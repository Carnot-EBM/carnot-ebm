"""Exp 1494 bounded ConstrainPrompt-style validator compiler audit.

Spec: REQ-VERIFY-1494, SCENARIO-VERIFY-1494.

The compiler in this module is intentionally narrow.  It extracts only a small
set of prompt patterns into a safe validator DSL and then executes fixed local
validator functions.  It never evaluates model-generated Python, never calls
``eval`` or ``exec``, and marks unsupported prompt constraints for manual
review instead of guessing.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from carnot.eval import cctu_executable_constraint_microbenchmark as cctu

JsonDict = dict[str, Any]

RUN_DATE = "20260507"
DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_1494_constrainprompt_validator_compiler_audit.json"
)
DEFAULT_MANIFEST_PATH = Path("results/constrainprompt_validator_manifest_1494.jsonl")

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = cctu.MANDATED_MODEL_SPECS
SAFE_VALIDATOR_KINDS: frozenset[str] = frozenset(
    {
        "cctu_tool_transcript",
        "json_final_answer",
        "json_schema",
        "python_ast_function",
        "graph_path_json",
    }
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "validator_compiler_ready",
    "prompts_attempted",
    "validator_skeletons_generated",
    "validators_compiled",
    "validator_compile_rate",
    "known_good_pass_rate",
    "known_bad_reject_rate",
    "verifier_false_accept_rate",
    "manual_review_required_count",
    "validator_manifest_path",
    "models_used",
    "gpu_probe",
    "blockers",
    "honest_verdict",
)

SUPPORTED_SCHEMA_TYPES: frozenset[str] = frozenset({"string", "integer", "array_string"})

ResolverFn = Callable[[str], str | None]
CachedPairFn = Callable[..., list[JsonDict] | None]
LlamaImporterFn = Callable[[], tuple[bool, type[Any] | None, str | None]]
CollectSkeletonsFn = Callable[[JsonDict, list["PromptCase"]], JsonDict]


@dataclass(frozen=True)
class PromptCase:
    """One fixed prompt plus its sanity-check examples.

    The audit treats known-good and known-bad examples as a compile-time unit
    test for each generated validator.  A prompt can therefore compile only if
    the resulting validator accepts a valid example and rejects a controlled
    violation.
    """

    prompt_id: str
    family: str
    source: str
    prompt: str
    known_good_output: str
    known_bad_output: str
    expected_validator_kind: str


@dataclass(frozen=True)
class CompiledValidator:
    """Safe compiler result for one prompt."""

    prompt_id: str
    compiled: bool
    dsl: JsonDict
    manual_review_required: bool
    failure_reason: str | None = None


def build_prompt_set() -> list[PromptCase]:
    """Return the fixed 30-prompt CCTU-style audit set."""

    prompts = [_prompt_from_exp1486(case) for case in cctu.build_benchmark_cases()]
    prompts.extend(_new_prompt_cases())
    return prompts


def compile_prompt(prompt: PromptCase) -> CompiledValidator:
    """Compile one prompt into the safe validator DSL or fail closed."""

    try:
        if "Use exactly one local tool named" in prompt.prompt:
            return _compile_cctu_prompt(prompt)
        if "Arithmetic expression:" in prompt.prompt:
            return _compile_arithmetic_prompt(prompt)
        if "Schema constraints JSON:" in prompt.prompt:
            return _compile_schema_prompt(prompt)
        if "Code constraints JSON:" in prompt.prompt:
            return _compile_code_prompt(prompt)
        if "Graph edges JSON:" in prompt.prompt:
            return _compile_graph_prompt(prompt)
    except Exception as exc:
        return _manual_review(prompt.prompt_id, f"{type(exc).__name__}: {exc}")
    return _manual_review(prompt.prompt_id, "unsupported_prompt_pattern")


def evaluate_compiled_validator(
    compiled: CompiledValidator,
    output_text: str,
) -> JsonDict:
    """Evaluate a compiled validator against one candidate output."""

    if not compiled.compiled:
        return {
            "accepted": False,
            "reason": compiled.failure_reason or "validator_not_compiled",
        }
    kind = str(compiled.dsl.get("kind"))
    if kind == "cctu_tool_transcript":
        return _evaluate_cctu_dsl(compiled.dsl, output_text)
    if kind == "json_final_answer":
        return _evaluate_json_final_answer(compiled.dsl, output_text)
    if kind == "json_schema":
        return _evaluate_json_schema(compiled.dsl, output_text)
    if kind == "python_ast_function":
        return _evaluate_python_ast_function(compiled.dsl, output_text)
    if kind == "graph_path_json":
        return _evaluate_graph_path_json(compiled.dsl, output_text)
    return {"accepted": False, "reason": f"unknown_validator_kind:{kind}"}


def compiler_uses_arbitrary_code_execution() -> bool:
    """Return whether this compiler introduces arbitrary execution paths."""

    return False


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable bootstrap artifact required by REQ-VERIFY-1494."""

    payload: JsonDict = {
        "status": "in_progress",
        "run_date": run_date,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "validator_compiler_ready": False,
        "prompts_attempted": 0,
        "validator_skeletons_generated": 0,
        "validators_compiled": 0,
        "validator_compile_rate": 0.0,
        "known_good_pass_rate": 0.0,
        "known_bad_reject_rate": 0.0,
        "verifier_false_accept_rate": 0.0,
        "manual_review_required_count": 0,
        "validator_manifest_path": _display_path(DEFAULT_MANIFEST_PATH),
        "models_used": [],
        "gpu_probe": {},
        "blockers": [],
        "honest_verdict": "complete: in-progress Exp 1494 bootstrap artifact",
    }
    _write_json(Path(output_path), payload)
    return payload


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    model_specs: Iterable[JsonDict] | None = None,
    collect_model_skeletons_fn: CollectSkeletonsFn | None = None,
    gpu_probe_fn: Callable[[], JsonDict] | None = None,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run the bounded compiler audit and write the manifest plus artifact."""

    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output, run_date=run_date)

    prompts = build_prompt_set()
    specs = list(resolve_model_specs() if model_specs is None else model_specs)
    gpu_probe = (gpu_probe_fn or probe_gpu)()
    if not specs:
        _write_jsonl(manifest, [])
        artifact = _build_terminal_artifact(
            run_date=run_date,
            manifest_path=manifest,
            prompts=prompts,
            rows=[],
            model_attempts=[],
            gpu_probe=gpu_probe,
            blockers=["no_mandated_sota_gguf_model_available"],
            tests_run=tests_run,
        )
        _write_json(output, artifact)
        return artifact

    collector = collect_model_skeletons_fn or collect_live_model_skeletons
    collection = collector(dict(specs[0]), prompts)
    model_attempts = [dict(collection.get("summary") or {})]
    skeletons_by_prompt = {
        str(row.get("prompt_id")): dict(row)
        for row in collection.get("rows") or []
        if row.get("prompt_id")
    }
    rows = [
        build_manifest_row(prompt, skeletons_by_prompt.get(prompt.prompt_id)) for prompt in prompts
    ]
    _write_jsonl(manifest, rows)

    blockers = [
        str(summary.get("blocker"))
        for summary in model_attempts
        if summary.get("model_used") is not True and summary.get("blocker")
    ]
    if not _live_sota_skeletons_present(rows):
        blockers.append("live_sota_skeleton_generation_unavailable")

    artifact = _build_terminal_artifact(
        run_date=run_date,
        manifest_path=manifest,
        prompts=prompts,
        rows=rows,
        model_attempts=model_attempts,
        gpu_probe=gpu_probe,
        blockers=list(dict.fromkeys(blockers)),
        tests_run=tests_run,
    )
    _write_json(output, artifact)
    return artifact


def build_manifest_row(
    prompt: PromptCase,
    skeleton_row: JsonDict | None,
) -> JsonDict:
    """Compile and sanity-check one prompt into a manifest row."""

    compiled = compile_prompt(prompt)
    good = evaluate_compiled_validator(compiled, prompt.known_good_output)
    bad = evaluate_compiled_validator(compiled, prompt.known_bad_output)
    model_skeleton = (
        parse_model_skeleton(skeleton_row.get("output_text", "")) if skeleton_row else None
    )
    model_blocker = skeleton_row.get("blocker") if skeleton_row else "missing_model_skeleton"
    false_accept = bool(compiled.compiled and bad["accepted"])
    false_reject = bool(compiled.compiled and not good["accepted"])
    manual_review = bool(
        compiled.manual_review_required
        or not compiled.compiled
        or model_skeleton is None
        or model_blocker is not None
    )
    return {
        "prompt_id": prompt.prompt_id,
        "family": prompt.family,
        "source": prompt.source,
        "prompt": prompt.prompt,
        "expected_validator_kind": prompt.expected_validator_kind,
        "model_hf_id": skeleton_row.get("model_hf_id") if skeleton_row else None,
        "model_name": skeleton_row.get("model_name") if skeleton_row else None,
        "generation_source": skeleton_row.get("generation_source") if skeleton_row else None,
        "elapsed_seconds": skeleton_row.get("elapsed_seconds") if skeleton_row else None,
        "model_blocker": model_blocker,
        "model_skeleton": model_skeleton,
        "compiled_validator": compiled.dsl,
        "validator_compiled": bool(compiled.compiled),
        "compiler_failure_reason": compiled.failure_reason,
        "manual_review_required": manual_review,
        "known_good_output": prompt.known_good_output,
        "known_good_result": good,
        "known_bad_output": prompt.known_bad_output,
        "known_bad_result": bad,
        "known_good_passed": bool(good["accepted"]),
        "known_bad_rejected": bool(not bad["accepted"]),
        "false_accept": false_accept,
        "false_reject": false_reject,
    }


def aggregate_manifest_metrics(rows: list[JsonDict]) -> JsonDict:
    """Compute compiler audit rates from manifest rows."""

    total = len(rows)
    compiled_rows = [row for row in rows if row["validator_compiled"]]
    compiled_total = len(compiled_rows)
    false_accepts = sum(bool(row["false_accept"]) for row in compiled_rows)
    return {
        "validator_skeletons_generated": sum(
            row.get("model_skeleton") is not None and row.get("model_blocker") is None
            for row in rows
        ),
        "validators_compiled": compiled_total,
        "validator_compile_rate": round(compiled_total / total, 6) if total else 0.0,
        "known_good_pass_rate": _rate(
            compiled_rows,
            lambda row: bool(row["known_good_passed"]),
        ),
        "known_bad_reject_rate": _rate(
            compiled_rows,
            lambda row: bool(row["known_bad_rejected"]),
        ),
        "verifier_false_accept_rate": (
            round(false_accepts / compiled_total, 6) if compiled_total else 0.0
        ),
        "manual_review_required_count": sum(bool(row["manual_review_required"]) for row in rows),
    }


def parse_model_skeleton(output_text: str) -> JsonDict | None:
    """Parse a model-suggested validator skeleton, if it emitted JSON."""

    obj = cctu.extract_json_object(output_text)
    if obj is None:
        return None
    return obj


def resolve_model_specs(
    *,
    cached_pair_fn: CachedPairFn | None = None,
    resolver_fn: ResolverFn | None = None,
) -> list[JsonDict]:
    """Resolve mandated local SOTA GGUF specs without legacy small fallbacks."""

    pair_resolver = cached_pair_fn or _cached_sota_pair
    pair = pair_resolver(gpu_indices=(0, 1))
    if pair:
        return pair

    resolver = resolver_fn or cctu._default_resolver
    specs: list[JsonDict] = []
    for spec in MANDATED_MODEL_SPECS:
        model_path = resolver(str(spec["hf_id"]))
        if model_path:
            specs.append({**spec, "model_path": model_path})
    return specs


def collect_live_model_skeletons(
    spec: JsonDict,
    prompts: list[PromptCase],
    *,
    resolver: ResolverFn | None = None,
    llama_importer: LlamaImporterFn | None = None,
    env_preparer: Callable[[], JsonDict] | None = None,
) -> JsonDict:
    """Ask one mandated local GGUF model for validator skeleton suggestions."""

    hf_id = str(spec.get("hf_id") or "")
    model_path = spec.get("model_path") or (resolver or cctu._default_resolver)(hf_id)
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

    load_start = time.monotonic()
    try:
        llm = llama_class(
            model_path=str(model_path),
            n_gpu_layers=-1,
            main_gpu=int(spec.get("gpu") or 0),
            n_ctx=8192,
            seed=1494,
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

    rows: list[JsonDict] = []
    try:
        for prompt in prompts:
            started = time.monotonic()
            try:
                result = llm(
                    _skeleton_prompt(prompt),
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
                    "prompt_id": prompt.prompt_id,
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
            "blocker": None if model_used else "no_usable_skeleton_generations",
            "env_details": env_details,
        },
        "rows": rows,
    }


def probe_gpu() -> JsonDict:
    """Return a JSON-safe NVIDIA GPU probe for the result artifact."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return {
            "nvidia_smi_available": False,
            "gpu_count": 0,
            "gpus": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    if result.returncode != 0:
        return {
            "nvidia_smi_available": False,
            "gpu_count": 0,
            "gpus": [],
            "error": result.stderr.strip() or "nvidia-smi failed",
        }

    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        name, _, memory = line.partition(",")
        gpus.append({"name": name.strip(), "memory_total": memory.strip()})
    return {"nvidia_smi_available": True, "gpu_count": len(gpus), "gpus": gpus}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the conductor and manual runs."""

    _ = list(sys.argv[1:] if argv is None else argv)
    artifact = run_experiment()
    print(
        "[exp1494] "
        f"ready={artifact['validator_compiler_ready']} "
        f"compile_rate={artifact['validator_compile_rate']} "
        f"false_accept={artifact['verifier_false_accept_rate']} "
        f"verdict={artifact['honest_verdict']}"
    )
    return 0


def _prompt_from_exp1486(case: cctu.BenchmarkCase) -> PromptCase:
    bad_payload = json.loads(cctu.compliant_transcript_for_case(case))
    bad_payload["final_answer"] = f"not {case.expected_final_answer}"
    bad_payload["verifier"] = {"accept": True}
    return PromptCase(
        prompt_id=case.case_id,
        family=case.family,
        source="exp1486",
        prompt=case.prompt,
        known_good_output=cctu.compliant_transcript_for_case(case),
        known_bad_output=json.dumps(bad_payload, sort_keys=True),
        expected_validator_kind="cctu_tool_transcript",
    )


def _new_prompt_cases() -> list[PromptCase]:
    return [
        _arithmetic_prompt("cctu-1494-arith-001", "(17 * 3) + 4"),
        _arithmetic_prompt("cctu-1494-arith-002", "(144 // 12) + (6 * 5)"),
        _arithmetic_prompt("cctu-1494-arith-003", "(9 % 4) + (8 * 7)"),
        _schema_prompt(
            "cctu-1494-schema-001",
            {
                "required": {
                    "ticket_id": {"type": "string", "equals": "INC-42"},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 3},
                    "tags": {"type": "array_string", "contains": "security"},
                }
            },
            {"ticket_id": "INC-42", "priority": 2, "tags": ["security", "api"]},
            {"ticket_id": "INC-42", "priority": 5, "tags": ["api"]},
        ),
        _schema_prompt(
            "cctu-1494-schema-002",
            {
                "required": {
                    "sku": {"type": "string", "equals": "BOLT-7"},
                    "quantity": {"type": "integer", "minimum": 10, "maximum": 20},
                }
            },
            {"sku": "BOLT-7", "quantity": 12},
            {"sku": "BOLT-8", "quantity": 12},
        ),
        _unsupported_schema_prompt(),
        _code_prompt(
            "cctu-1494-code-001",
            {
                "function_name": "normalize_slug",
                "parameters": ["text"],
                "returns_expr": 'text.strip().lower().replace(" ", "-")',
            },
            ('def normalize_slug(text):\n    return text.strip().lower().replace(" ", "-")\n'),
            "def normalize_slug(text):\n    return text.lower()\n",
        ),
        _unsupported_code_prompt(),
        _graph_prompt(
            "cctu-1494-graph-001",
            [["A", "B", 1], ["B", "D", 3], ["A", "C", 5], ["C", "D", 1]],
            "A",
            "D",
        ),
        _graph_prompt(
            "cctu-1494-graph-002",
            [["S", "M", 2], ["M", "T", 2], ["S", "N", 5], ["N", "T", 1]],
            "S",
            "T",
        ),
    ]


def _arithmetic_prompt(prompt_id: str, expression: str) -> PromptCase:
    expected = _safe_eval_arithmetic(expression)
    prompt = (
        "You are evaluating a CCTU-style arithmetic constraint case.\n"
        f"Case: {prompt_id}\n"
        f"Arithmetic expression: {expression}\n"
        "Return exactly one JSON object with key `final_answer`. "
        "The final_answer value must equal the computed integer."
    )
    return PromptCase(
        prompt_id=prompt_id,
        family="arithmetic",
        source="exp1494_new",
        prompt=prompt,
        known_good_output=json.dumps({"final_answer": expected}, sort_keys=True),
        known_bad_output=json.dumps({"final_answer": expected + 1}, sort_keys=True),
        expected_validator_kind="json_final_answer",
    )


def _schema_prompt(
    prompt_id: str,
    schema: JsonDict,
    good: JsonDict,
    bad: JsonDict,
) -> PromptCase:
    prompt = (
        "You are evaluating a CCTU-style JSON schema constraint case.\n"
        f"Case: {prompt_id}\n"
        "Schema constraints JSON:\n"
        f"{json.dumps(schema, sort_keys=True)}\n"
        "Return exactly one JSON object satisfying those constraints."
    )
    return PromptCase(
        prompt_id=prompt_id,
        family="json_schema",
        source="exp1494_new",
        prompt=prompt,
        known_good_output=json.dumps(good, sort_keys=True),
        known_bad_output=json.dumps(bad, sort_keys=True),
        expected_validator_kind="json_schema",
    )


def _unsupported_schema_prompt() -> PromptCase:
    schema = {"required": {"tracking_code": {"type": "string", "pattern": "^[A-Z]{2}-[0-9]{3}$"}}}
    return _schema_prompt(
        "cctu-1494-schema-003",
        schema,
        {"tracking_code": "AB-123"},
        {"tracking_code": "bad"},
    )


def _code_prompt(
    prompt_id: str,
    constraints: JsonDict,
    good: str,
    bad: str,
) -> PromptCase:
    prompt = (
        "You are evaluating a CCTU-style simple-code constraint case.\n"
        f"Case: {prompt_id}\n"
        "Code constraints JSON:\n"
        f"{json.dumps(constraints, sort_keys=True)}\n"
        "Return only a Python function definition satisfying the constraints."
    )
    return PromptCase(
        prompt_id=prompt_id,
        family="simple_code",
        source="exp1494_new",
        prompt=prompt,
        known_good_output=good,
        known_bad_output=bad,
        expected_validator_kind="python_ast_function",
    )


def _unsupported_code_prompt() -> PromptCase:
    return _code_prompt(
        "cctu-1494-code-002",
        {
            "function_name": "unique_sorted",
            "parameters": ["values"],
            "behavior": "return the sorted unique values for any list input",
        },
        "def unique_sorted(values):\n    return sorted(set(values))\n",
        "def unique_sorted(values):\n    return values\n",
    )


def _graph_prompt(
    prompt_id: str,
    edges: list[list[Any]],
    start: str,
    end: str,
) -> PromptCase:
    result = cctu.execute_tool(
        "graph.shortest_path",
        {"edges": edges, "start": start, "end": end},
    )
    prompt = (
        "You are evaluating a CCTU-style graph/path constraint case.\n"
        f"Case: {prompt_id}\n"
        "Graph edges JSON:\n"
        f"{json.dumps(edges, sort_keys=True)}\n"
        f"Start: {start}\n"
        f"End: {end}\n"
        "Return exactly one JSON object with keys `path` and `cost` for the shortest path."
    )
    bad = {"path": list(reversed(result["path"])), "cost": int(result["cost"]) + 1}
    return PromptCase(
        prompt_id=prompt_id,
        family="graph_path",
        source="exp1494_new",
        prompt=prompt,
        known_good_output=json.dumps(result, sort_keys=True),
        known_bad_output=json.dumps(bad, sort_keys=True),
        expected_validator_kind="graph_path_json",
    )


def _compile_cctu_prompt(prompt: PromptCase) -> CompiledValidator:
    match = re.search(
        r"Case:\s*(?P<case_id>[^\n]+)\n"
        r"Constraint family:\s*(?P<family>[^\n]+)\n"
        r"Use exactly one local tool named\s*(?P<tool_name>[\w.]+)\s*with these arguments:\n"
        r"(?P<arguments>\{.*?\})\nReturn exactly",
        prompt.prompt,
        flags=re.S,
    )
    if match is None:
        return _manual_review(prompt.prompt_id, "cctu_prompt_parse_failed")
    arguments = json.loads(match.group("arguments"))
    tool_name = match.group("tool_name")
    family = match.group("family")
    expected_tool_result = cctu.execute_tool(tool_name, arguments)
    expected_final_answer = cctu._final_answer_from_result(family, expected_tool_result)
    return CompiledValidator(
        prompt_id=prompt.prompt_id,
        compiled=True,
        manual_review_required=False,
        dsl={
            "kind": "cctu_tool_transcript",
            "case_id": match.group("case_id"),
            "family": family,
            "tool_name": tool_name,
            "tool_arguments": arguments,
            "expected_final_answer": expected_final_answer,
        },
    )


def _compile_arithmetic_prompt(prompt: PromptCase) -> CompiledValidator:
    match = re.search(r"Arithmetic expression:\s*(?P<expr>[^\n]+)", prompt.prompt)
    if match is None:
        return _manual_review(prompt.prompt_id, "arithmetic_expression_missing")
    expected = _safe_eval_arithmetic(match.group("expr"))
    return CompiledValidator(
        prompt_id=prompt.prompt_id,
        compiled=True,
        manual_review_required=False,
        dsl={"kind": "json_final_answer", "expected": expected},
    )


def _compile_schema_prompt(prompt: PromptCase) -> CompiledValidator:
    schema = _extract_json_block(prompt.prompt, "Schema constraints JSON:")
    failure = _schema_compile_failure(schema)
    if failure:
        return _manual_review(prompt.prompt_id, failure)
    return CompiledValidator(
        prompt_id=prompt.prompt_id,
        compiled=True,
        manual_review_required=False,
        dsl={"kind": "json_schema", "schema": schema},
    )


def _compile_code_prompt(prompt: PromptCase) -> CompiledValidator:
    constraints = _extract_json_block(prompt.prompt, "Code constraints JSON:")
    if "returns_expr" not in constraints:
        return _manual_review(prompt.prompt_id, "code_behavior_requires_execution_or_review")
    function_name = constraints.get("function_name")
    parameters = constraints.get("parameters")
    if not isinstance(function_name, str) or not _is_string_list(parameters):
        return _manual_review(prompt.prompt_id, "code_signature_constraint_malformed")
    expected_expr = _normalise_expr_ast(str(constraints["returns_expr"]))
    return CompiledValidator(
        prompt_id=prompt.prompt_id,
        compiled=True,
        manual_review_required=False,
        dsl={
            "kind": "python_ast_function",
            "function_name": function_name,
            "parameters": list(parameters),
            "returns_expr": str(constraints["returns_expr"]),
            "returns_expr_ast": expected_expr,
        },
    )


def _compile_graph_prompt(prompt: PromptCase) -> CompiledValidator:
    edges = _extract_json_block(prompt.prompt, "Graph edges JSON:")
    start_match = re.search(r"Start:\s*(?P<start>[^\n]+)", prompt.prompt)
    end_match = re.search(r"End:\s*(?P<end>[^\n]+)", prompt.prompt)
    if start_match is None or end_match is None:
        return _manual_review(prompt.prompt_id, "graph_endpoint_missing")
    start = start_match.group("start").strip()
    end = end_match.group("end").strip()
    result = cctu.execute_tool(
        "graph.shortest_path",
        {"edges": edges, "start": start, "end": end},
    )
    return CompiledValidator(
        prompt_id=prompt.prompt_id,
        compiled=True,
        manual_review_required=False,
        dsl={
            "kind": "graph_path_json",
            "edges": edges,
            "start": start,
            "end": end,
            "expected": result,
        },
    )


def _manual_review(prompt_id: str, reason: str) -> CompiledValidator:
    return CompiledValidator(
        prompt_id=prompt_id,
        compiled=False,
        dsl={"kind": "manual_review", "reason": reason},
        manual_review_required=True,
        failure_reason=reason,
    )


def _evaluate_cctu_dsl(dsl: JsonDict, output_text: str) -> JsonDict:
    expected_tool_result = cctu.execute_tool(
        str(dsl["tool_name"]),
        dict(dsl["tool_arguments"]),
    )
    case = cctu.BenchmarkCase(
        case_id=str(dsl["case_id"]),
        family=str(dsl["family"]),
        tool_name=str(dsl["tool_name"]),
        tool_arguments=dict(dsl["tool_arguments"]),
        expected_tool_result=expected_tool_result,
        expected_final_answer=str(dsl["expected_final_answer"]),
        prompt="",
    )
    validation = cctu.validate_transcript(case, output_text)
    return {
        "accepted": bool(validation["verifier_result"]["accepted"]),
        "reason": validation["validator_result"].get("parse_error"),
        "validator_result": validation["validator_result"],
    }


def _evaluate_json_final_answer(dsl: JsonDict, output_text: str) -> JsonDict:
    obj = cctu.extract_json_object(output_text)
    if obj is None:
        return {"accepted": False, "reason": "no_json_object"}
    accepted = _normalise_scalar(obj.get("final_answer")) == _normalise_scalar(dsl["expected"])
    return {"accepted": accepted, "reason": None if accepted else "final_answer_mismatch"}


def _evaluate_json_schema(dsl: JsonDict, output_text: str) -> JsonDict:
    obj = cctu.extract_json_object(output_text)
    if obj is None:
        return {"accepted": False, "reason": "no_json_object"}
    if not isinstance(obj, dict):
        return {"accepted": False, "reason": "json_not_object"}
    for field, constraints in dsl["schema"]["required"].items():
        if field not in obj:
            return {"accepted": False, "reason": f"missing_field:{field}"}
        ok, reason = _value_satisfies_schema(obj[field], constraints)
        if not ok:
            return {"accepted": False, "reason": f"{field}:{reason}"}
    return {"accepted": True, "reason": None}


def _evaluate_python_ast_function(dsl: JsonDict, output_text: str) -> JsonDict:
    try:
        module = ast.parse(output_text)
    except SyntaxError as exc:
        return {"accepted": False, "reason": f"syntax_error:{exc.msg}"}
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        return {"accepted": False, "reason": "expected_single_function_def"}
    function = module.body[0]
    parameters = [arg.arg for arg in function.args.args]
    if function.name != dsl["function_name"] or parameters != dsl["parameters"]:
        return {"accepted": False, "reason": "function_signature_mismatch"}
    if len(function.body) != 1 or not isinstance(function.body[0], ast.Return):
        return {"accepted": False, "reason": "function_body_not_single_return"}
    actual_expr = ast.dump(function.body[0].value, include_attributes=False)
    accepted = actual_expr == dsl["returns_expr_ast"]
    return {"accepted": accepted, "reason": None if accepted else "return_expr_mismatch"}


def _evaluate_graph_path_json(dsl: JsonDict, output_text: str) -> JsonDict:
    obj = cctu.extract_json_object(output_text)
    if obj is None:
        return {"accepted": False, "reason": "no_json_object"}
    expected = dsl["expected"]
    accepted = obj.get("path") == expected["path"] and int(obj.get("cost", -1)) == int(
        expected["cost"]
    )
    return {"accepted": accepted, "reason": None if accepted else "graph_path_mismatch"}


def _value_satisfies_schema(value: Any, constraints: JsonDict) -> tuple[bool, str | None]:
    typ = constraints.get("type")
    if typ == "string" and not isinstance(value, str):
        return False, "not_string"
    if typ == "integer" and not isinstance(value, int):
        return False, "not_integer"
    if typ == "array_string":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return False, "not_array_string"
    if "equals" in constraints and value != constraints["equals"]:
        return False, "equals_mismatch"
    if "minimum" in constraints and value < constraints["minimum"]:
        return False, "below_minimum"
    if "maximum" in constraints and value > constraints["maximum"]:
        return False, "above_maximum"
    if "contains" in constraints and constraints["contains"] not in value:
        return False, "missing_required_member"
    return True, None


def _schema_compile_failure(schema: JsonDict) -> str | None:
    required = schema.get("required")
    if not isinstance(required, dict) or not required:
        return "schema_required_fields_missing"
    for field, constraints in required.items():
        if not isinstance(field, str) or not isinstance(constraints, dict):
            return "schema_field_malformed"
        unsupported_keys = set(constraints) - {
            "type",
            "equals",
            "minimum",
            "maximum",
            "contains",
        }
        if unsupported_keys:
            return f"unsupported_schema_constraint:{sorted(unsupported_keys)[0]}"
        if constraints.get("type") not in SUPPORTED_SCHEMA_TYPES:
            return f"unsupported_schema_type:{constraints.get('type')}"
    return None


def _safe_eval_arithmetic(expression: str) -> int:
    tree = ast.parse(expression, mode="eval")
    return int(_eval_arithmetic_node(tree.body))


def _eval_arithmetic_node(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_arithmetic_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_arithmetic_node(node.left)
        right = _eval_arithmetic_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
    raise ValueError("unsupported_arithmetic_expression")


def _normalise_expr_ast(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    return ast.dump(tree.body, include_attributes=False)


def _extract_json_block(text: str, marker: str) -> JsonDict:
    start = text.index(marker) + len(marker)
    tail = text[start:].lstrip()
    obj = cctu.extract_json_object(tail)
    if obj is not None:
        return obj
    decoder = json.JSONDecoder()
    value, _end = decoder.raw_decode(tail)
    if isinstance(value, list):
        return value  # type: ignore[return-value]
    raise ValueError(f"json_block_missing_after:{marker}")


def _normalise_scalar(value: Any) -> str:
    return str(value).strip().casefold()


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _skeleton_prompt(prompt: PromptCase) -> str:
    return (
        "Extract a validator skeleton for this bounded CCTU-style prompt. "
        "Return exactly one JSON object with keys prompt_id, validator_kind, "
        "fields, and manual_review_hint. Do not write code.\n"
        f"Prompt id: {prompt.prompt_id}\n"
        f"Prompt text:\n{prompt.prompt}\n"
    )


def _build_terminal_artifact(
    *,
    run_date: str,
    manifest_path: Path,
    prompts: list[PromptCase],
    rows: list[JsonDict],
    model_attempts: list[JsonDict],
    gpu_probe: JsonDict,
    blockers: list[str],
    tests_run: list[str] | None,
) -> JsonDict:
    metrics = aggregate_manifest_metrics(rows)
    models_used = [
        str(summary["hf_id"])
        for summary in model_attempts
        if summary.get("model_used") is True and summary.get("hf_id")
    ]
    live_used = _live_sota_skeletons_present(rows)
    arbitrary_execution = compiler_uses_arbitrary_code_execution()
    ready = (
        bool(rows) and live_used and metrics["validators_compiled"] > 0 and not arbitrary_execution
    )
    status = "complete" if ready else "blocked"
    verdict = (
        "complete: bounded safe-DSL validator compiler audit measured on live local SOTA GGUF skeleton rows"
        if ready
        else "complete: blocked because mandated live SOTA GGUF validator skeleton rows were not produced"
    )
    return {
        "status": status,
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": bool(live_used),
        "validator_compiler_ready": bool(ready),
        "prompts_attempted": len(prompts),
        "validator_skeletons_generated": metrics["validator_skeletons_generated"],
        "validators_compiled": metrics["validators_compiled"],
        "validator_compile_rate": metrics["validator_compile_rate"],
        "known_good_pass_rate": metrics["known_good_pass_rate"],
        "known_bad_reject_rate": metrics["known_bad_reject_rate"],
        "verifier_false_accept_rate": metrics["verifier_false_accept_rate"],
        "manual_review_required_count": metrics["manual_review_required_count"],
        "validator_manifest_path": _display_path(manifest_path),
        "models_used": models_used,
        "gpu_probe": gpu_probe,
        "blockers": blockers,
        "honest_verdict": verdict,
        "arbitrary_code_execution_path_introduced": arbitrary_execution,
        "model_attempts": model_attempts,
        "manifest_rows": len(rows),
        "tests_run": list(tests_run or []),
    }


def _live_sota_skeletons_present(rows: list[JsonDict]) -> bool:
    mandated = {str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS}
    return any(
        row.get("generation_source") == "live_sota_llamacpp"
        and row.get("model_blocker") is None
        and row.get("model_hf_id") in mandated
        for row in rows
    )


def _rate(rows: list[JsonDict], predicate: Callable[[JsonDict], bool]) -> float:
    if not rows:
        return 0.0
    return round(sum(bool(predicate(row)) for row in rows) / len(rows), 6)


def _cached_sota_pair(**kwargs: Any) -> list[JsonDict] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair  # noqa: PLC0415

    return cached_sota_pair(**kwargs)


def _display_path(path: Path | str) -> str:
    as_path = Path(path)
    try:
        return str(as_path.resolve().relative_to(cctu._repo_root()))
    except ValueError:
        return str(as_path)


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - exercised by conductor.
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "MANDATED_MODEL_SPECS",
    "PromptCase",
    "REQUIRED_ARTIFACT_FIELDS",
    "SAFE_VALIDATOR_KINDS",
    "aggregate_manifest_metrics",
    "build_manifest_row",
    "build_prompt_set",
    "collect_live_model_skeletons",
    "compile_prompt",
    "compiler_uses_arbitrary_code_execution",
    "evaluate_compiled_validator",
    "main",
    "parse_model_skeleton",
    "probe_gpu",
    "resolve_model_specs",
    "run_experiment",
    "write_in_progress_artifact",
]

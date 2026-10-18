"""Exp 1144 CCTU-style constrained tool-use micro-benchmark adapter.

Spec: REQ-VERIFY-1144, SCENARIO-VERIFY-1144
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from carnot.inference.sota_models import resolve_cached_gguf
from carnot.verify.ast_structure_verifier import ASTStructureVerifier
from carnot.verify.semenergy_probe import SemEnergyProbe
from carnot.verify.z3_math_verifier import Z3MathVerifier

CCTU_SOURCE_DATASET_ID = "Junjie-Ye/CCTU"
CCTU_SOURCE_URL = "https://huggingface.co/datasets/Junjie-Ye/CCTU"
MANDATED_SOTA_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS = (
    "cctu_adapter_written",
    "n_tasks_defined",
    "n_tasks_evaluated",
    "model_used",
    "inference_mode",
    "constraint_types_tested",
    "per_constraint_tp_rate",
    "baseline_completion_rate",
    "carnot_guided_completion_rate",
    "carnot_delta_pp",
    "cctu_adapter_honest_result",
    "honest_verdict",
)
ALLOWED_HONEST_VERDICTS = {
    "carnot_positive_delta",
    "carnot_neutral",
    "carnot_negative",
    "mock_inference_only",
}

_SOURCE_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("0", "2007-11-7", "historical_event_finder", "Single-Hop"),
    ("1", "Mishal Al-Ahmad Al-Jaber Al-Sabah.", "political_successor_finder", "Single-Hop"),
    ("2", "2007-10-10", "game_release_date_finder", "Single-Hop"),
    ("3", "128", "election_results_analyzer", "Single-Hop"),
    ("4", "Al-Aqsa Mosque", "event_locator", "Single-Hop"),
    ("5", "Abdel Fattah el-Sisi.", "event_outcome_retriever", "Single-Hop"),
    ("6", "Michel Barnier", "political_event_search", "Single-Hop"),
    ("7", "Brazil, China", "international_agreement_finder", "Single-Hop"),
    ("8", "Fuad Shukr", "event_personnel_identifier", "Single-Hop"),
    ("9", "2007-09-13", "historical_event_finder", "Single-Hop"),
    ("10", "United Kingdom", "event_location_finder", "Single-Hop"),
    ("11", "2001-9-17", "historical_event_search", "Single-Hop"),
    ("12", "2023-10-13", "corporate_event_finder", "Single-Hop"),
    ("13", "2001-04-01", "historical_event_retriever", "Single-Hop"),
    ("14", "2023-10-22", "historical_event_date_finder", "Single-Hop"),
    ("15", "2024-6-23", "historical_event_finder", "Single-Hop"),
    ("16", "2023-10-31", "military_event_search", "Single-Hop"),
    ("17", "165", "historical_weather_data_retriever", "Single-Hop"),
    ("18", "2023-10-8", "historical_event_finder", "Single-Hop"),
    ("19", "75", "historical_event_data_retriever", "Single-Hop"),
    ("20", "Australia", "individual_movement_tracker", "Single-Hop"),
    ("21", "The Ohrid Agreement", "historical_agreement_finder", "Single-Hop"),
    ("22", "3", "political_term_tracker", "Single-Hop"),
    ("23", "Israel", "geopolitical_event_finder", "Single-Hop"),
    ("24", "2007-07-24", "historical_event_finder", "Single-Hop"),
)


class ResponseRunner(Protocol):
    """Minimal inference interface shared by live and mock CCTU runners."""

    inference_mode: str
    model_used: str

    def generate(self, task: dict[str, Any], prompt: str, *, guided: bool) -> str: ...


@dataclass(frozen=True)
class ConstraintCheck:
    """One executable constraint check result."""

    constraint_id: str
    constraint_type: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ResponseEvaluation:
    """Cascade and validator result for one task response."""

    passed: bool
    checks: tuple[ConstraintCheck, ...]
    verifier_scores: dict[str, float]
    by_type: dict[str, tuple[int, int]]

    @property
    def completion_rate(self) -> float:
        """Return fraction of constraints passed for this response."""

        if not self.checks:
            return 0.0
        return sum(1 for check in self.checks if check.passed) / len(self.checks)

    @property
    def violated_types(self) -> set[str]:
        """Return constraint types with at least one failed check."""

        return {check.constraint_type for check in self.checks if not check.passed}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for experiment artifacts."""

        return {
            "passed": self.passed,
            "completion_rate": self.completion_rate,
            "violated_types": sorted(self.violated_types),
            "verifier_scores": dict(self.verifier_scores),
            "checks": [
                {
                    "constraint_id": check.constraint_id,
                    "constraint_type": check.constraint_type,
                    "passed": check.passed,
                    "detail": check.detail,
                }
                for check in self.checks
            ],
        }


class MockCCTURunner:
    """Deterministic inference fallback used when the GGUF loader is unavailable."""

    inference_mode = "mock"

    def __init__(self, model_used: str = MANDATED_SOTA_MODELS[0]) -> None:
        self.model_used = model_used

    def generate(self, task: dict[str, Any], prompt: str, *, guided: bool) -> str:
        del prompt
        if guided:
            return compliant_response_for_task(task)
        return mock_baseline_response_for_task(task)


class LlamaCppCCTURunner:
    """Thin wrapper around the repo's llama.cpp-compatible GGUF loader."""

    inference_mode = "live_gpu"

    def __init__(self, model_used: str, model_path: str) -> None:  # pragma: no cover
        from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader

        self.model_used = model_used
        self.model_path = model_path
        self.loader = Gemma4QuantizedLoader(model_path=model_path, n_gpu_layers=-1, max_tokens=256)
        if not self.loader.load() or getattr(self.loader, "_stub_mode", False):
            raise RuntimeError("llama.cpp loader entered stub mode")

    def generate(
        self, task: dict[str, Any], prompt: str, *, guided: bool
    ) -> str:  # pragma: no cover
        del task, guided
        return self.loader.generate(prompt)


def build_micro_benchmark_tasks() -> list[dict[str, Any]]:
    """Build 25 CCTU-style tasks with explicit executable validators.

    The local adapter uses public CCTU source row identifiers and answers, then
    normalises the interaction into a compact single-turn transcript protocol so
    Carnot can validate tool-use constraints deterministically in CI.
    """

    tasks: list[dict[str, Any]] = []
    for idx, (source_id, answer, source_family, data_source) in enumerate(_SOURCE_ROWS):
        score = 20 + idx
        tool_call_count = 2 if idx % 3 else 1
        format_style = ("markdown_bold", "json_answer", "python_code", "answer_prefix", "table")[
            idx % 5
        ]
        task_id = f"cctu-micro-{idx:03d}"
        system_prompt = (
            "You are a constrained tool-use agent. Available tools are "
            f"`{source_family}` for source lookup and `calculate` for arithmetic checks. "
            "Emit tool calls as `TOOL_CALL name {json}` lines or as a JSON `tool_calls` "
            "array, then provide the final answer. Obey every explicit constraint."
        )
        user_request = (
            f"Answer representative public CCTU source task {source_id}. "
            f"Use {tool_call_count} tool call(s), include a numeric SCORE between 10 and 100, "
            f"and return the final target answer `{answer}` in the required format."
        )
        constraints = [
            {
                "id": f"{task_id}-numeric-score",
                "type": "numeric",
                "description": "The response must include SCORE in the inclusive range [10, 100].",
                "validator": {"name": "numeric_score_range", "min": 10, "max": 100},
            },
            {
                "id": f"{task_id}-semantic-answer",
                "type": "semantic",
                "description": "The final answer must mention the expected CCTU target answer.",
                "validator": {"name": "contains_expected_answer"},
            },
            {
                "id": f"{task_id}-resource-tools",
                "type": "resource",
                "description": (
                    f"The transcript must use exactly {tool_call_count} tool call(s), "
                    f"including `{source_family}`."
                ),
                "validator": {
                    "name": "tool_call_protocol",
                    "count": tool_call_count,
                    "required_tool": source_family,
                },
            },
            {
                "id": f"{task_id}-format-{format_style}",
                "type": "format",
                "description": f"The final response must satisfy the `{format_style}` format.",
                "validator": {"name": "format_style", "style": format_style},
            },
            {
                "id": f"{task_id}-length",
                "type": "length",
                "description": "The full response must be at most 90 whitespace-delimited words.",
                "validator": {"name": "word_count_max", "max": 90},
            },
        ]
        tasks.append(
            {
                "task_id": task_id,
                "source": {
                    "dataset": CCTU_SOURCE_DATASET_ID,
                    "url": CCTU_SOURCE_URL,
                    "source_row_id": source_id,
                    "data_source": data_source,
                    "license": "cc-by-4.0",
                    "adapter_derivation": "public-row answer plus local executable constraints",
                },
                "system_prompt": system_prompt,
                "user_request": user_request,
                "available_tools": [source_family, "calculate"],
                "expected_answer": answer,
                "score_value": score,
                "format_style": format_style,
                "constraints": constraints,
            }
        )
    return tasks


def write_tasks_json(tasks: list[dict[str, Any]], output_path: Path) -> Path:
    """Write the 25-task adapter JSON deterministically."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def load_or_write_tasks(data_path: Path) -> list[dict[str, Any]]:
    """Load existing task JSON or create it from the embedded source rows."""

    if data_path.exists():
        return json.loads(data_path.read_text(encoding="utf-8"))
    tasks = build_micro_benchmark_tasks()
    write_tasks_json(tasks, data_path)
    return tasks


def build_prompt(task: dict[str, Any]) -> str:
    """Return the single-turn prompt sent to the model for a task."""

    constraints = "\n".join(f"- {c['description']}" for c in task["constraints"])
    return f"{task['system_prompt']}\n\nUSER REQUEST:\n{task['user_request']}\n\nCONSTRAINTS:\n{constraints}"


def repair_prompt(task: dict[str, Any], evaluation: ResponseEvaluation) -> str:
    """Return verifier feedback used for the Carnot-guided repair condition."""

    violations = [
        f"{check.constraint_id} ({check.constraint_type}): {check.detail}"
        for check in evaluation.checks
        if not check.passed
    ]
    joined = "\n".join(f"- {violation}" for violation in violations) or "- none"
    return (
        f"Repair the response for {task['task_id']} so every constraint passes.\n"
        f"Verifier violations:\n{joined}\n\nOriginal prompt:\n{build_prompt(task)}"
    )


def compliant_response_for_task(task: dict[str, Any]) -> str:
    """Return a deterministic response that satisfies all validators for a task."""

    answer = str(task["expected_answer"])
    score = int(task["score_value"])
    tools = _tool_call_names_for_task(task)
    check = f"{score} + 0 = {score}"
    style = task["format_style"]
    if style == "json_answer":
        return json.dumps(
            {
                "tool_calls": [
                    {"name": name, "arguments": {"task_id": task["task_id"]}} for name in tools
                ],
                "answer": answer,
                "score": score,
                "check": check,
            },
            sort_keys=True,
        )

    tool_lines = "\n".join(f'TOOL_CALL {name} {{"task_id": "{task["task_id"]}"}}' for name in tools)
    if style == "markdown_bold":
        final = f"FINAL: **{answer}**\nSCORE: {score}.\nCHECK: {check}."
    elif style == "python_code":
        final = (
            "```python\n"
            "def final_answer():\n"
            f"    return {answer!r}\n"
            "```\n"
            f"FINAL: {answer}\nSCORE: {score}.\nCHECK: {check}."
        )
    elif style == "answer_prefix":
        final = f"FINAL: ANSWER: {answer}\nSCORE: {score}.\nCHECK: {check}."
    else:
        final = f"| answer | score |\n| --- | --- |\n| {answer} | {score} |\nSCORE: {score}.\nCHECK: {check}."
    return f"{tool_lines}\n{final}"


def mock_baseline_response_for_task(task: dict[str, Any]) -> str:
    """Return a deterministic raw-model response with representative violations."""

    del task
    return "TOOL_CALL wrong_tool {}\nFINAL: wrong target\nSCORE: 4\nCHECK: 2 + 2 = 5."


def evaluate_response(task: dict[str, Any], response: str) -> ResponseEvaluation:
    """Run the Carnot verifier cascade and executable validators for one response."""

    verifier_scores: dict[str, float] = {}
    constraint_types = {constraint["type"] for constraint in task["constraints"]}
    if "numeric" in constraint_types:
        verifier_scores["z3_math"] = Z3MathVerifier().score(response)
    if "semantic" in constraint_types:
        verifier_scores["semenergy"] = SemEnergyProbe().score_response_proxy(response)
    if "format" in constraint_types:
        verifier_scores["ast_structure"] = ASTStructureVerifier().score(response)

    checks = tuple(
        _check_constraint(task, constraint, response) for constraint in task["constraints"]
    )
    by_type: dict[str, tuple[int, int]] = {}
    for check in checks:
        passed, total = by_type.get(check.constraint_type, (0, 0))
        by_type[check.constraint_type] = (passed + int(check.passed), total + 1)
    return ResponseEvaluation(
        passed=all(check.passed for check in checks),
        checks=checks,
        verifier_scores=verifier_scores,
        by_type=by_type,
    )


def build_exp1144_artifact(
    *,
    n_tasks_defined: int,
    n_tasks_evaluated: int,
    model_used: str,
    inference_mode: str,
    constraint_types_tested: list[str],
    per_constraint_tp_rate: dict[str, float],
    baseline_completion_rate: float,
    carnot_guided_completion_rate: float,
    baseline_results: list[dict[str, Any]],
    guided_results: list[dict[str, Any]],
    model_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the required Exp 1144 artifact payload."""

    delta = carnot_guided_completion_rate - baseline_completion_rate
    if inference_mode != "live_gpu":
        verdict = "mock_inference_only"
    elif delta > 0:
        verdict = "carnot_positive_delta"
    elif delta < 0:
        verdict = "carnot_negative"
    else:
        verdict = "carnot_neutral"

    return {
        "cctu_adapter_written": True,
        "n_tasks_defined": int(n_tasks_defined),
        "n_tasks_evaluated": int(n_tasks_evaluated),
        "model_used": model_used,
        "inference_mode": inference_mode,
        "constraint_types_tested": sorted(constraint_types_tested),
        "per_constraint_tp_rate": dict(sorted(per_constraint_tp_rate.items())),
        "baseline_completion_rate": round(float(baseline_completion_rate), 6),
        "carnot_guided_completion_rate": round(float(carnot_guided_completion_rate), 6),
        "carnot_delta_pp": round(float(delta), 6),
        "cctu_adapter_honest_result": True,
        "honest_verdict": verdict,
        "baseline_results": baseline_results,
        "guided_results": guided_results,
        "model_resolution": model_resolution or {},
    }


def run_micro_benchmark(
    *,
    data_path: Path,
    output_path: Path,
    force_mock: bool = False,
    template: Any | None = None,
) -> dict[str, Any]:
    """Run baseline and Carnot-guided conditions and write the result artifact."""

    tasks = load_or_write_tasks(data_path)
    runner, model_resolution = _select_runner(force_mock=force_mock)
    baseline_results: list[dict[str, Any]] = []
    guided_results: list[dict[str, Any]] = []
    guided_by_type: dict[str, list[int]] = {}

    for task in tasks:
        prompt = build_prompt(task)
        baseline_response = runner.generate(task, prompt, guided=False)
        baseline_eval = evaluate_response(task, baseline_response)
        if baseline_eval.passed:
            guided_response = baseline_response
            guided_eval = baseline_eval
            repair_used = False
        else:
            guided_response = runner.generate(task, repair_prompt(task, baseline_eval), guided=True)
            guided_eval = evaluate_response(task, guided_response)
            repair_used = True

        baseline_results.append(_result_row(task, baseline_eval, repair_used=False))
        guided_results.append(_result_row(task, guided_eval, repair_used=repair_used))
        for constraint_type, (passed, total) in guided_eval.by_type.items():
            bucket = guided_by_type.setdefault(constraint_type, [0, 0])
            bucket[0] += passed
            bucket[1] += total

    n = len(tasks)
    baseline_completion_rate = sum(row["passed"] for row in baseline_results) / n if n else 0.0
    guided_completion_rate = sum(row["passed"] for row in guided_results) / n if n else 0.0
    per_constraint_tp_rate = {
        constraint_type: passed / total if total else 0.0
        for constraint_type, (passed, total) in guided_by_type.items()
    }
    artifact = build_exp1144_artifact(
        n_tasks_defined=len(tasks),
        n_tasks_evaluated=len(guided_results),
        model_used=runner.model_used,
        inference_mode=runner.inference_mode,
        constraint_types_tested=sorted(guided_by_type),
        per_constraint_tp_rate=per_constraint_tp_rate,
        baseline_completion_rate=baseline_completion_rate,
        carnot_guided_completion_rate=guided_completion_rate,
        baseline_results=baseline_results,
        guided_results=guided_results,
        model_resolution=model_resolution,
    )
    if template is not None:
        artifact = template.build_result(
            artifact,
            status="success",
            decision_class=["verify", "repair"],
            metrics_used=["constraint_tp_rate", "completion_rate"],
            code_files=["python/carnot/eval/cctu_micro_benchmark_adapter.py"],
            data_path=str(data_path),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _tool_call_names_for_task(task: dict[str, Any]) -> list[str]:
    required = str(task["available_tools"][0])
    count = int(task["constraints"][2]["validator"]["count"])
    names = [required]
    while len(names) < count:
        names.append("calculate")
    return names


def _check_constraint(
    task: dict[str, Any], constraint: dict[str, Any], response: str
) -> ConstraintCheck:
    spec = constraint["validator"]
    name = spec["name"]
    if name == "numeric_score_range":
        score = _extract_score(response)
        passed = score is not None and int(spec["min"]) <= score <= int(spec["max"])
        detail = f"score={score!r}, expected_range=[{spec['min']}, {spec['max']}]"
    elif name == "contains_expected_answer":
        expected = str(task["expected_answer"])
        passed = _normalise(expected) in _normalise(response)
        detail = f"expected_answer={expected!r}"
    elif name == "tool_call_protocol":
        names = _extract_tool_call_names(response)
        passed = len(names) == int(spec["count"]) and str(spec["required_tool"]) in names
        detail = f"tool_calls={names}, expected_count={spec['count']}"
    elif name == "format_style":
        passed, detail = _check_format_style(task, response, str(spec["style"]))
    elif name == "word_count_max":
        count = len(response.split())
        passed = count <= int(spec["max"])
        detail = f"word_count={count}, max={spec['max']}"
    else:
        passed = False
        detail = f"unknown validator {name!r}"
    return ConstraintCheck(
        constraint_id=str(constraint["id"]),
        constraint_type=str(constraint["type"]),
        passed=passed,
        detail=detail,
    )


def _check_format_style(task: dict[str, Any], response: str, style: str) -> tuple[bool, str]:
    answer = str(task["expected_answer"])
    if style == "markdown_bold":
        return f"**{answer}**" in response, "expected bold markdown answer"
    if style == "json_answer":
        try:
            obj = json.loads(response)
        except json.JSONDecodeError as exc:
            return False, f"invalid json: {exc.msg}"
        return obj.get("answer") == answer, "expected JSON object with matching answer"
    if style == "python_code":
        code = _extract_code_block(response)
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return False, f"invalid python code: {exc.msg}"
        return answer in response, "expected parseable Python block containing answer"
    if style == "answer_prefix":
        return bool(re.search(r"(^|\n)FINAL:\s*ANSWER:", response)), "expected FINAL: ANSWER:"
    if style == "table":
        has_table = bool(re.search(r"(?m)^\|\s*answer\s*\|\s*score\s*\|", response))
        return has_table and answer in response, "expected markdown answer table"
    return False, f"unknown style {style!r}"


def _extract_score(response: str) -> int | None:
    try:
        obj = json.loads(response)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("score"), int | float):
        return int(obj["score"])
    match = re.search(r"\bSCORE\s*:\s*(-?\d+)\b", response, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_tool_call_names(response: str) -> list[str]:
    names = re.findall(r"(?m)^TOOL_CALL\s+([A-Za-z_]\w*)\b", response)
    try:
        obj = json.loads(response)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
        for call in obj["tool_calls"]:
            if isinstance(call, dict) and isinstance(call.get("name"), str):
                names.append(call["name"])
    return names


def _extract_code_block(response: str) -> str:
    match = re.search(r"```(?:python|py)?\s*(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else response


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _result_row(
    task: dict[str, Any],
    evaluation: ResponseEvaluation,
    *,
    repair_used: bool,
) -> dict[str, Any]:
    row = evaluation.to_dict()
    row.update(
        {
            "task_id": task["task_id"],
            "source_row_id": task["source"]["source_row_id"],
            "repair_used": repair_used,
        }
    )
    return row


def _select_runner(force_mock: bool) -> tuple[ResponseRunner, dict[str, Any]]:
    preferred = MANDATED_SOTA_MODELS[0]
    if force_mock or os.getenv("CARNOT_CCTU_FORCE_MOCK") == "1":
        return MockCCTURunner(preferred), {
            "loader_path": "mock",
            "llama_cpp_loader_attempted": False,
            "reason": "forced_mock",
        }
    return _select_live_or_mock_runner()  # pragma: no cover - avoids loading GGUF in unit tests.


def _select_live_or_mock_runner() -> tuple[ResponseRunner, dict[str, Any]]:  # pragma: no cover
    last_error = None
    for model_id in MANDATED_SOTA_MODELS:
        model_path = resolve_cached_gguf(model_id)
        if model_path is None:
            last_error = f"{model_id} not cached"
            continue
        try:
            return LlamaCppCCTURunner(model_id, model_path), {
                "loader_path": "carnot.pipeline.gemma4_quantized_loader.Gemma4QuantizedLoader",
                "llama_cpp_loader_attempted": True,
                "model_path": model_path,
                "loader_status": "live",
            }
        except Exception as exc:
            last_error = str(exc)
    return MockCCTURunner(MANDATED_SOTA_MODELS[0]), {
        "loader_path": "carnot.pipeline.gemma4_quantized_loader.Gemma4QuantizedLoader",
        "llama_cpp_loader_attempted": True,
        "loader_status": "mock_fallback",
        "loader_error": last_error,
    }


__all__ = [
    "ALLOWED_HONEST_VERDICTS",
    "CCTU_SOURCE_DATASET_ID",
    "MANDATED_SOTA_MODELS",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_exp1144_artifact",
    "build_micro_benchmark_tasks",
    "build_prompt",
    "compliant_response_for_task",
    "evaluate_response",
    "load_or_write_tasks",
    "mock_baseline_response_for_task",
    "repair_prompt",
    "run_micro_benchmark",
    "write_tasks_json",
]

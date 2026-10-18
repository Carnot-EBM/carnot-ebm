"""SWE-Bench Lite helpers for Exp 1742 verify-repair evaluation.

The module keeps the experiment script thin: dataset rows are normalized here,
candidate patches are checked before evaluation, model attempts are recorded in
a stable schema, and the terminal JSON payload is assembled in one place.

Spec: REQ-BENCH-1742, SCENARIO-BENCH-1742
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

EXPERIMENT_ID = "1742"
SPEC_REFS = ["REQ-BENCH-1742", "SCENARIO-BENCH-1742"]
SWE_BENCH_LITE_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_TARGET_INSTANCE_IDS: tuple[str, ...] = (
    "django__django-11099",
    "sympy__sympy-11400",
    "matplotlib__matplotlib-18869",
    "pytest-dev__pytest-11143",
    "sphinx-doc__sphinx-10325",
)
DEFAULT_MODEL_HF_IDS: tuple[str, str] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)


@dataclass(frozen=True)
class SweBenchProblem:
    """Normalized subset of one SWE-Bench Lite row."""

    dataset_idx: int
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    version: str
    environment_setup_commit: str

    def to_metadata(self) -> dict[str, Any]:
        """Return the fields that belong in the terminal artifact."""
        return {
            "dataset_idx": self.dataset_idx,
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "version": self.version,
            "environment_setup_commit": self.environment_setup_commit,
            "fail_to_pass": list(self.fail_to_pass),
            "pass_to_pass": list(self.pass_to_pass),
        }


@dataclass(frozen=True)
class PatchVerification:
    """Bounded Carnot patch-check result before expensive SWE-Bench evaluation."""

    accepted: bool
    n_constraints: int
    violations: list[str]
    feedback: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "n_constraints": self.n_constraints,
            "violations": list(self.violations),
            "feedback": list(self.feedback),
        }


@dataclass(frozen=True)
class PatchEvaluation:
    """Result from a SWE-Bench-compatible patch evaluator."""

    resolved: bool
    status: str
    fail_to_pass_passed: bool = False
    pass_to_pass_passed: bool = False
    error_type: str = "none"
    error_message: str = ""
    stdout: str = ""
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "status": self.status,
            "fail_to_pass_passed": self.fail_to_pass_passed,
            "pass_to_pass_passed": self.pass_to_pass_passed,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stdout": self.stdout,
            "report_path": self.report_path,
        }


GeneratorFn = Callable[..., str]
EvaluatorFn = Callable[[SweBenchProblem, str, str], PatchEvaluation]


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _problem_from_row(row: Mapping[str, Any], dataset_idx: int) -> SweBenchProblem:
    return SweBenchProblem(
        dataset_idx=dataset_idx,
        instance_id=str(row["instance_id"]),
        repo=str(row["repo"]),
        base_commit=str(row.get("base_commit", "")),
        problem_statement=str(row.get("problem_statement", "")),
        hints_text=str(row.get("hints_text", "")),
        test_patch=str(row.get("test_patch", "")),
        fail_to_pass=_parse_json_list(row.get("FAIL_TO_PASS", [])),
        pass_to_pass=_parse_json_list(row.get("PASS_TO_PASS", [])),
        version=str(row.get("version", "")),
        environment_setup_commit=str(row.get("environment_setup_commit", "")),
    )


def _load_rows_with_datasets(split: str) -> list[Mapping[str, Any]]:  # pragma: no cover
    from datasets import load_dataset

    dataset = load_dataset(SWE_BENCH_LITE_DATASET, split=split)
    return [dict(dataset[idx]) for idx in range(len(dataset))]


def _fetch_rows_from_viewer(split: str) -> list[Mapping[str, Any]]:  # pragma: no cover
    rows: list[Mapping[str, Any]] = []
    offset = 0
    page_size = 100
    encoded_dataset = urllib.parse.quote(SWE_BENCH_LITE_DATASET, safe="")
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": encoded_dataset,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": page_size,
            },
            safe="%",
        )
        with urllib.request.urlopen(
            f"https://datasets-server.huggingface.co/rows?{query}",
            timeout=30,
        ) as handle:
            payload = json.load(handle)
        page = [item["row"] for item in payload.get("rows", [])]
        rows.extend(page)
        if offset + len(page) >= int(payload.get("num_rows_total", len(rows))) or not page:
            return rows
        offset += len(page)


def load_swebench_lite_problems(
    *,
    rows: Sequence[Mapping[str, Any]] | None = None,
    split: str = "test",
    target_instance_ids: Sequence[str] = DEFAULT_TARGET_INSTANCE_IDS,
    limit: int = 5,
    dataset_loader: Callable[[str], Sequence[Mapping[str, Any]]] | None = None,
    viewer_fetcher: Callable[[str], Sequence[Mapping[str, Any]]] | None = None,
) -> list[SweBenchProblem]:
    """Load and deterministically select the targeted SWE-Bench Lite cohort."""
    source_rows: Sequence[Mapping[str, Any]]
    if rows is not None:
        source_rows = rows
    else:
        loader = dataset_loader or _load_rows_with_datasets
        try:
            source_rows = loader(split)
        except ImportError:
            fetcher = viewer_fetcher or _fetch_rows_from_viewer
            source_rows = fetcher(split)

    problems = [_problem_from_row(row, idx) for idx, row in enumerate(source_rows)]
    by_id = {problem.instance_id: problem for problem in problems}
    selected = [
        by_id[instance_id] for instance_id in target_instance_ids if instance_id in by_id
    ][:limit]
    if len(selected) < limit:
        selected_ids = {problem.instance_id for problem in selected}
        selected.extend(
            problem
            for problem in problems
            if problem.instance_id not in selected_ids
        )
    return selected[:limit]


def build_swebench_prompt(problem: SweBenchProblem, *, eqm_decoding_enabled: bool) -> str:
    """Build the baseline patch-generation prompt for one SWE-Bench Lite issue."""
    fail_to_pass = "\n".join(f"- {test}" for test in problem.fail_to_pass) or "- unavailable"
    pass_to_pass = "\n".join(f"- {test}" for test in problem.pass_to_pass) or "- unavailable"
    hints = problem.hints_text.strip() or "(none)"
    eqm = str(bool(eqm_decoding_enabled)).lower()
    return "\n".join(
        [
            "You are resolving a SWE-Bench Lite Python issue.",
            "Return ONLY a unified diff patch. Do not include markdown or commentary.",
            f"EqM decoding enabled: {eqm}",
            "",
            f"Repository: {problem.repo}",
            f"Base commit: {problem.base_commit}",
            f"Instance: {problem.instance_id}",
            "",
            "Issue:",
            problem.problem_statement.strip(),
            "",
            "Hints:",
            hints,
            "",
            "Fail-to-pass tests:",
            fail_to_pass,
            "",
            "Pass-to-pass tests:",
            pass_to_pass,
        ]
    )


def extract_unified_diff(raw_output: str) -> str:
    """Extract a unified diff from model output, stripping fences and prose."""
    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)```", raw_output, flags=re.DOTALL)
    text = fenced.group(1) if fenced else raw_output
    lines = text.strip().splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("--- "):
            start_idx = idx
            break
    if start_idx is None:
        return text.strip()
    return "\n".join(lines[start_idx:]).strip()


def _modified_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) >= 4:
            for raw_path in parts[2:4]:
                path = raw_path[2:] if raw_path.startswith(("a/", "b/")) else raw_path
                paths.append(path)
    return paths


def _is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        path.startswith("tests/")
        or "/tests/" in path
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def verify_patch_candidate(problem: SweBenchProblem, patch: str) -> PatchVerification:
    """Run bounded patch checks before invoking the expensive evaluator."""
    del problem
    normalized = extract_unified_diff(patch)
    violations: list[str] = []
    feedback: list[str] = []
    n_constraints = 4

    if not normalized:
        violations.append("empty_patch")
        feedback.append("The model returned no patch text.")
        return PatchVerification(False, n_constraints, violations, feedback)

    has_header = normalized.startswith("diff --git ") or (
        "--- " in normalized and "+++ " in normalized and "@@" in normalized
    )
    if not has_header:
        violations.append("missing_unified_diff_header")
        feedback.append("Return a valid unified diff beginning with diff --git or ---/+++ headers.")

    has_edit = any(
        line.startswith(("+", "-"))
        and not line.startswith(("+++", "---"))
        for line in normalized.splitlines()
    )
    if not has_edit:
        violations.append("no_code_edits")
        feedback.append("The patch must contain at least one added or removed source line.")

    paths = _modified_paths(normalized)
    if paths and all(_is_test_path(path) for path in paths):
        violations.append("test_only_patch")
        feedback.append("SWE-Bench predictions must fix implementation code, not only tests.")

    return PatchVerification(not violations, n_constraints, violations, feedback)


def _blocked_evaluation(error_type: str, error_message: str) -> PatchEvaluation:
    return PatchEvaluation(
        resolved=False,
        status="blocked",
        error_type=error_type,
        error_message=error_message,
    )


def _attempt_patch(
    problem: SweBenchProblem,
    *,
    model_name: str,
    raw_output: str,
    evaluator: EvaluatorFn,
    attempt_index: int,
) -> dict[str, Any]:
    patch = extract_unified_diff(raw_output)
    verification = verify_patch_candidate(problem, patch)
    if verification.accepted:
        evaluation = evaluator(problem, patch, model_name)
    else:
        evaluation = _blocked_evaluation(
            "patch_verification_failed",
            "; ".join(verification.feedback),
        )
    return {
        "attempt_index": attempt_index,
        "patch": patch,
        "verification": verification.to_dict(),
        "evaluation": evaluation.to_dict(),
    }


def build_repair_prompt(
    problem: SweBenchProblem,
    previous_patch: str,
    evaluation: PatchEvaluation,
    verification: PatchVerification,
    repair_idx: int,
) -> str:
    """Build a patch-repair prompt from evaluator and Carnot verifier feedback."""
    feedback = verification.feedback or [evaluation.error_message or evaluation.error_type]
    return "\n".join(
        [
            "Repair the previous SWE-Bench Lite unified diff patch.",
            f"Repair attempt: {repair_idx + 1}",
            f"Instance: {problem.instance_id}",
            "",
            "Issue:",
            problem.problem_statement.strip(),
            "",
            "Previous patch:",
            previous_patch.strip() or "(empty)",
            "",
            "Evaluator status:",
            f"{evaluation.status}: {evaluation.error_message or evaluation.error_type}",
            "",
            "Carnot patch feedback:",
            *[f"- {item}" for item in feedback],
            "",
            "Return ONLY the corrected unified diff patch.",
        ]
    )


def _call_generator(
    generator: GeneratorFn,
    prompt: str,
    *,
    model_name: str,
    eqm_decoding_enabled: bool,
) -> str:
    return generator(
        prompt,
        model_name=model_name,
        eqm_decoding_enabled=eqm_decoding_enabled,
    )


def run_verify_repair_case(
    problem: SweBenchProblem,
    *,
    model_name: str,
    generator: GeneratorFn,
    evaluator: EvaluatorFn,
    max_repairs: int,
    eqm_decoding_enabled: bool,
) -> dict[str, Any]:
    """Run baseline patch generation plus bounded verify-repair for one issue."""
    baseline_prompt = build_swebench_prompt(problem, eqm_decoding_enabled=eqm_decoding_enabled)
    baseline_raw = _call_generator(
        generator,
        baseline_prompt,
        model_name=model_name,
        eqm_decoding_enabled=eqm_decoding_enabled,
    )
    attempts = [
        _attempt_patch(
            problem,
            model_name=model_name,
            raw_output=baseline_raw,
            evaluator=evaluator,
            attempt_index=0,
        )
    ]

    current = attempts[0]
    current_evaluation = PatchEvaluation(**current["evaluation"])
    current_verification = PatchVerification(**current["verification"])
    current_patch = str(current["patch"])

    repaired = False
    for repair_idx in range(max_repairs):
        if current_evaluation.resolved:
            break
        repair_prompt = build_repair_prompt(
            problem,
            current_patch,
            current_evaluation,
            current_verification,
            repair_idx,
        )
        raw_repair = _call_generator(
            generator,
            repair_prompt,
            model_name=model_name,
            eqm_decoding_enabled=eqm_decoding_enabled,
        )
        current = _attempt_patch(
            problem,
            model_name=model_name,
            raw_output=raw_repair,
            evaluator=evaluator,
            attempt_index=repair_idx + 1,
        )
        attempts.append(current)
        current_evaluation = PatchEvaluation(**current["evaluation"])
        current_verification = PatchVerification(**current["verification"])
        current_patch = str(current["patch"])
        repaired = current_evaluation.resolved
        if repaired:
            break

    baseline_eval = attempts[0]["evaluation"]
    final_eval = attempts[-1]["evaluation"]
    return {
        "instance_id": problem.instance_id,
        "repo": problem.repo,
        "dataset_idx": problem.dataset_idx,
        "baseline": {
            "resolved": bool(baseline_eval["resolved"]),
            "evaluation_status": baseline_eval["status"],
            "error_type": baseline_eval["error_type"],
            "error_message": baseline_eval["error_message"],
            "verification": attempts[0]["verification"],
        },
        "verify_repair": {
            "resolved": bool(final_eval["resolved"]),
            "evaluation_status": final_eval["status"],
            "repaired": repaired and not bool(baseline_eval["resolved"]),
            "n_repairs": len(attempts) - 1,
            "error_type": final_eval["error_type"],
            "error_message": final_eval["error_message"],
        },
        "attempts": attempts,
    }


def run_model_on_problems(
    problems: Sequence[SweBenchProblem],
    *,
    model_spec: Mapping[str, str],
    generator: GeneratorFn,
    evaluator: EvaluatorFn,
    max_repairs: int,
    eqm_decoding_enabled: bool,
) -> dict[str, Any]:
    """Run one model over the selected Exp 1742 problem slice."""
    cases = [
        run_verify_repair_case(
            problem,
            model_name=str(model_spec["name"]),
            generator=generator,
            evaluator=evaluator,
            max_repairs=max_repairs,
            eqm_decoding_enabled=eqm_decoding_enabled,
        )
        for problem in problems
    ]
    return {
        "model_name": str(model_spec["name"]),
        "model_hf_id": str(model_spec["hf_id"]),
        "n_cases": len(cases),
        "baseline_resolved": sum(1 for case in cases if case["baseline"]["resolved"]),
        "verify_repair_resolved": sum(
            1 for case in cases if case["verify_repair"]["resolved"]
        ),
        "cases": cases,
    }


def summarize_model_results(model_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate resolve-rate metrics across model results."""
    n_models = len(model_results)
    n_instances = int(model_results[0]["n_cases"]) if model_results else 0
    total_cases = sum(int(result["n_cases"]) for result in model_results)
    baseline_resolved = sum(int(result["baseline_resolved"]) for result in model_results)
    verify_repair_resolved = sum(
        int(result["verify_repair_resolved"]) for result in model_results
    )
    rates_available = total_cases > 0
    return {
        "n_models": n_models,
        "n_instances": n_instances,
        "total_model_instance_pairs": total_cases,
        "baseline_resolved": baseline_resolved,
        "verify_repair_resolved": verify_repair_resolved,
        "baseline_resolve_rate": baseline_resolved / total_cases if rates_available else None,
        "verify_repair_resolve_rate": (
            verify_repair_resolved / total_cases if rates_available else None
        ),
        "signed_improvement": (
            (verify_repair_resolved - baseline_resolved) / total_cases
            if rates_available
            else None
        ),
        "headline_resolve_rates_available": rates_available,
    }


def build_results_payload(
    *,
    status: str,
    honest_verdict: str,
    timestamp: str,
    runtime_seconds: float,
    selected_problems: Sequence[SweBenchProblem],
    model_results: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    blockers: Sequence[str],
    evaluator_backend: str,
    eqm_decoding_enabled: bool,
    max_repairs: int = 1,
) -> dict[str, Any]:
    """Build the Exp 1742 terminal JSON artifact."""
    return {
        "experiment_id": EXPERIMENT_ID,
        "title": "SWE-Bench Lite verify-repair baseline with EqM decoding disabled",
        "status": status,
        "honest_verdict": honest_verdict,
        "spec_refs": list(SPEC_REFS),
        "metadata": {
            "timestamp": timestamp,
            "runtime_seconds": runtime_seconds,
        },
        "dataset": {
            "source": SWE_BENCH_LITE_DATASET,
            "split": "test",
            "targeted": True,
            "selected_instance_ids": [problem.instance_id for problem in selected_problems],
            "selected_problems": [problem.to_metadata() for problem in selected_problems],
        },
        "config": {
            "eqm_decoding_enabled": bool(eqm_decoding_enabled),
            "baseline_condition": "sota_greedy_no_eqm",
            "max_repairs": max_repairs,
            "evaluator_backend": evaluator_backend,
            "target_model_hf_ids": list(DEFAULT_MODEL_HF_IDS),
        },
        "metrics": dict(metrics),
        "blockers": list(blockers),
        "per_model_results": [dict(result) for result in model_results],
    }

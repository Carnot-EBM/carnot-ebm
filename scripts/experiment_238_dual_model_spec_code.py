#!/usr/bin/env python3
"""Experiment 238: paired spec-aware HumanEval benchmark on Qwen and Gemma.

Reuses the checked-in Exp 227 cohort so `Qwen/Qwen3.5-0.8B` and
`google/gemma-4-E4B-it` run on the same ordered HumanEval slice with the same
prompt seeds, the same repair budget, and the same verifier stack:
official tests, additive PBT, and additive explicit specs.

Spec: REQ-CODE-028, REQ-CODE-029, REQ-CODE-030,
SCENARIO-CODE-026, SCENARIO-CODE-027, SCENARIO-CODE-028
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

_humaneval_live_benchmark = import_module("carnot.pipeline.humaneval_live_benchmark")
PBTCodeVerifier = import_module("carnot.pipeline.pbt_code_verifier").PBTCodeVerifier
SpecCodeVerifier = import_module("carnot.pipeline.spec_code_verifier").SpecCodeVerifier

RUN_DATE = "20260413"
EXPERIMENT_ID = 238
DEFAULT_MAX_REPAIRS = 3
DEFAULT_MAX_NEW_TOKENS = 220
DEFAULT_PBT_MAX_EXAMPLES = 64
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_CHECKPOINT_INTERVAL = 10
RESULTS_DIR = REPO_ROOT / "results"
VERIFIER_STACK = ["official_tests", "pbt", "explicit_specs"]
MODEL_SPECS: tuple[dict[str, str], ...] = (
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B"},
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it"},
)
HarnessResult = _humaneval_live_benchmark.HarnessResult
bootstrap_ci = _humaneval_live_benchmark.bootstrap_ci
bootstrap_delta_ci = _humaneval_live_benchmark.bootstrap_delta_ci
build_candidate_code = _humaneval_live_benchmark.build_candidate_code
execute_humaneval = _humaneval_live_benchmark.execute_humaneval
run_instrumentation = _humaneval_live_benchmark.run_instrumentation


def get_repo_root() -> Path:
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return REPO_ROOT


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else get_repo_root() / candidate


def _display_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(get_repo_root().resolve()))
    except ValueError:
        return str(candidate)


def default_reference_artifact_path() -> Path:
    """Return the checked-in Exp 227 reference artifact path."""
    return RESULTS_DIR / "experiment_227_results.json"


def default_output_path() -> Path:
    """Return the default Exp 238 artifact path."""
    return RESULTS_DIR / "experiment_238_results.json"


def default_checkpoint_dir() -> Path:
    """Return the default Exp 238 checkpoint directory."""
    return RESULTS_DIR / "checkpoints" / "experiment_238"


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_parser() -> argparse.ArgumentParser:
    """Build the Exp 238 CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a paired spec-aware HumanEval benchmark on the checked-in "
            "Exp 227 cohort for Qwen/Qwen3.5-0.8B and google/gemma-4-E4B-it."
        ),
    )
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        default=default_reference_artifact_path(),
        help="Reference artifact whose checked-in cohort is reused verbatim.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Artifact path for results/experiment_238_results.json.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=default_checkpoint_dir(),
        help="Directory for model-specific resume checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Save a resume checkpoint every N completed cases per model.",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=DEFAULT_MAX_REPAIRS,
        help="Maximum repair attempts per failing spec-aware case.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated tokens for baseline and repair generations.",
    )
    parser.add_argument(
        "--pbt-max-examples",
        type=int,
        default=DEFAULT_PBT_MAX_EXAMPLES,
        help="Hypothesis max_examples for the additive PBT verifier.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap sample count for paired confidence intervals.",
    )
    return parser


def load_shared_cohort(
    reference_artifact: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the checked-in shared cohort from the reference artifact."""
    path = resolve_path(reference_artifact)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Reference artifact must be a JSON object: {_display_path(path)}")
    cohort = payload.get("cohort")
    if not isinstance(cohort, dict):
        raise ValueError(f"Reference artifact missing valid cohort block: {_display_path(path)}")
    cases = cohort.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Reference artifact missing valid cohort cases: {_display_path(path)}")

    loaded_cases: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"Reference artifact has malformed cohort case: {_display_path(path)}")
        prompt_seeds = case.get("prompt_seeds")
        if not isinstance(prompt_seeds, dict):
            raise ValueError(f"Reference artifact case missing prompt seeds: {_display_path(path)}")
        baseline_seed = prompt_seeds.get("baseline")
        verify_only_seed = prompt_seeds.get("verify_only")
        verify_repair_seed = prompt_seeds.get("verify_repair")
        if baseline_seed != verify_only_seed or verify_only_seed != verify_repair_seed:
            raise ValueError(
                f"Reference artifact case has mismatched prompt seeds: {_display_path(path)}"
            )
        loaded_cases.append(dict(case))
    return (
        loaded_cases,
        {
            "source_artifact": str(path),
            "source_experiment": int(payload.get("experiment", 0)),
            "reference_experiment": int(
                payload.get("metadata", {}).get("reference_experiment", 208)
            ),
            "reference_run_date": str(payload.get("run_date", "")),
            "case_count": len(loaded_cases),
        },
    )


def checkpoint_path(checkpoint_dir: str | Path, model_name: str) -> Path:
    """Return the per-model checkpoint path."""
    return _checkpoint_path(checkpoint_dir, model_name)


def _checkpoint_path(checkpoint_dir: str | Path, model_name: str) -> Path:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in model_name).strip("_")
    return Path(checkpoint_dir) / f"{slug}.json"


def _model_checkpoint_path(checkpoint_dir: str | Path, model_name: str) -> Path:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in model_name).strip("_")
    return Path(checkpoint_dir) / f"humaneval_spec_dual__{slug}.json"


def load_checkpoint(path: str | Path, expected_case_ids: list[str]) -> dict[str, Any]:
    """Load a checkpoint only when the cohort metadata still matches."""
    checkpoint = resolve_path(path)
    fresh = {
        "case_ids": list(expected_case_ids),
        "results_by_case": {},
    }
    if not checkpoint.exists():
        return fresh

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if payload.get("case_ids") != expected_case_ids:
        return fresh
    results_by_case = payload.get("results_by_case")
    if not isinstance(results_by_case, dict):
        return fresh
    return {
        "case_ids": list(expected_case_ids),
        "results_by_case": dict(results_by_case),
    }


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist a checkpoint atomically with a trailing newline."""
    checkpoint = resolve_path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp_path.replace(checkpoint)


def _seed_runtime(seed: int) -> None:  # pragma: no cover
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed & 0xFFFFFFFF)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _best_cuda_device() -> str:  # pragma: no cover
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for Exp 238 live inference.")

    best_index = 0
    best_free = -1
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(index)
        except Exception:
            free_bytes = 0
        if free_bytes > best_free:
            best_free = free_bytes
            best_index = index
    return f"cuda:{best_index}"


def _load_live_model(model_hf_id: str) -> tuple[Any, Any, str]:  # pragma: no cover
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    os.environ["CARNOT_FORCE_CPU"] = "0"

    from carnot.inference.model_loader import load_model

    device_str = _best_cuda_device()
    model, tokenizer = load_model(model_hf_id, device=device_str)
    if model is None or tokenizer is None:
        raise RuntimeError(f"Failed to load live model: {model_hf_id}")
    return model, tokenizer, device_str


def _unload_live_model(model: Any, tokenizer: Any) -> None:  # pragma: no cover
    del model, tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _generate_text(  # pragma: no cover
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    prompt_seed: int,
    max_new_tokens: int,
) -> str:
    _seed_runtime(prompt_seed)
    from carnot.inference.model_loader import generate

    return str(generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens))


def build_generation_prompt(case: dict[str, Any]) -> str:
    """Build the baseline code-generation prompt for one HumanEval case."""
    return (
        "You are an expert Python programmer.\n"
        "Complete the following function.\n"
        "Return ONLY the function body lines. No def line. No markdown fences.\n"
        "Indent with 4 spaces.\n\n"
        f"{case['prompt']}"
    )


def build_spec_verifier(pbt_max_examples: int) -> SpecCodeVerifier:
    """Build the explicit-spec verifier used for the additive spec layer."""
    verifier = SpecCodeVerifier(include_official_tests=False, include_pbt=False)
    verifier._pbt_verifier = PBTCodeVerifier(max_examples=pbt_max_examples)  # type: ignore[attr-defined]
    return verifier


def _serialize_pbt_properties(properties: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(getattr(prop, "name", "")),
            "source": str(getattr(prop, "source", "")),
            "description": str(getattr(prop, "description", "")),
        }
        for prop in properties
    ]


def _serialize_pbt_failures(failures: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "property_name": str(getattr(failure, "property_name", "")),
            "source": str(getattr(failure, "source", "")),
            "description": str(getattr(failure, "description", "")),
            "input_args": list(getattr(failure, "input_args", ()) or ()),
            "actual": str(getattr(failure, "actual", "")),
            "expected": str(getattr(failure, "expected", "")),
            "error": getattr(failure, "error", None),
        }
        for failure in failures
    ]


def _serialize_repair_hints(hints: list[Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for hint in hints:
        if hasattr(hint, "to_dict"):
            serialized.append(dict(hint.to_dict()))
        elif isinstance(hint, dict):
            serialized.append(dict(hint))
    return serialized


def _spec_violation_texts(spec_result: Any) -> list[str]:
    violations: list[str] = []
    for clause in getattr(spec_result, "spec_clause_results", ()):
        if getattr(clause, "status", "") != "violated":
            continue
        constraint = (
            clause.to_constraint_result() if hasattr(clause, "to_constraint_result") else None
        )
        if constraint is not None and getattr(constraint, "description", ""):
            violations.append(str(constraint.description))
        else:
            violations.append(
                f"{getattr(clause, 'kind', '')} ({getattr(clause, 'family', '')}) failed: "
                f"{getattr(clause, 'detail', '')}"
            )
    return violations


def evaluate_candidate(
    case: dict[str, Any],
    candidate_code: str,
    *,
    pbt_max_examples: int,
) -> dict[str, Any]:
    """Run official tests, instrumentation, additive PBT, and additive explicit specs."""
    started = time.perf_counter()
    official = execute_humaneval(candidate_code, case, timeout=5.0)
    instrumentation = run_instrumentation(
        candidate_code,
        str(case["prompt"]),
        str(case["entry_point"]),
        official_tests=None,
    )
    pbt_result = PBTCodeVerifier(max_examples=pbt_max_examples).verify(
        candidate_code,
        str(case["prompt"]),
        str(case["entry_point"]),
        str(case["test"]),
    )
    spec_result = build_spec_verifier(pbt_max_examples).verify(
        candidate_code,
        str(case["prompt"]),
        str(case["entry_point"]),
        str(case["test"]),
        task_id=str(case["task_id"]),
        case_id=str(case["case_id"]),
    )

    explicit_violations = _spec_violation_texts(spec_result)
    stage_acceptance = {
        "official_tests_verify_only": bool(official.passed),
        "pbt_verify_only": bool(official.passed and pbt_result.verified),
        "spec_aware_verify_only": bool(
            official.passed and pbt_result.verified and not explicit_violations
        ),
    }
    return {
        "official_tests": {
            "passed": bool(official.passed),
            "error_type": str(official.error_type),
            "error_message": str(official.error_message),
            "stdout": str(official.stdout),
        },
        "instrumentation": dict(instrumentation),
        "pbt": {
            "verified": bool(pbt_result.verified),
            "n_failures": len(pbt_result.failures),
            "violations": [result.description for result in pbt_result.to_constraint_results()],
            "repair_feedback": pbt_result.repair_feedback(),
            "derived_properties": _serialize_pbt_properties(list(pbt_result.derived_properties)),
            "failure_records": _serialize_pbt_failures(list(pbt_result.failures)),
            "wall_clock_seconds": round(float(pbt_result.wall_clock_seconds), 6),
            "max_examples": int(getattr(pbt_result, "max_examples", pbt_max_examples)),
        },
        "explicit_specs": {
            "matched": getattr(spec_result, "spec", None) is not None,
            "n_violations": len(explicit_violations),
            "violations": explicit_violations,
            "repair_hints": _serialize_repair_hints(getattr(spec_result, "repair_hints", ())),
        },
        "stage_acceptance": stage_acceptance,
        "latency_seconds": round(time.perf_counter() - started, 6),
    }


def build_spec_repair_prompt(
    case: dict[str, Any],
    *,
    previous_body: str,
    evaluation: dict[str, Any],
    repair_idx: int,
) -> str:
    """Build a shared repair prompt with official, PBT, and explicit-spec feedback."""
    official = evaluation["official_tests"]
    instrumentation = evaluation["instrumentation"]
    pbt = evaluation["pbt"]
    explicit_specs = evaluation["explicit_specs"]

    lines = [
        f"You are fixing a Python function (repair attempt {repair_idx + 1}).",
        "",
        "Function prompt:",
        str(case["prompt"]).rstrip(),
        "",
        "Previous function body:",
        "    " + (previous_body.strip() or "pass").replace("\n", "\n    "),
        "",
        "Official test failure:",
        f"  - {official.get('error_message') or official.get('error_type')}",
    ]

    for heading, key in (
        ("Static constraint findings:", "constraint_feedback"),
        ("Runtime instrumentation findings:", "dynamic_violations"),
    ):
        findings = list(instrumentation.get(key, []))
        if findings:
            lines.extend(["", heading])
            lines.extend(f"  - {finding}" for finding in findings[:5])

    pbt_lines = list(pbt.get("violations", []))
    if pbt_lines:
        lines.extend(["", "Hypothesis-backed PBT counterexamples:"])
        lines.extend(f"  - {line}" for line in pbt_lines[:5])

    spec_lines = list(explicit_specs.get("violations", []))
    if spec_lines:
        lines.extend(["", "Explicit spec findings:"])
        lines.extend(f"  - {line}" for line in spec_lines[:5])

    repair_hints = list(explicit_specs.get("repair_hints", []))
    if repair_hints:
        lines.extend(["", "Trace-ranked repair hints:"])
        for hint in repair_hints[:5]:
            strategy_name = str(hint.get("strategy_name", "")).strip()
            rationale = str(hint.get("rationale", "")).strip()
            rendered = strategy_name or "generic_fix"
            if rationale:
                rendered = f"{rendered}: {rationale}"
            lines.append(f"  - {rendered}")

    repair_feedback = str(pbt.get("repair_feedback", "")).strip()
    if repair_feedback:
        lines.extend(["", "PBT repair feedback:", repair_feedback])

    lines.extend(
        [
            "",
            "Write ONLY the corrected function body. No markdown fences.",
            "Indent with 4 spaces.",
        ]
    )
    return "\n".join(lines)


def _baseline_record(body: str, candidate_code: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "official_passed": bool(evaluation["official_tests"]["passed"]),
        "body": body,
        "candidate_code": candidate_code,
    }


def _history_entry(
    *,
    iteration: int,
    body: str,
    candidate_code: str,
    evaluation: dict[str, Any],
    repair_prompt: str | None = None,
) -> dict[str, Any]:
    entry = {
        "iteration": iteration,
        "body": body,
        "candidate_code": candidate_code,
        "evaluation": dict(evaluation),
    }
    if repair_prompt is not None:
        entry["repair_prompt"] = repair_prompt
    return entry


def run_case(
    case: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    device_str: str,
    max_repairs: int,
    pbt_max_examples: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run baseline generation plus spec-aware verify-repair for one case."""
    del device_str
    baseline_prompt = build_generation_prompt(case)
    baseline_body = _generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=baseline_prompt,
        prompt_seed=int(case["prompt_seeds"]["baseline"]),
        max_new_tokens=max_new_tokens,
    )
    baseline_code = build_candidate_code(str(case["prompt"]), baseline_body)
    baseline_eval = evaluate_candidate(case, baseline_code, pbt_max_examples=pbt_max_examples)

    result = {
        "case_id": str(case["case_id"]),
        "dataset_idx": int(case["dataset_idx"]),
        "task_id": str(case["task_id"]),
        "entry_point": str(case["entry_point"]),
        "baseline": _baseline_record(baseline_body, baseline_code, baseline_eval),
        "official_tests_verify_only": {
            "accepted": bool(baseline_eval["stage_acceptance"]["official_tests_verify_only"]),
        },
        "pbt_verify_only": {
            "accepted": bool(baseline_eval["stage_acceptance"]["pbt_verify_only"]),
            "harness_passing_rejected_by_pbt": bool(
                baseline_eval["official_tests"]["passed"]
                and not baseline_eval["stage_acceptance"]["pbt_verify_only"]
            ),
        },
        "spec_aware_verify_only": {
            "accepted": bool(baseline_eval["stage_acceptance"]["spec_aware_verify_only"]),
            "harness_passing_rejected_by_specs": bool(
                baseline_eval["stage_acceptance"]["pbt_verify_only"]
                and not baseline_eval["stage_acceptance"]["spec_aware_verify_only"]
            ),
        },
        "verify_repair": {
            "accepted": bool(baseline_eval["stage_acceptance"]["spec_aware_verify_only"]),
            "official_passed": bool(baseline_eval["official_tests"]["passed"]),
            "repaired": False,
            "n_repairs": 0,
            "final_body": baseline_body,
            "final_code": baseline_code,
        },
        "history": [
            _history_entry(
                iteration=0,
                body=baseline_body,
                candidate_code=baseline_code,
                evaluation=baseline_eval,
            )
        ],
    }

    if bool(baseline_eval["stage_acceptance"]["spec_aware_verify_only"]):
        return result

    current_body = baseline_body
    current_code = baseline_code
    current_eval = baseline_eval

    for repair_idx in range(max_repairs):
        repair_prompt = build_spec_repair_prompt(
            case,
            previous_body=current_body,
            evaluation=current_eval,
            repair_idx=repair_idx,
        )
        current_body = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=repair_prompt,
            prompt_seed=int(case["prompt_seeds"]["verify_repair"]) + repair_idx + 1,
            max_new_tokens=max_new_tokens,
        )
        current_code = build_candidate_code(str(case["prompt"]), current_body)
        current_eval = evaluate_candidate(case, current_code, pbt_max_examples=pbt_max_examples)
        result["history"].append(
            _history_entry(
                iteration=repair_idx + 1,
                body=current_body,
                candidate_code=current_code,
                evaluation=current_eval,
                repair_prompt=repair_prompt,
            )
        )
        result["verify_repair"].update(
            {
                "accepted": bool(current_eval["stage_acceptance"]["spec_aware_verify_only"]),
                "official_passed": bool(current_eval["official_tests"]["passed"]),
                "repaired": bool(current_eval["stage_acceptance"]["spec_aware_verify_only"]),
                "n_repairs": repair_idx + 1,
                "final_body": current_body,
                "final_code": current_code,
            }
        )
        if bool(current_eval["stage_acceptance"]["spec_aware_verify_only"]):
            break

    return result


def run_benchmark(
    cases: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    device_str: str,
    checkpoint_path: str | Path,
    checkpoint_interval: int,
    max_repairs: int,
    pbt_max_examples: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Run one model over the shared cohort with checkpointing and resume."""
    case_ids = [str(case["case_id"]) for case in cases]
    checkpoint = load_checkpoint(checkpoint_path, case_ids)
    results_by_case = dict(checkpoint["results_by_case"])
    since_last_checkpoint = 0

    for case in cases:
        case_id = str(case["case_id"])
        if case_id in results_by_case:
            continue
        try:
            results_by_case[case_id] = run_case(
                case,
                model=model,
                tokenizer=tokenizer,
                device_str=device_str,
                max_repairs=max_repairs,
                pbt_max_examples=pbt_max_examples,
                max_new_tokens=max_new_tokens,
            )
        except Exception:
            save_checkpoint(
                checkpoint_path,
                {
                    "case_ids": case_ids,
                    "results_by_case": results_by_case,
                },
            )
            raise
        since_last_checkpoint += 1
        if since_last_checkpoint >= checkpoint_interval:
            save_checkpoint(
                checkpoint_path,
                {
                    "case_ids": case_ids,
                    "results_by_case": results_by_case,
                },
            )
            since_last_checkpoint = 0

    if since_last_checkpoint > 0 or not resolve_path(checkpoint_path).exists():
        save_checkpoint(
            checkpoint_path,
            {
                "case_ids": case_ids,
                "results_by_case": results_by_case,
            },
        )

    return [dict(results_by_case[case_id]) for case_id in case_ids]


def _stage_flags(case: dict[str, Any]) -> dict[str, bool]:
    return {
        "baseline": bool(case["baseline"]["official_passed"]),
        "official_tests_verify_only": bool(case["official_tests_verify_only"]["accepted"]),
        "pbt_verify_only": bool(case["pbt_verify_only"]["accepted"]),
        "spec_aware_verify_only": bool(case["spec_aware_verify_only"]["accepted"]),
        "verify_repair": bool(case["verify_repair"]["accepted"]),
    }


def _stage_summary(flags: list[bool], *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    point, ci_lower, ci_upper = bootstrap_ci(flags, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "accepted_pass_at_1": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _zero_stage_summary() -> dict[str, Any]:
    return {
        "accepted_pass_at_1": 0.0,
        "ci_lower": 0.0,
        "ci_upper": 0.0,
    }


def _delta_summary(
    baseline_flags: list[bool],
    comparison_flags: list[bool],
    *,
    n_bootstrap: int,
    seed: int,
    delta_key: str,
) -> dict[str, float]:
    point, ci_lower, ci_upper = bootstrap_delta_ci(
        baseline_flags,
        comparison_flags,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        delta_key: point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _zero_comparison_summary(repair_budget: int) -> dict[str, Any]:
    zero_delta = {"gemma_minus_qwen": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    zero_outcome = {
        "gemma_only": 0,
        "qwen_only": 0,
        "both": 0,
        "neither": 0,
    }
    return {
        "paired_case_count": 0,
        "shared_repair_budget": repair_budget,
        "shared_verifier_stack": list(VERIFIER_STACK),
        "stage_deltas": {
            "baseline": dict(zero_delta),
            "official_tests_verify_only": dict(zero_delta),
            "pbt_verify_only": dict(zero_delta),
            "spec_aware_verify_only": dict(zero_delta),
            "verify_repair": dict(zero_delta),
        },
        "stage_outcomes": {
            "baseline": dict(zero_outcome),
            "official_tests_verify_only": dict(zero_outcome),
            "pbt_verify_only": dict(zero_outcome),
            "spec_aware_verify_only": dict(zero_outcome),
            "verify_repair": dict(zero_outcome),
        },
        "technical_report_summary": {"paragraph": "No paired cases completed.", "bullets": []},
    }


def summarize_model_results(
    cases: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize one model's stage-wise outcomes and paired deltas."""
    if not cases:
        return {
            "stages": {
                "baseline": _zero_stage_summary(),
                "official_tests_verify_only": _zero_stage_summary(),
                "pbt_verify_only": _zero_stage_summary() | {"harness_passing_rejected_by_pbt": 0},
                "spec_aware_verify_only": _zero_stage_summary()
                | {
                    "added_rejections_over_pbt": 0,
                    "harness_passing_rejected_by_specs": 0,
                },
                "verify_repair": _zero_stage_summary(),
            },
            "paired_deltas": {
                "spec_over_pbt": {"delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
                "repair_over_baseline": {"delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            },
            "verify_repair": {"n_repaired": 0, "accepted_pass_at_1": 0.0},
            "technical_report_summary": {"paragraph": "No completed cases.", "bullets": []},
        }

    flags_by_stage = {stage: [] for stage in VERIFIER_STACK}
    all_stage_flags = {
        "baseline": [],
        "official_tests_verify_only": [],
        "pbt_verify_only": [],
        "spec_aware_verify_only": [],
        "verify_repair": [],
    }
    del flags_by_stage
    for case in cases:
        flags = _stage_flags(case)
        for stage, accepted in flags.items():
            all_stage_flags[stage].append(accepted)

    stages = {
        stage: _stage_summary(all_stage_flags[stage], n_bootstrap=n_bootstrap, seed=seed + idx)
        for idx, stage in enumerate(all_stage_flags)
    }
    stages["spec_aware_verify_only"]["added_rejections_over_pbt"] = sum(
        1
        for case in cases
        if bool(case["pbt_verify_only"]["accepted"])
        and not bool(case["spec_aware_verify_only"]["accepted"])
    )

    paired_deltas = {
        "spec_over_pbt": _delta_summary(
            all_stage_flags["pbt_verify_only"],
            all_stage_flags["spec_aware_verify_only"],
            n_bootstrap=n_bootstrap,
            seed=seed + 10,
            delta_key="delta",
        ),
        "repair_over_baseline": _delta_summary(
            all_stage_flags["baseline"],
            all_stage_flags["verify_repair"],
            n_bootstrap=n_bootstrap,
            seed=seed + 11,
            delta_key="delta",
        ),
    }
    verify_repair = {
        "n_repaired": sum(1 for case in cases if bool(case["verify_repair"]["repaired"])),
        "accepted_pass_at_1": stages["verify_repair"]["accepted_pass_at_1"],
    }
    baseline_pct = stages["baseline"]["accepted_pass_at_1"] * 100.0
    pbt_pct = stages["pbt_verify_only"]["accepted_pass_at_1"] * 100.0
    spec_pct = stages["spec_aware_verify_only"]["accepted_pass_at_1"] * 100.0
    spec_over_pbt_pp = paired_deltas["spec_over_pbt"]["delta"] * 100.0
    paragraph = (
        f"Baseline accepted pass@1 was {baseline_pct:.1f}%, "
        f"additive PBT verify-only reached {pbt_pct:.1f}%, "
        f"and the explicit spec layer moved accepted pass@1 to {spec_pct:.1f}% "
        f"({spec_over_pbt_pp:+.1f}pp vs PBT)."
    )
    return {
        "stages": stages,
        "paired_deltas": paired_deltas,
        "verify_repair": verify_repair,
        "technical_report_summary": {
            "paragraph": paragraph,
            "bullets": [],
        },
    }


def build_comparison_summary(
    model_runs: dict[str, dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
    repair_budget: int,
) -> dict[str, Any]:
    """Build the apples-to-apples Gemma-versus-Qwen comparison block."""
    gemma = model_runs.get("Gemma4-E4B-it", {})
    qwen = model_runs.get("Qwen3.5-0.8B", {})
    gemma_results = gemma.get("per_problem_results", [])
    qwen_results = qwen.get("per_problem_results", [])
    if not isinstance(gemma_results, list):
        return _zero_comparison_summary(repair_budget)
    if not isinstance(qwen_results, list):
        return _zero_comparison_summary(repair_budget)
    gemma_lookup = {str(case["case_id"]): case for case in gemma_results if isinstance(case, dict)}
    qwen_lookup = {str(case["case_id"]): case for case in qwen_results if isinstance(case, dict)}
    paired_case_ids = [case_id for case_id in qwen_lookup if case_id in gemma_lookup]
    if not paired_case_ids:
        return _zero_comparison_summary(repair_budget)

    stage_deltas: dict[str, Any] = {}
    stage_outcomes: dict[str, Any] = {}
    for idx, stage in enumerate(
        (
            "baseline",
            "official_tests_verify_only",
            "pbt_verify_only",
            "spec_aware_verify_only",
            "verify_repair",
        )
    ):
        qwen_flags = [_stage_flags(qwen_lookup[case_id])[stage] for case_id in paired_case_ids]
        gemma_flags = [_stage_flags(gemma_lookup[case_id])[stage] for case_id in paired_case_ids]
        stage_deltas[stage] = _delta_summary(
            qwen_flags,
            gemma_flags,
            n_bootstrap=n_bootstrap,
            seed=seed + idx,
            delta_key="gemma_minus_qwen",
        )
        stage_outcomes[stage] = {
            "gemma_only": sum(
                1
                for gemma_flag, qwen_flag in zip(gemma_flags, qwen_flags, strict=True)
                if gemma_flag and not qwen_flag
            ),
            "qwen_only": sum(
                1
                for gemma_flag, qwen_flag in zip(gemma_flags, qwen_flags, strict=True)
                if qwen_flag and not gemma_flag
            ),
            "both": sum(
                1
                for gemma_flag, qwen_flag in zip(gemma_flags, qwen_flags, strict=True)
                if gemma_flag and qwen_flag
            ),
            "neither": sum(
                1
                for gemma_flag, qwen_flag in zip(gemma_flags, qwen_flags, strict=True)
                if not gemma_flag and not qwen_flag
            ),
        }

    spec_delta_pp = stage_deltas["spec_aware_verify_only"]["gemma_minus_qwen"] * 100.0
    paragraph = (
        f"Across {len(paired_case_ids)} paired cases with the same official-test, "
        "PBT, and explicit spec stack plus the same repair budget, "
        f"the explicit spec layer changed Gemma-versus-Qwen accepted pass@1 by "
        f"{spec_delta_pp:+.1f}pp."
    )
    return {
        "paired_case_count": len(paired_case_ids),
        "shared_repair_budget": repair_budget,
        "shared_verifier_stack": list(VERIFIER_STACK),
        "stage_deltas": stage_deltas,
        "stage_outcomes": stage_outcomes,
        "technical_report_summary": {
            "paragraph": paragraph,
            "bullets": [],
        },
    }


def build_artifact_payload(
    *,
    output_path: Path,
    cohort: list[dict[str, Any]],
    cohort_meta: dict[str, Any],
    model_runs: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    blockers: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    runtime_seconds: float,
    checkpoint_dir: Path,
    max_repairs: int,
    pbt_max_examples: int,
    bootstrap_samples: int,
    run_status: str,
) -> dict[str, Any]:
    """Build the final Exp 238 artifact payload."""
    return {
        "experiment": EXPERIMENT_ID,
        "benchmark": "humaneval_dual_model_spec",
        "run_date": RUN_DATE,
        "schema": {
            "artifact": "carnot.humaneval_dual_model_spec.v1",
            "benchmark_case_schema": "humaneval_dual_model_spec.v1",
        },
        "metadata": {
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_seconds": round(runtime_seconds, 3),
            "checkpoint_dir": str(checkpoint_dir),
            "max_repairs": max_repairs,
            "pbt_max_examples": pbt_max_examples,
            "bootstrap_samples": bootstrap_samples,
            "output_path": str(output_path),
            "source_artifacts": [
                cohort_meta["source_artifact"],
                "data/research/code_spec_corpus_236.jsonl",
                "results/experiment_225_results.json",
                "results/experiment_226_results.json",
                "results/experiment_227_results.json",
            ],
        },
        "cohort": {
            "case_count": len(cohort),
            "case_ids": [str(case["case_id"]) for case in cohort],
            "task_ids": [str(case["task_id"]) for case in cohort],
            "cases": [dict(case) for case in cohort],
            "shared_with_reference_artifact": True,
        },
        "model_runs": dict(model_runs),
        "comparison": dict(comparison),
        "blockers": list(blockers),
        "run_status": run_status,
    }


def _run_live_benchmark(args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover
    started_at = utc_now()
    start = time.perf_counter()
    cohort, cohort_meta = load_shared_cohort(args.reference_artifact)
    model_runs: dict[str, dict[str, Any]] = {}
    blockers: list[dict[str, Any]] = []

    for model_spec in MODEL_SPECS:
        cell = run_model_benchmark_cell(
            model_spec=model_spec,
            cohort=cohort,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            max_repairs=args.max_repairs,
            pbt_max_examples=args.pbt_max_examples,
            max_new_tokens=args.max_new_tokens,
            bootstrap_samples=args.bootstrap_samples,
        )
        model_runs[str(model_spec["name"])] = cell
        blockers.extend(cell.get("blockers", []))

    comparison = build_comparison_summary(
        model_runs,
        n_bootstrap=args.bootstrap_samples,
        seed=13,
        repair_budget=args.max_repairs,
    )
    statuses = {str(run.get("run_status", "")) for run in model_runs.values()}
    if statuses == {"complete"}:
        run_status = "complete"
    elif statuses <= {"blocked"}:
        run_status = "blocked"
    else:
        run_status = "partial"
    return build_artifact_payload(
        output_path=args.output,
        cohort=cohort,
        cohort_meta=cohort_meta,
        model_runs=model_runs,
        comparison=comparison,
        blockers=blockers,
        started_at=started_at,
        finished_at=utc_now(),
        runtime_seconds=time.perf_counter() - start,
        checkpoint_dir=args.checkpoint_dir,
        max_repairs=args.max_repairs,
        pbt_max_examples=args.pbt_max_examples,
        bootstrap_samples=args.bootstrap_samples,
        run_status=run_status,
    )


def run_model_benchmark_cell(
    *,
    model_spec: dict[str, str],
    cohort: list[dict[str, Any]],
    checkpoint_dir: Path,
    checkpoint_interval: int,
    max_repairs: int,
    pbt_max_examples: int,
    max_new_tokens: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Run one model cell and preserve blockers honestly."""
    model_name = str(model_spec["name"])
    ckpt_path = _checkpoint_path(checkpoint_dir, model_name)
    model = tokenizer = None
    device = ""
    blockers: list[dict[str, Any]] = []
    try:
        model, tokenizer, device = _load_live_model(str(model_spec["hf_id"]))
        results = run_benchmark(
            cohort,
            model=model,
            tokenizer=tokenizer,
            device_str=device,
            checkpoint_path=ckpt_path,
            checkpoint_interval=checkpoint_interval,
            max_repairs=max_repairs,
            pbt_max_examples=pbt_max_examples,
            max_new_tokens=max_new_tokens,
        )
        run_status = "complete"
    except Exception as error:
        checkpoint = load_checkpoint(ckpt_path, [str(case["case_id"]) for case in cohort])
        results = [
            dict(checkpoint["results_by_case"][case["case_id"]])
            for case in cohort
            if case["case_id"] in checkpoint["results_by_case"]
        ]
        run_status = "partial" if results else "blocked"
        blockers.append(
            {
                "model_name": model_name,
                "stage": "model_load" if model is None else "run_benchmark",
                "error": str(error),
            }
        )
    finally:
        if model is not None and tokenizer is not None:
            _unload_live_model(model, tokenizer)

    return {
        "model_name": model_name,
        "model_hf_id": str(model_spec["hf_id"]),
        "device": device,
        "run_status": run_status,
        "completed_case_count": len(results),
        "pending_case_count": len(cohort) - len(results),
        "blockers": blockers,
        "checkpoint_path": str(ckpt_path),
        "statistics": summarize_model_results(
            results,
            n_bootstrap=bootstrap_samples,
            seed=7 if model_name.startswith("Qwen") else 11,
        ),
        "per_problem_results": results,
    }


def write_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    artifact_path = resolve_path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = _run_live_benchmark(args)
    output_path = payload.get("metadata", {}).get("output_path", str(args.output))
    write_artifact(output_path, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

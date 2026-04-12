"""Reusable helpers for the Exp 208 live HumanEval benchmark.

This module keeps the experiment script thin and pushes deterministic,
testable logic into the Python package so the HumanEval live benchmark can be
verified with full unit coverage before the GPU run happens.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-006
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from carnot.pipeline.extract import CodeExtractor

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from experiment_53_runtime_constraints import (  # type: ignore[import-not-found]  # noqa: E402
    execute_instrumented,
    instrument_code,
)


@dataclass(frozen=True)
class HarnessResult:
    """Outcome of running one candidate solution against HumanEval tests."""

    passed: bool
    error_type: str
    error_message: str
    stdout: str


def sample_problems(
    problems: list[dict[str, Any]],
    sample_size: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic shuffled sample of HumanEval problems."""
    shuffled = list(problems)
    random.Random(sample_seed).shuffle(shuffled)
    return shuffled[:sample_size]


def build_candidate_code(prompt: str, raw_body: str) -> str:
    """Attach a generated function body to the HumanEval prompt."""
    lines = []
    for line in raw_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        if stripped.startswith("def "):
            continue
        lines.append(line)

    body = textwrap.dedent("\n".join(lines)).strip("\n")
    if not body.strip():
        body = "pass"

    return f"{prompt}{textwrap.indent(body, '    ')}\n"


def execute_humaneval(
    code: str,
    problem: dict[str, Any],
    timeout: float = 5.0,
) -> HarnessResult:
    """Execute candidate code against the official HumanEval check() harness."""
    full_source = f"{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix="exp208_",
    ) as handle:
        handle.write(full_source)
        temp_path = handle.name

    try:
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (proc.stdout + proc.stderr)[:2000]
        if proc.returncode == 0:
            return HarnessResult(
                passed=True,
                error_type="none",
                error_message="",
                stdout=output,
            )

        lines = [line for line in output.splitlines() if line.strip()]
        error_message = lines[-1] if lines else "execution failed"
        return HarnessResult(
            passed=False,
            error_type="failure",
            error_message=error_message,
            stdout=output,
        )
    except subprocess.TimeoutExpired:
        return HarnessResult(
            passed=False,
            error_type="timeout",
            error_message=f"execution timeout after {timeout}s",
            stdout="",
        )
    finally:
        with suppress(OSError):
            os.unlink(temp_path)


def _split_signature_args(args_text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0

    for char in args_text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)

        if char == "," and depth == 0:
            candidate = "".join(current).strip()
            if candidate:
                parts.append(candidate)
            current = []
            continue
        current.append(char)

    candidate = "".join(current).strip()
    if candidate:
        parts.append(candidate)
    return parts


def _annotation_kind(annotation: str) -> str:
    normalized = annotation.strip().lower()
    if "dict" in normalized:
        return "dict"
    if "tuple" in normalized:
        return "tuple"
    if "list" in normalized or "sequence" in normalized:
        return "list"
    if "bool" in normalized:
        return "bool"
    if "float" in normalized:
        return "float"
    if "str" in normalized:
        return "str"
    return "int"


def generate_probe_inputs(prompt: str, entry_point: str) -> list[dict[str, Any]]:
    """Generate a small deterministic probe set for runtime instrumentation."""
    match = re.search(
        rf"def\s+{re.escape(entry_point)}\s*\((.*?)\)\s*(?:->|:)",
        prompt,
        flags=re.DOTALL,
    )
    params: list[tuple[str, str]] = []
    if match:
        for chunk in _split_signature_args(match.group(1)):
            name_text, _, ann_text = chunk.partition(":")
            name = name_text.strip()
            if not name or name == "self":
                continue
            params.append((name, _annotation_kind(ann_text)))

    if not params:
        return [{}]

    variants: dict[str, list[Any]] = {
        "int": [1, 0, -1],
        "float": [1.0, 0.0, -1.0],
        "str": ["x", "", " "],
        "list": [[1, 2], [], [0]],
        "bool": [True, False, True],
        "dict": [{"key": 1}, {}, {"key": 0}],
        "tuple": [(1,), (), (0, 1)],
    }

    probes: list[dict[str, Any]] = []
    for probe_idx in range(3):
        probe: dict[str, Any] = {}
        for name, kind in params:
            probe[name] = variants[kind][probe_idx]
        probes.append(probe)
    return probes


def run_instrumentation(
    code: str,
    prompt: str,
    entry_point: str,
) -> dict[str, Any]:
    """Run CodeExtractor plus Exp 53 runtime instrumentation on candidate code."""
    constraints = CodeExtractor().extract(code, domain="code")
    static_violations = [
        result.description for result in constraints if result.metadata.get("satisfied") is False
    ]
    probes = generate_probe_inputs(prompt, entry_point)
    instrumented = instrument_code(code)
    dynamic_result = execute_instrumented(instrumented, probes, entry_point)
    dynamic_violations = list(dynamic_result.get("violations", []))
    dynamic_count = int(dynamic_result.get("n_fail", 0))
    if dynamic_violations and dynamic_count == 0:
        dynamic_count = len(dynamic_violations)

    return {
        "n_constraints": len(constraints),
        "constraint_feedback": (
            static_violations or [result.description for result in constraints]
        )[:5],
        "n_static_violations": len(static_violations),
        "static_violations": static_violations[:5],
        "n_dynamic_violations": dynamic_count,
        "dynamic_violations": dynamic_violations[:5],
        "probe_inputs": probes,
        "detected": bool(static_violations or dynamic_violations),
    }


def build_repair_prompt(
    prompt: str,
    previous_body: str,
    harness_result: HarnessResult,
    instrumentation: dict[str, Any],
    repair_idx: int,
) -> str:
    """Build a code-repair prompt using tests plus instrumentation feedback."""
    previous = textwrap.dedent(previous_body).strip() or "pass"
    feedback_lines = [
        f"You are fixing a Python function (repair attempt {repair_idx + 1}).",
        "",
        "Function prompt:",
        prompt.rstrip(),
        "",
        "Previous function body:",
        textwrap.indent(previous, "    "),
        "",
        "HumanEval test failure:",
        f"  - {harness_result.error_message or harness_result.error_type}",
    ]

    if instrumentation.get("constraint_feedback"):
        feedback_lines.extend(["", "Static constraint findings:"])
        feedback_lines.extend(f"  - {line}" for line in instrumentation["constraint_feedback"][:5])

    if instrumentation.get("dynamic_violations"):
        feedback_lines.extend(["", "Runtime instrumentation findings:"])
        feedback_lines.extend(f"  - {line}" for line in instrumentation["dynamic_violations"][:5])

    feedback_lines.extend(
        [
            "",
            "Write ONLY the corrected function body. No markdown fences.",
            "Indent with 4 spaces.",
        ]
    )
    return "\n".join(feedback_lines)


def bootstrap_ci(
    flags: list[bool],
    n_bootstrap: int = 10_000,
    seed: int = 208,
) -> tuple[float, float, float]:
    """Compute a percentile bootstrap CI for a binary proportion."""
    arr = np.array(flags, dtype=float)
    point = float(arr.mean())
    samples = (
        np.random.default_rng(seed)
        .choice(
            arr,
            size=(n_bootstrap, len(arr)),
            replace=True,
        )
        .mean(axis=1)
    )
    return (
        point,
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def bootstrap_delta_ci(
    baseline_flags: list[bool],
    repair_flags: list[bool],
    n_bootstrap: int = 10_000,
    seed: int = 209,
) -> tuple[float, float, float]:
    """Compute a paired bootstrap CI for verify-repair improvement."""
    baseline = np.array(baseline_flags, dtype=float)
    repair = np.array(repair_flags, dtype=float)
    point = float((repair - baseline).mean())
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstrap):
        indices = rng.choice(len(baseline), size=len(baseline), replace=True)
        deltas.append(float((repair[indices] - baseline[indices]).mean()))
    delta_array = np.array(deltas)
    return (
        point,
        float(np.percentile(delta_array, 2.5)),
        float(np.percentile(delta_array, 97.5)),
    )


def summarize_cases(
    cases: list[dict[str, Any]],
    n_bootstrap: int = 10_000,
    seed: int = 208,
) -> dict[str, Any]:
    """Summarize pass@1 baseline vs verify-repair over the selected cohort."""
    if not cases:
        raise ValueError("Cannot summarize empty benchmark cohort.")

    baseline_flags = [bool(case["baseline"]["passed"]) for case in cases]
    repair_flags = [bool(case["verify_repair"]["passed"]) for case in cases]

    base_acc, base_lo, base_hi = bootstrap_ci(
        baseline_flags,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    repair_acc, repair_lo, repair_hi = bootstrap_ci(
        repair_flags,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )
    delta, delta_lo, delta_hi = bootstrap_delta_ci(
        baseline_flags,
        repair_flags,
        n_bootstrap=n_bootstrap,
        seed=seed + 2,
    )

    repairs = [
        int(case["verify_repair"].get("n_repairs", 0))
        for case in cases
        if int(case["verify_repair"].get("n_repairs", 0)) > 0
    ]
    n_needing_repair = sum(1 for flag in baseline_flags if not flag)
    n_repaired = sum(1 for case in cases if bool(case["verify_repair"].get("repaired", False)))

    return {
        "n_problems": len(cases),
        "baseline": {
            "pass_at_1": base_acc,
            "ci_lower": base_lo,
            "ci_upper": base_hi,
            "n_correct": int(sum(baseline_flags)),
        },
        "verify_repair": {
            "pass_at_1": repair_acc,
            "ci_lower": repair_lo,
            "ci_upper": repair_hi,
            "n_correct": int(sum(repair_flags)),
        },
        "improvement": {
            "delta": delta,
            "ci_lower": delta_lo,
            "ci_upper": delta_hi,
        },
        "repair_stats": {
            "n_problems_needing_repair": n_needing_repair,
            "n_repaired": n_repaired,
            "repair_success_rate": (n_repaired / n_needing_repair if n_needing_repair else 1.0),
            "avg_repair_iterations": float(np.mean(repairs)) if repairs else 0.0,
        },
        "instrumentation": {
            "problems_with_static_violations": sum(
                1 for case in cases if int(case["baseline"].get("n_static_violations", 0)) > 0
            ),
            "problems_with_dynamic_violations": sum(
                1 for case in cases if int(case["baseline"].get("n_dynamic_violations", 0)) > 0
            ),
            "total_static_violations": sum(
                int(case["baseline"].get("n_static_violations", 0)) for case in cases
            ),
            "total_dynamic_violations": sum(
                int(case["baseline"].get("n_dynamic_violations", 0)) for case in cases
            ),
        },
    }


def build_results_payload(
    *,
    experiment: int,
    title: str,
    timestamp: str,
    model_name: str,
    hf_id: str,
    device: str,
    inference_mode: str,
    sample_seed: int,
    sample_size: int,
    sample_dataset_indices: list[int],
    sample_task_ids: list[str],
    max_new_tokens: int,
    max_repairs: int,
    runtime_seconds: float,
    statistics: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the final JSON payload for the Exp 208 artifact."""
    return {
        "experiment": experiment,
        "title": title,
        "metadata": {
            "timestamp": timestamp,
            "dataset_source": "HumanEval (openai_humaneval)",
            "model_name": model_name,
            "model_hf_id": hf_id,
            "device": device,
            "inference_mode": inference_mode,
            "sample_seed": sample_seed,
            "sample_size": sample_size,
            "sample_dataset_indices": sample_dataset_indices,
            "sample_task_ids": sample_task_ids,
            "max_new_tokens": max_new_tokens,
            "max_repairs": max_repairs,
            "runtime_seconds": runtime_seconds,
        },
        "statistics": statistics,
        "per_problem_results": cases,
    }

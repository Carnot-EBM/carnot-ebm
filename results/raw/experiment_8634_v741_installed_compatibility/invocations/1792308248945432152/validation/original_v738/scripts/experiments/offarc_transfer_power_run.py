"""Exp 4045 OFF-ARC full-power execution-verifier transfer runner.

Spec refs: REQ-VERIFY-4045, SCENARIO-VERIFY-4045.

This runner scales the Exp 4032 OFF-ARC measurement to a HumanEval plus MBPP
cohort and adds a symbolic-equivalence-partition arm. The GAP-4 demo-fit
primitive is intentionally unchanged in spirit: candidate programs are
generated once, executed in the restricted namespace, selected by exact visible
test behavior, and scored on held-out hidden tests only after selection.
"""

from __future__ import annotations

import argparse
import ast
import doctest
import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "results" / "experiment_4045_offarc_transfer_power_raw.json"
CHECKPOINT = REPO_ROOT / "results" / "experiment_4045_offarc_transfer_power.checkpoint.json"
HUMANEVAL_MANIFEST = REPO_ROOT / "data" / "eval_manifests" / "humaneval_20260522.jsonl"
MBPP_MANIFEST = REPO_ROOT / "data" / "eval_manifests" / "mbpp_20260522.jsonl"
GGUF_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--gemma-4-12B-it-GGUF"

RANDOM_SEED = 4045
DEFAULT_K = 8
DEFAULT_N_TASKS = 192
FULL_TASK_FLOOR = 160
FULL_K_FLOOR = 8
INFERENCE_SUBSTRATE = "live_llm_inference"

ARMS_IMPLEMENTED = [
    "armA_vote",
    "armAplusplus_aces",
    "armB_demofit",
    "armC_symbolic_equivalence_partition",
]

REQUIRED_RAW_FIELDS = [
    "honest_verdict",
    "corpus",
    "n_tasks",
    "target_completed_tasks",
    "k_candidates_per_task",
    "arms_implemented",
    "armA_vote_passrate",
    "armA_vote_pass2",
    "armAplusplus_aces_passrate",
    "armAplusplus_aces_pass2",
    "armB_demofit_passrate",
    "armB_demofit_pass2",
    "armC_symbolic_partition_passrate",
    "armC_symbolic_partition_pass2",
    "bootstrap_ci95",
    "bootstrap_ci95_armC_vs_armA",
    "oracle_passrate",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "truncation_rate",
    "preconditions_checked",
    "per_task",
    "candidate_pool",
    "skipped_tasks",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix verdict; success only when a powered arm beats vote with CI > 0.",
    "runner_pool": "All four arms select from the same generated candidates for each task.",
    "armB_demofit": "The shared GAP-4 primitive: exact visible-test passers only, no LLM judge.",
    "armC_symbolic_partition": "SEP proxy: public-test filter plus behavioral equivalence partition.",
    "oracle_passrate": "Positive control: whether any candidate in the same pool passes hidden tests.",
    "truncation_rate": "Fraction of local GGUF generations that ended by token limit.",
    "reproducibility_checksum": "SHA-256 over deterministic task, candidate, and metric content.",
}


@dataclass(frozen=True)
class CodeTest:
    source: str
    func_name: str
    args: tuple[Any, ...]
    expected: Any


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    corpus: str
    prompt: str
    func_name: str
    visible_tests: list[CodeTest]
    hidden_tests: list[CodeTest]


@dataclass(frozen=True)
class GeneratedCandidate:
    draw_index: int
    raw_text: str
    code: str
    generation_seconds: float
    finish_reason: str | None
    truncated: bool


@dataclass(frozen=True)
class CandidateEvaluation:
    task_id: str
    draw_index: int
    status: str
    code: str
    visible_passes: list[bool]
    hidden_passes: list[bool]
    visible_outputs: list[Any]
    hidden_outputs: list[Any]
    fingerprint_outputs: list[Any]
    generation_seconds: float
    truncated: bool
    error: str | None


def parse_assertion_test(source: str) -> CodeTest | None:
    """Parse a simple `assert f(args) == expected` exact-output test."""
    stripped = source.strip()
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    expr = tree.body[0].test
    if not (
        isinstance(expr, ast.Compare)
        and len(expr.ops) == 1
        and isinstance(expr.ops[0], ast.Eq)
        and len(expr.comparators) == 1
    ):
        return None
    parsed = _call_and_expected(expr.left, expr.comparators[0])
    if parsed is None:
        parsed = _call_and_expected(expr.comparators[0], expr.left)
    if parsed is None:
        return None
    func_name, args, expected = parsed
    return CodeTest(stripped, func_name, args, expected)


def parse_humaneval_candidate_assert(source: str, func_name: str) -> CodeTest | None:
    """Parse a HumanEval `assert candidate(args) == expected` as an exact test."""
    stripped = source.strip()
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    expr = tree.body[0].test
    if isinstance(expr, ast.Call) and _is_candidate_call(expr):
        args = _literal_args(expr)
        if args is not None:
            return CodeTest(f"assert {func_name}{_args_repr(args)} == True", func_name, args, True)
    if (
        isinstance(expr, ast.UnaryOp)
        and isinstance(expr.op, ast.Not)
        and isinstance(expr.operand, ast.Call)
        and _is_candidate_call(expr.operand)
    ):
        args = _literal_args(expr.operand)
        if args is not None:
            return CodeTest(
                f"assert {func_name}{_args_repr(args)} == False", func_name, args, False
            )
    if not (
        isinstance(expr, ast.Compare)
        and len(expr.ops) == 1
        and isinstance(expr.ops[0], ast.Eq)
        and len(expr.comparators) == 1
    ):
        return None
    parsed = _candidate_call_and_expected(expr.left, expr.comparators[0])
    if parsed is None:
        parsed = _candidate_call_and_expected(expr.comparators[0], expr.left)
    if parsed is None:
        return None
    args, expected = parsed
    return CodeTest(
        f"assert {func_name}{_args_repr(args)} == {expected!r}", func_name, args, expected
    )


def _call_and_expected(
    call_node: ast.AST, expected_node: ast.AST
) -> tuple[str, tuple[Any, ...], Any] | None:
    if not isinstance(call_node, ast.Call) or not isinstance(call_node.func, ast.Name):
        return None
    if call_node.keywords:
        return None
    args = _literal_args(call_node)
    if args is None:
        return None
    try:
        expected = ast.literal_eval(expected_node)
    except (SyntaxError, ValueError):
        return None
    return call_node.func.id, args, expected


def _candidate_call_and_expected(
    call_node: ast.AST, expected_node: ast.AST
) -> tuple[tuple[Any, ...], Any] | None:
    if not isinstance(call_node, ast.Call) or not _is_candidate_call(call_node):
        return None
    args = _literal_args(call_node)
    if args is None:
        return None
    try:
        expected = ast.literal_eval(expected_node)
    except (SyntaxError, ValueError):
        return None
    return args, expected


def _is_candidate_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "candidate" and not node.keywords


def _literal_args(node: ast.Call) -> tuple[Any, ...] | None:
    try:
        return tuple(ast.literal_eval(arg) for arg in node.args)
    except (SyntaxError, ValueError):
        return None


def _args_repr(args: tuple[Any, ...]) -> str:
    return "(" + ", ".join(repr(arg) for arg in args) + ("," if len(args) == 1 else "") + ")"


def extract_code_block(text: str, func_name: str) -> str:
    """Extract the generated Python function from fenced or plain model text."""
    fence = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    for match in fence.finditer(text):
        block = match.group(1).strip()
        if re.search(rf"\bdef\s+{re.escape(func_name)}\s*\(", block):
            return block
    match = re.search(rf"\bdef\s+{re.escape(func_name)}\s*\(", text)
    if match:
        return text[match.start() :].strip().strip("`")
    return ""


def _stable_repr(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)


def default_executor(  # pragma: no cover - live integration path.
    code: str, func_name: str, args: tuple[Any, ...], timeout: float
) -> tuple[Any, Exception | None]:
    from carnot.verify.sandbox import sandboxed_exec_function

    return sandboxed_exec_function(code, func_name, args, timeout=timeout, allow_fallback=True)


def evaluate_candidate(
    task: CodeTask,
    candidate: GeneratedCandidate,
    *,
    executor: Callable[
        [str, str, tuple[Any, ...], float], tuple[Any, Exception | None]
    ] = default_executor,
    timeout: float = 2.0,
    fingerprint_tests: list[CodeTest] | None = None,
) -> CandidateEvaluation:
    """Run one candidate on visible, hidden, and public-derived fingerprint tests."""
    if not candidate.code:
        fingerprint_count = len(fingerprint_tests or [])
        return CandidateEvaluation(
            task_id=task.task_id,
            draw_index=candidate.draw_index,
            status="no_code",
            code="",
            visible_passes=[False for _ in task.visible_tests],
            hidden_passes=[False for _ in task.hidden_tests],
            visible_outputs=["no_code" for _ in task.visible_tests],
            hidden_outputs=["no_code" for _ in task.hidden_tests],
            fingerprint_outputs=["no_code" for _ in range(fingerprint_count)],
            generation_seconds=candidate.generation_seconds,
            truncated=candidate.truncated,
            error="no_code",
        )

    visible_passes, visible_outputs, visible_error = _run_tests(
        candidate.code, task.visible_tests, executor=executor, timeout=timeout
    )
    hidden_passes, hidden_outputs, hidden_error = _run_tests(
        candidate.code, task.hidden_tests, executor=executor, timeout=timeout
    )
    fingerprint_outputs, fingerprint_error = _run_fingerprint_tests(
        candidate.code, fingerprint_tests or [], executor=executor, timeout=timeout
    )
    return CandidateEvaluation(
        task_id=task.task_id,
        draw_index=candidate.draw_index,
        status="ok",
        code=candidate.code,
        visible_passes=visible_passes,
        hidden_passes=hidden_passes,
        visible_outputs=visible_outputs,
        hidden_outputs=hidden_outputs,
        fingerprint_outputs=fingerprint_outputs,
        generation_seconds=candidate.generation_seconds,
        truncated=candidate.truncated,
        error=visible_error or hidden_error or fingerprint_error,
    )


def _run_tests(
    code: str,
    tests: list[CodeTest],
    *,
    executor: Callable[[str, str, tuple[Any, ...], float], tuple[Any, Exception | None]],
    timeout: float,
) -> tuple[list[bool], list[Any], str | None]:
    passes: list[bool] = []
    outputs: list[Any] = []
    first_error: str | None = None
    for test in tests:
        result, error = executor(code, test.func_name, test.args, timeout)
        if error is not None:
            err = f"{type(error).__name__}: {error}"
            passes.append(False)
            outputs.append({"error": err})
            first_error = first_error or err
            continue
        passes.append(result == test.expected)
        outputs.append(result)
    return passes, outputs, first_error


def _run_fingerprint_tests(
    code: str,
    tests: list[CodeTest],
    *,
    executor: Callable[[str, str, tuple[Any, ...], float], tuple[Any, Exception | None]],
    timeout: float,
) -> tuple[list[Any], str | None]:
    outputs: list[Any] = []
    first_error: str | None = None
    for test in tests:
        result, error = executor(code, test.func_name, test.args, timeout)
        if error is not None:
            err = f"{type(error).__name__}: {error}"
            outputs.append({"error": err})
            first_error = first_error or err
        else:
            outputs.append(result)
    return outputs, first_error


def aces_leave_one_out_scores(evaluations: list[CandidateEvaluation]) -> dict[int, float]:
    """Compute a bounded ACES-style score from visible pass/fail consistency."""
    if not evaluations:
        return {}
    n_tests = max((len(ev.visible_passes) for ev in evaluations), default=0)
    if n_tests == 0:
        return {ev.draw_index: 0.0 for ev in evaluations}
    if n_tests == 1:
        return {ev.draw_index: float(ev.visible_passes[0]) for ev in evaluations}

    weights: list[float] = []
    for held_out in range(n_tests):
        other_scores = []
        for ev in evaluations:
            other = [passed for i, passed in enumerate(ev.visible_passes) if i != held_out]
            other_scores.append(sum(other) / max(1, len(other)))
        best = max(other_scores)
        top = [
            ev.visible_passes[held_out]
            for ev, score in zip(evaluations, other_scores, strict=False)
            if score == best
        ]
        rest = [
            ev.visible_passes[held_out]
            for ev, score in zip(evaluations, other_scores, strict=False)
            if score != best
        ]
        top_rate = sum(top) / max(1, len(top))
        rest_rate = sum(rest) / max(1, len(rest))
        weights.append(max(0.0, top_rate - rest_rate))
    if not any(weights):
        weights = [1.0 for _ in range(n_tests)]

    denom = sum(weights)
    return {
        ev.draw_index: (
            sum(
                weight * float(passed)
                for weight, passed in zip(weights, ev.visible_passes, strict=False)
            )
            / denom
            if denom
            else 0.0
        )
        for ev in evaluations
    }


def score_evaluated_tasks(
    tasks: list[CodeTask],
    evaluations_by_task: dict[str, list[CandidateEvaluation]],
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Score vote, ACES, demo-fit, SEP proxy, and oracle on hidden tests."""
    per_task = []
    a1: list[int] = []
    a2: list[int] = []
    app1: list[int] = []
    app2: list[int] = []
    b1: list[int] = []
    b2: list[int] = []
    c1: list[int] = []
    c2: list[int] = []
    oracle: list[int] = []
    n_candidates = 0
    n_truncated = 0
    missing_gaps: list[str] = []

    for task in tasks:
        evaluations = list(evaluations_by_task.get(task.task_id, []))
        n_candidates += len(evaluations)
        n_truncated += sum(1 for ev in evaluations if ev.truncated)
        vote_rank = _rank_vote(evaluations)
        aces_rank = _rank_aces(evaluations)
        demofit_rank = _rank_demofit(evaluations)
        sep_rank, sep_meta = _rank_symbolic_partition(evaluations)
        oracle_hit = any(_hidden_all(ev) for ev in evaluations)

        arm_a_1 = _pass_at(vote_rank, 1)
        arm_a_2 = _pass_at(vote_rank, 2)
        arm_app_1 = _pass_at(aces_rank, 1)
        arm_app_2 = _pass_at(aces_rank, 2)
        arm_b_1 = _pass_at(demofit_rank, 1)
        arm_b_2 = _pass_at(demofit_rank, 2)
        arm_c_1 = _pass_at(sep_rank, 1)
        arm_c_2 = _pass_at(sep_rank, 2)
        if oracle_hit and not (arm_b_1 or arm_c_1):
            missing_gaps.append(task.task_id)

        a1.append(int(arm_a_1))
        a2.append(int(arm_a_2))
        app1.append(int(arm_app_1))
        app2.append(int(arm_app_2))
        b1.append(int(arm_b_1))
        b2.append(int(arm_b_2))
        c1.append(int(arm_c_1))
        c2.append(int(arm_c_2))
        oracle.append(int(oracle_hit))

        per_task.append(
            {
                "task_id": task.task_id,
                "corpus": task.corpus,
                "func_name": task.func_name,
                "n_candidates": len(evaluations),
                "n_visible_tests": len(task.visible_tests),
                "n_hidden_tests": len(task.hidden_tests),
                "armA_vote_selected": [ev.draw_index for ev in vote_rank[:2]],
                "armAplusplus_aces_selected": [ev.draw_index for ev in aces_rank[:2]],
                "armB_demofit_selected": [ev.draw_index for ev in demofit_rank[:2]],
                "armC_symbolic_partition_selected": [ev.draw_index for ev in sep_rank[:2]],
                "armB_demo_perfect_count": sum(1 for ev in evaluations if _demo_perfect(ev)),
                "armC_partition_count": sep_meta["partition_count"],
                "armC_dominant_partition_size": sep_meta["dominant_partition_size"],
                "armC_partition_signature_sha256": sep_meta["dominant_signature_sha256"],
                "armA_vote_pass1": arm_a_1,
                "armA_vote_pass2": arm_a_2,
                "armAplusplus_aces_pass1": arm_app_1,
                "armAplusplus_aces_pass2": arm_app_2,
                "armB_demofit_pass1": arm_b_1,
                "armB_demofit_pass2": arm_b_2,
                "armC_symbolic_partition_pass1": arm_c_1,
                "armC_symbolic_partition_pass2": arm_c_2,
                "oracle_hidden_pass": oracle_hit,
            }
        )

    arm_a_rate = _rate(a1)
    arm_b_rate = _rate(b1)
    arm_c_rate = _rate(c1)
    return {
        "armA_vote_passrate": arm_a_rate,
        "armA_vote_pass2": _rate(a2),
        "armAplusplus_aces_passrate": _rate(app1),
        "armAplusplus_aces_pass2": _rate(app2),
        "armB_demofit_passrate": arm_b_rate,
        "armB_demofit_pass2": _rate(b2),
        "armC_symbolic_partition_passrate": arm_c_rate,
        "armC_symbolic_partition_pass2": _rate(c2),
        "delta_pp": round((arm_b_rate - arm_a_rate) * 100.0, 4),
        "delta_armC_vs_armA_pp": round((arm_c_rate - arm_a_rate) * 100.0, 4),
        "bootstrap_ci95": _bootstrap_ci_pp(
            [b - a for a, b in zip(a1, b1, strict=False)], seed=seed
        ),
        "bootstrap_ci95_armC_vs_armA": _bootstrap_ci_pp(
            [c - a for a, c in zip(a1, c1, strict=False)], seed=seed + 4
        ),
        "oracle_passrate": _rate(oracle),
        "truncation_rate": n_truncated / max(1, n_candidates),
        "missing_verifier_gaps": missing_gaps,
        "per_task": per_task,
    }


def _rank_vote(evaluations: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    counts = Counter(_visible_signature(ev) for ev in evaluations)
    return sorted(evaluations, key=lambda ev: (-counts[_visible_signature(ev)], ev.draw_index))


def _rank_aces(evaluations: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    scores = aces_leave_one_out_scores(evaluations)
    fingerprint_counts = Counter(_fingerprint_signature(ev) for ev in evaluations)
    return sorted(
        evaluations,
        key=lambda ev: (
            -scores.get(ev.draw_index, 0.0),
            -sum(ev.visible_passes),
            -fingerprint_counts[_fingerprint_signature(ev)],
            ev.draw_index,
        ),
    )


def _rank_demofit(evaluations: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    demo_perfect = [ev for ev in evaluations if _demo_perfect(ev)]
    if not demo_perfect:
        return []
    counts = Counter(_visible_signature(ev) for ev in demo_perfect)
    aces = aces_leave_one_out_scores(demo_perfect)
    return sorted(
        demo_perfect,
        key=lambda ev: (
            -aces.get(ev.draw_index, 0.0),
            -counts[_visible_signature(ev)],
            ev.draw_index,
        ),
    )


def _rank_symbolic_partition(
    evaluations: list[CandidateEvaluation],
) -> tuple[list[CandidateEvaluation], dict[str, Any]]:
    survivors = [ev for ev in evaluations if _demo_perfect(ev)]
    if not survivors:
        return [], {
            "partition_count": 0,
            "dominant_partition_size": 0,
            "dominant_signature_sha256": None,
        }
    partitions: dict[tuple[str, ...], list[CandidateEvaluation]] = {}
    for ev in survivors:
        partitions.setdefault(_fingerprint_signature(ev), []).append(ev)
    dominant_signature, dominant = sorted(
        partitions.items(), key=lambda item: (-len(item[1]), _signature_sha(item[0]))
    )[0]
    counts = Counter(_visible_signature(ev) for ev in dominant)
    aces = aces_leave_one_out_scores(dominant)
    ranked = sorted(
        dominant,
        key=lambda ev: (
            -aces.get(ev.draw_index, 0.0),
            -counts[_visible_signature(ev)],
            ev.draw_index,
        ),
    )
    return ranked, {
        "partition_count": len(partitions),
        "dominant_partition_size": len(dominant),
        "dominant_signature_sha256": _signature_sha(dominant_signature),
    }


def _visible_signature(ev: CandidateEvaluation) -> tuple[str, ...]:
    return tuple(_stable_repr(output) for output in ev.visible_outputs)


def _fingerprint_signature(ev: CandidateEvaluation) -> tuple[str, ...]:
    values = ev.fingerprint_outputs if ev.fingerprint_outputs else ev.visible_outputs
    return tuple(_stable_repr(output) for output in values)


def _signature_sha(signature: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _demo_perfect(ev: CandidateEvaluation) -> bool:
    return bool(ev.visible_passes) and all(ev.visible_passes)


def _hidden_all(ev: CandidateEvaluation) -> bool:
    return bool(ev.hidden_passes) and all(ev.hidden_passes)


def _pass_at(ranked: list[CandidateEvaluation], k: int) -> bool:
    return any(_hidden_all(ev) for ev in ranked[:k])


def _rate(values: list[int]) -> float:
    return round(sum(values) / max(1, len(values)), 6)


def _bootstrap_ci_pp(values: list[int], *, seed: int, n_boot: int = 2000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(draw) / len(draw) * 100.0)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [round(lo, 4), round(hi, 4)]


def build_raw_artifact(
    *,
    tasks: list[CodeTask],
    evaluations_by_task: dict[str, list[CandidateEvaluation]],
    preconditions_checked: list[dict[str, Any]],
    model_specs: dict[str, Any],
    k: int,
    started_s: float,
    ended_s: float,
    mode: str,
    skipped_tasks: list[dict[str, Any]] | None = None,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    scored = score_evaluated_tasks(tasks, evaluations_by_task, seed=seed)
    artifact: dict[str, Any] = {
        "experiment": "experiment_4045_offarc_transfer_power",
        "schema": "carnot.experiment_4045_offarc_transfer_power_raw.v1",
        "honest_verdict": _verdict(scored, mode=mode),
        "corpus": "humaneval_plus_mbpp",
        "n_tasks": len(tasks),
        "target_completed_tasks": FULL_TASK_FLOOR,
        "k_candidates_per_task": k,
        "arms_implemented": list(ARMS_IMPLEMENTED),
        "armA_vote_passrate": scored["armA_vote_passrate"],
        "armA_vote_pass2": scored["armA_vote_pass2"],
        "armAplusplus_aces_passrate": scored["armAplusplus_aces_passrate"],
        "armAplusplus_aces_pass2": scored["armAplusplus_aces_pass2"],
        "armB_demofit_passrate": scored["armB_demofit_passrate"],
        "armB_demofit_pass2": scored["armB_demofit_pass2"],
        "armC_symbolic_partition_passrate": scored["armC_symbolic_partition_passrate"],
        "armC_symbolic_partition_pass2": scored["armC_symbolic_partition_pass2"],
        "delta_pp": scored["delta_pp"],
        "delta_armC_vs_armA_pp": scored["delta_armC_vs_armA_pp"],
        "bootstrap_ci95": scored["bootstrap_ci95"],
        "bootstrap_ci95_armC_vs_armA": scored["bootstrap_ci95_armC_vs_armA"],
        "oracle_passrate": scored["oracle_passrate"],
        "model_specs": model_specs,
        "random_seed": seed,
        "reproducibility_checksum": "",
        "truncation_rate": scored["truncation_rate"],
        "preconditions_checked": preconditions_checked,
        "missing_verifier_gaps": scored["missing_verifier_gaps"],
        "skipped_tasks": skipped_tasks or [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "mode": mode,
        "duration_s": round(ended_s - started_s, 2),
        "per_task": scored["per_task"],
        "candidate_pool": _candidate_pool_records(evaluations_by_task),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _verdict(scored: dict[str, Any], *, mode: str) -> str:
    if mode == "smoke":
        return "success: offarc_power_smoke_raw_complete"
    if (
        scored["armB_demofit_passrate"] > scored["armA_vote_passrate"]
        and scored["bootstrap_ci95"][0] > 0.0
    ):
        return "success: offarc_power_demofit_beats_vote_ci_excl0"
    if (
        scored["armC_symbolic_partition_passrate"] > scored["armA_vote_passrate"]
        and scored["bootstrap_ci95_armC_vs_armA"][0] > 0.0
    ):
        return "success: offarc_power_sep_beats_vote_ci_excl0"
    if scored["oracle_passrate"] <= scored["armA_vote_passrate"] + 0.01:
        return "complete: offarc_power_uninformative_no_selectable_headroom"
    return "complete: offarc_power_no_ci_closed_transfer_headroom_exists"


def _candidate_pool_records(
    evaluations_by_task: dict[str, list[CandidateEvaluation]],
) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for task_id, evaluations in sorted(evaluations_by_task.items()):
        rows = []
        for ev in evaluations:
            rows.append(
                {
                    "draw_index": ev.draw_index,
                    "status": ev.status,
                    "code_sha256": hashlib.sha256(ev.code.encode("utf-8")).hexdigest()[:16],
                    "code": ev.code,
                    "visible_passes": ev.visible_passes,
                    "hidden_passes": ev.hidden_passes,
                    "visible_outputs": ev.visible_outputs,
                    "fingerprint_signature_sha256": _signature_sha(_fingerprint_signature(ev)),
                    "generation_seconds": ev.generation_seconds,
                    "truncated": ev.truncated,
                    "error": ev.error,
                }
            )
        records[task_id] = rows
    return records


def _artifact_checksum(artifact: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s", "field_principles"}
    }
    encoded = json.dumps(payload, sort_keys=True, default=repr)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_raw_artifact(artifact: dict[str, Any], *, require_full: bool) -> None:
    if "runner_ready" in artifact:
        raise ValueError("unexpected raw artifact field: runner_ready")
    for field in REQUIRED_RAW_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required raw field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("success:")
        or verdict.startswith("complete:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in ("n_tasks", "target_completed_tasks", "k_candidates_per_task", "random_seed"):
        if not isinstance(artifact[field], int) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare int")
    for field in (
        "armA_vote_passrate",
        "armA_vote_pass2",
        "armAplusplus_aces_passrate",
        "armAplusplus_aces_pass2",
        "armB_demofit_passrate",
        "armB_demofit_pass2",
        "armC_symbolic_partition_passrate",
        "armC_symbolic_partition_pass2",
        "oracle_passrate",
        "truncation_rate",
    ):
        if not isinstance(artifact[field], float) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare float")
    for field in ("bootstrap_ci95", "bootstrap_ci95_armC_vs_armA"):
        if not (isinstance(artifact[field], list) and len(artifact[field]) == 2):
            raise ValueError(f"{field} must be a two-element list")
        if not all(isinstance(value, (int, float)) for value in artifact[field]):
            raise ValueError(f"{field} values must be numeric")
    if artifact["arms_implemented"] != ARMS_IMPLEMENTED:
        raise ValueError("arms_implemented must list the four required arms")
    for field in ("model_specs",):
        if not isinstance(artifact[field], dict):
            raise ValueError(f"{field} must be an object")
    if (
        not isinstance(artifact["reproducibility_checksum"], str)
        or not artifact["reproducibility_checksum"]
    ):
        raise ValueError("reproducibility_checksum must be a non-empty string")
    for field in ("preconditions_checked", "per_task", "skipped_tasks"):
        if not isinstance(artifact[field], list):
            raise ValueError(f"{field} must be a list")
    if not isinstance(artifact["candidate_pool"], dict):
        raise ValueError("candidate_pool must be an object")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be live_llm_inference")
    if require_full and (
        artifact["n_tasks"] < FULL_TASK_FLOOR or artifact["k_candidates_per_task"] < FULL_K_FLOOR
    ):
        raise ValueError("full run must include at least 160 tasks and k>=8")


def resolve_gemma_gguf(cache_dir: Path = GGUF_CACHE) -> Path | None:  # pragma: no cover
    candidates = sorted(cache_dir.glob("**/*.gguf"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def check_preconditions() -> list[dict[str, Any]]:  # pragma: no cover - live build path.
    model_path = resolve_gemma_gguf()
    llama_ok = _import_ok("llama_cpp")
    sandbox_ok = _sandbox_import_ok()
    humaneval_ok, humaneval_detail = _corpus_probe("humaneval")
    mbpp_ok, mbpp_detail = _corpus_probe("mbpp")
    return [
        {
            "resource": "local_gguf_cached",
            "available": model_path is not None,
            "path": str(model_path) if model_path else None,
        },
        {"resource": "llama_cpp_importable", "available": llama_ok},
        {
            "resource": "humaneval_corpus_loadable",
            "available": humaneval_ok,
            "detail": humaneval_detail,
        },
        {"resource": "mbpp_corpus_loadable", "available": mbpp_ok, "detail": mbpp_detail},
        {"resource": "restricted_exec_importable", "available": sandbox_ok},
    ]


def _corpus_probe(corpus: str) -> tuple[bool, str]:  # pragma: no cover
    try:
        if corpus == "humaneval":
            tasks, skipped = load_humaneval_tasks(limit=164)
            return len(tasks) + len(skipped) >= 164, f"loaded={len(tasks)} skipped={len(skipped)}"
        if corpus == "mbpp":
            tasks = load_mbpp_tasks(limit=3)
            return bool(tasks), f"loaded={len(tasks)}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"
    return False, "unknown corpus"


def _import_ok(module_name: str) -> bool:  # pragma: no cover
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _sandbox_import_ok() -> bool:  # pragma: no cover
    try:
        from carnot.verify import sandbox  # noqa: F401

        return True
    except Exception:
        return False


def blocker_from_preconditions(preconditions: list[dict[str, Any]]) -> str | None:
    blockers = {
        "local_gguf_cached": "blocked_local_gguf_not_cached",
        "llama_cpp_importable": "blocked_llama_cpp_unavailable",
        "humaneval_corpus_loadable": "blocked_humaneval_corpus_unavailable",
        "mbpp_corpus_loadable": "blocked_mbpp_corpus_unavailable",
        "restricted_exec_importable": "blocked_sandbox_unavailable",
    }
    for row in preconditions:
        if not bool(row.get("available")):
            return blockers.get(
                str(row.get("resource")), f"blocked_{row.get('resource', 'resource')}"
            )
    return None


def load_code_tasks(  # pragma: no cover - live corpus path.
    *,
    limit: int = DEFAULT_N_TASKS,
) -> tuple[list[CodeTask], list[dict[str, Any]]]:
    human_tasks, skipped = load_humaneval_tasks(limit=164)
    remaining = max(0, limit - len(human_tasks))
    mbpp_tasks = load_mbpp_tasks(limit=remaining)
    return (human_tasks + mbpp_tasks)[:limit], skipped


def load_humaneval_tasks(  # pragma: no cover - live corpus path.
    *,
    limit: int = 164,
    manifest_path: Path = HUMANEVAL_MANIFEST,
) -> tuple[list[CodeTask], list[dict[str, Any]]]:
    tasks: list[CodeTask] = []
    skipped: list[dict[str, Any]] = []
    for row in _read_jsonl(manifest_path):
        task_id = str(row.get("stable_id") or row.get("task_id"))
        func_name = str(row.get("entry_point"))
        tests = _parse_humaneval_tests(str(row.get("tests", "")), func_name)
        doc_examples = _parse_docstring_examples(str(row.get("prompt", "")), func_name)
        if len(tests) >= 2:
            n_visible = min(2, len(tests) - 1)
            visible = tests[:n_visible]
            hidden = tests[n_visible:]
        elif tests and doc_examples:
            visible = doc_examples[:2]
            hidden = tests
        else:
            skipped.append(
                {
                    "task_id": task_id,
                    "corpus": "humaneval",
                    "reason": "unsupported_non_exact_or_insufficient_tests",
                }
            )
            continue
        tasks.append(
            CodeTask(
                task_id=task_id,
                corpus="humaneval",
                prompt=str(row.get("prompt", "")),
                func_name=func_name,
                visible_tests=visible,
                hidden_tests=hidden,
            )
        )
        if limit and len(tasks) >= limit:
            break
    return tasks, skipped


def load_mbpp_tasks(  # pragma: no cover - live corpus path.
    *,
    limit: int,
    manifest_path: Path = MBPP_MANIFEST,
) -> list[CodeTask]:
    if limit <= 0:
        return []
    tasks: list[CodeTask] = []
    for row in _read_jsonl(manifest_path):
        parsed = [parse_assertion_test(item) for item in row.get("tests", [])]
        tests = [test for test in parsed if test is not None]
        if len(tests) < 3:
            continue
        func_name = tests[0].func_name
        tests = [test for test in tests if test.func_name == func_name]
        if len(tests) < 3:
            continue
        tasks.append(
            CodeTask(
                task_id=str(row.get("stable_id") or f"mbpp-{len(tasks)}"),
                corpus="mbpp",
                prompt=str(row.get("prompt", "")),
                func_name=func_name,
                visible_tests=tests[:2],
                hidden_tests=tests[2:],
            )
        )
        if len(tasks) >= limit:
            break
    return tasks


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _parse_humaneval_tests(test_source: str, func_name: str) -> list[CodeTest]:
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []
    parsed: list[CodeTest] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            source = f"assert {ast.unparse(node.test)}"
            test = parse_humaneval_candidate_assert(source, func_name)
            if test is not None:
                parsed.append(test)
    return parsed


def _parse_docstring_examples(prompt: str, func_name: str) -> list[CodeTest]:
    tests: list[CodeTest] = []
    parser = doctest.DocTestParser()
    for example in parser.get_examples(prompt):
        source = example.source.strip()
        want = example.want.strip()
        if not source or not want:
            continue
        try:
            call = ast.parse(source, mode="eval").body
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if call.func.id != func_name:
                continue
            args = _literal_args(call)
            if args is None:
                continue
            expected = ast.literal_eval(want)
        except (SyntaxError, ValueError):
            continue
        tests.append(CodeTest(f"assert {source} == {expected!r}", func_name, args, expected))
    return tests


def build_fingerprint_tests(task: CodeTask, *, max_tests: int = 4) -> list[CodeTest]:
    """Build deterministic public-derived probes for the SEP proxy."""
    probes: list[CodeTest] = []
    for index, test in enumerate(task.visible_tests):
        mutated = tuple(_mutate_public_arg(arg) for arg in test.args)
        if mutated == test.args:
            continue
        probes.append(
            CodeTest(
                source=f"probe {task.func_name}{_args_repr(mutated)}",
                func_name=task.func_name,
                args=mutated,
                expected=None,
            )
        )
        if len(probes) >= max_tests:
            break
    return probes


def _mutate_public_arg(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, str):
        return value + ("a" if "a" not in value[-1:] else "b")
    if isinstance(value, list):
        if not value:
            return value
        return value[1:] + value[:1]
    if isinstance(value, tuple):
        if not value:
            return value
        return value[1:] + value[:1]
    if isinstance(value, dict):
        return dict(reversed(list(value.items())))
    return value


def build_prompt(task: CodeTask) -> str:  # pragma: no cover - live generation path.
    visible = "\n".join(test.source for test in task.visible_tests)
    return (
        "Write a correct Python function for this programming task.\n\n"
        f"Task:\n{task.prompt}\n\n"
        f"Function name: {task.func_name}\n\n"
        "Visible example tests:\n"
        f"{visible}\n\n"
        "Return only one Python code block defining the requested function. "
        "Do not include hidden tests, file IO, network access, or explanations."
    )


class LocalGemmaSampler:  # pragma: no cover - live model path.
    SYSTEM = (
        "You are an expert Python programmer. Produce only a Python code block "
        "containing the requested function."
    )

    def __init__(  # pragma: no cover - live model path.
        self, llama: Any, *, max_tokens: int = 1024, base_seed: int = RANDOM_SEED
    ) -> None:
        self._llama = llama
        self.max_tokens = max_tokens
        self.base_seed = base_seed

    def __call__(self, task: CodeTask, draw_index: int) -> GeneratedCandidate:  # pragma: no cover
        prompt = build_prompt(task)
        temperature = round(min(0.95, 0.25 + 0.07 * (draw_index % 8)), 3)
        seed = self.base_seed + draw_index * 1009
        started = time.time()
        try:
            out = self._llama.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=temperature,
                top_p=0.95,
                seed=seed,
            )
            choice = out["choices"][0]
            raw = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason")
        except Exception as exc:
            raw = f"__local_error__:{type(exc).__name__}: {exc}"
            finish_reason = "error"
        code = extract_code_block(raw, task.func_name)
        return GeneratedCandidate(
            draw_index=draw_index,
            raw_text=raw,
            code=code,
            generation_seconds=round(time.time() - started, 3),
            finish_reason=finish_reason,
            truncated=finish_reason == "length",
        )


def make_live_sampler() -> LocalGemmaSampler:  # pragma: no cover
    from llama_cpp import Llama

    model_path = resolve_gemma_gguf()
    if model_path is None:
        raise RuntimeError("local Gemma 4 12B GGUF not found")
    llama = Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_gpu_layers=-1,
        n_batch=512,
        verbose=False,
    )
    return LocalGemmaSampler(llama)


def run(  # pragma: no cover - live smoke/full command path.
    *,
    output_path: Path = OUTPUT,
    checkpoint_path: Path = CHECKPOINT,
    n_tasks: int = DEFAULT_N_TASKS,
    k: int = DEFAULT_K,
    mode: str = "full",
    preconditions_checked: list[dict[str, Any]] | None = None,
    sampler: Callable[[CodeTask, int], GeneratedCandidate] | None = None,
    executor: Callable[[str, str, tuple[Any, ...], float], tuple[Any, Exception | None]]
    | None = None,
) -> dict[str, Any]:
    started = time.time()
    tasks, skipped = load_code_tasks(limit=n_tasks)
    if mode == "smoke":
        tasks = tasks[:2]
    if not tasks:
        raise RuntimeError("no code tasks loaded")

    sampler = sampler or make_live_sampler()
    executor = executor or default_executor
    evaluations_by_task = _load_checkpoint(checkpoint_path)
    for task in tasks:
        if len(evaluations_by_task.get(task.task_id, [])) >= k:
            continue
        fingerprint_tests = build_fingerprint_tests(task)
        evaluations: list[CandidateEvaluation] = []
        for draw_index in range(k):
            candidate = sampler(task, draw_index)
            evaluations.append(
                evaluate_candidate(
                    task,
                    candidate,
                    executor=executor,
                    fingerprint_tests=fingerprint_tests,
                )
            )
        evaluations_by_task[task.task_id] = evaluations
        _write_checkpoint(
            checkpoint_path,
            tasks=tasks,
            evaluations_by_task=evaluations_by_task,
            skipped_tasks=skipped,
            k=k,
            mode=mode,
        )

    model_specs = {
        "local_generator": "unsloth/gemma-4-12B-it-GGUF",
        "verifier": (
            "model-free demo-fit + restricted-namespace execution + "
            "content-hash/fingerprint SEP proxy"
        ),
        "candidate_pool_policy": "same generated pool shared by all four arms",
        "llama_cpp": {"n_ctx": 4096, "n_batch": 512, "n_gpu_layers": -1},
    }
    artifact = build_raw_artifact(
        tasks=tasks,
        evaluations_by_task=evaluations_by_task,
        preconditions_checked=preconditions_checked or check_preconditions(),
        model_specs=model_specs,
        k=k,
        started_s=started,
        ended_s=time.time(),
        mode=mode,
        skipped_tasks=skipped,
    )
    validate_raw_artifact(artifact, require_full=mode == "full")
    _write_json(output_path, artifact)
    return artifact


def _load_checkpoint(path: Path) -> dict[str, list[CandidateEvaluation]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    evaluations_by_task: dict[str, list[CandidateEvaluation]] = {}
    for task_id, rows in payload.get("evaluations_by_task", {}).items():
        evaluations_by_task[task_id] = [CandidateEvaluation(**row) for row in rows]
    return evaluations_by_task


def _write_checkpoint(
    path: Path,
    *,
    tasks: list[CodeTask],
    evaluations_by_task: dict[str, list[CandidateEvaluation]],
    skipped_tasks: list[dict[str, Any]],
    k: int,
    mode: str,
) -> None:
    payload = {
        "experiment": "experiment_4045_offarc_transfer_power_checkpoint",
        "schema": "carnot.experiment_4045_offarc_transfer_power_checkpoint.v1",
        "mode": mode,
        "k_candidates_per_task": k,
        "completed_task_ids": sorted(evaluations_by_task),
        "ordered_task_ids": [task.task_id for task in tasks],
        "skipped_tasks": skipped_tasks,
        "evaluations_by_task": {
            task_id: [asdict(ev) for ev in evaluations]
            for task_id, evaluations in sorted(evaluations_by_task.items())
        },
        "updated_at_unix": time.time(),
    }
    _write_json(path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Exp 4045 OFF-ARC full-power runner")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--n-tasks", type=int, default=DEFAULT_N_TASKS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    artifact = run(
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        n_tasks=args.n_tasks,
        k=args.k,
        mode=args.mode,
    )
    print(
        f"-> {artifact['honest_verdict']} n={artifact['n_tasks']} k={artifact['k_candidates_per_task']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()

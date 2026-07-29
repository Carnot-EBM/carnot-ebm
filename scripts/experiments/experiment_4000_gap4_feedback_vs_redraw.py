"""Exp 4000 GAP-4 feedback-chain vs independent-redraw paired control.

This is the same-run paired mechanism control for the GAP-4 chain harness:
per chain-feasible ARC-2 task, arm A runs one <=3-iteration failure-feedback
chain and arm B runs three independent one-call redraws. Gold is used only
after induction, for task-level scoring and the paired McNemar test.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from arc3_gap3_stage2_transition_ebm import SEED, ghash
from arc3_gap4_rule_exec_verifier import (
    _extract_code,
    _failing_demos,
    ask_codex,
    demo_fit,
    induction_prompt,
    safe_transform_from_code,
)
from experiment_3998_gap4_deselection_coverage import audit_transcripts
from carnot.paths import repo_root


# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
REPO_ROOT = repo_root()
ARC2_POOL = REPO_ROOT / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
CHAIN_ARTIFACT = REPO_ROOT / "results" / "arc3_gap4_arc2_chain_ensemble.json"
OUTPUT = REPO_ROOT / "results" / "experiment_4000_gap4_feedback_vs_redraw.json"
TRANSCRIPTS_DIR = REPO_ROOT / "results" / "experiment_4000_gap4_feedback_vs_redraw_transcripts"
CHALLENGES = Path("/home/ianblenke/trm_src/kaggle/combined/arc-agi_evaluation2_challenges.json")
SOLUTIONS = Path("/home/ianblenke/trm_src/kaggle/combined/arc-agi_evaluation2_solutions.json")
INFERENCE_SUBSTRATE = "codex_program_induction_same_run_feedback_vs_iid_redraw_arc2"

REQUIRED_FIELDS = [
    "same_run_interleaved",
    "feedback_beats_redraw",
    "mcnemar_p",
    "n_discordant_pairs",
    "arm_a_gold_rate",
    "arm_b_gold_rate",
    "total_codex_calls",
    "total_codex_seconds",
    "leak_clean",
    "random_seed",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "same_run_interleaved": (
        "BARE BOOL - A and B were run interleaved in one experiment invocation."
    ),
    "feedback_beats_redraw": (
        "BARE BOOL - feedback chain beats three iid singles at equal budget."
    ),
    "mcnemar_p": "Exact two-sided paired McNemar p-value.",
    "n_discordant_pairs": "A-only plus B-only task count.",
    "arm_a_gold_rate": "Task-level gold rate for the feedback-chain arm.",
    "arm_b_gold_rate": "Task-level gold rate for any of three independent redraws.",
    "total_codex_calls": "Total Codex induction calls across both arms.",
    "total_codex_seconds": "Total Codex wall seconds reported by induction calls.",
    "leak_clean": "BARE BOOL - archived transcripts passed the word-boundary leak audit.",
    "random_seed": "Shared GAP-4 substrate seed.",
    "honest_verdict": "Terminal-prefix mechanism verdict.",
    "duration_s": "Wall-clock seconds for this runner.",
    "inference_substrate": "Codex program-induction substrate and paired-control mode.",
}


def load_eval_pool(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_preconditions(
    pool_path: Path,
    codex_available_override: bool | None = None,
) -> list[dict[str, Any]]:
    codex_available = (
        bool(codex_available_override)
        if codex_available_override is not None
        else shutil.which("codex") is not None
    )
    try:
        load_eval_pool(pool_path)
        pool_available = True
    except Exception:
        pool_available = False
    return [
        {"resource": "codex", "available": codex_available},
        {"resource": "eval_pool", "available": pool_available},
    ]


def blocker_from_preconditions(preconditions: list[dict[str, Any]]) -> str | None:
    by_resource = {row["resource"]: bool(row["available"]) for row in preconditions}
    if not by_resource.get("codex", False):
        return "blocked_codex_unavailable"
    if not by_resource.get("eval_pool", False):
        return "blocked_eval_pool_unreadable"
    return None


def selected_tasks_from_chain_artifact(artifact: dict[str, Any]) -> list[str]:
    prereg = artifact.get("preregistration", {})
    tasks = prereg.get("tasks")
    if tasks:
        return sorted(str(task) for task in tasks)
    return sorted(str(row["task"]) for row in artifact.get("per_task", []))


def _group_entries_by_task(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry["task"]), []).append(entry)
    return grouped


def _rate(successes: int, total: int) -> float:
    return round(successes / total, 4) if total else 0.0


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _grid_hash(grid: Any) -> str:
    return ghash(np.asarray(grid, dtype=np.int64))


def _score_grid(pred_grid: Any, target: np.ndarray | None) -> bool:
    return (
        target is not None
        and pred_grid is not None
        and np.array_equal(np.asarray(pred_grid, dtype=np.int64), target)
    )


def _gold_from_entry_candidates(entry: dict[str, Any]) -> np.ndarray | None:
    for candidate in entry.get("candidates", []):
        if candidate.get("correct") is True and "grid" in candidate:
            return np.asarray(candidate["grid"], dtype=np.int64)
    return None


def gold_for_entry(
    entry: dict[str, Any],
    challenges: dict[str, Any],
    solutions: dict[str, Any],
) -> np.ndarray | None:
    task = str(entry["task"])
    challenge = challenges.get(task)
    if challenge and task in solutions:
        target_hash = _grid_hash(entry["test_input"])
        for idx, pair in enumerate(challenge.get("test", [])):
            if _grid_hash(pair["input"]) == target_hash:
                return np.asarray(solutions[task][idx], dtype=np.int64)
    return _gold_from_entry_candidates(entry)


def _call_and_grade(
    prompt: str,
    transcript_path: Path,
    timeout: int,
    iter_idx: int,
    demos: list[dict[str, Any]],
) -> tuple[str | None, Any | None, float, dict[str, Any]]:
    raw, elapsed = ask_codex(prompt, timeout=timeout, transcript_path=str(transcript_path))
    code = _extract_code(raw)
    if code is None:
        return None, None, 0.0, {"iter": iter_idx, "status": "no_code", "codex_s": elapsed}
    fn = safe_transform_from_code(code)
    if fn is None:
        return (
            None,
            None,
            0.0,
            {"iter": iter_idx, "status": "unsafe_or_uncompilable", "codex_s": elapsed},
        )
    fit = demo_fit(fn, demos)
    return (
        code,
        fn,
        fit,
        {
            "iter": iter_idx,
            "status": "graded",
            "demo_fit": round(fit, 4),
            "codex_s": elapsed,
            "code_len": len(code),
        },
    )


def _predictions_for_arm(
    code: str | None,
    demo_perfect: bool,
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fn = safe_transform_from_code(code) if code and demo_perfect else None
    predictions = []
    for idx, entry in enumerate(entries):
        pred = fn(entry["test_input"]) if fn else None
        predictions.append(
            {
                "input_idx": idx,
                "pred_hash": _grid_hash(pred) if pred is not None else None,
                "pred_grid": pred.tolist() if pred is not None else None,
            }
        )
    return predictions


def _feedback_record(
    task: str,
    entries: list[dict[str, Any]],
    transcripts_dir: Path,
    iters: int,
    timeout: int,
    redraw_callback: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    demos = entries[0]["demos"]
    test_input = entries[0]["test_input"]
    task_dir = transcripts_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    best_fit = -1.0
    best_code: str | None = None
    best_fn = None
    prior_code: str | None = None
    failures: str | None = None
    history: list[dict[str, Any]] = []
    redraws: list[dict[str, Any]] = []
    feedback_done = False
    feedback_iter = 0
    redraw_idx = 1

    while (feedback_iter < iters and not feedback_done) or redraw_idx <= 3:
        if feedback_iter < iters and not feedback_done:
            prompt = induction_prompt(demos, test_input, prior_code, failures)
            code, fn, fit, row = _call_and_grade(
                prompt,
                task_dir / f"arm_a_feedback_iter{feedback_iter}.txt",
                timeout,
                feedback_iter,
                demos,
            )
            history.append(row)
            if fn is not None and fit > best_fit:
                best_fit = fit
                best_code = code
                best_fn = fn
            if best_fit >= 1.0:
                feedback_done = True
            elif best_fn is not None:
                prior_code = best_code
                failures = _failing_demos(best_fn, demos)
            feedback_iter += 1
        if redraw_idx <= 3:
            redraws.append(redraw_callback(redraw_idx))
            redraw_idx += 1

    demo_fit_value = round(max(best_fit, 0.0), 4)
    feedback = {
        "source": "arm_a_feedback",
        "code": best_code,
        "demo_fit": demo_fit_value,
        "demo_perfect": bool(best_fit >= 1.0),
        "n_calls": len(history),
        "codex_seconds": round(sum(float(row["codex_s"]) for row in history), 2),
        "history": history,
        "predictions": _predictions_for_arm(best_code, bool(best_fit >= 1.0), entries),
    }
    return feedback, redraws


def _redraw_record(
    task: str,
    entries: list[dict[str, Any]],
    transcripts_dir: Path,
    redraw_idx: int,
    timeout: int,
) -> dict[str, Any]:
    demos = entries[0]["demos"]
    test_input = entries[0]["test_input"]
    task_dir = transcripts_dir / task
    prompt = induction_prompt(demos, test_input)
    code, _fn, fit, row = _call_and_grade(
        prompt,
        task_dir / f"arm_b_redraw{redraw_idx}_iter0.txt",
        timeout,
        0,
        demos,
    )
    demo_perfect = bool(fit >= 1.0)
    return {
        "source": f"arm_b_redraw{redraw_idx}",
        "code": code,
        "demo_fit": round(fit, 4),
        "demo_perfect": demo_perfect,
        "n_calls": 1,
        "codex_seconds": round(float(row["codex_s"]), 2),
        "history": [row],
        "predictions": _predictions_for_arm(code, demo_perfect, entries),
    }


def _arm_task_correct(
    arm: dict[str, Any],
    entries: list[dict[str, Any]],
    challenges: dict[str, Any],
    solutions: dict[str, Any],
) -> bool:
    if not entries or not arm.get("demo_perfect"):
        return False
    for entry, pred in zip(entries, arm["predictions"], strict=True):
        target = gold_for_entry(entry, challenges, solutions)
        if not _score_grid(pred.get("pred_grid"), target):
            return False
    return True


def _paired_task(
    task: str,
    entries: list[dict[str, Any]],
    transcripts_dir: Path,
    challenges: dict[str, Any],
    solutions: dict[str, Any],
    iters: int,
    timeout: int,
) -> dict[str, Any]:
    def run_redraw(redraw_idx: int) -> dict[str, Any]:
        return _redraw_record(task, entries, transcripts_dir, redraw_idx, timeout)

    feedback, redraws = _feedback_record(
        task=task,
        entries=entries,
        transcripts_dir=transcripts_dir,
        iters=iters,
        timeout=timeout,
        redraw_callback=run_redraw,
    )
    arm_a_correct = _arm_task_correct(feedback, entries, challenges, solutions)
    redraw_correct = [
        _arm_task_correct(redraw, entries, challenges, solutions) for redraw in redraws
    ]
    arm_b_correct = any(redraw_correct)
    return {
        "task": task,
        "n_entries": len(entries),
        "arm_a_feedback": feedback,
        "arm_b_redraws": redraws,
        "arm_a_correct": arm_a_correct,
        "arm_b_correct": arm_b_correct,
        "arm_b_correct_sources": [
            redraw["source"] for redraw, ok in zip(redraws, redraw_correct, strict=True) if ok
        ],
        "n_calls": int(feedback["n_calls"]) + sum(int(redraw["n_calls"]) for redraw in redraws),
        "codex_seconds": round(
            float(feedback["codex_seconds"])
            + sum(float(redraw["codex_seconds"]) for redraw in redraws),
            2,
        ),
    }


def transcript_paths(transcripts_dir: Path) -> list[Path]:
    return sorted(path for path in transcripts_dir.glob("*/*.txt") if path.is_file())


def _format_p(value: float) -> str:
    if value == round(value):
        return f"{value:.1f}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _verdict(feedback_beats_redraw: bool, p_value: float, discordant: int) -> str:
    prefix = (
        "success: feedback_beats_redraw_p"
        if feedback_beats_redraw
        else "complete: feedback_no_better_than_redraw_p"
    )
    suffix = "_FALSE_NEGATIVE_RISK" if discordant < 10 and not feedback_beats_redraw else ""
    return prefix + _format_p(p_value) + suffix


def blocked_artifact(
    verdict: str,
    preconditions: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "experiment": "experiment_4000_gap4_feedback_vs_redraw",
        "schema": "carnot.experiment_4000_gap4_feedback_vs_redraw.v1",
        "title": "GAP-4 feedback-chain vs independent-redraw paired control",
        "same_run_interleaved": False,
        "feedback_beats_redraw": False,
        "mcnemar_p": 1.0,
        "n_discordant_pairs": 0,
        "arm_a_gold_rate": 0.0,
        "arm_b_gold_rate": 0.0,
        "paired_contingency": {
            "a_correct_b_correct": 0,
            "a_correct_b_wrong": 0,
            "a_wrong_b_correct": 0,
            "a_wrong_b_wrong": 0,
        },
        "per_task": [],
        "total_codex_calls": 0,
        "total_codex_seconds": 0.0,
        "leak_clean": False,
        "leak_audit": {"clean": False, "n_transcripts": 0, "violations": []},
        "preconditions_checked": preconditions,
        "random_seed": SEED,
        "honest_verdict": verdict,
        "duration_s": round(duration_s, 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def build_complete_artifact(
    records: list[dict[str, Any]],
    tasks: list[str],
    preconditions: list[dict[str, Any]],
    transcript_audit: dict[str, Any],
    started_s: float,
    now_s: float,
) -> dict[str, Any]:
    a_correct = sum(int(row["arm_a_correct"]) for row in records)
    b_correct = sum(int(row["arm_b_correct"]) for row in records)
    both = sum(int(row["arm_a_correct"] and row["arm_b_correct"]) for row in records)
    a_only = sum(int(row["arm_a_correct"] and not row["arm_b_correct"]) for row in records)
    b_only = sum(int((not row["arm_a_correct"]) and row["arm_b_correct"]) for row in records)
    neither = sum(int((not row["arm_a_correct"]) and (not row["arm_b_correct"])) for row in records)
    discordant = a_only + b_only
    p_value = round(exact_mcnemar_p(a_only, b_only), 6)
    feedback_beats = bool(a_only > b_only and p_value <= 0.05)
    artifact: dict[str, Any] = {
        "experiment": "experiment_4000_gap4_feedback_vs_redraw",
        "schema": "carnot.experiment_4000_gap4_feedback_vs_redraw.v1",
        "title": "GAP-4 feedback-chain vs independent-redraw paired control",
        "same_run_interleaved": True,
        "feedback_beats_redraw": feedback_beats,
        "mcnemar_p": p_value,
        "n_discordant_pairs": discordant,
        "arm_a_gold_rate": _rate(a_correct, len(records)),
        "arm_b_gold_rate": _rate(b_correct, len(records)),
        "paired_contingency": {
            "a_correct_b_correct": both,
            "a_correct_b_wrong": a_only,
            "a_wrong_b_correct": b_only,
            "a_wrong_b_wrong": neither,
        },
        "n_tasks": len(records),
        "task_set": tasks,
        "per_task": records,
        "total_codex_calls": sum(int(record["n_calls"]) for record in records),
        "total_codex_seconds": round(sum(float(record["codex_seconds"]) for record in records), 2),
        "leak_clean": bool(transcript_audit["clean"]),
        "leak_audit": transcript_audit,
        "preconditions_checked": preconditions,
        "random_seed": SEED,
        "honest_verdict": _verdict(feedback_beats, p_value, discordant),
        "duration_s": round(now_s - started_s, 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def _is_bare_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_bare_float(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def validate_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:")
        or verdict.startswith("success:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in ("same_run_interleaved", "feedback_beats_redraw", "leak_clean"):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in ("n_discordant_pairs", "total_codex_calls", "random_seed"):
        if not _is_bare_int(artifact[field]):
            raise ValueError(f"{field} must be a bare int")
    for field in (
        "mcnemar_p",
        "arm_a_gold_rate",
        "arm_b_gold_rate",
        "total_codex_seconds",
        "duration_s",
    ):
        if not _is_bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float")
    if not isinstance(artifact["inference_substrate"], str):
        raise ValueError("inference_substrate must be a string")


def run(
    root: Path = REPO_ROOT,
    pool_path: Path = ARC2_POOL,
    chain_artifact_path: Path = CHAIN_ARTIFACT,
    output_path: Path = OUTPUT,
    transcripts_dir: Path = TRANSCRIPTS_DIR,
    challenges_path: Path = CHALLENGES,
    solutions_path: Path = SOLUTIONS,
    workers: int = 4,
    iters: int = 3,
    timeout: int = 600,
    codex_available_override: bool | None = None,
    write: bool = True,
) -> dict[str, Any]:
    del root
    started = time.time()
    preconditions = check_preconditions(pool_path, codex_available_override)
    blocker = blocker_from_preconditions(preconditions)
    if blocker:
        artifact = blocked_artifact(blocker, preconditions, time.time() - started)
        if write:
            _write_json(output_path, artifact)
        return artifact

    pool = load_eval_pool(pool_path)
    chain_artifact = load_json(chain_artifact_path)
    entries_by_task = _group_entries_by_task(pool["entries"])
    tasks = selected_tasks_from_chain_artifact(chain_artifact)
    missing = [task for task in tasks if task not in entries_by_task]
    if missing:
        raise ValueError(f"chain-feasible tasks missing from eval pool: {missing}")

    challenges = load_json(challenges_path) if challenges_path.exists() else {}
    solutions = load_json(solutions_path) if solutions_path.exists() else {}
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[exp4000] paired feedback-vs-redraw on {len(tasks)} chain-feasible tasks "
        f"(feedback iters<={iters}, redraws=3, timeout={timeout}s, workers={workers})",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(
            lambda task: _paired_task(
                task,
                entries_by_task[task],
                transcripts_dir,
                challenges,
                solutions,
                iters,
                timeout,
            ),
            tasks,
        )
        for record in futures:
            records.append(record)
            print(
                f"  {record['task']}: A={int(record['arm_a_correct'])} "
                f"B={int(record['arm_b_correct'])} calls={record['n_calls']} "
                f"codex_s={record['codex_seconds']}",
                flush=True,
            )

    audit = audit_transcripts(transcript_paths(transcripts_dir))
    artifact = build_complete_artifact(
        records=records,
        tasks=tasks,
        preconditions=preconditions,
        transcript_audit=audit,
        started_s=started,
        now_s=time.time(),
    )
    if write:
        _write_json(output_path, artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(
        f"   McNemar p={artifact['mcnemar_p']} discordant={artifact['n_discordant_pairs']} "
        f"A_rate={artifact['arm_a_gold_rate']} B_rate={artifact['arm_b_gold_rate']} "
        f"leak_clean={artifact['leak_clean']}",
        flush=True,
    )
    return artifact


def main() -> None:  # pragma: no cover - exercised by the required script command.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    run(workers=args.workers, iters=args.iters, timeout=args.timeout)


if __name__ == "__main__":  # pragma: no cover
    main()

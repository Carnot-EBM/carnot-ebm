"""Exp 4011 GAP-4 feedback-chain vs independent-redraw powered paired control.

This is the powered v2 rerun of Exp 4000. Exp 4000 found only 3 discordant
pairs, so the null result was explicitly underpowered. This runner keeps the
same same-run interleaved A/B mechanism, but streams over the ARC-2 GAP-4 eval
pool until the paired discordant count reaches 10 or the pool is exhausted.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import experiment_4000_gap4_feedback_vs_redraw as v1
from arc3_gap3_stage2_transition_ebm import SEED
from carnot.paths import repo_root


# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
REPO_ROOT = repo_root()
ARC2_POOL = REPO_ROOT / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
PILOT_ARTIFACT = REPO_ROOT / "results" / "experiment_4000_gap4_feedback_vs_redraw.json"
OUTPUT = REPO_ROOT / "results" / "experiment_4011_gap4_feedback_vs_redraw_v2.json"
TRANSCRIPTS_DIR = REPO_ROOT / "results" / "experiment_4011_gap4_feedback_vs_redraw_v2_transcripts"
CHALLENGES = Path("/home/ianblenke/trm_src/kaggle/combined/arc-agi_evaluation2_challenges.json")
SOLUTIONS = Path("/home/ianblenke/trm_src/kaggle/combined/arc-agi_evaluation2_solutions.json")
INFERENCE_SUBSTRATE = "codex_program_induction_same_run_feedback_vs_iid_redraw_arc2_powered_v2"
DISCORDANT_TARGET = 10
POWER_TARGET = 0.8
ALPHA = 0.05

REQUIRED_FIELDS = [
    "same_run_interleaved",
    "feedback_beats_redraw",
    "n_discordant_pairs",
    "mcnemar_p",
    "achieved_power",
    "min_detectable_effect",
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
        "BARE BOOL -- A and B were run interleaved in ONE codex run "
        "(the control vs the 2.2x between-run variance; false => invalid comparison)."
    ),
    "feedback_beats_redraw": (
        "BARE BOOL -- the feedback chain beats 3 iid singles at equal budget "
        "(the mechanism verdict; false => deploy cheaper independent redraws)."
    ),
    "n_discordant_pairs": (
        "BARE INT -- discordant pairs achieved (>=10 => powered; the fix for the "
        ".370 underpowered run)."
    ),
    "mcnemar_p": "Exact paired-test p.",
    "achieved_power": (
        "Power achieved against the preregistered 10-discordant-pair detectable effect."
    ),
    "min_detectable_effect": ("Exact minimum discordant-pair win-probability shift for 80% power."),
    "arm_a_gold_rate": "Per-arm task-level gold rate for the feedback chain at equal budget.",
    "arm_b_gold_rate": "Per-arm task-level gold rate for three iid redraws at equal budget.",
    "total_codex_calls": "Total Codex induction calls across both arms.",
    "total_codex_seconds": "Total Codex wall seconds recorded by induction calls.",
    "leak_clean": "BARE BOOL -- transcript leak-audit provenance.",
    "random_seed": "Reproducibility seed.",
    "honest_verdict": "Terminal-prefix verdict.",
    "duration_s": "Wall-clock duration in seconds.",
    "inference_substrate": "Codex program-induction substrate.",
}

load_eval_pool = v1.load_eval_pool
load_json = v1.load_json
check_preconditions = v1.check_preconditions
blocker_from_preconditions = v1.blocker_from_preconditions
exact_mcnemar_p = v1.exact_mcnemar_p
_group_entries_by_task = v1._group_entries_by_task
_paired_task = v1._paired_task
audit_transcripts = v1.audit_transcripts
transcript_paths = v1.transcript_paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tasks_from_eval_pool(pool: dict[str, Any]) -> list[str]:
    """Return unique eval-pool tasks in committed pool order."""

    tasks: list[str] = []
    seen: set[str] = set()
    for entry in pool.get("entries", []):
        task = str(entry["task"])
        if task not in seen:
            seen.add(task)
            tasks.append(task)
    return tasks


def _binomial_pmf(n: int, k: int, p: float) -> float:
    return math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def _critical_indices(n: int, alpha: float = ALPHA) -> list[int]:
    return [k for k in range(n + 1) if exact_mcnemar_p(k, n - k) <= alpha]


def _power_at_probability(n: int, p_alt: float, alpha: float = ALPHA) -> float:
    if n <= 0:
        return 0.0
    return sum(_binomial_pmf(n, k, p_alt) for k in _critical_indices(n, alpha))


def _minimum_effect_raw(
    n: int,
    target_power: float = POWER_TARGET,
    alpha: float = ALPHA,
) -> float:
    if n <= 0 or _power_at_probability(n, 1.0, alpha) < target_power:
        return 1.0
    lo = 0.5
    hi = 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _power_at_probability(n, mid, alpha) >= target_power:
            hi = mid
        else:
            lo = mid
    return hi - 0.5


def target_effect_for_discordant_target() -> float:
    """Effect that makes the preregistered 10-discordant target 80% powered."""

    return _minimum_effect_raw(DISCORDANT_TARGET)


def achieved_power(n_discordant: int) -> float:
    """Exact power at the preregistered target effect for the observed discordants."""

    if n_discordant <= 0:
        return 0.0
    p_alt = 0.5 + target_effect_for_discordant_target()
    return float(round(_power_at_probability(n_discordant, p_alt), 4))


def min_detectable_effect(n_discordant: int) -> float:
    """Minimum absolute discordant-pair effect needed for 80% exact-test power."""

    if n_discordant <= 0:
        return 1.0
    return round(_minimum_effect_raw(n_discordant), 4)


def _rate(successes: int, total: int) -> float:
    return round(successes / total, 4) if total else 0.0


def _format_p(value: float) -> str:
    return v1._format_p(value)


def load_pilot_context(path: Path = PILOT_ARTIFACT) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    try:
        pilot = load_json(path)
    except Exception as exc:
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": True,
        "honest_verdict": pilot.get("honest_verdict"),
        "n_discordant_pairs": pilot.get("n_discordant_pairs"),
        "mcnemar_p": pilot.get("mcnemar_p"),
        "paired_contingency": pilot.get("paired_contingency"),
        "false_negative_risk": "FALSE_NEGATIVE_RISK" in str(pilot.get("honest_verdict", "")),
    }


def _verdict(
    feedback_beats_redraw: bool,
    p_value: float,
    discordant: int,
    pool_exhausted: bool,
) -> str:
    if feedback_beats_redraw:
        return "success: feedback_beats_redraw_p" + _format_p(p_value)
    if discordant >= DISCORDANT_TARGET:
        return "complete: feedback_no_better_than_redraw_powered_p" + _format_p(p_value)
    if pool_exhausted:
        return f"complete: feedback_vs_redraw_underpowered_n{discordant}"
    return f"blocked_discordant_target_unmet_n{discordant}"


def _base_artifact(
    honest_verdict: str,
    duration_s: float,
    pilot_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4011_gap4_feedback_vs_redraw_v2",
        "schema": "carnot.experiment_4011_gap4_feedback_vs_redraw_v2.v1",
        "title": "GAP-4 feedback-chain vs independent-redraw powered paired control v2",
        "same_run_interleaved": False,
        "feedback_beats_redraw": False,
        "n_discordant_pairs": 0,
        "mcnemar_p": 1.0,
        "achieved_power": 0.0,
        "min_detectable_effect": 1.0,
        "arm_a_gold_rate": 0.0,
        "arm_b_gold_rate": 0.0,
        "paired_contingency": {
            "a_correct_b_correct": 0,
            "a_correct_b_wrong": 0,
            "a_wrong_b_correct": 0,
            "a_wrong_b_wrong": 0,
        },
        "n_tasks": 0,
        "task_set": [],
        "pool_exhausted": False,
        "stop_reason": "not_started",
        "per_task": [],
        "total_codex_calls": 0,
        "total_codex_seconds": 0.0,
        "leak_clean": False,
        "leak_audit": {"clean": False, "n_transcripts": 0, "violations": []},
        "pilot_exp4000": pilot_context or {"available": False},
        "random_seed": SEED,
        "honest_verdict": honest_verdict,
        "duration_s": round(float(duration_s), 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
    }


def blocked_artifact(
    verdict: str,
    preconditions: list[dict[str, Any]],
    duration_s: float,
    pilot_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = _base_artifact(verdict, duration_s, pilot_context)
    artifact["preconditions_checked"] = preconditions
    validate_artifact(artifact)
    return artifact


def build_complete_artifact(
    records: list[dict[str, Any]],
    tasks: list[str],
    all_tasks: list[str],
    preconditions: list[dict[str, Any]],
    transcript_audit: dict[str, Any],
    pilot_context: dict[str, Any],
    started_s: float,
    now_s: float,
    pool_exhausted: bool,
) -> dict[str, Any]:
    a_correct = sum(int(row["arm_a_correct"]) for row in records)
    b_correct = sum(int(row["arm_b_correct"]) for row in records)
    both = sum(int(row["arm_a_correct"] and row["arm_b_correct"]) for row in records)
    a_only = sum(int(row["arm_a_correct"] and not row["arm_b_correct"]) for row in records)
    b_only = sum(int((not row["arm_a_correct"]) and row["arm_b_correct"]) for row in records)
    neither = sum(int((not row["arm_a_correct"]) and (not row["arm_b_correct"])) for row in records)
    discordant = a_only + b_only
    p_value = round(exact_mcnemar_p(a_only, b_only), 6)
    feedback_beats = bool(a_only > b_only and p_value <= ALPHA)
    stop_reason = (
        "discordant_target_met"
        if discordant >= DISCORDANT_TARGET
        else "pool_exhausted"
        if pool_exhausted
        else "stopped_before_target"
    )
    artifact = _base_artifact(
        _verdict(feedback_beats, p_value, discordant, pool_exhausted),
        now_s - started_s,
        pilot_context,
    )
    artifact.update(
        {
            "same_run_interleaved": True,
            "feedback_beats_redraw": feedback_beats,
            "n_discordant_pairs": discordant,
            "mcnemar_p": p_value,
            "achieved_power": achieved_power(discordant),
            "min_detectable_effect": min_detectable_effect(discordant),
            "arm_a_gold_rate": _rate(a_correct, len(records)),
            "arm_b_gold_rate": _rate(b_correct, len(records)),
            "paired_contingency": {
                "a_correct_b_correct": both,
                "a_correct_b_wrong": a_only,
                "a_wrong_b_correct": b_only,
                "a_wrong_b_wrong": neither,
            },
            "discordant_target": DISCORDANT_TARGET,
            "n_tasks": len(records),
            "task_set": tasks,
            "eligible_task_set": all_tasks,
            "remaining_task_count": max(0, len(all_tasks) - len(records)),
            "pool_exhausted": pool_exhausted,
            "stop_reason": stop_reason,
            "per_task": records,
            "total_codex_calls": sum(int(record["n_calls"]) for record in records),
            "total_codex_seconds": round(
                sum(float(record["codex_seconds"]) for record in records), 2
            ),
            "leak_clean": bool(transcript_audit["clean"]),
            "leak_audit": transcript_audit,
            "preconditions_checked": preconditions,
        }
    )
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
    if "FALSE_NEGATIVE_RISK" in verdict:
        raise ValueError("4011 verdicts must not use the Exp 4000 FALSE_NEGATIVE_RISK suffix")
    if verdict.startswith("complete: feedback_no_better_than_redraw_p") and not verdict.startswith(
        "complete: feedback_no_better_than_redraw_powered_p"
    ):
        raise ValueError("powered null verdict must include feedback_no_better_than_redraw_powered")
    for field in ("same_run_interleaved", "feedback_beats_redraw", "leak_clean"):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in ("n_discordant_pairs", "total_codex_calls", "random_seed"):
        if not _is_bare_int(artifact[field]):
            raise ValueError(f"{field} must be a bare int")
    for field in (
        "mcnemar_p",
        "achieved_power",
        "min_detectable_effect",
        "arm_a_gold_rate",
        "arm_b_gold_rate",
        "total_codex_seconds",
        "duration_s",
    ):
        if not _is_bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float")
        if field not in {"total_codex_seconds", "duration_s"} and not (
            0.0 <= artifact[field] <= 1.0
        ):
            raise ValueError(f"{field} must be in [0, 1]")
    if not isinstance(artifact["inference_substrate"], str):
        raise ValueError("inference_substrate must be a string")

    discordant = int(artifact["n_discordant_pairs"])
    if verdict.startswith("complete: feedback_no_better_than_redraw_powered_p"):
        if discordant < DISCORDANT_TARGET:
            raise ValueError("powered null verdict requires n_discordant_pairs>=10")
    if verdict.startswith("complete: feedback_vs_redraw_underpowered_n"):
        if discordant >= DISCORDANT_TARGET:
            raise ValueError("underpowered verdict is invalid after target is reached")
        if not artifact.get("pool_exhausted", False):
            raise ValueError("underpowered verdict requires pool_exhausted=true")
    if artifact["feedback_beats_redraw"] and not verdict.startswith("success:"):
        raise ValueError("feedback_beats_redraw=true requires a success verdict")


def _print_record(record: dict[str, Any]) -> None:
    print(
        f"  {record['task']}: A={int(record['arm_a_correct'])} "
        f"B={int(record['arm_b_correct'])} calls={record['n_calls']} "
        f"codex_s={record['codex_seconds']}",
        flush=True,
    )


def _discordant_count(records: list[dict[str, Any]]) -> int:
    return sum(int(row["arm_a_correct"] != row["arm_b_correct"]) for row in records)


def _run_one_task(
    task: str,
    entries_by_task: dict[str, list[dict[str, Any]]],
    transcripts_dir: Path,
    challenges: dict[str, Any],
    solutions: dict[str, Any],
    iters: int,
    timeout: int,
) -> dict[str, Any]:
    return _paired_task(
        task,
        entries_by_task[task],
        transcripts_dir,
        challenges,
        solutions,
        iters,
        timeout,
    )


def run(
    root: Path = REPO_ROOT,
    pool_path: Path = ARC2_POOL,
    pilot_artifact_path: Path = PILOT_ARTIFACT,
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
    pilot_context = load_pilot_context(pilot_artifact_path)
    preconditions = check_preconditions(pool_path, codex_available_override)
    blocker = blocker_from_preconditions(preconditions)
    if blocker:
        artifact = blocked_artifact(blocker, preconditions, time.time() - started, pilot_context)
        if write:
            _write_json(output_path, artifact)
        return artifact

    pool = load_eval_pool(pool_path)
    entries_by_task = _group_entries_by_task(pool["entries"])
    tasks = tasks_from_eval_pool(pool)
    challenges = load_json(challenges_path) if challenges_path.exists() else {}
    solutions = load_json(solutions_path) if solutions_path.exists() else {}
    if transcripts_dir.exists():
        shutil.rmtree(transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[exp4011] powered paired feedback-vs-redraw over {len(tasks)} eval-pool tasks "
        f"(target discordants={DISCORDANT_TARGET}, feedback iters<={iters}, "
        f"redraws=3, timeout={timeout}s, workers={workers})",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    pool_exhausted = True
    if workers <= 1:
        for task in tasks:
            record = _run_one_task(
                task,
                entries_by_task,
                transcripts_dir,
                challenges,
                solutions,
                iters,
                timeout,
            )
            records.append(record)
            _print_record(record)
            if _discordant_count(records) >= DISCORDANT_TARGET:
                pool_exhausted = False
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start_idx in range(0, len(tasks), workers):
                batch = tasks[start_idx : start_idx + workers]
                futures = executor.map(
                    lambda task: _run_one_task(
                        task,
                        entries_by_task,
                        transcripts_dir,
                        challenges,
                        solutions,
                        iters,
                        timeout,
                    ),
                    batch,
                )
                for record in futures:
                    records.append(record)
                    _print_record(record)
                if _discordant_count(records) >= DISCORDANT_TARGET:
                    pool_exhausted = False
                    break

    audit = audit_transcripts(transcript_paths(transcripts_dir))
    artifact = build_complete_artifact(
        records=records,
        tasks=[record["task"] for record in records],
        all_tasks=tasks,
        preconditions=preconditions,
        transcript_audit=audit,
        pilot_context=pilot_context,
        started_s=started,
        now_s=time.time(),
        pool_exhausted=pool_exhausted,
    )
    if write:
        _write_json(output_path, artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(
        f"   McNemar p={artifact['mcnemar_p']} discordant={artifact['n_discordant_pairs']} "
        f"power={artifact['achieved_power']} MDE={artifact['min_detectable_effect']} "
        f"A_rate={artifact['arm_a_gold_rate']} B_rate={artifact['arm_b_gold_rate']} "
        f"leak_clean={artifact['leak_clean']}",
        flush=True,
    )
    print(
        f"   stop={artifact['stop_reason']} pool_exhausted={artifact['pool_exhausted']} "
        f"calls={artifact['total_codex_calls']} codex_s={artifact['total_codex_seconds']}",
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

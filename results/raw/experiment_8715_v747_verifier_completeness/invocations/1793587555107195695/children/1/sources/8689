"""Exp 3998 GAP-4 de-selection coverage run.

This reruns the poison-skipped .369 de-selection follow-up as exp3998: k=2
fresh <=3-iteration Codex chains on the raw complement of the 12
probe-chain-feasible ARC-2 tasks. Gold is used only after induction for
scoring; prompts are delegated to the existing GAP-4 inducer, which contains
only demos and the test input.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from arc3_gap3_stage2_transition_ebm import SEED, ghash
from arc3_gap4_rule_exec_verifier import demo_fit, induce_program, safe_transform_from_code
from carnot.paths import repo_root


# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
REPO_ROOT = repo_root()
ARC2_POOL = REPO_ROOT / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
CHAIN_ARTIFACT = REPO_ROOT / "results" / "arc3_gap4_arc2_chain_ensemble.json"
OUTPUT = REPO_ROOT / "results" / "experiment_3998_gap4_deselection_coverage.json"
TRANSCRIPTS_DIR = REPO_ROOT / "results" / "experiment_3998_gap4_deselection_transcripts"
CHALLENGES = Path("/home/ianblenke/trm_src/kaggle/combined/arc-agi_evaluation2_challenges.json")
SOLUTIONS = Path("/home/ianblenke/trm_src/kaggle/combined/arc-agi_evaluation2_solutions.json")
INFERENCE_SUBSTRATE = "codex_program_induction_executed_consistency_vs_cached_arc_pool"

REQUIRED_FIELDS = [
    "fresh_chain_demo_perfect_rate_nonselected",
    "cp95_low",
    "cp95_high",
    "debiased_coverage_combined",
    "per_arm_gold_given_perfect",
    "iter0_vs_chainfinal",
    "leak_clean",
    "n_tasks_chained",
    "total_codex_calls",
    "total_codex_seconds",
    "preconditions_checked",
    "random_seed",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "fresh_chain_demo_perfect_rate_nonselected": (
        "BARE FLOAT - the demo-perfect rate on the 11 never-chained tasks."
    ),
    "cp95_low": "Clopper-Pearson 95% interval lower bound on that rate.",
    "cp95_high": "Clopper-Pearson 95% interval upper bound on that rate.",
    "debiased_coverage_combined": (
        "BARE FLOAT - selected fresh-chain arms plus nonselected fresh-chain arms."
    ),
    "per_arm_gold_given_perfect": "P(true-gold | demo-perfect) per fresh arm; gold is post-hoc.",
    "iter0_vs_chainfinal": "Decomposition of demo-perfect attributable to iter0 vs chain final.",
    "leak_clean": "BARE BOOL - every archived transcript passed the word-boundary leak audit.",
    "n_tasks_chained": "Number of raw complement tasks chained in this de-selection run.",
    "total_codex_calls": "Total Codex induction calls recorded by chain histories.",
    "total_codex_seconds": "Total Codex seconds recorded by chain histories.",
    "preconditions_checked": "List of {resource, available} for codex and eval pool.",
    "random_seed": "Shared GAP-4 substrate seed.",
    "honest_verdict": "Terminal-prefix verdict; complete regardless of measured rate.",
    "duration_s": "Wall-clock seconds for this runner.",
    "inference_substrate": "Codex program induction plus cached ARC-2 pool scoring.",
}

LEAK_TOKENS = (
    "arc-agi_evaluation2_solutions",
    "arc_agi_evaluation2_solutions",
    "solutions.json",
    "correct",
    "candidate",
    "gold",
    "type(",
    "os.",
    "open(",
    "__import__",
    "subprocess",
)


def load_eval_pool(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_tasks_from_chain_artifact(artifact: dict[str, Any]) -> list[str]:
    prereg = artifact.get("preregistration", {})
    tasks = prereg.get("tasks")
    if tasks:
        return sorted(str(task) for task in tasks)
    return sorted(str(row["task"]) for row in artifact.get("per_task", []))


def never_chained_tasks(entries: list[dict[str, Any]], chain_artifact: dict[str, Any]) -> list[str]:
    selected = set(selected_tasks_from_chain_artifact(chain_artifact))
    return sorted({str(entry["task"]) for entry in entries if str(entry["task"]) not in selected})


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k + 1))


def _binom_sf_ge(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k, n + 1))


def clopper_pearson_95(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must be in [0, total]")
    alpha = 0.05
    if successes == 0:
        low = 0.0
    else:
        lo, hi = 0.0, successes / total
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if _binom_sf_ge(successes, total, mid) < alpha / 2.0:
                lo = mid
            else:
                hi = mid
        low = hi
    if successes == total:
        high = 1.0
    else:
        lo, hi = successes / total, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if _binom_cdf(successes, total, mid) < alpha / 2.0:
                hi = mid
            else:
                lo = mid
        high = hi
    return round(low, 4), round(high, 4)


def check_preconditions(
    pool_path: Path, codex_available_override: bool | None = None
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


def blocked_artifact(
    verdict: str, preconditions: list[dict[str, Any]], duration_s: float
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "experiment": "experiment_3998_gap4_deselection_coverage",
        "schema": "carnot.experiment_3998_gap4_deselection_coverage.v1",
        "fresh_chain_demo_perfect_rate_nonselected": None,
        "cp95_low": None,
        "cp95_high": None,
        "debiased_coverage_combined": None,
        "per_arm_gold_given_perfect": {},
        "iter0_vs_chainfinal": {},
        "leak_clean": False,
        "n_tasks_chained": 0,
        "total_codex_calls": 0,
        "total_codex_seconds": 0.0,
        "preconditions_checked": preconditions,
        "random_seed": SEED,
        "honest_verdict": verdict,
        "duration_s": round(duration_s, 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def _group_entries_by_task(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry["task"]), []).append(entry)
    return grouped


def _history_iter0_demo_perfect(history: list[dict[str, Any]]) -> bool:
    for row in history:
        if int(row.get("iter", -1)) == 0 and row.get("status") == "graded":
            return float(row.get("demo_fit", 0.0)) >= 1.0
    return False


def _chain_task(
    task: str,
    entries: list[dict[str, Any]],
    transcripts_dir: Path,
    n_fresh: int,
    iters: int,
    timeout: int,
) -> dict[str, Any]:
    arms = []
    demos = entries[0]["demos"]
    for arm_idx in range(1, n_fresh + 1):
        arm_name = f"fresh_chain{arm_idx}"
        arm_dir = transcripts_dir / f"arm{arm_idx}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        rec = induce_program(
            task,
            demos,
            entries[0]["test_input"],
            iters=iters,
            timeout=timeout,
            transcripts_dir=str(arm_dir),
        )
        fn = safe_transform_from_code(rec["code"]) if rec.get("code") else None
        demo_score = demo_fit(fn, demos) if fn else 0.0
        demo_perfect = bool(fn and demo_score >= 1.0)
        arms.append(
            {
                "source": arm_name,
                "code": rec.get("code"),
                "demo_fit": round(demo_score, 4),
                "demo_perfect": demo_perfect,
                "n_calls": int(rec.get("n_calls", 0)),
                "codex_seconds": round(float(rec.get("codex_seconds", 0.0)), 1),
                "history": rec.get("history", []),
                "iter0_demo_perfect": _history_iter0_demo_perfect(rec.get("history", [])),
            }
        )
    return {
        "task": task,
        "arms": arms,
        "n_calls": sum(arm["n_calls"] for arm in arms),
        "codex_seconds": round(sum(arm["codex_seconds"] for arm in arms), 1),
    }


def _ratio_from_text(text: str) -> tuple[int, int]:
    left, right = str(text).split("/", 1)
    return int(left), int(right)


def selected_fresh_counts(chain_artifact: dict[str, Any]) -> tuple[int, int]:
    value = chain_artifact.get("fresh_chain_arms_demo_perfect")
    if value is not None:
        return _ratio_from_text(str(value))
    successes = total = 0
    for row in chain_artifact.get("per_task", []):
        for arm in row.get("arms", []):
            if str(arm.get("source", "")).startswith("fresh_chain"):
                total += 1
                successes += int(bool(arm.get("demo_perfect")))
    return successes, total


def load_gold(challenges_path: Path, solutions_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_json(challenges_path), load_json(solutions_path)


def gold_for(
    task: str,
    test_input: Any,
    challenges: dict[str, Any],
    solutions: dict[str, Any],
) -> np.ndarray | None:
    task_challenge = challenges.get(task)
    if not task_challenge or task not in solutions:
        return None
    target_hash = ghash(np.asarray(test_input))
    for idx, pair in enumerate(task_challenge.get("test", [])):
        if ghash(np.asarray(pair["input"])) == target_hash:
            return np.asarray(solutions[task][idx])
    return None


def _score_gold_for_arm(
    arm: dict[str, Any],
    entries: list[dict[str, Any]],
    challenges: dict[str, Any],
    solutions: dict[str, Any],
) -> tuple[int, int]:
    if not arm.get("demo_perfect") or not arm.get("code"):
        return 0, 0
    fn = safe_transform_from_code(str(arm["code"]))
    if fn is None:
        return 0, 0
    gold = total = 0
    for entry in entries:
        target = gold_for(str(entry["task"]), entry["test_input"], challenges, solutions)
        pred = fn(entry["test_input"])
        total += 1
        gold += int(pred is not None and target is not None and np.array_equal(pred, target))
    return gold, total


def _rate(successes: int, total: int) -> float | None:
    return round(successes / total, 4) if total else None


def transcript_paths(transcripts_dir: Path) -> list[Path]:
    return sorted(path for path in transcripts_dir.glob("arm*/*.txt") if path.is_file())


def _token_pattern(token: str) -> re.Pattern[str]:
    if token in {"type(", "os."}:
        return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(token))
    if re.fullmatch(r"[A-Za-z0-9_]+", token):
        return re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", re.IGNORECASE
        )
    return re.compile(re.escape(token), re.IGNORECASE)


def audit_transcripts(paths: list[Path]) -> dict[str, Any]:
    violations = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in LEAK_TOKENS:
            if _token_pattern(token).search(text):
                violations.append({"path": str(path), "token": token})
                break
    return {"clean": not violations, "n_transcripts": len(paths), "violations": violations}


def build_complete_artifact(
    records: list[dict[str, Any]],
    entries_by_task: dict[str, list[dict[str, Any]]],
    selected_successes: int,
    selected_total: int,
    transcript_audit: dict[str, Any],
    preconditions: list[dict[str, Any]],
    challenges: dict[str, Any],
    solutions: dict[str, Any],
    started_s: float,
    now_s: float,
) -> dict[str, Any]:
    arm_successes = 0
    arm_total = 0
    per_arm_counts: dict[str, list[int]] = {}
    iter0 = chain_final = recovered = 0
    for record in records:
        for arm in record["arms"]:
            source = str(arm["source"])
            arm_total += 1
            arm_successes += int(bool(arm["demo_perfect"]))
            iter0_ok = bool(arm.get("iter0_demo_perfect"))
            final_ok = bool(arm.get("demo_perfect"))
            iter0 += int(iter0_ok)
            chain_final += int(final_ok)
            recovered += int(final_ok and not iter0_ok)
            gold, total = _score_gold_for_arm(
                arm,
                entries_by_task[str(record["task"])],
                challenges,
                solutions,
            )
            slot = per_arm_counts.setdefault(source, [0, 0])
            slot[0] += gold
            slot[1] += total
    fresh_gold = [
        sum(v[0] for v in per_arm_counts.values()),
        sum(v[1] for v in per_arm_counts.values()),
    ]
    per_arm_counts["fresh"] = fresh_gold
    rate = _rate(arm_successes, arm_total)
    cp95_low, cp95_high = clopper_pearson_95(arm_successes, arm_total)
    combined = _rate(selected_successes + arm_successes, selected_total + arm_total)
    iter_summary = {
        "total_arms": arm_total,
        "iter0_demo_perfect": iter0,
        "chain_final_demo_perfect": chain_final,
        "recovered_by_chain": recovered,
        "failed_after_chain": arm_total - chain_final,
        "iter0_rate": _rate(iter0, arm_total),
        "chain_final_rate": _rate(chain_final, arm_total),
    }
    per_arm_gold = {
        source: {"gold": counts[0], "n": counts[1], "rate": _rate(counts[0], counts[1])}
        for source, counts in sorted(per_arm_counts.items())
    }
    duration_s = round(now_s - started_s, 1)
    verdict_rate = str(rate).rstrip("0").rstrip(".") if rate is not None else "none"
    artifact: dict[str, Any] = {
        "experiment": "experiment_3998_gap4_deselection_coverage",
        "schema": "carnot.experiment_3998_gap4_deselection_coverage.v1",
        "title": "GAP-4 de-selection fresh-chain coverage on never-chained ARC-2 tasks",
        "fresh_chain_demo_perfect_rate_nonselected": rate,
        "cp95_low": cp95_low,
        "cp95_high": cp95_high,
        "debiased_coverage_combined": combined,
        "per_arm_gold_given_perfect": per_arm_gold,
        "iter0_vs_chainfinal": iter_summary,
        "leak_clean": bool(transcript_audit["clean"]),
        "leak_audit": transcript_audit,
        "n_tasks_chained": len(records),
        "n_chain_arms_nonselected": arm_total,
        "demo_perfect_arms_nonselected": arm_successes,
        "selected_chain_feasible_demo_perfect_arms": selected_successes,
        "selected_chain_feasible_total_arms": selected_total,
        "never_chained_tasks": [str(row["task"]) for row in records],
        "per_task": records,
        "total_codex_calls": sum(int(record["n_calls"]) for record in records),
        "total_codex_seconds": round(sum(float(record["codex_seconds"]) for record in records), 1),
        "preconditions_checked": preconditions,
        "random_seed": SEED,
        "honest_verdict": f"complete: gap4_deselection_coverage_{verdict_rate}_n{len(records)}",
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


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
    if artifact["fresh_chain_demo_perfect_rate_nonselected"] is not None and not _is_bare_float(
        artifact["fresh_chain_demo_perfect_rate_nonselected"]
    ):
        raise ValueError("fresh_chain_demo_perfect_rate_nonselected must be a bare float or null")
    for field in ("cp95_low", "cp95_high", "debiased_coverage_combined", "duration_s"):
        if artifact[field] is not None and not _is_bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float or null")
    if not isinstance(artifact["leak_clean"], bool):
        raise ValueError("leak_clean must be a bare bool")
    for field in ("n_tasks_chained", "total_codex_calls"):
        if not isinstance(artifact[field], int) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare int")
    if not _is_bare_float(artifact["total_codex_seconds"]):
        raise ValueError("total_codex_seconds must be a bare float")
    if not isinstance(artifact["preconditions_checked"], list):
        raise ValueError("preconditions_checked must be a list")


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
    n_fresh: int = 2,
    expected_nonselected_count: int = 11,
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
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        return artifact

    pool = load_eval_pool(pool_path)
    chain_artifact = load_json(chain_artifact_path)
    entries = pool["entries"]
    tasks = never_chained_tasks(entries, chain_artifact)
    if len(tasks) != expected_nonselected_count:
        raise ValueError(
            f"expected {expected_nonselected_count} never-chained tasks, found {len(tasks)}"
        )
    entries_by_task = _group_entries_by_task(entries)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[exp3998] k={n_fresh} fresh chains on {len(tasks)} never-chained tasks "
        f"(iters<={iters}, timeout={timeout}s, workers={workers})",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(
            lambda task: _chain_task(
                task,
                entries_by_task[task],
                transcripts_dir,
                n_fresh,
                iters,
                timeout,
            ),
            tasks,
        )
        for record in futures:
            records.append(record)
            print(
                f"  {record['task']}: demo-perfect "
                f"{sum(int(arm['demo_perfect']) for arm in record['arms'])}/{n_fresh}, "
                f"calls={record['n_calls']} codex_s={record['codex_seconds']}",
                flush=True,
            )

    challenges, solutions = load_gold(challenges_path, solutions_path)
    selected_successes, selected_total = selected_fresh_counts(chain_artifact)
    audit = audit_transcripts(transcript_paths(transcripts_dir))
    artifact = build_complete_artifact(
        records=records,
        entries_by_task=entries_by_task,
        selected_successes=selected_successes,
        selected_total=selected_total,
        transcript_audit=audit,
        preconditions=preconditions,
        challenges=challenges,
        solutions=solutions,
        started_s=started,
        now_s=time.time(),
    )
    if write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(
        f"   nonselected fresh-chain rate="
        f"{artifact['fresh_chain_demo_perfect_rate_nonselected']} "
        f"CP95=[{artifact['cp95_low']}, {artifact['cp95_high']}]",
        flush=True,
    )
    print(
        f"   combined debiased coverage={artifact['debiased_coverage_combined']}; "
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

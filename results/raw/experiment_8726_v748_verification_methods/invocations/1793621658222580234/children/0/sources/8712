"""Exp 3999 GAP-4 precision confirmation v2.

This is the pre-registered follow-up to the k=3 chain-arms ARC-2 round. It
uses only all-fresh Codex chains, commits the task set and gates to the output
artifact before the first Codex call, then scores agreement precision post-hoc.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from arc3_gap3_stage2_transition_ebm import SEED, ghash
from arc3_gap4_rule_exec_verifier import demo_fit, induce_program, safe_transform_from_code
from experiment_3998_gap4_deselection_coverage import (
    audit_transcripts,
    gold_for,
    load_json,
    transcript_paths,
)
from carnot.paths import repo_root


# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
REPO_ROOT = repo_root()
ARC2_POOL = REPO_ROOT / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
CHAIN_ARTIFACT = REPO_ROOT / "results" / "arc3_gap4_arc2_chain_ensemble.json"
OUTPUT = REPO_ROOT / "results" / "experiment_3999_gap4_precision_confirmation_v2.json"
TRANSCRIPTS_DIR = REPO_ROOT / "results" / "experiment_3999_gap4_precision_transcripts"
CHALLENGES = Path("/home/ianblenke/trm_src/kaggle/combined/arc-agi_evaluation2_challenges.json")
SOLUTIONS = Path("/home/ianblenke/trm_src/kaggle/combined/arc-agi_evaluation2_solutions.json")
INFERENCE_SUBSTRATE = "codex_program_induction_all_fresh_k3_posthoc_arc2_gold_scoring"
EXCLUDED_REUSE_TASKS = ("aa4ec2a5", "16b78196")

REQUIRED_FIELDS = [
    "protocol_preregistered",
    "n_agreement_events",
    "n_gold_given_agreement",
    "primary_gate_passed",
    "precision_vs_fresharm_base",
    "sibling_disagreement_tripwire_gold_rate",
    "agreement_is_selector_not_label",
    "missing_verifier_gaps",
    "total_codex_calls",
    "total_codex_seconds",
    "leak_clean",
    "random_seed",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "protocol_preregistered": (
        "BARE BOOL - the gate + task set were committed before any codex call."
    ),
    "n_agreement_events": "BARE INT - agreement events observed; the binomial n.",
    "n_gold_given_agreement": "BARE INT - gold agreement events; the binomial successes.",
    "primary_gate_passed": "BARE BOOL - n>=19 and gold>=14.",
    "precision_vs_fresharm_base": "BARE FLOAT - agreement precision minus fresh-arm base rate.",
    "sibling_disagreement_tripwire_gold_rate": (
        "BARE FLOAT - gold rate after unanimity/sibling-disagreement abstention."
    ),
    "agreement_is_selector_not_label": (
        "BARE BOOL - true only when the powered primary selector gate passes."
    ),
    "missing_verifier_gaps": "Residual wrong-agreement cases for missing-verifier logging.",
    "total_codex_calls": "Total Codex induction calls recorded by fresh chains.",
    "total_codex_seconds": "Total Codex wall seconds recorded by fresh chains.",
    "leak_clean": "BARE BOOL - every archived transcript passed the leak audit.",
    "random_seed": "Shared GAP-4 substrate seed.",
    "honest_verdict": "Terminal-prefix verdict.",
    "duration_s": "Wall-clock seconds for this runner.",
    "inference_substrate": "Codex program induction substrate and post-hoc scoring mode.",
}


def load_eval_pool(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def selected_tasks_from_chain_artifact(artifact: dict[str, Any]) -> list[str]:
    prereg = artifact.get("preregistration", {})
    tasks = prereg.get("tasks")
    if tasks:
        return sorted(str(task) for task in tasks)
    return sorted(str(row["task"]) for row in artifact.get("per_task", []))


def clean_new_tasks(
    entries: list[dict[str, Any]],
    chain_artifact: dict[str, Any],
    excluded: tuple[str, ...] = EXCLUDED_REUSE_TASKS,
) -> list[str]:
    selected = set(selected_tasks_from_chain_artifact(chain_artifact))
    blocked = set(excluded)
    return sorted(
        {
            str(entry["task"])
            for entry in entries
            if str(entry["task"]) not in selected and str(entry["task"]) not in blocked
        }
    )


def primary_gate_passed(n_gold: int, n_events: int) -> bool:
    return n_events >= 19 and n_gold >= 14


def verdict_for(n_gold: int, n_events: int, primary_passed: bool) -> str:
    if primary_passed:
        return f"success: gap4_precision_confirmed_{n_gold}of{n_events}_gold"
    return f"complete: gap4_agreement_confidence_label_only_{n_gold}of{n_events}"


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


def _base_artifact(
    honest_verdict: str,
    duration_s: float,
    protocol_preregistered: bool,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_3999_gap4_precision_confirmation_v2",
        "schema": "carnot.experiment_3999_gap4_precision_confirmation_v2.v1",
        "title": "GAP-4 k=3 all-fresh agreement precision confirmation v2",
        "protocol_preregistered": protocol_preregistered,
        "n_agreement_events": 0,
        "n_gold_given_agreement": 0,
        "primary_gate_passed": False,
        "precision_vs_fresharm_base": 0.0,
        "sibling_disagreement_tripwire_gold_rate": 0.0,
        "agreement_is_selector_not_label": False,
        "missing_verifier_gaps": [],
        "total_codex_calls": 0,
        "total_codex_seconds": 0.0,
        "leak_clean": False,
        "random_seed": SEED,
        "honest_verdict": honest_verdict,
        "duration_s": round(duration_s, 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
    }


def blocked_artifact(
    verdict: str,
    preconditions: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    artifact = _base_artifact(verdict, duration_s, protocol_preregistered=False)
    artifact["preconditions_checked"] = preconditions
    validate_artifact(artifact)
    return artifact


def preregistered_artifact(
    tasks: list[str],
    entries_by_task: dict[str, list[dict[str, Any]]],
    preconditions: list[dict[str, Any]],
    started_s: float,
    n_fresh: int,
    timeout: int,
) -> dict[str, Any]:
    n_entries = sum(len(entries_by_task[task]) for task in tasks)
    artifact = _base_artifact(
        "complete: protocol_preregistered_pending_execution",
        time.time() - started_s,
        protocol_preregistered=True,
    )
    artifact.update(
        {
            "preregistration": {
                "registered_before_codex_call": True,
                "task_set": tasks,
                "excluded_reuse_tasks": list(EXCLUDED_REUSE_TASKS),
                "selection_rule": (
                    "all tasks in the ARC-2 GAP-4 eval pool not used in the prior "
                    "chain-ensemble round, excluding flagged reuse contamination"
                ),
                "primary_binomial_gate": {
                    "min_agreement_events": 19,
                    "min_gold_given_agreement": 14,
                    "null_precision": 0.52,
                    "size": 0.046,
                },
                "secondary": "agreement precision minus in-run fresh-arm gold rate",
                "tertiary": "task-level unanimity-with-abstention sibling-input tripwire",
                "n_fresh_chains": n_fresh,
                "timeout_s_per_chain": timeout,
                "committed_event_capacity_entries": n_entries,
                "powered_target_reachable_from_committed_pool": n_entries >= 19,
            },
            "preconditions_checked": preconditions,
        }
    )
    validate_artifact(artifact)
    return artifact


def _group_entries_by_task(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry["task"]), []).append(entry)
    return grouped


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
    target = gold_for(task, entry["test_input"], challenges, solutions)
    if target is not None:
        return target
    return _gold_from_entry_candidates(entry)


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
        predictions = []
        for input_idx, entry in enumerate(entries):
            pred = fn(entry["test_input"]) if demo_perfect else None
            predictions.append(
                {
                    "input_idx": input_idx,
                    "pred_hash": _grid_hash(pred) if pred is not None else None,
                    "pred_grid": pred.tolist() if pred is not None else None,
                }
            )
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
                "predictions": predictions,
            }
        )
    return {
        "task": task,
        "arms": arms,
        "n_calls": sum(arm["n_calls"] for arm in arms),
        "codex_seconds": round(sum(float(arm["codex_seconds"]) for arm in arms), 1),
    }


def _agreement_for_input(
    task: str,
    input_idx: int,
    arms: list[dict[str, Any]],
    target: np.ndarray | None,
) -> dict[str, Any]:
    outputs = []
    for arm in arms:
        if not arm.get("demo_perfect"):
            continue
        pred = arm["predictions"][input_idx]
        if pred.get("pred_hash") is None:
            continue
        outputs.append((str(pred["pred_hash"]), pred["pred_grid"], str(arm["source"])))
    counts = Counter(row[0] for row in outputs)
    agreed_hash = None
    n_matching = 0
    if counts:
        agreed_hash, n_matching = counts.most_common(1)[0]
    agreement = bool(agreed_hash is not None and n_matching >= 2)
    agreed_pred = None
    agreeing_sources: list[str] = []
    if agreement:
        for pred_hash, pred_grid, source in outputs:
            if pred_hash == agreed_hash:
                agreed_pred = pred_grid
                agreeing_sources.append(source)
    n_outputs = len(counts)
    return {
        "task": task,
        "input_idx": input_idx,
        "agreement": agreement,
        "agreed_hash": agreed_hash if agreement else None,
        "agreed_pred": agreed_pred,
        "agreed_is_gold": _score_grid(agreed_pred, target) if agreement else False,
        "agreeing_sources": agreeing_sources,
        "n_matching_outputs": n_matching if agreement else 0,
        "n_demo_perfect_arms": len(outputs),
        "n_outputs": n_outputs,
        "unanimous_demo_perfect_output": bool(len(outputs) >= 2 and n_outputs == 1),
    }


def _rate(successes: int, total: int) -> float:
    return round(successes / total, 4) if total else 0.0


def _summarize_records(
    records: list[dict[str, Any]],
    entries_by_task: dict[str, list[dict[str, Any]]],
    challenges: dict[str, Any],
    solutions: dict[str, Any],
) -> dict[str, Any]:
    agreement_events = []
    fresh_gold = 0
    fresh_total = 0
    per_task = []
    for record in records:
        task = str(record["task"])
        entries = entries_by_task[task]
        per_input = []
        for input_idx, entry in enumerate(entries):
            target = gold_for_entry(entry, challenges, solutions)
            row = _agreement_for_input(task, input_idx, record["arms"], target)
            per_input.append(row)
            if row["agreement"]:
                agreement_events.append(row)
            for arm in record["arms"]:
                pred_grid = arm["predictions"][input_idx]["pred_grid"]
                if arm.get("demo_perfect") and pred_grid is not None and target is not None:
                    fresh_total += 1
                    fresh_gold += int(_score_grid(pred_grid, target))
        task_abstains = any(
            row["n_demo_perfect_arms"] >= 2 and row["n_outputs"] > 1 for row in per_input
        )
        for row in per_input:
            row["sibling_disagreement_abstains"] = task_abstains
        per_task.append(
            {
                "task": task,
                "n_entries": len(entries),
                "sibling_disagreement_abstains": task_abstains,
                "arms": record["arms"],
                "per_input": per_input,
                "n_calls": record["n_calls"],
                "codex_seconds": record["codex_seconds"],
            }
        )
    kept = [row for row in agreement_events if not row["sibling_disagreement_abstains"]]
    n_gold = sum(int(row["agreed_is_gold"]) for row in agreement_events)
    kept_gold = sum(int(row["agreed_is_gold"]) for row in kept)
    missing = []
    for row in agreement_events:
        if row["agreed_is_gold"]:
            continue
        if row["sibling_disagreement_abstains"]:
            failure_mode = "agreement_wrong_but_unanimity_tripwire_abstains"
            discriminator = "GAP-5 demo-underdetermination sibling-input tripwire"
        else:
            failure_mode = "agreement_wrong_and_tripwire_keeps"
            discriminator = "residual rule underdetermination beyond sibling unanimity"
        missing.append(
            {
                "task": row["task"],
                "input_idx": row["input_idx"],
                "failure_mode": failure_mode,
                "missing_discriminator": discriminator,
            }
        )
    return {
        "agreement_events": agreement_events,
        "n_agreement_events": len(agreement_events),
        "n_gold_given_agreement": n_gold,
        "fresh_arm_gold": fresh_gold,
        "fresh_arm_total": fresh_total,
        "fresh_arm_base_rate": _rate(fresh_gold, fresh_total),
        "sibling_disagreement_tripwire_gold_rate": _rate(kept_gold, len(kept)),
        "sibling_tripwire_kept_events": len(kept),
        "missing_verifier_gaps": missing,
        "per_task": per_task,
    }


def build_complete_artifact(
    records: list[dict[str, Any]],
    entries_by_task: dict[str, list[dict[str, Any]]],
    preregistration: dict[str, Any],
    preconditions: list[dict[str, Any]],
    transcript_audit: dict[str, Any],
    challenges: dict[str, Any],
    solutions: dict[str, Any],
    started_s: float,
    now_s: float,
) -> dict[str, Any]:
    summary = _summarize_records(records, entries_by_task, challenges, solutions)
    n_events = int(summary["n_agreement_events"])
    n_gold = int(summary["n_gold_given_agreement"])
    agreement_precision = _rate(n_gold, n_events)
    precision_vs_base = round(agreement_precision - float(summary["fresh_arm_base_rate"]), 4)
    primary_passed = primary_gate_passed(n_gold, n_events)
    artifact = _base_artifact(
        verdict_for(n_gold, n_events, primary_passed),
        now_s - started_s,
        protocol_preregistered=True,
    )
    artifact.update(
        {
            "preregistration": preregistration,
            "preconditions_checked": preconditions,
            "n_agreement_events": n_events,
            "n_gold_given_agreement": n_gold,
            "primary_gate_passed": primary_passed,
            "agreement_precision": agreement_precision,
            "fresh_arm_base_rate": summary["fresh_arm_base_rate"],
            "fresh_arm_gold": summary["fresh_arm_gold"],
            "fresh_arm_total": summary["fresh_arm_total"],
            "precision_vs_fresharm_base": precision_vs_base,
            "sibling_disagreement_tripwire_gold_rate": summary[
                "sibling_disagreement_tripwire_gold_rate"
            ],
            "sibling_tripwire_kept_events": summary["sibling_tripwire_kept_events"],
            "agreement_is_selector_not_label": primary_passed,
            "missing_verifier_gaps": summary["missing_verifier_gaps"],
            "agreement_events": summary["agreement_events"],
            "per_task": summary["per_task"],
            "total_codex_calls": sum(int(record["n_calls"]) for record in records),
            "total_codex_seconds": round(
                sum(float(record["codex_seconds"]) for record in records), 1
            ),
            "leak_clean": bool(transcript_audit["clean"]),
            "leak_audit": transcript_audit,
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
    for field in (
        "protocol_preregistered",
        "primary_gate_passed",
        "agreement_is_selector_not_label",
        "leak_clean",
    ):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in (
        "n_agreement_events",
        "n_gold_given_agreement",
        "total_codex_calls",
        "random_seed",
    ):
        if not _is_bare_int(artifact[field]):
            raise ValueError(f"{field} must be a bare int")
    for field in (
        "precision_vs_fresharm_base",
        "sibling_disagreement_tripwire_gold_rate",
        "total_codex_seconds",
        "duration_s",
    ):
        if not _is_bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float")
    if not isinstance(artifact["missing_verifier_gaps"], list):
        raise ValueError("missing_verifier_gaps must be a list")
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
    n_fresh: int = 3,
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
    entries = pool["entries"]
    tasks = clean_new_tasks(entries, chain_artifact)
    entries_by_task = _group_entries_by_task(entries)
    committed_entries_by_task = {task: entries_by_task[task] for task in tasks}
    prereg = preregistered_artifact(
        tasks=tasks,
        entries_by_task=committed_entries_by_task,
        preconditions=preconditions,
        started_s=started,
        n_fresh=n_fresh,
        timeout=timeout,
    )
    if write:
        _write_json(output_path, prereg)

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[exp3999] preregistered {len(tasks)} clean new tasks / "
        f"{sum(len(committed_entries_by_task[t]) for t in tasks)} entries; "
        f"k={n_fresh}, iters<={iters}, timeout={timeout}s, workers={workers}",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(
            lambda task: _chain_task(
                task,
                committed_entries_by_task[task],
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

    challenges = load_json(challenges_path) if challenges_path.exists() else {}
    solutions = load_json(solutions_path) if solutions_path.exists() else {}
    audit = audit_transcripts(transcript_paths(transcripts_dir))
    artifact = build_complete_artifact(
        records=records,
        entries_by_task=committed_entries_by_task,
        preregistration=prereg["preregistration"],
        preconditions=preconditions,
        transcript_audit=audit,
        challenges=challenges,
        solutions=solutions,
        started_s=started,
        now_s=time.time(),
    )
    if write:
        _write_json(output_path, artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(
        f"   agreement={artifact['n_gold_given_agreement']}/"
        f"{artifact['n_agreement_events']} precision={artifact['agreement_precision']} "
        f"fresh_base={artifact['fresh_arm_base_rate']} "
        f"delta={artifact['precision_vs_fresharm_base']}",
        flush=True,
    )
    print(
        f"   tripwire_gold_rate={artifact['sibling_disagreement_tripwire_gold_rate']} "
        f"kept={artifact['sibling_tripwire_kept_events']} leak_clean={artifact['leak_clean']}",
        flush=True,
    )
    return artifact


def main() -> None:  # pragma: no cover - exercised by required script command.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    run(workers=args.workers, iters=args.iters, timeout=args.timeout)


if __name__ == "__main__":  # pragma: no cover
    main()

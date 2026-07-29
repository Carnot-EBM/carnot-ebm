"""Exp 4009 GAP-4 precision confirmation v3.

This reruns the Exp 3999 precision confirmation with the missing execution
floor fixed. The task set and statistical gates are written before any Codex
call, but a terminal complete/success verdict is valid only after real Codex
calls produce at least one agreement event.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import experiment_3999_gap4_precision_confirmation_v2 as v2
from arc3_gap3_stage2_transition_ebm import SEED
from carnot.paths import repo_root


# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
REPO_ROOT = repo_root()
ARC2_POOL = REPO_ROOT / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
CHAIN_ARTIFACT = REPO_ROOT / "results" / "arc3_gap4_arc2_chain_ensemble.json"
OUTPUT = REPO_ROOT / "results" / "experiment_4009_gap4_precision_confirmation_v3.json"
TRANSCRIPTS_DIR = REPO_ROOT / "results" / "experiment_4009_gap4_precision_transcripts"
CHALLENGES = Path("/home/ianblenke/trm_src/kaggle/combined/arc-agi_evaluation2_challenges.json")
SOLUTIONS = Path("/home/ianblenke/trm_src/kaggle/combined/arc-agi_evaluation2_solutions.json")
INFERENCE_SUBSTRATE = "codex_program_induction_all_fresh_k3_execution_floor_v3"
EXCLUDED_REUSE_TASKS = v2.EXCLUDED_REUSE_TASKS
AGREEMENT_EVENT_TARGET = 19
PRIMARY_GOLD_TARGET = 14

REQUIRED_FIELDS = [
    "execution_floor_met",
    "protocol_preregistered",
    "n_agreement_events",
    "n_gold_given_agreement",
    "primary_gate_passed",
    "precision_vs_fresharm_base",
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
    "execution_floor_met": (
        "BARE BOOL -- total_codex_calls>0 AND n_agreement_events>0 "
        "(the .370-fix guard; false with codex available forbids complete verdicts)."
    ),
    "protocol_preregistered": (
        "BARE BOOL -- the gate + task set were committed before any codex call."
    ),
    "n_agreement_events": "BARE INT -- agreement events observed; the binomial n.",
    "n_gold_given_agreement": "BARE INT -- gold among agreement events.",
    "primary_gate_passed": (
        "BARE BOOL -- gold-given-agreement cleared the n>=19, >=14 critical value."
    ),
    "precision_vs_fresharm_base": (
        "BARE FLOAT -- agreement precision minus the in-run fresh-arm base rate."
    ),
    "agreement_is_selector_not_label": (
        "BARE BOOL -- true only when the powered primary selector gate passes."
    ),
    "missing_verifier_gaps": "Residual cases agreement mis-labels.",
    "total_codex_calls": "Total Codex induction calls; 0 calls is invalid for completion.",
    "total_codex_seconds": "Total Codex wall seconds recorded by fresh chains.",
    "leak_clean": "BARE BOOL -- every archived transcript passed the leak audit.",
    "random_seed": "Shared GAP-4 substrate seed.",
    "honest_verdict": "Terminal-prefix verdict.",
    "duration_s": "Wall-clock seconds for this runner.",
    "inference_substrate": "Codex program-induction substrate and post-hoc scoring mode.",
}

load_eval_pool = v2.load_eval_pool
load_json = v2.load_json
clean_new_tasks = v2.clean_new_tasks
primary_gate_passed = v2.primary_gate_passed
verdict_for = v2.verdict_for
group_entries_by_task = v2._group_entries_by_task
grid_hash = v2._grid_hash


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_preconditions(
    pool_path: Path,
    codex_available_override: bool | None = None,
) -> list[dict[str, Any]]:
    return v2.check_preconditions(pool_path, codex_available_override)


def blocker_from_preconditions(preconditions: list[dict[str, Any]]) -> str | None:
    return v2.blocker_from_preconditions(preconditions)


def execution_floor_met(total_codex_calls: int, n_agreement_events: int) -> bool:
    return total_codex_calls > 0 and n_agreement_events > 0


def _base_artifact(
    honest_verdict: str,
    duration_s: float,
    protocol_preregistered: bool,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4009_gap4_precision_confirmation_v3",
        "schema": "carnot.experiment_4009_gap4_precision_confirmation_v3.v1",
        "title": "GAP-4 k=3 all-fresh agreement precision confirmation v3",
        "execution_floor_met": False,
        "protocol_preregistered": protocol_preregistered,
        "n_agreement_events": 0,
        "n_gold_given_agreement": 0,
        "primary_gate_passed": False,
        "precision_vs_fresharm_base": 0.0,
        "agreement_is_selector_not_label": False,
        "missing_verifier_gaps": [],
        "total_codex_calls": 0,
        "total_codex_seconds": 0.0,
        "leak_clean": False,
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
    now_s: float,
    n_fresh: int,
    timeout: int,
) -> dict[str, Any]:
    n_entries = sum(len(entries_by_task[task]) for task in tasks)
    artifact = _base_artifact(
        "blocked_execution_floor_unmet",
        now_s - started_s,
        protocol_preregistered=True,
    )
    artifact.update(
        {
            "artifact_phase": "preregistered_before_execution",
            "preregistration": {
                "registered_before_codex_call": True,
                "task_set": tasks,
                "excluded_reuse_tasks": list(EXCLUDED_REUSE_TASKS),
                "selection_rule": (
                    "all tasks in the ARC-2 GAP-4 eval pool not used in the prior "
                    "chain-ensemble round, excluding flagged reuse contamination"
                ),
                "primary_binomial_gate": {
                    "min_agreement_events": AGREEMENT_EVENT_TARGET,
                    "min_gold_given_agreement": PRIMARY_GOLD_TARGET,
                    "null_precision": 0.52,
                    "size": 0.046,
                },
                "secondary": "agreement precision minus in-run fresh-arm gold rate",
                "n_fresh_chains": n_fresh,
                "timeout_s_per_chain": timeout,
                "committed_event_capacity_entries": n_entries,
                "powered_target_reachable_from_committed_pool": (
                    n_entries >= AGREEMENT_EVENT_TARGET
                ),
            },
            "preconditions_checked": preconditions,
        }
    )
    validate_artifact(artifact)
    return artifact


def agreement_events_so_far(
    records: list[dict[str, Any]],
    entries_by_task: dict[str, list[dict[str, Any]]],
) -> int:
    count = 0
    for record in records:
        task = str(record["task"])
        for input_idx, _entry in enumerate(entries_by_task[task]):
            row = v2._agreement_for_input(task, input_idx, record["arms"], target=None)
            count += int(row["agreement"])
    return count


def _rate(successes: int, total: int) -> float:
    return round(successes / total, 4) if total else 0.0


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
    pool_exhausted: bool,
) -> dict[str, Any]:
    summary = v2._summarize_records(records, entries_by_task, challenges, solutions)
    n_events = int(summary["n_agreement_events"])
    n_gold = int(summary["n_gold_given_agreement"])
    total_calls = sum(int(record["n_calls"]) for record in records)
    total_seconds = round(sum(float(record["codex_seconds"]) for record in records), 1)
    floor = execution_floor_met(total_calls, n_events)
    agreement_precision = _rate(n_gold, n_events)
    precision_vs_base = round(agreement_precision - float(summary["fresh_arm_base_rate"]), 4)
    primary_passed = floor and primary_gate_passed(n_gold, n_events)
    honest_verdict = (
        verdict_for(n_gold, n_events, primary_passed) if floor else "blocked_execution_floor_unmet"
    )

    artifact = _base_artifact(
        honest_verdict,
        now_s - started_s,
        protocol_preregistered=True,
    )
    artifact.update(
        {
            "artifact_phase": "terminal",
            "preregistration": preregistration,
            "preconditions_checked": preconditions,
            "execution_floor_met": floor,
            "n_agreement_events": n_events,
            "n_gold_given_agreement": n_gold,
            "primary_gate_passed": primary_passed,
            "agreement_precision": agreement_precision,
            "fresh_arm_base_rate": summary["fresh_arm_base_rate"],
            "fresh_arm_gold": summary["fresh_arm_gold"],
            "fresh_arm_total": summary["fresh_arm_total"],
            "precision_vs_fresharm_base": precision_vs_base,
            "agreement_event_target": AGREEMENT_EVENT_TARGET,
            "powered_target_reached": n_events >= AGREEMENT_EVENT_TARGET,
            "draw_stop_reason": (
                "powered_target_met"
                if n_events >= AGREEMENT_EVENT_TARGET
                else "pool_exhausted"
                if pool_exhausted
                else "stopped_before_target"
            ),
            "sibling_disagreement_tripwire_gold_rate": summary[
                "sibling_disagreement_tripwire_gold_rate"
            ],
            "sibling_tripwire_kept_events": summary["sibling_tripwire_kept_events"],
            "agreement_is_selector_not_label": primary_passed,
            "missing_verifier_gaps": summary["missing_verifier_gaps"],
            "agreement_events": summary["agreement_events"],
            "per_task": summary["per_task"],
            "total_codex_calls": total_calls,
            "total_codex_seconds": total_seconds,
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
    if "pending_execution" in verdict:
        raise ValueError("honest_verdict must never emit pending_execution")
    for field in (
        "execution_floor_met",
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
        "total_codex_seconds",
        "duration_s",
    ):
        if not _is_bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float")
    if not isinstance(artifact["missing_verifier_gaps"], list):
        raise ValueError("missing_verifier_gaps must be a list")
    if not isinstance(artifact["inference_substrate"], str):
        raise ValueError("inference_substrate must be a string")

    computed_floor = execution_floor_met(
        int(artifact["total_codex_calls"]),
        int(artifact["n_agreement_events"]),
    )
    if artifact["execution_floor_met"] is not computed_floor:
        raise ValueError(
            "execution_floor_met must equal total_codex_calls>0 AND n_agreement_events>0"
        )
    if verdict.startswith(("complete:", "success:")) and not artifact["execution_floor_met"]:
        raise ValueError("complete/success verdict requires execution floor")
    if artifact["primary_gate_passed"] and not primary_gate_passed(
        int(artifact["n_gold_given_agreement"]),
        int(artifact["n_agreement_events"]),
    ):
        raise ValueError("primary_gate_passed must follow the preregistered critical value")


def _print_record(record: dict[str, Any], n_fresh: int) -> None:
    print(
        f"  {record['task']}: demo-perfect "
        f"{sum(int(arm['demo_perfect']) for arm in record['arms'])}/{n_fresh}, "
        f"calls={record['n_calls']} codex_s={record['codex_seconds']}",
        flush=True,
    )


def _chain_one(
    task: str,
    entries_by_task: dict[str, list[dict[str, Any]]],
    transcripts_dir: Path,
    n_fresh: int,
    iters: int,
    timeout: int,
) -> dict[str, Any]:
    return v2._chain_task(task, entries_by_task[task], transcripts_dir, n_fresh, iters, timeout)


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
    entries_by_task = group_entries_by_task(entries)
    committed_entries_by_task = {task: entries_by_task[task] for task in tasks}
    prereg = preregistered_artifact(
        tasks=tasks,
        entries_by_task=committed_entries_by_task,
        preconditions=preconditions,
        started_s=started,
        now_s=time.time(),
        n_fresh=n_fresh,
        timeout=timeout,
    )
    if write:
        _write_json(output_path, prereg)

    if transcripts_dir.exists():
        shutil.rmtree(transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[exp4009] preregistered {len(tasks)} clean new tasks / "
        f"{sum(len(committed_entries_by_task[t]) for t in tasks)} entries; "
        f"k={n_fresh}, iters<={iters}, timeout={timeout}s, workers={workers}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    pool_exhausted = True
    if workers <= 1:
        for task in tasks:
            record = _chain_one(
                task, committed_entries_by_task, transcripts_dir, n_fresh, iters, timeout
            )
            records.append(record)
            _print_record(record, n_fresh)
            if (
                agreement_events_so_far(records, committed_entries_by_task)
                >= AGREEMENT_EVENT_TARGET
            ):
                pool_exhausted = False
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start_idx in range(0, len(tasks), workers):
                batch = tasks[start_idx : start_idx + workers]
                futures = executor.map(
                    lambda task: _chain_one(
                        task,
                        committed_entries_by_task,
                        transcripts_dir,
                        n_fresh,
                        iters,
                        timeout,
                    ),
                    batch,
                )
                for record in futures:
                    records.append(record)
                    _print_record(record, n_fresh)
                if (
                    agreement_events_so_far(records, committed_entries_by_task)
                    >= AGREEMENT_EVENT_TARGET
                ):
                    pool_exhausted = False
                    break

    challenges = load_json(challenges_path) if challenges_path.exists() else {}
    solutions = load_json(solutions_path) if solutions_path.exists() else {}
    audit = v2.audit_transcripts(v2.transcript_paths(transcripts_dir))
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
        pool_exhausted=pool_exhausted,
    )
    if write:
        _write_json(output_path, artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(
        f"   floor={artifact['execution_floor_met']} agreement="
        f"{artifact['n_gold_given_agreement']}/{artifact['n_agreement_events']} "
        f"precision={artifact['agreement_precision']} "
        f"fresh_base={artifact['fresh_arm_base_rate']} "
        f"delta={artifact['precision_vs_fresharm_base']}",
        flush=True,
    )
    print(
        f"   target={artifact['powered_target_reached']} "
        f"stop={artifact['draw_stop_reason']} leak_clean={artifact['leak_clean']}",
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

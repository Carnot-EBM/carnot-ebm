"""Exp 4316 budget-aware cascade deployment for the ARC energy verifier.

Spec refs: REQ-VERIFY-4316, SCENARIO-VERIFY-4316.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import resolve_cached_gguf
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244
from carnot.reporting import verifier_efficiency_harden_strong_judge_4294 as exp4294
from carnot.reporting import verifier_efficiency_vs_llm_judge_4284 as exp4284


RANDOM_SEED = 4316
OUTPUT_REL = Path("results/experiment_4316_efficiency_cascade_router_deploy.json")
CHECKPOINT_REL = Path("results/experiment_4316_efficiency_cascade_router_deploy_checkpoint.json")
SPEC_REFS = ["REQ-VERIFY-4316", "SCENARIO-VERIFY-4316"]
INFERENCE_SUBSTRATE = "live_strong_llm_judge_budget_aware_arc_energy_cascade"
PROMPT_VERSION = exp4294.PROMPT_VERSION
QWEN_JUDGE_ID = exp4294.QWEN_JUDGE_ID
GEMMA_JUDGE_ID = exp4294.GEMMA_JUDGE_ID
REQUESTED_JUDGE_IDS = (QWEN_JUDGE_ID, GEMMA_JUDGE_ID)
MIN_TASKS = 40
MIN_EVAL_TASKS = 40
LIVE_SELECTION_TASKS = 52
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_WINDOW_S = 6900.0

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A cascade Pareto-win (most of the judge's accuracy "
        "at a fraction of the cost), a parity-at-lower-cost, an 'always-energy "
        "already dominates' (the cascade is unnecessary -- a clean finding), "
        "and an honest blocked_judge_models_not_cached / blocked_window_exceeded "
        "are ALL COMPLETE."
    ),
    "cascade_dominates_controls": (
        "BARE bool: the capstone reads this (gated-fields-must-be-bare); true "
        "iff the cascade reaches always-judge accuracy (within CI) at a cost "
        "strictly between always-energy and always-judge AND is Pareto-non-"
        "dominated -- the deployed §5 efficiency operating point."
    ),
    "accuracy_cascade": (
        "BARE float: the cascade policy's selection accuracy -- the deployed "
        "accuracy (target: >= always-judge within CI)."
    ),
    "accuracy_always_energy": (
        "BARE float: the cheap always-energy baseline accuracy (compare to "
        "exp4303's 0.8)."
    ),
    "accuracy_always_judge": (
        "BARE float: the well-prompted always-judge accuracy (the expensive "
        "ceiling; compare to exp4303's 0.5)."
    ),
    "cost_ratio_cascade": (
        "BARE float: cascade cost / always-judge cost (wall-clock + FLOPs-proxy "
        "+ $/1k) -- the deployed efficiency multiplier (the cascade buys "
        "accuracy by escalating only the hard cases)."
    ),
    "escalation_rate": (
        "BARE float: fraction of tasks escalated to the judge -- the cascade's "
        "cost knob (low escalation + near-judge accuracy is the win)."
    ),
    "pareto_curve": (
        "Accuracy-vs-cost for always-energy, cascade, always-judge (and "
        "intermediate thresholds) -- the honest Pareto frontier (Budget-aware "
        "Discriminative Verification 2510.14913)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the energy verifier on cross-family/ARC-GEN/FoVer "
        "selection is oracle-distinct (NOT the executable oracle); keeps the "
        "efficiency measurement non-circular (unlike the retired code-efficiency)."
    ),
    "preconditions_checked": (
        "Records the judge GGUF caches + corpus load + TRM-stand-down verified; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the selection + bootstrap + threshold tuning.",
    "reproducibility_checksum": (
        "Hash of the corpora + the policies' outputs + the cascade threshold + "
        "the cost accounting; lets a third party re-run."
    ),
    "model_specs": (
        "The judge GGUF ids + the strong prompt + the energy verifier + the "
        "cascade threshold + the cost-accounting method + the corpora; required "
        "methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "cascade_dominates_controls",
    "accuracy_cascade",
    "accuracy_always_energy",
    "accuracy_always_judge",
    "cost_ratio_cascade",
    "escalation_rate",
    "pareto_curve",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "accuracy_ci95s",
    "cost_ci95s",
    "cost_always_energy",
    "cost_always_judge",
    "cost_cascade",
    "field_principles",
    "spec_refs",
    "inference_substrate",
    "duration_s",
    "adversarial_verify",
)


class BlockedRun(RuntimeError):
    """Expected precondition or window failure that still writes an artifact."""

    def __init__(self, reason: str, preconditions: list[dict[str, Any]] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.preconditions = preconditions or []


def _round_metric(value: float) -> float:
    return exp4284._round_metric(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    return exp4284._safe_float(value, default)


def _bare_float(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _checksum(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def default_judge_specs_provider() -> list[dict[str, Any]]:  # pragma: no cover - local cache dependent.
    specs: list[dict[str, Any]] = []
    for hf_id in REQUESTED_JUDGE_IDS:
        model_path = resolve_cached_gguf(hf_id)
        path = Path(str(model_path)) if model_path else Path("")
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            specs.append(
                {
                    "name": hf_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
                    "hf_id": hf_id,
                    "model_path": str(path),
                    "active_params_b": exp4284._model_active_params_b({"hf_id": hf_id}),
                }
            )
    return specs


def default_llama_import_checker() -> bool:  # pragma: no cover - environment dependent.
    try:
        __import__("llama_cpp")
    except Exception:
        return False
    return True


def default_trm_stand_down_checker(repo_root: Path) -> tuple[bool, str]:  # pragma: no cover - process dependent.
    proc = subprocess.run(
        ["pgrep", "-af", "trm|TRM|src/nn/train.py"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if str(repo_root / "results" / "trm_runs") in line]
    if lines:
        return False, "; ".join(lines[:3])
    return True, "no active TRM training process writing this repo's results/trm_runs"


def _normalize_judge_specs(
    raw_specs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    by_id = {
        str(spec.get("hf_id")): dict(spec)
        for spec in raw_specs or []
        if isinstance(spec, dict) and spec.get("hf_id")
    }
    available: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    skipped: list[str] = []
    for hf_id in REQUESTED_JUDGE_IDS:
        spec = by_id.get(hf_id)
        model_path = Path(str(spec.get("model_path") if spec else ""))
        if spec and model_path.exists() and model_path.is_file() and model_path.stat().st_size > 0:
            normalized = dict(spec)
            normalized.setdefault("name", hf_id.rsplit("/", 1)[-1].removesuffix("-GGUF"))
            normalized.setdefault("active_params_b", exp4284._model_active_params_b(normalized))
            available.append(normalized)
            checks.append(
                {
                    "resource": f"cached_judge_gguf:{hf_id}",
                    "available": True,
                    "detail": str(model_path),
                }
            )
        else:
            checks.append(
                {
                    "resource": f"cached_judge_gguf:{hf_id}",
                    "available": False,
                    "detail": str(model_path) if spec else "not resolved",
                }
            )
            skipped.append(hf_id)
    return available, checks, skipped


def _empty_checkpoint() -> dict[str, Any]:
    return {"version": 1, "experiment_id": 4316, "judges": {}}


def _read_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_checkpoint()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("judges"), dict):  # pragma: no cover
        return _empty_checkpoint()
    return payload


def _write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_rows(checkpoint: dict[str, Any], judge_id: str) -> list[dict[str, Any]]:
    judge_block = checkpoint.setdefault("judges", {}).setdefault(judge_id, {"selections": []})
    rows = judge_block.setdefault("selections", [])
    return rows if isinstance(rows, list) else []


def _selection_row(
    case: exp4284.SelectionCase,
    *,
    chosen_index: int,
    judge_id: str,
    record: dict[str, Any],
    checkpoint_resumed: bool,
) -> dict[str, Any]:
    if chosen_index < 0 or chosen_index >= len(case.finalists):  # pragma: no cover
        chosen_index = 0
    chosen = case.finalists[chosen_index]
    energy_index = next(
        index for index, candidate in enumerate(case.finalists) if candidate.candidate_id == case.energy_candidate_id
    )
    return {
        "task_id": case.task_id,
        "family_id": case.family_id,
        "fold": case.fold,
        "candidate_count": len(case.finalists),
        "all_candidate_count": len(case.all_candidates),
        "energy_candidate_id": case.energy_candidate_id,
        "energy_finalist_index": energy_index,
        "energy_correct": bool(case.energy_correct),
        "judge_id": judge_id,
        "judge_chosen_index": chosen_index,
        "judge_candidate_id": chosen.candidate_id,
        "judge_correct": bool(chosen.correct),
        "judge_cost": record,
        "checkpoint_resumed": checkpoint_resumed,
    }


def run_checkpointed_strong_llm_judge(
    cases: list[exp4284.SelectionCase],
    judge_client: Any,
    *,
    judge_id: str,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    deadline_monotonic: float | None,
    min_completed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    rows = _checkpoint_rows(checkpoint, judge_id)
    by_task = {str(row.get("task_id")): dict(row) for row in rows if isinstance(row, dict) and row.get("task_id")}
    selections: list[dict[str, Any]] = []
    resumed_any = False
    for case_index, case in enumerate(cases, start=1):
        cached = by_task.get(case.task_id)
        if cached:  # pragma: no cover - exercised by resumed live runs, not fresh unit fixtures.
            cached["checkpoint_resumed"] = True
            selections.append(cached)
            resumed_any = True
            continue
        if deadline_monotonic is not None and time.perf_counter() >= deadline_monotonic:  # pragma: no cover
            if len(selections) >= int(min_completed):
                break
            raise BlockedRun("blocked_window_exceeded")
        if case_index == 1 or case_index % 5 == 0 or case_index == len(cases):
            print(f"[exp4316] {judge_id} judging {case_index}/{len(cases)}", flush=True)
        candidate_texts = [
            exp4284._candidate_prompt_text(candidate, index) for index, candidate in enumerate(case.finalists)
        ]
        chosen_index = int(judge_client.judge(exp4284._problem_text(case), candidate_texts))
        records = getattr(judge_client, "records", [])
        record = dict(records[-1]) if records else {"chosen_index": chosen_index}
        record.setdefault("latency_s", 0.0)
        record.setdefault("prompt_tokens", 0)
        record.setdefault("completion_tokens", 0)
        record.setdefault("total_tokens", 0)
        record.setdefault("raw_output", "")
        record.setdefault("parse_status", "record_missing")
        row = _selection_row(
            case,
            chosen_index=chosen_index,
            judge_id=judge_id,
            record=record,
            checkpoint_resumed=False,
        )
        rows.append(row)
        _write_checkpoint(checkpoint_path, checkpoint)
        selections.append(row)
    costs = [row["judge_cost"] for row in selections]
    return selections, {
        "total_wall_clock_s": round(sum(_safe_float(cost.get("latency_s")) for cost in costs), 6),
        "total_tokens": int(sum(_safe_float(cost.get("total_tokens")) for cost in costs)),
        "prompt_tokens": int(sum(_safe_float(cost.get("prompt_tokens")) for cost in costs)),
        "completion_tokens": int(sum(_safe_float(cost.get("completion_tokens")) for cost in costs)),
    }, resumed_any


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):  # pragma: no cover
        raise BlockedRun("blocked_malformed_json_artifact")
    return payload


def _load_margin_map(root: Path) -> dict[str, float]:
    path = root / exp4284.CROSS_FAMILY_REL
    payload = _read_json_object(path)
    rows = payload.get("task_rows")
    if not isinstance(rows, list):  # pragma: no cover
        return {}
    margins: dict[str, float] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("task_id"):
            margins[str(row["task_id"])] = abs(_safe_float(row.get("set_encoder_score_margin_vs_vote")))
    return margins


def _fallback_energy_margin(case: exp4284.SelectionCase, model_payload: dict[str, Any]) -> float:  # pragma: no cover
    model = model_payload.get("model", model_payload)
    scores = exp4244._score_with_payload(model, exp4284._grown_rows(case.all_candidates))
    if len(scores) < 2:
        return 0.0
    ordered = sorted((float(score) for score in scores.values()), reverse=True)
    return max(0.0, ordered[0] - ordered[1])


def _energy_margin(case: exp4284.SelectionCase, margin_by_task: dict[str, float], model_payload: dict[str, Any]) -> float:
    if case.task_id in margin_by_task:
        return _round_metric(margin_by_task[case.task_id])
    return _round_metric(_fallback_energy_margin(case, model_payload))  # pragma: no cover


def _split_tune_eval(
    cases: list[exp4284.SelectionCase],
    *,
    min_eval_tasks: int,
) -> tuple[list[exp4284.SelectionCase], list[exp4284.SelectionCase]]:
    if len(cases) <= int(min_eval_tasks):  # pragma: no cover
        raise BlockedRun("blocked_window_exceeded")
    tune_n = max(1, len(cases) - int(min_eval_tasks))
    return cases[:tune_n], cases[tune_n:]


def _rate(values: list[bool]) -> float:
    return _round_metric(exp4284._rate(values))


def _ci95(samples: list[float]) -> list[float]:
    if not samples:  # pragma: no cover
        return [0.0, 0.0]
    samples = sorted(float(sample) for sample in samples)
    if len(samples) == 1:  # pragma: no cover
        point = _round_metric(samples[0])
        return [point, point]
    return [
        _round_metric(samples[int(0.025 * (len(samples) - 1))]),
        _round_metric(samples[int(0.975 * (len(samples) - 1))]),
    ]


def _judge_task_cost(row: dict[str, Any], judge_spec: dict[str, Any]) -> dict[str, float]:
    cost = row.get("judge_cost") if isinstance(row.get("judge_cost"), dict) else {}
    tokens = int(_safe_float(cost.get("total_tokens")))
    active_params_b = exp4284._model_active_params_b(judge_spec)
    return {
        "wall_clock_s": _safe_float(cost.get("latency_s")),
        "flops_proxy": float(2.0 * active_params_b * 1_000_000_000.0 * tokens),
        "estimated_dollars": float(tokens) / 1000.0 * exp4284.LLM_DOLLARS_PER_1K_TOKENS,
        "tokens": float(tokens),
    }


def _energy_task_costs(energy_cost: dict[str, Any], n_tasks: int) -> list[dict[str, float]]:
    n = max(1, int(n_tasks))
    return [
        {
            "wall_clock_s": float(energy_cost["total_wall_clock_s"]) / n,
            "flops_proxy": float(energy_cost["flops_proxy"]) / n,
            "estimated_dollars": float(energy_cost["estimated_dollars_per_1k_selections"]) / n,
            "tokens": 0.0,
        }
        for _ in range(n)
    ]


def _sum_costs(costs: list[dict[str, float]]) -> dict[str, float]:
    n = max(1, len(costs))
    total_dollars = sum(float(cost["estimated_dollars"]) for cost in costs)
    return {
        "total_wall_clock_s": _round_metric(sum(float(cost["wall_clock_s"]) for cost in costs)),
        "flops_proxy": float(sum(float(cost["flops_proxy"]) for cost in costs)),
        "total_tokens": int(sum(float(cost["tokens"]) for cost in costs)),
        "total_estimated_dollars": total_dollars,
        "estimated_dollars_per_1k_selections": total_dollars / n * 1000.0,
    }


def _combine_costs(*costs: dict[str, float]) -> dict[str, float]:
    return {
        "wall_clock_s": sum(float(cost["wall_clock_s"]) for cost in costs),
        "flops_proxy": sum(float(cost["flops_proxy"]) for cost in costs),
        "estimated_dollars": sum(float(cost["estimated_dollars"]) for cost in costs),
        "tokens": sum(float(cost["tokens"]) for cost in costs),
    }


def _decision_rows(
    cases: list[exp4284.SelectionCase],
    selections: list[dict[str, Any]],
    *,
    margin_by_task: dict[str, float],
    model_payload: dict[str, Any],
    energy_cost: dict[str, Any],
    judge_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    by_task = {str(row["task_id"]): row for row in selections}
    energy_costs = _energy_task_costs(energy_cost, len(cases))
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        judge_row = by_task.get(case.task_id)
        if judge_row is None:  # pragma: no cover
            continue
        judge_cost = _judge_task_cost(judge_row, judge_spec)
        energy_cost_one = energy_costs[index]
        rows.append(
            {
                "task_id": case.task_id,
                "family_id": case.family_id,
                "fold": case.fold,
                "energy_margin": _energy_margin(case, margin_by_task, model_payload),
                "energy_correct": bool(case.energy_correct),
                "judge_correct": bool(judge_row["judge_correct"]),
                "energy_candidate_id": case.energy_candidate_id,
                "judge_candidate_id": judge_row["judge_candidate_id"],
                "judge_chosen_index": int(judge_row["judge_chosen_index"]),
                "energy_cost": energy_cost_one,
                "judge_cost": judge_cost,
            }
        )
    return rows


def _threshold_grid(rows: list[dict[str, Any]]) -> list[float]:
    margins = sorted({_round_metric(float(row["energy_margin"])) for row in rows})
    if not margins:  # pragma: no cover
        return [0.0]
    below = max(0.0, margins[0] - 1e-9)
    above = margins[-1] + 1e-9
    return [_round_metric(value) for value in [below, *margins, above]]


def _evaluate_policy(rows: list[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    hits: list[bool] = []
    escalated: list[bool] = []
    costs: list[dict[str, float]] = []
    for row in rows:
        should_escalate = threshold is not None and float(row["energy_margin"]) <= float(threshold)
        escalated.append(should_escalate)
        hits.append(bool(row["judge_correct"] if should_escalate else row["energy_correct"]))
        costs.append(
            _combine_costs(row["energy_cost"], row["judge_cost"]) if should_escalate else dict(row["energy_cost"])
        )
    return {
        "accuracy": _rate(hits),
        "escalation_rate": _round_metric(sum(escalated) / float(len(escalated))) if escalated else 0.0,
        "cost": _sum_costs(costs),
        "hits": hits,
        "escalated": escalated,
    }


def _always_judge_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [dict(row["judge_cost"]) for row in rows]
    hits = [bool(row["judge_correct"]) for row in rows]
    return {
        "accuracy": _rate(hits),
        "escalation_rate": 1.0,
        "cost": _sum_costs(costs),
        "hits": hits,
        "escalated": [True for _ in rows],
    }


def _tune_threshold(rows: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    points = []
    for threshold in _threshold_grid(rows):
        policy = _evaluate_policy(rows, threshold)
        points.append(
            {
                "threshold": threshold,
                "accuracy": policy["accuracy"],
                "escalation_rate": policy["escalation_rate"],
                "estimated_dollars_per_1k_selections": policy["cost"]["estimated_dollars_per_1k_selections"],
            }
        )
    best = max(points, key=lambda point: (point["accuracy"], -point["estimated_dollars_per_1k_selections"]))
    return float(best["threshold"]), points


def _point(policy: str, accuracy: float, cost: dict[str, Any], *, threshold: float | None, escalation_rate: float) -> dict[str, Any]:
    return {
        "policy": policy,
        "threshold": threshold,
        "accuracy": float(accuracy),
        "estimated_dollars_per_1k_selections": float(cost["estimated_dollars_per_1k_selections"]),
        "total_wall_clock_s": float(cost["total_wall_clock_s"]),
        "flops_proxy": float(cost["flops_proxy"]),
        "escalation_rate": float(escalation_rate),
    }


def _mark_pareto(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        dominated = False
        for other_index, other in enumerate(points):
            if index == other_index:
                continue
            other_better_or_equal = (
                float(other["accuracy"]) >= float(point["accuracy"])
                and float(other["estimated_dollars_per_1k_selections"])
                <= float(point["estimated_dollars_per_1k_selections"])
            )
            other_strict = (
                float(other["accuracy"]) > float(point["accuracy"])
                or float(other["estimated_dollars_per_1k_selections"])
                < float(point["estimated_dollars_per_1k_selections"])
            )
            if other_better_or_equal and other_strict:
                dominated = True
                break
        marked.append({**point, "pareto_non_dominated": not dominated})
    return marked


def _pareto_curve(
    rows: list[dict[str, Any]],
    *,
    selected_threshold: float,
    always_energy: dict[str, Any],
    always_judge: dict[str, Any],
    cascade: dict[str, Any],
) -> dict[str, Any]:
    points = [
        _point(
            "always_energy",
            always_energy["accuracy"],
            always_energy["cost"],
            threshold=None,
            escalation_rate=0.0,
        )
    ]
    for threshold in _threshold_grid(rows):
        policy = _evaluate_policy(rows, threshold)
        points.append(
            _point(
                "cascade_threshold",
                policy["accuracy"],
                policy["cost"],
                threshold=threshold,
                escalation_rate=policy["escalation_rate"],
            )
        )
    points.append(
        _point(
            "cascade",
            cascade["accuracy"],
            cascade["cost"],
            threshold=selected_threshold,
            escalation_rate=cascade["escalation_rate"],
        )
    )
    points.append(
        _point(
            "always_judge",
            always_judge["accuracy"],
            always_judge["cost"],
            threshold=None,
            escalation_rate=1.0,
        )
    )
    marked = _mark_pareto(points)
    return {
        "x_axis": "estimated_dollars_per_1k_selections",
        "y_axis": "selection_accuracy",
        "threshold_rule": "escalate_to_judge_when_energy_margin <= threshold",
        "selected_threshold": float(selected_threshold),
        "points": marked,
    }


def _bootstrap_cis(
    rows: list[dict[str, Any]],
    *,
    selected_threshold: float,
    random_seed: int,
    resamples: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    rng = random.Random(random_seed)
    n = len(rows)
    acc_energy: list[float] = []
    acc_judge: list[float] = []
    acc_cascade: list[float] = []
    acc_cascade_minus_judge: list[float] = []
    acc_cascade_minus_energy: list[float] = []
    cost_energy: list[float] = []
    cost_judge: list[float] = []
    cost_cascade: list[float] = []
    cost_ratio_cascade: list[float] = []
    for _ in range(int(resamples)):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        energy = _evaluate_policy(sample, None)
        judge = _always_judge_policy(sample)
        cascade = _evaluate_policy(sample, selected_threshold)
        acc_energy.append(float(energy["accuracy"]))
        acc_judge.append(float(judge["accuracy"]))
        acc_cascade.append(float(cascade["accuracy"]))
        acc_cascade_minus_judge.append(float(cascade["accuracy"]) - float(judge["accuracy"]))
        acc_cascade_minus_energy.append(float(cascade["accuracy"]) - float(energy["accuracy"]))
        energy_dollars = float(energy["cost"]["estimated_dollars_per_1k_selections"])
        judge_dollars = max(float(judge["cost"]["estimated_dollars_per_1k_selections"]), 1e-18)
        cascade_dollars = float(cascade["cost"]["estimated_dollars_per_1k_selections"])
        cost_energy.append(energy_dollars)
        cost_judge.append(judge_dollars)
        cost_cascade.append(cascade_dollars)
        cost_ratio_cascade.append(cascade_dollars / judge_dollars)
    return {
        "always_energy": _ci95(acc_energy),
        "always_judge": _ci95(acc_judge),
        "cascade": _ci95(acc_cascade),
        "cascade_minus_judge": _ci95(acc_cascade_minus_judge),
        "cascade_minus_energy": _ci95(acc_cascade_minus_energy),
    }, {
        "always_energy_dollars_per_1k": _ci95(cost_energy),
        "always_judge_dollars_per_1k": _ci95(cost_judge),
        "cascade_dollars_per_1k": _ci95(cost_cascade),
        "cost_ratio_cascade": _ci95(cost_ratio_cascade),
    }


def _selected_cascade_pareto_non_dominated(curve: dict[str, Any]) -> bool:
    for point in curve.get("points", []):
        if isinstance(point, dict) and point.get("policy") == "cascade":
            return bool(point.get("pareto_non_dominated"))
    return False  # pragma: no cover


def _blocked_artifact(
    reason: str,
    preconditions: list[dict[str, Any]],
    *,
    random_seed: int,
    duration_s: float,
    checkpoint_path: Path,
) -> dict[str, Any]:
    zero_cost = {
        "total_wall_clock_s": 0.0,
        "flops_proxy": 0.0,
        "total_tokens": 0,
        "total_estimated_dollars": 0.0,
        "estimated_dollars_per_1k_selections": 0.0,
    }
    return {
        "honest_verdict": reason,
        "cascade_dominates_controls": False,
        "accuracy_cascade": 0.0,
        "accuracy_always_energy": 0.0,
        "accuracy_always_judge": 0.0,
        "cost_ratio_cascade": 0.0,
        "escalation_rate": 0.0,
        "pareto_curve": {
            "x_axis": "estimated_dollars_per_1k_selections",
            "y_axis": "selection_accuracy",
            "threshold_rule": "not computed because the run blocked before judge measurement",
            "selected_threshold": None,
            "points": [],
        },
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _checksum(
            {"blocked": reason, "preconditions": preconditions, "checkpoint_path": str(checkpoint_path)}
        ),
        "model_specs": {
            "status": "blocked",
            "blocked_reason": reason,
            "requested_judge_ggufs": list(REQUESTED_JUDGE_IDS),
            "strong_prompt": {"version": PROMPT_VERSION, "summary": exp4294.STRONG_PROMPT_SUMMARY},
        },
        "accuracy_ci95s": {
            "always_energy": [0.0, 0.0],
            "always_judge": [0.0, 0.0],
            "cascade": [0.0, 0.0],
            "cascade_minus_judge": [0.0, 0.0],
            "cascade_minus_energy": [0.0, 0.0],
        },
        "cost_ci95s": {
            "always_energy_dollars_per_1k": [0.0, 0.0],
            "always_judge_dollars_per_1k": [0.0, 0.0],
            "cascade_dollars_per_1k": [0.0, 0.0],
            "cost_ratio_cascade": [0.0, 0.0],
        },
        "cost_always_energy": dict(zero_cost),
        "cost_always_judge": dict(zero_cost),
        "cost_cascade": dict(zero_cost),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": "precondition_block",
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "not_run", "reason": reason},
        "acceptance_gate": reason in {"blocked_judge_models_not_cached", "blocked_window_exceeded"},
        "selection_task_n": 0,
        "tuning_task_n": 0,
        "checkpoint_path": str(checkpoint_path),
        "judge_metrics": [],
        "per_task": [],
    }


def _complete_artifact(
    *,
    all_cases: list[exp4284.SelectionCase],
    tune_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    selected_threshold: float,
    tuning_curve: list[dict[str, Any]],
    judge_result: dict[str, Any],
    energy_cost_eval: dict[str, Any],
    checksums: dict[str, str],
    model_path: Path,
    build: dict[str, Any],
    preconditions: list[dict[str, Any]],
    skipped_judge_ids: list[str],
    random_seed: int,
    bootstrap_resamples: int,
    duration_s: float,
    checkpoint_path: Path,
) -> dict[str, Any]:
    always_energy = _evaluate_policy(eval_rows, None)
    always_judge = _always_judge_policy(eval_rows)
    cascade = _evaluate_policy(eval_rows, selected_threshold)
    curve = _pareto_curve(
        eval_rows,
        selected_threshold=selected_threshold,
        always_energy=always_energy,
        always_judge=always_judge,
        cascade=cascade,
    )
    accuracy_ci95s, cost_ci95s = _bootstrap_cis(
        eval_rows,
        selected_threshold=selected_threshold,
        random_seed=random_seed,
        resamples=bootstrap_resamples,
    )
    judge_cost_per_1k = max(float(always_judge["cost"]["estimated_dollars_per_1k_selections"]), 1e-18)
    cascade_cost_per_1k = float(cascade["cost"]["estimated_dollars_per_1k_selections"])
    energy_cost_per_1k = float(always_energy["cost"]["estimated_dollars_per_1k_selections"])
    cost_ratio_cascade = _round_metric(cascade_cost_per_1k / judge_cost_per_1k)
    parity_with_judge = bool(accuracy_ci95s["cascade_minus_judge"][1] >= 0.0)
    strict_cost_between = bool(energy_cost_per_1k < cascade_cost_per_1k < judge_cost_per_1k)
    cascade_non_dominated = _selected_cascade_pareto_non_dominated(curve)
    cascade_dominates_controls = bool(parity_with_judge and strict_cost_between and cascade_non_dominated)
    if cascade_dominates_controls:
        verdict = "complete: cascade_pareto_win"
    elif (  # pragma: no cover
        float(always_energy["accuracy"]) >= float(cascade["accuracy"])
        and energy_cost_per_1k <= cascade_cost_per_1k
    ):
        verdict = "complete: always_energy_already_dominates"
    elif parity_with_judge and cascade_cost_per_1k < judge_cost_per_1k:  # pragma: no cover
        verdict = "complete: cascade_parity_at_lower_cost"
    elif accuracy_ci95s["cascade_minus_judge"][1] < 0.0:  # pragma: no cover
        verdict = "complete: judge_accuracy_ceiling_requires_more_budget"
    else:  # pragma: no cover
        verdict = "complete: no_pareto_cascade_win"

    per_task = [
        {
            "task_id": row["task_id"],
            "family_id": row["family_id"],
            "fold": row["fold"],
            "energy_margin": row["energy_margin"],
            "energy_correct": row["energy_correct"],
            "judge_correct": row["judge_correct"],
            "cascade_escalated": bool(row["energy_margin"] <= selected_threshold),
            "cascade_correct": bool(row["judge_correct"] if row["energy_margin"] <= selected_threshold else row["energy_correct"]),
            "energy_candidate_id": row["energy_candidate_id"],
            "judge_candidate_id": row["judge_candidate_id"],
            "judge_chosen_index": row["judge_chosen_index"],
        }
        for row in eval_rows
    ]
    compact_judge_metric = {
        "judge_id": judge_result["judge_id"],
        "accuracy": _rate([bool(row["judge_correct"]) for row in eval_rows]),
        "selection_task_n": len(eval_rows),
        "checkpoint_resumed": any(bool(row.get("checkpoint_resumed")) for row in judge_result["selections"]),
        "total_wall_clock_s": sum(float(row["judge_cost"]["wall_clock_s"]) for row in eval_rows),
        "total_tokens": int(sum(float(row["judge_cost"]["tokens"]) for row in eval_rows)),
        "estimated_dollars_per_1k_selections": always_judge["cost"]["estimated_dollars_per_1k_selections"],
        "flops_proxy": always_judge["cost"]["flops_proxy"],
    }
    cost_accounting_method = (
        "Policy costs are estimated per 1k selections from per-task wall-clock, "
        "FLOPs proxy, and dollars: energy uses Exp 4284 set-encoder forward-pass "
        "constants; judge uses tokens * active_params_b with the Exp 4284 local "
        "GGUF dollar proxy; cascade pays energy for every task plus judge cost "
        "only for rows whose energy margin is <= the tuned threshold."
    )
    checksum_payload = {
        "checksums": checksums,
        "selected_threshold": selected_threshold,
        "per_task": per_task,
        "costs": {
            "always_energy": always_energy["cost"],
            "always_judge": always_judge["cost"],
            "cascade": cascade["cost"],
        },
        "curve": curve,
        "random_seed": int(random_seed),
    }
    return {
        "honest_verdict": (
            f"{verdict}_acc_cascade_{float(cascade['accuracy']):.4f}_"
            f"cost_ratio_{cost_ratio_cascade:.10f}"
        ),
        "cascade_dominates_controls": cascade_dominates_controls,
        "accuracy_cascade": float(cascade["accuracy"]),
        "accuracy_always_energy": float(always_energy["accuracy"]),
        "accuracy_always_judge": float(always_judge["accuracy"]),
        "cost_ratio_cascade": float(cost_ratio_cascade),
        "escalation_rate": float(cascade["escalation_rate"]),
        "pareto_curve": curve,
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _checksum(checksum_payload),
        "model_specs": {
            "requested_judge_ggufs": list(REQUESTED_JUDGE_IDS),
            "available_judge_ggufs": [judge_result["judge_id"]],
            "skipped_judge_ggufs": list(skipped_judge_ids),
            "judge_gguf": dict(judge_result["judge_spec"]),
            "strong_prompt": {
                "version": PROMPT_VERSION,
                "summary": exp4294.STRONG_PROMPT_SUMMARY,
                "template_sha256": _checksum(
                    {"prompt": exp4294._build_strong_prompt("{problem}", ["Candidate 0: ..."])}
                ),
                "final_answer_contract": "Final answer: <index>",
            },
            "energy_verifier": {
                "path": str(model_path),
                "verifier_is_oracle": False,
                "model_specs": build.get("model_specs", {}),
            },
            "cascade_threshold": float(selected_threshold),
            "threshold_rule": "escalate_to_judge_when_energy_margin <= threshold",
            "cost_accounting_method": cost_accounting_method,
            "corpora": [
                {
                    "corpus_id": "arc_cross_family_existing_pool",
                    "source_artifacts": [
                        str(exp4284.POOL_REL),
                        str(exp4284.MANIFEST_REL),
                        str(exp4284.CROSS_FAMILY_REL),
                    ],
                    "total_judged_task_n": len(all_cases),
                    "tuning_task_n": len(tune_rows),
                    "evaluation_task_n": len(eval_rows),
                    "partial_one_corpus_complete": len(eval_rows) >= MIN_EVAL_TASKS,
                }
            ],
        },
        "accuracy_ci95s": accuracy_ci95s,
        "cost_ci95s": cost_ci95s,
        "cost_always_energy": always_energy["cost"],
        "cost_always_judge": always_judge["cost"],
        "cost_cascade": cascade["cost"],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(
            max(float(duration_s), float(judge_result["judge_cost"]["total_wall_clock_s"])),
            6,
        ),
        "adversarial_verify": {"status": "pending"},
        "acceptance_gate": True,
        "selection_task_n": len(eval_rows),
        "tuning_task_n": len(tune_rows),
        "total_judge_task_n": len(all_cases),
        "bootstrap_resamples": int(bootstrap_resamples),
        "cascade_threshold": float(selected_threshold),
        "threshold_tuning": {
            "tuning_task_n": len(tune_rows),
            "objective": "maximize held-out tuning accuracy, tie-break by lower estimated dollars per 1k selections",
            "curve": tuning_curve,
        },
        "checkpoint_path": str(checkpoint_path),
        "judge_metrics": [compact_judge_metric],
        "per_task": per_task,
        "precondition_checksums": checksums,
        "cost_accounting": {
            "method": cost_accounting_method,
            "energy_forward_cost_eval": energy_cost_eval,
            "constants": {
                "energy_dollars_per_tflop": exp4284.ENERGY_DOLLARS_PER_TFLOP,
                "llm_dollars_per_1k_tokens": exp4284.LLM_DOLLARS_PER_1K_TOKENS,
            },
        },
    }


def _run_adversarial_verify(repo_root: Path, artifact_path: Path) -> dict[str, Any]:  # pragma: no cover
    script = repo_root / "scripts" / "adversarial_verify.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--json", str(artifact_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = proc.returncode
    return payload


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:  # pragma: no cover
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (verdict.startswith("complete:") or verdict.startswith("blocked_")):  # pragma: no cover
        raise ValueError("honest_verdict must have a terminal prefix")
    if type(artifact["cascade_dominates_controls"]) is not bool:  # pragma: no cover
        raise ValueError("cascade_dominates_controls must be a bare bool")
    for field in (
        "accuracy_cascade",
        "accuracy_always_energy",
        "accuracy_always_judge",
        "cost_ratio_cascade",
        "escalation_rate",
    ):
        if not _bare_float(artifact[field]):  # pragma: no cover
            raise ValueError(f"{field} must be a bare float")
    if artifact["verifier_is_oracle"] is not False:  # pragma: no cover
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:  # pragma: no cover
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["pareto_curve"], dict) or not isinstance(artifact["pareto_curve"].get("points"), list):  # pragma: no cover
        raise ValueError("pareto_curve must be an object with points")
    for field in ("preconditions_checked",):
        if not isinstance(artifact[field], list):  # pragma: no cover
            raise ValueError(f"{field} must be a list")
    for field in (
        "model_specs",
        "adversarial_verify",
        "accuracy_ci95s",
        "cost_ci95s",
        "cost_always_energy",
        "cost_always_judge",
        "cost_cascade",
    ):
        if not isinstance(artifact[field], dict):  # pragma: no cover
            raise ValueError(f"{field} must be an object")
    checksum = artifact["reproducibility_checksum"]
    if not isinstance(checksum, str) or len(checksum) != 64:  # pragma: no cover
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if artifact["field_principles"] != FIELD_PRINCIPLES:  # pragma: no cover
        raise ValueError("field_principles do not match REQ-VERIFY-4316")
    if artifact["spec_refs"] != SPEC_REFS:  # pragma: no cover
        raise ValueError("spec_refs do not match REQ-VERIFY-4316")


def _write_artifact(repo_root: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    validate_artifact(artifact)
    output = repo_root / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def run(
    repo_root: Path | str = Path("."),
    *,
    judge_specs_provider: Callable[[], list[dict[str, Any]] | tuple[dict[str, Any], ...] | None] = (
        default_judge_specs_provider
    ),
    llama_import_checker: Callable[[], bool] = default_llama_import_checker,
    judge_factory: Callable[[dict[str, Any]], Any] = exp4294.StrongPromptCostMeteredLlmJudge,
    trm_stand_down_checker: Callable[[Path], tuple[bool, str]] = default_trm_stand_down_checker,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    min_tasks: int = MIN_TASKS,
    min_eval_tasks: int = MIN_EVAL_TASKS,
    max_tasks: int | None = LIVE_SELECTION_TASKS,
    window_s: float = DEFAULT_WINDOW_S,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    checkpoint_path = root / CHECKPOINT_REL
    preconditions: list[dict[str, Any]] = []
    skipped_judge_ids: list[str] = []
    checksums: dict[str, str] = {}

    try:
        available_specs, cache_checks, skipped_judge_ids = _normalize_judge_specs(judge_specs_provider())
        preconditions.extend(cache_checks)
        trm_ok, trm_detail = trm_stand_down_checker(root)
        preconditions.append(
            {"resource": "trm_training_stand_down", "available": bool(trm_ok), "detail": trm_detail}
        )
        if not trm_ok:  # pragma: no cover
            raise BlockedRun("blocked_trm_training_active", preconditions)
        if not available_specs:
            raise BlockedRun("blocked_judge_models_not_cached", preconditions)
        if not llama_import_checker():  # pragma: no cover
            preconditions.append(
                {"resource": "llama_cpp", "available": False, "detail": "import llama_cpp failed"}
            )
            raise BlockedRun("blocked_llama_cpp_unavailable", preconditions)
        preconditions.append({"resource": "llama_cpp", "available": True, "detail": "import OK"})
        cases, checksums, energy_model, energy_model_path, build = exp4284.load_selection_cases(
            root,
            min_tasks=min_tasks,
            max_tasks=max_tasks,
        )
        preconditions.extend(
            [
                {
                    "resource": "arc_cross_family_existing_pool",
                    "available": True,
                    "detail": f"{len(cases)} tasks loaded",
                },
                {
                    "resource": "energy_verifier_cpu",
                    "available": True,
                    "detail": str(energy_model_path),
                },
                {
                    "resource": "trm_runs_not_touched",
                    "available": True,
                    "detail": "Exp 4316 performs no TRM training and writes no results/trm_runs files",
                },
            ]
        )
        margin_by_task = _load_margin_map(root)
        judge_spec = available_specs[0]
        judge_id = str(judge_spec["hf_id"])
        deadline = start + float(window_s) if window_s > 0 else None
        checkpoint = _read_checkpoint(checkpoint_path)
        judge_client = None
        try:
            judge_client = judge_factory(judge_spec)
            selections, judge_cost, checkpoint_resumed = run_checkpointed_strong_llm_judge(
                cases,
                judge_client,
                judge_id=judge_id,
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                deadline_monotonic=deadline,
                min_completed=max(int(min_tasks), int(min_eval_tasks) + 1),
            )
        finally:
            judge_client = None
            gc.collect()
        if len(selections) < max(int(min_tasks), int(min_eval_tasks) + 1):  # pragma: no cover
            raise BlockedRun("blocked_window_exceeded", preconditions)
        measured_cases = cases[: len(selections)]
        tune_cases, eval_cases = _split_tune_eval(measured_cases, min_eval_tasks=min_eval_tasks)
        if len(eval_cases) < int(min_eval_tasks):  # pragma: no cover
            raise BlockedRun("blocked_window_exceeded", preconditions)
        energy_cost_tune = exp4284.measure_energy_forward_cost(tune_cases, energy_model)
        energy_cost_eval = exp4284.measure_energy_forward_cost(eval_cases, energy_model)
        tune_rows = _decision_rows(
            tune_cases,
            selections,
            margin_by_task=margin_by_task,
            model_payload=energy_model,
            energy_cost=energy_cost_tune,
            judge_spec=judge_spec,
        )
        eval_rows = _decision_rows(
            eval_cases,
            selections,
            margin_by_task=margin_by_task,
            model_payload=energy_model,
            energy_cost=energy_cost_eval,
            judge_spec=judge_spec,
        )
        if len(tune_rows) != len(tune_cases) or len(eval_rows) != len(eval_cases):  # pragma: no cover
            raise BlockedRun("blocked_window_exceeded", preconditions)
        selected_threshold, tuning_curve = _tune_threshold(tune_rows)
        artifact = _complete_artifact(
            all_cases=measured_cases,
            tune_rows=tune_rows,
            eval_rows=eval_rows,
            selected_threshold=selected_threshold,
            tuning_curve=tuning_curve,
            judge_result={
                "judge_id": judge_id,
                "judge_spec": dict(judge_spec),
                "selections": selections,
                "judge_cost": judge_cost,
                "checkpoint_resumed": checkpoint_resumed,
            },
            energy_cost_eval=energy_cost_eval,
            checksums=checksums,
            model_path=energy_model_path,
            build=build,
            preconditions=preconditions,
            skipped_judge_ids=sorted(set(skipped_judge_ids + [str(spec["hf_id"]) for spec in available_specs[1:]])),
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
            duration_s=time.perf_counter() - start,
            checkpoint_path=checkpoint_path,
        )
    except exp4284.BlockedRun as blocked:  # pragma: no cover
        artifact = _blocked_artifact(
            blocked.reason,
            preconditions,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
            checkpoint_path=checkpoint_path,
        )
    except BlockedRun as blocked:
        artifact = _blocked_artifact(
            blocked.reason,
            blocked.preconditions or preconditions,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
            checkpoint_path=checkpoint_path,
        )

    output_path = root / OUTPUT_REL
    artifact = _write_artifact(root, artifact)
    if artifact["honest_verdict"].startswith("complete:"):
        if adversarial_runner is not None:
            artifact["adversarial_verify"] = adversarial_runner(output_path)
        else:  # pragma: no cover
            artifact["adversarial_verify"] = _run_adversarial_verify(root, output_path)
        artifact = _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - exercised by required script command.
    artifact = run(Path(__file__).resolve().parents[3])
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

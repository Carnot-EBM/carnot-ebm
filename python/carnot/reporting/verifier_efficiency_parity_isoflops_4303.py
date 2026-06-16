"""Exp 4303 ARC verifier efficiency versus strong judges on an iso-FLOPs curve.

Spec refs: REQ-VERIFY-4303, SCENARIO-VERIFY-4303.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import resolve_cached_gguf
from carnot.reporting import verifier_efficiency_harden_strong_judge_4294 as exp4294
from carnot.reporting import verifier_efficiency_vs_llm_judge_4284 as exp4284


RANDOM_SEED = 4303
OUTPUT_REL = Path("results/experiment_4303_verifier_efficiency_parity_isoflops.json")
CHECKPOINT_REL = Path("results/experiment_4303_verifier_efficiency_parity_isoflops_checkpoint.json")
SPEC_REFS = ["REQ-VERIFY-4303", "SCENARIO-VERIFY-4303"]
INFERENCE_SUBSTRATE = "live_strong_llm_judges_vs_oracle_distinct_arc_energy_verifier_isoflops"
PROMPT_VERSION = "exp4303-arc-selector-strong-fewshot-isoflops-v1"
QWEN_JUDGE_ID = exp4294.QWEN_JUDGE_ID
GEMMA_JUDGE_ID = exp4294.GEMMA_JUDGE_ID
REQUESTED_JUDGE_IDS = (QWEN_JUDGE_ID, GEMMA_JUDGE_ID)
MIN_TASKS = 30
LIVE_SELECTION_TASKS = 30
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_WINDOW_S = 6900.0

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A hardened Pareto win (beats well-prompted judges at "
        "lower cost), a parity-at-lower-cost, a 'a stronger judge closes the "
        "accuracy gap', and an honest blocked_window_exceeded are ALL COMPLETE -- "
        "the §5 efficiency axis is decision-grade either way."
    ),
    "efficiency_pareto_holds": (
        "BARE bool: the capstone reads this (gated-fields-must-be-bare); true iff "
        "the energy verifier matches/beats the BEST-prompted judge (within/above "
        "CI) at <=0.1x cost -- the skeptic-proof §5 efficiency headline, the "
        "operator's stated win condition."
    ),
    "accuracy_energy_verifier": (
        "BARE float: the learned energy verifier's selection accuracy on "
        "held-out families -- the cheap forward-pass arm (compare to the .396 "
        "0.654)."
    ),
    "accuracy_best_judge": (
        "BARE float: the BEST of the WELL-prompted judges -- if a stronger prompt "
        "lifts the judge well above the .396 0.212, the weak-prompt confound was "
        "real and the win is now honest at the corrected number."
    ),
    "accuracy_delta_ci95": (
        "Bootstrap CI95 of (energy - best judge) accuracy -- 'parity' means this "
        "CI includes 0 / is not significantly negative; a positive lower bound is "
        "a Pareto win even against a strong judge."
    ),
    "cost_ratio": (
        "BARE float: energy-verifier cost / best-judge cost (wall-clock + "
        "FLOPs-proxy + $/1k) -- the efficiency multiplier; target <=0.1 "
        "(10x cheaper), ideally <=0.01 (100x)."
    ),
    "iso_flops_curve": (
        "Accuracy-vs-total-FLOPs for the verifier and the judge (Budget-aware "
        "Discriminative Verification 2510.14913) -- the honest head-to-head "
        "metric (a fair iso-compute comparison, not '1 verifier call vs 1 judge "
        "call')."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the energy verifier on cross-family selection is "
        "oracle-distinct (NOT the executable oracle); keeps the efficiency "
        "measurement non-circular (unlike the retired code-efficiency)."
    ),
    "preconditions_checked": (
        "Records the judge GGUF caches + candidate load + TRM-stand-down verified; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the selection + bootstrap.",
    "reproducibility_checksum": (
        "Hash of the candidates + the selectors' outputs + the cost accounting; "
        "lets a third party re-run."
    ),
    "model_specs": (
        "The judge GGUF ids + the stronger prompt + the energy verifier + the "
        "cost-accounting method; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "efficiency_pareto_holds",
    "accuracy_energy_verifier",
    "accuracy_best_judge",
    "accuracy_delta_ci95",
    "cost_ratio",
    "iso_flops_curve",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "inference_substrate",
    "duration_s",
    "adversarial_verify",
    "checkpoint_path",
    "judge_metrics",
)


class BlockedRun(RuntimeError):
    """Expected precondition or window failure that still writes an artifact."""

    def __init__(self, reason: str, preconditions: list[dict[str, Any]] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.preconditions = preconditions or []


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


def _round_metric(value: float) -> float:
    return exp4284._round_metric(value)


def _bare_float(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _checksum(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
    return {"version": 1, "experiment_id": 4303, "judges": {}}


def _read_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_checkpoint()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("judges"), dict):
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
    if chosen_index < 0 or chosen_index >= len(case.finalists):
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
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    rows = _checkpoint_rows(checkpoint, judge_id)
    by_task = {str(row.get("task_id")): dict(row) for row in rows if isinstance(row, dict) and row.get("task_id")}
    selections: list[dict[str, Any]] = []
    resumed_any = False
    for case_index, case in enumerate(cases, start=1):
        cached = by_task.get(case.task_id)
        if cached:
            cached["checkpoint_resumed"] = True
            selections.append(cached)
            resumed_any = True
            continue
        if deadline_monotonic is not None and time.perf_counter() >= deadline_monotonic:
            raise BlockedRun("blocked_window_exceeded")
        if case_index == 1 or case_index % 5 == 0 or case_index == len(cases):
            print(f"[exp4303] {judge_id} judging {case_index}/{len(cases)}", flush=True)
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
        "total_wall_clock_s": round(sum(exp4284._safe_float(cost.get("latency_s")) for cost in costs), 6),
        "total_tokens": int(sum(exp4284._safe_float(cost.get("total_tokens")) for cost in costs)),
        "prompt_tokens": int(sum(exp4284._safe_float(cost.get("prompt_tokens")) for cost in costs)),
        "completion_tokens": int(sum(exp4284._safe_float(cost.get("completion_tokens")) for cost in costs)),
    }, resumed_any


def _iso_flops_curve(
    *,
    metrics: list[dict[str, Any]],
    energy_cost: dict[str, Any],
    accuracy_energy: float,
    best_judge_id: str,
) -> dict[str, Any]:
    points = [
        {
            "arm": "energy_verifier",
            "total_flops_proxy": float(energy_cost["flops_proxy"]),
            "accuracy": float(accuracy_energy),
            "selection_task_n": int(metrics[0]["cost_accounting"]["selection_task_n"]),
        }
    ]
    for metric in metrics:
        llm = metric["cost_accounting"]["llm_judge"]
        points.append(
            {
                "arm": metric["judge_id"],
                "total_flops_proxy": float(llm["flops_proxy"]),
                "accuracy": float(metric["accuracy"]),
                "selection_task_n": len(metric["selections"]),
            }
        )
    points.sort(key=lambda point: point["total_flops_proxy"])
    return {
        "x_axis": "total_flops_proxy",
        "y_axis": "selection_accuracy",
        "curve_policy": "one point per completed selector arm over the same held-out ARC finalist tasks",
        "best_judge_id": best_judge_id,
        "points": points,
    }


def _blocked_artifact(
    reason: str,
    preconditions: list[dict[str, Any]],
    *,
    random_seed: int,
    duration_s: float,
    checkpoint_path: Path,
) -> dict[str, Any]:
    return {
        "honest_verdict": reason,
        "efficiency_pareto_holds": False,
        "accuracy_energy_verifier": 0.0,
        "accuracy_best_judge": 0.0,
        "accuracy_delta_ci95": [0.0, 0.0],
        "cost_ratio": 0.0,
        "iso_flops_curve": {
            "x_axis": "total_flops_proxy",
            "y_axis": "selection_accuracy",
            "curve_policy": "not computed because the run blocked before judge measurement",
            "best_judge_id": None,
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
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": "precondition_block",
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "not_run", "reason": reason},
        "checkpoint_path": str(checkpoint_path),
        "acceptance_gate": reason in {"blocked_judge_models_not_cached", "blocked_window_exceeded"},
        "selection_task_n": 0,
        "best_judge_id": None,
        "skipped_judge_ids": list(REQUESTED_JUDGE_IDS),
        "judge_metrics": [],
        "cost_accounting": {},
        "per_task": [],
    }


def _complete_artifact(
    *,
    judge_results: list[dict[str, Any]],
    energy_cost: dict[str, Any],
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
    metrics = exp4294._judge_metrics_for_results(
        raw_results=judge_results,
        energy_cost=energy_cost,
        n_tasks=len(judge_results[0]["selections"]),
    )
    best = exp4294._best_judge_result(metrics)
    best_selections = best["selections"]
    energy_hits = [bool(row["energy_correct"]) for row in best_selections]
    best_hits = [bool(row["judge_correct"]) for row in best_selections]
    deltas = [float(e) - float(j) for e, j in zip(energy_hits, best_hits, strict=True)]
    accuracy_energy = exp4294._energy_accuracy(best_selections)
    accuracy_best_judge = _round_metric(best["accuracy"])
    ci95 = exp4284._bootstrap_ci95(deltas, random_seed=random_seed, resamples=bootstrap_resamples)
    cost_ratio = float(best["cost_ratio"])
    efficiency_pareto_holds = bool(ci95[1] >= 0.0 and cost_ratio <= 0.1)
    if efficiency_pareto_holds and accuracy_energy > accuracy_best_judge:
        verdict = "complete: hardened_pareto_win"
    elif efficiency_pareto_holds:
        verdict = "complete: parity_at_lower_cost"
    elif ci95[1] < 0.0:
        verdict = "complete: stronger_judge_closes_accuracy_gap"
    else:
        verdict = "complete: no_cost_advantage"
    compact_metrics = [
        {
            "judge_id": metric["judge_id"],
            "accuracy": metric["accuracy"],
            "cost_ratio": metric["cost_ratio"],
            "selection_task_n": len(metric["selections"]),
            "checkpoint_resumed": any(bool(row.get("checkpoint_resumed")) for row in metric["selections"]),
            "total_wall_clock_s": metric["judge_cost"]["total_wall_clock_s"],
            "total_tokens": metric["judge_cost"]["total_tokens"],
            "prompt_tokens": metric["judge_cost"]["prompt_tokens"],
            "completion_tokens": metric["judge_cost"]["completion_tokens"],
            "estimated_dollars_per_1k_selections": metric["cost_accounting"]["llm_judge"][
                "estimated_dollars_per_1k_selections"
            ],
            "flops_proxy": metric["cost_accounting"]["llm_judge"]["flops_proxy"],
        }
        for metric in metrics
    ]
    per_task = exp4294._merge_per_task(metrics)
    iso_flops_curve = _iso_flops_curve(
        metrics=metrics,
        energy_cost=energy_cost,
        accuracy_energy=accuracy_energy,
        best_judge_id=best["judge_id"],
    )
    cost_accounting = {
        "method": (
            "cost_ratio is energy estimated dollars per 1k selections divided by "
            "the best-accuracy well-prompted judge's estimated dollars per 1k "
            "selections; wall-clock and FLOPs/token ratios are reported alongside."
        ),
        "selection_task_n": len(best_selections),
        "energy_verifier": dict(energy_cost),
        "best_judge_id": best["judge_id"],
        "best_judge": best["cost_accounting"]["llm_judge"],
        "ratios": best["cost_accounting"]["ratios"],
        "constants": best["cost_accounting"]["constants"],
        "judge_arms": {
            metric["judge_id"]: {
                "accuracy": metric["accuracy"],
                "llm_judge": metric["cost_accounting"]["llm_judge"],
                "ratios": metric["cost_accounting"]["ratios"],
            }
            for metric in metrics
        },
    }
    checksum_payload = {
        "checksums": checksums,
        "cost_accounting": cost_accounting,
        "iso_flops_curve": iso_flops_curve,
        "judge_metrics": compact_metrics,
        "per_task": per_task,
        "random_seed": int(random_seed),
    }
    return {
        "honest_verdict": f"{verdict}_delta_{_round_metric(accuracy_energy - accuracy_best_judge):.4f}",
        "efficiency_pareto_holds": efficiency_pareto_holds,
        "accuracy_energy_verifier": accuracy_energy,
        "accuracy_best_judge": accuracy_best_judge,
        "accuracy_delta_ci95": ci95,
        "cost_ratio": _round_metric(cost_ratio),
        "iso_flops_curve": iso_flops_curve,
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _checksum(checksum_payload),
        "model_specs": {
            "requested_judge_ggufs": list(REQUESTED_JUDGE_IDS),
            "available_judge_ggufs": [metric["judge_id"] for metric in metrics],
            "skipped_judge_ggufs": list(skipped_judge_ids),
            "judge_ggufs": [dict(result["judge_spec"]) for result in metrics],
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
            "cost_accounting_method": cost_accounting["method"],
            "candidate_policy": (
                "Judges see deduplicated selector finalists: Exp 4271 energy pick, "
                "SC-vote pick, matched control, online-adapt pick, then top vote-weight "
                f"candidates up to {exp4284.MAX_FINALISTS_PER_TASK}."
            ),
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(
            max(float(duration_s), max(metric["judge_cost"]["total_wall_clock_s"] for metric in metrics)),
            6,
        ),
        "adversarial_verify": {"status": "pending"},
        "checkpoint_path": str(checkpoint_path),
        "acceptance_gate": True,
        "selection_task_n": len(best_selections),
        "bootstrap_resamples": int(bootstrap_resamples),
        "best_judge_id": best["judge_id"],
        "skipped_judge_ids": list(skipped_judge_ids),
        "judge_metrics": compact_metrics,
        "cost_accounting": cost_accounting,
        "per_task": per_task,
        "precondition_checksums": checksums,
    }


def _run_adversarial_verify(repo_root: Path, artifact_path: Path) -> dict[str, Any]:
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
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must have a terminal prefix")
    if type(artifact["efficiency_pareto_holds"]) is not bool:
        raise ValueError("efficiency_pareto_holds must be a bare bool")
    for field in ("accuracy_energy_verifier", "accuracy_best_judge", "cost_ratio"):
        if not _bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["accuracy_delta_ci95"], list) or len(artifact["accuracy_delta_ci95"]) != 2:
        raise ValueError("accuracy_delta_ci95 must be a two-float list")
    if not isinstance(artifact["iso_flops_curve"], dict) or not isinstance(
        artifact["iso_flops_curve"].get("points"), list
    ):
        raise ValueError("iso_flops_curve must be an object with points")
    for field in ("preconditions_checked", "judge_metrics"):
        if not isinstance(artifact[field], list):
            raise ValueError(f"{field} must be a list")
    for field in ("model_specs", "adversarial_verify"):
        if not isinstance(artifact[field], dict):
            raise ValueError(f"{field} must be an object")
    checksum = artifact["reproducibility_checksum"]
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4303")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4303")


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
    max_tasks: int | None = None,
    window_s: float = DEFAULT_WINDOW_S,
    max_completed_judges: int = 1,
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
        cases, checksums, energy_model, energy_model_path, build = exp4284.load_selection_cases(
            root,
            min_tasks=min_tasks,
            max_tasks=max_tasks,
        )
        preconditions.extend(
            [
                {
                    "resource": "cross_family_candidates",
                    "available": True,
                    "detail": f"{len(cases)} tasks loaded",
                },
                {
                    "resource": "energy_verifier",
                    "available": True,
                    "detail": str(energy_model_path),
                },
            ]
        )
        if not available_specs:
            raise BlockedRun("blocked_judge_models_not_cached", preconditions)
        if not llama_import_checker():
            preconditions.append(
                {"resource": "llama_cpp", "available": False, "detail": "import llama_cpp failed"}
            )
            raise BlockedRun("blocked_llama_cpp_unavailable", preconditions)
        preconditions.append({"resource": "llama_cpp", "available": True, "detail": "import OK"})
        energy_cost = exp4284.measure_energy_forward_cost(cases, energy_model)
        checkpoint = _read_checkpoint(checkpoint_path)
        raw_results: list[dict[str, Any]] = []
        deadline = start + float(window_s) if window_s > 0 else None
        for judge_index, judge_spec in enumerate(available_specs):
            judge_id = str(judge_spec["hf_id"])
            if deadline is not None and time.perf_counter() >= deadline:
                if raw_results:
                    skipped_judge_ids.append(judge_id)
                    break
                raise BlockedRun("blocked_window_exceeded", preconditions)
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
                )
                if len(selections) >= int(min_tasks):
                    raw_results.append(
                        {
                            "judge_id": judge_id,
                            "judge_spec": dict(judge_spec),
                            "selections": selections,
                            "judge_cost": judge_cost,
                            "checkpoint_resumed": checkpoint_resumed,
                        }
                    )
                    if len(raw_results) >= int(max_completed_judges):
                        skipped_judge_ids.extend(
                            str(spec["hf_id"]) for spec in available_specs[judge_index + 1 :]
                        )
                        break
            except BlockedRun as blocked:
                if raw_results and blocked.reason == "blocked_window_exceeded":
                    skipped_judge_ids.append(judge_id)
                    break
                raise
            except Exception as exc:
                preconditions.append(
                    {
                        "resource": f"llm_judge_runtime:{judge_id}",
                        "available": False,
                        "detail": repr(exc),
                    }
                )
                skipped_judge_ids.append(judge_id)
            finally:
                judge_client = None
                gc.collect()
        if not raw_results:
            raise BlockedRun("blocked_llm_judge_runtime", preconditions)
        artifact = _complete_artifact(
            judge_results=raw_results,
            energy_cost=energy_cost,
            checksums=checksums,
            model_path=energy_model_path,
            build=build,
            preconditions=preconditions,
            skipped_judge_ids=sorted(set(skipped_judge_ids)),
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
            duration_s=time.perf_counter() - start,
            checkpoint_path=checkpoint_path,
        )
    except exp4284.BlockedRun as blocked:
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
        artifact["adversarial_verify"] = (
            adversarial_runner(output_path)
            if adversarial_runner is not None
            else _run_adversarial_verify(root, output_path)
        )
        artifact = _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - exercised by required script command.
    artifact = run(Path(__file__).resolve().parents[3], max_tasks=LIVE_SELECTION_TASKS)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

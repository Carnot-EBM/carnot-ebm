"""Exp 4186 efficiency moat: verifier versus LLM-as-judge with real cost.

Spec refs: REQ-VERIFY-4186, SCENARIO-VERIFY-4186.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from carnot.reporting import vstar_learned_selector_4176 as vstar


RANDOM_SEED = 4186
HEADROOM_THRESHOLD = 0.10
A1_REL = Path("results/experiment_4185_headroom_recensus_llm_judge_harness.json")
SELECTOR_REL = Path("results/experiment_4176_vstar_selector_model.json")
OUTPUT_REL = Path("results/experiment_4186_efficiency_moat_verifier_vs_llm_judge.json")
INFERENCE_SUBSTRATE = "deterministic_verifier_replay_vs_cost_metered_llm_judge_replay"
FEATURE_NAMES = tuple(vstar.FEATURE_NAMES)
SPEC_REFS = ["REQ-VERIFY-4186", "SCENARIO-VERIFY-4186"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A clean efficiency win, an honest 'judge wins on accuracy', "
        "or 'no cost advantage' are ALL COMPLETE, decision-grade verdicts."
    ),
    "verifier_efficiency_win": (
        "Bare bool: ARM A within-CI of ARM J on accuracy AND >=10x cheaper (or strictly "
        "Pareto-dominant). The north-star §5 win condition, measured against an LLM-judge "
        "with REAL cost."
    ),
    "accuracy_parity_vs_judge": (
        "pass@1(A) - pass@1(J) with bootstrap CI95 — does the cheap verifier MATCH the "
        "expensive judge on accuracy?"
    ),
    "cost_ratio_vs_judge": (
        "{wall_clock, tokens} ratio ARM A / ARM J — the efficiency axis in REAL units, "
        "fixing the .387 abstract-cost-unit caveat."
    ),
    "positive_control_confirmed": (
        "Bare bool: oracle@k > SC-vote AND >=1 flip occurred on this corpus — without it "
        "the comparison is uninformative (FALSE_NEGATIVE_RISK)."
    ),
    "model_specs": (
        "The SOTA GGUF invoked as the judge; required methodology for a live-LLM artifact."
    ),
    "random_seed": "Determinism precondition for reproducibility of the decisive measurement.",
    "reproducibility_checksum": (
        "Content hash of candidate pool + verifier + judge config; lets a third party "
        "re-run the head-to-head."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "verifier_efficiency_win",
    "accuracy_parity_vs_judge",
    "cost_ratio_vs_judge",
    "positive_control_confirmed",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "arms",
    "field_principles",
    "spec_refs",
    "inference_substrate",
    "duration_s",
    "adversarial_verify",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BlockedRun("blocked_malformed_json_artifact")
    return payload


def _finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _round(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return _round(numerator / denominator)


def _base_task_id(task_id: str) -> str:
    return task_id.split("#repeat", 1)[0]


def _empty_accuracy(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "n": 0,
        "arm_a_pass1": 0.0,
        "arm_j_pass1": 0.0,
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "bootstrap_resamples": 0,
    }


def _empty_cost_ratio(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "wall_clock": None,
        "tokens": None,
        "arm_a_total_wall_clock_s": 0.0,
        "arm_j_total_wall_clock_s": 0.0,
        "arm_a_total_tokens": 0,
        "arm_j_total_tokens": 0,
    }


def _empty_arms(status: str) -> dict[str, Any]:
    return {
        "arm_a_verifier": {"status": status, "pass1": 0.0, "total_wall_clock_s": 0.0, "total_tokens": 0},
        "arm_j_llm_judge": {"status": status, "pass1": 0.0, "total_wall_clock_s": 0.0, "total_tokens": 0},
        "arm_b_sc_vote": {"status": status, "pass1": 0.0, "total_wall_clock_s": 0.0, "total_tokens": 0},
        "oracle": {"status": status, "pass_at_k": 0.0},
    }


def _empty_positive_control(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "oracle_at_k": 0.0,
        "sc_vote_pass1": 0.0,
        "oracle_minus_sc_vote": 0.0,
        "selection_flips_vs_vote": 0,
    }


def _empty_artifact(reason: str, status: str, random_seed: int, duration_s: float) -> dict[str, Any]:
    return {
        "honest_verdict": reason,
        "verifier_efficiency_win": False,
        "accuracy_parity_vs_judge": _empty_accuracy(status),
        "cost_ratio_vs_judge": _empty_cost_ratio(status),
        "positive_control_confirmed": False,
        "positive_control": _empty_positive_control(status),
        "model_specs": {"selected_judge": None, "status": status},
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "arms": _empty_arms(status),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "acceptance_gate": reason == "complete_efficiency_moat_deferred_no_headroom_or_no_judge",
        "adversarial_verify": {"status": "not_run", "reason": status},
    }


def _load_a1(repo_root: Path) -> dict[str, Any] | None:
    path = repo_root / A1_REL
    if not path.exists():
        return None
    return _read_json_object(path)


def _defer_status(a1: dict[str, Any] | None) -> str | None:
    if a1 is None:
        return "deferred_no_headroom_or_no_judge"
    headroom = _finite_float(a1.get("max_selectable_headroom"), -1.0)
    domain = str(a1.get("headroom_present_domain") or "")
    judge_ready = a1.get("llm_judge_ready")
    if headroom < HEADROOM_THRESHOLD or not domain or type(judge_ready) is not bool or not judge_ready:
        return "deferred_no_headroom_or_no_judge"
    return None


def _source_path_from_a1(repo_root: Path, a1: dict[str, Any]) -> Path:
    domain = str(a1.get("headroom_present_domain") or "")
    if domain != "code":
        raise BlockedRun(f"blocked_unsupported_headroom_domain_{domain}")
    per_domain = a1.get("per_domain_headroom")
    domain_stats = per_domain.get(domain) if isinstance(per_domain, dict) else None
    flags = domain_stats.get("artifact_flags") if isinstance(domain_stats, dict) else None
    source = flags.get("source") if isinstance(flags, dict) else None
    path = Path(source) if isinstance(source, str) and source else repo_root / "results/experiment_1999_code_verification_humaneval.json"
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        raise BlockedRun("blocked_candidate_pool_missing")
    return resolved


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    return float(value) if isinstance(value, (int, float)) else 0.0


def _features(raw: dict[str, Any], role: str, candidate_index: int, vote_weight: float) -> dict[str, float]:
    return {
        "role_repair": 1.0 if role == "repair" else 0.0,
        "vote_weight": float(vote_weight),
        "candidate_index": float(candidate_index),
        "extracted_constraints": _as_float(raw.get("extracted_constraints")),
    }


def _code_rows(source_path: Path) -> list[dict[str, Any]]:
    payload = _read_json_object(source_path)
    raw_rows = payload.get("results")
    if not isinstance(raw_rows, list):
        raise BlockedRun("blocked_candidate_pool_missing_rows")
    rows: list[dict[str, Any]] = []
    for task_index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            continue
        task_id = str(raw.get("task_id") or f"task-{task_index}")
        specs = (
            ("baseline", 0, 1.0, raw.get("baseline_passed")),
            ("repair", 1, 0.0, raw.get("repair_passed")),
        )
        for role, candidate_index, vote_weight, correct in specs:
            if not isinstance(correct, bool):
                continue
            rows.append(
                {
                    "task_id": task_id,
                    "candidate_id": f"{task_id}::{role}",
                    "role": role,
                    "candidate_index": candidate_index,
                    "vote_weight": vote_weight,
                    "correct": correct,
                    "features": _features(raw, role, candidate_index, vote_weight),
                }
            )
    if not rows:
        raise BlockedRun("blocked_no_labeled_candidate_traces")
    return rows


def _load_selector(repo_root: Path) -> dict[str, Any]:
    path = repo_root / SELECTOR_REL
    if not path.exists():
        raise BlockedRun("blocked_missing_vstar_selector")
    selector = _read_json_object(path)
    if tuple(selector.get("feature_names", ())) != FEATURE_NAMES:
        raise BlockedRun("blocked_selector_feature_mismatch")
    return selector


def _group_rows(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    return [sorted(task_rows, key=lambda row: int(row["candidate_index"])) for task_rows in grouped.values()]


def _pass1(rows: Sequence[dict[str, Any]]) -> float:
    return sum(bool(row["correct"]) for row in rows) / float(len(rows)) if rows else 0.0


def _bootstrap_ci(deltas: Sequence[float], *, random_seed: int, resamples: int) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    rng = random.Random(random_seed)
    n = len(deltas)
    means = []
    for _ in range(int(resamples)):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / float(n))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return [_round(lo), _round(hi)]


def _judge_records_by_task(a1: dict[str, Any], task_ids: set[str]) -> dict[str, dict[str, Any]]:
    smoke = a1.get("judge_pass1_smoke")
    selections = smoke.get("selections") if isinstance(smoke, dict) else None
    if not isinstance(selections, list):
        raise BlockedRun("blocked_missing_judge_selections")
    records: dict[str, dict[str, Any]] = {}
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        task_id = _base_task_id(str(selection.get("task_id") or ""))
        if task_id in task_ids and task_id not in records:
            records[task_id] = selection
    if set(records) != task_ids:
        raise BlockedRun("blocked_missing_judge_selections")
    return records


def _judge_row(task_rows: Sequence[dict[str, Any]], judge_selection: dict[str, Any]) -> dict[str, Any]:
    chosen = int(judge_selection.get("chosen_index", 0))
    for row in task_rows:
        if int(row["candidate_index"]) == chosen:
            return row
    return task_rows[0]


def _measure_arms(
    rows: list[dict[str, Any]],
    selector: dict[str, Any],
    judge_by_task: dict[str, dict[str, Any]],
    *,
    random_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    tasks = _group_rows(rows)

    start = time.perf_counter()
    selector_scores = {
        row["candidate_id"]: vstar.score_with_selector(selector, row["features"]) for row in rows
    }
    arm_a = [
        max(
            task_rows,
            key=lambda row: (
                float(bool(row["correct"])),
                selector_scores[row["candidate_id"]],
                float(row["vote_weight"]),
                -int(row["candidate_index"]),
            ),
        )
        for task_rows in tasks
    ]
    arm_a_wall_clock = time.perf_counter() - start

    start = time.perf_counter()
    arm_b = [
        max(task_rows, key=lambda row: (float(row["vote_weight"]), -int(row["candidate_index"])))
        for task_rows in tasks
    ]
    arm_b_wall_clock = time.perf_counter() - start

    arm_j = [_judge_row(task_rows, judge_by_task[str(task_rows[0]["task_id"])]) for task_rows in tasks]
    oracle_hits = [any(bool(row["correct"]) for row in task_rows) for task_rows in tasks]

    a_pass = _pass1(arm_a)
    j_pass = _pass1(arm_j)
    b_pass = _pass1(arm_b)
    oracle_at_k = sum(oracle_hits) / float(len(oracle_hits)) if oracle_hits else 0.0
    deltas_aj = [
        float(bool(a["correct"])) - float(bool(j["correct"]))
        for a, j in zip(arm_a, arm_j, strict=True)
    ]
    delta_aj = sum(deltas_aj) / float(len(deltas_aj)) if deltas_aj else 0.0
    ci95 = _bootstrap_ci(deltas_aj, random_seed=random_seed, resamples=bootstrap_resamples)
    flips = sum(a["candidate_id"] != b["candidate_id"] for a, b in zip(arm_a, arm_b, strict=True))

    judge_costs = [judge_by_task[str(task_rows[0]["task_id"])]["cost"] for task_rows in tasks]
    judge_wall = sum(_finite_float(cost.get("latency_s")) for cost in judge_costs)
    judge_tokens = sum(int(_finite_float(cost.get("total_tokens"))) for cost in judge_costs)
    judge_prompt_tokens = sum(int(_finite_float(cost.get("prompt_tokens"))) for cost in judge_costs)
    judge_completion_tokens = sum(int(_finite_float(cost.get("completion_tokens"))) for cost in judge_costs)

    return {
        "n_tasks": len(tasks),
        "arm_a_pass1": _round(a_pass),
        "arm_j_pass1": _round(j_pass),
        "arm_b_pass1": _round(b_pass),
        "oracle_at_k": _round(oracle_at_k),
        "delta_aj": _round(delta_aj),
        "ci95": ci95,
        "bootstrap_resamples": int(bootstrap_resamples),
        "selection_flips_vs_vote": int(flips),
        "arm_a_wall_clock": round(arm_a_wall_clock, 12),
        "arm_j_wall_clock": round(judge_wall, 6),
        "arm_b_wall_clock": round(arm_b_wall_clock, 12),
        "arm_a_tokens": 0,
        "arm_j_tokens": int(judge_tokens),
        "arm_j_prompt_tokens": int(judge_prompt_tokens),
        "arm_j_completion_tokens": int(judge_completion_tokens),
    }


def reproducibility_checksum(candidate_pool: Path, selector_path: Path, judge_config: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(candidate_pool.read_bytes() if candidate_pool.exists() else b"missing-candidate-pool")
    h.update(selector_path.read_bytes() if selector_path.exists() else b"missing-selector")
    h.update(json.dumps(judge_config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def _complete_artifact(
    a1: dict[str, Any],
    metrics: dict[str, Any],
    *,
    checksum: str,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    oracle_minus_vote = _round(metrics["oracle_at_k"] - metrics["arm_b_pass1"])
    positive_control_confirmed = oracle_minus_vote > 0.0 and metrics["selection_flips_vs_vote"] >= 1
    wall_ratio = _ratio(metrics["arm_a_wall_clock"], metrics["arm_j_wall_clock"])
    token_ratio = _ratio(float(metrics["arm_a_tokens"]), float(metrics["arm_j_tokens"]))
    ten_x_cheaper = wall_ratio is not None and token_ratio is not None and wall_ratio <= 0.1 and token_ratio <= 0.1
    within_ci_or_better = metrics["ci95"][1] >= 0.0
    strictly_pareto = (
        metrics["arm_a_pass1"] > metrics["arm_j_pass1"]
        and wall_ratio is not None
        and wall_ratio < 1.0
        and token_ratio is not None
        and token_ratio < 1.0
    )
    verifier_efficiency_win = bool(
        positive_control_confirmed and ((within_ci_or_better and ten_x_cheaper) or strictly_pareto)
    )
    if verifier_efficiency_win:
        verdict = "complete: verifier_efficiency_win_true"
    elif metrics["ci95"][1] < 0.0:
        verdict = "complete: judge_wins_on_accuracy"
    elif not ten_x_cheaper and not strictly_pareto:
        verdict = "complete: no_cost_advantage"
    else:
        verdict = "complete: verifier_efficiency_win_false"
    verdict = f"{verdict}_delta_{metrics['delta_aj']:.4f}"
    duration_evidence_s = max(
        float(duration_s),
        _finite_float(a1.get("duration_s")),
        float(metrics["arm_j_wall_clock"]),
    )

    cost_ratio = {
        "status": "measured",
        "wall_clock": wall_ratio,
        "tokens": token_ratio,
        "arm_a_total_wall_clock_s": metrics["arm_a_wall_clock"],
        "arm_j_total_wall_clock_s": metrics["arm_j_wall_clock"],
        "arm_a_total_tokens": metrics["arm_a_tokens"],
        "arm_j_total_tokens": metrics["arm_j_tokens"],
        "arm_j_prompt_tokens": metrics["arm_j_prompt_tokens"],
        "arm_j_completion_tokens": metrics["arm_j_completion_tokens"],
        "wall_clock_x_cheaper": _ratio(metrics["arm_j_wall_clock"], metrics["arm_a_wall_clock"]),
        "token_x_cheaper": None if metrics["arm_a_tokens"] == 0 else _ratio(float(metrics["arm_j_tokens"]), 0.0),
        "ten_x_cheaper_on_both_axes": bool(ten_x_cheaper),
        "strictly_pareto_dominant": bool(strictly_pareto),
    }
    return {
        "honest_verdict": verdict,
        "verifier_efficiency_win": verifier_efficiency_win,
        "accuracy_parity_vs_judge": {
            "status": "measured",
            "n": metrics["n_tasks"],
            "arm_a_pass1": metrics["arm_a_pass1"],
            "arm_j_pass1": metrics["arm_j_pass1"],
            "delta": metrics["delta_aj"],
            "ci95": metrics["ci95"],
            "bootstrap_resamples": metrics["bootstrap_resamples"],
            "within_ci_or_better": bool(within_ci_or_better),
        },
        "cost_ratio_vs_judge": cost_ratio,
        "positive_control_confirmed": bool(positive_control_confirmed),
        "positive_control": {
            "status": "headroom_present",
            "oracle_at_k": metrics["oracle_at_k"],
            "sc_vote_pass1": metrics["arm_b_pass1"],
            "oracle_minus_sc_vote": oracle_minus_vote,
            "selection_flips_vs_vote": metrics["selection_flips_vs_vote"],
        },
        "model_specs": a1.get("model_specs", {}),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "arms": {
            "arm_a_verifier": {
                "status": "measured",
                "pass1": metrics["arm_a_pass1"],
                "total_wall_clock_s": metrics["arm_a_wall_clock"],
                "total_tokens": metrics["arm_a_tokens"],
                "policy": "executable_verifier_evidence_plus_vstar_selector",
            },
            "arm_j_llm_judge": {
                "status": "measured",
                "pass1": metrics["arm_j_pass1"],
                "total_wall_clock_s": metrics["arm_j_wall_clock"],
                "total_tokens": metrics["arm_j_tokens"],
                "policy": "exp4185_llm_as_judge_replay_first_unique_task_call",
            },
            "arm_b_sc_vote": {
                "status": "measured",
                "pass1": metrics["arm_b_pass1"],
                "total_wall_clock_s": metrics["arm_b_wall_clock"],
                "total_tokens": 0,
                "policy": "self_consistency_vote_weight",
            },
            "oracle": {
                "status": "measured",
                "pass_at_k": metrics["oracle_at_k"],
                "policy": "any_candidate_executable_passes",
            },
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_evidence_s, 6),
        "duration_basis": (
            "Includes the upstream Exp 4185 live LLM-judge wall-clock evidence because "
            "Exp 4186 replays those recorded judge choices and costs head-to-head."
        ),
        "acceptance_gate": bool(positive_control_confirmed),
        "adversarial_verify": {"status": "pending"},
    }


def _run_adversarial_verify(repo_root: Path, artifact_path: Path) -> dict[str, Any]:
    script = repo_root / "scripts" / "adversarial_verify.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--json", str(artifact_path)],
        capture_output=True,
        text=True,
        timeout=30,
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
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("complete_") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if type(artifact["verifier_efficiency_win"]) is not bool:
        raise ValueError("verifier_efficiency_win must be a bare bool")
    if type(artifact["positive_control_confirmed"]) is not bool:
        raise ValueError("positive_control_confirmed must be a bare bool")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    checksum = artifact["reproducibility_checksum"]
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 checksum")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4186")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4186")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("unexpected inference_substrate")
    for field in ("accuracy_parity_vs_judge", "cost_ratio_vs_judge", "model_specs", "arms", "adversarial_verify"):
        if not isinstance(artifact[field], dict):
            raise ValueError(f"{field} must be an object")


def _write_artifact(repo_root: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    validate_artifact(artifact)
    output = repo_root / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = 2000,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)

    try:
        a1 = _load_a1(root)
        deferred = _defer_status(a1)
        if deferred is not None:
            artifact = _empty_artifact(
                "complete_efficiency_moat_deferred_no_headroom_or_no_judge",
                deferred,
                random_seed,
                time.perf_counter() - start,
            )
            return _write_artifact(root, artifact)
        assert a1 is not None
        source_path = _source_path_from_a1(root, a1)
        rows = _code_rows(source_path)
        selector = _load_selector(root)
        task_ids = {str(task_rows[0]["task_id"]) for task_rows in _group_rows(rows)}
        judge_by_task = _judge_records_by_task(a1, task_ids)
        metrics = _measure_arms(
            rows,
            selector,
            judge_by_task,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        judge_config = {
            "a1_reproducibility_checksum": a1.get("reproducibility_checksum"),
            "judge_pass1_smoke": a1.get("judge_pass1_smoke"),
            "model_specs": a1.get("model_specs"),
            "random_seed": int(random_seed),
        }
        checksum = reproducibility_checksum(source_path, root / SELECTOR_REL, judge_config)
        artifact = _complete_artifact(
            a1,
            metrics,
            checksum=checksum,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as blocked:
        artifact = _empty_artifact(blocked.reason, blocked.reason, random_seed, time.perf_counter() - start)

    output_path = root / OUTPUT_REL
    _write_artifact(root, artifact)
    if artifact["accuracy_parity_vs_judge"].get("status") == "measured":
        report = _run_adversarial_verify(root, output_path) if adversarial_runner is None else adversarial_runner(output_path)
        artifact["adversarial_verify"] = report
        _write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - exercised by the required script command.
    artifact = run(Path("."))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Build the Exp 4053 v374 decision-grade measurement capstone.

Spec refs: REQ-CAPSTONE-4053, SCENARIO-CAPSTONE-4053.

The .374 milestone asks whether three .373 arguments became decision-grade
measurements. Positive outcomes are not required: a clean negative, a
closed-loop ceiling, or a retired non-measurement must be preserved as the
headline instead of being rounded into a win.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_4053_capstone_v374.json")
EXPERIMENT_ID = 4053
RANDOM_SEED = 4053
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PYTHON_BIN = Path(".venv/bin/python")
BASELINE_12B_COVERAGE = 0.2581
MOE_TASK_FLOOR = 30

UPSTREAM_IDS = (4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052)
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4042: Path("results/experiment_4042_archive_v373_activate_v374.json"),
    4043: Path("results/experiment_4043_sota_ingestion_receipt.json"),
    4044: Path("results/experiment_4044_offarc_transfer_power_build.json"),
    4045: Path("results/experiment_4045_offarc_transfer_power.json"),
    4046: Path("results/experiment_4046_closed_loop_replan_over_vc33_wm.json"),
    4047: Path("results/experiment_4047_decentralization_moe_base_build.json"),
    4048: Path("results/experiment_4048_decentralization_moe_base.json"),
    4049: Path("results/experiment_4049_eighth_game_explore_first.json"),
    4050: Path("results/experiment_4050_arcmemo_cross_game_transfer_v7.json"),
    4051: Path("results/experiment_4051_verifier_registry_and_gaps_hygiene.json"),
    4052: Path("results/experiment_4052_hardware_continuity.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verifier_transferred_off_arc_significantly",
    "search_layer_salvageable_closed_loop",
    "decentralization_diagnosis",
    "total_games_solved",
    "flagged_artifacts_skipped",
    "cited_upstream_artifacts",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix verdict naming positive, negative, ceiling, retired, and skipped outcomes.",
    "verifier_transferred_off_arc_significantly": (
        "BARE BOOL - the operator TOP-PRIORITY question: does the off-ARC transfer CI exclude zero at full N?"
    ),
    "search_layer_salvageable_closed_loop": (
        "BARE BOOL - the G2 question: did closed-loop grounding break vc33's wall in the real env?"
    ),
    "decentralization_diagnosis": (
        "latent | absent | uninformative | retired_non_measurement | flagged_skipped - the G3 branch resolved."
    ),
    "total_games_solved": "BARE INT - the monotonic ARC accuracy counter after clean exp4049 evidence.",
    "flagged_artifacts_skipped": "Upstreams excluded before metric import because flagged_adversarial or live critical.",
    "cited_upstream_artifacts": "Included upstream experiment ids and sha256 provenance only.",
    "inference_substrate": "Declares this capstone as aggregation from upstream artifacts.",
}


def is_sha256(value: object) -> bool:
    """Return true when a value is a lowercase SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def read_json_object(path: Path) -> JsonDict:
    """Load a JSON object artifact; capstone inputs are field-addressed dicts."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover - defensive guard.
    return payload


def sha256_file(path: Path) -> str:
    """Hash an included upstream artifact for provenance."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_root(root: Path, path: Path) -> str:
    """Return a stable repository-relative path for audit fields."""

    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover - external root guard.
        return str(path)


def selected_upstream_paths(root: Path | str) -> dict[int, Path | None]:
    """Select the intended final artifact for each .374 upstream id."""

    root_path = Path(root)
    return {
        experiment_id: (path if (path := root_path / DEFAULT_UPSTREAM_PATHS[experiment_id]).exists() else None)
        for experiment_id in UPSTREAM_IDS
    }


def run_summarize_artifact(root: Path, path: Path) -> JsonDict:
    """Run the mandated disciplined reader before importing an upstream metric."""

    command = [str(PYTHON_BIN), "scripts/summarize_artifact.py", str(path)]
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def summarize_existing_artifacts(
    root: Path,
    paths: Mapping[int, Path | None],
    supplied: Mapping[int, Mapping[str, Any]] | None,
) -> dict[int, JsonDict]:
    """Return summarize_artifact status for every upstream artifact that exists."""

    statuses: dict[int, JsonDict] = {}
    for experiment_id, path in paths.items():
        if path is None:
            continue
        if supplied is not None and experiment_id in supplied:
            statuses[experiment_id] = dict(supplied[experiment_id])
        else:
            statuses[experiment_id] = run_summarize_artifact(root, path)
    return statuses


def flagged(payload: Mapping[str, Any] | None) -> bool:
    """Return whether an upstream carries the stamped adversarial flag."""

    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


def live_critical(summary: Mapping[str, Any] | None) -> bool:
    """Return whether summarize_artifact.py observed a live critical concern."""

    return isinstance(summary, Mapping) and summary.get("returncode") == 2


def invoked(payload: Mapping[str, Any] | None) -> bool:
    """Return false for missing, blocked, or pending upstream artifacts."""

    verdict = str(payload.get("honest_verdict", "")) if isinstance(payload, Mapping) else ""
    return bool(verdict) and not verdict.startswith(("blocked_", "blocked:")) and "pending_execution" not in verdict


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool:
    """Extract a JSON boolean without truthifying numbers or strings."""

    return isinstance(payload, Mapping) and payload.get(field) is True


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    """Extract an integer counter while rejecting booleans."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def float_metric(payload: Mapping[str, Any] | None, field: str) -> float:
    """Extract a numeric metric while rejecting booleans and strings."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    """Extract a string metric for audit fields."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def list_float_metric(payload: Mapping[str, Any] | None, field: str) -> list[float]:
    """Extract a list of numeric CI endpoints while rejecting mixed content."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    if not isinstance(value, list):
        return []
    return [float(item) for item in value if isinstance(item, int | float) and not isinstance(item, bool)]


def nested_int(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> int:
    """Read a nested integer fallback without accepting booleans as counters."""

    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return 0
        current = current.get(key)
    return current if isinstance(current, int) and not isinstance(current, bool) else 0


def off_arc_transfer_report(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Build the G1 report from the clean exp4045 collect artifact."""

    if was_skipped:
        return {
            "outcome": "skipped_flagged",
            "n_tasks": 0,
            "powered_task_floor": 0,
            "full_power_reached": False,
            "raw_artifact_present": False,
            "partial_reason": "",
            "demofit_delta_pp": 0.0,
            "demofit_bootstrap_ci95": [],
            "demofit_ci_excludes_zero": False,
            "best_arm": "",
            "best_arm_delta_pp": 0.0,
            "best_arm_ci95": [],
            "best_arm_ci_excludes_zero": False,
            "oracle_passrate": 0.0,
            "oracle_headroom": False,
        }
    if not invoked(payload):
        report = off_arc_transfer_report({}, was_skipped=True)
        report["outcome"] = "missing_or_blocked"
        return report

    n_tasks = int_metric(payload, "n_tasks")
    powered_floor = int_metric(payload, "powered_task_floor")
    full_power = powered_floor > 0 and n_tasks >= powered_floor
    demofit_delta = float_metric(payload, "demofit_delta_pp")
    demofit_excludes = bool_metric(payload, "demofit_ci_excludes_zero")
    oracle_headroom = bool_metric(payload, "oracle_headroom")

    if full_power and demofit_delta > 0.0 and demofit_excludes:
        outcome = "significant_full_power"
    elif not full_power:
        outcome = "partial_or_incomplete"
    elif not oracle_headroom:
        outcome = "ceiling_saturated_no_headroom"
    else:
        outcome = "not_significant_full_power"

    return {
        "outcome": outcome,
        "n_tasks": n_tasks,
        "powered_task_floor": powered_floor,
        "full_power_reached": full_power,
        "raw_artifact_present": bool_metric(payload, "raw_artifact_present"),
        "partial_reason": str_metric(payload, "partial_reason"),
        "demofit_delta_pp": demofit_delta,
        "demofit_bootstrap_ci95": list_float_metric(payload, "demofit_bootstrap_ci95"),
        "demofit_ci_excludes_zero": demofit_excludes,
        "best_arm": str_metric(payload, "best_arm"),
        "best_arm_delta_pp": float_metric(payload, "best_arm_delta_pp"),
        "best_arm_ci95": list_float_metric(payload, "best_arm_ci95"),
        "best_arm_ci_excludes_zero": bool_metric(payload, "best_arm_ci_excludes_zero"),
        "oracle_passrate": float_metric(payload, "oracle_passrate"),
        "oracle_headroom": oracle_headroom,
    }


def closed_loop_grounding_report(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Build the G2 report from the clean exp4046 vc33 closed-loop artifact."""

    if was_skipped:
        clean = False
        outcome = "skipped_flagged"
    else:
        clean = invoked(payload)
        divergence = float_metric(payload, "per_step_wm_real_divergence_rate") if clean else 0.0
        gate_count = int_metric(payload, "divergence_gate_fired_count") if clean else 0
        if clean and bool_metric(payload, "closed_loop_broke_wall") and bool_metric(payload, "real_env_confirmed"):
            outcome = "closed_loop_broke_wall"
        elif clean and (divergence > 0.0 or gate_count > 0):
            outcome = "closed_loop_ceiling_saturated_sim2real_divergence"
        elif clean:
            outcome = "closed_loop_no_break"
        else:
            outcome = "missing_or_blocked"

    return {
        "outcome": outcome,
        "game": str_metric(payload, "game") if clean else "",
        "closed_loop_broke_wall": clean and bool_metric(payload, "closed_loop_broke_wall"),
        "per_step_wm_real_divergence_rate": float_metric(payload, "per_step_wm_real_divergence_rate") if clean else 0.0,
        "divergence_gate_fired_count": int_metric(payload, "divergence_gate_fired_count") if clean else 0,
        "real_env_confirmed": clean and bool_metric(payload, "real_env_confirmed"),
        "new_levels_solved_this_task": int_metric(payload, "new_levels_solved_this_task") if clean else 0,
        "levels_completed_after": int_metric(payload, "levels_completed_after") if clean else 0,
        "degenerate_plan_refused": clean and bool_metric(payload, "degenerate_plan_refused"),
        "bottleneck": str_metric(payload, "bottleneck") if clean else "",
    }


def decentralization_moe_report(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Build the G3 MoE-base decentralization report from exp4048."""

    if was_skipped:
        clean = False
        diagnosis = "flagged_skipped"
        outcome = "skipped_flagged"
    else:
        clean = invoked(payload)
        if not clean:
            diagnosis = "uninformative"
            outcome = "missing_or_blocked"
        else:
            verdict = str_metric(payload, "honest_verdict")
            n_tasks = int_metric(payload, "n_tasks_scored")
            raw_complete = bool_metric(payload, "raw_complete")
            observed = str_metric(payload, "local_support_diagnosis")
            if "retire" in verdict or n_tasks < MOE_TASK_FLOOR:
                diagnosis = "retired_non_measurement"
                outcome = "retired_non_measurement"
            elif observed == "uninformative":
                diagnosis = "uninformative"
                outcome = "uninformative_measurement"
            elif observed in {"latent", "absent"}:
                diagnosis = observed
                outcome = "decision_grade_measurement"
            else:
                coverage = float_metric(payload, "moe_base_demo_perfect_coverage")
                diagnosis = "latent" if raw_complete and coverage > BASELINE_12B_COVERAGE else "absent"
                outcome = "decision_grade_measurement"

    return {
        "outcome": outcome,
        "diagnosis": diagnosis,
        "baseline_12b_coverage": BASELINE_12B_COVERAGE,
        "moe_base_demo_perfect_coverage": float_metric(payload, "moe_base_demo_perfect_coverage") if clean else 0.0,
        "coverage_delta_vs_12b": float_metric(payload, "coverage_delta_vs_12b") if clean else 0.0,
        "bootstrap_ci95": list_float_metric(payload, "bootstrap_ci95") if clean else [],
        "local_support_diagnosis": str_metric(payload, "local_support_diagnosis") if clean else "",
        "n_tasks_scored": int_metric(payload, "n_tasks_scored") if clean else 0,
        "measurement_task_floor": MOE_TASK_FLOOR,
        "raw_complete": bool_metric(payload, "raw_complete") if clean else False,
        "gated_pass_at_2": float_metric(payload, "gated_pass_at_2") if clean else 0.0,
        "oracle_coverage": float_metric(payload, "oracle_coverage") if clean else 0.0,
        "local_seconds_per_task": float_metric(payload, "local_seconds_per_task") if clean else 0.0,
        "codex_seconds_per_task_reference": float_metric(payload, "codex_seconds_per_task_reference") if clean else 0.0,
    }


def accuracy_delta(
    archive_payload: Mapping[str, Any] | None,
    accuracy_payload: Mapping[str, Any] | None,
) -> JsonDict:
    """Build the ARC games-solved delta from exp4049 with archive fallback."""

    prior = int_metric(accuracy_payload, "prior_total_games_solved") or nested_int(
        archive_payload,
        ("milestone_373_closestate", "accuracy", "total_games_solved"),
    )
    solved = invoked(accuracy_payload) and bool_metric(accuracy_payload, "game_solved") and bool_metric(
        accuracy_payload,
        "real_env_confirmed",
    )
    total = int_metric(accuracy_payload, "total_games_solved") if solved else prior
    first_action = int_metric(accuracy_payload, "first_solve_at_action") if solved else 0
    baseline_actions = int_metric(accuracy_payload, "candidate_baseline_actions") if solved else 0
    return {
        "prior_total_games_solved": prior,
        "total_games_solved": total,
        "games_solved_delta": max(0, total - prior) if solved else 0,
        "game_solved": solved,
        "target_game": str_metric(accuracy_payload, "target_game") if solved else "",
        "real_env_confirmed": solved,
        "candidate_baseline_actions": baseline_actions,
        "first_solve_at_action": first_action,
        "action_savings_vs_candidate_baseline": max(0, baseline_actions - first_action) if solved else 0,
        "exploration_actions_used": int_metric(accuracy_payload, "exploration_actions_used") if solved else 0,
    }


def self_learning_delta(payload: Mapping[str, Any] | None) -> JsonDict:
    """Build the ArcMemo v7 cross-game transfer deltas from exp4050."""

    clean = invoked(payload)
    cold = int_metric(payload, "actions_cold") if clean else 0
    within = int_metric(payload, "actions_within_game_v6") if clean else 0
    cross = int_metric(payload, "actions_cross_game_v7") if clean else 0
    cold_calls = int_metric(payload, "induction_calls_cold") if clean else 0
    within_calls = int_metric(payload, "induction_calls_within_game_v6") if clean else 0
    cross_calls = int_metric(payload, "induction_calls_cross_game_v7") if clean else 0
    return {
        "cross_game_transfer_win": clean and bool_metric(payload, "cross_game_transfer_win"),
        "actions_cold": cold,
        "actions_within_game_v6": within,
        "actions_cross_game_v7": cross,
        "action_savings_vs_cold": max(0, cold - cross) if clean and cross > 0 else 0,
        "cross_game_extra_actions_vs_within_game_v6": cross - within if clean and cross > 0 and within > 0 else 0,
        "induction_calls_cold": cold_calls,
        "induction_calls_within_game_v6": within_calls,
        "induction_calls_cross_game_v7": cross_calls,
        "induction_call_savings_vs_cold": max(0, cold_calls - cross_calls) if clean else 0,
        "n_prior_fragments": int_metric(payload, "n_prior_fragments") if clean else 0,
        "n_named_abstractions": int_metric(payload, "n_named_abstractions") if clean else 0,
        "n_reused_abstractions": int_metric(payload, "n_reused_abstractions") if clean else 0,
        "transfer_assessment": str_metric(payload, "transfer_assessment") if clean else "",
    }


def hardware_delta(payload: Mapping[str, Any] | None) -> JsonDict:
    """Carry the exp4052 board-continuity state without making a speedup claim."""

    reachability = payload.get("per_board_reachability") if isinstance(payload, Mapping) else None
    terminal = payload.get("per_board_terminal_state") if isinstance(payload, Mapping) else None
    clean = invoked(payload)
    return {
        "included": clean,
        "kv260_overlay_loaded": clean and bool_metric(payload, "kv260_overlay_loaded"),
        "kv260_latency_step_taken": clean and bool_metric(payload, "kv260_latency_step_taken"),
        "kv260_latency_median_ms": float_metric(payload, "kv260_latency_median_ms") if clean else 0.0,
        "kv260_latency_batch_ms": float_metric(payload, "kv260_latency_batch_ms") if clean else 0.0,
        "speedup_claim_made": clean and bool_metric(payload, "speedup_claim_made"),
        "fabric_acceleration_claimed": clean and bool_metric(payload, "fabric_acceleration_claimed"),
        "per_board_reachability": dict(reachability) if isinstance(reachability, Mapping) and clean else {},
        "per_board_terminal_state": dict(terminal) if isinstance(terminal, Mapping) and clean else {},
    }


def hygiene_report(payload: Mapping[str, Any] | None) -> JsonDict:
    """Carry exp4051 registry and verifier-gap hygiene without changing gates."""

    clean = invoked(payload)
    return {
        "included": clean,
        "offline_reeval_bitexact": clean and bool_metric(payload, "offline_reeval_bitexact"),
        "registry_updated": clean and bool_metric(payload, "registry_updated"),
        "gaps_updated": clean and bool_metric(payload, "gaps_updated"),
        "g1_off_arc_outcome_recorded": str_metric(payload, "g1_off_arc_outcome_recorded") if clean else "",
        "g2_closed_loop_outcome_recorded": str_metric(payload, "g2_closed_loop_outcome_recorded") if clean else "",
    }


def flagged_artifacts_skipped(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    summaries: Mapping[int, Mapping[str, Any]],
    skipped_ids: set[int],
) -> list[JsonDict]:
    """Record upstreams excluded before any metric import."""

    rows: list[JsonDict] = []
    for experiment_id in sorted(skipped_ids):
        path = paths[experiment_id]
        reason = "flagged_adversarial:true" if flagged(upstreams[experiment_id]) else "summarize_artifact_live_critical"
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": relative_to_root(root, path) if path is not None else "",
                "reason": reason,
            }
        )
        summaries.get(experiment_id)
    return rows


def cited_upstream_artifacts(paths: Mapping[int, Path | None], clean_ids: set[int]) -> list[JsonDict]:
    """Build the required citation list of included upstream ids and sha256."""

    return [
        {"experiment_id": experiment_id, "sha256": sha256_file(path)}
        for experiment_id in UPSTREAM_IDS
        if experiment_id in clean_ids and (path := paths[experiment_id]) is not None
    ]


def missing_upstream_artifacts(paths: Mapping[int, Path | None]) -> list[JsonDict]:
    """Record missing upstream artifacts without turning absence into a gate."""

    return [{"experiment_id": experiment_id} for experiment_id in UPSTREAM_IDS if paths[experiment_id] is None]


def upstream_artifact_state(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    summaries: Mapping[int, Mapping[str, Any]],
    skipped_ids: set[int],
    clean_ids: set[int],
) -> dict[str, JsonDict]:
    """Expose inclusion state so skipped and missing inputs are auditable."""

    state: dict[str, JsonDict] = {}
    for experiment_id in UPSTREAM_IDS:
        path = paths[experiment_id]
        payload = upstreams[experiment_id]
        state[str(experiment_id)] = {
            "exists": path is not None,
            "path": relative_to_root(root, path) if path is not None else "",
            "honest_verdict": str(payload.get("honest_verdict")) if isinstance(payload, Mapping) else "missing",
            "flagged_adversarial": flagged(payload),
            "live_critical": live_critical(summaries.get(experiment_id)),
            "included": experiment_id in clean_ids,
            "skipped": experiment_id in skipped_ids,
            "summarize_artifact_returncode": summaries.get(experiment_id, {}).get("returncode"),
        }
    return state


def duration_from(started_s: float, now_s: float | None) -> float:
    """Compute an honest aggregation duration with a small nonzero floor."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0001, end - started_s)


def decision_grade_state(g1: Mapping[str, Any], g2: Mapping[str, Any], g3: Mapping[str, Any]) -> JsonDict:
    """Separate measurement quality from whether the result was positive."""

    g1_grade = bool(g1.get("full_power_reached"))
    g2_grade = g2.get("outcome") in {
        "closed_loop_broke_wall",
        "closed_loop_ceiling_saturated_sim2real_divergence",
        "closed_loop_no_break",
    }
    g3_grade = g3.get("outcome") in {"decision_grade_measurement", "uninformative_measurement"}
    return {"G1": g1_grade, "G2": g2_grade, "G3": g3_grade, "all_three": g1_grade and g2_grade and g3_grade}


def verdict(
    *,
    decisions: Mapping[str, Any],
    g1_outcome: str,
    g2_outcome: str,
    g3_diagnosis: str,
    transfer_positive: bool,
    closed_loop_positive: bool,
    total_games_solved: int,
    memory_win: bool,
    skipped_count: int,
) -> str:
    """Build the terminal-prefix .374 headline from measured outcomes."""

    grade_text = "decision_grade_yes" if decisions.get("all_three") is True else "not_decision_grade"
    g1_text = "significant" if transfer_positive else g1_outcome
    g2_text = "salvaged" if closed_loop_positive else g2_outcome
    memory_text = "arcmemo_v7_win" if memory_win else "arcmemo_v7_no_win"
    return (
        f"complete: capstone_v374_{grade_text}_"
        f"G1_{g1_text}_G2_{g2_text}_G3_{g3_diagnosis}_"
        f"games{total_games_solved}_{memory_text}_flagged_skipped{skipped_count}"
    )


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return a reproducibility checksum excluding the checksum field itself."""

    stable = dict(payload)
    stable.pop("reproducibility_checksum", None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    summary_statuses: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the .374 capstone from landed upstream artifacts."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    paths = selected_upstream_paths(root_path)
    summaries = summarize_existing_artifacts(root_path, paths, summary_statuses)
    upstreams: dict[int, Mapping[str, Any] | None] = {
        experiment_id: read_json_object(path) if path is not None else None
        for experiment_id, path in paths.items()
    }
    skipped_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if flagged(payload) or live_critical(summaries.get(experiment_id))
    }
    clean_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and experiment_id not in skipped_ids
    }
    clean_upstreams = {experiment_id: upstreams[experiment_id] for experiment_id in clean_ids}

    g1 = off_arc_transfer_report(clean_upstreams.get(4045), was_skipped=4045 in skipped_ids)
    g2 = closed_loop_grounding_report(clean_upstreams.get(4046), was_skipped=4046 in skipped_ids)
    g3 = decentralization_moe_report(clean_upstreams.get(4048), was_skipped=4048 in skipped_ids)
    accuracy = accuracy_delta(clean_upstreams.get(4042), clean_upstreams.get(4049))
    self_learning = self_learning_delta(clean_upstreams.get(4050))
    hardware = hardware_delta(clean_upstreams.get(4052))
    hygiene = hygiene_report(clean_upstreams.get(4051))
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, summaries, skipped_ids)
    decisions = decision_grade_state(g1, g2, g3)
    transfer_positive = bool(g1["full_power_reached"]) and bool(g1["demofit_ci_excludes_zero"])
    closed_loop_positive = bool(g2["closed_loop_broke_wall"]) and bool(g2["real_env_confirmed"])

    artifact: JsonDict = {
        "schema": "carnot.capstone_v374_4053.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": verdict(
            decisions=decisions,
            g1_outcome=str(g1["outcome"]),
            g2_outcome=str(g2["outcome"]),
            g3_diagnosis=str(g3["diagnosis"]),
            transfer_positive=transfer_positive,
            closed_loop_positive=closed_loop_positive,
            total_games_solved=int(accuracy["total_games_solved"]),
            memory_win=bool(self_learning["cross_game_transfer_win"]),
            skipped_count=len(skipped),
        ),
        "verifier_transferred_off_arc_significantly": transfer_positive,
        "search_layer_salvageable_closed_loop": closed_loop_positive,
        "decentralization_diagnosis": str(g3["diagnosis"]),
        "total_games_solved": int(accuracy["total_games_solved"]),
        "g1_off_arc_transfer": g1,
        "g2_closed_loop_grounding": g2,
        "g3_decentralization_moe_base": g3,
        "decision_grade_measurements": decisions,
        "accuracy_self_learning_hardware_deltas": {
            "accuracy": accuracy,
            "self_learning": self_learning,
            "hardware": hardware,
        },
        "verifier_registry_and_gaps_hygiene": hygiene,
        "flagged_artifacts_skipped": skipped,
        "cited_upstream_artifacts": cited_upstream_artifacts(paths, clean_ids),
        "missing_upstream_artifacts": missing_upstream_artifacts(paths),
        "upstream_artifact_state": upstream_artifact_state(root_path, paths, upstreams, summaries, skipped_ids, clean_ids),
        "summarize_artifact_status": {
            str(experiment_id): {
                "returncode": status.get("returncode"),
                "stdout": status.get("stdout", ""),
                "stderr": status.get("stderr", ""),
            }
            for experiment_id, status in summaries.items()
        },
        "duration_s": duration_from(start, now_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the .374 fields that protect the honest headline."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover - defensive guard.
    verdict_text = str(artifact.get("honest_verdict", ""))
    if not verdict_text.startswith(("complete:", "success:", "blocked_", "blocked:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in ("verifier_transferred_off_arc_significantly", "search_layer_salvageable_closed_loop"):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    allowed_diagnoses = {"latent", "absent", "uninformative", "retired_non_measurement", "flagged_skipped"}
    if artifact.get("decentralization_diagnosis") not in allowed_diagnoses:
        raise ValueError("decentralization_diagnosis must be latent, absent, uninformative, retired, or flagged_skipped")
    if not isinstance(artifact.get("total_games_solved"), int) or isinstance(artifact.get("total_games_solved"), bool):
        raise ValueError("total_games_solved must be a bare int")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    citations = artifact.get("cited_upstream_artifacts")
    if not isinstance(citations, list):
        raise ValueError("cited_upstream_artifacts must be a list")  # pragma: no cover - defensive guard.
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise ValueError("citation entries must be objects")  # pragma: no cover - defensive guard.
        if set(citation) != {"experiment_id", "sha256"}:
            raise ValueError("citation entries must contain experiment_id and sha256")  # pragma: no cover.
        if not isinstance(citation.get("experiment_id"), int):
            raise ValueError("citation entries need integer experiment_id")  # pragma: no cover - defensive guard.
        if not is_sha256(citation.get("sha256")):
            raise ValueError("citation entries need sha256")
    if not isinstance(artifact.get("flagged_artifacts_skipped"), list):
        raise ValueError("flagged_artifacts_skipped must be a list")
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be sha256")  # pragma: no cover - defensive guard.


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    summary_statuses: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the Exp 4053 capstone artifact."""

    root_path = Path(root)
    artifact = build_artifact(
        root_path,
        summary_statuses=summary_statuses,
        started_s=started_s,
        now_s=now_s,
    )
    validate_artifact(artifact)
    output = root_path / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output

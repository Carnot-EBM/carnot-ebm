"""Build the Exp 4065 v375 capstone aggregation.

Spec refs: REQ-CAPSTONE-4065, SCENARIO-CAPSTONE-4065.

The .375 milestone keeps resume-not-restart accounting explicit. A short-N
result is an accumulating measurement state, not a retirement, and flagged
upstream artifacts are excluded before any metric is imported.
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
OUTPUT_REL_PATH = Path("results/experiment_4065_capstone_v375.json")
EXPERIMENT_ID = 4065
RANDOM_SEED = 4065
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PYTHON_BIN = Path(".venv/bin/python")
BASELINE_12B_COVERAGE = 0.2581
DEFAULT_G3_TARGET_N = 30

UPSTREAM_IDS = tuple(range(4054, 4065))
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4054: Path("results/experiment_4054_archive_v374_activate_v375.json"),
    4055: Path("results/experiment_4055_sota_ingestion_receipt.json"),
    4056: Path("results/experiment_4056_offarc_power_evalplus_build.json"),
    4057: Path("results/experiment_4057_offarc_power_evalplus.json"),
    4058: Path("results/experiment_4058_offarc_power_evalplus_resume.json"),
    4059: Path("results/experiment_4059_decentralization_moe_resume.json"),
    4060: Path("results/experiment_4060_ninth_game_explore_first.json"),
    4061: Path("results/experiment_4061_verifier_action_pruner_efficiency.json"),
    4062: Path("results/experiment_4062_arcmemo_cross_game_transfer_v8.json"),
    4063: Path("results/experiment_4063_verifier_registry_and_gaps_hygiene.json"),
    4064: Path("results/experiment_4064_hardware_continuity.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verifier_transferred_off_arc_significantly",
    "off_arc_accumulated_n",
    "decentralization_diagnosis",
    "verifier_pruner_efficiency_gain",
    "total_games_solved",
    "flagged_artifacts_skipped",
    "cited_upstream_artifacts",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefix verdict preserving positive, negative, uninformative, skipped, and accumulating outcomes."
    ),
    "verifier_transferred_off_arc_significantly": (
        "BARE BOOL - the operator TOP-PRIORITY question: does the EvalPlus off-ARC best-arm CI exclude zero "
        "with oracle headroom at accumulated N?"
    ),
    "off_arc_accumulated_n": "BARE INT - resume-not-restart progress accumulated for EvalPlus off-ARC transfer.",
    "decentralization_diagnosis": (
        "latent | absent | accumulating | uninformative | flagged_skipped - the sovereign-base branch state."
    ),
    "verifier_pruner_efficiency_gain": (
        "BARE BOOL - the efficient-axis datum: action reduction is positive and solverate parity held."
    ),
    "total_games_solved": "BARE INT - the monotonic ARC accuracy counter after clean .375 evidence.",
    "flagged_artifacts_skipped": "Upstreams excluded before metric import because they are stamped flagged_adversarial:true.",
    "cited_upstream_artifacts": "Included upstream experiment ids and sha256 provenance only.",
    "inference_substrate": "Declares this capstone as aggregation from upstream artifacts.",
}


def is_sha256(value: object) -> bool:
    """Return true when a value is a lowercase SHA-256 hex digest."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def read_json_object(path: Path) -> JsonDict:
    """Load a JSON object artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} did not contain a JSON object"
        )  # pragma: no cover - defensive guard.
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


def _fallback_path(root: Path, experiment_id: int) -> Path | None:
    """Find a single top-level result artifact if the default path is absent."""

    matches = sorted((root / "results").glob(f"experiment_{experiment_id}_*.json"))
    return matches[0] if matches else None


def selected_upstream_paths(root: Path | str) -> dict[int, Path | None]:
    """Select the intended artifact for each .375 upstream id."""

    root_path = Path(root)
    paths: dict[int, Path | None] = {}
    for experiment_id in UPSTREAM_IDS:
        default = root_path / DEFAULT_UPSTREAM_PATHS[experiment_id]
        paths[experiment_id] = (
            default if default.exists() else _fallback_path(root_path, experiment_id)
        )
    return paths


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
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


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
    """Return false for missing or blocked upstream artifacts."""

    verdict = str(payload.get("honest_verdict", "")) if isinstance(payload, Mapping) else ""
    return (
        bool(verdict)
        and not verdict.startswith(("blocked_", "blocked:"))
        and "pending_execution" not in verdict
    )


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
    return [
        float(item)
        for item in value
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]


def nested_mapping(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> Mapping[str, Any]:
    """Read a nested object and return an empty mapping if any hop is absent."""

    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def nested_int(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> int:
    """Read a nested integer fallback without accepting booleans as counters."""

    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return 0
        current = current.get(key)
    return current if isinstance(current, int) and not isinstance(current, bool) else 0


def _empty_g1(status: str) -> JsonDict:
    return {
        "status": status,
        "accumulated_n": 0,
        "powered_task_floor": 0,
        "decision_grade": False,
        "best_arm": "",
        "best_arm_delta_pp": 0.0,
        "best_arm_ci95": [],
        "best_arm_ci_excludes_zero": False,
        "excludes_zero_with_headroom": False,
        "demofit_delta_pp": 0.0,
        "demofit_bootstrap_ci95": [],
        "demofit_ci_excludes_zero": False,
        "oracle_passrate": 0.0,
        "oracle_headroom_present": False,
        "raw_artifact_present": False,
        "partial_reason": "",
    }


def g1_off_arc_report(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Build the G1 EvalPlus off-ARC transfer report from exp4057."""

    if was_skipped:
        return _empty_g1("skipped_flagged")
    if not invoked(payload):
        return _empty_g1("missing_or_blocked")

    accumulated_n = int_metric(payload, "accumulated_n_tasks") or int_metric(payload, "n_tasks")
    powered_floor = int_metric(payload, "powered_task_floor")
    best_delta = float_metric(payload, "best_arm_delta_pp")
    best_excludes = bool_metric(payload, "best_arm_ci_excludes_zero")
    headroom = bool_metric(payload, "oracle_headroom_present") or bool_metric(
        payload, "oracle_headroom"
    )
    full_n = powered_floor > 0 and accumulated_n >= powered_floor
    transferred = best_excludes and best_delta > 0.0 and headroom

    if transferred:
        status = "excludes_zero_with_headroom"
    elif not full_n:
        status = f"accumulating_n_{accumulated_n}"
    else:
        status = "uninformative_or_not_significant"

    return {
        "status": status,
        "accumulated_n": accumulated_n,
        "powered_task_floor": powered_floor,
        "decision_grade": full_n and headroom,
        "best_arm": str_metric(payload, "best_arm"),
        "best_arm_delta_pp": best_delta,
        "best_arm_ci95": list_float_metric(payload, "best_arm_ci95"),
        "best_arm_ci_excludes_zero": best_excludes,
        "excludes_zero_with_headroom": transferred,
        "demofit_delta_pp": float_metric(payload, "demofit_delta_pp"),
        "demofit_bootstrap_ci95": list_float_metric(payload, "demofit_bootstrap_ci95"),
        "demofit_ci_excludes_zero": bool_metric(payload, "demofit_ci_excludes_zero"),
        "oracle_passrate": float_metric(payload, "oracle_passrate"),
        "oracle_headroom_present": headroom,
        "raw_artifact_present": bool_metric(payload, "raw_artifact_present"),
        "partial_reason": str_metric(payload, "partial_reason"),
    }


def _empty_g3(status: str, diagnosis: str) -> JsonDict:
    return {
        "status": status,
        "diagnosis": diagnosis,
        "source": "",
        "baseline_12b_coverage": BASELINE_12B_COVERAGE,
        "accumulated_coverage": 0.0,
        "coverage_delta_vs_12b": 0.0,
        "bootstrap_ci95": [],
        "accumulated_n": 0,
        "target_task_floor": DEFAULT_G3_TARGET_N,
        "decision_grade": False,
        "raw_complete": False,
        "local_support_diagnosis": "",
    }


def _g3_from_clean_4059(payload: Mapping[str, Any]) -> JsonDict:
    accumulated_n = (
        int_metric(payload, "accumulated_n")
        or int_metric(payload, "n_tasks_scored")
        or int_metric(payload, "accumulated_n_tasks")
    )
    target = int_metric(payload, "target_task_floor") or DEFAULT_G3_TARGET_N
    coverage = (
        float_metric(payload, "accumulated_coverage")
        or float_metric(payload, "moe_base_demo_perfect_coverage")
        or float_metric(payload, "coverage")
    )
    delta = float_metric(payload, "coverage_delta_vs_12b") or (
        coverage - BASELINE_12B_COVERAGE if coverage else 0.0
    )
    observed = (
        str_metric(payload, "local_support_diagnosis")
        or str_metric(payload, "decentralization_diagnosis")
        or str_metric(payload, "diagnosis")
    )

    if observed == "uninformative":
        diagnosis = "uninformative"
        status = "uninformative"
    elif accumulated_n < target:
        diagnosis = "accumulating"
        status = "accumulating"
    elif observed in {"latent", "absent"}:
        diagnosis = observed
        status = "decision_grade"
    else:
        diagnosis = "latent" if coverage > BASELINE_12B_COVERAGE else "absent"
        status = "decision_grade"

    return {
        "status": status,
        "diagnosis": diagnosis,
        "source": "exp4059",
        "baseline_12b_coverage": BASELINE_12B_COVERAGE,
        "accumulated_coverage": coverage,
        "coverage_delta_vs_12b": delta,
        "bootstrap_ci95": list_float_metric(payload, "bootstrap_ci95"),
        "accumulated_n": accumulated_n,
        "target_task_floor": target,
        "decision_grade": status == "decision_grade" and diagnosis in {"latent", "absent"},
        "raw_complete": bool_metric(payload, "raw_complete"),
        "local_support_diagnosis": observed,
    }


def _g3_from_activation_checkpoint(activation_payload: Mapping[str, Any] | None) -> JsonDict:
    g3 = nested_mapping(
        activation_payload, ("milestone_374_closestate", "g3_decentralization_moe_base")
    )
    if not g3:
        return _empty_g3("missing_or_blocked", "accumulating")

    accumulated_n = int_metric(g3, "checkpoint_n_tasks") or int_metric(g3, "accumulated_n")
    target = int_metric(g3, "target_task_floor") or DEFAULT_G3_TARGET_N
    coverage = float_metric(g3, "moe_base_coverage") or float_metric(g3, "accumulated_coverage")
    delta = coverage - BASELINE_12B_COVERAGE if coverage else 0.0
    diagnosis = (
        "accumulating"
        if accumulated_n < target
        else "latent"
        if coverage > BASELINE_12B_COVERAGE
        else "absent"
    )

    return {
        "status": "accumulating_from_resume_checkpoint"
        if accumulated_n < target
        else "decision_grade_from_checkpoint",
        "diagnosis": diagnosis,
        "source": "exp4054_resume_checkpoint",
        "baseline_12b_coverage": BASELINE_12B_COVERAGE,
        "accumulated_coverage": coverage,
        "coverage_delta_vs_12b": delta,
        "bootstrap_ci95": list_float_metric(g3, "bootstrap_ci95"),
        "accumulated_n": accumulated_n,
        "target_task_floor": target,
        "decision_grade": accumulated_n >= target and diagnosis in {"latent", "absent"},
        "raw_complete": False,
        "local_support_diagnosis": str_metric(g3, "operator_corrected_diagnosis"),
    }


def g3_decentralization_report(
    payload: Mapping[str, Any] | None,
    activation_payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    """Build the G3 MoE resume report from exp4059 or the resume checkpoint."""

    if was_skipped:
        return _empty_g3("skipped_flagged", "flagged_skipped")
    if invoked(payload):
        return _g3_from_clean_4059(payload)
    return _g3_from_activation_checkpoint(activation_payload)


def efficiency_report(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Build the verifier-as-action-pruner efficiency report from exp4061."""

    if was_skipped:
        status = "skipped_flagged"
        clean = False
    else:
        clean = invoked(payload)
        status = "measured" if clean else "missing_or_blocked"

    baseline_actions = int_metric(payload, "baseline_actions") if clean else 0
    pruned_actions = int_metric(payload, "pruned_actions") if clean else 0
    reduction = float_metric(payload, "action_reduction_pct") if clean else 0.0
    if reduction == 0.0 and baseline_actions > 0 and pruned_actions >= 0:
        reduction = max(0.0, (baseline_actions - pruned_actions) * 100.0 / baseline_actions)
    parity = clean and bool_metric(payload, "solverate_parity_held")
    gain = clean and parity and reduction > 0.0

    return {
        "status": "gain" if gain else status if not clean else "no_gain_or_no_parity",
        "action_reduction_pct": reduction,
        "solverate_parity_held": parity,
        "efficiency_gain": gain,
        "baseline_solverate": float_metric(payload, "baseline_solverate") if clean else 0.0,
        "pruned_solverate": float_metric(payload, "pruned_solverate") if clean else 0.0,
        "baseline_actions": baseline_actions,
        "pruned_actions": pruned_actions,
    }


def accuracy_delta(
    activation_payload: Mapping[str, Any] | None,
    accuracy_payload: Mapping[str, Any] | None,
) -> JsonDict:
    """Build the ARC games-solved delta from exp4060 with activation fallback."""

    prior = int_metric(accuracy_payload, "prior_total_games_solved") or nested_int(
        activation_payload,
        ("milestone_374_closestate", "accuracy", "total_games_solved"),
    )
    clean = invoked(accuracy_payload)
    solved = (
        clean
        and bool_metric(accuracy_payload, "game_solved")
        and bool_metric(accuracy_payload, "real_env_confirmed")
    )
    total = int_metric(accuracy_payload, "total_games_solved") if solved else prior
    first_action = int_metric(accuracy_payload, "first_solve_at_action") if solved else 0
    baseline_actions = int_metric(accuracy_payload, "candidate_baseline_actions") if solved else 0
    return {
        "status": "new_game_solved"
        if solved
        else "no_new_solve"
        if clean
        else "missing_or_blocked",
        "prior_total_games_solved": prior,
        "total_games_solved": total,
        "games_solved_delta": max(0, total - prior) if solved else 0,
        "game_solved": solved,
        "target_game": str_metric(accuracy_payload, "target_game") if solved else "",
        "real_env_confirmed": solved,
        "candidate_baseline_actions": baseline_actions,
        "first_solve_at_action": first_action,
        "action_savings_vs_candidate_baseline": max(0, baseline_actions - first_action)
        if solved
        else 0,
        "exploration_actions_used": int_metric(accuracy_payload, "exploration_actions_used")
        if solved
        else 0,
    }


def self_learning_delta(payload: Mapping[str, Any] | None) -> JsonDict:
    """Build the ArcMemo v8 cross-game transfer deltas from exp4062."""

    clean = invoked(payload)
    cold = int_metric(payload, "actions_cold") if clean else 0
    within = int_metric(payload, "actions_within_game") if clean else 0
    cross = int_metric(payload, "actions_cross_game_v8") if clean else 0
    cold_calls = int_metric(payload, "induction_calls_cold") if clean else 0
    within_calls = int_metric(payload, "induction_calls_within_game") if clean else 0
    cross_calls = int_metric(payload, "induction_calls_cross_game_v8") if clean else 0
    return {
        "status": "measured" if clean else "missing_or_blocked",
        "cross_game_transfer_win": clean and bool_metric(payload, "cross_game_transfer_win"),
        "actions_cold": cold,
        "actions_within_game": within,
        "actions_cross_game_v8": cross,
        "action_savings_vs_cold": max(0, cold - cross) if clean and cross > 0 else 0,
        "cross_game_extra_actions_vs_within_game": cross - within
        if clean and cross > 0 and within > 0
        else 0,
        "induction_calls_cold": cold_calls,
        "induction_calls_within_game": within_calls,
        "induction_calls_cross_game_v8": cross_calls,
        "induction_call_savings_vs_cold": max(0, cold_calls - cross_calls) if clean else 0,
        "n_prior_fragments": int_metric(payload, "n_prior_fragments") if clean else 0,
        "n_named_abstractions": int_metric(payload, "n_named_abstractions") if clean else 0,
        "n_reused_abstractions": int_metric(payload, "n_reused_abstractions") if clean else 0,
        "transfer_assessment": str_metric(payload, "transfer_assessment") if clean else "",
    }


def hardware_delta(payload: Mapping[str, Any] | None) -> JsonDict:
    """Carry the exp4064 board-continuity state without making a speedup claim."""

    reachability = payload.get("per_board_reachability") if isinstance(payload, Mapping) else None
    terminal = payload.get("per_board_terminal_state") if isinstance(payload, Mapping) else None
    clean = invoked(payload)
    return {
        "included": clean,
        "kv260_terminal_confirmed": clean and bool_metric(payload, "kv260_terminal_confirmed"),
        "kv260_step_taken": str_metric(payload, "kv260_step_taken") if clean else "",
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken") if clean else "",
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken") if clean else "",
        "speedup_claim_made": clean and bool_metric(payload, "speedup_claim_made"),
        "fabric_acceleration_claimed": clean
        and bool_metric(payload, "fabric_acceleration_claimed"),
        "per_board_reachability": dict(reachability)
        if isinstance(reachability, Mapping) and clean
        else {},
        "per_board_terminal_state": dict(terminal)
        if isinstance(terminal, Mapping) and clean
        else {},
    }


def hygiene_report(payload: Mapping[str, Any] | None) -> JsonDict:
    """Carry exp4063 registry and verifier-gap hygiene without changing gates."""

    clean = invoked(payload)
    return {
        "included": clean,
        "offline_reeval_bitexact": clean and bool_metric(payload, "offline_reeval_bitexact"),
        "registry_updated": clean and bool_metric(payload, "registry_updated"),
        "gaps_updated": clean and bool_metric(payload, "gaps_updated"),
        "g1_off_arc_outcome_recorded": str_metric(payload, "g1_off_arc_outcome_recorded")
        if clean
        else "",
        "g3_decentralization_outcome_recorded": str_metric(
            payload, "g3_decentralization_outcome_recorded"
        )
        if clean
        else "",
    }


def flagged_artifacts_skipped(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    skipped_ids: set[int],
) -> list[JsonDict]:
    """Record upstreams excluded before any metric import."""

    rows: list[JsonDict] = []
    for experiment_id in sorted(skipped_ids):
        path = paths[experiment_id]
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": relative_to_root(root, path) if path is not None else "",
                "reason": "flagged_adversarial:true"
                if flagged(upstreams[experiment_id])
                else "unknown",
            }
        )
    return rows


def cited_upstream_artifacts(
    paths: Mapping[int, Path | None], clean_ids: set[int]
) -> list[JsonDict]:
    """Build the required citation list of included upstream ids and sha256."""

    return [
        {"experiment_id": experiment_id, "sha256": sha256_file(path)}
        for experiment_id in UPSTREAM_IDS
        if experiment_id in clean_ids and (path := paths[experiment_id]) is not None
    ]


def missing_upstream_artifacts(paths: Mapping[int, Path | None]) -> list[JsonDict]:
    """Record missing upstream artifacts without turning absence into a gate."""

    return [
        {"experiment_id": experiment_id}
        for experiment_id in UPSTREAM_IDS
        if paths[experiment_id] is None
    ]


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
            "honest_verdict": str(payload.get("honest_verdict"))
            if isinstance(payload, Mapping)
            else "missing",
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


def verdict(
    *,
    g1: Mapping[str, Any],
    g3: Mapping[str, Any],
    efficiency_gain: bool,
    total_games_solved: int,
    skipped_count: int,
) -> str:
    """Build the terminal-prefix .375 headline from measured outcomes."""

    accumulated_n = int(g1.get("accumulated_n", 0))
    powered_floor = int(g1.get("powered_task_floor", 0))
    if g1.get("excludes_zero_with_headroom") is True:
        g1_text = "excl0"
    elif powered_floor > 0 and accumulated_n < powered_floor:
        g1_text = f"accumulating_n{accumulated_n}"
    else:
        g1_text = "not_excl0"
    efficiency_text = "gain" if efficiency_gain else "null"
    return (
        f"complete: capstone_v375_offarc_{g1_text}_g3_{g3.get('diagnosis')}_"
        f"efficiency_{efficiency_text}_games{total_games_solved}_flagged_skipped{skipped_count}"
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
    """Build the .375 capstone from landed upstream artifacts."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    paths = selected_upstream_paths(root_path)
    summaries = summarize_existing_artifacts(root_path, paths, summary_statuses)
    upstreams: dict[int, Mapping[str, Any] | None] = {
        experiment_id: read_json_object(path) if path is not None else None
        for experiment_id, path in paths.items()
    }
    skipped_ids = {
        experiment_id for experiment_id, payload in upstreams.items() if flagged(payload)
    }
    clean_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and experiment_id not in skipped_ids
    }
    clean_upstreams = {experiment_id: upstreams[experiment_id] for experiment_id in clean_ids}

    activation = clean_upstreams.get(4054)
    g1 = g1_off_arc_report(clean_upstreams.get(4057), was_skipped=4057 in skipped_ids)
    g3 = g3_decentralization_report(
        clean_upstreams.get(4059), activation, was_skipped=4059 in skipped_ids
    )
    efficiency = efficiency_report(clean_upstreams.get(4061), was_skipped=4061 in skipped_ids)
    accuracy = accuracy_delta(activation, clean_upstreams.get(4060))
    self_learning = self_learning_delta(clean_upstreams.get(4062))
    hardware = hardware_delta(clean_upstreams.get(4064))
    hygiene = hygiene_report(clean_upstreams.get(4063))
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    efficiency_gain = bool(efficiency["efficiency_gain"])

    artifact: JsonDict = {
        "schema": "carnot.capstone_v375_4065.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": verdict(
            g1=g1,
            g3=g3,
            efficiency_gain=efficiency_gain,
            total_games_solved=int(accuracy["total_games_solved"]),
            skipped_count=len(skipped),
        ),
        "verifier_transferred_off_arc_significantly": bool(g1["excludes_zero_with_headroom"]),
        "off_arc_accumulated_n": int(g1["accumulated_n"]),
        "decentralization_diagnosis": str(g3["diagnosis"]),
        "verifier_pruner_efficiency_gain": efficiency_gain,
        "total_games_solved": int(accuracy["total_games_solved"]),
        "g1_off_arc_evalplus": g1,
        "g3_decentralization_moe_resume": g3,
        "efficiency_action_pruner": efficiency,
        "decision_grade_measurements": {
            "G1": bool(g1["decision_grade"]),
            "G3": bool(g3["decision_grade"]),
            "G1_and_G3": bool(g1["decision_grade"]) and bool(g3["decision_grade"]),
        },
        "accuracy_self_learning_hardware_deltas": {
            "accuracy": accuracy,
            "self_learning": self_learning,
            "hardware": hardware,
        },
        "verifier_registry_and_gaps_hygiene": hygiene,
        "flagged_artifacts_skipped": skipped,
        "cited_upstream_artifacts": cited_upstream_artifacts(paths, clean_ids),
        "missing_upstream_artifacts": missing_upstream_artifacts(paths),
        "upstream_artifact_state": upstream_artifact_state(
            root_path, paths, upstreams, summaries, skipped_ids, clean_ids
        ),
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
    """Validate the .375 fields that protect the honest headline."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(
            f"missing required fields: {missing}"
        )  # pragma: no cover - defensive guard.
    verdict_text = str(artifact.get("honest_verdict", ""))
    if not verdict_text.startswith(("complete:", "success:", "blocked_", "blocked:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in ("verifier_transferred_off_arc_significantly", "verifier_pruner_efficiency_gain"):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in ("off_arc_accumulated_n", "total_games_solved"):
        if not isinstance(artifact.get(field), int) or isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare int")
    allowed_diagnoses = {"latent", "absent", "accumulating", "uninformative", "flagged_skipped"}
    if artifact.get("decentralization_diagnosis") not in allowed_diagnoses:
        raise ValueError(
            "decentralization_diagnosis must be latent, absent, accumulating, uninformative, or flagged_skipped"
        )
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    citations = artifact.get("cited_upstream_artifacts")
    if not isinstance(citations, list):
        raise ValueError(
            "cited_upstream_artifacts must be a list"
        )  # pragma: no cover - defensive guard.
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise ValueError(
                "citation entries must be objects"
            )  # pragma: no cover - defensive guard.
        if set(citation) != {"experiment_id", "sha256"}:
            raise ValueError(
                "citation entries must contain experiment_id and sha256"
            )  # pragma: no cover.
        if not isinstance(citation.get("experiment_id"), int):
            raise ValueError(
                "citation entries need integer experiment_id"
            )  # pragma: no cover - defensive guard.
        if not is_sha256(citation.get("sha256")):
            raise ValueError("citation entries need sha256")
    if not isinstance(artifact.get("flagged_artifacts_skipped"), list):
        raise ValueError("flagged_artifacts_skipped must be a list")
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError(
            "reproducibility_checksum must be sha256"
        )  # pragma: no cover - defensive guard.


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    summary_statuses: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the Exp 4065 capstone artifact."""

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

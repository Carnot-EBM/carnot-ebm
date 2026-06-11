"""Build the Exp 4041 v373 argument-measurement capstone.

Spec refs: REQ-CAPSTONE-4041, SCENARIO-CAPSTONE-4041.

The .373 milestone was not a search for a flattering headline. It was a
measurement pass over three open arguments: whether the execution verifier
transfers off ARC, whether the search layer generalizes past the r11l win, and
whether a stronger sovereign base exposes latent local support. This module
keeps that distinction explicit by aggregating only landed upstream artifacts
and by preserving negative or underpowered outcomes as real results.
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
OUTPUT_REL_PATH = Path("results/experiment_4041_capstone_v373.json")
EXPERIMENT_ID = 4041
RANDOM_SEED = 4041
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PYTHON_BIN = Path(".venv/bin/python")
BASELINE_12B_COVERAGE = 0.2581

UPSTREAM_IDS = tuple(range(4029, 4041))
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4029: Path("results/experiment_4029_archive_v372_activate_v373.json"),
    4030: Path("results/experiment_4030_sota_ingestion_receipt.json"),
    4031: Path("results/experiment_4031_offarc_transfer_build.json"),
    4032: Path("results/experiment_4032_offarc_exec_verifier_transfer_collect.json"),
    4033: Path("results/experiment_4033_verifier_registry_harness_registration.json"),
    4034: Path("results/experiment_4034_vc33_goal_predicate_induction.json"),
    4035: Path("results/experiment_4035_hierarchical_search_over_vc33_wm.json"),
    4036: Path("results/experiment_4036_decentralization_stronger_base_build.json"),
    4037: Path("results/experiment_4037_decentralization_stronger_base.json"),
    4038: Path("results/experiment_4038_seventh_game_explore_first.json"),
    4039: Path("results/experiment_4039_arcmemo_concept_library_v6.json"),
    4040: Path("results/experiment_4040_hardware_continuity.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verifier_generalized_off_arc",
    "search_layer_generalized",
    "decentralization_diagnosis",
    "total_games_solved",
    "flagged_artifacts_skipped",
    "cited_upstream_artifacts",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix verdict naming measured, negative, underpowered, and skipped outcomes.",
    "verifier_generalized_off_arc": (
        "BARE BOOL - the operator TOP-PRIORITY question: did the GAP-4 primitive measurably transfer to code?"
    ),
    "search_layer_generalized": (
        "BARE BOOL - the G2 depth question: did a general heuristic break a second game's wall?"
    ),
    "decentralization_diagnosis": (
        "latent | absent | partial | flagged_skipped - the sovereign-base branch G3 resolved."
    ),
    "total_games_solved": "BARE INT - the monotonic ARC accuracy counter after clean exp4038 evidence.",
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
    """Select the one intended final artifact for each .373 upstream id."""

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
    """Build the G1 report from the clean exp4032 collect artifact."""

    if was_skipped:
        return {
            "outcome": "skipped_flagged",
            "n_tasks": 0,
            "delta_pp": 0.0,
            "bootstrap_ci95_pp": [],
            "ci_excludes_zero": False,
            "positive_control_passes": False,
            "three_outcome_verdict": "",
        }
    if not invoked(payload):
        return {
            "outcome": "missing_or_blocked",
            "n_tasks": 0,
            "delta_pp": 0.0,
            "bootstrap_ci95_pp": [],
            "ci_excludes_zero": False,
            "positive_control_passes": False,
            "three_outcome_verdict": "",
        }

    delta = float_metric(payload, "delta_pp")
    ci_excludes_zero = bool_metric(payload, "ci_excludes_zero")
    positive_control = bool_metric(payload, "positive_control_passes")
    confirmed = positive_control and delta > 0.0 and ci_excludes_zero
    outcome = "confirmed_ci_excludes_zero" if confirmed else "no_confirmed_transfer"
    if positive_control and delta > 0.0 and not ci_excludes_zero:
        outcome = "directional_underpowered_ci_touches_zero"

    return {
        "outcome": outcome,
        "n_tasks": int_metric(payload, "n_tasks"),
        "delta_pp": delta,
        "bootstrap_ci95_pp": list_float_metric(payload, "bootstrap_ci95_pp"),
        "ci_excludes_zero": ci_excludes_zero,
        "positive_control_passes": positive_control,
        "three_outcome_verdict": str_metric(payload, "three_outcome_verdict"),
    }


def search_generalization_report(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Build the G2 report from the clean exp4035 vc33 search artifact."""

    clean = invoked(payload) and not was_skipped
    return {
        "outcome": "skipped_flagged" if was_skipped else ("included" if clean else "missing_or_blocked"),
        "game": str_metric(payload, "game") if clean else "",
        "search_layer_generalizes": clean and bool_metric(payload, "search_layer_generalizes"),
        "heuristic_was_non_bespoke": clean and bool_metric(payload, "heuristic_was_non_bespoke"),
        "nodes_expanded": int_metric(payload, "nodes_expanded") if clean else 0,
        "search_found_plan": clean and bool_metric(payload, "search_found_plan"),
        "real_env_confirmed": clean and bool_metric(payload, "real_env_confirmed"),
        "levels_completed_after": int_metric(payload, "levels_completed_after") if clean else 0,
        "new_levels_solved_this_task": int_metric(payload, "new_levels_solved_this_task") if clean else 0,
        "goal_predicate_heldout_precision": float_metric(payload, "goal_predicate_heldout_precision") if clean else 0.0,
    }


def decentralization_scaling_report(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Build the G3 stronger-base coverage report from exp4037."""

    if was_skipped:
        diagnosis = "flagged_skipped"
        clean = False
    else:
        clean = invoked(payload)
        observed = str_metric(payload, "local_support_diagnosis") if clean else ""
        coverage = float_metric(payload, "stronger_base_demo_perfect_coverage") if clean else 0.0
        diagnosis = observed if observed in {"latent", "absent", "partial"} else ("latent" if coverage > BASELINE_12B_COVERAGE else "partial")
        if clean and coverage == 0.0:
            diagnosis = "absent"

    return {
        "diagnosis": diagnosis,
        "baseline_12b_coverage": BASELINE_12B_COVERAGE,
        "stronger_base_demo_perfect_coverage": float_metric(payload, "stronger_base_demo_perfect_coverage") if clean else 0.0,
        "coverage_delta_vs_12b": float_metric(payload, "coverage_delta_vs_12b") if clean else 0.0,
        "gated_pass_at_2": float_metric(payload, "gated_pass_at_2") if clean else 0.0,
        "n_tasks_scored": int_metric(payload, "n_tasks_scored") if clean else 0,
        "local_seconds_per_task": float_metric(payload, "local_seconds_per_task") if clean else 0.0,
    }


def accuracy_delta(
    archive_payload: Mapping[str, Any] | None,
    accuracy_payload: Mapping[str, Any] | None,
) -> JsonDict:
    """Build the ARC games-solved delta from exp4038 with archive fallback."""

    prior = int_metric(accuracy_payload, "prior_total_games_solved") or nested_int(
        archive_payload,
        ("milestone_372_closestate", "arc3", "total_games_solved"),
    )
    solved = invoked(accuracy_payload) and bool_metric(accuracy_payload, "game_solved") and bool_metric(
        accuracy_payload,
        "real_env_confirmed",
    )
    total = int_metric(accuracy_payload, "total_games_solved") if solved else prior
    return {
        "prior_total_games_solved": prior,
        "total_games_solved": total,
        "games_solved_delta": max(0, total - prior) if solved else 0,
        "game_solved": solved,
        "target_game": str_metric(accuracy_payload, "target_game") if solved else "",
        "real_env_confirmed": solved,
        "candidate_baseline_actions": int_metric(accuracy_payload, "candidate_baseline_actions") if solved else 0,
        "first_solve_at_action": int_metric(accuracy_payload, "first_solve_at_action") if solved else 0,
    }


def self_learning_delta(payload: Mapping[str, Any] | None) -> JsonDict:
    """Build the ArcMemo v6 action and induction-call deltas from exp4039."""

    clean = invoked(payload)
    cold = int_metric(payload, "actions_cold") if clean else 0
    v5 = int_metric(payload, "actions_v5") if clean else 0
    v6 = int_metric(payload, "actions_v6") if clean else 0
    cold_calls = int_metric(payload, "induction_calls_cold") if clean else 0
    v6_calls = int_metric(payload, "induction_calls_v6") if clean else 0
    win = clean and bool_metric(payload, "solve_transfer_win")
    return {
        "solve_transfer_win": win,
        "actions_cold": cold,
        "actions_v5": v5,
        "actions_v6": v6,
        "action_savings_vs_cold": max(0, cold - v6) if win else 0,
        "v6_action_savings_vs_v5": max(0, v5 - v6) if win else 0,
        "induction_calls_cold": cold_calls,
        "induction_calls_v5": int_metric(payload, "induction_calls_v5") if clean else 0,
        "induction_calls_v6": v6_calls,
        "induction_call_savings_vs_cold": max(0, cold_calls - v6_calls) if win else 0,
        "n_named_abstractions": int_metric(payload, "n_named_abstractions") if clean else 0,
    }


def hardware_delta(payload: Mapping[str, Any] | None) -> JsonDict:
    """Carry the exp4040 board-continuity state without making a speedup claim."""

    reachability = payload.get("per_board_reachability") if isinstance(payload, Mapping) else None
    terminal = payload.get("per_board_terminal_state") if isinstance(payload, Mapping) else None
    clean = invoked(payload)
    return {
        "included": clean,
        "kv260_overlay_loaded": clean and bool_metric(payload, "kv260_overlay_loaded"),
        "kv260_latency_step_taken": clean and bool_metric(payload, "kv260_latency_step_taken"),
        "speedup_claim_made": clean and bool_metric(payload, "speedup_claim_made"),
        "fabric_acceleration_claimed": clean and bool_metric(payload, "fabric_acceleration_claimed"),
        "per_board_reachability": dict(reachability) if isinstance(reachability, Mapping) and clean else {},
        "per_board_terminal_state": dict(terminal) if isinstance(terminal, Mapping) and clean else {},
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


def verdict(
    *,
    g1_outcome: str,
    g2_generalized: bool,
    g3_diagnosis: str,
    total_games_solved: int,
    memory_win: bool,
    skipped_count: int,
) -> str:
    """Build the terminal-prefix .373 headline from measured outcomes."""

    g2_text = "generalized" if g2_generalized else "no_generalization"
    memory_text = "arcmemo_win" if memory_win else "arcmemo_no_win"
    return (
        "complete: capstone_v373_arguments_measured_"
        f"G1_{g1_outcome}_G2_{g2_text}_G3_{g3_diagnosis}_"
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
    """Build the .373 capstone from landed upstream artifacts."""

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

    g1 = off_arc_transfer_report(clean_upstreams.get(4032), was_skipped=4032 in skipped_ids)
    g2 = search_generalization_report(clean_upstreams.get(4035), was_skipped=4035 in skipped_ids)
    g3 = decentralization_scaling_report(clean_upstreams.get(4037), was_skipped=4037 in skipped_ids)
    accuracy = accuracy_delta(clean_upstreams.get(4029), clean_upstreams.get(4038))
    self_learning = self_learning_delta(clean_upstreams.get(4039))
    hardware = hardware_delta(clean_upstreams.get(4040))
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, summaries, skipped_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v373_4041.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": verdict(
            g1_outcome=str(g1["outcome"]),
            g2_generalized=bool(g2["search_layer_generalizes"]),
            g3_diagnosis=str(g3["diagnosis"]),
            total_games_solved=int(accuracy["total_games_solved"]),
            memory_win=bool(self_learning["solve_transfer_win"]),
            skipped_count=len(skipped),
        ),
        "verifier_generalized_off_arc": g1["outcome"] == "confirmed_ci_excludes_zero",
        "search_layer_generalized": bool(g2["search_layer_generalizes"]),
        "decentralization_diagnosis": str(g3["diagnosis"]),
        "total_games_solved": int(accuracy["total_games_solved"]),
        "g1_off_arc_transfer": g1,
        "g2_search_layer_generalization": g2,
        "g3_decentralization_scaling": g3,
        "accuracy_self_learning_hardware_deltas": {
            "accuracy": accuracy,
            "self_learning": self_learning,
            "hardware": hardware,
        },
        "arguments_became_measurements": {
            "G1": g1["outcome"] != "missing_or_blocked" and g1["outcome"] != "skipped_flagged",
            "G2": g2["outcome"] == "included",
            "G3": g3["diagnosis"] in {"latent", "absent", "partial"},
        },
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
    """Validate the .373 fields that protect the honest headline."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover - defensive guard.
    verdict_text = str(artifact.get("honest_verdict", ""))
    if not verdict_text.startswith(("complete:", "success:", "blocked_", "blocked:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in ("verifier_generalized_off_arc", "search_layer_generalized"):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    if artifact.get("decentralization_diagnosis") not in {"latent", "absent", "partial", "flagged_skipped"}:
        raise ValueError("decentralization_diagnosis must be latent, absent, partial, or flagged_skipped")
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
    """Build, validate, and write the Exp 4041 capstone artifact."""

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

"""Build the Exp 4028 v372 Deep-Think pivot capstone.

Spec refs: REQ-CAPSTONE-4028, SCENARIO-CAPSTONE-4028.

This module reads already-landed upstream JSON artifacts and asks one narrow
question: did the new search navigator over a verified simulator move ARC-3
past the planning wall? Keeping this as an aggregation pass matters because
the capstone should not re-run ARC, hide a flagged artifact, or turn a blocked
upstream into a forward-facing claim.
"""

from __future__ import annotations

from collections.abc import Mapping
import glob
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_4028_capstone_v372.json")
EXPERIMENT_ID = 4028
RANDOM_SEED = 4028
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PYTHON_BIN = Path(".venv/bin/python")

UPSTREAM_IDS = tuple(range(4019, 4028))
FIELDS_IMPORTED: Mapping[int, list[str]] = {
    4019: ["active_milestone_confirmed", "milestone_371_closestate"],
    4020: ["game", "goal_predicate_heldout_precision", "heldout_recall", "n_levelup_transitions"],
    4021: [
        "new_levels_solved_this_task",
        "wall_was_search_not_representation",
        "search_advanced_past_single_step_stall",
        "search_found_plan",
        "real_env_confirmed",
        "nodes_expanded",
        "levels_completed_after",
        "representation_vs_search_diagnosis",
    ],
    4022: ["branch_taken", "decentralization_next_step", "local_support_diagnostic"],
    4023: [
        "retired_r_and_d_line",
        "agreement_role_after_retirement",
        "agreement_is_precision_selector",
        "no_precision_confirmation_v4_proposed",
        "safety_gate_kept",
        "registry_updated",
        "retire_if_same_verdict_triggered",
    ],
    4024: [
        "prior_total_games_solved",
        "total_games_solved",
        "game_solved",
        "target_game",
        "real_env_confirmed",
    ],
    4025: ["solve_transfer_win", "actions_cold", "actions_seeded", "induction_calls_cold", "induction_calls_seeded"],
    4026: [
        "accuracy_parity",
        "accuracy_gap",
        "wallclock_seconds_ratio_judge_over_verifier",
        "token_ratio_judge_over_verifier",
        "verifier_gold_rate",
        "judge_gold_rate",
    ],
    4027: ["per_board_reachability", "honest_verdict"],
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "pivot_central_bet_advanced",
    "new_levels_this_milestone",
    "inference_substrate",
    "planning_result",
    "search_vs_representation_diagnosis",
    "decentralization_branch_taken",
    "selection_retirement",
    "accuracy_memory_efficiency_deltas",
    "cited_upstream_artifacts",
    "flagged_artifacts_skipped",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix outcome so the conductor can classify the capstone without rereading prose.",
    "pivot_central_bet_advanced": (
        "BARE BOOL - the milestone's one question: did adding a navigator over the verified simulator "
        "break a planning wall?"
    ),
    "new_levels_this_milestone": (
        "BARE INT - clean search-layer levels solved by the Deep-Think navigator, excluding flagged inputs."
    ),
    "inference_substrate": "Declares this as aggregation only, preventing live-inference duration false positives.",
    "planning_result": "Carries the exp4020 goal predicate and exp4021 search result that decide the pivot.",
    "search_vs_representation_diagnosis": (
        "Preserves whether exp4021 diagnosed the wall as search/planning rather than representation."
    ),
    "decentralization_branch_taken": "Reports the exp4022 branch only if that artifact is clean; flagged branches are skipped.",
    "selection_retirement": "Shows agreement was retired as a selector while keeping the execution safety gate.",
    "accuracy_memory_efficiency_deltas": "Collects the exp4024 accuracy, exp4025 memory, and exp4026 efficiency deltas.",
    "cited_upstream_artifacts": "List of included upstream artifact ids, imported fields, and sha256 provenance.",
    "flagged_artifacts_skipped": "Lists upstreams excluded before any metric import because flagged_adversarial is true.",
}


def is_sha256(value: object) -> bool:
    """Return true when a value is a lowercase SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def read_json_object(path: Path) -> JsonDict:
    """Load one artifact and reject non-object JSON because capstones cite fields."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover - defensive
    return payload


def sha256_file(path: Path) -> str:
    """Hash an upstream artifact so imported fields are traceable."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_root(root: Path, path: Path) -> str:
    """Return a stable repository-relative path for audit fields."""

    return str(path.relative_to(root))


def matching_artifact_paths(root: Path, experiment_id: int) -> list[Path]:
    """Find candidate JSON artifacts for one upstream experiment."""

    matches: list[Path] = []
    for pattern in (
        root / "results" / f"experiment_{experiment_id}_*.json",
        root / "results" / f"experiment_{experiment_id}.json",
    ):
        matches.extend(Path(path) for path in glob.glob(str(pattern)))
    return sorted(matches)


def selected_upstream_paths(root: Path) -> dict[int, Path | None]:
    """Select one artifact per expected upstream, preferring the final sorted hit."""

    return {
        experiment_id: (matches[-1] if (matches := matching_artifact_paths(root, experiment_id)) else None)
        for experiment_id in UPSTREAM_IDS
    }


def run_summarize_artifact(root: Path, path: Path) -> JsonDict:
    """Run the mandated artifact reader before importing any upstream metric."""

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
    """Return whether an upstream is stamped adversarial and must be skipped."""

    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


def invoked(payload: Mapping[str, Any] | None) -> bool:
    """Return false for missing, blocked, or pending upstream artifacts."""

    if not isinstance(payload, Mapping):
        return False
    verdict_text = str(payload.get("honest_verdict", ""))
    return not verdict_text.startswith(("blocked_", "blocked:")) and "pending_execution" not in verdict_text


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool:
    """Extract a boolean metric without truthifying non-bool values."""

    return isinstance(payload, Mapping) and payload.get(field) is True


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    """Extract an integer metric while rejecting booleans and missing values."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def float_metric(payload: Mapping[str, Any] | None, field: str) -> float:
    """Extract a numeric metric while rejecting booleans and missing values."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    """Extract a string metric for audit fields."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def nested_int(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> int:
    """Read a nested integer without accepting booleans as counters."""

    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return 0
        current = current.get(key)
    return current if isinstance(current, int) and not isinstance(current, bool) else 0


def search_new_levels(payload: Mapping[str, Any] | None) -> int:
    """Count only clean, real-env-confirmed search solves as pivot progress."""

    if not (
        invoked(payload)
        and bool_metric(payload, "real_env_confirmed")
        and bool_metric(payload, "search_found_plan")
        and bool_metric(payload, "search_advanced_past_single_step_stall")
    ):
        return 0
    return int_metric(payload, "new_levels_solved_this_task")


def planning_result(goal_payload: Mapping[str, Any] | None, search_payload: Mapping[str, Any] | None) -> JsonDict:
    """Build the planning evidence block from exp4020 and exp4021."""

    new_levels = search_new_levels(search_payload)
    search_is_clean = invoked(search_payload)
    return {
        "goal_predicate_game": str_metric(goal_payload, "game") if invoked(goal_payload) else "",
        "goal_predicate_heldout_precision": float_metric(goal_payload, "goal_predicate_heldout_precision")
        if invoked(goal_payload)
        else 0.0,
        "new_levels_via_search": new_levels,
        "search_found_plan": search_is_clean and bool_metric(search_payload, "search_found_plan"),
        "search_real_env_confirmed": search_is_clean and bool_metric(search_payload, "real_env_confirmed"),
        "search_advanced_past_single_step_stall": search_is_clean
        and bool_metric(search_payload, "search_advanced_past_single_step_stall"),
        "wall_was_search_not_representation": search_is_clean
        and bool_metric(search_payload, "wall_was_search_not_representation"),
        "levels_completed_after": int_metric(search_payload, "levels_completed_after") if search_is_clean else 0,
        "nodes_expanded": int_metric(search_payload, "nodes_expanded") if search_is_clean else 0,
    }


def decentralization_result(payload: Mapping[str, Any] | None, was_flagged: bool) -> JsonDict:
    """Report the data-gated decentralization branch without importing flagged data."""

    if was_flagged:
        return {"status": "skipped_flagged_artifact", "branch_taken": ""}
    if not invoked(payload):
        return {"status": "missing_or_blocked", "branch_taken": ""}
    return {
        "status": "included",
        "branch_taken": str_metric(payload, "branch_taken"),
        "decentralization_next_step": str_metric(payload, "decentralization_next_step"),
        "local_support_diagnostic": str_metric(payload, "local_support_diagnostic"),
    }


def selection_retirement(payload: Mapping[str, Any] | None) -> JsonDict:
    """Summarize the selection-line retirement without reviving agreement as a selector."""

    return {
        "retired_r_and_d_line": str_metric(payload, "retired_r_and_d_line") if invoked(payload) else "",
        "agreement_role_after_retirement": str_metric(payload, "agreement_role_after_retirement") if invoked(payload) else "",
        "agreement_is_precision_selector": invoked(payload) and bool_metric(payload, "agreement_is_precision_selector"),
        "no_precision_confirmation_v4_proposed": invoked(payload)
        and bool_metric(payload, "no_precision_confirmation_v4_proposed"),
        "safety_gate_kept": invoked(payload) and bool_metric(payload, "safety_gate_kept"),
        "registry_updated": invoked(payload) and bool_metric(payload, "registry_updated"),
        "retire_if_same_verdict_triggered": invoked(payload)
        and bool_metric(payload, "retire_if_same_verdict_triggered"),
    }


def accuracy_delta(archive_payload: Mapping[str, Any] | None, payload: Mapping[str, Any] | None) -> JsonDict:
    """Build the ARC-3 games-solved delta from exp4024, falling back to exp4019 baseline."""

    baseline = int_metric(payload, "prior_total_games_solved") or nested_int(
        archive_payload,
        ("milestone_371_closestate", "arc3", "total_games_solved"),
    )
    solved = invoked(payload) and bool_metric(payload, "game_solved") and bool_metric(payload, "real_env_confirmed")
    total = int_metric(payload, "total_games_solved") if solved else baseline
    return {
        "prior_total_games_solved": baseline,
        "total_games_solved": total,
        "games_solved_delta": max(0, total - baseline) if solved else 0,
        "game_solved": solved,
        "target_game": str_metric(payload, "target_game") if solved else "",
        "real_env_confirmed": solved,
    }


def memory_delta(payload: Mapping[str, Any] | None) -> JsonDict:
    """Build the ArcMemo cost delta from exp4025 only when the transfer win is clean."""

    win = invoked(payload) and bool_metric(payload, "solve_transfer_win")
    cold = int_metric(payload, "actions_cold") if invoked(payload) else 0
    seeded = int_metric(payload, "actions_seeded") if invoked(payload) else 0
    cold_calls = int_metric(payload, "induction_calls_cold") if invoked(payload) else 0
    seeded_calls = int_metric(payload, "induction_calls_seeded") if invoked(payload) else 0
    return {
        "solve_transfer_win": win,
        "actions_cold": cold,
        "actions_seeded": seeded,
        "action_savings": max(0, cold - seeded) if win else 0,
        "induction_calls_cold": cold_calls,
        "induction_calls_seeded": seeded_calls,
        "induction_call_savings": max(0, cold_calls - seeded_calls) if win else 0,
    }


def efficiency_delta(payload: Mapping[str, Any] | None) -> JsonDict:
    """Build the corrected verifier-vs-judge efficiency delta from exp4026."""

    clean = invoked(payload)
    return {
        "accuracy_parity": clean and bool_metric(payload, "accuracy_parity"),
        "accuracy_gap": float_metric(payload, "accuracy_gap") if clean else 0.0,
        "wallclock_seconds_ratio_judge_over_verifier": float_metric(
            payload,
            "wallclock_seconds_ratio_judge_over_verifier",
        )
        if clean
        else 0.0,
        "token_ratio_judge_over_verifier": float_metric(payload, "token_ratio_judge_over_verifier") if clean else 0.0,
        "verifier_gold_rate": float_metric(payload, "verifier_gold_rate") if clean else 0.0,
        "judge_gold_rate": float_metric(payload, "judge_gold_rate") if clean else 0.0,
    }


def accuracy_memory_efficiency_deltas(
    archive_payload: Mapping[str, Any] | None,
    accuracy_payload: Mapping[str, Any] | None,
    memory_payload: Mapping[str, Any] | None,
    efficiency_payload: Mapping[str, Any] | None,
) -> JsonDict:
    """Collect the non-planning deltas the milestone asked the capstone to report."""

    return {
        "accuracy": accuracy_delta(archive_payload, accuracy_payload),
        "memory": memory_delta(memory_payload),
        "efficiency": efficiency_delta(efficiency_payload),
    }


def cited_upstream_artifacts(
    paths: Mapping[int, Path | None],
    clean_upstreams: Mapping[int, Mapping[str, Any]],
) -> list[JsonDict]:
    """Build the provenance list for every non-flagged upstream artifact that exists."""

    cited: list[JsonDict] = []
    for experiment_id in UPSTREAM_IDS:
        path = paths[experiment_id]
        if path is not None and experiment_id in clean_upstreams:
            cited.append(
                {
                    "experiment_id": experiment_id,
                    "fields_imported": FIELDS_IMPORTED[experiment_id],
                    "sha256": sha256_file(path),
                }
            )
    return cited


def flagged_artifacts_skipped(root: Path, paths: Mapping[int, Path | None], flagged_ids: set[int]) -> list[JsonDict]:
    """Record artifacts skipped before aggregation due to adversarial stamps."""

    return [
        {
            "experiment_id": experiment_id,
            "path": relative_to_root(root, path) if (path := paths[experiment_id]) is not None else "",
            "reason": "flagged_adversarial:true",
        }
        for experiment_id in sorted(flagged_ids)
    ]


def missing_upstream_artifacts(paths: Mapping[int, Path | None]) -> list[JsonDict]:
    """Record missing upstream artifacts without treating absence as a fabricated gate."""

    return [{"experiment_id": experiment_id} for experiment_id in UPSTREAM_IDS if paths[experiment_id] is None]


def upstream_artifact_state(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    summaries: Mapping[int, Mapping[str, Any]],
    flagged_ids: set[int],
    clean_upstreams: Mapping[int, Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Expose inclusion state so missing and skipped inputs are auditable."""

    state: dict[str, JsonDict] = {}
    for experiment_id in UPSTREAM_IDS:
        path = paths[experiment_id]
        payload = upstreams[experiment_id]
        state[str(experiment_id)] = {
            "exists": path is not None,
            "path": relative_to_root(root, path) if path is not None else "",
            "honest_verdict": str(payload.get("honest_verdict")) if isinstance(payload, Mapping) else "missing",
            "flagged_adversarial": experiment_id in flagged_ids,
            "included": experiment_id in clean_upstreams,
            "invoked": invoked(payload),
            "summarize_artifact_returncode": summaries.get(experiment_id, {}).get("returncode"),
        }
    return state


def hardware_continuity(payload: Mapping[str, Any] | None) -> JsonDict:
    """Carry exp4027 continuity state without treating it as inference progress."""

    reachability = payload.get("per_board_reachability") if isinstance(payload, Mapping) else None
    return {
        "included": invoked(payload),
        "honest_verdict": str_metric(payload, "honest_verdict") if invoked(payload) else "",
        "per_board_reachability": dict(reachability) if isinstance(reachability, Mapping) else {},
    }


def duration_from(started_s: float, now_s: float | None) -> float:
    """Compute an honest aggregation duration with a small nonzero floor."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0001, end - started_s)


def verdict(
    pivot_advanced: bool,
    new_levels: int,
    branch_taken: str,
    games_delta: int,
    memory_win: bool,
    efficiency_win: bool,
    missing_count: int,
    skipped_count: int,
) -> str:
    """Build the terminal-prefix milestone verdict from the .372 headline axes."""

    prefix = "success" if pivot_advanced or games_delta > 0 or memory_win or efficiency_win else "complete"
    pivot_text = "ADVANCED" if pivot_advanced else "NOT_ADVANCED"
    memory_text = "memory_win" if memory_win else "memory_no_win"
    efficiency_text = "efficiency_win" if efficiency_win else "efficiency_no_win"
    return (
        f"{prefix}: capstone_v372_deep_think_pivot_{pivot_text}_search_levels{new_levels}_"
        f"decentralization_{branch_taken}_games_delta{games_delta}_{memory_text}_{efficiency_text}_"
        f"missing{missing_count}_flagged_skipped{skipped_count}"
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
    """Build the .372 capstone from whatever upstream artifacts exist."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    paths = selected_upstream_paths(root_path)
    summaries = summarize_existing_artifacts(root_path, paths, summary_statuses)
    upstreams: dict[int, Mapping[str, Any] | None] = {
        experiment_id: read_json_object(path) if path is not None else None
        for experiment_id, path in paths.items()
    }
    flagged_ids = {experiment_id for experiment_id, payload in upstreams.items() if flagged(payload)}
    clean_upstreams: dict[int, Mapping[str, Any]] = {
        experiment_id: payload
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and experiment_id not in flagged_ids
    }

    planning = planning_result(clean_upstreams.get(4020), clean_upstreams.get(4021))
    pivot_advanced = (
        planning["new_levels_via_search"] > 0
        and planning["search_real_env_confirmed"] is True
        and planning["search_advanced_past_single_step_stall"] is True
    )
    decentralization = decentralization_result(clean_upstreams.get(4022), 4022 in flagged_ids)
    branch_taken = (
        "skipped_flagged_exp4022"
        if 4022 in flagged_ids
        else str(decentralization.get("branch_taken") or "missing_or_blocked_exp4022")
    )
    deltas = accuracy_memory_efficiency_deltas(
        clean_upstreams.get(4019),
        clean_upstreams.get(4024),
        clean_upstreams.get(4025),
        clean_upstreams.get(4026),
    )
    missing = missing_upstream_artifacts(paths)
    skipped = flagged_artifacts_skipped(root_path, paths, flagged_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v372_4028.v1",
        "experiment_id": EXPERIMENT_ID,
        "pivot_central_bet_advanced": pivot_advanced,
        "new_levels_this_milestone": int(planning["new_levels_via_search"]),
        "planning_result": planning,
        "search_vs_representation_diagnosis": str_metric(
            clean_upstreams.get(4021),
            "representation_vs_search_diagnosis",
        )
        if invoked(clean_upstreams.get(4021))
        else "not_established",
        "decentralization_branch_taken": branch_taken,
        "decentralization_result": decentralization,
        "selection_retirement": selection_retirement(clean_upstreams.get(4023)),
        "accuracy_memory_efficiency_deltas": deltas,
        "hardware_continuity": hardware_continuity(clean_upstreams.get(4027)),
        "cited_upstream_artifacts": cited_upstream_artifacts(paths, clean_upstreams),
        "flagged_artifacts_skipped": skipped,
        "missing_upstream_artifacts": missing,
        "upstream_artifact_state": upstream_artifact_state(
            root_path,
            paths,
            upstreams,
            summaries,
            flagged_ids,
            clean_upstreams,
        ),
        "summarize_artifact_status": {
            str(experiment_id): {
                "returncode": status.get("returncode"),
                "stdout": status.get("stdout", ""),
                "stderr": status.get("stderr", ""),
            }
            for experiment_id, status in summaries.items()
        },
        "honest_verdict": verdict(
            pivot_advanced,
            int(planning["new_levels_via_search"]),
            branch_taken,
            int(deltas["accuracy"]["games_solved_delta"]),
            bool(deltas["memory"]["solve_transfer_win"]),
            bool(deltas["efficiency"]["accuracy_parity"])
            and float(deltas["efficiency"]["wallclock_seconds_ratio_judge_over_verifier"]) > 1.0,
            len(missing),
            len(skipped),
        ),
        "duration_s": duration_from(start, now_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the .372 fields that prevent false headline confirmation."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover - defensive
    verdict_text = str(artifact.get("honest_verdict", ""))
    if not verdict_text.startswith(("complete:", "success:", "blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(artifact.get("pivot_central_bet_advanced"), bool):
        raise ValueError("pivot_central_bet_advanced must be a bare bool")
    if not isinstance(artifact.get("new_levels_this_milestone"), int) or isinstance(
        artifact.get("new_levels_this_milestone"),
        bool,
    ):
        raise ValueError("new_levels_this_milestone must be a bare int")
    if not isinstance(artifact.get("inference_substrate"), str):
        raise ValueError("inference_substrate must be a string")
    for field in ("planning_result", "selection_retirement", "accuracy_memory_efficiency_deltas"):
        if not isinstance(artifact.get(field), Mapping):
            raise ValueError(f"{field} must be an object")
    citations = artifact.get("cited_upstream_artifacts")
    if not isinstance(citations, list):
        raise ValueError("cited_upstream_artifacts must be a list")  # pragma: no cover - defensive
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise ValueError("citation entries must be objects")  # pragma: no cover - defensive
        if not isinstance(citation.get("experiment_id"), int):
            raise ValueError("citation entries need integer experiment_id")  # pragma: no cover - defensive
        if not isinstance(citation.get("fields_imported"), list):
            raise ValueError("citation entries need fields_imported list")
        if not is_sha256(citation.get("sha256")):
            raise ValueError("citation entries need sha256")
    if not isinstance(artifact.get("flagged_artifacts_skipped"), list):
        raise ValueError("flagged_artifacts_skipped must be a list")
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be sha256")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    summary_statuses: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the Exp 4028 capstone artifact."""

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

"""Build the Exp 4105 v379 TRM-pivot capstone aggregation.

Spec refs: REQ-CAPSTONE-4105, SCENARIO-CAPSTONE-4105.

This capstone is deliberately an aggregation layer, not a second experiment. It
turns the .379 TRM artifacts into one decision-grade read while preserving the
important failure mode: a verifier can be useful elsewhere and still fail to
rank native TRM grids. That negative is a real result, so flagged upstreams are
kept as provenance but their metrics never enter the headline.
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
OUTPUT_REL_PATH = Path("results/experiment_4105_capstone_v379.json")
EXPERIMENT_ID = 4105
RANDOM_SEED = 4105
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PYTHON_BIN = Path(".venv/bin/python")

HEADLINE_OUTCOMES = {
    "verifier_rft_on_trm_validated",
    "honest_negative_no_grid_discrimination",
    "trainer_derisked_science_open",
}
PRIOR_CAPTURED_PP = 0.0

UPSTREAM_IDS = (4099, 4100, 4101, 4102, 4103, 4104)
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4099: Path("results/experiment_4099_trm_pool_verifier_discrimination_probe.json"),
    4100: Path("results/experiment_4100_trm_verifier_rft_conditional.json"),
    4101: Path("results/experiment_4101_eleventh_game_explore_first.json"),
    4102: Path("results/experiment_4102_sota_ingestion_trm_self_training.json"),
    4103: Path("results/experiment_4103_verifier_registry_gaps_hygiene.json"),
    4104: Path("results/experiment_4104_hardware_continuity.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headline_outcome",
    "upstream_provenance",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. The milestone's decision-grade summary; an honest "
        "negative ('verifier does not discriminate on TRM grids') is a COMPLETE, "
        "valuable verdict."
    ),
    "headline_outcome": (
        "One of {verifier_rft_on_trm_validated, "
        "honest_negative_no_grid_discrimination, trainer_derisked_science_open} -- "
        "forces a single unambiguous read of the pivot's first decision-grade result."
    ),
    "upstream_provenance": (
        "{experiment_id, fields_imported, sha256} per cited upstream -- the audit "
        "trail that proves the capstone synthesizes real measurements, not invented "
        "numbers."
    ),
}


def is_sha256(value: object) -> bool:
    """Return true only for lowercase SHA-256 hex digests."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def read_json_object(path: Path) -> JsonDict:
    """Load one upstream JSON artifact as an object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover
    return payload


def sha256_file(path: Path) -> str:
    """Hash an upstream artifact so the capstone can be audited later."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_root(root: Path, path: Path) -> str:
    """Return stable repository-relative paths in the artifact."""

    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover
        return str(path)


def _fallback_path(root: Path, experiment_id: int) -> Path | None:
    matches = sorted((root / "results").glob(f"experiment_{experiment_id}_*.json"))
    return matches[0] if matches else None


def selected_upstream_paths(root: Path | str) -> dict[int, Path | None]:
    """Select the intended artifact for every .379 upstream id."""

    root_path = Path(root)
    paths: dict[int, Path | None] = {}
    for experiment_id in UPSTREAM_IDS:
        default = root_path / DEFAULT_UPSTREAM_PATHS[experiment_id]
        paths[experiment_id] = (
            default if default.exists() else _fallback_path(root_path, experiment_id)
        )
    return paths


def run_summarize_artifact(root: Path, path: Path) -> JsonDict:  # pragma: no cover
    """Run the mandated artifact reader before importing upstream fields."""

    completed = subprocess.run(
        [str(PYTHON_BIN), "scripts/summarize_artifact.py", str(path)],
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
        else:  # pragma: no cover
            statuses[experiment_id] = run_summarize_artifact(root, path)
    return statuses


def flagged(payload: Mapping[str, Any] | None) -> bool:
    """Return whether an upstream is stamped adversarial."""

    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


def verdict_text(payload: Mapping[str, Any] | None) -> str:
    """Extract an upstream honest verdict without coercing non-strings."""

    value = payload.get("honest_verdict") if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def invoked(payload: Mapping[str, Any] | None) -> bool:
    """Return false for missing or explicitly blocked upstream artifacts."""

    text = verdict_text(payload)
    return (
        bool(text)
        and not text.startswith(("blocked_", "blocked:"))
        and str_metric(payload, "status") != "blocked"
    )


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    """Extract a string metric without accepting numbers or booleans."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool:
    """Extract a JSON boolean without truthifying strings or integers."""

    return isinstance(payload, Mapping) and payload.get(field) is True


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    """Extract an integer counter while rejecting booleans."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def float_metric(payload: Mapping[str, Any] | None, field: str) -> float:
    """Extract a numeric metric while rejecting booleans and strings."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def list_float_metric(payload: Mapping[str, Any] | None, field: str) -> list[float]:
    """Extract numeric confidence interval endpoints without coercing strings."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    if not isinstance(value, list):
        return []
    return [
        float(item)
        for item in value
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]


def _nested(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def nested_str(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> str:
    """Read a nested string metric without coercion."""

    value = _nested(payload, path)
    return value if isinstance(value, str) else ""


def nested_bool(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> bool:
    """Read a nested boolean metric without coercion."""

    return _nested(payload, path) is True


def nested_float(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> float:
    """Read a nested numeric metric without accepting booleans."""

    value = _nested(payload, path)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def nested_int(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> int:
    """Read a nested integer counter without accepting booleans."""

    value = _nested(payload, path)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def nested_float_list(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> list[float]:
    """Read nested numeric list values such as bootstrap CIs."""

    value = _nested(payload, path)
    if not isinstance(value, list):
        return []
    return [
        float(item)
        for item in value
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]


def _ci_excludes_zero_positive(ci: list[float]) -> bool:
    return len(ci) == 2 and ci[0] > 0.0 and ci[1] > 0.0


def trm_grid_discrimination(
    payload: Mapping[str, Any] | None,
    registry_payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    """Summarize whether a verifier reranker discriminates native TRM grids."""

    if was_skipped:
        status = "skipped_flagged"
    elif not isinstance(payload, Mapping):
        status = "missing"
    elif not invoked(payload):
        status = "blocked"
    else:
        status = (
            "verifier_beats_trm_vote"
            if bool_metric(payload, "verifier_beats_trm_vote")
            else "no_grid_discrimination"
        )

    best_reranker = str_metric(payload, "best_reranker")
    captured = float_metric(payload, "captured_pp_directional")
    bottleneck = ""
    if status == "no_grid_discrimination":
        bottleneck = nested_str(registry_payload, ("exp4099_gap", "missing_discriminator")) or (
            "signal_separating_correct_trm_grid_from_confident_wrong_trm_grid_on_pool"
        )
    return {
        "status": status,
        "verifier_beats_trm_vote": status == "verifier_beats_trm_vote",
        "captured_pp": captured if status in {"verifier_beats_trm_vote", "no_grid_discrimination"} else 0.0,
        "prior_captured_pp": PRIOR_CAPTURED_PP,
        "delta_vs_prior_pp": (
            captured - PRIOR_CAPTURED_PP
            if status in {"verifier_beats_trm_vote", "no_grid_discrimination"}
            else 0.0
        ),
        "best_reranker": best_reranker,
        "pool_n_tasks": int_metric(payload, "pool_n_tasks") or int_metric(payload, "n_tasks_scored"),
        "underpowered": bool_metric(payload, "underpowered"),
        "trm_vote_pass2": nested_float(payload, ("per_reranker", "TRM_VOTE", "pass@2")),
        "best_reranker_pass2": nested_float(payload, ("per_reranker", best_reranker, "pass@2")),
        "captured_pp_ci95": nested_float_list(
            payload, ("per_reranker", best_reranker, "captured_pp_ci95")
        ),
        "bottleneck": bottleneck,
        "honest_verdict": verdict_text(payload),
    }


def verifier_rft_followthrough(
    payload: Mapping[str, Any] | None,
    *,
    verifier_beat_vote: bool,
    was_skipped: bool,
) -> JsonDict:
    """Summarize Exp 4100 only when Exp 4099 made RFT scientifically warranted."""

    if was_skipped:
        return {
            "status": "skipped_flagged",
            "branch_taken": "",
            "rft_vs_ablation_delta": None,
            "native_trainer_checkpoint": "skipped_flagged",
            "trm_native_trainer_checkpoint_ok": False,
            "honest_verdict": "",
        }
    if not verifier_beat_vote:
        return {
            "status": "not_applicable_no_grid_discrimination",
            "branch_taken": "",
            "rft_vs_ablation_delta": None,
            "native_trainer_checkpoint": "not_applicable_no_grid_discrimination",
            "trm_native_trainer_checkpoint_ok": False,
            "honest_verdict": "",
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "missing",
            "branch_taken": "",
            "rft_vs_ablation_delta": None,
            "native_trainer_checkpoint": "missing",
            "trm_native_trainer_checkpoint_ok": False,
            "honest_verdict": "",
        }
    if not invoked(payload):
        return {
            "status": "blocked",
            "branch_taken": str_metric(payload, "branch_taken"),
            "rft_vs_ablation_delta": None,
            "native_trainer_checkpoint": "blocked",
            "trm_native_trainer_checkpoint_ok": False,
            "honest_verdict": verdict_text(payload),
        }

    delta_payload = payload.get("rft_vs_ablation_delta")
    delta_map = delta_payload if isinstance(delta_payload, Mapping) else {}
    delta = float_metric(delta_map, "delta")
    ci = list_float_metric(delta_map, "ci95")
    checkpoint_ok = bool_metric(payload, "trm_native_trainer_checkpoint_ok") or nested_bool(
        payload, ("native_smoke", "checkpoint_reload_ok")
    )
    status = "rft_beats_vote_sft" if delta > 0.0 and _ci_excludes_zero_positive(ci) else (
        str_metric(delta_map, "status") or "measured_null"
    )
    if status == "rft_beats_vote_sft" and not checkpoint_ok:
        status = "rft_beats_without_checkpoint"
    return {
        "status": status,
        "branch_taken": str_metric(payload, "branch_taken"),
        "rft_vs_ablation_delta": {
            "delta": delta,
            "ci95": ci,
            "metric": str_metric(delta_map, "metric"),
        },
        "native_trainer_checkpoint": "checkpoint_ok" if checkpoint_ok else "checkpoint_missing",
        "trm_native_trainer_checkpoint_ok": checkpoint_ok,
        "honest_verdict": verdict_text(payload),
    }


def arc_games(
    payload: Mapping[str, Any] | None,
    prior_payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    """Carry the clean ARC games-solved count without inferring hidden progress."""

    if was_skipped:
        prior = int_metric(prior_payload, "total_games_solved")
        return {
            "status": "skipped_flagged",
            "prior_total_games_solved": prior,
            "total_games_solved": prior,
            "game_solved": False,
            "real_env_confirmed": False,
            "target_game": "",
            "first_solve_at_action": 0,
            "honest_verdict": "",
        }
    if not isinstance(payload, Mapping):
        prior = int_metric(prior_payload, "total_games_solved")
        return {
            "status": "missing",
            "prior_total_games_solved": prior,
            "total_games_solved": prior,
            "game_solved": False,
            "real_env_confirmed": False,
            "target_game": "",
            "first_solve_at_action": 0,
            "honest_verdict": "",
        }

    clean = invoked(payload)
    prior = int_metric(payload, "prior_total_games_solved") or int_metric(
        prior_payload, "total_games_solved"
    )
    solved = clean and bool_metric(payload, "game_solved") and bool_metric(
        payload, "real_env_confirmed"
    )
    total = int_metric(payload, "total_games_solved") if clean else prior
    return {
        "status": "new_game_solved" if solved else "measured_no_new_solve" if clean else "blocked",
        "prior_total_games_solved": prior,
        "total_games_solved": total or prior,
        "game_solved": solved,
        "real_env_confirmed": solved,
        "target_game": str_metric(payload, "target_game") if clean else "",
        "first_solve_at_action": int_metric(payload, "first_solve_at_action") if clean else 0,
        "honest_verdict": verdict_text(payload),
    }


def sota_ingestion(payload: Mapping[str, Any] | None) -> JsonDict:
    """Summarize clean .380 method candidates from the SOTA ingestion artifact."""

    methods = payload.get("methods_mapped") if isinstance(payload, Mapping) else None
    method_rows = [dict(item) for item in methods if isinstance(item, Mapping)] if isinstance(methods, list) else []
    return {
        "included": invoked(payload),
        "flagged_for_v380": str_metric(payload, "flagged_for_v380"),
        "methods_mapped": [
            {"name": str_metric(row, "name"), "arxiv_id": str_metric(row, "arxiv_id")}
            for row in method_rows
        ],
        "honest_verdict": verdict_text(payload),
    }


def registry_gap_hygiene(payload: Mapping[str, Any] | None) -> JsonDict:
    """Summarize registry and gap reconciliation without re-importing flagged Exp 4100."""

    gaps = payload.get("gaps_updated") if isinstance(payload, Mapping) else None
    gap_rows = [item for item in gaps if isinstance(item, str)] if isinstance(gaps, list) else []
    return {
        "included": invoked(payload),
        "gaps_updated": gap_rows,
        "regression_guard_passed": bool_metric(payload, "regression_guard_passed"),
        "trm_grid_gap_status": nested_str(payload, ("exp4099_gap", "status")),
        "missing_discriminator": nested_str(payload, ("exp4099_gap", "missing_discriminator")),
        "honest_verdict": verdict_text(payload),
    }


def hardware_continuity(payload: Mapping[str, Any] | None) -> JsonDict:
    """Carry hardware continuity separately from the scientific TRM verdict."""

    reachability = payload.get("per_board_reachability") if isinstance(payload, Mapping) else None
    return {
        "included": invoked(payload),
        "kv260_terminal_confirmed": bool_metric(payload, "kv260_terminal_confirmed"),
        "per_board_reachability": dict(reachability) if isinstance(reachability, Mapping) else {},
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken"),
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken"),
        "honest_verdict": verdict_text(payload),
    }


def candidate_v380_directions(
    discrimination: Mapping[str, Any],
    sota: Mapping[str, Any],
    registry: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> list[str]:
    """List next directions implied by clean upstreams, led by the named bottleneck."""

    directions: list[str] = []
    if discrimination.get("status") == "no_grid_discrimination":
        gap = registry.get("gaps_updated", [])
        gap_name = "GAP-TRM-GRID-DISCRIMINATION" if "GAP-TRM-GRID-DISCRIMINATION" in gap else (
            "TRM-grid-discrimination"
        )
        directions.append(
            f"{gap_name}: build a discriminator that separates correct TRM grids from "
            "confident wrong TRM grids before another verifier-RFT run."
        )
    flagged = str(sota.get("flagged_for_v380", ""))
    if flagged:
        directions.append(f"{flagged}: test rejected-trace selection only after grid discrimination improves.")
    method_names = [row.get("name", "") for row in sota.get("methods_mapped", []) if isinstance(row, Mapping)]
    for name in method_names:
        if "process rewards" in str(name).lower():
            directions.append(f"{name}: score recursive grid-edit steps instead of only final grids.")
        if "imperfect-verifier" in str(name).lower():
            directions.append(f"{name}: carry FP/FN correction into any TRM reward labels.")
    if hardware.get("included") is True:
        directions.append("Keep hardware continuity separate from TRM science claims until a clean trainer run exists.")
    return directions


def headline_outcome(discrimination: Mapping[str, Any], rft: Mapping[str, Any]) -> str:
    """Choose the single enumerated outcome the milestone must report."""

    if discrimination.get("status") == "no_grid_discrimination":
        return "honest_negative_no_grid_discrimination"
    if (
        discrimination.get("status") == "verifier_beats_trm_vote"
        and rft.get("status") == "rft_beats_vote_sft"
        and rft.get("trm_native_trainer_checkpoint_ok") is True
    ):
        return "verifier_rft_on_trm_validated"
    return "trainer_derisked_science_open"


def headline_answer(
    discrimination: Mapping[str, Any],
    rft: Mapping[str, Any],
    games: Mapping[str, Any],
) -> JsonDict:
    """Answer the three concrete headline questions in machine-checkable fields."""

    return {
        "verifier_beats_trm_vote": discrimination.get("verifier_beats_trm_vote") is True,
        "captured_pp_vs_prior": {
            "prior_captured_pp": PRIOR_CAPTURED_PP,
            "captured_pp": float(discrimination.get("captured_pp", 0.0)),
            "delta_vs_prior_pp": float(discrimination.get("delta_vs_prior_pp", 0.0)),
        },
        "best_reranker": str(discrimination.get("best_reranker", "")),
        "trm_vote_pass2": float(discrimination.get("trm_vote_pass2", 0.0)),
        "best_reranker_pass2": float(discrimination.get("best_reranker_pass2", 0.0)),
        "verifier_rft_beat_vote_sft_ablation": rft.get("status") == "rft_beats_vote_sft",
        "native_trm_trainer_checkpoint_produced": (
            rft.get("native_trainer_checkpoint") == "checkpoint_ok"
        ),
        "total_arc_games_solved": int(games.get("total_games_solved", 0)),
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
                "sha256": sha256_file(path) if path is not None else "",
            }
        )
    return rows


def imported_fields_by_id(
    discrimination: Mapping[str, Any],
    rft: Mapping[str, Any],
    games: Mapping[str, Any],
    clean_ids: set[int],
) -> dict[int, list[str]]:
    """Declare exactly which upstream fields shaped the capstone headline."""

    fields: dict[int, list[str]] = {experiment_id: [] for experiment_id in UPSTREAM_IDS}
    if 4099 in clean_ids:
        fields[4099] = [
            "verifier_beats_trm_vote",
            "captured_pp_directional",
            "best_reranker",
            "pool_n_tasks",
            "per_reranker.TRM_VOTE.pass@2",
            f"per_reranker.{discrimination.get('best_reranker', '')}.pass@2",
        ]
    if 4100 in clean_ids and rft.get("status") not in {
        "not_applicable_no_grid_discrimination",
        "skipped_flagged",
        "missing",
    }:
        fields[4100] = [
            "branch_taken",
            "rft_vs_ablation_delta",
            "trm_native_trainer_checkpoint_ok",
        ]
    if 4101 in clean_ids:
        fields[4101] = [
            "game_solved",
            "real_env_confirmed",
            "prior_total_games_solved",
            "total_games_solved",
            "target_game",
            "first_solve_at_action",
        ]
    if 4102 in clean_ids:
        fields[4102] = ["flagged_for_v380", "methods_mapped"]
    if 4103 in clean_ids:
        fields[4103] = [
            "gaps_updated",
            "regression_guard_passed",
            "exp4099_gap.missing_discriminator",
        ]
    if 4104 in clean_ids:
        fields[4104] = [
            "kv260_terminal_confirmed",
            "per_board_reachability",
            "gatemate_step_taken",
            "polarfire_step_taken",
        ]
    return fields


def upstream_provenance(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    summaries: Mapping[int, Mapping[str, Any]],
    skipped_ids: set[int],
    fields_by_id: Mapping[int, list[str]],
) -> list[JsonDict]:
    """Cite every existing upstream sha and name the fields actually imported."""

    rows: list[JsonDict] = []
    for experiment_id in UPSTREAM_IDS:
        path = paths[experiment_id]
        if path is None:
            continue
        skipped = experiment_id in skipped_ids
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": relative_to_root(root, path),
                "sha256": sha256_file(path),
                "fields_imported": [] if skipped else list(fields_by_id.get(experiment_id, [])),
                "skipped": skipped,
                "skip_reason": "flagged_adversarial:true" if skipped else "",
                "honest_verdict": verdict_text(upstreams[experiment_id])
                if isinstance(upstreams[experiment_id], Mapping)
                else "",
                "summarize_artifact_returncode": summaries.get(experiment_id, {}).get("returncode"),
            }
        )
    return rows


def missing_upstream_artifacts(paths: Mapping[int, Path | None]) -> list[JsonDict]:
    """Record missing upstream artifacts without turning absence into a hidden gate."""

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
            "honest_verdict": verdict_text(payload) if isinstance(payload, Mapping) else "missing",
            "flagged_adversarial": flagged(payload),
            "included": experiment_id in clean_ids,
            "skipped": experiment_id in skipped_ids,
            "summarize_artifact_returncode": summaries.get(experiment_id, {}).get("returncode"),
        }
    return state


def verdict(
    outcome: str,
    discrimination: Mapping[str, Any],
    games_solved_total: int,
    skipped_count: int,
) -> str:
    """Build the terminal-prefix .379 headline."""

    prefix = "success:" if outcome == "verifier_rft_on_trm_validated" else "complete:"
    best = str(discrimination.get("best_reranker", "")) or "missing"
    captured = float(discrimination.get("captured_pp", 0.0))
    return (
        f"{prefix} capstone_v379_{outcome}_best_{best}_captured_{captured:.4f}_"
        f"games{games_solved_total}_flagged_skipped{skipped_count}"
    )


def duration_from(started_s: float, now_s: float | None) -> float:
    """Compute an honest aggregation duration with a small nonzero floor."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0001, end - started_s)


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
    """Build the .379 TRM pivot capstone from landed upstream artifacts."""

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

    discrimination = trm_grid_discrimination(
        clean_upstreams.get(4099),
        clean_upstreams.get(4103),
        was_skipped=4099 in skipped_ids,
    )
    rft = verifier_rft_followthrough(
        clean_upstreams.get(4100),
        verifier_beat_vote=bool(discrimination["verifier_beats_trm_vote"]),
        was_skipped=4100 in skipped_ids,
    )
    games = arc_games(clean_upstreams.get(4101), None, was_skipped=4101 in skipped_ids)
    sota = sota_ingestion(clean_upstreams.get(4102))
    registry = registry_gap_hygiene(clean_upstreams.get(4103))
    hardware = hardware_continuity(clean_upstreams.get(4104))
    outcome = headline_outcome(discrimination, rft)
    directions = candidate_v380_directions(discrimination, sota, registry, hardware)
    fields_by_id = imported_fields_by_id(discrimination, rft, games, clean_ids)
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    total_games = int(games["total_games_solved"])

    artifact: JsonDict = {
        "schema": "carnot.capstone_v379_4105.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": verdict(outcome, discrimination, total_games, len(skipped)),
        "headline_outcome": outcome,
        "headline_answer": headline_answer(discrimination, rft, games),
        "total_arc_games_solved": total_games,
        "trm_grid_discrimination": discrimination,
        "verifier_rft_followthrough": rft,
        "arc_games": games,
        "candidate_v380_directions": directions,
        "sota_ingestion": sota,
        "registry_gap_hygiene": registry,
        "hardware_continuity": hardware,
        "flagged_artifacts_skipped": skipped,
        "upstream_provenance": upstream_provenance(
            root_path, paths, upstreams, summaries, skipped_ids, fields_by_id
        ),
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
    """Validate the fields that protect the .379 headline from drift."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover
    if "gated_on" in artifact:
        raise ValueError("capstone must not emit gated_on")  # pragma: no cover
    verdict_value = str(artifact.get("honest_verdict", ""))
    if not verdict_value.startswith(("complete:", "success:", "blocked_", "blocked:")):
        raise ValueError("honest_verdict must use a terminal prefix")  # pragma: no cover
    if artifact.get("headline_outcome") not in HEADLINE_OUTCOMES:
        raise ValueError("headline_outcome must be one of the enumerated values")
    if not isinstance(artifact.get("total_arc_games_solved"), int) or isinstance(
        artifact.get("total_arc_games_solved"), bool
    ):
        raise ValueError("total_arc_games_solved must be a bare int")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")  # pragma: no cover
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be an object")  # pragma: no cover
    for field, principle in FIELD_PRINCIPLES.items():
        if principles.get(field) != principle:
            raise ValueError(f"field_principles.{field} mismatch")  # pragma: no cover
    provenance = artifact.get("upstream_provenance")
    if not isinstance(provenance, list):
        raise ValueError("upstream_provenance must be a list")  # pragma: no cover
    for row in provenance:
        if not isinstance(row, Mapping):
            raise ValueError("upstream_provenance entries must be objects")  # pragma: no cover
        if not isinstance(row.get("experiment_id"), int):
            raise ValueError("upstream_provenance entries need integer experiment_id")  # pragma: no cover
        if not isinstance(row.get("fields_imported"), list) or not all(
            isinstance(item, str) for item in row.get("fields_imported", [])
        ):
            raise ValueError("upstream_provenance fields_imported must be strings")  # pragma: no cover
        if row.get("skipped") is True and row.get("fields_imported") != []:
            raise ValueError("skipped upstreams must import no fields")  # pragma: no cover
        if not is_sha256(row.get("sha256")):
            raise ValueError("upstream_provenance entries need sha256")
    if not isinstance(artifact.get("flagged_artifacts_skipped"), list):
        raise ValueError("flagged_artifacts_skipped must be a list")  # pragma: no cover
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be sha256")  # pragma: no cover


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    summary_statuses: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the Exp 4105 capstone artifact."""

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

"""Build the Exp 4085 v377 capstone aggregation.

Spec refs: REQ-CAPSTONE-4085, SCENARIO-CAPSTONE-4085.

The capstone answers the verifier-as-reward pivot from landed artifacts only.
It does not trust stamped adversarial artifacts for metrics, because those
artifacts already failed the repository's fabrication checks. Blocked artifacts
are still read as state: a blocked RFT eval is an honest outcome, not a reason
for the capstone itself to skip.
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
OUTPUT_REL_PATH = Path("results/experiment_4085_capstone_v377.json")
EXPERIMENT_ID = 4085
RANDOM_SEED = 4085
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PYTHON_BIN = Path(".venv/bin/python")

UPSTREAM_IDS = tuple(range(4076, 4085))
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4076: Path("results/experiment_4076_archive_v376_activate_v377.json"),
    4077: Path("results/experiment_4077_verifier_reward_rft_corpus_build.json"),
    4078: Path("results/experiment_4078_verifier_reward_rft_train_launch.json"),
    4079: Path("results/experiment_4079_verifier_reward_rft_eval_collect.json"),
    4080: Path("results/experiment_4080_sudoku_rft_positive_control.json"),
    4081: Path("results/experiment_4081_sota_ingestion_verifier_as_reward_receipt.json"),
    4082: Path("results/experiment_4082_ninth_game_explore_first.json"),
    4083: Path("results/experiment_4083_verifier_registry_gaps_hygiene.json"),
    4084: Path("results/experiment_4084_hardware_continuity.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "pivot_result",
    "sudoku_control_reproduced",
    "decentralization_distillation_outcome",
    "games_solved_total",
    "cited_upstream_artifacts",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefix verdict that distinguishes measured wins, measured nulls, blocked "
        "pivot state, accumulating training, and skipped flagged controls."
    ),
    "pivot_result": (
        "principle: the headline -- the verifier's training-time value is now measured one "
        "way or the other, or honestly recorded as blocked/accumulating when no clean held-out "
        "RFT eval exists."
    ),
    "sudoku_control_reproduced": (
        "BARE BOOL - true only from a clean, non-flagged Sudoku positive-control artifact."
    ),
    "decentralization_distillation_outcome": (
        "States whether clean RFT moved local induction toward Codex, or that the movement is unmeasured."
    ),
    "games_solved_total": "BARE INT - monotonic ARC-AGI-3 solved-game count from clean exp4082 evidence.",
    "cited_upstream_artifacts": "Included non-flagged upstream experiment ids and sha256 provenance only.",
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
    """Load a JSON object artifact and reject malformed top-level values."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover
    return payload


def sha256_file(path: Path) -> str:
    """Hash an upstream artifact for provenance."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_root(root: Path, path: Path) -> str:
    """Return a stable repository-relative path for audit fields."""

    try:
        return str(path.relative_to(root))
    except ValueError:  # pragma: no cover
        return str(path)


def _fallback_path(root: Path, experiment_id: int) -> Path | None:
    matches = sorted((root / "results").glob(f"experiment_{experiment_id}_*.json"))
    return matches[0] if matches else None


def selected_upstream_paths(root: Path | str) -> dict[int, Path | None]:
    """Select the intended artifact for each .377 upstream id."""

    root_path = Path(root)
    paths: dict[int, Path | None] = {}
    for experiment_id in UPSTREAM_IDS:
        default = root_path / DEFAULT_UPSTREAM_PATHS[experiment_id]
        paths[experiment_id] = (
            default if default.exists() else _fallback_path(root_path, experiment_id)
        )
    return paths


def run_summarize_artifact(root: Path, path: Path) -> JsonDict:  # pragma: no cover
    """Run the mandated disciplined reader before importing an upstream metric."""

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
    """Return whether an upstream carries the stamped adversarial flag."""

    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


def live_critical(summary: Mapping[str, Any] | None) -> bool:
    """Return whether summarize_artifact.py observed a live critical concern."""

    return isinstance(summary, Mapping) and summary.get("returncode") == 2


def verdict_text(payload: Mapping[str, Any] | None) -> str:
    """Extract the honest verdict text without treating non-strings as verdicts."""

    value = payload.get("honest_verdict") if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def invoked(payload: Mapping[str, Any] | None) -> bool:
    """Return false for missing or blocked upstream artifacts."""

    text = verdict_text(payload)
    status = str_metric(payload, "status")
    return (
        bool(text)
        and not text.startswith(("blocked_", "blocked:"))
        and status != "blocked"
        and "pending_execution" not in text
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
    """Extract numeric confidence-interval endpoints while rejecting mixed content."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    if not isinstance(value, list):
        return []
    return [
        float(item)
        for item in value
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]


def nested_int(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> int:
    """Read a nested integer fallback without accepting booleans as counters."""

    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return 0
        current = current.get(key)
    return current if isinstance(current, int) and not isinstance(current, bool) else 0


def _ci_excludes_zero_positive(ci: list[float]) -> bool:
    return len(ci) == 2 and ci[0] > 0.0 and ci[1] > 0.0


def _ci_touches_zero(ci: list[float]) -> bool:
    return len(ci) == 2 and ci[0] <= 0.0 <= ci[1]


def pivot_comparison(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize the ARC held-out RFT comparison without manufacturing missing metrics."""

    if was_skipped:
        status = "skipped_flagged"
    elif not isinstance(payload, Mapping):
        status = "missing"
    elif str_metric(payload, "status") == "blocked" or verdict_text(payload).startswith(
        ("blocked_", "blocked:")
    ):
        status = "blocked"
    elif bool_metric(payload, "training_accumulating") or "accumulating" in verdict_text(payload):
        status = "accumulating"
    else:
        status = "measured"

    measured = status == "measured"
    rft_rate = float_metric(payload, "rft_correct_induction_rate") if measured else 0.0
    cold_rate = float_metric(payload, "cold_base_induction_rate") if measured else 0.0
    gold_rate = float_metric(payload, "gold_sft_induction_rate") if measured else 0.0
    cold_delta = (
        float_metric(payload, "rft_vs_cold_delta") or rft_rate - cold_rate
        if measured
        else 0.0
    )
    gold_delta = (
        float_metric(payload, "rft_vs_gold_delta") or rft_rate - gold_rate
        if measured
        else 0.0
    )
    cold_ci = list_float_metric(payload, "rft_vs_cold_ci95") if measured else []
    gold_ci = list_float_metric(payload, "rft_vs_gold_ci95") if measured else []
    beats_cold = measured and cold_delta > 0.0 and _ci_excludes_zero_positive(cold_ci)
    matches_gold = measured and (
        rft_rate >= gold_rate or (gold_delta >= -0.01 and _ci_touches_zero(gold_ci))
    )

    return {
        "status": "measured_positive"
        if beats_cold and matches_gold
        else "measured_null"
        if measured
        else status,
        "base_model": str_metric(payload, "base_model") if isinstance(payload, Mapping) else "",
        "rft_correct_induction_rate": rft_rate,
        "cold_base_induction_rate": cold_rate,
        "gold_sft_induction_rate": gold_rate,
        "rft_vs_cold_delta": cold_delta,
        "rft_vs_cold_ci95": cold_ci,
        "rft_beats_cold_ci_excludes_zero": beats_cold,
        "rft_vs_gold_delta": gold_delta,
        "rft_vs_gold_ci95": gold_ci,
        "rft_beats_or_matches_gold_sft": matches_gold,
        "training_accumulating": status == "accumulating",
        "blocked_at_layer": str_metric(payload, "blocked_at_layer"),
        "gate_check_summary": str_metric(payload, "gate_check_summary"),
        "honest_verdict": verdict_text(payload),
        "codex_induction_rate": float_metric(payload, "codex_induction_rate")
        if measured
        else 0.0,
        "prior_local_induction_rate": float_metric(payload, "prior_local_induction_rate")
        if measured
        else 0.0,
    }


def pivot_result_text(pivot: Mapping[str, Any]) -> str:
    """Write the load-bearing headline in the requested principle-prefixed form."""

    status = str(pivot.get("status", ""))
    if status == "measured_positive":
        return (
            "principle: verifier-certified RFT beat cold held-out with CI excluding zero "
            "and matched gold-SFT; the verifier's training-time value is measured."
        )
    if status == "measured_null":
        return (
            "principle: verifier-certified RFT was measured on held-out ARC but did not "
            "clear both the cold-base and gold-SFT comparisons."
        )
    if status == "accumulating":
        return (
            "principle: ARC verifier-certified RFT training is still accumulating; the first "
            "window is not decision-grade yet."
        )
    if status == "blocked":
        return (
            "principle: ARC verifier-certified RFT training-time value was not measured; "
            "exp4079 landed only a blocked gate-check because exp4078 never launched training."
        )
    return (
        "principle: ARC verifier-certified RFT training-time value was not measured from the "
        "landed non-flagged artifacts."
    )


def sudoku_control(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Report whether the positive control reproduced from clean evidence."""

    if was_skipped:
        clean = False
        status = "skipped_flagged"
    else:
        clean = invoked(payload)
        status = "measured" if clean else "missing_or_blocked"
    rft_rate = float_metric(payload, "rft_rate") if clean else 0.0
    sft_rate = float_metric(payload, "sft_rate") if clean else 0.0
    n_seeds = int_metric(payload, "n_seeds") if clean else 0
    reproduced = clean and bool_metric(payload, "reproduces_beachhead") and rft_rate >= sft_rate
    return {
        "status": "reproduced" if reproduced else status if not clean else "failed",
        "reproduces_beachhead": reproduced,
        "rft_rate": rft_rate,
        "sft_rate": sft_rate,
        "n_seeds": n_seeds,
        "honest_verdict": verdict_text(payload) if clean else "",
    }


def decentralization_distillation_outcome(pivot: Mapping[str, Any]) -> str:
    """State whether clean local induction moved toward Codex after RFT."""

    if pivot.get("status") not in {"measured_positive", "measured_null"}:
        return (
            "unmeasured: no clean ARC RFT induction-rate delta; cannot assess "
            "Invisible-Leash latent movement toward Codex."
        )
    prior = float(pivot.get("prior_local_induction_rate", 0.0))
    local = float(pivot.get("rft_correct_induction_rate", 0.0))
    codex = float(pivot.get("codex_induction_rate", 0.0))
    if prior > 0.0 and local > prior and codex > local:
        return f"moved_toward_codex: local induction {prior:.4f}->{local:.4f} toward codex {codex:.4f}"
    return f"not_moved_toward_codex: local induction {prior:.4f}->{local:.4f} vs codex {codex:.4f}"


def arc_accuracy(
    activation_payload: Mapping[str, Any] | None,
    accuracy_payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    """Build the ARC games-solved result from clean exp4082, with activation fallback."""

    prior = int_metric(accuracy_payload, "prior_total_games_solved") or nested_int(
        activation_payload,
        ("milestone_376_closestate", "accuracy", "total_games_solved"),
    )
    clean = not was_skipped and invoked(accuracy_payload)
    solved = (
        clean
        and bool_metric(accuracy_payload, "game_solved")
        and bool_metric(accuracy_payload, "real_env_confirmed")
    )
    total = int_metric(accuracy_payload, "total_games_solved") if clean else prior
    return {
        "status": "new_game_solved"
        if solved
        else "measured_no_new_solve"
        if clean
        else "missing_or_blocked",
        "prior_total_games_solved": prior,
        "games_solved_total": total or prior,
        "game_solved": solved,
        "target_game": str_metric(accuracy_payload, "target_game") if clean else "",
        "real_env_confirmed": solved,
        "levels_completed": int_metric(accuracy_payload, "levels_completed") if clean else 0,
        "first_solve_at_action": int_metric(accuracy_payload, "first_solve_at_action")
        if clean
        else 0,
        "candidate_baseline_actions": int_metric(accuracy_payload, "candidate_baseline_actions")
        if clean
        else 0,
    }


def hardware_continuity(payload: Mapping[str, Any] | None) -> JsonDict:
    """Carry clean board-continuity state without making an acceleration claim."""

    clean = invoked(payload)
    reachability = payload.get("per_board_reachability") if isinstance(payload, Mapping) else None
    return {
        "included": clean,
        "kv260_terminal_confirmed": clean and bool_metric(payload, "kv260_terminal_confirmed"),
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken") if clean else "",
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken") if clean else "",
        "fabric_acceleration_claimed": clean
        and bool_metric(payload, "fabric_acceleration_claimed"),
        "per_board_reachability": dict(reachability)
        if clean and isinstance(reachability, Mapping)
        else {},
    }


def sota_ingestion(payload: Mapping[str, Any] | None) -> JsonDict:
    """Carry the non-metric SOTA ingestion receipt for verifier-as-reward planning."""

    methods = payload.get("methods_mapped") if isinstance(payload, Mapping) else None
    clean = invoked(payload)
    return {
        "included": clean,
        "methods_mapped_count": len(methods) if clean and isinstance(methods, list) else 0,
        "honest_verdict": verdict_text(payload) if clean else "",
    }


def registry_hygiene(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Carry registry/gaps state only when the artifact is clean."""

    clean = not was_skipped and invoked(payload)
    return {
        "included": clean,
        "registry_updated": clean and bool_metric(payload, "registry_updated"),
        "gaps_updated": clean and bool_metric(payload, "gaps_updated"),
        "pivot_outcome_recorded": clean and bool_metric(payload, "pivot_outcome_recorded"),
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
            "honest_verdict": verdict_text(payload) if isinstance(payload, Mapping) else "missing",
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
    pivot: Mapping[str, Any],
    sudoku: Mapping[str, Any],
    games_solved_total: int,
    skipped_count: int,
) -> str:
    """Build the terminal-prefix .377 headline from measured outcomes."""

    pivot_status = str(pivot.get("status", ""))
    sudoku_status = str(sudoku.get("status", ""))
    sudoku_verdict_status = "flagged_skipped" if sudoku_status == "skipped_flagged" else sudoku_status
    if pivot_status == "measured_positive" and sudoku_status == "reproduced":
        return (
            "success: capstone_v377_verifier_as_reward_rft_beats_cold_matches_gold_"
            f"sudoku_reproduced_games{games_solved_total}_flagged_skipped{skipped_count}"
        )
    if pivot_status == "accumulating":
        return (
            "complete: capstone_v377_pivot_train_accumulating_"
            f"sudoku_{sudoku_verdict_status}_games{games_solved_total}_flagged_skipped{skipped_count}"
        )
    if pivot_status == "blocked":
        return (
            "complete: capstone_v377_pivot_blocked_no_arc_rft_eval_"
            f"sudoku_{sudoku_verdict_status}_games{games_solved_total}_flagged_skipped{skipped_count}"
        )
    return (
        "complete: capstone_v377_verifier_as_reward_rft_measured_null_"
        f"sudoku_{sudoku_verdict_status}_games{games_solved_total}_flagged_skipped{skipped_count}"
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
    """Build the .377 capstone from landed upstream artifacts."""

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

    pivot = pivot_comparison(clean_upstreams.get(4079), was_skipped=4079 in skipped_ids)
    sudoku = sudoku_control(clean_upstreams.get(4080), was_skipped=4080 in skipped_ids)
    accuracy = arc_accuracy(
        clean_upstreams.get(4076),
        clean_upstreams.get(4082),
        was_skipped=4082 in skipped_ids,
    )
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    games_solved_total = int(accuracy["games_solved_total"])

    artifact: JsonDict = {
        "schema": "carnot.capstone_v377_4085.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": verdict(
            pivot=pivot,
            sudoku=sudoku,
            games_solved_total=games_solved_total,
            skipped_count=len(skipped),
        ),
        "pivot_result": pivot_result_text(pivot),
        "sudoku_control_reproduced": bool(sudoku["reproduces_beachhead"]),
        "decentralization_distillation_outcome": decentralization_distillation_outcome(pivot),
        "games_solved_total": games_solved_total,
        "pivot_comparison": pivot,
        "sudoku_control": sudoku,
        "arc_accuracy": accuracy,
        "sota_ingestion": sota_ingestion(clean_upstreams.get(4081)),
        "registry_hygiene": registry_hygiene(
            clean_upstreams.get(4083), was_skipped=4083 in skipped_ids
        ),
        "hardware_continuity": hardware_continuity(clean_upstreams.get(4084)),
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
    """Validate the .377 fields that protect the honest headline."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover
    verdict_value = str(artifact.get("honest_verdict", ""))
    if not verdict_value.startswith(("complete:", "success:", "blocked_", "blocked:")):
        raise ValueError("honest_verdict must use a terminal prefix")  # pragma: no cover
    if not isinstance(artifact.get("pivot_result"), str):
        raise ValueError("pivot_result must be a string")  # pragma: no cover
    if not isinstance(artifact.get("sudoku_control_reproduced"), bool):
        raise ValueError("sudoku_control_reproduced must be a bare bool")  # pragma: no cover
    if not isinstance(artifact.get("decentralization_distillation_outcome"), str):
        raise ValueError("decentralization_distillation_outcome must be a string")  # pragma: no cover
    if not isinstance(artifact.get("games_solved_total"), int) or isinstance(
        artifact.get("games_solved_total"), bool
    ):
        raise ValueError("games_solved_total must be a bare int")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    citations = artifact.get("cited_upstream_artifacts")
    if not isinstance(citations, list):
        raise ValueError("cited_upstream_artifacts must be a list")  # pragma: no cover
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise ValueError("citation entries must be objects")  # pragma: no cover
        if set(citation) != {"experiment_id", "sha256"}:
            raise ValueError("citation entries must contain experiment_id and sha256")  # pragma: no cover
        if not isinstance(citation.get("experiment_id"), int):
            raise ValueError("citation entries need integer experiment_id")  # pragma: no cover
        if not is_sha256(citation.get("sha256")):
            raise ValueError("citation entries need sha256")  # pragma: no cover
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
    """Build, validate, and write the Exp 4085 capstone artifact."""

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

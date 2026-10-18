"""Build the Exp 4097 v378 capstone aggregation.

Spec refs: REQ-CAPSTONE-4097, SCENARIO-CAPSTONE-4097.

The capstone answers the .378 milestone's load-bearing question from landed
artifacts only: whether Exp 4087 rescued ARC certification precision enough to
make verifier labels eligible for RFT, and whether a clean Exp 4090 A-vs-B eval
then measured the verifier label carrying training signal. Missing or blocked
Phase B artifacts stay explicit state, not invented gates.
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
OUTPUT_REL_PATH = Path("results/experiment_4097_capstone_v378.json")
EXPERIMENT_ID = 4097
RANDOM_SEED = 4097
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PYTHON_BIN = Path(".venv/bin/python")

PRECISION_BASELINE = 0.68
PRECISION_FLOOR = 0.85
RECALL_FLOOR = 0.20

UPSTREAM_IDS = tuple(range(4086, 4097))
DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    4086: Path("results/experiment_4086_archive_v377_activate_v378.json"),
    4087: Path("results/experiment_4087_certification_precision_rescue.json"),
    4088: Path("results/experiment_4088_verifier_reward_rft_corpus_build.json"),
    4089: Path("results/experiment_4089_verifier_reward_rft_train.json"),
    4090: Path("results/experiment_4090_verifier_reward_rft_eval_gate.json"),
    4091: Path("results/experiment_4091_sudoku_rft_pipeline_sanity.json"),
    4092: Path("results/experiment_4092_tenth_game_explore_first.json"),
    4093: Path("results/experiment_4093_offarc_demofit_precision_transfer.json"),
    4094: Path("results/experiment_4094_sota_ingestion_precision_calibration_receipt.json"),
    4095: Path("results/experiment_4095_verifier_registry_gaps_hygiene.json"),
    4096: Path("results/experiment_4096_hardware_continuity.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "precision_rescue_outcome",
    "pivot_result",
    "offarc_precision_transfer",
    "games_solved_total",
    "sudoku_sanity_reproduced",
    "cited_upstream_artifacts",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefix verdict that distinguishes precision rescue, bounded rescue, "
        "clean RFT A-vs-B signal, and absent/blocked Phase B state."
    ),
    "precision_rescue_outcome": (
        "principle: THE gate -- whether certification precision moved from the 0.68 "
        "baseline to at least 0.85 at recall at least 0.20."
    ),
    "pivot_result": (
        "principle: the A-vs-B verifier-label result when clean exp4090 exists, "
        "or the honest skipped/bounded Phase B state."
    ),
    "offarc_precision_transfer": "principle: whether the precision primitive transfers off-ARC.",
    "games_solved_total": "BARE INT - monotonic ARC-AGI-3 solved-game count.",
    "sudoku_sanity_reproduced": "BARE BOOL - true only from a clean Sudoku sanity artifact.",
    "cited_upstream_artifacts": "Included non-flagged upstream experiment ids and sha256 provenance.",
    "inference_substrate": "Declares this capstone as aggregation from upstream artifacts.",
}


def is_sha256(value: object) -> bool:
    """Return true for lowercase SHA-256 hex digests."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def read_json_object(path: Path) -> JsonDict:
    """Load one JSON object artifact."""

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
    """Select the intended artifact for every .378 upstream id."""

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
    """Return whether an upstream is stamped adversarial."""

    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


def live_critical(summary: Mapping[str, Any] | None) -> bool:
    """Return whether summarize_artifact.py observed a live critical concern."""

    return isinstance(summary, Mapping) and summary.get("returncode") == 2


def verdict_text(payload: Mapping[str, Any] | None) -> str:
    """Extract an honest verdict string."""

    value = payload.get("honest_verdict") if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    """Extract a string metric without coercion."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool:
    """Extract a JSON boolean without truthifying strings or numbers."""

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
    """Extract numeric confidence-interval endpoints while rejecting mixed content."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    if not isinstance(value, list):
        return []
    return [
        float(item)
        for item in value
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]


def nested_bool(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> bool:
    """Read a nested boolean without coercion."""

    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return False
        current = current.get(key)
    return current is True


def nested_int(payload: Mapping[str, Any] | None, path: tuple[str, ...]) -> int:
    """Read a nested integer without accepting booleans."""

    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return 0
        current = current.get(key)
    return current if isinstance(current, int) and not isinstance(current, bool) else 0


def invoked(payload: Mapping[str, Any] | None) -> bool:
    """Return false for missing or blocked upstream artifacts."""

    text = verdict_text(payload)
    return (
        bool(text)
        and not text.startswith(("blocked_", "blocked:"))
        and str_metric(payload, "status") != "blocked"
        and "pending_execution" not in text
    )


def _first_float(payload: Mapping[str, Any] | None, fields: tuple[str, ...]) -> float:
    for field in fields:
        value = float_metric(payload, field)
        if value:
            return value
    return 0.0


def _first_ci(payload: Mapping[str, Any] | None, fields: tuple[str, ...]) -> list[float]:
    for field in fields:
        value = list_float_metric(payload, field)
        if value:
            return value
    return []


def _ci_excludes_zero_positive(ci: list[float]) -> bool:
    return len(ci) == 2 and ci[0] > 0.0 and ci[1] > 0.0


def precision_rescue(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize the Exp 4087 precision rescue gate."""

    if was_skipped:
        status = "skipped_flagged"
    elif not isinstance(payload, Mapping):
        status = "missing"
    elif not invoked(payload) and not verdict_text(payload).startswith("complete: precision_rescue"):
        status = "blocked"
    else:
        status = "rescued" if bool_metric(payload, "precision_rescue_succeeded") else "bounded"
    precision = float_metric(payload, "best_certified_precision")
    recall = float_metric(payload, "best_op_point_recall")
    rescued = (
        status == "rescued"
        and precision >= PRECISION_FLOOR
        and recall >= RECALL_FLOOR
    )
    return {
        "status": "rescued" if rescued else status,
        "rescued": rescued,
        "baseline_precision": PRECISION_BASELINE,
        "precision_floor": PRECISION_FLOOR,
        "recall_floor": RECALL_FLOOR,
        "best_certified_precision": precision,
        "best_op_point_recall": recall,
        "honest_verdict": verdict_text(payload),
    }


def precision_rescue_outcome_text(rescue: Mapping[str, Any]) -> str:
    """Build the principle-prefixed precision rescue headline."""

    precision = float(rescue.get("best_certified_precision", 0.0))
    recall = float(rescue.get("best_op_point_recall", 0.0))
    if rescue.get("rescued") is True:
        return (
            "principle: THE gate passed; certification precision moved from the 0.68 "
            f"baseline to {precision:.4f} at recall {recall:.4f}."
        )
    return (
        "principle: THE gate failed; max certification precision was "
        f"{precision:.4f} at recall {recall:.4f}, below the {PRECISION_FLOOR:.4f} "
        "precision floor."
    )


def pivot_comparison(
    payload: Mapping[str, Any] | None,
    *,
    precision_rescued: bool,
    was_skipped: bool,
) -> JsonDict:
    """Summarize the clean RFT A-vs-B eval without manufacturing missing metrics."""

    if not precision_rescued:
        status = "skipped_bounded"
    elif was_skipped:
        status = "skipped_flagged"
    elif not isinstance(payload, Mapping):
        status = "missing"
    elif bool_metric(payload, "training_accumulating") or "accumulating" in verdict_text(payload):
        status = "accumulating"
    elif not invoked(payload):
        status = "blocked"
    else:
        delta = _first_float(
            payload,
            (
                "a_vs_b_delta",
                "rft_correct_vs_ablation_delta",
                "rft_a_vs_b_delta",
                "rft_correct_minus_ablation_delta",
            ),
        )
        ci = _first_ci(
            payload,
            (
                "a_vs_b_ci95",
                "rft_correct_vs_ablation_ci95",
                "rft_a_vs_b_ci95",
                "rft_correct_minus_ablation_ci95",
            ),
        )
        status = "a_gt_b" if delta > 0.0 and _ci_excludes_zero_positive(ci) else "measured_null"

    measured = status in {"a_gt_b", "measured_null"}
    delta = _first_float(
        payload,
        (
            "a_vs_b_delta",
            "rft_correct_vs_ablation_delta",
            "rft_a_vs_b_delta",
            "rft_correct_minus_ablation_delta",
        ),
    ) if measured else 0.0
    ci = _first_ci(
        payload,
        (
            "a_vs_b_ci95",
            "rft_correct_vs_ablation_ci95",
            "rft_a_vs_b_ci95",
            "rft_correct_minus_ablation_ci95",
        ),
    ) if measured else []
    return {
        "status": status,
        "a_vs_b_delta": delta,
        "a_vs_b_ci95": ci,
        "a_vs_b_ci_excludes_zero": status == "a_gt_b",
        "rft_correct_pass_at_1": float_metric(payload, "rft_correct_pass_at_1")
        if measured
        else 0.0,
        "rft_ablation_pass_at_1": float_metric(payload, "rft_ablation_pass_at_1")
        if measured
        else 0.0,
        "n_heldout_tasks": int_metric(payload, "n_heldout_tasks") if measured else 0,
        "honest_verdict": verdict_text(payload),
    }


def pivot_result_text(pivot: Mapping[str, Any]) -> str:
    """Write the load-bearing RFT A-vs-B result."""

    status = str(pivot.get("status", ""))
    if status == "a_gt_b":
        return (
            "principle: clean exp4090 measured A>B with CI excluding zero; the "
            "verifier label carries training signal."
        )
    if status == "measured_null":
        return (
            "principle: clean exp4090 did not measure A>B with CI excluding zero; "
            "verifier-label signal remains unproven."
        )
    if status == "accumulating":
        return "principle: Phase B is still accumulating; A-vs-B is not decision-grade yet."
    if status == "skipped_bounded":
        return (
            "principle: verifier-as-reward is precision-bounded on ARC; Phase B is skipped "
            "honestly, and the forward path is step-level process-reward / outcome-verifier pairing."
        )
    if status == "blocked":
        return (
            "principle: precision rescue passed, but Phase B blocked before a clean held-out "
            "RFT A-vs-B measurement."
        )
    if status == "skipped_flagged":
        return "principle: exp4090 was flagged_adversarial and skipped; A-vs-B is unmeasured."
    return (
        "principle: precision rescue passed, but no clean exp4090 A-vs-B artifact "
        "exists; exp4088/exp4089 stopped before a decision-grade held-out RFT eval, "
        "so verifier-label training signal is unmeasured."
    )


def offarc_precision(payload: Mapping[str, Any] | None, *, was_skipped: bool) -> JsonDict:
    """Summarize Exp 4093 off-ARC precision transfer."""

    if was_skipped:
        status = "skipped_flagged"
    elif not isinstance(payload, Mapping):
        status = "missing"
    elif not invoked(payload):
        status = "blocked"
    elif bool_metric(payload, "primitive_is_domain_general"):
        status = "transfers"
    else:
        status = "bounded"
    raw = float_metric(payload, "demofit_precision_raw") if status in {"transfers", "bounded"} else 0.0
    filtered = (
        float_metric(payload, "demofit_precision_filtered") if status in {"transfers", "bounded"} else 0.0
    )
    floor = float_metric(payload, "domain_general_precision_floor") or PRECISION_BASELINE
    return {
        "status": status,
        "primitive_is_domain_general": status == "transfers",
        "demofit_precision_raw": raw,
        "demofit_precision_filtered": filtered,
        "filter_recall": float_metric(payload, "filter_recall")
        if status in {"transfers", "bounded"}
        else 0.0,
        "domain_general_precision_floor": floor,
        "n_tasks_scored": int_metric(payload, "n_tasks_scored")
        if status in {"transfers", "bounded"}
        else 0,
        "honest_verdict": verdict_text(payload),
    }


def offarc_precision_transfer_text(offarc: Mapping[str, Any]) -> str:
    """Write the off-ARC transfer result."""

    status = str(offarc.get("status", ""))
    if status == "transfers":
        return (
            "principle: precision primitive transfers off-ARC; raw demo-fit precision "
            f"{float(offarc.get('demofit_precision_raw', 0.0)):.4f} and filtered precision "
            f"{float(offarc.get('demofit_precision_filtered', 0.0)):.4f} clear the "
            f"{float(offarc.get('domain_general_precision_floor', PRECISION_BASELINE)):.4f} floor."
        )
    if status == "bounded":
        return (
            "principle: off-ARC precision replay landed but did not clear the domain-general "
            "precision floor."
        )
    return "principle: off-ARC precision transfer is unmeasured from clean landed artifacts."


def arc_accuracy(
    activation_payload: Mapping[str, Any] | None,
    accuracy_payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    """Build the monotonic games-solved result from clean Exp 4092 evidence."""

    prior = int_metric(accuracy_payload, "prior_total_games_solved") or int_metric(
        activation_payload, "total_games_solved"
    ) or nested_int(activation_payload, ("milestone_377_closestate", "accuracy", "total_games_solved"))
    clean = not was_skipped and invoked(accuracy_payload)
    solved = (
        clean
        and bool_metric(accuracy_payload, "game_solved")
        and bool_metric(accuracy_payload, "real_env_confirmed")
    )
    total = int_metric(accuracy_payload, "total_games_solved") if clean else prior
    return {
        "status": "new_game_solved" if solved else "measured_no_new_solve" if clean else "missing",
        "prior_total_games_solved": prior,
        "games_solved_total": total or prior,
        "game_solved": solved,
        "real_env_confirmed": solved,
        "target_game": str_metric(accuracy_payload, "target_game") if clean else "",
        "honest_verdict": verdict_text(accuracy_payload),
    }


def sudoku_sanity(
    payload: Mapping[str, Any] | None,
    activation_payload: Mapping[str, Any] | None,
    *,
    was_skipped: bool,
) -> JsonDict:
    """Report whether the .378 Sudoku pipeline sanity reproduced from clean evidence."""

    previous = nested_bool(
        activation_payload,
        ("milestone_377_closestate", "sudoku_control", "reproduces_beachhead"),
    )
    if was_skipped:
        status = "skipped_flagged"
        reproduced = False
    elif not isinstance(payload, Mapping):
        status = "missing"
        reproduced = False
    elif not invoked(payload):
        status = "blocked"
        reproduced = False
    else:
        reproduced = bool_metric(payload, "sudoku_sanity_reproduced") or bool_metric(
            payload, "reproduces_beachhead"
        )
        status = "reproduced" if reproduced else "failed"
    return {
        "status": status,
        "sudoku_sanity_reproduced": reproduced,
        "previous_v377_reproduced": previous,
        "honest_verdict": verdict_text(payload),
    }


def rft_pipeline_state(
    corpus_payload: Mapping[str, Any] | None,
    train_payload: Mapping[str, Any] | None,
) -> JsonDict:
    """Carry Phase B corpus/train state without treating it as a capstone gate."""

    return {
        "corpus_status": "ready" if bool_metric(corpus_payload, "runner_ready") else "blocked_or_missing",
        "corpus_honest_verdict": verdict_text(corpus_payload),
        "trainer_smoke_passed": bool_metric(corpus_payload, "trainer_smoke_passed"),
        "train_status": "launched"
        if bool_metric(train_payload, "train_launched")
        else "accumulating"
        if bool_metric(train_payload, "training_accumulating")
        else "blocked_or_missing",
        "train_honest_verdict": verdict_text(train_payload),
    }


def hardware_continuity(payload: Mapping[str, Any] | None) -> JsonDict:
    """Carry clean board-continuity state without making an acceleration claim."""

    reachability = payload.get("per_board_reachability") if isinstance(payload, Mapping) else None
    clean = invoked(payload)
    return {
        "included": clean,
        "kv260_terminal_confirmed": clean and bool_metric(payload, "kv260_terminal_confirmed"),
        "gatemate_step_taken": str_metric(payload, "gatemate_step_taken") if clean else "",
        "polarfire_step_taken": str_metric(payload, "polarfire_step_taken") if clean else "",
        "per_board_reachability": dict(reachability)
        if clean and isinstance(reachability, Mapping)
        else {},
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
    rescue: Mapping[str, Any],
    pivot: Mapping[str, Any],
    offarc: Mapping[str, Any],
    sudoku: Mapping[str, Any],
    games_solved_total: int,
    skipped_count: int,
) -> str:
    """Build the terminal-prefix .378 headline."""

    precision = float(rescue.get("best_certified_precision", 0.0))
    offarc_token = (
        "offarc_transfer" if offarc.get("primitive_is_domain_general") is True else "offarc_bounded"
    )
    pivot_status = str(pivot.get("status", ""))
    if rescue.get("rescued") is not True:
        return (
            f"complete: capstone_v378_precision_bounded_max_{precision:.4f}_"
            f"phaseB_skipped_honest_bound_games{games_solved_total}_{offarc_token}_"
            f"flagged_skipped{skipped_count}"
        )
    if pivot_status == "a_gt_b":
        sudoku_token = (
            "sudoku_reproduced" if sudoku.get("sudoku_sanity_reproduced") is True else "sudoku_unreproduced"
        )
        return (
            f"success: capstone_v378_precision_rescued_{precision:.4f}_"
            f"rft_A_gt_B_games{games_solved_total}_{offarc_token}_{sudoku_token}_"
            f"flagged_skipped{skipped_count}"
        )
    if pivot_status == "accumulating":
        return (
            f"complete: capstone_v378_precision_rescued_{precision:.4f}_"
            f"phaseB_accumulating_games{games_solved_total}_{offarc_token}_"
            f"flagged_skipped{skipped_count}"
        )
    return (
        f"complete: capstone_v378_precision_rescued_{precision:.4f}_"
        f"phaseB_no_clean_A_vs_B_games{games_solved_total}_{offarc_token}_"
        f"flagged_skipped{skipped_count}"
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
    """Build the .378 capstone from landed upstream artifacts."""

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

    rescue = precision_rescue(clean_upstreams.get(4087), was_skipped=4087 in skipped_ids)
    pivot = pivot_comparison(
        clean_upstreams.get(4090),
        precision_rescued=bool(rescue["rescued"]),
        was_skipped=4090 in skipped_ids,
    )
    offarc = offarc_precision(clean_upstreams.get(4093), was_skipped=4093 in skipped_ids)
    accuracy = arc_accuracy(
        clean_upstreams.get(4086),
        clean_upstreams.get(4092),
        was_skipped=4092 in skipped_ids,
    )
    sudoku = sudoku_sanity(
        clean_upstreams.get(4091),
        clean_upstreams.get(4086),
        was_skipped=4091 in skipped_ids,
    )
    skipped = flagged_artifacts_skipped(root_path, paths, upstreams, skipped_ids)
    games_solved_total = int(accuracy["games_solved_total"])

    artifact: JsonDict = {
        "schema": "carnot.capstone_v378_4097.v1",
        "experiment_id": EXPERIMENT_ID,
        "honest_verdict": verdict(
            rescue=rescue,
            pivot=pivot,
            offarc=offarc,
            sudoku=sudoku,
            games_solved_total=games_solved_total,
            skipped_count=len(skipped),
        ),
        "precision_rescue_outcome": precision_rescue_outcome_text(rescue),
        "pivot_result": pivot_result_text(pivot),
        "offarc_precision_transfer": offarc_precision_transfer_text(offarc),
        "games_solved_total": games_solved_total,
        "sudoku_sanity_reproduced": bool(sudoku["sudoku_sanity_reproduced"]),
        "precision_rescue": rescue,
        "pivot_comparison": pivot,
        "offarc_precision": offarc,
        "arc_accuracy": accuracy,
        "sudoku_sanity": sudoku,
        "rft_pipeline_state": rft_pipeline_state(clean_upstreams.get(4088), clean_upstreams.get(4089)),
        "sota_ingestion": {
            "included": invoked(clean_upstreams.get(4094)),
            "honest_verdict": verdict_text(clean_upstreams.get(4094)),
        },
        "hardware_continuity": hardware_continuity(clean_upstreams.get(4096)),
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
    """Validate the .378 fields that protect the honest headline."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover
    if "gated_on" in artifact:
        raise ValueError("capstone must not emit gated_on")  # pragma: no cover
    verdict_value = str(artifact.get("honest_verdict", ""))
    if not verdict_value.startswith(("complete:", "success:", "blocked_", "blocked:")):
        raise ValueError("honest_verdict must use a terminal prefix")  # pragma: no cover
    for field in ("precision_rescue_outcome", "pivot_result", "offarc_precision_transfer"):
        if not isinstance(artifact.get(field), str):
            raise ValueError(f"{field} must be a string")  # pragma: no cover
    if not isinstance(artifact.get("games_solved_total"), int) or isinstance(
        artifact.get("games_solved_total"), bool
    ):
        raise ValueError("games_solved_total must be a bare int")
    if not isinstance(artifact.get("sudoku_sanity_reproduced"), bool):
        raise ValueError("sudoku_sanity_reproduced must be a bare bool")  # pragma: no cover
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")  # pragma: no cover
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
    """Build, validate, and write the Exp 4097 capstone artifact."""

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

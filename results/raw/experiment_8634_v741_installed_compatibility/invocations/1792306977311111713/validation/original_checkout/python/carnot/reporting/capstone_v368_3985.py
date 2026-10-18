"""Build the Exp 3985 v368 GAP-4 execution-verifier capstone.

Spec refs: REQ-CAPSTONE-3985, SCENARIO-CAPSTONE-3985.

This is an aggregation artifact, not a new ARC run. Every existing upstream in
the .368 range is routed through ``scripts/summarize_artifact.py`` before any
metric is imported. Missing upstreams are recorded as missing state, and stamped
``flagged_adversarial:true`` artifacts are excluded before owed-axis gates are
computed.
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
OUTPUT_REL_PATH = Path("results/experiment_3985_capstone_v368.json")
EXPERIMENT_ID = 3985
RANDOM_SEED = 3985
BANKED_GAMES_SOLVED_BEFORE_V368 = 3
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts_via_summarize_artifact_py"
PYTHON_BIN = Path(".venv/bin/python")

UPSTREAM_IDS = tuple(range(3975, 3985))
FIELDS_IMPORTED: Mapping[int, list[str]] = {
    3975: ["positive_control_passed", "program_synthesis_coverage"],
    3976: [
        "gap4_beats_vote",
        "executed_consistency_pass2",
        "headroom_capture_fraction",
        "program_synthesis_coverage",
    ],
    3977: ["gap4_positive_confirmed"],
    3978: [
        "accuracy_parity",
        "cost_ratio_judge_over_verifier",
        "verifier_actually_invoked",
    ],
    3979: ["n_trustworthy_at_0.15", "positive_control_passed"],
    3980: ["new_levels_solved_this_task"],
    3981: ["game_solved", "ACCURACY_levels_solved"],
    3982: ["solve_transfer_win"],
    3983: [],
    3984: [],
}

REQUIRED_ARTIFACT_FIELDS = (
    "verifier_earns_accuracy",
    "verifier_earns_efficiency",
    "gap4_program_synthesis_coverage",
    "n_trustworthy_world_models",
    "total_games_solved",
    "total_new_levels_this_milestone",
    "arcmemo_solve_transfer_win",
    "cited_upstream_artifacts",
    "flagged_artifacts_skipped",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

BOOL_FIELDS = (
    "verifier_earns_accuracy",
    "verifier_earns_efficiency",
    "arcmemo_solve_transfer_win",
)
INT_FIELDS = (
    "n_trustworthy_world_models",
    "total_games_solved",
    "total_new_levels_this_milestone",
)
FLOAT_FIELDS = ("gap4_program_synthesis_coverage",)

FIELD_PRINCIPLES = {
    "verifier_earns_accuracy": (
        "BARE BOOL - did the GAP-4 executed-rule verifier select past vote "
        "toward the 16pp headroom AND survive the re-derivation audit (the "
        "accuracy-moat existential answer; a flagged/unconfirmed result is FALSE)."
    ),
    "verifier_earns_efficiency": (
        "BARE BOOL - did the energy verifier reach accuracy parity with the "
        "LLM-judge at >=10x lower cost WITH the verifier provably invoked (the "
        "efficiency existential answer; the .366 fabrication + .367 block did "
        "not recur)."
    ),
    "gap4_program_synthesis_coverage": (
        "BARE FLOAT - the binding constraint on the accuracy axis (how many "
        "tasks a program induced for)."
    ),
    "n_trustworthy_world_models": (
        "BARE INT - how many of the 6 non-spatial games reached a plan-able "
        "model with execution-guided synthesis (vs exp3968's 0)."
    ),
    "total_games_solved": (
        "BARE INT - the ARC-AGI-3 games-solved headline, starting from the 3 "
        "banked games and adding one only for a real fourth-game solve."
    ),
    "total_new_levels_this_milestone": (
        "BARE INT - monotonic level progress from incremental re-induction and "
        "any fourth-game first solve."
    ),
    "arcmemo_solve_transfer_win": (
        "BARE BOOL - did concept memory make a SOLVE cheaper (the self-learning "
        "datum)."
    ),
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported, sha256} - the audit trail so "
        "the capstone numbers trace to real measurements."
    ),
    "flagged_artifacts_skipped": "Which upstream artifacts were flagged_adversarial and EXCLUDED.",
    "honest_verdict": "Terminal-prefix verdict.",
    "duration_s": "Aggregation wall-clock duration.",
    "inference_substrate": "Aggregation substrate.",
}


def is_sha256(value: object) -> bool:
    """Return true when a value is a lowercase SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object artifact from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover - defensive.
    return payload


def sha256_file(path: Path) -> str:
    """Hash an upstream artifact so every imported number has provenance."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_root(root: Path, path: Path) -> str:
    """Return a stable repository-relative path for audit fields."""

    return str(path.relative_to(root))


def matching_artifact_paths(root: Path, experiment_id: int) -> list[Path]:
    """Find candidate artifact files for one upstream experiment."""

    patterns = (
        root / "results" / f"experiment_{experiment_id}_*.json",
        root / "results" / f"experiment_{experiment_id}.json",
    )
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path(path) for path in glob.glob(str(pattern)))
    return sorted(matches)


def selected_upstream_paths(root: Path) -> dict[int, Path | None]:
    """Select one upstream artifact per expected experiment, if it exists."""

    selected: dict[int, Path | None] = {}
    for experiment_id in UPSTREAM_IDS:
        matches = matching_artifact_paths(root, experiment_id)
        selected[experiment_id] = matches[-1] if matches else None
    return selected


def run_summarize_artifact(root: Path, path: Path) -> JsonDict:
    """Run the mandated disciplined reader for a single existing artifact."""

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
    """Return whether the upstream is stamped as adversarial and must be skipped."""

    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


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


def gap4_program_synthesis_coverage(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> float:
    """Return the GAP-4 coverage metric, preferring the rerank-eval field."""

    coverage = float_metric(clean_upstreams.get(3976), "program_synthesis_coverage")
    if coverage:
        return coverage
    return float_metric(clean_upstreams.get(3975), "program_synthesis_coverage")


def verifier_earns_accuracy(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Apply the milestone's accuracy-axis gate exactly from exp3976 and exp3977."""

    return bool_metric(clean_upstreams.get(3976), "gap4_beats_vote") and bool_metric(
        clean_upstreams.get(3977),
        "gap4_positive_confirmed",
    )


def verifier_earns_efficiency(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Apply the efficiency-axis gate, requiring parity, >=10x cost ratio, and invocation."""

    payload = clean_upstreams.get(3978)
    return (
        bool_metric(payload, "accuracy_parity")
        and float_metric(payload, "cost_ratio_judge_over_verifier") >= 10.0
        and bool_metric(payload, "verifier_actually_invoked")
    )


def trustworthy_world_models(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> int:
    """Return trustworthy model count only when the positive control passed."""

    payload = clean_upstreams.get(3979)
    if not bool_metric(payload, "positive_control_passed"):
        return 0
    return int_metric(payload, "n_trustworthy_at_0.15")


def fourth_game_solved(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Return true only for a real fourth-game solve, not a no-solve artifact."""

    payload = clean_upstreams.get(3981)
    game = str(payload.get("game_solved", "")) if isinstance(payload, Mapping) else ""
    return (
        bool_metric(payload, "real_env_confirmed")
        and game not in ("", "none")
        and int_metric(payload, "ACCURACY_levels_solved") > 0
    )


def total_new_levels_this_milestone(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> int:
    """Sum incremental new levels plus any fourth-game levels that actually solved."""

    total = int_metric(clean_upstreams.get(3980), "new_levels_solved_this_task")
    if fourth_game_solved(clean_upstreams):
        total += int_metric(clean_upstreams.get(3981), "ACCURACY_levels_solved")
    return total


def cited_upstream_artifacts(
    paths: Mapping[int, Path | None],
    clean_upstreams: Mapping[int, Mapping[str, Any]],
) -> list[JsonDict]:
    """Build the provenance list for every clean upstream artifact that exists."""

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
    """Record upstream artifacts skipped before aggregation due to adversarial stamp."""

    skipped: list[JsonDict] = []
    for experiment_id in sorted(flagged_ids):
        path = paths[experiment_id]
        skipped.append(
            {
                "experiment_id": experiment_id,
                "path": relative_to_root(root, path) if path is not None else "",
                "reason": "flagged_adversarial:true",
            }
        )
    return skipped


def missing_upstream_artifacts(paths: Mapping[int, Path | None]) -> list[JsonDict]:
    """Record missing upstream artifacts without turning absence into a gate."""

    return [{"experiment_id": experiment_id} for experiment_id in UPSTREAM_IDS if paths[experiment_id] is None]


def upstream_artifact_state(
    root: Path,
    paths: Mapping[int, Path | None],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    summaries: Mapping[int, Mapping[str, Any]],
    flagged_ids: set[int],
    clean_upstreams: Mapping[int, Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Expose per-upstream inclusion state so missing and skipped inputs are auditable."""

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
            "summarize_artifact_returncode": summaries.get(experiment_id, {}).get("returncode"),
        }
    return state


def duration_from(started_s: float, now_s: float | None) -> float:
    """Compute an honest aggregation duration with a small nonzero floor."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0001, end - started_s)


def verdict(
    earns_accuracy: bool,
    earns_efficiency: bool,
    total_games_solved: int,
    total_new_levels: int,
    missing_count: int,
    skipped_count: int,
) -> str:
    """Build the terminal-prefix milestone verdict."""

    prefix = "success" if earns_accuracy or earns_efficiency else "complete"
    earned = (
        "accuracy_and_efficiency"
        if earns_accuracy and earns_efficiency
        else "accuracy_only"
        if earns_accuracy
        else "efficiency_only"
        if earns_efficiency
        else "no_owed_axis"
    )
    return (
        f"{prefix}: capstone_v368_verifier_earned_{earned}_"
        f"games{total_games_solved}_new_levels{total_new_levels}_"
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
    """Build the .368 capstone from whatever upstream artifacts exist."""

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

    earns_accuracy = verifier_earns_accuracy(clean_upstreams)
    earns_efficiency = verifier_earns_efficiency(clean_upstreams)
    fourth_solved = fourth_game_solved(clean_upstreams)
    total_games = BANKED_GAMES_SOLVED_BEFORE_V368 + (1 if fourth_solved else 0)
    total_new_levels = total_new_levels_this_milestone(clean_upstreams)
    missing = missing_upstream_artifacts(paths)
    skipped = flagged_artifacts_skipped(root_path, paths, flagged_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v368_3985.v1",
        "experiment_id": EXPERIMENT_ID,
        "verifier_earns_accuracy": earns_accuracy,
        "gap4_beats_vote": bool_metric(clean_upstreams.get(3976), "gap4_beats_vote"),
        "gap4_executed_consistency_pass2": bool_metric(
            clean_upstreams.get(3976),
            "executed_consistency_pass2",
        ),
        "gap4_headroom_capture_fraction": float_metric(
            clean_upstreams.get(3976),
            "headroom_capture_fraction",
        ),
        "gap4_program_synthesis_coverage": gap4_program_synthesis_coverage(clean_upstreams),
        "gap4_positive_confirmed": bool_metric(clean_upstreams.get(3977), "gap4_positive_confirmed"),
        "verifier_earns_efficiency": earns_efficiency,
        "efficiency_accuracy_parity": bool_metric(clean_upstreams.get(3978), "accuracy_parity"),
        "efficiency_cost_ratio_judge_over_verifier": float_metric(
            clean_upstreams.get(3978),
            "cost_ratio_judge_over_verifier",
        ),
        "efficiency_verifier_actually_invoked": bool_metric(
            clean_upstreams.get(3978),
            "verifier_actually_invoked",
        ),
        "n_trustworthy_world_models": trustworthy_world_models(clean_upstreams),
        "world_model_positive_control_passed": bool_metric(clean_upstreams.get(3979), "positive_control_passed"),
        "total_games_solved": total_games,
        "banked_games_solved_before_v368": BANKED_GAMES_SOLVED_BEFORE_V368,
        "fourth_game_solved": fourth_solved,
        "total_new_levels_this_milestone": total_new_levels,
        "incremental_new_levels": int_metric(clean_upstreams.get(3980), "new_levels_solved_this_task"),
        "fourth_game_accuracy_levels_solved": int_metric(clean_upstreams.get(3981), "ACCURACY_levels_solved")
        if fourth_solved
        else 0,
        "arcmemo_solve_transfer_win": bool_metric(clean_upstreams.get(3982), "solve_transfer_win"),
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
            earns_accuracy,
            earns_efficiency,
            total_games,
            total_new_levels,
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
    """Validate the .368 capstone fields that prevent metric laundering."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover - defensive.
    verdict_text = str(artifact.get("honest_verdict", ""))
    if not verdict_text.startswith(("complete:", "success:", "blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in BOOL_FIELDS:
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in INT_FIELDS:
        if not isinstance(artifact.get(field), int) or isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare int")
    for field in FLOAT_FIELDS:
        if not isinstance(artifact.get(field), int | float) or isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare float")
    if not isinstance(artifact.get("duration_s"), int | float) or isinstance(artifact.get("duration_s"), bool):
        raise ValueError("duration_s must be a bare number")  # pragma: no cover - defensive.
    if not isinstance(artifact.get("inference_substrate"), str):
        raise ValueError("inference_substrate must be a string")  # pragma: no cover - defensive.
    citations = artifact.get("cited_upstream_artifacts")
    if not isinstance(citations, list):
        raise ValueError("cited_upstream_artifacts must be a list")  # pragma: no cover - defensive.
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise ValueError("citation entries must be objects")  # pragma: no cover - defensive.
        if not isinstance(citation.get("experiment_id"), int):
            raise ValueError("citation entries need integer experiment_id")  # pragma: no cover - defensive.
        if not isinstance(citation.get("fields_imported"), list):
            raise ValueError("citation entries need fields_imported list")  # pragma: no cover - defensive.
        if not is_sha256(citation.get("sha256")):
            raise ValueError("citation entries need sha256")
    if not isinstance(artifact.get("flagged_artifacts_skipped"), list):
        raise ValueError("flagged_artifacts_skipped must be a list")  # pragma: no cover - defensive.
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be sha256")  # pragma: no cover - defensive.


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    summary_statuses: Mapping[int, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the Exp 3985 capstone artifact."""

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

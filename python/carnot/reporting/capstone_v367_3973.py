"""Build the Exp 3973 v367 ARC accuracy plus HONEST-efficiency capstone.

Spec refs: REQ-CAPSTONE-3973, SCENARIO-CAPSTONE-3973.

The capstone is an aggregation step, not a new ARC run. It first routes every
existing upstream artifact through ``scripts/summarize_artifact.py`` so the
artifact's own verdict, flags, gates, duration, and headline metrics are visible
before any field is imported. Missing upstreams are recorded as missing state,
and artifacts stamped ``flagged_adversarial:true`` are excluded before their
numbers can influence the milestone verdict.
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
OUTPUT_REL_PATH = Path("results/experiment_3973_capstone_v367.json")
EXPERIMENT_ID = 3973
RANDOM_SEED = 3973
V366_BASELINE_TOTAL_REAL_LEVELS_SOLVED = 2
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts_via_summarize_artifact_py"
PYTHON_BIN = Path(".venv/bin/python")

UPSTREAM_IDS = (3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971)
REAL_LEVEL_UPSTREAM_IDS = (3964, 3965, 3966)

FIELDS_IMPORTED: Mapping[int, list[str]] = {
    3964: ["ACCURACY_levels_solved", "new_levels_solved_this_task"],
    3965: ["ACCURACY_levels_solved", "level_summaries"],
    3966: ["game_solved", "ACCURACY_levels_solved"],
    3967: [
        "efficiency_ratio_with_over_without",
        "verifier_invoked_in_loop",
        "actions_from_real_env",
        "cis_non_overlapping_pruner_helps",
    ],
    3968: ["n_trustworthy_at_0.15"],
    3969: ["energy_drop", "positive_control_passed"],
    3970: ["transfer_win"],
    3971: ["quota_gate_cleared"],
}

REQUIRED_ARTIFACT_FIELDS = (
    "total_real_levels_solved",
    "third_game_solved",
    "verifier_earns_efficiency_on_real_games",
    "m3_artifact_clean_not_flagged",
    "n_trustworthy_world_models",
    "hidden_state_fixed",
    "transfer_win",
    "quota_gate_cleared",
    "cited_upstream_artifacts",
    "flagged_artifacts_skipped",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

INT_FIELDS = (
    "total_real_levels_solved",
    "n_trustworthy_world_models",
)
BOOL_FIELDS = (
    "third_game_solved",
    "verifier_earns_efficiency_on_real_games",
    "m3_artifact_clean_not_flagged",
    "hidden_state_fixed",
    "transfer_win",
    "quota_gate_cleared",
)

FIELD_PRINCIPLES = {
    "total_real_levels_solved": (
        "BARE INT - total real ARC-AGI-3 levels solved this milestone across all "
        "games (the accuracy headline; .366 baseline was 2)."
    ),
    "third_game_solved": (
        "BARE BOOL - did a third distinct game solve (games-solved 2->3, the "
        "method-generalizes datum)."
    ),
    "verifier_earns_efficiency_on_real_games": (
        "BARE BOOL - exp3967's HONEST M3 gate: did the verifier action-pruner "
        "significantly cut REAL actions WITH the verifier provably in the loop "
        "(the existential 'verifier earns its place' answer; a flagged/not-in-loop "
        "M3 counts as FALSE)."
    ),
    "m3_artifact_clean_not_flagged": (
        "BARE BOOL - is the exp3967 M3 artifact NON-flagged AND "
        "verifier_invoked_in_loop=true (the direct check that .366's fabrication "
        "did not recur)."
    ),
    "n_trustworthy_world_models": (
        "BARE INT - how many of the 6 non-spatial games reached a plan-able model "
        "(generalization of vc33's 0.005)."
    ),
    "hidden_state_fixed": (
        "BARE BOOL - did Pinductor (exp3969) drop hidden-state energy with a "
        "passing positive control (one broken mechanism fixed?)."
    ),
    "transfer_win": (
        "BARE BOOL - did ArcMemo (exp3970) show cross-game transfer (the other "
        "broken mechanism fixed? the self-learning mandate)."
    ),
    "quota_gate_cleared": (
        "BARE BOOL - is offline accuracy past the operator gate for justifying an "
        "online run (exp3971)."
    ),
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported, sha256} - the audit trail so "
        "the capstone numbers trace to real measurements."
    ),
    "flagged_artifacts_skipped": (
        "Which upstream artifacts were flagged_adversarial and therefore EXCLUDED "
        "from aggregation."
    ),
    "honest_verdict": "Terminal-prefix verdict.",
    "duration_s": "Aggregation wall-clock duration.",
    "inference_substrate": "Aggregation substrate.",
}


def is_sha256(value: object) -> bool:
    """Return whether a value is a lowercase SHA-256 hex digest."""

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
    """Return a stable path for JSON audit fields."""

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
        if supplied is not None:
            statuses[experiment_id] = dict(supplied[experiment_id])
        else:
            statuses[experiment_id] = run_summarize_artifact(root, path)
    return statuses


def flagged(payload: Mapping[str, Any] | None) -> bool:
    """Return whether the upstream is stamped as adversarial and must be skipped."""

    return isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True


def int_metric(payload: Mapping[str, Any] | None, field: str) -> int:
    """Extract an integer metric while rejecting booleans and missing values."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def float_metric(payload: Mapping[str, Any] | None, field: str) -> float:
    """Extract a floating-point metric while rejecting booleans and missing values."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def bool_metric(payload: Mapping[str, Any] | None, field: str) -> bool:
    """Extract a boolean metric without truthifying non-bool values."""

    return isinstance(payload, Mapping) and payload.get(field) is True


def real_levels_solved(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> int:
    """Sum real ARC levels from the three accuracy upstreams."""

    total = 0
    for experiment_id in REAL_LEVEL_UPSTREAM_IDS:
        payload = clean_upstreams.get(experiment_id)
        if bool_metric(payload, "real_env_confirmed"):
            total += int_metric(payload, "ACCURACY_levels_solved")
    return total


def m3_artifact_clean(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Return the direct clean-M3 guard from exp3967."""

    payload = clean_upstreams.get(3967)
    return bool_metric(payload, "verifier_invoked_in_loop")


def verifier_earns_efficiency(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Apply the HONEST M3 gate without importing flagged or not-in-loop claims."""

    payload = clean_upstreams.get(3967)
    return (
        bool_metric(payload, "cis_non_overlapping_pruner_helps")
        and bool_metric(payload, "verifier_invoked_in_loop")
        and bool_metric(payload, "actions_from_real_env")
    )


def third_game_solved(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Return whether exp3966 establishes a third solved ARC-AGI-3 game."""

    payload = clean_upstreams.get(3966)
    return bool_metric(payload, "real_env_confirmed") and int_metric(payload, "ACCURACY_levels_solved") > 0


def hidden_state_fixed(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Return whether Pinductor lowered hidden-state energy with a positive control."""

    payload = clean_upstreams.get(3969)
    return float_metric(payload, "energy_drop") > 0.0 and bool_metric(payload, "positive_control_passed")


def cited_upstream_artifacts(
    paths: Mapping[int, Path | None],
    clean_upstreams: Mapping[int, Mapping[str, Any]],
) -> list[JsonDict]:
    """Build the minimal provenance list for every imported clean upstream."""

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
    """Record upstream artifacts skipped before aggregation due to the adversarial stamp."""

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
    """Record missing upstream artifacts without converting absence into a gate."""

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
    flagged_ids: set[int],
    clean_upstreams: Mapping[int, Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Expose per-upstream inclusion state so missing and skipped inputs are auditable."""

    state: dict[str, JsonDict] = {}
    for experiment_id in UPSTREAM_IDS:
        path = paths[experiment_id]
        state[str(experiment_id)] = {
            "exists": path is not None,
            "path": relative_to_root(root, path) if path is not None else "",
            "honest_verdict": str(upstreams[experiment_id].get("honest_verdict"))
            if isinstance(upstreams[experiment_id], Mapping)
            else "missing",
            "flagged_adversarial": experiment_id in flagged_ids,
            "included": experiment_id in clean_upstreams,
            "summarize_artifact_returncode": summaries.get(experiment_id, {}).get("returncode"),
        }
    return state


def duration_from(started_s: float, now_s: float | None) -> float:
    """Compute an honest aggregation duration with a small nonzero floor."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0001, end - started_s)


def verdict(total_real_levels_solved: int, verifier_earns: bool, missing_count: int, skipped_count: int) -> str:
    """Build the terminal-prefix verdict for an ungated aggregation capstone."""

    progress = total_real_levels_solved - V366_BASELINE_TOTAL_REAL_LEVELS_SOLVED
    return (
        "complete: capstone_v367_"
        f"accuracy_progress{progress}_total_real_levels{total_real_levels_solved}_"
        f"verifier_earns_efficiency{str(verifier_earns).lower()}_"
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
    """Build the .367 capstone from whatever upstream artifacts exist."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    paths = selected_upstream_paths(root_path)
    summaries = summarize_existing_artifacts(root_path, paths, summary_statuses)
    upstreams: dict[int, Mapping[str, Any] | None] = {
        experiment_id: read_json_object(path) if path is not None else None
        for experiment_id, path in paths.items()
    }
    flagged_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if flagged(payload)
    }
    clean_upstreams: dict[int, Mapping[str, Any]] = {
        experiment_id: payload
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and experiment_id not in flagged_ids
    }

    total_solved = real_levels_solved(clean_upstreams)
    verifier_earns = verifier_earns_efficiency(clean_upstreams)
    missing = missing_upstream_artifacts(paths)
    skipped = flagged_artifacts_skipped(root_path, paths, flagged_ids)
    artifact: JsonDict = {
        "schema": "carnot.capstone_v367_3973.v1",
        "experiment_id": EXPERIMENT_ID,
        "total_real_levels_solved": total_solved,
        "v366_baseline_total_real_levels_solved": V366_BASELINE_TOTAL_REAL_LEVELS_SOLVED,
        "accuracy_progress_vs_v366_baseline": total_solved - V366_BASELINE_TOTAL_REAL_LEVELS_SOLVED,
        "third_game_solved": third_game_solved(clean_upstreams),
        "verifier_earns_efficiency_on_real_games": verifier_earns,
        "m3_artifact_clean_not_flagged": m3_artifact_clean(clean_upstreams),
        "m3_efficiency_ratio_with_over_without": float_metric(
            clean_upstreams.get(3967),
            "efficiency_ratio_with_over_without",
        ),
        "n_trustworthy_world_models": int_metric(clean_upstreams.get(3968), "n_trustworthy_at_0.15"),
        "hidden_state_fixed": hidden_state_fixed(clean_upstreams),
        "transfer_win": bool_metric(clean_upstreams.get(3970), "transfer_win"),
        "quota_gate_cleared": bool_metric(clean_upstreams.get(3971), "quota_gate_cleared"),
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
        "honest_verdict": verdict(total_solved, verifier_earns, len(missing), len(skipped)),
        "duration_s": duration_from(start, now_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the .367 capstone fields that prevent metric laundering."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover - defensive.
    verdict_text = str(artifact.get("honest_verdict", ""))
    if not verdict_text.startswith(("complete:", "success:", "blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in INT_FIELDS:
        if not isinstance(artifact.get(field), int) or isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare int")
    for field in BOOL_FIELDS:
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
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
    """Build, validate, and write the Exp 3973 capstone artifact."""

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

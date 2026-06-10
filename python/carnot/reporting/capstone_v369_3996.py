"""Build the Exp 3996 v369 GAP-4 confirm/decentralize/deploy capstone.

Spec refs: REQ-CAPSTONE-3996, SCENARIO-CAPSTONE-3996.

This module does not run a new ARC solver or verifier. It performs the
milestone-close accounting step: read each existing upstream artifact through
``scripts/summarize_artifact.py``, exclude artifacts stamped
``flagged_adversarial:true``, and compute the GAP-4 headline from the surviving
measurements. Missing upstreams stay visible in the artifact because absence is
part of the honest milestone verdict.
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
OUTPUT_REL_PATH = Path("results/experiment_3996_capstone_v369.json")
EXPERIMENT_ID = 3996
RANDOM_SEED = 3996
BANKED_GAMES_SOLVED_BEFORE_V369 = 3
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PYTHON_BIN = Path(".venv/bin/python")

UPSTREAM_IDS = tuple(range(3987, 3996))
FIELDS_IMPORTED: Mapping[int, list[str]] = {
    3987: ["debiased_coverage_combined"],
    3988: [
        "primary_gate_passed",
        "confidence_label_only_retired",
        "agreement_is_selector_not_label",
        "n_gold_given_agreement",
    ],
    3989: ["feedback_beats_redraw", "mcnemar_p"],
    3990: [
        "verifier_registered",
        "arc2_reproduced_19of31",
        "arc1_reproduced_28of31",
        "gap5_entry_appended",
    ],
    3991: [
        "real_local_gguf_inducer",
        "local_beats_vote",
        "local_induction_demo_perfect_rate",
        "local_gated_pass2",
        "cost_local_seconds",
        "cost_codex_seconds_ref",
    ],
    3992: ["new_levels_solved_this_task", "ACCURACY_levels_solved"],
    3993: ["game_solved", "ACCURACY_levels_solved"],
    3994: ["solve_transfer_win", "actions_cold_start", "actions_with_memory"],
    3995: [],
}

REQUIRED_ARTIFACT_FIELDS = (
    "gap4_confirmed",
    "gap4_decentralized",
    "gap4_deployed",
    "local_generator_beats_vote",
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
    "gap4_confirmed",
    "gap4_decentralized",
    "gap4_deployed",
    "local_generator_beats_vote",
    "arcmemo_solve_transfer_win",
)
INT_FIELDS = (
    "total_games_solved",
    "total_new_levels_this_milestone",
)

FIELD_PRINCIPLES = {
    "gap4_confirmed": (
        "BARE BOOL - did the pre-registered precision confirmation reach a powered answer "
        "(selector-confirmed OR confidence-label-only-retired); a flagged/blocked result is FALSE."
    ),
    "gap4_decentralized": (
        "BARE BOOL - did a REAL local open-weight GGUF inducer drive the verifier and report a "
        "gated rerank number (the sovereignty milestone; blocked_local_gguf_not_cached is FALSE)."
    ),
    "gap4_deployed": (
        "BARE BOOL - gap4_program_induction_stack registered AND ARC-2 19/31 + ARC-1 28/31 "
        "reproduced bit-exact offline."
    ),
    "local_generator_beats_vote": (
        "BARE BOOL - did the OPEN local inducer beat vote (the strong decentralization-clean "
        "accuracy claim); false-but-induced is the honest open-vs-closed gap."
    ),
    "total_games_solved": (
        "BARE INT - the ARC-AGI-3 accuracy headline (games-solved + monotonic level progress; "
        "did .369 break the 3xL1 plateau)."
    ),
    "total_new_levels_this_milestone": (
        "BARE INT - the ARC-AGI-3 accuracy headline (games-solved + monotonic level progress; "
        "did .369 break the 3xL1 plateau)."
    ),
    "arcmemo_solve_transfer_win": (
        "BARE BOOL - did concept memory make a NEW game's solve cheaper (the self-learning datum "
        "compounding across games)."
    ),
    "cited_upstream_artifacts": (
        "list of {experiment_id, fields_imported, sha256} - the audit trail so the capstone "
        "numbers trace to real measurements."
    ),
    "flagged_artifacts_skipped": "Which upstream artifacts were flagged_adversarial and EXCLUDED.",
    "honest_verdict": "Terminal-prefix verdict + aggregation substrate.",
    "duration_s": "Terminal-prefix verdict + aggregation substrate.",
    "inference_substrate": "Terminal-prefix verdict + aggregation substrate.",
}


def is_sha256(value: object) -> bool:
    """Return true when a value is a lowercase SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def read_json_object(path: Path) -> JsonDict:
    """Load an artifact and fail closed if it is not a JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover - defensive.
    return payload


def sha256_file(path: Path) -> str:
    """Hash an upstream artifact so imported numbers remain audit-traceable."""

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
    """Select one artifact per expected upstream, using the latest sorted hit."""

    selected: dict[int, Path | None] = {}
    for experiment_id in UPSTREAM_IDS:
        matches = matching_artifact_paths(root, experiment_id)
        selected[experiment_id] = matches[-1] if matches else None
    return selected


def run_summarize_artifact(root: Path, path: Path) -> JsonDict:
    """Run the mandated disciplined reader for one existing artifact."""

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
    """Return whether an upstream is stamped adversarial and must be skipped."""

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


def gap4_confirmed(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Return true for either selector confirmation or clean retirement."""

    payload = clean_upstreams.get(3988)
    return bool_metric(payload, "primary_gate_passed") or bool_metric(payload, "confidence_label_only_retired")


def gap4_decentralized(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Return true when a real local GGUF inducer reports a gated rerank number."""

    payload = clean_upstreams.get(3991)
    return bool_metric(payload, "real_local_gguf_inducer") and float_metric(payload, "local_gated_pass2") > 0.0


def gap4_deployed(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Return true only for registered and bit-exact reproduced deployment."""

    payload = clean_upstreams.get(3990)
    return (
        bool_metric(payload, "verifier_registered")
        and bool_metric(payload, "arc2_reproduced_19of31")
        and bool_metric(payload, "arc1_reproduced_28of31")
    )


def fourth_game_solved(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> bool:
    """Return true only when exp3993 reports a real solved additional game."""

    payload = clean_upstreams.get(3993)
    game = str(payload.get("game_solved", "")) if isinstance(payload, Mapping) else ""
    return (
        bool_metric(payload, "real_env_confirmed")
        and game not in ("", "none")
        and int_metric(payload, "ACCURACY_levels_solved") > 0
    )


def total_new_levels_this_milestone(clean_upstreams: Mapping[int, Mapping[str, Any]]) -> int:
    """Sum validated incremental progress and fourth-game levels, if solved."""

    total = int_metric(clean_upstreams.get(3992), "new_levels_solved_this_task")
    if fourth_game_solved(clean_upstreams):
        total += int_metric(clean_upstreams.get(3993), "ACCURACY_levels_solved")
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
    """Record artifacts skipped before aggregation due to adversarial stamps."""

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
            "summarize_artifact_returncode": summaries.get(experiment_id, {}).get("returncode"),
        }
    return state


def duration_from(started_s: float, now_s: float | None) -> float:
    """Compute an honest aggregation duration with a small nonzero floor."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0001, end - started_s)


def verdict(
    confirmed: bool,
    decentralized: bool,
    deployed: bool,
    local_beats_vote: bool,
    total_games_solved: int,
    total_new_levels: int,
    missing_count: int,
    skipped_count: int,
) -> str:
    """Build the terminal-prefix milestone verdict from the owed axes."""

    prefix = "success" if confirmed or decentralized or deployed else "complete"
    confirmed_text = "CONFIRMED" if confirmed else "UNCONFIRMED"
    decentralized_text = "DECENTRALIZED" if decentralized else "NOT_DECENTRALIZED"
    deployed_text = "DEPLOYED" if deployed else "NOT_DEPLOYED"
    local_text = "local_beats_vote" if local_beats_vote else "local_not_beats_vote"
    return (
        f"{prefix}: capstone_v369_gap4_{confirmed_text}_{decentralized_text}_{deployed_text}_"
        f"{local_text}_games{total_games_solved}_new_levels{total_new_levels}_"
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
    """Build the .369 capstone from whatever upstream artifacts exist."""

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

    confirmed = gap4_confirmed(clean_upstreams)
    decentralized = gap4_decentralized(clean_upstreams)
    deployed = gap4_deployed(clean_upstreams)
    local_beats_vote = bool_metric(clean_upstreams.get(3991), "local_beats_vote")
    fourth_solved = fourth_game_solved(clean_upstreams)
    total_games = BANKED_GAMES_SOLVED_BEFORE_V369 + (1 if fourth_solved else 0)
    total_new_levels = total_new_levels_this_milestone(clean_upstreams)
    missing = missing_upstream_artifacts(paths)
    skipped = flagged_artifacts_skipped(root_path, paths, flagged_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v369_3996.v1",
        "experiment_id": EXPERIMENT_ID,
        "gap4_confirmed": confirmed,
        "confirmed_primary_gate_passed": bool_metric(clean_upstreams.get(3988), "primary_gate_passed"),
        "confirmed_confidence_label_only_retired": bool_metric(
            clean_upstreams.get(3988),
            "confidence_label_only_retired",
        ),
        "confirmed_agreement_is_selector_not_label": bool_metric(
            clean_upstreams.get(3988),
            "agreement_is_selector_not_label",
        ),
        "confirmed_n_gold_given_agreement": int_metric(clean_upstreams.get(3988), "n_gold_given_agreement"),
        "confirmed_debiased_coverage_combined": float_metric(
            clean_upstreams.get(3987),
            "debiased_coverage_combined",
        ),
        "confirmed_feedback_beats_redraw": bool_metric(clean_upstreams.get(3989), "feedback_beats_redraw"),
        "confirmed_mcnemar_p": float_metric(clean_upstreams.get(3989), "mcnemar_p"),
        "gap4_decentralized": decentralized,
        "local_generator_beats_vote": local_beats_vote,
        "local_induction_demo_perfect_rate": float_metric(
            clean_upstreams.get(3991),
            "local_induction_demo_perfect_rate",
        ),
        "local_gated_pass2": float_metric(clean_upstreams.get(3991), "local_gated_pass2"),
        "local_cost_seconds": float_metric(clean_upstreams.get(3991), "cost_local_seconds"),
        "codex_cost_seconds_ref": float_metric(clean_upstreams.get(3991), "cost_codex_seconds_ref"),
        "gap4_deployed": deployed,
        "deployed_verifier_registered": bool_metric(clean_upstreams.get(3990), "verifier_registered"),
        "deployed_arc2_reproduced_19of31": bool_metric(clean_upstreams.get(3990), "arc2_reproduced_19of31"),
        "deployed_arc1_reproduced_28of31": bool_metric(clean_upstreams.get(3990), "arc1_reproduced_28of31"),
        "deployed_gap5_entry_appended": bool_metric(clean_upstreams.get(3990), "gap5_entry_appended"),
        "total_games_solved": total_games,
        "banked_games_solved_before_v369": BANKED_GAMES_SOLVED_BEFORE_V369,
        "fourth_game_solved": fourth_solved,
        "total_new_levels_this_milestone": total_new_levels,
        "incremental_new_levels": int_metric(clean_upstreams.get(3992), "new_levels_solved_this_task"),
        "fourth_game_accuracy_levels_solved": int_metric(clean_upstreams.get(3993), "ACCURACY_levels_solved")
        if fourth_solved
        else 0,
        "arcmemo_solve_transfer_win": bool_metric(clean_upstreams.get(3994), "solve_transfer_win"),
        "arcmemo_actions_cold_start": int_metric(clean_upstreams.get(3994), "actions_cold_start"),
        "arcmemo_actions_with_memory": int_metric(clean_upstreams.get(3994), "actions_with_memory"),
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
            confirmed,
            decentralized,
            deployed,
            local_beats_vote,
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
    """Validate the .369 capstone fields that prevent metric laundering."""

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
    """Build, validate, and write the Exp 3996 capstone artifact."""

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

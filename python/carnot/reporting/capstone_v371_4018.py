"""Build the Exp 4018 v371 GAP-4 precision/decentralization capstone.

Spec refs: REQ-CAPSTONE-4018, SCENARIO-CAPSTONE-4018.

This module is an accounting pass over already-landed experiment artifacts. It
does not re-run ARC, a local model, or an LLM judge. The reason the code is
strict about missing, blocked, and flagged upstreams is that the milestone
headline depends on not laundering an absent execution as a clean confirmation.
Every existing upstream is first routed through ``scripts/summarize_artifact.py``
so the artifact's own verdict, flags, gates, and duration are visible before
any number is imported.
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
OUTPUT_REL_PATH = Path("results/experiment_4018_capstone_v371.json")
EXPERIMENT_ID = 4018
RANDOM_SEED = 4018
BANKED_GAMES_SOLVED_BEFORE_V371 = 4
LOCAL_PASS2_APPROACHING_CODEX_FLOOR = 0.45
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PYTHON_BIN = Path(".venv/bin/python")

UPSTREAM_IDS = tuple(range(4008, 4018))
FIELDS_IMPORTED: Mapping[int, list[str]] = {
    4008: ["pretest_suite_green", "quarantined_tests"],
    4009: [
        "execution_floor_met",
        "primary_gate_passed",
        "confidence_label_only_retired",
        "agreement_is_selector_not_label",
        "n_gold_given_agreement",
        "total_codex_calls",
        "n_agreement_events",
    ],
    4010: ["selector_beats_output_agreement"],
    4011: ["feedback_beats_redraw", "n_discordant_pairs"],
    4012: [
        "local_beats_vote",
        "local_demo_perfect_coverage_bestofn",
        "coverage_gain_vs_3attempt",
        "local_gated_pass2",
        "cost_local_seconds",
        "cost_codex_seconds_ref",
    ],
    4013: ["selection_accuracy_parity", "cost_ratio_judge_over_verifier"],
    4014: ["new_levels_this_task", "ACCURACY_total_levels_solved"],
    4015: ["game_solved", "ACCURACY_levels_solved"],
    4016: ["solve_transfer_win", "actions_cold_start", "actions_with_memory"],
    4017: [],
}

REQUIRED_ARTIFACT_FIELDS = (
    "confirmation_executed_this_time",
    "gap4_confirmed",
    "gap4_decentralization_effective",
    "local_generator_beats_vote",
    "cross_example_selector_helped",
    "verifier_cheaper_than_judge",
    "total_games_solved",
    "total_levels_solved",
    "arcmemo_solve_transfer_win",
    "pretest_suite_green",
    "cited_upstream_artifacts",
    "flagged_artifacts_skipped",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

BOOL_FIELDS = (
    "confirmation_executed_this_time",
    "gap4_confirmed",
    "gap4_decentralization_effective",
    "local_generator_beats_vote",
    "cross_example_selector_helped",
    "verifier_cheaper_than_judge",
    "arcmemo_solve_transfer_win",
    "pretest_suite_green",
)
INT_FIELDS = ("total_games_solved", "total_levels_solved")

FIELD_PRINCIPLES = {
    "confirmation_executed_this_time": (
        "BARE BOOL - did exp4009 clear the execution floor (total_codex_calls>0, "
        "n_agreement_events>0), unlike .370's pending_execution / .369's poison-skip."
    ),
    "gap4_confirmed": (
        "BARE BOOL - did the precision confirmation reach a powered answer "
        "(selector-confirmed OR confidence-label-only-retired); a flagged/blocked/missing/"
        "pending result is FALSE."
    ),
    "gap4_decentralization_effective": (
        "BARE BOOL - did best-of-N local close the induction gap (local_beats_vote strong "
        "form, OR coverage_gain_vs_3attempt>0 approaching codex)."
    ),
    "local_generator_beats_vote": (
        "BARE BOOL - did the OPEN local inducer (best-of-N) beat vote (the strong sovereign "
        "accuracy claim)."
    ),
    "cross_example_selector_helped": (
        "BARE BOOL - did exp4010's cross-example-consistency selector beat plain "
        "output-agreement (the GAP-5 progress datum)."
    ),
    "verifier_cheaper_than_judge": (
        "BARE BOOL - did exp4013 show the model-free verifier cheaper than an LLM-judge at "
        "matched selection accuracy (north-star section 5 efficiency)."
    ),
    "total_games_solved": (
        "BARE INT - the ARC-AGI-3 accuracy headline (games-solved + monotonic level progress; "
        "did .371 raise the frontier past 4 games / 5 levels)."
    ),
    "total_levels_solved": (
        "BARE INT - the ARC-AGI-3 accuracy headline (games-solved + monotonic level progress; "
        "did .371 raise the frontier past 4 games / 5 levels)."
    ),
    "arcmemo_solve_transfer_win": (
        "BARE BOOL - did concept memory make NEW content's solve cheaper (the self-learning "
        "datum compounding across the milestone)."
    ),
    "pretest_suite_green": (
        "BARE BOOL - did the poison-guard hold (exp4008), so no GAP-4 task was cascade-skipped "
        "this time."
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
    """Load an artifact and reject non-object JSON, which cannot be safely cited."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover - defensive
    return payload


def sha256_file(path: Path) -> str:
    """Hash an upstream artifact so imported numbers remain audit-traceable."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_root(root: Path, path: Path) -> str:
    """Return a stable repository-relative path for audit fields."""

    return str(path.relative_to(root))


def matching_artifact_paths(root: Path, experiment_id: int) -> list[Path]:
    """Find candidate artifact files for one upstream experiment."""

    matches: list[Path] = []
    for pattern in (
        root / "results" / f"experiment_{experiment_id}_*.json",
        root / "results" / f"experiment_{experiment_id}.json",
    ):
        matches.extend(Path(path) for path in glob.glob(str(pattern)))
    return sorted(matches)


def selected_upstream_paths(root: Path) -> dict[int, Path | None]:
    """Select one artifact per expected upstream, preferring the final sorted hit."""

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
    return not verdict_text.startswith("blocked_") and "pending_execution" not in verdict_text


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


def list_metric(payload: Mapping[str, Any] | None, field: str) -> list[Any]:
    """Extract a list metric without accepting scalar lookalikes."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return list(value) if isinstance(value, list) else []


def str_metric(payload: Mapping[str, Any] | None, field: str) -> str:
    """Extract a string metric for audit fields."""

    value = payload.get(field) if isinstance(payload, Mapping) else None
    return value if isinstance(value, str) else ""


def confirmation_executed(payload: Mapping[str, Any] | None) -> bool:
    """Return true only when exp4009 actually cleared the execution floor."""

    return (
        invoked(payload)
        and bool_metric(payload, "execution_floor_met")
        and int_metric(payload, "total_codex_calls") > 0
        and int_metric(payload, "n_agreement_events") > 0
    )


def gap4_confirmed_from(payload: Mapping[str, Any] | None) -> bool:
    """Return true for selector confirmation or clean confidence-label retirement."""

    return confirmation_executed(payload) and (
        bool_metric(payload, "primary_gate_passed") or bool_metric(payload, "confidence_label_only_retired")
    )


def decentralization_effective(payload: Mapping[str, Any] | None) -> bool:
    """Return true for the strong local win or the honest best-of-N closing-gap form."""

    if not invoked(payload):
        return False
    return bool_metric(payload, "local_beats_vote") or (
        float_metric(payload, "coverage_gain_vs_3attempt") > 0.0
        and float_metric(payload, "local_gated_pass2") >= LOCAL_PASS2_APPROACHING_CODEX_FLOOR
    )


def verifier_cheaper_than_judge_from(payload: Mapping[str, Any] | None) -> bool:
    """Return true only for parity with a strictly cheaper model-free verifier."""

    return (
        invoked(payload)
        and bool_metric(payload, "selection_accuracy_parity")
        and float_metric(payload, "cost_ratio_judge_over_verifier") > 1.0
    )


def fifth_game_solved(payload: Mapping[str, Any] | None) -> bool:
    """Return true only when exp4015 reports an additional solved game."""

    game = str_metric(payload, "game_solved")
    return invoked(payload) and game not in ("", "none") and int_metric(payload, "ACCURACY_levels_solved") > 0


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
            "invoked": invoked(payload),
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
    local_beats_vote: bool,
    verifier_cheaper: bool,
    total_games_solved: int,
    total_levels_solved: int,
    arcmemo_win: bool,
    pretest_green: bool,
    missing_count: int,
    skipped_count: int,
) -> str:
    """Build the terminal-prefix milestone verdict from the .371 headline axes."""

    prefix = (
        "success"
        if confirmed
        or decentralized
        or total_games_solved > BANKED_GAMES_SOLVED_BEFORE_V371
        or total_levels_solved > 5
        or arcmemo_win
        else "complete"
    )
    confirmed_text = "CONFIRMED" if confirmed else "UNCONFIRMED"
    decentralized_text = "DECENTRALIZATION_EFFECTIVE" if decentralized else "NOT_DECENTRALIZATION_EFFECTIVE"
    local_text = "local_beats_vote" if local_beats_vote else "local_not_beats_vote"
    efficiency_text = "verifier_cheaper" if verifier_cheaper else "verifier_not_cheaper"
    pretest_text = "pretest_green" if pretest_green else "pretest_not_green"
    arcmemo_text = "arcmemo_transfer_win" if arcmemo_win else "arcmemo_no_transfer_win"
    return (
        f"{prefix}: capstone_v371_gap4_{confirmed_text}_{decentralized_text}_{local_text}_"
        f"{efficiency_text}_games{total_games_solved}_levels{total_levels_solved}_{arcmemo_text}_"
        f"{pretest_text}_missing{missing_count}_flagged_skipped{skipped_count}"
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
    """Build the .371 capstone from whatever upstream artifacts exist."""

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

    confirmation_payload = clean_upstreams.get(4009)
    local_payload = clean_upstreams.get(4012)
    efficiency_payload = clean_upstreams.get(4013)
    arcmemo_payload = clean_upstreams.get(4016)
    executed = confirmation_executed(confirmation_payload)
    confirmed = gap4_confirmed_from(confirmation_payload)
    decentralized = decentralization_effective(local_payload)
    local_beats_vote = invoked(local_payload) and bool_metric(local_payload, "local_beats_vote")
    cross_selector = invoked(clean_upstreams.get(4010)) and bool_metric(
        clean_upstreams.get(4010),
        "selector_beats_output_agreement",
    )
    verifier_cheaper = verifier_cheaper_than_judge_from(efficiency_payload)
    fifth_solved = fifth_game_solved(clean_upstreams.get(4015))
    total_games = BANKED_GAMES_SOLVED_BEFORE_V371 + (1 if fifth_solved else 0)
    total_levels = int_metric(clean_upstreams.get(4014), "ACCURACY_total_levels_solved")
    arcmemo_win = invoked(arcmemo_payload) and bool_metric(arcmemo_payload, "solve_transfer_win")
    pretest_green = invoked(clean_upstreams.get(4008)) and bool_metric(clean_upstreams.get(4008), "pretest_suite_green")
    missing = missing_upstream_artifacts(paths)
    skipped = flagged_artifacts_skipped(root_path, paths, flagged_ids)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v371_4018.v1",
        "experiment_id": EXPERIMENT_ID,
        "confirmation_executed_this_time": executed,
        "gap4_confirmed": confirmed,
        "confirmed_execution_floor_met": executed,
        "confirmed_primary_gate_passed": confirmation_executed(confirmation_payload)
        and bool_metric(confirmation_payload, "primary_gate_passed"),
        "confirmed_confidence_label_only_retired": confirmation_executed(confirmation_payload)
        and bool_metric(confirmation_payload, "confidence_label_only_retired"),
        "confirmed_agreement_is_selector_not_label": confirmation_executed(confirmation_payload)
        and bool_metric(confirmation_payload, "agreement_is_selector_not_label"),
        "confirmed_n_gold_given_agreement": int_metric(confirmation_payload, "n_gold_given_agreement")
        if confirmation_executed(confirmation_payload)
        else 0,
        "confirmation_total_codex_calls": int_metric(confirmation_payload, "total_codex_calls")
        if confirmation_executed(confirmation_payload)
        else 0,
        "confirmation_n_agreement_events": int_metric(confirmation_payload, "n_agreement_events")
        if confirmation_executed(confirmation_payload)
        else 0,
        "cross_example_selector_helped": cross_selector,
        "feedback_beats_redraw": invoked(clean_upstreams.get(4011))
        and bool_metric(clean_upstreams.get(4011), "feedback_beats_redraw"),
        "feedback_n_discordant_pairs": int_metric(clean_upstreams.get(4011), "n_discordant_pairs")
        if invoked(clean_upstreams.get(4011))
        else 0,
        "gap4_decentralization_effective": decentralized,
        "local_generator_beats_vote": local_beats_vote,
        "local_demo_perfect_coverage_bestofn": float_metric(local_payload, "local_demo_perfect_coverage_bestofn")
        if invoked(local_payload)
        else 0.0,
        "local_coverage_gain_vs_3attempt": float_metric(local_payload, "coverage_gain_vs_3attempt")
        if invoked(local_payload)
        else 0.0,
        "local_gated_pass2": float_metric(local_payload, "local_gated_pass2") if invoked(local_payload) else 0.0,
        "local_cost_seconds": float_metric(local_payload, "cost_local_seconds") if invoked(local_payload) else 0.0,
        "codex_cost_seconds_ref": float_metric(local_payload, "cost_codex_seconds_ref")
        if invoked(local_payload)
        else 0.0,
        "verifier_cheaper_than_judge": verifier_cheaper,
        "efficiency_selection_accuracy_parity": invoked(efficiency_payload)
        and bool_metric(efficiency_payload, "selection_accuracy_parity"),
        "efficiency_cost_ratio_judge_over_verifier": float_metric(efficiency_payload, "cost_ratio_judge_over_verifier")
        if invoked(efficiency_payload)
        else 0.0,
        "total_games_solved": total_games,
        "banked_games_solved_before_v371": BANKED_GAMES_SOLVED_BEFORE_V371,
        "total_levels_solved": total_levels,
        "new_levels_this_task": int_metric(clean_upstreams.get(4014), "new_levels_this_task"),
        "fifth_game_solved": fifth_solved,
        "fifth_game_accuracy_levels_solved": int_metric(clean_upstreams.get(4015), "ACCURACY_levels_solved")
        if fifth_solved
        else 0,
        "arcmemo_solve_transfer_win": arcmemo_win,
        "arcmemo_actions_cold_start": int_metric(arcmemo_payload, "actions_cold_start") if invoked(arcmemo_payload) else 0,
        "arcmemo_actions_with_memory": int_metric(arcmemo_payload, "actions_with_memory")
        if invoked(arcmemo_payload)
        else 0,
        "pretest_suite_green": pretest_green,
        "poison_guard_quarantined_tests": list_metric(clean_upstreams.get(4008), "quarantined_tests"),
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
            local_beats_vote,
            verifier_cheaper,
            total_games,
            total_levels,
            arcmemo_win,
            pretest_green,
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
    """Validate the .371 fields that prevent false headline confirmation."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")  # pragma: no cover - defensive
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
        raise ValueError("duration_s must be a bare number")
    if not isinstance(artifact.get("inference_substrate"), str):
        raise ValueError("inference_substrate must be a string")
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
    """Build, validate, and write the Exp 4018 capstone artifact."""

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

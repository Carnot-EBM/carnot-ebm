"""Experiment 4631: refresh the ARC operator-resubmit package.

Spec refs: REQ-CAPSTONE-4631, SCENARIO-CAPSTONE-4631,
SCENARIO-CAPSTONE-4631-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_4619_refresh_submission_package as previous


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4631_refresh_submission_package"
SCHEMA = "carnot.exp4631.refresh_submission_package.v1"
PACKAGE_SCHEMA = "carnot.exp4631.submission_package.v1"
RANDOM_SEED = 4631

RESULT_RELATIVE_PATH = "results/experiment_4631_refresh_submission_package.json"
PACKAGE_RELATIVE_PATH = "results/experiment_4631_submission_package_operator_resubmit.json"
BASELINE_RESULT_RELATIVE_PATH = previous.RESULT_RELATIVE_PATH
BASELINE_PACKAGE_RELATIVE_PATH = previous.PACKAGE_RELATIVE_PATH
A3_LEVELUP_RELATIVE_PATH = "results/experiment_4630_levelup_selfplay.json"
A1_VARIANT_RELATIVE_PATH = "results/experiment_4628_dense_curiosity_progress_loop.json"
A2_VARIANT_RELATIVE_PATH = "results/experiment_4629_graduate_action_effect_predictor_live.json"
REGISTRY_RELATIVE_PATH = previous.REGISTRY_RELATIVE_PATH
BANK_DIR_RELATIVE_PATH = previous.BANK_DIR_RELATIVE_PATH

LIVE_SUBMITTABLE_PREV = 55
SUBMISSION_SCORE_GATE = previous.SUBMISSION_SCORE_GATE
INFERENCE_SUBSTRATE = previous.INFERENCE_SUBSTRATE
TERMINAL_PREFIXES = previous.TERMINAL_PREFIXES
REQUIRED_ARTIFACT_FIELDS = previous.REQUIRED_ARTIFACT_FIELDS
SPEC_REFS = [
    "REQ-CAPSTONE-4631",
    "SCENARIO-CAPSTONE-4631",
    "SCENARIO-CAPSTONE-4631-FIELD-PRINCIPLES",
]
A1_MEASUREMENT_KEYS = ("loop_measurement", "bare_measurement")
A2_MEASUREMENT_KEYS = ("live_measurement", "graduated_measurement", "predictor_measurement")

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    field: dict(principle) for field, principle in previous.FIELD_PRINCIPLES.items()
}
FIELD_PRINCIPLES["live_submittable_count_prev"] = {
    "principle": "55 -- the .426 A4 count, the apples-to-apples comparison."
}
FIELD_PRINCIPLES["count_delta"] = {
    "principle": (
        "live_submittable_level_count - 55 (>=0; positive = more submittable levels "
        "folded in), emitted explicitly so a null is annotated."
    )
}
FIELD_PRINCIPLES["levels_folded_in"] = {
    "principle": (
        "names the games whose new banks (A3 + A1/A2 variant solves) were folded into "
        "the refreshed package this milestone."
    )
}


def _as_int(value: Any, default: int = 0) -> int:
    return previous._as_int(value, default)


def _load_json(path: Path) -> JsonDict:
    return previous._load_json(path)


def _load_yaml(path: Path) -> JsonDict:
    return previous._load_yaml(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    previous._write_json(path, payload)


def _terminal_prefixed(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(TERMINAL_PREFIXES)


def _sha256(value: Any) -> str:
    return previous.base._sha256(value)


def _sha256_hex(value: Any) -> bool:
    return previous.base._sha256_hex(value)


def _iter_reproduced_variants(
    artifact: Mapping[str, Any],
    *,
    measurement_keys: Sequence[str],
) -> Iterable[tuple[str, int, JsonDict]]:
    """REQ-CAPSTONE-4631: yield unique offline-reproduced variant candidates."""

    seen: set[tuple[str, str, int]] = set()
    for measurement_key in measurement_keys:
        measurement = artifact.get(measurement_key)
        if not isinstance(measurement, Mapping):
            continue
        attempts = measurement.get("variant_attempts")
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, Mapping) or not attempt.get("game"):
                continue
            target_level = previous.base._candidate_level(attempt)
            if target_level <= 0 or not previous.base._is_candidate_reproduced(
                attempt, target_level
            ):
                continue
            game = str(attempt["game"])
            signature = str(attempt.get("variant_signature") or f"{game}:L{target_level}")
            key = (game, signature, target_level)
            if key in seen:
                continue
            seen.add(key)
            payload = dict(attempt)
            payload["source_measurement"] = measurement_key
            yield game, target_level, payload


def _iter_a2_newly_solved_variants(a2: Mapping[str, Any]) -> Iterable[tuple[str, int, JsonDict]]:
    offline = a2.get("offline_reproduced")
    offline = offline if isinstance(offline, Mapping) else {}
    rows = offline.get("newly_solved_variants")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping) or not row.get("game"):
                continue
            target_level = previous.base._candidate_level(row)
            if target_level <= 0:
                target_level = _as_int(row.get("reached_level"), 1)
            if target_level <= 0:
                continue
            payload = dict(row)
            yield str(row["game"]), target_level, payload
    yield from _iter_reproduced_variants(a2, measurement_keys=A2_MEASUREMENT_KEYS)


def collect_upstream_candidates(root: Path) -> dict[str, list[JsonDict]]:
    """SCENARIO-CAPSTONE-4631: read A3 plus any A1/A2 reproduced variants."""

    out: dict[str, list[JsonDict]] = {}

    a3 = _load_json(root / A3_LEVELUP_RELATIVE_PATH)
    game = str(a3.get("target_game") or "")
    if game:
        out.setdefault(game, []).append(
            previous.base._candidate(
                game=game,
                source_artifact=A3_LEVELUP_RELATIVE_PATH,
                source_payload_path=str(a3.get("standing_loop_result_path") or ""),
                payload=a3,
                level=previous.base._candidate_level(a3),
            )
        )

    a1 = _load_json(root / A1_VARIANT_RELATIVE_PATH)
    for game, target_level, payload in _iter_reproduced_variants(
        a1, measurement_keys=A1_MEASUREMENT_KEYS
    ):
        out.setdefault(game, []).append(
            previous.base._candidate(
                game=game,
                source_artifact=A1_VARIANT_RELATIVE_PATH,
                payload=payload,
                level=target_level,
            )
        )

    a2 = _load_json(root / A2_VARIANT_RELATIVE_PATH)
    for game, target_level, payload in _iter_a2_newly_solved_variants(a2):
        out.setdefault(game, []).append(
            previous.base._candidate(
                game=game,
                source_artifact=A2_VARIANT_RELATIVE_PATH,
                payload=payload,
                level=target_level,
            )
        )
    return out


def build_refreshed_rows(
    root: Path,
    *,
    registry: Mapping[str, Any],
    previous_package: Mapping[str, Any],
    candidates_by_game: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[list[JsonDict], list[str], list[JsonDict], dict[str, JsonDict], list[JsonDict]]:
    return previous.base.build_refreshed_rows(
        root,
        registry=registry,
        previous_package=previous_package,
        candidates_by_game=candidates_by_game or collect_upstream_candidates(root),
    )


def build_package_payload(rows: Sequence[Mapping[str, Any]], *, result_path: str) -> JsonDict:
    """REQ-CAPSTONE-4631: build the operator-only live-submit package manifest."""

    manifest: list[JsonDict] = []
    for row in rows:
        levels = _as_int(row.get("submittable_level"))
        if levels <= 0:
            continue
        manifest.append(
            {
                "game": str(row.get("game") or ""),
                "levels": levels,
                "offline_reproduced_level": _as_int(row.get("offline_reproduced_level")),
                "registry_reproduced_level": _as_int(row.get("registry_reproduced_level")),
                "trajectory_path": str(row.get("trajectory_path") or ""),
                "action_count": _as_int(row.get("trajectory_action_count")),
                "source": str(row.get("source") or ""),
                "env_matched": True,
                "env_match_basis": "offline_reproduction_gated_package_refresh_4631",
                "has_trajectory": bool(row.get("has_replayable_trajectory")),
                "has_env_adaptive_resolver": bool(row.get("has_env_adaptive_resolver")),
                "adaptive_solver": str(row.get("adaptive_solver") or ""),
                "claim_capped": bool(row.get("claim_capped")),
            }
        )
    return {
        "experiment": "experiment_4631_submission_package_operator_resubmit",
        "schema": PACKAGE_SCHEMA,
        "source_result_path": result_path,
        "package_manifest": manifest,
        "claimed_total_levels": sum(_as_int(row.get("levels")) for row in manifest),
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "reproducibility_checksum": _sha256(manifest),
    }


def _ready_for_operator(rows: Sequence[Mapping[str, Any]], live_count: int) -> bool:
    return previous.base._ready_for_operator(rows, live_count)


def _honest_verdict(live_count: int, count_delta: int) -> str:
    if live_count > SUBMISSION_SCORE_GATE and count_delta > 0:
        return f"success: package_refreshed_live_submittable_{live_count}_above_33"
    return "complete: package_refreshed_unchanged_depth."


def compute_reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    return _sha256(
        {
            "live_submittable_level_count": artifact.get("live_submittable_level_count"),
            "live_submittable_count_prev": artifact.get("live_submittable_count_prev"),
            "count_delta": artifact.get("count_delta"),
            "levels_folded_in": artifact.get("levels_folded_in"),
            "per_game_submittable": artifact.get("per_game_submittable"),
            "claimed_caps": artifact.get("claimed_caps"),
            "trajectory_refreshes": artifact.get("trajectory_refreshes"),
            "env_adaptive_recovery": artifact.get("env_adaptive_recovery"),
            "upstream_fold_audit": artifact.get("upstream_fold_audit"),
            "random_seed": artifact.get("random_seed"),
        }
    )


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    previous_count: int,
    levels_folded_in: Sequence[str],
    trajectory_refreshes: Sequence[Mapping[str, Any]],
    claimed_caps: Mapping[str, Mapping[str, Any]],
    upstream_fold_audit: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    live_count = sum(_as_int(row.get("submittable_level")) for row in rows)
    count_delta = live_count - previous_count
    ready = _ready_for_operator(rows, live_count)
    package_payload = build_package_payload(rows, result_path=RESULT_RELATIVE_PATH)
    env_adaptive_recovery = [
        {
            "game": str(row.get("game")),
            "adaptive_solver": str(row.get("adaptive_solver") or ""),
            "drift_robust": bool(row.get("drift_robust")),
        }
        for row in rows
        if row.get("has_env_adaptive_resolver") and _as_int(row.get("submittable_level")) > 0
    ]
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": _honest_verdict(live_count, count_delta),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "live_submittable_level_count": live_count,
        "live_submittable_count_prev": previous_count,
        "count_delta": count_delta,
        "levels_folded_in": list(levels_folded_in),
        "refreshed_package_path": PACKAGE_RELATIVE_PATH,
        "per_game_submittable": [dict(row) for row in rows],
        "ready_for_operator_submit": ready,
        "offline_reproduced": ready,
        "offline_reproduction_by_game": {
            str(row.get("game")): _as_int(row.get("offline_reproduced_level"))
            for row in rows
            if _as_int(row.get("submittable_level")) > 0
        },
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions_checked),
        "null_delta_methodology_note": (
            "count_delta==0 is an unchanged-depth refresh measured against the same .426 A4 count."
            if count_delta == 0
            else ""
        ),
        "submitted_to_leaderboard": False,
        "package_manifest": package_payload["package_manifest"],
        "claimed_caps": {key: dict(value) for key, value in claimed_caps.items()},
        "trajectory_refreshes": [dict(item) for item in trajectory_refreshes],
        "env_adaptive_recovery": env_adaptive_recovery,
        "upstream_fold_audit": [dict(item) for item in upstream_fold_audit],
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "result_path": RESULT_RELATIVE_PATH,
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": "precondition_check_no_inference",
        "verifier_is_oracle": False,
        "live_submittable_level_count": 0,
        "live_submittable_count_prev": LIVE_SUBMITTABLE_PREV,
        "count_delta": -LIVE_SUBMITTABLE_PREV,
        "levels_folded_in": [],
        "refreshed_package_path": "",
        "per_game_submittable": [],
        "ready_for_operator_submit": False,
        "offline_reproduced": False,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions_checked),
        "null_delta_methodology_note": "",
        "submitted_to_leaderboard": False,
        "package_manifest": [],
        "claimed_caps": {},
        "trajectory_refreshes": [],
        "env_adaptive_recovery": [],
        "upstream_fold_audit": [],
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "result_path": RESULT_RELATIVE_PATH,
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    if not _terminal_prefixed(artifact.get("honest_verdict")):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") not in {
        INFERENCE_SUBSTRATE,
        "precondition_check_no_inference",
    }:
        errors.append("inference_substrate must equal the declared offline packaging substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    for field in (
        "live_submittable_level_count",
        "live_submittable_count_prev",
        "count_delta",
        "random_seed",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    for field in ("levels_folded_in", "per_game_submittable"):
        if not isinstance(artifact.get(field), list):
            errors.append(f"{field} must be list")
    if type(artifact.get("ready_for_operator_submit")) is not bool:
        errors.append("ready_for_operator_submit must be bare bool")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be mapping")
    if not _sha256_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be sha256 hex")
    if artifact.get("reproducibility_checksum") != compute_reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum must match the artifact payload")

    live = artifact.get("live_submittable_level_count")
    prev_count = artifact.get("live_submittable_count_prev")
    delta = artifact.get("count_delta")
    if type(live) is int and type(prev_count) is int and type(delta) is int:
        if delta != live - prev_count:
            errors.append(
                "count_delta must equal live_submittable_level_count - live_submittable_count_prev"
            )
        if artifact.get("ready_for_operator_submit") is True and live <= SUBMISSION_SCORE_GATE:
            errors.append("ready_for_operator_submit requires count strictly above 33")
        if delta == 0 and not artifact.get("null_delta_methodology_note"):
            errors.append("null_delta_methodology_note required when count_delta is zero")

    rows = artifact.get("per_game_submittable")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                errors.append("per_game_submittable rows must be mappings")
                continue
            game = str(row.get("game") or "<unknown>")
            submittable = _as_int(row.get("submittable_level"))
            offline = _as_int(row.get("offline_reproduced_level"))
            registry = _as_int(row.get("registry_reproduced_level"))
            if submittable > offline:
                errors.append(f"{game} submittable exceeds offline reproduction")
            if submittable > registry:
                errors.append(f"{game} submittable exceeds registry claim")
            if submittable > 0 and not (
                row.get("has_replayable_trajectory") is True
                or row.get("has_env_adaptive_resolver") is True
            ):
                errors.append(f"{game} counted without trajectory or adaptive resolver")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    _write_json(path, artifact)
    return path


def write_package(root: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    payload = build_package_payload(rows, result_path=RESULT_RELATIVE_PATH)
    path = root / PACKAGE_RELATIVE_PATH
    _write_json(path, payload)
    return path


def check_preconditions(
    root: Path,
    *,
    offline_arcade_checker: Callable[[], bool] | None = None,
) -> JsonDict:
    return previous.base.check_preconditions(root, offline_arcade_checker=offline_arcade_checker)


def first_precondition_miss(checked: Mapping[str, Any]) -> str | None:
    return previous.base.first_precondition_miss(checked)


def _previous_live_count(previous_result: Mapping[str, Any]) -> int:
    return _as_int(previous_result.get("live_submittable_level_count"), LIVE_SUBMITTABLE_PREV)


def run(
    root: Path = REPO_ROOT,
    *,
    offline_arcade_checker: Callable[[], bool] | None = None,
    now: Callable[[], float] = time.perf_counter,
) -> JsonDict:
    """SCENARIO-CAPSTONE-4631: refresh the package without online submission."""

    started = now()
    preconditions = check_preconditions(root, offline_arcade_checker=offline_arcade_checker)
    miss = first_precondition_miss(preconditions)
    if miss is not None:
        artifact = _blocked_artifact(
            reason=miss,
            preconditions_checked=preconditions,
            duration_s=now() - started,
        )
        write_artifact(root, artifact)
        return artifact

    registry = _load_yaml(root / REGISTRY_RELATIVE_PATH)
    previous_package = _load_json(root / BASELINE_PACKAGE_RELATIVE_PATH)
    previous_result = _load_json(root / BASELINE_RESULT_RELATIVE_PATH)
    rows, folded, trajectory_refreshes, claimed_caps, upstream_audit = build_refreshed_rows(
        root,
        registry=registry,
        previous_package=previous_package,
    )
    artifact = build_artifact(
        rows=rows,
        previous_count=_previous_live_count(previous_result),
        levels_folded_in=folded,
        trajectory_refreshes=trajectory_refreshes,
        claimed_caps=claimed_caps,
        upstream_fold_audit=upstream_audit,
        preconditions_checked=preconditions,
        duration_s=now() - started,
    )
    write_artifact(root, artifact)
    write_package(root, rows)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run(REPO_ROOT)
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    print(f"live_submittable_level_count={artifact['live_submittable_level_count']}")
    print(f"ready_for_operator_submit={artifact['ready_for_operator_submit']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())

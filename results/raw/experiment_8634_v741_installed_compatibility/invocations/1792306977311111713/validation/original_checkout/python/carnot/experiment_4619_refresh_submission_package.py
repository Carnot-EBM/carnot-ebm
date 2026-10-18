"""Experiment 4619: refresh the ARC operator-resubmit package.

Spec refs: REQ-CAPSTONE-4619, SCENARIO-CAPSTONE-4619,
SCENARIO-CAPSTONE-4619-FIELD-PRINCIPLES.
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

from carnot import experiment_4607_refresh_submission_package as base


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4619_refresh_submission_package"
SCHEMA = "carnot.exp4619.refresh_submission_package.v1"
PACKAGE_SCHEMA = "carnot.exp4619.submission_package.v1"
RANDOM_SEED = 4619

RESULT_RELATIVE_PATH = "results/experiment_4619_refresh_submission_package.json"
PACKAGE_RELATIVE_PATH = "results/experiment_4619_submission_package_operator_resubmit.json"
BASELINE_RESULT_RELATIVE_PATH = "results/experiment_4607_refresh_submission_package.json"
BASELINE_PACKAGE_RELATIVE_PATH = "results/experiment_4607_submission_package_operator_resubmit.json"
A3_LEVELUP_RELATIVE_PATH = "results/experiment_4618_levelup_selfplay.json"
A2_VARIANT_RELATIVE_PATH = "results/experiment_4617_graduate_spatial_value_head_live.json"
REGISTRY_RELATIVE_PATH = base.REGISTRY_RELATIVE_PATH
BANK_DIR_RELATIVE_PATH = base.BANK_DIR_RELATIVE_PATH

LIVE_SUBMITTABLE_PREV = 55
SUBMISSION_SCORE_GATE = base.SUBMISSION_SCORE_GATE
INFERENCE_SUBSTRATE = base.INFERENCE_SUBSTRATE
TERMINAL_PREFIXES = base.TERMINAL_PREFIXES
REQUIRED_ARTIFACT_FIELDS = base.REQUIRED_ARTIFACT_FIELDS
SPEC_REFS = [
    "REQ-CAPSTONE-4619",
    "SCENARIO-CAPSTONE-4619",
    "SCENARIO-CAPSTONE-4619-FIELD-PRINCIPLES",
]
A2_MEASUREMENT_KEYS = ("graduated_measurement", "linear_measurement", "bare_measurement")

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    field: dict(principle) for field, principle in base.FIELD_PRINCIPLES.items()
}
FIELD_PRINCIPLES["live_submittable_count_prev"] = {
    "principle": "55 -- the .425 A4 count, the apples-to-apples comparison."
}
FIELD_PRINCIPLES["levels_folded_in"] = {
    "principle": (
        "names the games whose new banks (A3 + A2 variant solves) were folded into "
        "the refreshed package this milestone."
    )
}


def _as_int(value: Any, default: int = 0) -> int:
    return base._as_int(value, default)


def _load_json(path: Path) -> JsonDict:
    return base._load_json(path)


def _load_yaml(path: Path) -> JsonDict:
    return base._load_yaml(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._write_json(path, payload)


def _iter_a2_reproduced_variants(a2: Mapping[str, Any]) -> Iterable[tuple[str, int, JsonDict]]:
    seen: set[tuple[str, str, int]] = set()
    for measurement_key in A2_MEASUREMENT_KEYS:
        measurement = a2.get(measurement_key)
        if not isinstance(measurement, Mapping):  # pragma: no cover - defensive malformed input
            continue
        attempts = measurement.get("variant_attempts")
        if not isinstance(attempts, list):  # pragma: no cover - defensive malformed input
            continue
        for attempt in attempts:
            if not isinstance(attempt, Mapping) or not attempt.get("game"):
                continue
            target_level = base._candidate_level(attempt)
            if target_level <= 0 or not base._is_candidate_reproduced(attempt, target_level):
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


def collect_upstream_candidates(root: Path) -> dict[str, list[JsonDict]]:
    out: dict[str, list[JsonDict]] = {}

    a3 = _load_json(root / A3_LEVELUP_RELATIVE_PATH)
    game = str(a3.get("target_game") or "")
    if game:
        out.setdefault(game, []).append(
            base._candidate(
                game=game,
                source_artifact=A3_LEVELUP_RELATIVE_PATH,
                source_payload_path=str(a3.get("standing_loop_result_path") or ""),
                payload=a3,
                level=base._candidate_level(a3),
            )
        )

    a2 = _load_json(root / A2_VARIANT_RELATIVE_PATH)
    for game, target_level, payload in _iter_a2_reproduced_variants(a2):
        out.setdefault(game, []).append(
            base._candidate(
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
    return base.build_refreshed_rows(
        root,
        registry=registry,
        previous_package=previous_package,
        candidates_by_game=candidates_by_game or collect_upstream_candidates(root),
    )


def build_package_payload(rows: Sequence[Mapping[str, Any]], *, result_path: str) -> JsonDict:
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
                "env_match_basis": "offline_reproduction_gated_package_refresh_4619",
                "has_trajectory": bool(row.get("has_replayable_trajectory")),
                "has_env_adaptive_resolver": bool(row.get("has_env_adaptive_resolver")),
                "adaptive_solver": str(row.get("adaptive_solver") or ""),
                "claim_capped": bool(row.get("claim_capped")),
            }
        )
    return {
        "experiment": "experiment_4619_submission_package_operator_resubmit",
        "schema": PACKAGE_SCHEMA,
        "source_result_path": result_path,
        "package_manifest": manifest,
        "claimed_total_levels": sum(_as_int(row.get("levels")) for row in manifest),
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "reproducibility_checksum": base._sha256(manifest),
    }


def _honest_verdict(live_count: int, count_delta: int) -> str:
    if live_count > SUBMISSION_SCORE_GATE and count_delta > 0:
        return f"success: package_refreshed_live_submittable_{live_count}_above_33"
    return "complete: package_refreshed_unchanged_depth."


def compute_reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    return base._sha256(
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
    ready = base._ready_for_operator(rows, live_count)
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
            "count_delta==0 is an unchanged-depth refresh measured against the same .425 A4 count."
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
    errors = base.artifact_schema_errors(artifact)
    if artifact.get("reproducibility_checksum") != compute_reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum must match the artifact payload")
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


def run(
    root: Path = REPO_ROOT,
    *,
    offline_arcade_checker: Callable[[], bool] | None = None,
    now: Callable[[], float] = time.perf_counter,
) -> JsonDict:
    started = now()
    preconditions = base.check_preconditions(root, offline_arcade_checker=offline_arcade_checker)
    miss = base.first_precondition_miss(preconditions)
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
        previous_count=base._previous_live_count(previous_result),
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

#!/usr/bin/env python3
"""Experiment 7492: reduce timed and compatible E6 cost evidence.

This script calls no model or game. It imports the Experiment 7490 reducer.

Spec: REQ-ARC-WMTE-7492 and SCENARIO-ARC-WMTE-7492-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import importlib.util
import json
import random
from pathlib import Path
import time
from types import ModuleType
from typing import Any


EXPERIMENT_ID = 7492
RANDOM_SEED = 7492
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7492_e6_timed_cost_profile.json")
EXP7490_PATH = Path("results/experiment_7490_e6_live_loop_cost_profile.json")
EXP7491_PATH = Path("results/experiment_7491_e6_timed_live_profile.json")
EXP7490_SCRIPT = Path("scripts/experiments/experiment_7490_e6_live_loop_cost_profile.py")
REPLACEABLE_POINTS = (
    "candidate_selection",
    "induction_and_generation",
    "supervisor",
    "planner",
)


def _load_7490() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "carnot_experiment_7490", REPO_ROOT / EXP7490_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Experiment 7490 reducer is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e6_base = _load_7490()


def progress(message: str) -> None:
    print(message, flush=True)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _checksum(value: Mapping[str, Any]) -> str:
    stable = {
        key: child
        for key, child in value.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return "sha256:" + hashlib.sha256(_canonical_bytes(stable)).hexdigest()


def write_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one artifact only to the caller's explicit path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _unique_complete(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        episode_id = row.get("episode_id")
        if (
            isinstance(episode_id, str)
            and row.get("complete") is True
            and row.get("current_model") is True
            and row.get("numeric_eligible") is True
        ):
            unique.setdefault(episode_id, row)
    return list(unique.values())


def _cluster_intervals(
    rows: Sequence[Mapping[str, Any]], *, draws: int = 2_000
) -> dict[str, dict[str, float | int | None]]:
    """Bootstrap games as clusters with the fixed reducer seed."""

    by_game: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[str(row.get("game"))].append(row)
    games = sorted(by_game)
    output: dict[str, dict[str, float | int | None]] = {}
    if not games:
        return output
    rng = random.Random(RANDOM_SEED)
    for point in e6_base.DECISION_POINTS:
        samples: list[float] = []
        for _ in range(draws):
            sampled_games = [games[rng.randrange(len(games))] for _ in games]
            wall = 0.0
            phase = 0.0
            for game in sampled_games:
                for row in by_game[game]:
                    wall += float(row.get("episode_wall_s") or 0.0)
                    for item in row.get("phases") or []:
                        if isinstance(item, Mapping) and item.get("decision_point") == point:
                            phase += float(item.get("wall_s") or 0.0)
            samples.append(phase / wall if wall > 0 else 0.0)
        samples.sort()
        low = samples[int(0.025 * (len(samples) - 1))]
        high = samples[int(0.975 * (len(samples) - 1))]
        output[point] = {
            "lower": low,
            "upper": high,
            "bootstrap_draws": draws,
            "game_clusters": len(games),
        }
    return output


def _support(
    timed: Sequence[Mapping[str, Any]], earlier: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    combined = _unique_complete([*earlier, *timed])
    timed_complete = _unique_complete(timed)
    games = {str(row.get("game")) for row in combined if row.get("game")}
    timed_games = {str(row.get("game")) for row in timed_complete if row.get("game")}
    return {
        "complete_current_model_episodes": len(combined),
        "current_model_games": len(games),
        "fully_timed_complete_episodes": len(timed_complete),
        "fully_timed_games": len(timed_games),
        "earlier_compatible_complete_episodes": len(_unique_complete(earlier)),
    }


def _publication_gates(
    timed: Sequence[Mapping[str, Any]],
    earlier: Sequence[Mapping[str, Any]],
    positive_control: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    support = _support(timed, earlier)
    complete_timed = _unique_complete(timed)
    required = set(e6_base.DECISION_POINTS)
    separation = all(
        {
            str(phase.get("decision_point"))
            for phase in row.get("phases") or []
            if isinstance(phase, Mapping)
        }
        >= required
        for row in complete_timed
    )
    reconciliation = bool(complete_timed) and all(
        row.get("reconciliation_passed") is True for row in complete_timed
    )
    token_join = bool(complete_timed) and all(
        row.get("token_join_passed") is True for row in complete_timed
    )
    total_wall = sum(float(row.get("episode_wall_s") or 0.0) for row in complete_timed)
    attributed_wall = sum(
        float(phase.get("wall_s") or 0.0)
        for row in complete_timed
        for phase in row.get("phases") or []
        if isinstance(phase, Mapping) and phase.get("concurrent") is not True
    )
    coverage = attributed_wall / total_wall if total_wall > 0 else 0.0
    backend_tokens = sum(int(row.get("backend_usage_tokens") or 0) for row in complete_timed)
    attributed_tokens = sum(
        int(phase.get("tokens") or 0)
        for row in complete_timed
        for phase in row.get("phases") or []
        if isinstance(phase, Mapping)
    )
    token_coverage = attributed_tokens / backend_tokens if backend_tokens > 0 else 1.0
    gates = [
        {
            "check": "complete_current_model_episode_floor",
            "expected": 30,
            "observed": support["complete_current_model_episodes"],
            "passed": support["complete_current_model_episodes"] >= 30,
        },
        {
            "check": "current_model_game_floor",
            "expected": 10,
            "observed": support["current_model_games"],
            "passed": support["current_model_games"] >= 10,
        },
        {
            "check": "fully_timed_episode_floor",
            "expected": 30,
            "observed": support["fully_timed_complete_episodes"],
            "passed": support["fully_timed_complete_episodes"] >= 30,
        },
        {
            "check": "fully_timed_game_floor",
            "expected": 10,
            "observed": support["fully_timed_games"],
            "passed": support["fully_timed_games"] >= 10,
        },
        {
            "check": "positive_control",
            "expected": True,
            "observed": positive_control.get("passed"),
            "passed": positive_control.get("passed") is True,
        },
        {
            "check": "span_reconciliation",
            "expected": True,
            "observed": reconciliation,
            "passed": reconciliation,
        },
        {
            "check": "backend_token_join",
            "expected": True,
            "observed": token_join,
            "passed": token_join,
        },
        {
            "check": "required_seams_separate",
            "expected": True,
            "observed": separation,
            "passed": separation,
        },
        {
            "check": "wall_attribution_coverage",
            "expected": 0.95,
            "observed": coverage,
            "passed": coverage >= 0.95,
        },
        {
            "check": "token_attribution_coverage",
            "expected": 0.99,
            "observed": token_coverage,
            "passed": token_coverage >= 0.99,
        },
    ]
    details = {
        "support": support,
        "wall_attribution_fraction": coverage,
        "token_attribution_fraction": token_coverage,
        "required_seams_separate": separation,
        "reconciliation_passed": reconciliation,
        "token_join_passed": token_join,
    }
    return details, gates


def build_profile(
    *,
    timed_episodes: Sequence[Mapping[str, Any]],
    earlier_episodes: Sequence[Mapping[str, Any]],
    cited_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply every E6 publication gate and build the terminal profile."""

    started = time.perf_counter()
    positive = e6_base.run_positive_control()
    details, gates = _publication_gates(timed_episodes, earlier_episodes, positive)
    complete_timed = _unique_complete(timed_episodes)
    gates_before_kill_pass = all(row["passed"] for row in gates)
    provisional = e6_base.reduce_episodes(
        complete_timed,
        publish_numeric=gates_before_kill_pass,
    )
    intervals = _cluster_intervals(complete_timed) if gates_before_kill_pass else {}
    if gates_before_kill_pass:
        for point, interval in intervals.items():
            provisional["decision_points"][point]["wall_fraction_interval_95"] = interval
    replaceable_fraction: float | None = None
    replaceable_amdahl: float | None = None
    if gates_before_kill_pass:
        replaceable_fraction = sum(
            float(provisional["decision_points"][point]["wall_fraction"] or 0.0)
            for point in REPLACEABLE_POINTS
        )
        if replaceable_fraction < 1.0:
            replaceable_amdahl = 1.0 / (1.0 - replaceable_fraction)
    kill_fired = replaceable_fraction is not None and replaceable_fraction < 0.05
    numeric_published = gates_before_kill_pass and not kill_fired
    profile = (
        provisional
        if numeric_published
        else e6_base.reduce_episodes(complete_timed, publish_numeric=False)
    )
    if numeric_published:
        for point, interval in intervals.items():
            profile["decision_points"][point]["wall_fraction_interval_95"] = interval
    else:
        for point in e6_base.DECISION_POINTS:
            profile["decision_points"][point]["wall_fraction_interval_95"] = {
                "lower": None,
                "upper": None,
                "bootstrap_draws": 0,
                "game_clusters": details["support"]["fully_timed_games"],
            }
    verdict = (
        "complete_numeric_profile_published"
        if numeric_published
        else "complete_coverage_only_publication_gate_or_kill_rule_fired"
    )
    failed = [row["check"] for row in gates if row["passed"] is not True]
    if kill_fired:
        failed.append("replaceable_work_at_least_five_percent")
    artifact: dict[str, Any] = {
        "schema": "carnot.arc.e6_timed_cost_profile.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": "positive" if numeric_published else "null",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "model_invoked": False,
        "MODEL_SPECS": [],
        "model_specs": [],
        "current_invocation_counts": {
            "model_calls": 0,
            "game_runs": 0,
            "gpu_jobs": 0,
            "network_calls": 0,
            "submissions": 0,
        },
        "random_seed": RANDOM_SEED,
        "sample_size": details["support"],
        "cohort_accounting": {
            "timed_episode_ids": [row["episode_id"] for row in complete_timed],
            "earlier_compatible_episode_ids": [
                row["episode_id"] for row in _unique_complete(earlier_episodes)
            ],
            "numeric_shares_use_fully_timed_cohort_only": True,
            "earlier_missing_seams_not_imputed": True,
        },
        "decision_point_profile": profile,
        "positive_control": positive,
        "reconciliation": {
            "wall_attribution_fraction": details["wall_attribution_fraction"],
            "token_attribution_fraction": details["token_attribution_fraction"],
            "span_reconciliation_passed": details["reconciliation_passed"],
            "backend_token_join_passed": details["token_join_passed"],
        },
        "gate_results": {
            "numeric_share_published": numeric_published,
            "all_pre_kill_publication_gates_passed": gates_before_kill_pass,
            "failed_checks": failed,
        },
        "acceptance_gate_results": gates,
        "kill_rule": {
            "replaceable_points": list(REPLACEABLE_POINTS),
            "replaceable_work_fraction": replaceable_fraction,
            "under_five_percent": kill_fired,
            "speed_claims_stopped": not numeric_published,
        },
        "amdahl_ceiling_all_replaceable_work": (replaceable_amdahl if numeric_published else None),
        "cited_upstream_artifacts": deepcopy(list(cited_artifacts)),
        "missing_verifier_gaps": (
            [] if details["required_seams_separate"] else ["world_model_verification.exclusive_ns"]
        ),
        "speed_claim": (
            "measured_cost_share_and_conditional_zero-overhead_ceiling"
            if numeric_published
            else None
        ),
        "hidden_game_efficacy_claim": False,
        "flagged_adversarial": False,
        "duration_s": time.perf_counter() - started,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _load_inputs(project_root: Path) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    exp7490_path = project_root / EXP7490_PATH
    exp7491_path = project_root / EXP7491_PATH
    exp7490 = json.loads(exp7490_path.read_text())
    exp7491 = json.loads(exp7491_path.read_text())
    if exp7490.get("flagged_adversarial") is True or exp7491.get("flagged_adversarial") is True:
        raise ValueError("flagged E6 input cannot be reduced")
    timed = [
        dict(row) for row in exp7491.get("normalized_cost_rows") or [] if isinstance(row, Mapping)
    ]
    earlier = [
        {
            "episode_id": str(row["episode_id"]),
            "game": row.get("game"),
            "complete": row.get("complete") is True,
            "current_model": True,
            "numeric_eligible": row.get("complete") is True,
            "cohort": "experiment_7490_compatible",
        }
        for row in exp7490.get("normalized_episode_coverage_rows") or []
        if isinstance(row, Mapping) and row.get("episode_id")
    ]
    cited = [
        {
            "experiment_id": 7490,
            "path": EXP7490_PATH.as_posix(),
            "fields_imported": ["normalized_episode_coverage_rows", "gate_results"],
            "sha256": _sha256_file(exp7490_path),
        },
        {
            "experiment_id": 7491,
            "path": EXP7491_PATH.as_posix(),
            "fields_imported": ["normalized_cost_rows", "model_specs", "span_shards"],
            "sha256": _sha256_file(exp7491_path),
        },
    ]
    for source in exp7491.get("span_shards") or []:
        if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
            continue
        shard_path = Path(str(source["path"]))
        if not shard_path.is_absolute():
            shard_path = project_root / shard_path
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing Experiment 7491 span shard: {shard_path}")
        observed_sha = _sha256_file(shard_path)
        if source.get("sha256") != observed_sha:
            raise ValueError(f"Experiment 7491 span shard hash mismatch: {shard_path}")
        cited.append(
            {
                "experiment_id": 7491,
                "path": str(source["path"]),
                "fields_imported": ["exclusive_span_rows"],
                "sha256": observed_sha,
                "role": "exclusive_span_shard",
            }
        )
    return timed, earlier, cited


def build_artifact(project_root: Path) -> dict[str, Any]:
    """Read immutable upstream inputs and run the E6 reducer."""

    progress("STEP 1: load Experiment 7491 and the 26 compatible Experiment 7490 rows")
    timed, earlier, cited = _load_inputs(project_root)
    progress(f"STEP 2: retain {len(timed)} timed rows and {len(earlier)} earlier rows")
    progress("STEP 3: run the Experiment 7490 positive control")
    artifact = build_profile(
        timed_episodes=timed,
        earlier_episodes=earlier,
        cited_artifacts=cited,
    )
    progress("STEP 4: apply sample, reconciliation, separation, and five-percent gates")
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    output = args.output or project_root / RESULT_PATH
    if not output.is_absolute():
        output = project_root / output
    artifact = build_artifact(project_root)
    progress(f"STEP 5: write {output}")
    write_artifact(output, artifact)
    progress("STEP 6: Experiment 7492 reduction complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

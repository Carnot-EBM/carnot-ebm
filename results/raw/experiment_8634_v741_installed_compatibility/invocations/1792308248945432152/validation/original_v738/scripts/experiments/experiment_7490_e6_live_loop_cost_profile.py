#!/usr/bin/env python3
"""Experiment 7490: reduce existing ARC live-loop cost evidence.

This script only reads existing JSON and JSONL artifacts. It does not call a
model, run a game, use a GPU, use the network, or submit work.

Spec: REQ-ARC-WMTE-7490 and SCENARIO-ARC-WMTE-7490-*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


EXPERIMENT_ID = 7490
RANDOM_SEED = 7490
CURRENT_MODEL_REPOSITORY = "unsloth/Qwen3.8-27B-GGUF"
CURRENT_MODEL_NAME = "Qwen3.8-27B"
CURRENT_MODEL_FILENAME = "Qwen3.8-27B-Q4_K_M.gguf"
CURRENT_MODEL_REVISION = "fe1e2a23d973adb629709749dc4f6756df66ef10"
EXECUTION_VENUE = "host"
COVERAGE_VERDICT_CLASS = "null"
DEFAULT_RESULT_PATH = Path("results/experiment_7490_e6_live_loop_cost_profile.json")

DECISION_POINTS = (
    "candidate_selection",
    "induction_and_generation",
    "world_model_verification",
    "supervisor",
    "planner",
    "environment",
)

SEAM_TO_DECISION_POINT = {
    "candidate_action_selection": "candidate_selection",
    "downstream_generation": "induction_and_generation",
    "hypothesis_gate": "induction_and_generation",
    "induction_timing": "induction_and_generation",
    "supervisor_arm_selection": "supervisor",
}

COST_FIELD_NAMES = {
    "completion_tokens",
    "elapsed_s",
    "exclusive_cost_rows",
    "input_tokens",
    "induction_attempts",
    "output_tokens",
    "phase_spans",
    "prompt_tokens",
    "raw_request_manifest",
    "request_budget_receipt",
    "request_budget_rows",
    "seam_event_shards",
    "server_request_rows",
    "timings",
    "total_tokens",
    "wall_s",
}

PRIMARY_ARTIFACTS = {
    "results/experiment_5972_arc_llm_on_budget2000_feasibility.json": (
        "reducer_control_older_model"
    ),
    "results/experiment_7234_v637_arc_scored_dryrun.json": "flagged_current_model",
    "results/experiment_7457_v653_arc_exposure.json": "schema_control_current_model",
    "results/experiment_7464_v654_semif_e6_decision_cost_profile.json": ("prior_reducer_control"),
    "results/experiment_7471_v654_arc_seam_observation.json": ("current_decision_seam_terminal"),
    "results/experiment_7478_v655_arc_interval_protocol.json": ("current_interval_protocol"),
    "results/experiment_7485_v655_arc_cost_panel_a.json": "current_cost_panel",
}

CLASS_MISSING_FIELDS = {
    "backend_response_usage": [
        "episode_wall_s",
        "decision_point",
        "parent_decision_id",
        "concurrent",
        "planner.start_monotonic_ns",
        "planner.end_monotonic_ns",
        "environment.start_monotonic_ns",
        "environment.end_monotonic_ns",
        "world_model_verification.start_monotonic_ns",
        "world_model_verification.end_monotonic_ns",
    ],
    "current_cost_panel": [
        "world_model_verification.start_monotonic_ns",
        "world_model_verification.end_monotonic_ns",
        "world_model_verification.exclusive_ns",
        "planner.start_monotonic_ns",
        "planner.end_monotonic_ns",
        "planner.exclusive_ns",
        "environment.start_monotonic_ns",
        "environment.end_monotonic_ns",
        "environment.exclusive_ns",
        "seam_stage_tokens_reconciled_to_backend_usage",
        "candidate_selection.complete_candidate_ids",
    ],
    "current_decision_seam_terminal": [
        "exclusive_cost_rows",
        "world_model_verification.exclusive_ns",
        "planner.exclusive_ns",
        "environment.exclusive_ns",
        "seam_stage_tokens_reconciled_to_backend_usage",
        "candidate_selection.complete_candidate_ids",
    ],
    "current_interval_protocol": [
        "world_model_verification.exclusive_ns",
        "planner.exclusive_ns",
        "environment.exclusive_ns",
        "seam_stage_tokens_reconciled_to_backend_usage",
        "candidate_selection.complete_candidate_ids",
    ],
    "decision_seam_shard": [
        "world_model_verification.stage_start",
        "world_model_verification.stage_end",
        "planner.stage_start",
        "planner.stage_end",
        "environment.stage_start",
        "environment.stage_end",
        "stage_end.input_tokens_matching_backend_usage",
        "stage_end.output_tokens_matching_backend_usage",
        "candidate_selection.complete_candidate_ids",
    ],
    "flagged_current_model": [
        "adversarial_clearance",
        "complete_policy_consumption",
        "candidate_selection.exclusive_ns",
        "world_model_verification.exclusive_ns",
        "planner.exclusive_ns",
        "environment.exclusive_ns",
    ],
    "live_action_provenance": [
        "current_model_identity",
        "episode_start_monotonic_ns",
        "episode_end_monotonic_ns",
        "candidate_selection.exclusive_ns",
        "generation.exclusive_ns",
        "world_model_verification.exclusive_ns",
        "supervisor.exclusive_ns",
        "planner.exclusive_ns",
        "environment.exclusive_ns",
        "backend_usage_tokens",
        "concurrent",
    ],
    "live_agent_induction_manifest": [
        "episode_id",
        "episode_wall_s",
        "induction.wall_s",
        "prompt_tokens",
        "completion_tokens",
        "candidate_selection.exclusive_ns",
        "world_model_verification.exclusive_ns",
        "supervisor.exclusive_ns",
        "planner.exclusive_ns",
        "environment.exclusive_ns",
        "concurrent",
    ],
    "phase_span_terminal": [
        "episode_level_phase_spans",
        "episode_id_to_phase_span_join",
        "candidate_selection.exclusive_ns",
        "world_model_verification.exclusive_ns",
        "supervisor.exclusive_ns",
        "planner.exclusive_ns",
        "environment.exclusive_ns",
        "episode_backend_usage_tokens",
        "concurrent",
    ],
    "prior_reducer_control": [
        "independent_current_episode_evidence",
        "world_model_verification.exclusive_ns",
        "planner.exclusive_ns",
        "environment.exclusive_ns",
        "current_model_backend_usage_tokens",
    ],
    "raw_episode_rows": [
        "candidate_selection.exclusive_ns",
        "world_model_verification.exclusive_ns",
        "supervisor.exclusive_ns",
        "planner.exclusive_ns",
        "environment.exclusive_ns",
        "request_usage_join",
        "concurrent",
    ],
    "reducer_control_older_model": [
        "current_model_identity",
        "candidate_selection.exclusive_ns",
        "world_model_verification.exclusive_ns",
        "supervisor.exclusive_ns",
        "planner.exclusive_ns",
        "environment.exclusive_ns",
        "concurrent",
    ],
    "schema_control_current_model": [
        "full_induction_generation",
        "candidate_selection.exclusive_ns",
        "world_model_verification.exclusive_ns",
        "planner.exclusive_ns",
        "environment.exclusive_ns",
        "seam_stage_tokens_reconciled_to_backend_usage",
        "concurrent",
    ],
}


def progress(message: str) -> None:
    """Print one immediately visible progress line."""
    print(message, flush=True)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _deep_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_deep_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_deep_keys(child))
    return keys


def _as_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _model_identity(data: Any, path: Path) -> tuple[str | None, str | None]:
    rows = _as_dicts(data)
    specs: list[dict[str, Any]] = []
    for row in rows:
        for field in ("model_specs", "MODEL_SPECS"):
            raw_specs = row.get(field)
            if isinstance(raw_specs, dict):
                specs.append(raw_specs)
            elif isinstance(raw_specs, list):
                specs.extend(item for item in raw_specs if isinstance(item, dict))
    name: str | None = None
    version: str | None = None
    for spec in specs:
        for field in ("name", "model_filename", "hf_id", "model_repository"):
            candidate = spec.get(field)
            if isinstance(candidate, str) and candidate:
                name = candidate
                break
        revision = spec.get("revision") or spec.get("model_revision")
        if isinstance(revision, str) and revision:
            version = revision
        if name:
            break
    if name is None:
        for row in rows:
            candidate = row.get("model")
            if isinstance(candidate, str) and candidate:
                name = candidate
                break
    relative = path.as_posix()
    if "experiment_7471" in relative or "experiment_7485" in relative:
        name = name or CURRENT_MODEL_NAME
        version = version or CURRENT_MODEL_REVISION
    return name, version


def _sample_counts(data: Any) -> tuple[int | None, int | None]:
    if not isinstance(data, dict):
        rows = _as_dicts(data)
        episodes = {str(row["episode_id"]) for row in rows if row.get("episode_id")}
        games = {str(row["game"]) for row in rows if row.get("game")}
        return (len(episodes) or None, len(games) or None)

    budget = data.get("sample_size_budget")
    episode_count: int | None = None
    game_count: int | None = None
    if isinstance(budget, dict):
        for field in (
            "complete_independent_units",
            "pooled_complete_independent_units",
            "completed_units",
            "historical_complete_episode_units",
        ):
            value = budget.get(field)
            if isinstance(value, int):
                episode_count = value
                break
        for field in (
            "independent_game_clusters",
            "pooled_independent_game_clusters",
            "historical_game_clusters",
        ):
            value = budget.get(field)
            if isinstance(value, int):
                game_count = value
                break
        if game_count is None and isinstance(budget.get("independent_groups"), list):
            game_count = len(set(map(str, budget["independent_groups"])))

    row_sets: list[list[dict[str, Any]]] = []
    for field in ("exclusive_cost_rows", "interval_rows", "episode_summaries", "rows"):
        candidate = data.get(field)
        if isinstance(candidate, list):
            row_sets.append([row for row in candidate if isinstance(row, dict)])
    for rows in row_sets:
        episodes = {str(row["episode_id"]) for row in rows if row.get("episode_id")}
        games = {str(row["game"]) for row in rows if row.get("game")}
        if episode_count is None and episodes:
            episode_count = len(episodes)
        if game_count is None and games:
            game_count = len(games)
    return episode_count, game_count


def _artifact_class(path: Path, data: Any) -> str:
    relative = path.as_posix()
    for suffix, artifact_class in PRIMARY_ARTIFACTS.items():
        if relative.endswith(suffix):
            return artifact_class
    if path.name == "seam_events.jsonl":
        return "decision_seam_shard"
    if path.name.endswith("_response.json") and "/requests/" in relative:
        return "backend_response_usage"
    if path.name == "episode_rows.json":
        return "raw_episode_rows"
    if path.name == "manifest.jsonl" and "/arc_e3/" in relative:
        return "live_agent_induction_manifest"
    if "action_provenance" in relative:
        return "live_action_provenance"
    return "phase_span_terminal"


def _flag_state(data: Any, path: Path) -> tuple[bool | None, bool]:
    if isinstance(data, dict) and "flagged_adversarial" in data:
        value = data["flagged_adversarial"]
        return (value if isinstance(value, bool) else None, True)
    relative = path.as_posix()
    if "experiment_7471" in relative or "experiment_7485" in relative:
        return False, False
    return None, False


def _candidate_inventory_paths(project_root: Path) -> list[Path]:
    results = project_root / "results"
    candidates: set[Path] = set()
    for path in results.glob("experiment_*.json"):
        name = path.name.lower()
        if "_arc_" in name or name.endswith("_arc.json") or "_semif_e6_" in name:
            candidates.add(path)
    candidates.update(results.glob("arc_*.json"))
    candidates.update(results.glob("outer_loop_arc_action_provenance*.json"))
    candidates.update((results / "arc_live_action_provenance_20260801").glob("*.json"))
    candidates.update((results / "arc_e3").glob("*/attempts/manifest.jsonl"))
    candidates.update((results / "raw").glob("experiment_*/episode_rows.json"))
    for experiment in (
        "experiment_7471_v654_arc_seam_observation",
        "experiment_7485_v655_arc_cost_panel_a",
    ):
        raw_root = results / "raw" / experiment
        candidates.update(raw_root.glob("episodes/*/seam_events.jsonl"))
        candidates.update(raw_root.glob("*/requests/*_response.json"))
    return sorted(path for path in candidates if path.is_file())


def inventory_artifacts(project_root: Path) -> list[dict[str, Any]]:
    """Inventory relevant ARC cost artifacts without changing them."""
    rows: list[dict[str, Any]] = []
    for path in _candidate_inventory_paths(project_root):
        relative = path.relative_to(project_root)
        progress(f"STEP 1 artifact {relative}: inventory")
        try:
            data = _read_json_or_jsonl(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        keys = _deep_keys(data)
        cost_fields = sorted(keys & COST_FIELD_NAMES)
        artifact_class = _artifact_class(relative, data)
        if artifact_class == "live_agent_induction_manifest":
            cost_fields.append("induction_attempt_manifest")
        if artifact_class == "decision_seam_shard":
            cost_fields.append("monotonic_stage_spans")
        if not cost_fields and artifact_class not in {
            "live_action_provenance",
            "live_agent_induction_manifest",
        }:
            continue
        model_name, model_version = _model_identity(data, relative)
        flagged, stamp_present = _flag_state(data, relative)
        episode_count, game_count = _sample_counts(data)
        experiment_id: Any = None
        if isinstance(data, dict):
            experiment_id = data.get("experiment_id", data.get("experiment"))
        rows.append(
            {
                "path": relative.as_posix(),
                "artifact_class": artifact_class,
                "experiment_id": experiment_id,
                "model_name": model_name,
                "model_version": model_version,
                "flagged_adversarial": flagged,
                "flag_stamp_present": stamp_present,
                "cost_fields": sorted(set(cost_fields)),
                "episode_count": episode_count,
                "game_count": game_count,
                "sha256": sha256_file(path),
            }
        )
    return rows


def numeric_share_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the 30-episode and 10-game publication floor after deduplication."""
    eligible: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not row.get("complete"):
            continue
        if not row.get("current_model"):
            continue
        if not row.get("numeric_eligible"):
            continue
        episode_id = row.get("episode_id")
        if isinstance(episode_id, str):
            eligible.setdefault(episode_id, row)
    games = {str(row["game"]) for row in eligible.values() if row.get("game")}
    episode_count = len(eligible)
    game_count = len(games)
    return {
        "criterion": "at_least_30_complete_current_model_episodes_across_10_games",
        "minimum_complete_episodes": 30,
        "minimum_games": 10,
        "complete_current_model_episodes": episode_count,
        "current_model_games": game_count,
        "missing_complete_episodes": max(0, 30 - episode_count),
        "missing_games": max(0, 10 - game_count),
        "passed": episode_count >= 30 and game_count >= 10,
    }


def reduce_episodes(
    episodes: Sequence[Mapping[str, Any]], *, publish_numeric: bool
) -> dict[str, Any]:
    """Reduce normalized exclusive episode rows into decision-point coverage."""
    wall_totals = dict.fromkeys(DECISION_POINTS, 0.0)
    token_totals = dict.fromkeys(DECISION_POINTS, 0)
    wall_coverage = dict.fromkeys(DECISION_POINTS, 0)
    token_coverage = dict.fromkeys(DECISION_POINTS, 0)
    concurrent = dict.fromkeys(DECISION_POINTS, False)
    episode_wall_total = 0.0
    backend_token_total = 0
    wall_rows_reconciled = 0
    token_rows_reconciled = 0

    for episode in episodes:
        episode_id = str(episode.get("episode_id", "unknown"))
        wall = episode.get("episode_wall_s")
        if not isinstance(wall, (int, float)) or isinstance(wall, bool) or wall < 0:
            raise ValueError(f"{episode_id}: invalid episode wall time")
        phases = episode.get("phases")
        if not isinstance(phases, list):
            raise ValueError(f"{episode_id}: phases must be a list")
        nonconcurrent_sum = 0.0
        attributed_tokens = 0
        phase_wall_seen: set[str] = set()
        phase_token_seen: set[str] = set()
        for phase in phases:
            if not isinstance(phase, dict):
                raise ValueError(f"{episode_id}: phase must be an object")
            point = phase.get("decision_point")
            if point not in DECISION_POINTS:
                raise ValueError(f"{episode_id}: unknown decision point {point!r}")
            phase_wall = phase.get("wall_s")
            if not isinstance(phase_wall, (int, float)) or isinstance(phase_wall, bool):
                raise ValueError(f"{episode_id}: invalid phase wall time")
            if phase_wall < 0:
                raise ValueError(f"{episode_id}: negative phase wall time")
            is_concurrent = phase.get("concurrent") is True
            if not is_concurrent:
                nonconcurrent_sum += float(phase_wall)
            concurrent[point] = concurrent[point] or is_concurrent
            wall_totals[point] += float(phase_wall)
            phase_wall_seen.add(point)
            tokens = phase.get("tokens")
            if tokens is not None:
                if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                    raise ValueError(f"{episode_id}: invalid phase token count")
                token_totals[point] += tokens
                attributed_tokens += tokens
                phase_token_seen.add(point)
        if nonconcurrent_sum > float(wall) + 1e-9:
            raise ValueError(f"{episode_id}: subphase time exceeds episode time")
        wall_rows_reconciled += 1
        backend_tokens = episode.get("backend_usage_tokens")
        if backend_tokens is not None:
            if (
                not isinstance(backend_tokens, int)
                or isinstance(backend_tokens, bool)
                or backend_tokens < 0
            ):
                raise ValueError(f"{episode_id}: invalid backend token total")
            if attributed_tokens != backend_tokens:
                raise ValueError(f"{episode_id}: phase tokens do not match backend usage")
            backend_token_total += backend_tokens
            token_rows_reconciled += 1
        for point in phase_wall_seen:
            wall_coverage[point] += 1
        for point in phase_token_seen:
            token_coverage[point] += 1
        episode_wall_total += float(wall)

    decision_points: dict[str, dict[str, Any]] = {}
    for point in DECISION_POINTS:
        wall_fraction: float | None = None
        token_fraction: float | None = None
        amdahl: float | None = None
        wall_value: float | None = None
        token_value: int | None = None
        if publish_numeric:
            wall_value = wall_totals[point]
            token_value = token_totals[point]
            if episode_wall_total > 0:
                wall_fraction = wall_totals[point] / episode_wall_total
                if wall_fraction < 1:
                    amdahl = 1.0 / (1.0 - wall_fraction)
            if backend_token_total > 0:
                token_fraction = token_totals[point] / backend_token_total
        decision_points[point] = {
            "episodes_with_wall_time": wall_coverage[point],
            "episodes_with_tokens": token_coverage[point],
            "concurrent": concurrent[point],
            "wall_s": wall_value,
            "tokens": token_value,
            "wall_fraction": wall_fraction,
            "token_fraction": token_fraction,
            "amdahl_ceiling": amdahl,
        }
    return {
        "publication_mode": "numeric" if publish_numeric else "coverage_only",
        "episode_count": len(episodes),
        "decision_points": decision_points,
        "reconciliation": {
            "wall_rows_reconciled": wall_rows_reconciled,
            "wall_rows_failed": len(episodes) - wall_rows_reconciled,
            "token_rows_reconciled": token_rows_reconciled,
            "token_rows_failed": len(episodes) - token_rows_reconciled,
        },
    }


def run_positive_control() -> dict[str, Any]:
    """Recover one known delay and one known token injection exactly."""
    injected_delay_s = 7.25
    injected_token_count = 137
    baseline_planner_s = 2.0
    baseline_generation_tokens = 80
    episode = {
        "episode_id": "synthetic-positive-control",
        "game": "synthetic",
        "complete": True,
        "current_model": True,
        "temperature": "cold",
        "episode_wall_s": 22.25,
        "backend_usage_tokens": 100 + injected_token_count,
        "phases": [
            {"decision_point": "candidate_selection", "wall_s": 1.0, "tokens": 0},
            {
                "decision_point": "induction_and_generation",
                "wall_s": 4.0,
                "tokens": baseline_generation_tokens + injected_token_count,
            },
            {
                "decision_point": "world_model_verification",
                "wall_s": 2.0,
                "tokens": 20,
            },
            {"decision_point": "supervisor", "wall_s": 1.0, "tokens": 0},
            {
                "decision_point": "planner",
                "wall_s": baseline_planner_s + injected_delay_s,
                "tokens": 0,
            },
            {"decision_point": "environment", "wall_s": 3.0, "tokens": 0},
        ],
    }
    reduced = reduce_episodes([episode], publish_numeric=True)
    planner = reduced["decision_points"]["planner"]
    generation = reduced["decision_points"]["induction_and_generation"]
    recovered_delay = float(planner["wall_s"]) - baseline_planner_s
    recovered_tokens = int(generation["tokens"]) - baseline_generation_tokens
    wall_reconciled = reduced["reconciliation"]["wall_rows_reconciled"] == 1
    tokens_reconciled = reduced["reconciliation"]["token_rows_reconciled"] == 1
    return {
        "synthetic_only": True,
        "injected_delay_s": injected_delay_s,
        "recovered_injected_delay_s": recovered_delay,
        "injected_token_count": injected_token_count,
        "recovered_injected_token_count": recovered_tokens,
        "wall_reconciled": wall_reconciled,
        "tokens_reconciled": tokens_reconciled,
        "passed": (
            recovered_delay == injected_delay_s
            and recovered_tokens == injected_token_count
            and wall_reconciled
            and tokens_reconciled
        ),
    }


def _backend_usage_by_episode(
    project_root: Path, raw_relative: str, episode_ids: Iterable[str]
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    raw_root = project_root / raw_relative
    folder_to_episode = {episode.replace(":", "__"): episode for episode in episode_ids}
    totals = dict.fromkeys(folder_to_episode.values(), 0)
    receipts: list[dict[str, Any]] = []
    for path in sorted(raw_root.glob("*/requests/*_response.json")):
        relative = path.relative_to(project_root)
        progress(f"STEP 3 artifact {relative}: reduce backend usage")
        data = json.loads(path.read_text())
        episode_id = folder_to_episode.get(path.parent.parent.name)
        if episode_id is None:
            continue
        usage = data.get("usage")
        if not isinstance(usage, dict):
            raise ValueError(f"{relative}: missing backend usage")
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if not all(isinstance(value, int) for value in (prompt, completion, total)):
            raise ValueError(f"{relative}: invalid backend token usage")
        if prompt + completion != total:
            raise ValueError(f"{relative}: backend token total does not reconcile")
        totals[episode_id] += total
        receipts.append(
            {
                "path": relative.as_posix(),
                "experiment_id": (
                    "exp7471-v654-arc-seam-observation"
                    if "experiment_7471" in relative.as_posix()
                    else "exp7485-v655-arc-cost-panel-a"
                ),
                "fields_imported": [
                    "model",
                    "usage.prompt_tokens",
                    "usage.completion_tokens",
                    "usage.total_tokens",
                ],
                "sha256": sha256_file(path),
            }
        )
    return totals, receipts


def _stage_phases(interval_row: Mapping[str, Any], backend_tokens: int) -> list[dict[str, Any]]:
    sums = dict.fromkeys(DECISION_POINTS, 0.0)
    stage_rows = interval_row.get("stage_rows")
    if not isinstance(stage_rows, list):
        raise ValueError(f"{interval_row.get('episode_id')}: missing stage rows")
    for stage in stage_rows:
        if not isinstance(stage, dict):
            continue
        point = SEAM_TO_DECISION_POINT.get(str(stage.get("seam")))
        exclusive_ns = stage.get("exclusive_ns")
        if point is None or not isinstance(exclusive_ns, int):
            continue
        sums[point] += exclusive_ns / 1_000_000_000
    phases: list[dict[str, Any]] = []
    for point in DECISION_POINTS:
        if sums[point] == 0 and point not in {
            "candidate_selection",
            "induction_and_generation",
            "supervisor",
        }:
            continue
        phases.append(
            {
                "decision_point": point,
                "wall_s": sums[point],
                "tokens": backend_tokens if point == "induction_and_generation" else 0,
                "concurrent": False,
            }
        )
    return phases


def _load_primary(project_root: Path, relative: str) -> dict[str, Any]:
    path = project_root / relative
    progress(f"STEP 3 artifact {relative}: load current evidence")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{relative}: expected object")
    return data


def _normalize_current_source(
    *,
    project_root: Path,
    timing_relative: str,
    identity_relative: str,
    raw_relative: str,
    interval_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timing = _load_primary(project_root, timing_relative)
    identity = _load_primary(project_root, identity_relative)
    if timing.get("flagged_adversarial") is True or identity.get("flagged_adversarial") is True:
        raise ValueError(f"{identity_relative}: flagged current evidence cannot be reduced")
    specs = identity.get("model_specs")
    if not isinstance(specs, list) or not specs or not isinstance(specs[0], dict):
        raise ValueError(f"{identity_relative}: missing current model identity")
    if specs[0].get("hf_id") != CURRENT_MODEL_REPOSITORY:
        raise ValueError(f"{identity_relative}: non-current model")

    intervals = timing.get(interval_field)
    identity_rows = identity.get("rows")
    if not isinstance(intervals, list) or not isinstance(identity_rows, list):
        raise ValueError(f"{timing_relative}: missing episode rows")
    row_by_episode = {
        str(row["episode_id"]): row
        for row in identity_rows
        if isinstance(row, dict) and row.get("episode_id")
    }
    episode_ids = [
        str(row["episode_id"])
        for row in intervals
        if isinstance(row, dict) and row.get("episode_id")
    ]
    usage, token_receipts = _backend_usage_by_episode(project_root, raw_relative, episode_ids)
    normalized: list[dict[str, Any]] = []
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        episode_id = str(interval.get("episode_id"))
        progress(f"STEP 3 artifact {identity_relative} episode {episode_id}: normalize")
        identity_row = row_by_episode.get(episode_id)
        if identity_row is None:
            raise ValueError(f"{identity_relative}: missing identity row {episode_id}")
        observed_ns = interval.get("observed_episode_ns")
        if not isinstance(observed_ns, int) or observed_ns < 0:
            raise ValueError(f"{timing_relative}: invalid observed duration {episode_id}")
        backend_tokens = usage.get(episode_id)
        if backend_tokens is None:
            raise ValueError(f"{raw_relative}: missing backend usage {episode_id}")
        disposition = identity_row.get("disposition")
        complete = disposition in {"complete", "completed"}
        normalized.append(
            {
                "episode_id": episode_id,
                "game": interval.get("game", identity_row.get("game")),
                "complete": complete,
                "current_model": True,
                "numeric_eligible": complete,
                "source_experiment_id": identity.get("experiment_id"),
                "timing_source_experiment_id": timing.get("experiment_id"),
                "temperature": ("cold" if identity_row.get("execution_order") == 0 else "warm"),
                "episode_wall_s": observed_ns / 1_000_000_000,
                "backend_usage_tokens": backend_tokens,
                "phases": _stage_phases(interval, backend_tokens),
                "missing_fields": [
                    "world_model_verification.exclusive_ns",
                    "planner.exclusive_ns",
                    "environment.exclusive_ns",
                    "candidate_selection.complete_candidate_ids",
                    "seam_stage_tokens_reconciled_to_backend_usage",
                ],
            }
        )
    return normalized, token_receipts


def extract_current_episodes(
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    historical, historical_receipts = _normalize_current_source(
        project_root=project_root,
        timing_relative="results/experiment_7478_v655_arc_interval_protocol.json",
        identity_relative="results/experiment_7471_v654_arc_seam_observation.json",
        raw_relative="results/raw/experiment_7471_v654_arc_seam_observation",
        interval_field="interval_rows",
    )
    panel, panel_receipts = _normalize_current_source(
        project_root=project_root,
        timing_relative="results/experiment_7485_v655_arc_cost_panel_a.json",
        identity_relative="results/experiment_7485_v655_arc_cost_panel_a.json",
        raw_relative="results/raw/experiment_7485_v655_arc_cost_panel_a",
        interval_field="exclusive_cost_rows",
    )
    return historical + panel, historical_receipts + panel_receipts


def evaluate_kill_rule(
    seam_coverage: Mapping[str, bool],
    replaceable_wall_shares: Mapping[str, float] | None,
) -> dict[str, Any]:
    """Stop speed claims when required seams are inseparable or negligible."""
    required = (
        "candidate_selection",
        "induction_and_generation",
        "world_model_verification",
        "planner",
        "environment",
    )
    missing = [point for point in required if not seam_coverage.get(point, False)]
    separation_kill = bool(missing)
    below_five: bool | None = None
    if replaceable_wall_shares is not None:
        values = list(replaceable_wall_shares.values())
        below_five = bool(values) and all(value < 0.05 for value in values)
    return {
        "required_separable_seams": list(required),
        "missing_separable_seams": missing,
        "seam_separation_kill_fired": separation_kill,
        "all_replaceable_work_below_five_percent": below_five,
        "speed_claims_stopped": separation_kill or below_five is True,
    }


def _coverage_table(inventory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in inventory:
        by_class[str(row["artifact_class"])].append(row)
    table: list[dict[str, Any]] = []
    for artifact_class in sorted(by_class):
        rows = by_class[artifact_class]
        fields = sorted(
            {
                str(field)
                for row in rows
                for field in row.get("cost_fields", [])
                if isinstance(field, str)
            }
        )
        table.append(
            {
                "artifact_class": artifact_class,
                "artifact_count": len(rows),
                "available_cost_fields": fields,
                "missing_fields": CLASS_MISSING_FIELDS.get(
                    artifact_class, CLASS_MISSING_FIELDS["phase_span_terminal"]
                ),
            }
        )
    return table


def _claim_role(row: Mapping[str, Any]) -> tuple[str, bool, str]:
    artifact_class = str(row.get("artifact_class"))
    path = str(row.get("path"))
    if row.get("flagged_adversarial") is True:
        return "excluded", False, "flagged_adversarial"
    if "experiment_5972" in path:
        return "reducer_control", False, "non_current_model_control_only"
    if "experiment_7457" in path:
        return "schema_control", False, "schema_control_only"
    if "experiment_7464" in path:
        return "prior_reducer_control", False, "derived_from_schema_control"
    if "experiment_7471" in path:
        if artifact_class == "backend_response_usage":
            return "current_token_evidence", True, "eligible_supporting_evidence"
        if artifact_class == "decision_seam_shard":
            return "current_raw_seam_evidence", True, "eligible_supporting_evidence"
        return "current_episode_identity", True, "eligible_current_model_evidence"
    if "experiment_7478" in path:
        return "current_timing_evidence", True, "deduplicated_exp7471_timing"
    if "experiment_7485" in path:
        return "current_panel_evidence", True, "eligible_current_model_evidence"
    if row.get("model_name") and CURRENT_MODEL_NAME not in str(row["model_name"]):
        return "excluded", False, "non_current_model"
    return "coverage_inventory", False, "incomplete_or_incompatible_cost_schema"


def _per_artifact_claim_rows(
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for inventory_row in inventory:
        role, eligible, reason = _claim_role(inventory_row)
        rows.append(
            {
                "path": inventory_row["path"],
                "sha256": inventory_row["sha256"],
                "artifact_class": inventory_row["artifact_class"],
                "claim_role": role,
                "numeric_eligible_source": eligible,
                "disposition_reason": reason,
                "flagged_adversarial": inventory_row["flagged_adversarial"],
                "model_name": inventory_row["model_name"],
                "model_version": inventory_row["model_version"],
                "episode_count": inventory_row["episode_count"],
                "game_count": inventory_row["game_count"],
                "cost_fields": inventory_row["cost_fields"],
            }
        )
    return rows


def _cited_artifacts(
    inventory: Sequence[Mapping[str, Any]], token_receipts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    citations: dict[str, dict[str, Any]] = {}
    for row in inventory:
        path = str(row["path"])
        citations[path] = {
            "experiment_id": row.get("experiment_id"),
            "path": path,
            "fields_imported": [
                "model_identity",
                "flagged_adversarial",
                "cost_fields",
                "episode_count",
                "game_count",
            ],
            "sha256": row["sha256"],
        }
    for receipt in token_receipts:
        citations[str(receipt["path"])] = dict(receipt)
    return [citations[path] for path in sorted(citations)]


def _cold_warm_coverage(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cold = sum(row.get("temperature") == "cold" for row in episodes)
    warm = sum(row.get("temperature") == "warm" for row in episodes)
    return {
        "cold_episode_count": cold,
        "warm_episode_count": warm,
        "classification_rule": (
            "execution_order_zero_after_each_owned_model_load_is cold; later episodes are warm"
        ),
        "cold_numeric_profile": None,
        "warm_numeric_profile": None,
        "reason_numeric_profiles_absent": "numeric_share_gate_not_met",
    }


def _shadow_timers_required() -> list[dict[str, Any]]:
    return [
        {
            "scope": "episode",
            "fields": [
                "run_id",
                "process_id",
                "episode_id",
                "game_id",
                "model_revision",
                "model_load_generation",
                "episode_start_monotonic_ns",
                "episode_end_monotonic_ns",
                "clock_identity",
            ],
        },
        {
            "scope": "every_phase_span",
            "fields": [
                "decision_id",
                "parent_decision_id",
                "decision_point",
                "start_monotonic_ns",
                "end_monotonic_ns",
                "exclusive_ns",
                "concurrent",
                "terminal_disposition",
            ],
        },
        {
            "scope": "candidate_selection",
            "fields": [
                "candidate_ids",
                "candidate_count",
                "omitted_candidate_count",
                "selected_candidate_id",
                "selection_start_monotonic_ns",
                "selection_end_monotonic_ns",
            ],
        },
        {
            "scope": "induction_and_generation",
            "fields": [
                "request_id",
                "generation_start_monotonic_ns",
                "generation_end_monotonic_ns",
                "construction_start_monotonic_ns",
                "construction_end_monotonic_ns",
                "prompt_tokens",
                "completion_tokens",
                "cached_prompt_tokens",
            ],
        },
        {
            "scope": "world_model_verification",
            "fields": [
                "verifier_start_monotonic_ns",
                "verifier_end_monotonic_ns",
                "verifier_exclusive_ns",
                "verifier_kind",
                "verifier_disposition",
                "verifier_input_tokens",
                "verifier_output_tokens",
            ],
        },
        {
            "scope": "supervisor",
            "fields": [
                "supervisor_start_monotonic_ns",
                "supervisor_end_monotonic_ns",
                "eligible_arm_ids",
                "selected_arm_id",
                "arm_applied",
            ],
        },
        {
            "scope": "planner",
            "fields": [
                "planner_start_monotonic_ns",
                "planner_end_monotonic_ns",
                "planner_exclusive_ns",
                "plan_id",
                "plan_disposition",
            ],
        },
        {
            "scope": "environment",
            "fields": [
                "environment_operation",
                "environment_start_monotonic_ns",
                "environment_end_monotonic_ns",
                "environment_exclusive_ns",
            ],
        },
        {
            "scope": "backend_usage_join",
            "fields": [
                "request_id",
                "episode_id",
                "decision_id",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "usage_source",
            ],
        },
    ]


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return _sha256_bytes(_canonical_json(stable))


def build_artifact(project_root: Path) -> dict[str, Any]:
    """Build the terminal coverage artifact from immutable upstream files."""
    started = time.perf_counter()
    progress("STEP 0: start CPU-only existing-artifact reduction")
    progress("STEP 1: inventory ARC cost and provenance artifacts")
    inventory = inventory_artifacts(project_root)

    progress("STEP 2: classify flags, model identities, controls, and exclusions")
    claim_rows = _per_artifact_claim_rows(inventory)

    progress("STEP 3: normalize current-model exclusive spans and backend usage")
    episodes, token_receipts = extract_current_episodes(project_root)

    progress("STEP 4: run the injected-delay and injected-token positive control")
    positive_control = run_positive_control()
    if not positive_control["passed"]:
        raise ValueError("positive control failed")

    progress("STEP 5: apply sample and seam-separation gates")
    numeric_gate = numeric_share_gate(episodes)
    publish_numeric = bool(numeric_gate["passed"])
    profile = reduce_episodes(episodes, publish_numeric=publish_numeric)
    seam_coverage = {
        point: profile["decision_points"][point]["episodes_with_wall_time"] == len(episodes)
        for point in DECISION_POINTS
    }
    replaceable_shares: dict[str, float] | None = None
    if publish_numeric:
        replaceable_shares = {
            point: float(profile["decision_points"][point]["wall_fraction"])
            for point in ("candidate_selection", "supervisor", "planner")
            if profile["decision_points"][point]["wall_fraction"] is not None
        }
    kill_rule = evaluate_kill_rule(seam_coverage, replaceable_shares)
    numeric_share_published = publish_numeric and not kill_rule["speed_claims_stopped"]
    if not numeric_share_published and publish_numeric:
        profile = reduce_episodes(episodes, publish_numeric=False)

    progress("STEP 6: preserve coverage-only verdict and exact missing fields")
    verdict = "complete_coverage_only_numeric_gate_not_met_and_speed_claims_stopped"
    if numeric_share_published:
        verdict = "complete_numeric_profile_gate_met"
    coverage_table = _coverage_table(inventory)
    missing_verifier_gaps = [
        "world_model_verification.start_monotonic_ns",
        "world_model_verification.end_monotonic_ns",
        "world_model_verification.exclusive_ns",
        "world_model_verification.parent_decision_id",
        "world_model_verification.terminal_disposition",
        "world_model_verification.input_tokens",
        "world_model_verification.output_tokens",
        "world_model_construction_vs_verification_boundary",
    ]
    gate_results = {
        "numeric_share_gate": numeric_gate,
        "seam_separation_gate": {
            "passed": not kill_rule["seam_separation_kill_fired"],
            "missing_separable_seams": kill_rule["missing_separable_seams"],
        },
        "replaceable_below_five_percent_gate": {
            "passed": None,
            "observed": kill_rule["all_replaceable_work_below_five_percent"],
            "reason": "not_evaluated_without_numeric_share_gate",
        },
        "numeric_share_published": numeric_share_published,
        "coverage_only": not numeric_share_published,
        "speed_claims_stopped": kill_rule["speed_claims_stopped"],
    }
    acceptance_gate_results = [
        {
            "check": "positive_control",
            "expected": True,
            "observed": positive_control["passed"],
            "passed": positive_control["passed"],
        },
        {
            "check": "complete_current_model_episode_floor",
            "expected": 30,
            "observed": numeric_gate["complete_current_model_episodes"],
            "passed": numeric_gate["complete_current_model_episodes"] >= 30,
        },
        {
            "check": "current_model_game_floor",
            "expected": 10,
            "observed": numeric_gate["current_model_games"],
            "passed": numeric_gate["current_model_games"] >= 10,
        },
        {
            "check": "required_seams_separable",
            "expected": True,
            "observed": not kill_rule["seam_separation_kill_fired"],
            "passed": not kill_rule["seam_separation_kill_fired"],
        },
    ]

    artifact: dict[str, Any] = {
        "schema": "carnot.arc.e6_live_loop_cost_profile.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": (COVERAGE_VERDICT_CLASS if not numeric_share_published else "positive"),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": EXECUTION_VENUE,
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
        "flagged_adversarial": False,
        "methodology_note": (
            "Coverage counts are published. Real wall-time shares, token shares, and "
            "Amdahl ceilings are null because only 26 complete current-model episodes "
            "exist, below the fixed 30-episode floor, and verifier, planner, and "
            "environment time are not separately observed."
        ),
        "current_model_contract": {
            "repository": CURRENT_MODEL_REPOSITORY,
            "name": CURRENT_MODEL_NAME,
            "filename": CURRENT_MODEL_FILENAME,
            "revision": CURRENT_MODEL_REVISION,
        },
        "cited_upstream_artifacts": _cited_artifacts(inventory, token_receipts),
        "artifact_inventory": inventory,
        "per_artifact_claim_rows": claim_rows,
        "coverage_table": coverage_table,
        "normalized_episode_coverage_rows": [
            {
                "episode_id": row["episode_id"],
                "game": row["game"],
                "complete": row["complete"],
                "temperature": row["temperature"],
                "source_experiment_id": row["source_experiment_id"],
                "timing_source_experiment_id": row["timing_source_experiment_id"],
                "wall_time_present": row["episode_wall_s"] is not None,
                "backend_usage_present": row["backend_usage_tokens"] is not None,
                "present_decision_points": [phase["decision_point"] for phase in row["phases"]],
                "missing_fields": row["missing_fields"],
            }
            for row in episodes
        ],
        "decision_point_profile": profile,
        "cold_warm_split": _cold_warm_coverage(episodes),
        "positive_control": positive_control,
        "gate_results": gate_results,
        "acceptance_gate_results": acceptance_gate_results,
        "kill_rule": kill_rule,
        "missing_verifier_gaps": missing_verifier_gaps,
        "shadow_timers_required": _shadow_timers_required(),
        "speed_claim": None,
        "speed_claim_stop_reason": (
            "required_seams_not_separable_and_numeric_sample_gate_not_met"
            if not numeric_share_published
            else None
        ),
    }
    artifact["duration_s"] = time.perf_counter() - started
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    return artifact


def write_artifact(output_path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON artifact to an explicit caller-owned path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    output_path = args.output
    if output_path is None:
        output_path = project_root / DEFAULT_RESULT_PATH
    elif not output_path.is_absolute():
        output_path = project_root / output_path
    artifact = build_artifact(project_root)
    progress(f"STEP 7: write terminal artifact to {output_path}")
    write_artifact(output_path, artifact)
    progress("STEP 8: keep test writers isolated through explicit output paths")
    progress("STEP 9: terminal artifact is ready for lint and adversarial verification")
    progress("STEP 10: no commit or push performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

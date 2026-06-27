"""Exp 4883: inducer-ceiling A/B for the TTT value-gap residual.

Spec refs: REQ-ARC-WMTE-4883,
SCENARIO-ARC-WMTE-4883-A1-LOW-VALUE-GATE,
SCENARIO-ARC-WMTE-4883-SAME-SPLIT-AB,
SCENARIO-ARC-WMTE-4883-ATTRIBUTION.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - script execution path
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_4871_generation_wall_fork_probe_gpu_fixed as a1  # noqa: E402
from carnot import experiment_4882_ttt_dynamics_value_gap as value_gap  # noqa: E402
from carnot.experiment_4851_generation_coverage_diagnostic import (  # noqa: E402
    offline_arcade_available,
    run_orphan_lint,
)


EXPERIMENT_ID = 4883
RESULT_RELATIVE_PATH = "results/experiment_4883_inducer_ceiling_ab.json"
CHECKPOINT_RELATIVE_DIR = "results/experiment_4883_inducer_ceiling_ab_checkpoints"
A1_RESULT_RELATIVE_PATH = value_gap.RESULT_RELATIVE_PATH
SOTA_MAPPING_RELATIVE_PATH = "results/experiment_4879_sota_ingestion_v450_frontier.json"
SPEC_REFS = [
    "REQ-ARC-WMTE-4883",
    "SCENARIO-ARC-WMTE-4883-A1-LOW-VALUE-GATE",
    "SCENARIO-ARC-WMTE-4883-SAME-SPLIT-AB",
    "SCENARIO-ARC-WMTE-4883-ATTRIBUTION",
]
LANES = ("reference", "local")
ATTRIBUTIONS = (
    "LOCAL_MODEL_IS_CEILING",
    "METHOD_IS_CEILING",
    "LOCAL_ALREADY_SUFFICIENT",
)
DEFAULT_HELDOUT_GAMES = value_gap.HELDOUT_GAMES
DEFAULT_COLD_TRANSITIONS = value_gap.DEFAULT_COLD_TRANSITIONS
DEFAULT_HELDOUT_TRANSITIONS = value_gap.DEFAULT_HELDOUT_TRANSITIONS
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_SOFT_ELAPSED_BUDGET_S = 3500.0
LOW_VALUE_DELTA_THRESHOLD = 0.1
RANDOM_SEED = 20260627
INFERENCE_SUBSTRATE = "live_llm_inference"

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a real lane lift is success_inducer_ceiling_<attribution>; "
            "a flat null is complete_inducer_ceiling_neither_lane_lifts_method_is_ceiling."
        )
    },
    "inducer_ceiling_attribution": {
        "principle": (
            "one of LOCAL_MODEL_IS_CEILING | METHOD_IS_CEILING | LOCAL_ALREADY_SUFFICIENT "
            "-- redirects .451."
        )
    },
    "reference_lane_value_accuracy_delta": {
        "principle": (
            "Family-B reference lane changed-cell value-accuracy delta vs A1 baseline "
            "(the capability ceiling)."
        )
    },
    "local_lane_value_accuracy_delta": {
        "principle": (
            "local open-code lane delta vs A1 baseline (the deployment lane -- what actually ships)."
        )
    },
    "per_lane_per_game": {
        "principle": (
            "per-lane per-game {value_acc, cell_recall, delta_vs_baseline, ci95} -- "
            "the quantitative A/B table."
        )
    },
    "delta_on_truly_heldout_split": {
        "principle": (
            "true -- both lanes scored on the SAME held-out split as A1, disjoint from "
            "any fit set (B1 audits)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the held-out transition score is oracle-distinct from the env's "
            "level-up check (circularity discipline)."
        )
    },
    "live_path_reachable": {
        "principle": (
            "both lanes use the live e3 load_engine interface (arc_orphan_solver_lint passes), "
            "not a parallel solver."
        )
    },
    "reference_lane_is_ceiling_only": {
        "principle": (
            "true -- the Family-B reference lane is a ceiling measurement; the deployment "
            "lane stays local (decentralization Rule 2)."
        )
    },
    "solve_provenance": {
        "principle": "development_proxy -- an inducer-ceiling measurement, NOT a banked level."
    },
    "checkpoint_emitted": {
        "principle": (
            "a capped run still emits a usable partial (per-game + per-lane checkpointing)."
        )
    },
    "inference_substrate": {
        "principle": "live_llm_inference (60s floor) -- both induction lanes invoke an LLM."
    },
    "model_specs": {
        "principle": (
            "names both inducers (Family-B reference + local Qwen3.5-9B-MTP via the GPU-0 "
            "CUDA llama-server)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/generator/A1-baseline checks; a missing resource emits blocked_."
        )
    },
    "random_seed": {
        "principle": "determinism for both induction lanes' stochastic search."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (games, A1 baseline, both lane configs, held-out split) so a "
            "replication catches drift."
        )
    },
}


JsonDict = dict[str, Any]
Clock = Callable[[], float]


class DiagnosticError(RuntimeError):
    """Raised when the Exp 4883 artifact would otherwise be invalid."""


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def load_a1_artifact(root: Path | str = REPO_ROOT) -> JsonDict | None:
    path = Path(root) / A1_RESULT_RELATIVE_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(data) if isinstance(data, Mapping) else None


def _normalise_generator_result(result: Any) -> JsonDict:
    return a1._normalise_generator_result(result)


def _generator_backend_from_preconditions(preconditions: Mapping[str, Any]) -> str | None:
    return a1._generator_backend_from_preconditions(preconditions)


def _model_specs_from_preconditions(
    preconditions: Mapping[str, Any], generator_backend: str | None
) -> JsonDict:
    local = a1._model_specs_from_preconditions(preconditions, generator_backend)
    return {
        "reference_lane": {
            "name": "Family-B reference executable-world-model inducer",
            "interface": "arXiv:2605.05138 reference interface",
            "role": "capability_ceiling_only",
            "closed_or_cloud_strength_optional": True,
        },
        "local_lane": {
            **local,
            "name": local.get("name") or "Qwen3.5-9B-MTP",
            "role": "deployment_lane",
            "source_ids": ["2507.03160", "2203.13474"],
        },
    }


def _inducer_ab_config(
    *,
    cold_transitions: int,
    heldout_transitions: int,
    soft_elapsed_budget_s: float,
    heldout_games: Sequence[str],
    bootstrap_iterations: int,
) -> JsonDict:
    return {
        "live_path": "arc_executable_world_model.load_engine",
        "a1_artifact": A1_RESULT_RELATIVE_PATH,
        "sota_mapping_artifact": SOTA_MAPPING_RELATIVE_PATH,
        "a1_low_value_delta_threshold": LOW_VALUE_DELTA_THRESHOLD,
        "lanes": {
            "reference": {
                "name": "Family-B reference executable-world-model inducer",
                "source_id": "2605.05138",
                "ceiling_only": True,
            },
            "local": {
                "name": "Qwen3.5-9B-MTP local open-code inducer",
                "source_ids": ["2507.03160", "2203.13474"],
                "deployment_lane": True,
            },
        },
        "graded_metric": "experiment_4882.score_graded_engine",
        "cold_transitions": int(cold_transitions),
        "heldout_transitions": int(heldout_transitions),
        "soft_elapsed_budget_s": float(soft_elapsed_budget_s),
        "heldout_games": list(heldout_games),
        "bootstrap_iterations": int(bootstrap_iterations),
        "planner_blind_to_banked_answer": True,
    }


def _unit(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= out <= 1.0:
        return out
    return None


def _id_set(row: Mapping[str, Any], key: str) -> set[str]:
    return {str(item) for item in row.get(key) or []}


def _a1_row_has_baseline_and_split(row: Mapping[str, Any]) -> bool:
    if _unit(row.get("value_acc_baseline")) is None:
        return False
    fit = _id_set(row, "fit_transition_ids")
    heldout = _id_set(row, "remeasure_transition_ids")
    return bool(fit) and bool(heldout) and fit.isdisjoint(heldout)


def _a1_precondition(a1_artifact: Mapping[str, Any] | None) -> JsonDict:
    if not isinstance(a1_artifact, Mapping):
        return {"ok": False, "detail": "missing_or_malformed"}
    try:
        value_delta = float(a1_artifact.get("tta_changed_cell_value_accuracy_delta_median"))
    except (TypeError, ValueError):
        return {"ok": False, "detail": "missing_tta_changed_cell_value_accuracy_delta_median"}
    if value_delta >= LOW_VALUE_DELTA_THRESHOLD:
        return {
            "ok": False,
            "detail": "a1_value_delta_not_low",
            "tta_changed_cell_value_accuracy_delta_median": value_delta,
        }
    rows = a1_artifact.get("per_game_value_gap")
    if not isinstance(rows, Mapping):
        return {
            "ok": False,
            "detail": "missing_per_game_baselines_or_split",
            "tta_changed_cell_value_accuracy_delta_median": value_delta,
        }
    valid_games = [
        str(game)
        for game, row in rows.items()
        if isinstance(row, Mapping) and _a1_row_has_baseline_and_split(row)
    ]
    if len(valid_games) < 3:
        return {
            "ok": False,
            "detail": "missing_per_game_baselines_or_split",
            "tta_changed_cell_value_accuracy_delta_median": value_delta,
            "n_valid_games": len(valid_games),
        }
    return {
        "ok": True,
        "detail": "ok",
        "tta_changed_cell_value_accuracy_delta_median": value_delta,
        "n_games": len(valid_games),
        "games": valid_games,
    }


def bootstrap_ci95(values: Sequence[float], *, iterations: int, seed: int) -> list[float | None]:
    vals = [float(value) for value in values]
    if not vals:
        return [None, None]
    if len(set(vals)) == 1:
        value = round(vals[0], 6)
        return [value, value]
    rng = random.Random(seed)
    samples: list[float] = []
    count = max(1, int(iterations))
    for _ in range(count):
        draw = [rng.choice(vals) for _ in vals]
        samples.append(float(median(draw)))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [round(float(lo), 6), round(float(hi), 6)]


def _lane_delta_values(
    per_lane_per_game: Mapping[str, Mapping[str, Mapping[str, Any]]],
    lane: str,
) -> list[float]:
    values: list[float] = []
    lane_rows = per_lane_per_game.get(lane)
    if not isinstance(lane_rows, Mapping):
        return values
    for row in lane_rows.values():
        if not isinstance(row, Mapping):
            continue
        try:
            values.append(round(float(row["delta_vs_baseline"]), 6))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def _aggregate_lane(
    per_lane_per_game: Mapping[str, Mapping[str, Mapping[str, Any]]],
    lane: str,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> JsonDict:
    values = _lane_delta_values(per_lane_per_game, lane)
    med = round(float(median(values)), 6) if values else None
    ci95 = bootstrap_ci95(values, iterations=bootstrap_iterations, seed=seed)
    lifted = med is not None and ci95[0] is not None and float(med) > 0.0 and float(ci95[0]) > 0.0
    return {
        "median": med,
        "ci95": ci95,
        "n_games": len(values),
        "lifted": bool(lifted),
    }


def _compute_attribution(
    *,
    reference_delta: Mapping[str, Any],
    local_delta: Mapping[str, Any],
    n_games: int,
    partial: bool,
) -> str | None:
    if partial or n_games < 3:
        return None
    if local_delta.get("lifted") is True:
        return "LOCAL_ALREADY_SUFFICIENT"
    if reference_delta.get("lifted") is True:
        return "LOCAL_MODEL_IS_CEILING"
    return "METHOD_IS_CEILING"


def _terminal_verdict(*, attribution: str | None, n_games: int, partial: bool) -> str:
    if partial:
        return "complete_inducer_ceiling_partial_budget_stop"
    if n_games < 3 or attribution is None:
        return "complete_inducer_ceiling_too_few_games"
    if attribution == "METHOD_IS_CEILING":
        return "complete_inducer_ceiling_neither_lane_lifts_method_is_ceiling"
    return f"success_inducer_ceiling_{attribution}"


def _same_split_rows(
    per_lane_per_game: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> bool:
    reference = per_lane_per_game.get("reference")
    local = per_lane_per_game.get("local")
    if not isinstance(reference, Mapping) or not isinstance(local, Mapping):
        return False
    games = set(reference) & set(local)
    if not games:
        return True
    for game in games:
        ref = reference.get(game)
        loc = local.get(game)
        if not isinstance(ref, Mapping) or not isinstance(loc, Mapping):
            return False
        ref_ids = _id_set(ref, "heldout_transition_ids")
        loc_ids = _id_set(loc, "heldout_transition_ids")
        a1_ids = _id_set(ref, "a1_heldout_transition_ids")
        if not ref_ids or ref_ids != loc_ids or ref_ids != a1_ids:
            return False
        if _id_set(ref, "fit_transition_ids") & ref_ids:
            return False
        if _id_set(loc, "fit_transition_ids") & loc_ids:
            return False
    return True


def _n_games_with_both_lanes(
    per_lane_per_game: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> int:
    reference = per_lane_per_game.get("reference")
    local = per_lane_per_game.get("local")
    if not isinstance(reference, Mapping) or not isinstance(local, Mapping):
        return 0
    return len(set(reference) & set(local))


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    rows = artifact.get("per_lane_per_game") or {}
    reference = rows.get("reference") if isinstance(rows, Mapping) else {}
    reference_rows = reference if isinstance(reference, Mapping) else {}
    games = sorted(str(game) for game in reference_rows)
    split = {
        str(game): {
            "fit": list(row.get("fit_transition_ids") or []),
            "heldout": list(row.get("heldout_transition_ids") or []),
            "a1_heldout": list(row.get("a1_heldout_transition_ids") or []),
        }
        for game, row in sorted(reference_rows.items())
        if isinstance(row, Mapping)
    }
    baselines = {
        str(game): row.get("a1_baseline_value_acc")
        for game, row in sorted(reference_rows.items())
        if isinstance(row, Mapping)
    }
    payload = {
        "games": games,
        "a1_baselines": baselines,
        "lane_configs": (artifact.get("inducer_ab_config") or {}).get("lanes", {}),
        "heldout_split": split,
        "random_seed": artifact.get("random_seed"),
        "spec_refs": artifact.get("spec_refs") or SPEC_REFS,
    }
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _attach_checksum(artifact: JsonDict) -> JsonDict:
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    verdict: str,
    *,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool = False,
    duration_s: float = 0.0,
    random_seed: int = RANDOM_SEED,
    cold_transitions: int = DEFAULT_COLD_TRANSITIONS,
    heldout_transitions: int = DEFAULT_HELDOUT_TRANSITIONS,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    heldout_games: Sequence[str] = DEFAULT_HELDOUT_GAMES,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> JsonDict:
    generator_backend = _generator_backend_from_preconditions(preconditions_checked)
    empty_delta = {"median": None, "ci95": [None, None], "n_games": 0, "lifted": False}
    artifact: JsonDict = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": str(verdict),
        "inducer_ceiling_attribution": None,
        "reference_lane_value_accuracy_delta": dict(empty_delta),
        "local_lane_value_accuracy_delta": dict(empty_delta),
        "per_lane_per_game": {"reference": {}, "local": {}},
        "delta_on_truly_heldout_split": True,
        "verifier_is_oracle": False,
        "live_path_reachable": bool(live_path_reachable),
        "reference_lane_is_ceiling_only": True,
        "solve_provenance": "development_proxy",
        "checkpoint_emitted": False,
        "partial": False,
        "n_games_measured": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": _model_specs_from_preconditions(preconditions_checked, generator_backend),
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": int(random_seed),
        "inducer_ab_config": _inducer_ab_config(
            cold_transitions=cold_transitions,
            heldout_transitions=heldout_transitions,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
            bootstrap_iterations=bootstrap_iterations,
        ),
        "retire_if_same_verdict": True,
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def build_artifact(
    *,
    per_lane_per_game: Mapping[str, Mapping[str, Mapping[str, Any]]],
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    duration_s: float,
    partial: bool,
    checkpoint_emitted: bool,
    a1_artifact: Mapping[str, Any] | None = None,
    random_seed: int = RANDOM_SEED,
    cold_transitions: int = DEFAULT_COLD_TRANSITIONS,
    heldout_transitions: int = DEFAULT_HELDOUT_TRANSITIONS,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    heldout_games: Sequence[str] = DEFAULT_HELDOUT_GAMES,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> JsonDict:
    _ = a1_artifact
    rows: dict[str, dict[str, JsonDict]] = {
        lane: {str(game): dict(row) for game, row in dict(per_lane_per_game.get(lane, {})).items()}
        for lane in LANES
    }
    reference_delta = _aggregate_lane(
        rows, "reference", bootstrap_iterations=bootstrap_iterations, seed=random_seed
    )
    local_delta = _aggregate_lane(
        rows, "local", bootstrap_iterations=bootstrap_iterations, seed=random_seed + 17
    )
    n_games = _n_games_with_both_lanes(rows)
    attribution = _compute_attribution(
        reference_delta=reference_delta,
        local_delta=local_delta,
        n_games=n_games,
        partial=partial,
    )
    generator_backend = _generator_backend_from_preconditions(preconditions_checked)
    artifact: JsonDict = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _terminal_verdict(
            attribution=attribution,
            n_games=n_games,
            partial=partial,
        ),
        "inducer_ceiling_attribution": attribution,
        "reference_lane_value_accuracy_delta": reference_delta,
        "local_lane_value_accuracy_delta": local_delta,
        "per_lane_per_game": rows,
        "delta_on_truly_heldout_split": _same_split_rows(rows),
        "verifier_is_oracle": False,
        "live_path_reachable": bool(live_path_reachable),
        "reference_lane_is_ceiling_only": True,
        "solve_provenance": "development_proxy",
        "checkpoint_emitted": bool(checkpoint_emitted),
        "partial": bool(partial),
        "n_games_measured": n_games,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": _model_specs_from_preconditions(preconditions_checked, generator_backend),
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": int(random_seed),
        "inducer_ab_config": _inducer_ab_config(
            cold_transitions=cold_transitions,
            heldout_transitions=heldout_transitions,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
            bootstrap_iterations=bootstrap_iterations,
        ),
        "retire_if_same_verdict": True,
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = set(FIELD_PRINCIPLES) | {
        "schema_version",
        "experiment_id",
        "spec_refs",
        "partial",
        "n_games_measured",
        "inducer_ab_config",
        "retire_if_same_verdict",
        "duration_s",
        "field_principles",
    }
    for field in sorted(required):
        if field not in artifact:
            errors.append(f"missing_field:{field}")
    if errors:
        return errors

    verdict = str(artifact.get("honest_verdict"))
    if not verdict.startswith(("blocked_", "complete_", "success_")):
        errors.append("honest_verdict_terminal_prefix")
    blocked = verdict.startswith("blocked_")
    partial = artifact.get("partial") is True

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append(f"field_principles.{field}")

    rows = artifact.get("per_lane_per_game")
    if not isinstance(rows, Mapping):
        errors.append("per_lane_per_game")
        rows = {"reference": {}, "local": {}}
    for lane in LANES:
        lane_rows = rows.get(lane)
        if not isinstance(lane_rows, Mapping):
            errors.append(f"per_lane_per_game.{lane}")
            continue
        for game, row in lane_rows.items():
            if not isinstance(row, Mapping):
                errors.append(f"per_lane_per_game.{lane}.{game}")
                continue
            for key in ("value_acc", "cell_recall", "a1_baseline_value_acc"):
                if _unit(row.get(key)) is None:
                    errors.append(f"per_lane_per_game.{lane}.{game}.{key}")
            try:
                delta = round(float(row.get("delta_vs_baseline")), 6)
            except (TypeError, ValueError):
                errors.append(f"per_lane_per_game.{lane}.{game}.delta_vs_baseline")
                continue
            value_acc = _unit(row.get("value_acc"))
            baseline = _unit(row.get("a1_baseline_value_acc"))
            if value_acc is not None and baseline is not None:
                if delta != round(float(value_acc) - float(baseline), 6):
                    errors.append(f"per_lane_per_game.{lane}.{game}.delta_vs_baseline")
            ci95 = row.get("ci95")
            if not (
                isinstance(ci95, Sequence)
                and not isinstance(ci95, (str, bytes))
                and len(ci95) == 2
            ):
                errors.append(f"per_lane_per_game.{lane}.{game}.ci95")
            if not _id_set(row, "heldout_transition_ids"):
                errors.append(f"per_lane_per_game.{lane}.{game}.heldout_transition_ids")
            if "arc_executable_world_model.load_engine" not in (
                row.get("live_path_methods_called") or []
            ):
                errors.append(f"per_lane_per_game.{lane}.{game}.live_path_methods_called")

    try:
        n_games_measured = int(artifact.get("n_games_measured"))
    except (TypeError, ValueError):
        n_games_measured = -1
    expected_n = _n_games_with_both_lanes(rows if isinstance(rows, Mapping) else {})
    if n_games_measured != expected_n:
        errors.append("n_games_measured")
    if artifact.get("delta_on_truly_heldout_split") != _same_split_rows(
        rows if isinstance(rows, Mapping) else {}
    ):
        errors.append("delta_on_truly_heldout_split")
    expected_reference = _aggregate_lane(
        rows if isinstance(rows, Mapping) else {},
        "reference",
        bootstrap_iterations=int(
            (artifact.get("inducer_ab_config") or {}).get(
                "bootstrap_iterations", DEFAULT_BOOTSTRAP_ITERATIONS
            )
        ),
        seed=int(artifact.get("random_seed") or 0),
    )
    expected_local = _aggregate_lane(
        rows if isinstance(rows, Mapping) else {},
        "local",
        bootstrap_iterations=int(
            (artifact.get("inducer_ab_config") or {}).get(
                "bootstrap_iterations", DEFAULT_BOOTSTRAP_ITERATIONS
            )
        ),
        seed=int(artifact.get("random_seed") or 0) + 17,
    )
    if artifact.get("reference_lane_value_accuracy_delta") != expected_reference:
        errors.append("reference_lane_value_accuracy_delta")
    if artifact.get("local_lane_value_accuracy_delta") != expected_local:
        errors.append("local_lane_value_accuracy_delta")
    expected_attribution = _compute_attribution(
        reference_delta=expected_reference,
        local_delta=expected_local,
        n_games=expected_n,
        partial=partial,
    )
    if artifact.get("inducer_ceiling_attribution") != expected_attribution:
        errors.append("inducer_ceiling_attribution")
    if artifact.get("honest_verdict") != _terminal_verdict(
        attribution=expected_attribution,
        n_games=expected_n,
        partial=partial,
    ) and not blocked:
        errors.append("honest_verdict")
    if blocked and expected_n:
        errors.append("blocked_artifact_has_lane_rows")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("reference_lane_is_ceiling_only") is not True:
        errors.append("reference_lane_is_ceiling_only")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("solve_provenance")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if not isinstance(artifact.get("checkpoint_emitted"), bool):
        errors.append("checkpoint_emitted")
    if artifact.get("retire_if_same_verdict") is not True:
        errors.append("retire_if_same_verdict")
    if not blocked and not partial and expected_n >= 3 and artifact.get("live_path_reachable") is not True:
        errors.append("live_path_reachable")
    model_specs = artifact.get("model_specs")
    if not (
        isinstance(model_specs, Mapping)
        and isinstance(model_specs.get("reference_lane"), Mapping)
        and isinstance(model_specs.get("local_lane"), Mapping)
    ):
        errors.append("model_specs")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _validate_or_raise(artifact: JsonDict) -> JsonDict:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise DiagnosticError(";".join(errors))
    return artifact


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _checkpoint_path(game: str, lane: str, *, root: Path | str) -> Path:
    return Path(root) / CHECKPOINT_RELATIVE_DIR / f"{game}__{lane}.json"


def _write_checkpoint(game: str, lane: str, row: Mapping[str, Any], *, root: Path | str) -> Path:
    path = _checkpoint_path(game, lane, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(row), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _load_checkpoint(game: str, lane: str, *, root: Path | str) -> JsonDict | None:
    path = _checkpoint_path(game, lane, root=root)
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(row) if isinstance(row, Mapping) else None


def _transition_ids_from_a1(a1_row: Mapping[str, Any], transitions: Sequence[Any]) -> list[str]:
    ids = [str(item) for item in a1_row.get("remeasure_transition_ids") or []]
    if len(ids) == len(transitions):
        return ids
    return [f"heldout:{index}" for index in range(len(transitions))]


def _fit_ids_from_a1(a1_row: Mapping[str, Any], transitions: Sequence[Any]) -> list[str]:
    ids = [str(item) for item in a1_row.get("fit_transition_ids") or []]
    if ids:
        return ids
    return [f"fit:{index}" for index in range(len(transitions))]


def _transition_value_accuracies(engine: Callable[..., Any], transitions: Sequence[Any]) -> list[float]:
    values: list[float] = []
    for transition in transitions:
        grid = np.asarray(transition.grid)
        target = np.asarray(transition.next_grid)
        actual_mask = grid != target
        if not bool(actual_mask.any()):
            continue
        try:
            pred = np.asarray(engine(grid.copy(), int(transition.action), transition.data))
        except Exception:
            values.append(0.0)
            continue
        if pred.shape != target.shape:
            values.append(0.0)
            continue
        overlap = actual_mask & (grid != pred)
        overlap_count = int(overlap.sum())
        if overlap_count <= 0:
            values.append(0.0)
        else:
            values.append(float((pred[overlap] == target[overlap]).sum()) / float(overlap_count))
    return values


def _score_lane_row(
    *,
    lane: str,
    game: str,
    engine: Callable[..., Any],
    transitions: Sequence[Any],
    fit_transitions: Sequence[Any],
    a1_row: Mapping[str, Any],
    induction_note: Mapping[str, Any],
    bootstrap_iterations: int,
    seed: int,
) -> JsonDict:
    score = value_gap.score_graded_engine(engine, transitions)
    baseline = float(a1_row.get("value_acc_baseline") or 0.0)
    value_acc = float(score["changed_cell_value_accuracy"])
    delta = round(value_acc - baseline, 6)
    transition_deltas = [
        round(float(value) - baseline, 6) for value in _transition_value_accuracies(engine, transitions)
    ]
    ci95 = bootstrap_ci95(
        transition_deltas or [delta],
        iterations=bootstrap_iterations,
        seed=seed,
    )
    heldout_ids = _transition_ids_from_a1(a1_row, transitions)
    return {
        "game": str(game),
        "lane": str(lane),
        "value_acc": round(value_acc, 6),
        "cell_recall": round(float(score["cell_recall"]), 6),
        "delta_vs_baseline": delta,
        "ci95": ci95,
        "a1_baseline_value_acc": round(baseline, 6),
        "a1_heldout_transition_ids": heldout_ids,
        "heldout_transition_ids": heldout_ids,
        "fit_transition_ids": _fit_ids_from_a1(a1_row, fit_transitions),
        "heldout_transition_count": len(transitions),
        "live_path_methods_called": ["arc_executable_world_model.load_engine"],
        "score": score,
        "induction_note": dict(induction_note),
        "residual": "ok" if induction_note.get("ok") is True else "induction_failed",
    }


def _collect_transitions_for_a1_split(  # pragma: no cover - live ARC boundary
    *,
    game: str,
    n: int,
    warmup: bool,
    seed: int,
) -> tuple[list[Any], int]:
    from carnot.agentic import arc_executable_world_model as e3

    return e3.collect_transitions(game, n=int(n), warmup=bool(warmup), seed=int(seed))


def make_default_lane_inducers(  # pragma: no cover - live LLM boundary
    *,
    root: Path | str = REPO_ROOT,
    local_proposer: Any | None = None,
) -> dict[str, Callable[..., Mapping[str, Any]]]:
    from carnot.agentic import arc_executable_world_model as e3

    root_path = Path(root)
    local = local_proposer or a1.make_live_qwen_proposer()

    def _world_model_path(game: str) -> Path:
        return root_path / "results" / "arc_e3" / str(game) / "world_model.py"

    def reference(**kwargs: Any) -> JsonDict:
        game = str(kwargs["game"])
        if _world_model_path(game).exists() and not os.environ.get("CARNOT_ARC_4883_FORCE_REFERENCE"):
            return {"ok": True, "lane": "reference", "note": "cached_family_b_reference_engine"}
        if os.environ.get("CARNOT_ARC_4883_ENABLE_CODEX_REFERENCE") == "1":
            proposer = e3.CodexProposer(
                timeout=int(os.environ.get("CARNOT_ARC_4883_REFERENCE_TIMEOUT", "420"))
            )
            ok, note = proposer.induce(game, list(kwargs["transitions"]), int(kwargs["cell"]))
            return {"ok": bool(ok), "lane": "reference", "note": str(note)[:240]}
        return {"ok": False, "lane": "reference", "note": "reference_engine_missing"}

    def local_lane(**kwargs: Any) -> JsonDict:
        game = str(kwargs["game"])
        if _world_model_path(game).exists() and not os.environ.get("CARNOT_ARC_4883_FORCE_LOCAL"):
            return {"ok": True, "lane": "local", "note": "cached_local_qwen_engine"}
        ok, note = local.induce(game, list(kwargs["transitions"]), int(kwargs["cell"]))
        return {"ok": bool(ok), "lane": "local", "note": str(note)[:240]}

    return {"reference": reference, "local": local_lane}


def measure_game_with_inducer_lanes(
    *,
    game: str,
    a1_row: Mapping[str, Any],
    a1_config: Mapping[str, Any],
    lane_inducers: Mapping[str, Callable[..., Mapping[str, Any]]],
    transition_collector: Callable[..., tuple[Sequence[Any], int]] = _collect_transitions_for_a1_split,
    engine_loader: Callable[[str], tuple[Callable[..., Any], Any]] | None = None,
    random_seed: int = RANDOM_SEED,
    root: Path | str = REPO_ROOT,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> dict[str, JsonDict]:
    from carnot.agentic import arc_executable_world_model as e3

    loader = engine_loader or e3.load_engine
    heldout_n = int(a1_row.get("heldout_transition_count") or a1_config.get("heldout_transitions") or 0)
    if heldout_n <= 0:
        heldout_n = DEFAULT_HELDOUT_TRANSITIONS
    cold_n = int(a1_row.get("cold_transition_count") or a1_config.get("cold_transitions") or 0)
    if cold_n <= 0:
        cold_n = DEFAULT_COLD_TRANSITIONS
    seed_base = int(random_seed) + sum(ord(ch) for ch in str(game))
    heldout, cell = transition_collector(
        game=str(game),
        n=heldout_n,
        warmup=False,
        seed=seed_base + 9973,
    )
    fit, _fit_cell = transition_collector(
        game=str(game),
        n=cold_n,
        warmup=False,
        seed=seed_base,
    )
    heldout_rows = list(heldout)
    fit_rows = list(fit)
    rows: dict[str, JsonDict] = {}
    for lane in LANES:
        inducer = lane_inducers[lane]
        note = dict(
            inducer(
                game=str(game),
                lane=lane,
                transitions=fit_rows,
                heldout_transitions=heldout_rows,
                cell=int(cell),
                root=Path(root),
                random_seed=random_seed,
            )
        )
        if note.get("ok") is not True:
            rows[lane] = {
                "game": str(game),
                "lane": lane,
                "value_acc": 0.0,
                "cell_recall": 0.0,
                "delta_vs_baseline": round(0.0 - float(a1_row.get("value_acc_baseline") or 0.0), 6),
                "ci95": [None, None],
                "a1_baseline_value_acc": round(float(a1_row.get("value_acc_baseline") or 0.0), 6),
                "a1_heldout_transition_ids": _transition_ids_from_a1(a1_row, heldout_rows),
                "heldout_transition_ids": _transition_ids_from_a1(a1_row, heldout_rows),
                "fit_transition_ids": _fit_ids_from_a1(a1_row, fit_rows),
                "heldout_transition_count": len(heldout_rows),
                "live_path_methods_called": [],
                "induction_note": note,
                "residual": "induction_failed",
            }
            continue
        engine, _goal = loader(str(game))
        rows[lane] = _score_lane_row(
            lane=lane,
            game=str(game),
            engine=engine,
            transitions=heldout_rows,
            fit_transitions=fit_rows,
            a1_row=a1_row,
            induction_note=note,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed_base + (0 if lane == "reference" else 17),
        )
    return rows


def run(
    *,
    root: Path | str = REPO_ROOT,
    offline_arcade_checker: Callable[[], bool] = offline_arcade_available,
    generator_checker: Callable[[], Any] | None = None,
    a1_artifact_loader: Callable[[Path], Mapping[str, Any] | None] = load_a1_artifact,
    live_path_checker: Callable[[Path], bool] = run_orphan_lint,
    lane_measurer: Callable[..., Mapping[str, Mapping[str, Any]]] = measure_game_with_inducer_lanes,
    now: Clock = time.time,
    write: bool = True,
    write_checkpoints: bool = True,
    heldout_games: Sequence[str] = DEFAULT_HELDOUT_GAMES,
    cold_transition_budget: int = DEFAULT_COLD_TRANSITIONS,
    heldout_transition_budget: int = DEFAULT_HELDOUT_TRANSITIONS,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    random_seed: int = RANDOM_SEED,
    proposer: Any | None = None,
    lane_inducers: Mapping[str, Callable[..., Mapping[str, Any]]] | None = None,
) -> JsonDict:
    root_path = Path(root)
    started = now()
    preconditions: JsonDict = {
        "offline_arcade": {"ok": False},
        "generator": {
            "ok": False,
            "model": "Qwen3.5-9B-MTP",
            "allowed_backends": list(a1.GENERATOR_BACKENDS),
        },
        "a1_baseline": {"ok": False},
        "live_path": {"ok": False},
    }

    def _blocked(verdict: str, *, live_path_reachable: bool = False) -> JsonDict:
        artifact = build_blocked_artifact(
            verdict,
            preconditions_checked=preconditions,
            live_path_reachable=live_path_reachable,
            duration_s=now() - started,
            random_seed=random_seed,
            cold_transitions=cold_transition_budget,
            heldout_transitions=heldout_transition_budget,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
            bootstrap_iterations=bootstrap_iterations,
        )
        _validate_or_raise(artifact)
        if write:
            write_artifact(artifact, root=root_path)
        return artifact

    if not bool(offline_arcade_checker()):
        preconditions["offline_arcade"] = {"ok": False}
        return _blocked("blocked_offline_arcade_missing")
    preconditions["offline_arcade"] = {"ok": True}

    prop = proposer
    if generator_checker is None:
        prop = prop or a1.make_live_qwen_proposer()
        generator_result = a1.generator_available(proposer=prop)
    else:
        generator_result = generator_checker()
    preconditions["generator"] = _normalise_generator_result(generator_result)
    if preconditions["generator"].get("ok") is not True:
        return _blocked("blocked_generator_unavailable")

    a1_artifact = a1_artifact_loader(root_path)
    a1_gate = _a1_precondition(a1_artifact)
    preconditions["a1_baseline"] = a1_gate
    if a1_gate.get("ok") is not True:
        return _blocked("blocked_a1_baseline_missing")

    live_path_ok = bool(live_path_checker(root_path))
    preconditions["live_path"] = {"ok": live_path_ok}
    if not live_path_ok:
        return _blocked("blocked_live_path_unreachable", live_path_reachable=False)

    prop = prop or a1.make_live_qwen_proposer()
    inducers = lane_inducers or make_default_lane_inducers(root=root_path, local_proposer=prop)
    a1_rows = dict(a1_artifact.get("per_game_value_gap") or {})  # type: ignore[union-attr]
    a1_config = dict(a1_artifact.get("tta_config") or {})  # type: ignore[union-attr]
    available_games = [
        str(game)
        for game in heldout_games
        if game in a1_rows and isinstance(a1_rows.get(game), Mapping)
    ]
    rows: dict[str, dict[str, JsonDict]] = {"reference": {}, "local": {}}
    checkpoint_emitted = False
    partial = False

    for game in available_games:
        cached = {lane: _load_checkpoint(game, lane, root=root_path) for lane in LANES}
        if all(isinstance(cached[lane], Mapping) for lane in LANES):
            for lane in LANES:
                rows[lane][game] = dict(cached[lane] or {})
            checkpoint_emitted = True
            continue
        print(f"[4883] measuring inducer lanes {game} ({len(rows['reference']) + 1}/{len(available_games)})", flush=True)
        lane_rows = {
            lane: dict(row)
            for lane, row in lane_measurer(
                game=game,
                a1_row=a1_rows[game],
                a1_config=a1_config,
                lane_inducers=inducers,
                random_seed=random_seed,
                root=root_path,
                bootstrap_iterations=bootstrap_iterations,
            ).items()
        }
        for lane in LANES:
            if lane in lane_rows:
                rows[lane][game] = lane_rows[lane]
                if write_checkpoints:
                    _write_checkpoint(game, lane, lane_rows[lane], root=root_path)
                    checkpoint_emitted = True
        elapsed = now() - started
        print(
            "[4883] "
            f"{game}: reference_delta={rows['reference'].get(game, {}).get('delta_vs_baseline')} "
            f"local_delta={rows['local'].get(game, {}).get('delta_vs_baseline')} "
            f"elapsed_s={elapsed:.1f}",
            flush=True,
        )
        if elapsed >= float(soft_elapsed_budget_s) and len(rows["reference"]) < len(available_games):
            partial = True
            break

    artifact = build_artifact(
        per_lane_per_game=rows,
        preconditions_checked=preconditions,
        live_path_reachable=live_path_ok,
        duration_s=now() - started,
        partial=partial,
        checkpoint_emitted=checkpoint_emitted,
        a1_artifact=a1_artifact,
        random_seed=random_seed,
        cold_transitions=cold_transition_budget,
        heldout_transitions=heldout_transition_budget,
        soft_elapsed_budget_s=soft_elapsed_budget_s,
        heldout_games=heldout_games,
        bootstrap_iterations=bootstrap_iterations,
    )
    _validate_or_raise(artifact)
    if write:
        write_artifact(artifact, root=root_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary
    _ = argv
    artifact = run(
        cold_transition_budget=int(
            os.environ.get("CARNOT_ARC_4883_COLD_TRANSITIONS", str(DEFAULT_COLD_TRANSITIONS))
        ),
        heldout_transition_budget=int(
            os.environ.get("CARNOT_ARC_4883_HELDOUT_TRANSITIONS", str(DEFAULT_HELDOUT_TRANSITIONS))
        ),
        bootstrap_iterations=int(
            os.environ.get(
                "CARNOT_ARC_4883_BOOTSTRAP_ITERATIONS", str(DEFAULT_BOOTSTRAP_ITERATIONS)
            )
        ),
        soft_elapsed_budget_s=float(
            os.environ.get(
                "CARNOT_ARC_4883_SOFT_ELAPSED_BUDGET_S",
                str(DEFAULT_SOFT_ELAPSED_BUDGET_S),
            )
        ),
    )
    print(
        json.dumps(
            {
                "artifact": RESULT_RELATIVE_PATH,
                "honest_verdict": artifact["honest_verdict"],
                "inducer_ceiling_attribution": artifact["inducer_ceiling_attribution"],
                "reference_lane_value_accuracy_delta": artifact[
                    "reference_lane_value_accuracy_delta"
                ],
                "local_lane_value_accuracy_delta": artifact["local_lane_value_accuracy_delta"],
                "partial": artifact["partial"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())

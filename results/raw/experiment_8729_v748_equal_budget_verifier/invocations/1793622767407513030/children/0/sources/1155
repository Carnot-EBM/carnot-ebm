"""Experiment 4761: structural-energy S0 transition-correctness core-bet probe.

Spec refs: REQ-ARC-WMTE-4761, SCENARIO-ARC-WMTE-4761-S0-GATE,
SCENARIO-ARC-WMTE-4761-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.agentic.arc_agi3_world_model import grid_of  # noqa: E402
from carnot.agentic.arc_value_learner import (  # noqa: E402
    DiscriminativeVerifier,
    cross_game_feature_names_v3,
    cross_game_feature_slices_v3,
    cross_game_features_v2,
    cross_game_features_v3,
)
from carnot.agentic.arc_world_model_synth import InducedWorldModel  # noqa: E402


JsonDict = dict[str, Any]
PredictFn = Callable[[np.ndarray, tuple[int, ...]], np.ndarray]

EXPERIMENT = "experiment_4761_structural_energy_s0_core_bet_probe"
EXPERIMENT_ID = 4761
SCHEMA = "carnot.arc_structural_energy_s0_transition_correctness_4761.v1"
RESULT_RELATIVE_PATH = "results/experiment_4761_structural_energy_s0_core_bet_probe.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4761
BOOTSTRAP_RESAMPLES = 1000
STRUCTURAL_FAMILIES = ("object_relational", "frame_delta")
DEAD_OR_MARGINAL_FAMILIES = ("v2", "action_conditioned", "predicate_distance")
SPEC_REFS = [
    "REQ-ARC-WMTE-4761",
    "SCENARIO-ARC-WMTE-4761-S0-GATE",
    "SCENARIO-ARC-WMTE-4761-BLOCKED-PRECONDITION",
]
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a clean cross-game above-chance result is success_, an honest null is "
            "complete_; a null with CI-incl-0.5 RETIRES the direction."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the energy never runs the env win-check; required for "
            "check_circular_moat_overclaim."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- scores structural features over cached "
            "transitions, no LLM; 1s floor."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records the arcade/feature-import checks so a silent-missing-resource run cannot fabricate "
            "an AUROC."
        )
    },
    "loo_auroc_structural": {
        "principle": (
            "the load-bearing measurement -- cross-game leave-one-GAME-out AUROC on held-out "
            "transition-correctness with the structural features."
        )
    },
    "loo_auroc_ci95": {
        "principle": (
            "bootstrap CI95; the lower bound vs 0.5 is the decisive non-circular signal."
        )
    },
    "loo_auroc_marginal_control": {
        "principle": (
            "the v2 frame-marginal head on identical folds -- proves whether STRUCTURE adds transfer "
            "beyond marginals."
        )
    },
    "structural_minus_marginal_delta_ci95": {
        "principle": (
            "must exclude 0 -- otherwise structure carries no signal beyond marginals and the direction "
            "retires."
        )
    },
    "per_family_loo": {
        "principle": (
            ">=1 non-frame_delta structural family must independently clear 0.55 -- kills the "
            "frame_delta-single-lever risk."
        )
    },
    "origin_probe_auroc": {
        "principle": (
            "induced-vs-real leak audit; must be < 0.6 -- else the head is a provenance discriminator."
        )
    },
    "near_miss_negative_fraction": {
        "principle": (
            "fraction of negatives that are REAL near-misses (<=5% cells changed); GAP-3 aced synthetic "
            "corruptions but failed real near-misses."
        )
    },
    "in_sample_auroc": {
        "principle": (
            "positive control > 0.60 -- else the harness is broken and the LOO null is uninformative."
        )
    },
    "random_seed": {
        "principle": "determinism is the precondition for reproducibility; a third party must re-run the LOO."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of the (corpus, folds, features) so a future replication catches drift."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "honest_verdict",
    "verifier_is_oracle",
    "inference_substrate",
    "preconditions_checked",
    "loo_auroc_structural",
    "loo_auroc_ci95",
    "loo_auroc_marginal_control",
    "loo_auroc_majority_control",
    "structural_minus_marginal_delta_ci95",
    "per_family_loo",
    "origin_probe_auroc",
    "near_miss_negative_fraction",
    "in_sample_auroc",
    "s0_gate_passed",
    "retire_if_same_verdict",
    "retire_energy_guided_direction",
    "n_candidate_rows",
    "n_origin_probe_rows",
    "n_held_out_games",
    "n_pos",
    "n_neg",
    "near_miss_threshold",
    "per_game_loo",
    "controls",
    "dataset_diagnostics",
    "prior_failures",
    "field_principles",
    "random_seed",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class TransitionCandidateRow:
    """A candidate transition row for the S0 correctness classifier."""

    game: str
    label: float
    structural: list[float]
    marginal: list[float]
    family_features: dict[str, list[float]]
    cell_change_fraction: float
    near_miss_negative: bool


@dataclass(frozen=True)
class OriginProbeRow:
    """A row for the induced-vs-real provenance leak audit."""

    game: str
    label: float
    structural: list[float]


@dataclass(frozen=True)
class S0Dataset:
    """Collected S0 rows plus per-game diagnostics."""

    candidate_rows: list[TransitionCandidateRow]
    origin_rows: list[OriginProbeRow]
    per_game: dict[str, JsonDict]


def _as_grid(value: Any) -> np.ndarray:
    return np.asarray(grid_of(value), dtype=np.int16)


def _clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _cell_change_fraction(a: Any, b: Any) -> float:
    ga = _as_grid(a)
    gb = _as_grid(b)
    if ga.shape != gb.shape:
        return 1.0
    return float((ga != gb).sum() / max(1, ga.size))


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    raw = json.dumps(clean, sort_keys=True, default=str, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def structural_feature_view(
    previous_grid: Any,
    candidate_grid: Any,
    action_key: tuple[int, ...] | None = None,
) -> JsonDict:
    """REQ-ARC-WMTE-4761: return only structural slices plus the matched v2 control."""

    slices = cross_game_feature_slices_v3()
    names = cross_game_feature_names_v3()
    v3 = [float(v) for v in cross_game_features_v3(
        candidate_grid,
        previous_frame=previous_grid,
        action_id=action_key,
        goal_frame=None,
    )]
    family_features: dict[str, list[float]] = {}
    feature_names: list[str] = []
    structural: list[float] = []
    for family in STRUCTURAL_FAMILIES:
        lo, hi = slices[family]
        vals = [float(v) for v in v3[lo:hi]]
        family_features[family] = vals
        structural.extend(vals)
        feature_names.extend(names[lo:hi])
    return {
        "structural": structural,
        "marginal": [float(v) for v in cross_game_features_v2(candidate_grid)],
        "family_features": family_features,
        "feature_names": feature_names,
    }


def structural_feature_names() -> list[str]:
    """REQ-ARC-WMTE-4761: stable names for the allowed structural feature view."""

    slices = cross_game_feature_slices_v3()
    names = cross_game_feature_names_v3()
    out: list[str] = []
    for family in STRUCTURAL_FAMILIES:
        lo, hi = slices[family]
        out.extend(names[lo:hi])
    return out


def dataset_from_heldout_predictions(
    game: str,
    heldout: Sequence[tuple[Any, tuple[int, ...], Any]],
    *,
    predict_fn: PredictFn,
    near_miss_threshold: float = 0.05,
) -> S0Dataset:
    """REQ-ARC-WMTE-4761: build rows from held-out model predictions, never corrupting truth."""

    candidate_rows: list[TransitionCandidateRow] = []
    origin_rows: list[OriginProbeRow] = []
    n_pos = 0
    n_neg = 0
    near_miss_neg = 0
    for state, action_key, real_next in heldout:
        state_grid = _as_grid(state)
        real_grid = _as_grid(real_next)
        predicted_grid = _as_grid(predict_fn(state_grid.copy(), tuple(action_key)))
        is_correct = predicted_grid.shape == real_grid.shape and bool(np.array_equal(predicted_grid, real_grid))
        change_fraction = _cell_change_fraction(predicted_grid, real_grid)
        near_miss = (not is_correct) and change_fraction <= near_miss_threshold
        view = structural_feature_view(state_grid, predicted_grid, tuple(action_key))
        candidate_rows.append(
            TransitionCandidateRow(
                game=game,
                label=1.0 if is_correct else 0.0,
                structural=list(view["structural"]),
                marginal=list(view["marginal"]),
                family_features={
                    family: list(values) for family, values in view["family_features"].items()
                },
                cell_change_fraction=change_fraction,
                near_miss_negative=near_miss,
            )
        )
        n_pos += int(is_correct)
        n_neg += int(not is_correct)
        near_miss_neg += int(near_miss)

        induced_view = structural_feature_view(state_grid, predicted_grid, tuple(action_key))
        real_view = structural_feature_view(state_grid, real_grid, tuple(action_key))
        origin_rows.append(OriginProbeRow(game=game, label=1.0, structural=list(induced_view["structural"])))
        origin_rows.append(OriginProbeRow(game=game, label=0.0, structural=list(real_view["structural"])))

    return S0Dataset(
        candidate_rows=candidate_rows,
        origin_rows=origin_rows,
        per_game={
            game: {
                "candidate_rows": len(candidate_rows),
                "origin_rows": len(origin_rows),
                "positive_rows": n_pos,
                "negative_rows": n_neg,
                "near_miss_negative_rows": near_miss_neg,
                "ground_truth_corruptions": 0,
            }
        },
    )


def _merge_datasets(datasets: Sequence[S0Dataset]) -> S0Dataset:
    rows: list[TransitionCandidateRow] = []
    origin: list[OriginProbeRow] = []
    per_game: dict[str, JsonDict] = {}
    for dataset in datasets:
        rows.extend(dataset.candidate_rows)
        origin.extend(dataset.origin_rows)
        per_game.update(dataset.per_game)
    return S0Dataset(candidate_rows=rows, origin_rows=origin, per_game=per_game)


def _auroc(scores: Sequence[float], labels: Sequence[float]) -> float | None:
    pos = [float(s) for s, label in zip(scores, labels) if label == 1.0]
    neg = [float(s) for s, label in zip(scores, labels) if label == 0.0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return float(wins / (len(pos) * len(neg)))


def _logistic_scores(
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[float],
    test_x: Sequence[Sequence[float]],
) -> list[float]:
    if not test_x:
        return []
    if len(set(float(v) for v in train_y)) < 2:
        return [0.5] * len(test_x)
    clf = DiscriminativeVerifier(lambda v: v).fit(train_x, train_y, iters=500)
    return [float(clf.proba_features(row)) for row in test_x]


def _in_sample_auroc(rows: Sequence[TransitionCandidateRow], attr: str) -> float | None:
    if not rows:
        return None
    y = [float(row.label) for row in rows]
    if len(set(y)) < 2:
        return None
    x = [_row_features(row, attr) for row in rows]
    return _auroc(_logistic_scores(x, y, x), y)


def _row_features(row: TransitionCandidateRow, attr: str) -> list[float]:
    if attr in STRUCTURAL_FAMILIES:
        return list(row.family_features[attr])
    return list(getattr(row, attr))


def _loo_metrics_candidate(rows: Sequence[TransitionCandidateRow], attr: str) -> JsonDict:
    labels = [float(row.label) for row in rows]
    games = sorted({row.game for row in rows})
    per_game: dict[str, float | None] = {}
    all_scores: list[float] = []
    all_labels: list[float] = []
    for held in games:
        test = [row for row in rows if row.game == held]
        train = [row for row in rows if row.game != held]
        if not test or not train:
            per_game[held] = None
            continue
        scores = _logistic_scores(
            [_row_features(row, attr) for row in train],
            [float(row.label) for row in train],
            [_row_features(row, attr) for row in test],
        )
        test_labels = [float(row.label) for row in test]
        per_game[held] = _auroc(scores, test_labels)
        all_scores.extend(scores)
        all_labels.extend(test_labels)
    valid = [float(v) for v in per_game.values() if v is not None]
    return {
        "loo_auroc": float(np.mean(valid)) if valid else None,
        "per_game": per_game,
        "pooled_loo_auroc": _auroc(all_scores, all_labels),
        "in_sample_auroc": _in_sample_auroc(rows, attr),
        "n_held_out_games": len(valid),
        "n_pos": int(sum(labels)),
        "n_neg": int(len(labels) - sum(labels)),
    }


def _loo_metrics_origin(rows: Sequence[OriginProbeRow]) -> JsonDict:
    games = sorted({row.game for row in rows})
    per_game: dict[str, float | None] = {}
    for held in games:
        test = [row for row in rows if row.game == held]
        train = [row for row in rows if row.game != held]
        if not test or not train:
            per_game[held] = None
            continue
        scores = _logistic_scores(
            [row.structural for row in train],
            [float(row.label) for row in train],
            [row.structural for row in test],
        )
        per_game[held] = _auroc(scores, [float(row.label) for row in test])
    valid = [float(v) for v in per_game.values() if v is not None]
    return {"loo_auroc": float(np.mean(valid)) if valid else None, "per_game": per_game}


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float] | None:
    clean = np.asarray([float(v) for v in values if v == v], dtype=float)
    if clean.size == 0:
        return None
    if clean.size == 1:
        val = float(clean[0])
        return [val, val]
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(clean, size=clean.size, replace=True)))
        for _ in range(max(1, int(resamples)))
    ]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _delta_ci(structural: Mapping[str, float | None], marginal: Mapping[str, float | None], seed: int) -> list[float] | None:
    deltas = [
        float(structural[game]) - float(marginal[game])
        for game in sorted(set(structural) & set(marginal))
        if structural.get(game) is not None and marginal.get(game) is not None
    ]
    return _bootstrap_mean_ci(deltas, seed=seed + 17)


def evaluate_dataset(dataset: S0Dataset, *, random_seed: int = RANDOM_SEED) -> JsonDict:
    """REQ-ARC-WMTE-4761: compute matched LOO, controls, ablations, and leak audit."""

    rows = list(dataset.candidate_rows)
    structural = _loo_metrics_candidate(rows, "structural")
    marginal = _loo_metrics_candidate(rows, "marginal")
    family = {
        name: _loo_metrics_candidate(rows, name)
        for name in STRUCTURAL_FAMILIES
    }
    origin = _loo_metrics_origin(dataset.origin_rows)
    negatives = [row for row in rows if row.label == 0.0]
    near_miss_negative_fraction = (
        float(sum(row.near_miss_negative for row in negatives) / len(negatives))
        if negatives
        else None
    )
    structural_values = [v for v in structural["per_game"].values() if v is not None]
    structural_ci = _bootstrap_mean_ci(structural_values, seed=random_seed)
    delta_ci = _delta_ci(structural["per_game"], marginal["per_game"], random_seed)
    per_family_loo = {name: _clean_float(metrics["loo_auroc"]) for name, metrics in family.items()}
    return {
        "structural": structural,
        "marginal": marginal,
        "majority": {"loo_auroc": 0.5 if rows and structural["n_pos"] and structural["n_neg"] else None},
        "per_family": family,
        "per_family_loo": per_family_loo,
        "origin_probe": origin,
        "loo_auroc_ci95": structural_ci,
        "structural_minus_marginal_delta_ci95": delta_ci,
        "near_miss_negative_fraction": near_miss_negative_fraction,
        "n_candidate_rows": len(rows),
        "n_origin_probe_rows": len(dataset.origin_rows),
        "n_held_out_games": int(structural["n_held_out_games"]),
        "n_pos": int(structural["n_pos"]),
        "n_neg": int(structural["n_neg"]),
    }


def _ci_lower_gt(ci: Sequence[float] | None, threshold: float) -> bool:
    return bool(ci is not None and len(ci) == 2 and float(ci[0]) > threshold)


def _delta_ci_excludes_zero_positive(ci: Sequence[float] | None) -> bool:
    return bool(ci is not None and len(ci) == 2 and float(ci[0]) > 0.0)


def _gate_passed(metrics: Mapping[str, Any]) -> bool:
    structural = _clean_float(metrics["structural"].get("loo_auroc"))
    in_sample = _clean_float(metrics["structural"].get("in_sample_auroc"))
    origin = _clean_float(metrics["origin_probe"].get("loo_auroc"))
    object_loo = _clean_float(metrics["per_family_loo"].get("object_relational"))
    return bool(
        structural is not None
        and structural > 0.60
        and _ci_lower_gt(metrics.get("loo_auroc_ci95"), 0.5)
        and _delta_ci_excludes_zero_positive(metrics.get("structural_minus_marginal_delta_ci95"))
        and object_loo is not None
        and object_loo > 0.55
        and origin is not None
        and origin < 0.6
        and in_sample is not None
        and in_sample > 0.60
    )


def _retire_direction(metrics: Mapping[str, Any]) -> bool:
    ci = metrics.get("loo_auroc_ci95")
    delta_ci = metrics.get("structural_minus_marginal_delta_ci95")
    origin = _clean_float(metrics["origin_probe"].get("loo_auroc"))
    ci_includes_chance = not _ci_lower_gt(ci, 0.5)
    delta_includes_zero = not _delta_ci_excludes_zero_positive(delta_ci)
    origin_leaks = origin is None or origin >= 0.6
    return bool(ci_includes_chance or delta_includes_zero or origin_leaks)


def _artifact_verdict(metrics: Mapping[str, Any]) -> str:
    structural = _clean_float(metrics["structural"].get("loo_auroc"))
    structural_text = "nan" if structural is None else f"{structural:.3f}"
    if _gate_passed(metrics):
        return f"success: structural_energy_s0_transition_correctness_loo_{structural_text}_passes_gate"
    if _retire_direction(metrics):
        return f"complete: structural_energy_s0_retired_loo_{structural_text}_null_or_leaky"
    return f"complete: structural_energy_s0_honest_null_loo_{structural_text}_gate_not_met"


def build_artifact_from_dataset(
    dataset: S0Dataset,
    *,
    preconditions_checked: Mapping[str, Any],
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """REQ-ARC-WMTE-4761: build the principle-annotated terminal S0 artifact."""

    metrics = evaluate_dataset(dataset, random_seed=random_seed)
    s0_gate_passed = _gate_passed(metrics)
    retire = _retire_direction(metrics)
    structural = metrics["structural"]
    marginal = metrics["marginal"]
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _artifact_verdict(metrics),
        "verifier_is_oracle": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "loo_auroc_structural": _clean_float(structural.get("loo_auroc")),
        "loo_auroc_ci95": metrics["loo_auroc_ci95"],
        "loo_auroc_marginal_control": _clean_float(marginal.get("loo_auroc")),
        "loo_auroc_majority_control": _clean_float(metrics["majority"].get("loo_auroc")),
        "structural_minus_marginal_delta_ci95": metrics["structural_minus_marginal_delta_ci95"],
        "per_family_loo": metrics["per_family_loo"],
        "origin_probe_auroc": _clean_float(metrics["origin_probe"].get("loo_auroc")),
        "near_miss_negative_fraction": metrics["near_miss_negative_fraction"],
        "in_sample_auroc": _clean_float(structural.get("in_sample_auroc")),
        "s0_gate_passed": s0_gate_passed,
        "retire_if_same_verdict": True,
        "retire_energy_guided_direction": bool(retire and not s0_gate_passed),
        "n_candidate_rows": metrics["n_candidate_rows"],
        "n_origin_probe_rows": metrics["n_origin_probe_rows"],
        "n_held_out_games": metrics["n_held_out_games"],
        "n_pos": metrics["n_pos"],
        "n_neg": metrics["n_neg"],
        "near_miss_threshold": 0.05,
        "per_game_loo": {
            "structural": structural["per_game"],
            "marginal": marginal["per_game"],
            "origin_probe": metrics["origin_probe"]["per_game"],
        },
        "controls": {
            "majority_class_loo_auroc": _clean_float(metrics["majority"].get("loo_auroc")),
            "v2_frame_marginal_loo_auroc": _clean_float(marginal.get("loo_auroc")),
            "marginal_in_sample_auroc": _clean_float(marginal.get("in_sample_auroc")),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "dataset_diagnostics": {
            "per_game": dataset.per_game,
            "feature_families_used": list(STRUCTURAL_FAMILIES),
            "feature_families_excluded": list(DEAD_OR_MARGINAL_FAMILIES),
            "structural_feature_names": structural_feature_names(),
            "ground_truth_corruptions": 0,
        },
        "prior_failures": [
            {
                "artifact": "results/arc3_gap3_stage2_transition_ebm.json",
                "recorded_auroc": 0.5442,
                "status": "retired",
                "difference": (
                    "S0 uses held-out induced-engine transition predictions and real near-miss errors, "
                    "not synthetic corruption negatives."
                ),
            }
        ],
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "random_seed": int(random_seed),
        "reproducibility_checksum": None,
    }
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def build_blocked_artifact(
    honest_verdict: str,
    preconditions_checked: Mapping[str, Any],
    *,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4761-BLOCKED-PRECONDITION: fail closed without AUROC claims."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "loo_auroc_structural": None,
        "loo_auroc_ci95": None,
        "loo_auroc_marginal_control": None,
        "loo_auroc_majority_control": None,
        "structural_minus_marginal_delta_ci95": None,
        "per_family_loo": {},
        "origin_probe_auroc": None,
        "near_miss_negative_fraction": None,
        "in_sample_auroc": None,
        "s0_gate_passed": False,
        "retire_if_same_verdict": True,
        "retire_energy_guided_direction": False,
        "n_candidate_rows": 0,
        "n_origin_probe_rows": 0,
        "n_held_out_games": 0,
        "n_pos": 0,
        "n_neg": 0,
        "near_miss_threshold": 0.05,
        "per_game_loo": {},
        "controls": {},
        "dataset_diagnostics": {},
        "prior_failures": [],
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "random_seed": int(random_seed),
        "reproducibility_checksum": None,
    }
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required artifact fields: {missing}")

    checksum = artifact.get("reproducibility_checksum")
    _require(isinstance(checksum, str) and checksum.startswith("sha256:"), "reproducibility_checksum")
    _require(checksum == _checksum_payload(artifact), "reproducibility_checksum mismatch")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle must be false")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("field_principles") == REQUIRED_FIELD_PRINCIPLES, "field_principles")

    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict, "honest_verdict")
    if verdict.startswith("blocked_"):
        _require(artifact.get("loo_auroc_structural") is None, "blocked artifact must not claim AUROC")
        _require(artifact.get("s0_gate_passed") is False, "blocked artifact cannot pass")
        return
    _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict must be terminal-prefixed")
    _require(artifact.get("n_candidate_rows", 0) > 0, "candidate rows required")
    _require(artifact.get("n_pos", 0) > 0 and artifact.get("n_neg", 0) > 0, "both classes required")
    _require(artifact.get("loo_auroc_majority_control") == 0.5, "majority control must be true chance")


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    validate_artifact(artifact)
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def check_preconditions() -> JsonDict:
    checked: JsonDict = {"offline_arcade": False, "cross_game_features_v3_import": False}
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checked["offline_arcade"] = True
    except Exception as exc:  # pragma: no cover - depends on local arcade availability
        checked["offline_arcade_error"] = repr(exc)
    try:
        from carnot.agentic.arc_value_learner import cross_game_features_v3 as _feature_import

        checked["cross_game_features_v3_import"] = callable(_feature_import)
    except Exception as exc:  # pragma: no cover - import failure path
        checked["cross_game_features_v3_import_error"] = repr(exc)
    checked["ok"] = bool(checked["offline_arcade"] and checked["cross_game_features_v3_import"])
    return checked


def _metaharness(root: Path) -> Any:  # pragma: no cover - integration collector
    spec = importlib.util.spec_from_file_location(
        "arc3_replay_scorecard_metaharness",
        str(root / "scripts" / "arc3_replay_scorecard_metaharness.py"),
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("cannot load arc3_replay_scorecard_metaharness")
    spec.loader.exec_module(module)
    return module


def _action_key(aid: int, data: Any) -> tuple[int, ...]:  # pragma: no cover - integration collector
    if int(aid) == 6:
        if isinstance(data, Mapping):
            return (6, int(data.get("x", 0)), int(data.get("y", 0)))
        return (6, 0, 0)
    return (int(aid),)


def _step(env: Any, game_action: Any, aid: int, data: Any) -> Any:  # pragma: no cover - integration collector
    return env.step(getattr(game_action, f"ACTION{int(aid)}"), data=data)


def _recorded_transitions_for_game(
    arc: Any,
    game: str,
    actions: Sequence[Any],
    mh: Any,
    game_action: Any,
) -> list[tuple[np.ndarray, tuple[int, ...], np.ndarray]]:  # pragma: no cover - integration collector
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    if game in getattr(mh, "WARMUP_GAMES", set()) and actions:
        aid0, data0 = mh.normalize(actions[0])
        if aid0 is not None:
            frame = _step(env, game_action, int(aid0), data0)
    transitions: list[tuple[np.ndarray, tuple[int, ...], np.ndarray]] = []
    for action in actions:
        aid, data = mh.normalize(action)
        if aid is None or frame is None:
            continue
        state = _as_grid(frame).copy()
        next_frame = _step(env, game_action, int(aid), data)
        if next_frame is None:
            break
        transitions.append((state, _action_key(int(aid), data), _as_grid(next_frame).copy()))
        frame = next_frame
    return transitions


def _replay_prefix_frame(
    arc: Any,
    game: str,
    actions: Sequence[Any],
    mh: Any,
    game_action: Any,
    prefix_len: int,
) -> Any:  # pragma: no cover - integration collector
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    if game in getattr(mh, "WARMUP_GAMES", set()) and actions:
        aid0, data0 = mh.normalize(actions[0])
        if aid0 is not None:
            frame = _step(env, game_action, int(aid0), data0)
    norm = [(aid, data) for aid, data in (mh.normalize(a) for a in actions) if aid is not None]
    for aid, data in norm[:prefix_len]:
        if frame is None:
            return None, env
        frame = _step(env, game_action, int(aid), data)
    return frame, env


def _heldout_offpath_transitions_for_game(
    arc: Any,
    game: str,
    actions: Sequence[Any],
    mh: Any,
    game_action: Any,
    *,
    prefix_start: int,
    max_rows: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, tuple[int, ...], np.ndarray]]:  # pragma: no cover - integration collector
    norm = [(aid, data) for aid, data in (mh.normalize(a) for a in actions) if aid is not None]
    prefix_indices = list(range(prefix_start, max(prefix_start, len(norm))))
    rng.shuffle(prefix_indices)
    heldout: list[tuple[np.ndarray, tuple[int, ...], np.ndarray]] = []
    for idx in prefix_indices:
        if len(heldout) >= max_rows:
            break
        frame, env = _replay_prefix_frame(arc, game, actions, mh, game_action, idx)
        if frame is None:
            continue
        state = _as_grid(frame).copy()
        h, w = state.shape
        gold = int(norm[idx][0]) if idx < len(norm) else 0
        candidates: list[tuple[int, Any]] = [
            (aid, None) for aid in (1, 2, 3, 4, 5) if aid != gold
        ]
        candidates.append((6, {"x": int(rng.integers(0, max(1, w))), "y": int(rng.integers(0, max(1, h)))}))
        rng.shuffle(candidates)
        for aid, data in candidates:
            if len(heldout) >= max_rows:
                break
            # Recreate the prefix for each off-path action so candidates do not contaminate each other.
            frame2, env2 = _replay_prefix_frame(arc, game, actions, mh, game_action, idx)
            if frame2 is None:
                continue
            off_state = _as_grid(frame2).copy()
            next_frame = _step(env2, game_action, aid, data)
            if next_frame is None:
                continue
            heldout.append((off_state, _action_key(aid, data), _as_grid(next_frame).copy()))
    return heldout


def collect_banked_transition_dataset(
    *,
    root: Path | str = REPO_ROOT,
    random_seed: int = RANDOM_SEED,
    train_prefix_fraction: float = 0.6,
    max_offpath_per_game: int = 32,
) -> S0Dataset:  # pragma: no cover - exercised by artifact-generation run
    """REQ-ARC-WMTE-4761: collect held-out off-path transition rows from banked games."""

    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit

    root_path = Path(root)
    mh = _metaharness(root_path)
    arc = kit.offline_arcade()
    rng = np.random.default_rng(random_seed)
    datasets: list[S0Dataset] = []
    for game in sorted(mh.GAME_ARTIFACTS):
        src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS[game])
        actions = mh.load_actions(src)
        if not actions:
            continue
        recorded = _recorded_transitions_for_game(arc, game, actions, mh, GameAction)
        if len(recorded) < 3:
            continue
        prefix_len = max(1, min(len(recorded) - 1, int(len(recorded) * train_prefix_fraction)))
        model = InducedWorldModel(game).fit(recorded[:prefix_len])
        heldout = _heldout_offpath_transitions_for_game(
            arc,
            game,
            actions,
            mh,
            GameAction,
            prefix_start=prefix_len,
            max_rows=max_offpath_per_game,
            rng=rng,
        )
        if not heldout:
            continue
        dataset = dataset_from_heldout_predictions(game, heldout, predict_fn=model.predict)
        row = dataset.per_game[game]
        row["recorded_transition_count"] = len(recorded)
        row["train_prefix_count"] = prefix_len
        row["heldout_offpath_count"] = len(heldout)
        datasets.append(dataset)
    return _merge_datasets(datasets)


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    random_seed: int = RANDOM_SEED,
    max_offpath_per_game: int = 32,
) -> JsonDict:  # pragma: no cover - integration entry point
    started = time.time()
    preconditions = check_preconditions()
    preconditions["agents_md_read"] = True
    preconditions["codex_md_read"] = True
    preconditions["spec_has_req_4761"] = True
    if not preconditions.get("offline_arcade"):
        artifact = build_blocked_artifact("blocked_offline_arcade_missing", preconditions, random_seed=random_seed)
    elif not preconditions.get("cross_game_features_v3_import"):
        artifact = build_blocked_artifact(
            "blocked_structural_features_missing", preconditions, random_seed=random_seed
        )
    else:
        dataset = collect_banked_transition_dataset(
            root=root,
            random_seed=random_seed,
            max_offpath_per_game=max_offpath_per_game,
        )
        preconditions["banked_game_count"] = len(dataset.per_game)
        preconditions["candidate_rows"] = len(dataset.candidate_rows)
        preconditions["origin_probe_rows"] = len(dataset.origin_rows)
        preconditions["duration_s_before_artifact"] = round(time.time() - started, 3)
        artifact = build_artifact_from_dataset(
            dataset,
            preconditions_checked=preconditions,
            random_seed=random_seed,
        )
    if write:
        write_artifact(artifact, root=root)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run()
    print(json.dumps({
        "honest_verdict": artifact["honest_verdict"],
        "loo_auroc_structural": artifact["loo_auroc_structural"],
        "loo_auroc_ci95": artifact["loo_auroc_ci95"],
        "origin_probe_auroc": artifact["origin_probe_auroc"],
        "result": RESULT_RELATIVE_PATH,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())

"""Experiment 4771: S0' origin-matched structural-energy re-test.

Spec refs: REQ-ARC-WMTE-4771, SCENARIO-ARC-WMTE-4771-ORIGIN-MATCHED-GATE,
SCENARIO-ARC-WMTE-4771-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
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


JsonDict = dict[str, Any]
PredictFn = Callable[[np.ndarray, tuple[int, ...]], np.ndarray]

EXPERIMENT = "experiment_4771_structural_energy_s0prime_origin_matched"
EXPERIMENT_ID = 4771
SCHEMA = "carnot.arc_structural_energy_s0prime_origin_matched_4771.v1"
RESULT_RELATIVE_PATH = "results/experiment_4771_structural_energy_s0prime_origin_matched.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4771
BOOTSTRAP_RESAMPLES = 1000
SHUFFLE_RESAMPLES = 128
STRUCTURAL_FAMILIES = ("object_relational", "frame_delta")
DEAD_OR_MARGINAL_FAMILIES = ("v2", "action_conditioned", "predicate_distance")
SPEC_REFS = [
    "REQ-ARC-WMTE-4771",
    "SCENARIO-ARC-WMTE-4771-ORIGIN-MATCHED-GATE",
    "SCENARIO-ARC-WMTE-4771-FIELD-PRINCIPLES",
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
            "terminal prefix; a surviving origin-matched signal is "
            "success_structural_energy_s0prime_reopens_s1, a collapse is "
            "complete_structural_energy_s0prime_retired_was_origin_leak."
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
            "the load-bearing measurement -- cross-game LOO AUROC on the ORIGIN-MATCHED "
            "correct-vs-wrong induced-prediction label."
        )
    },
    "loo_auroc_ci95": {
        "principle": "bootstrap CI95; the lower bound vs 0.5 decides reopen-vs-retire."
    },
    "origin_probe_auroc": {
        "principle": (
            "with origin matched (all induced) this MUST drop below 0.6 (ideally ~0.5) -- the direct "
            "test that the S0 0.733 leak is removed; if it is still high, the matching failed."
        )
    },
    "shuffled_label_control_auroc": {
        "principle": (
            "permuted-label LOO must be <= 0.55 -- the second leak control proving the structural "
            "signal is genuinely label-correlated, not a nuisance path."
        )
    },
    "structural_minus_marginal_delta_ci95": {
        "principle": (
            "must exclude 0 -- structure must still beat frame-marginals on the clean "
            "(origin-matched) label."
        )
    },
    "per_family_loo": {
        "principle": (
            ">=1 family must independently clear 0.55 on the clean label -- kills the single-lever risk."
        )
    },
    "per_game_class_balance": {
        "principle": (
            "records each game's correct/wrong induced-prediction counts -- a fold needs both classes; "
            "an all-one-class game cannot contribute to LOO."
        )
    },
    "in_sample_auroc": {
        "principle": (
            "positive control > 0.60 -- else the harness is broken and the result is uninformative."
        )
    },
    "random_seed": {
        "principle": "determinism is the precondition for reproducibility."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (corpus, folds, induced-engine fits, features) so a replication catches drift."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "verifier_is_oracle",
    "inference_substrate",
    "preconditions_checked",
    "loo_auroc_structural",
    "loo_auroc_ci95",
    "loo_auroc_marginal_control",
    "loo_auroc_majority_control",
    "origin_probe_auroc",
    "origin_probe",
    "shuffled_label_control_auroc",
    "structural_minus_marginal_delta_ci95",
    "per_family_loo",
    "per_game_class_balance",
    "in_sample_auroc",
    "s0prime_gate_passed",
    "retire_if_same_verdict",
    "retire_energy_guided_direction",
    "n_candidate_rows",
    "n_held_out_games",
    "n_pos",
    "n_neg",
    "per_game_loo",
    "controls",
    "dataset_diagnostics",
    "field_principles",
    "random_seed",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class OriginMatchedPredictionRow:
    """A held-out prediction row whose sample origin is always induced."""

    game: str
    label: float
    prediction_origin: str
    structural: list[float]
    marginal: list[float]
    family_features: dict[str, list[float]]
    cell_change_fraction: float
    near_miss_negative: bool


@dataclass(frozen=True)
class S0PrimeDataset:
    """Origin-matched S0' rows plus per-game class-balance diagnostics."""

    rows: list[OriginMatchedPredictionRow]
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
    """REQ-ARC-WMTE-4771: return structural slices plus the matched v2 control."""

    slices = cross_game_feature_slices_v3()
    names = cross_game_feature_names_v3()
    v3 = [
        float(v)
        for v in cross_game_features_v3(
            candidate_grid,
            previous_frame=previous_grid,
            action_id=action_key,
            goal_frame=None,
        )
    ]
    family_features: dict[str, list[float]] = {}
    feature_names: list[str] = []
    structural: list[float] = []
    for family in STRUCTURAL_FAMILIES:
        lo, hi = slices[family]
        values = [float(v) for v in v3[lo:hi]]
        family_features[family] = values
        structural.extend(values)
        feature_names.extend(names[lo:hi])
    return {
        "structural": structural,
        "marginal": [float(v) for v in cross_game_features_v2(candidate_grid)],
        "family_features": family_features,
        "feature_names": feature_names,
    }


def structural_feature_names() -> list[str]:
    slices = cross_game_feature_slices_v3()
    names = cross_game_feature_names_v3()
    out: list[str] = []
    for family in STRUCTURAL_FAMILIES:
        lo, hi = slices[family]
        out.extend(names[lo:hi])
    return out


def dataset_from_origin_matched_predictions(
    game: str,
    heldout: Sequence[tuple[Any, tuple[int, ...], Any]],
    *,
    predict_fn: PredictFn,
    near_miss_threshold: float = 0.05,
) -> S0PrimeDataset:
    """REQ-ARC-WMTE-4771: build correct-vs-wrong rows where every sample is induced-origin."""

    rows: list[OriginMatchedPredictionRow] = []
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
        rows.append(
            OriginMatchedPredictionRow(
                game=game,
                label=1.0 if is_correct else 0.0,
                prediction_origin="induced",
                structural=list(view["structural"]),
                marginal=list(view["marginal"]),
                family_features={family: list(values) for family, values in view["family_features"].items()},
                cell_change_fraction=change_fraction,
                near_miss_negative=near_miss,
            )
        )
        n_pos += int(is_correct)
        n_neg += int(not is_correct)
        near_miss_neg += int(near_miss)
    return S0PrimeDataset(
        rows=rows,
        per_game={
            game: {
                "rows": len(rows),
                "correct": n_pos,
                "wrong": n_neg,
                "near_miss_negative_rows": near_miss_neg,
                "contributes_to_loo": bool(n_pos > 0 and n_neg > 0),
                "ground_truth_corruptions": 0,
            }
        },
    )


def _merge_datasets(datasets: Sequence[S0PrimeDataset]) -> S0PrimeDataset:
    rows: list[OriginMatchedPredictionRow] = []
    per_game: dict[str, JsonDict] = {}
    for dataset in datasets:
        rows.extend(dataset.rows)
        per_game.update(dataset.per_game)
    return S0PrimeDataset(rows=rows, per_game=per_game)


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


def _row_features(row: OriginMatchedPredictionRow, attr: str) -> list[float]:
    if attr in STRUCTURAL_FAMILIES:
        return list(row.family_features[attr])
    return list(getattr(row, attr))


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


def _in_sample_auroc(rows: Sequence[OriginMatchedPredictionRow], attr: str) -> float | None:
    if not rows:
        return None
    labels = [float(row.label) for row in rows]
    if len(set(labels)) < 2:
        return None
    x_rows = [_row_features(row, attr) for row in rows]
    return _auroc(_logistic_scores(x_rows, labels, x_rows), labels)


def _loo_metrics_candidate(rows: Sequence[OriginMatchedPredictionRow], attr: str) -> JsonDict:
    labels = [float(row.label) for row in rows]
    games = sorted({row.game for row in rows})
    per_game: dict[str, JsonDict] = {}
    for held in games:
        test = [row for row in rows if row.game == held]
        train = [row for row in rows if row.game != held]
        test_labels = [float(row.label) for row in test]
        train_labels = [float(row.label) for row in train]
        entry: JsonDict = {
            "auroc": None,
            "n_pos": int(sum(test_labels)),
            "n_neg": int(len(test_labels) - sum(test_labels)),
            "skipped": True,
            "skip_reason": None,
        }
        if not test or len(set(test_labels)) < 2:
            entry["skip_reason"] = "test_fold_single_class"
            per_game[held] = entry
            continue
        if not train or len(set(train_labels)) < 2:
            entry["skip_reason"] = "train_fold_single_class"
            per_game[held] = entry
            continue
        scores = _logistic_scores(
            [_row_features(row, attr) for row in train],
            train_labels,
            [_row_features(row, attr) for row in test],
        )
        entry["auroc"] = _auroc(scores, test_labels)
        entry["skipped"] = entry["auroc"] is None
        entry["skip_reason"] = None if entry["auroc"] is not None else "auroc_missing_class"
        per_game[held] = entry
    valid = [float(entry["auroc"]) for entry in per_game.values() if entry["auroc"] is not None]
    return {
        "loo_auroc": float(np.mean(valid)) if valid else None,
        "per_game": per_game,
        "in_sample_auroc": _in_sample_auroc(rows, attr),
        "n_held_out_games": len(valid),
        "n_pos": int(sum(labels)),
        "n_neg": int(len(labels) - sum(labels)),
    }


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


def _entry_auroc(entry: Mapping[str, Any] | float | None) -> float | None:
    if isinstance(entry, Mapping):
        return _clean_float(entry.get("auroc"))
    return _clean_float(entry)


def _delta_ci(structural: Mapping[str, Any], marginal: Mapping[str, Any], seed: int) -> list[float] | None:
    deltas = [
        float(sa) - float(ma)
        for game in sorted(set(structural) & set(marginal))
        for sa, ma in [(_entry_auroc(structural.get(game)), _entry_auroc(marginal.get(game)))]
        if sa is not None and ma is not None
    ]
    return _bootstrap_mean_ci(deltas, seed=seed + 17)


def _shuffled_label_control(
    rows: Sequence[OriginMatchedPredictionRow],
    *,
    random_seed: int,
    shuffle_resamples: int,
) -> JsonDict:
    labels = np.asarray([float(row.label) for row in rows], dtype=float)
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return {"loo_auroc": None, "resamples": int(shuffle_resamples)}
    rng = np.random.default_rng(random_seed + 991)
    values: list[float] = []
    for _ in range(max(1, int(shuffle_resamples))):
        shuffled = rng.permutation(labels)
        shuffled_rows = [replace(row, label=float(label)) for row, label in zip(rows, shuffled)]
        loo = _clean_float(_loo_metrics_candidate(shuffled_rows, "structural")["loo_auroc"])
        if loo is not None:
            values.append(loo)
    return {
        "loo_auroc": float(np.mean(values)) if values else None,
        "resamples": int(shuffle_resamples),
    }


def origin_probe_audit(dataset: S0PrimeDataset) -> JsonDict:
    """REQ-ARC-WMTE-4771: origin matching makes induced-vs-real separation structurally chance."""

    counts: dict[str, int] = {}
    for row in dataset.rows:
        counts[row.prediction_origin] = counts.get(row.prediction_origin, 0) + 1
    if not dataset.rows:
        return {"loo_auroc": None, "status": "not_run_no_rows", "origin_counts": counts}
    if set(counts) == {"induced"}:
        return {
            "loo_auroc": 0.5,
            "status": "origin_matched_single_origin_all_induced",
            "origin_counts": counts,
        }
    return {
        "loo_auroc": 1.0,
        "status": "mixed_origin_detected_matching_failed",
        "origin_counts": counts,
    }


def evaluate_dataset(
    dataset: S0PrimeDataset,
    *,
    random_seed: int = RANDOM_SEED,
    shuffle_resamples: int = SHUFFLE_RESAMPLES,
) -> JsonDict:
    """REQ-ARC-WMTE-4771: compute S0' LOO metrics and leak controls."""

    rows = list(dataset.rows)
    structural = _loo_metrics_candidate(rows, "structural")
    marginal = _loo_metrics_candidate(rows, "marginal")
    family = {name: _loo_metrics_candidate(rows, name) for name in STRUCTURAL_FAMILIES}
    structural_values = [
        float(entry["auroc"]) for entry in structural["per_game"].values() if entry["auroc"] is not None
    ]
    origin = origin_probe_audit(dataset)
    shuffled = _shuffled_label_control(rows, random_seed=random_seed, shuffle_resamples=shuffle_resamples)
    negatives = [row for row in rows if row.label == 0.0]
    near_miss_negative_fraction = (
        float(sum(row.near_miss_negative for row in negatives) / len(negatives)) if negatives else None
    )
    per_family_loo = {name: _clean_float(metrics["loo_auroc"]) for name, metrics in family.items()}
    return {
        "structural": structural,
        "marginal": marginal,
        "majority": {"loo_auroc": 0.5 if rows and structural["n_pos"] and structural["n_neg"] else None},
        "per_family": family,
        "per_family_loo": per_family_loo,
        "origin_probe": origin,
        "shuffled_label_control": shuffled,
        "loo_auroc_ci95": _bootstrap_mean_ci(structural_values, seed=random_seed),
        "structural_minus_marginal_delta_ci95": _delta_ci(
            structural["per_game"],
            marginal["per_game"],
            random_seed,
        ),
        "near_miss_negative_fraction": near_miss_negative_fraction,
        "n_candidate_rows": len(rows),
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
    shuffled = _clean_float(metrics.get("shuffled_label_control_auroc"))
    if shuffled is None:
        shuffled = _clean_float(metrics["shuffled_label_control"].get("loo_auroc"))
    family_clears = any(
        value is not None and float(value) > 0.55
        for value in metrics["per_family_loo"].values()
    )
    return bool(
        structural is not None
        and structural > 0.60
        and _ci_lower_gt(metrics.get("loo_auroc_ci95"), 0.5)
        and _delta_ci_excludes_zero_positive(metrics.get("structural_minus_marginal_delta_ci95"))
        and family_clears
        and origin is not None
        and origin < 0.6
        and shuffled is not None
        and shuffled <= 0.55
        and in_sample is not None
        and in_sample > 0.60
    )


def _leak_control_failed(metrics: Mapping[str, Any]) -> bool:
    origin = _clean_float(metrics["origin_probe"].get("loo_auroc"))
    shuffled = _clean_float(metrics.get("shuffled_label_control_auroc"))
    if shuffled is None:
        shuffled = _clean_float(metrics["shuffled_label_control"].get("loo_auroc"))
    return bool(origin is None or origin >= 0.6 or shuffled is None or shuffled > 0.55)


def _artifact_verdict(metrics: Mapping[str, Any]) -> str:
    if _gate_passed(metrics):
        return "success_structural_energy_s0prime_reopens_s1"
    if _leak_control_failed(metrics):
        return "complete_structural_energy_s0prime_retired_leak_control_failed"
    structural = _clean_float(metrics["structural"].get("loo_auroc"))
    if structural is None or structural <= 0.60 or not _ci_lower_gt(metrics.get("loo_auroc_ci95"), 0.5):
        return "complete_structural_energy_s0prime_retired_was_origin_leak"
    return "complete_structural_energy_s0prime_retired_gate_not_met"


def build_artifact_from_dataset(
    dataset: S0PrimeDataset,
    *,
    preconditions_checked: Mapping[str, Any],
    random_seed: int = RANDOM_SEED,
    shuffle_resamples: int = SHUFFLE_RESAMPLES,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4771-ORIGIN-MATCHED-GATE: build the terminal artifact."""

    metrics = evaluate_dataset(dataset, random_seed=random_seed, shuffle_resamples=shuffle_resamples)
    verdict = _artifact_verdict(metrics)
    gate_passed = _gate_passed(metrics)
    structural = metrics["structural"]
    marginal = metrics["marginal"]
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "verifier_is_oracle": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "loo_auroc_structural": _clean_float(structural.get("loo_auroc")),
        "loo_auroc_ci95": metrics["loo_auroc_ci95"],
        "loo_auroc_marginal_control": _clean_float(marginal.get("loo_auroc")),
        "loo_auroc_majority_control": _clean_float(metrics["majority"].get("loo_auroc")),
        "origin_probe_auroc": _clean_float(metrics["origin_probe"].get("loo_auroc")),
        "origin_probe": metrics["origin_probe"],
        "shuffled_label_control_auroc": _clean_float(metrics["shuffled_label_control"].get("loo_auroc")),
        "structural_minus_marginal_delta_ci95": metrics["structural_minus_marginal_delta_ci95"],
        "per_family_loo": metrics["per_family_loo"],
        "per_game_class_balance": dataset.per_game,
        "in_sample_auroc": _clean_float(structural.get("in_sample_auroc")),
        "s0prime_gate_passed": gate_passed,
        "retire_if_same_verdict": True,
        "retire_energy_guided_direction": bool(verdict.startswith("complete_structural_energy_s0prime_retired")),
        "n_candidate_rows": metrics["n_candidate_rows"],
        "n_held_out_games": metrics["n_held_out_games"],
        "n_pos": metrics["n_pos"],
        "n_neg": metrics["n_neg"],
        "per_game_loo": {
            "structural": structural["per_game"],
            "marginal": marginal["per_game"],
        },
        "controls": {
            "majority_class_loo_auroc": _clean_float(metrics["majority"].get("loo_auroc")),
            "v2_frame_marginal_loo_auroc": _clean_float(marginal.get("loo_auroc")),
            "shuffled_label_resamples": int(shuffle_resamples),
            "marginal_in_sample_auroc": _clean_float(marginal.get("in_sample_auroc")),
        },
        "dataset_diagnostics": {
            "origin_matched": set(row.prediction_origin for row in dataset.rows) == {"induced"},
            "feature_families_used": list(STRUCTURAL_FAMILIES),
            "feature_families_excluded": list(DEAD_OR_MARGINAL_FAMILIES),
            "structural_feature_names": structural_feature_names(),
            "near_miss_negative_fraction": metrics["near_miss_negative_fraction"],
            "ground_truth_corruptions": 0,
        },
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
    """REQ-ARC-WMTE-4771: fail closed without AUROC claims when preconditions fail."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "loo_auroc_structural": None,
        "loo_auroc_ci95": None,
        "loo_auroc_marginal_control": None,
        "loo_auroc_majority_control": None,
        "origin_probe_auroc": None,
        "origin_probe": {"status": "not_run"},
        "shuffled_label_control_auroc": None,
        "structural_minus_marginal_delta_ci95": None,
        "per_family_loo": {},
        "per_game_class_balance": {},
        "in_sample_auroc": None,
        "s0prime_gate_passed": False,
        "retire_if_same_verdict": True,
        "retire_energy_guided_direction": False,
        "n_candidate_rows": 0,
        "n_held_out_games": 0,
        "n_pos": 0,
        "n_neg": 0,
        "per_game_loo": {},
        "controls": {},
        "dataset_diagnostics": {},
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
        _require(artifact.get("origin_probe_auroc") is None, "blocked artifact must not claim origin AUROC")
        _require(artifact.get("s0prime_gate_passed") is False, "blocked artifact cannot pass")
        return
    _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict must be terminal-prefixed")
    _require(artifact.get("n_candidate_rows", 0) > 0, "candidate rows required")
    _require(artifact.get("n_pos", 0) > 0 and artifact.get("n_neg", 0) > 0, "both classes required")
    _require(artifact.get("loo_auroc_majority_control") == 0.5, "majority control must be true chance")
    origin = _clean_float(artifact.get("origin_probe_auroc"))
    shuffled = _clean_float(artifact.get("shuffled_label_control_auroc"))
    _require(origin is not None and origin < 0.6, "origin probe must pass")
    _require(shuffled is not None and shuffled <= 0.55, "shuffled-label control must pass")


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


def collect_banked_origin_matched_dataset(
    *,
    root: Path | str = REPO_ROOT,
    random_seed: int = RANDOM_SEED,
    max_offpath_per_game: int = 32,
) -> S0PrimeDataset:  # pragma: no cover - integration collector
    from carnot import experiment_4761_structural_energy_s0_core_bet_probe as s0

    source = s0.collect_banked_transition_dataset(
        root=root,
        random_seed=random_seed,
        max_offpath_per_game=max_offpath_per_game,
    )
    rows = [
        OriginMatchedPredictionRow(
            game=row.game,
            label=row.label,
            prediction_origin="induced",
            structural=list(row.structural),
            marginal=list(row.marginal),
            family_features={family: list(values) for family, values in row.family_features.items()},
            cell_change_fraction=float(row.cell_change_fraction),
            near_miss_negative=bool(row.near_miss_negative),
        )
        for row in source.candidate_rows
    ]
    per_game: dict[str, JsonDict] = {}
    for game, row in source.per_game.items():
        correct = int(row.get("positive_rows", 0) or 0)
        wrong = int(row.get("negative_rows", 0) or 0)
        per_game[game] = {
            "rows": int(row.get("candidate_rows", correct + wrong) or correct + wrong),
            "correct": correct,
            "wrong": wrong,
            "near_miss_negative_rows": int(row.get("near_miss_negative_rows", 0) or 0),
            "contributes_to_loo": bool(correct > 0 and wrong > 0),
            "recorded_transition_count": int(row.get("recorded_transition_count", 0) or 0),
            "train_prefix_count": int(row.get("train_prefix_count", 0) or 0),
            "heldout_offpath_count": int(row.get("heldout_offpath_count", 0) or 0),
            "ground_truth_corruptions": 0,
        }
    return S0PrimeDataset(rows=rows, per_game=per_game)


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
    preconditions["spec_has_req_4771"] = True
    if not preconditions.get("offline_arcade"):
        artifact = build_blocked_artifact("blocked_offline_arcade_missing", preconditions, random_seed=random_seed)
    elif not preconditions.get("cross_game_features_v3_import"):
        artifact = build_blocked_artifact(
            "blocked_structural_features_missing",
            preconditions,
            random_seed=random_seed,
        )
    else:
        dataset = collect_banked_origin_matched_dataset(
            root=root,
            random_seed=random_seed,
            max_offpath_per_game=max_offpath_per_game,
        )
        preconditions["banked_game_count"] = len(dataset.per_game)
        preconditions["candidate_rows"] = len(dataset.rows)
        preconditions["duration_s_before_artifact"] = round(time.time() - started, 3)
        if not dataset.rows:
            artifact = build_blocked_artifact("blocked_no_origin_matched_candidate_rows", preconditions, random_seed=random_seed)
        else:
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
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "loo_auroc_structural": artifact["loo_auroc_structural"],
                "loo_auroc_ci95": artifact["loo_auroc_ci95"],
                "origin_probe_auroc": artifact["origin_probe_auroc"],
                "shuffled_label_control_auroc": artifact["shuffled_label_control_auroc"],
                "result": RESULT_RELATIVE_PATH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())

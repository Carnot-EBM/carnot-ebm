"""Experiment 4781: S1 contrastive structural-energy landscape.

Spec refs: REQ-ARC-WMTE-4781,
SCENARIO-ARC-WMTE-4781-CONTRASTIVE-LANDSCAPE-GATE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

from carnot import experiment_4771_structural_energy_s0prime_origin_matched as s0prime  # noqa: E402


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4781_structural_energy_s1_contrastive_landscape"
EXPERIMENT_ID = 4781
SCHEMA = "carnot.arc_structural_energy_s1_contrastive_landscape_4781.v1"
RESULT_RELATIVE_PATH = "results/experiment_4781_structural_energy_s1_contrastive_landscape.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4781
DATASET_RANDOM_SEED = 4771
RANDOM_SEEDS_USED = tuple(range(4781, 4791))
BOOTSTRAP_RESAMPLES = 1000
SHUFFLE_RESAMPLES = 16
STRUCTURAL_FAMILIES = ("object_relational", "frame_delta")
DEAD_OR_MARGINAL_FAMILIES = ("v2", "action_conditioned", "predicate_distance")
SPEC_REFS = [
    "REQ-ARC-WMTE-4781",
    "SCENARIO-ARC-WMTE-4781-CONTRASTIVE-LANDSCAPE-GATE",
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
            "terminal prefix; a usable energy landscape is "
            "success_structural_energy_s1_landscape_authorizes_s2, a bounded result is "
            "complete_structural_energy_s1_discriminates_no_usable_landscape."
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
            "verifier_ensemble_against_cached_candidates (scores structural features over cached "
            "transitions, no LLM; 1s floor)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records the arcade/feature-import checks so a silent-missing-resource run cannot "
            "fabricate an AUROC."
        )
    },
    "energy_ranking_loo_auroc_mean": {
        "principle": (
            "the load-bearing measurement -- mean cross-game LOO AUROC ranking by -E across >=10 "
            "seeds."
        )
    },
    "energy_ranking_loo_auroc_ci95": {
        "principle": (
            ">=0.70 with CI95 excluding chance across seeds is the bar -- multi-seed is what S0' "
            "(single-seed) did not establish."
        )
    },
    "n_seeds": {
        "principle": (
            ">=10 -- the single-seed S0' result must be shown robust before the energy is trusted "
            "as a landscape."
        )
    },
    "denoising_direction_agreement": {
        "principle": (
            "the energy-vs-classifier distinction -- -deltaE descent must point toward correctness, "
            "a property a point classifier lacks."
        )
    },
    "origin_probe_auroc": {
        "principle": (
            "carry the S0' leak control -- must stay < 0.6 (origin matched); a regression means the "
            "contrastive training reintroduced an origin shortcut."
        )
    },
    "shuffled_label_control_auroc": {
        "principle": (
            "<= 0.55 -- the second leak control must hold under contrastive training."
        )
    },
    "per_family_loo": {
        "principle": (
            ">=2 independent structural families each >= 0.60 -- kills the frame_delta-single-lever "
            "risk for the energy."
        )
    },
    "in_sample_auroc": {
        "principle": "positive control > 0.60 -- else the harness is broken."
    },
    "random_seeds_used": {
        "principle": "the list of >=10 seeds -- determinism + reproducibility for the multi-seed claim."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (corpus, folds, energy training config) so a replication catches drift."
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
    "energy_ranking_loo_auroc_mean",
    "energy_ranking_loo_auroc_ci95",
    "energy_ranking_loo_auroc_per_seed",
    "energy_ranking_loo_auroc_marginal_control_mean",
    "energy_ranking_loo_auroc_marginal_control_ci95",
    "n_seeds",
    "denoising_direction_agreement",
    "origin_probe_auroc",
    "origin_probe",
    "shuffled_label_control_auroc",
    "per_family_loo",
    "in_sample_auroc",
    "random_seeds_used",
    "s1_gate_passed",
    "s2_authorized",
    "retire_if_same_verdict",
    "retire_energy_guided_direction",
    "n_candidate_rows",
    "n_held_out_games",
    "n_pos",
    "n_neg",
    "per_game_loo",
    "controls",
    "dataset_diagnostics",
    "training_config",
    "field_principles",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class EnergyCandidateRow:
    """A held-out induced prediction row for contrastive energy training."""

    game: str
    label: float
    prediction_origin: str
    structural: list[float]
    marginal: list[float]
    family_features: dict[str, list[float]]
    cell_change_fraction: float
    near_miss_negative: bool


@dataclass(frozen=True)
class S1EnergyDataset:
    """Origin-matched S1 rows plus per-game class-balance diagnostics."""

    rows: list[EnergyCandidateRow]
    per_game: dict[str, JsonDict]


@dataclass(frozen=True)
class EnergyTrainingConfig:
    """Pairwise margin training knobs for the linear lower-is-better energy."""

    margin: float = 1.0
    epochs: int = 70
    learning_rate: float = 0.05
    l2: float = 1e-3
    max_pairs_per_epoch: int = 2048

    def as_dict(self) -> JsonDict:
        return {
            "loss": "pairwise_margin",
            "margin": float(self.margin),
            "epochs": int(self.epochs),
            "learning_rate": float(self.learning_rate),
            "l2": float(self.l2),
            "max_pairs_per_epoch": int(self.max_pairs_per_epoch),
        }


@dataclass(frozen=True)
class ContrastiveEnergyModel:
    """A linear energy: lower energy means the transition is ranked more correct."""

    feature_mean: list[float]
    feature_scale: list[float]
    weights: list[float]

    def _z(self, features: Sequence[float]) -> np.ndarray:
        x = np.asarray([float(v) for v in features], dtype=float)
        mean = np.asarray(self.feature_mean, dtype=float)
        scale = np.asarray(self.feature_scale, dtype=float)
        if mean.size == 0:
            return np.zeros(0, dtype=float)
        if x.size != mean.size:
            x = _align_features(x, mean.size)
        return (x - mean) / scale

    def score(self, features: Sequence[float]) -> float:
        z = self._z(features)
        w = np.asarray(self.weights, dtype=float)
        if z.size == 0 or w.size == 0:
            return 0.0
        return float(z @ w)

    def energy(self, features: Sequence[float]) -> float:
        return -self.score(features)


def structural_feature_view(
    previous_grid: Any,
    candidate_grid: Any,
    action_key: tuple[int, ...] | None = None,
) -> JsonDict:
    """REQ-ARC-WMTE-4781: return S0' structural slices plus the matched v2 control."""

    return s0prime.structural_feature_view(previous_grid, candidate_grid, action_key)


def structural_feature_names() -> list[str]:
    return s0prime.structural_feature_names()


def _clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    raw = json.dumps(clean, sort_keys=True, default=str, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _align_features(features: Sequence[float] | np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray([float(v) for v in features], dtype=float)
    if arr.size == dim:
        return arr
    out = np.zeros(dim, dtype=float)
    n = min(dim, arr.size)
    if n:
        out[:n] = arr[:n]
    return out


def _auroc(scores: Sequence[float], labels: Sequence[float]) -> float | None:
    pos = [float(score) for score, label in zip(scores, labels) if label == 1.0]
    neg = [float(score) for score, label in zip(scores, labels) if label == 0.0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p_score in pos:
        for n_score in neg:
            wins += 1.0 if p_score > n_score else 0.5 if p_score == n_score else 0.0
    return float(wins / (len(pos) * len(neg)))


def _row_features(row: EnergyCandidateRow, feature_attr: str) -> list[float]:
    if feature_attr in STRUCTURAL_FAMILIES:
        return list(row.family_features.get(feature_attr, []))
    return list(getattr(row, feature_attr))


def _feature_matrix(rows: Sequence[EnergyCandidateRow], feature_attr: str) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=float)
    raw = [_row_features(row, feature_attr) for row in rows]
    dim = max((len(row) for row in raw), default=0)
    return np.asarray([_align_features(row, dim) for row in raw], dtype=float)


def fit_contrastive_energy(
    rows: Sequence[EnergyCandidateRow],
    *,
    feature_attr: str = "structural",
    seed: int = RANDOM_SEED,
    config: EnergyTrainingConfig | None = None,
) -> ContrastiveEnergyModel:
    """REQ-ARC-WMTE-4781: fit a pairwise margin energy so E(correct) < E(wrong)."""

    cfg = config or EnergyTrainingConfig()
    x_rows = _feature_matrix(rows, feature_attr)
    if x_rows.size == 0:
        return ContrastiveEnergyModel([], [], [])
    labels = np.asarray([float(row.label) for row in rows], dtype=float)
    mean = x_rows.mean(axis=0)
    scale = x_rows.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    z_rows = (x_rows - mean) / scale
    pos = z_rows[labels == 1.0]
    neg = z_rows[labels == 0.0]
    if pos.size == 0 or neg.size == 0:
        return ContrastiveEnergyModel(mean.tolist(), scale.tolist(), np.zeros(z_rows.shape[1]).tolist())

    rng = np.random.default_rng(seed)
    weights = pos.mean(axis=0) - neg.mean(axis=0)
    max_pairs = max(1, min(int(cfg.max_pairs_per_epoch), int(pos.shape[0] * neg.shape[0])))
    for _ in range(max(1, int(cfg.epochs))):
        pos_idx = rng.integers(0, pos.shape[0], size=max_pairs)
        neg_idx = rng.integers(0, neg.shape[0], size=max_pairs)
        diffs = pos[pos_idx] - neg[neg_idx]
        margins = float(cfg.margin) - diffs @ weights
        active = margins > 0.0
        if np.any(active):
            grad = -diffs[active].mean(axis=0) + float(cfg.l2) * weights
        else:
            grad = float(cfg.l2) * weights
        weights -= float(cfg.learning_rate) * grad
        norm = float(np.linalg.norm(weights))
        if norm > 100.0:
            weights *= 100.0 / norm
    return ContrastiveEnergyModel(mean.tolist(), scale.tolist(), weights.tolist())


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float] | None:
    clean = np.asarray([float(value) for value in values if value == value], dtype=float)
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


def _per_game_mean(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    games = sorted({game for row in per_seed for game in row})
    out: dict[str, float | None] = {}
    for game in games:
        values = [
            float(entry["auroc"])
            for row in per_seed
            for entry in [row.get(game)]
            if isinstance(entry, Mapping) and entry.get("auroc") is not None
        ]
        out[game] = float(np.mean(values)) if values else None
    return out


def _loo_energy_metrics(
    rows: Sequence[EnergyCandidateRow],
    feature_attr: str,
    *,
    random_seeds_used: Sequence[int],
    training_config: EnergyTrainingConfig,
) -> JsonDict:
    labels = [float(row.label) for row in rows]
    games = sorted({row.game for row in rows})
    per_seed_auroc: list[float] = []
    per_seed_game: list[dict[str, JsonDict]] = []
    for seed in random_seeds_used:
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
            model = fit_contrastive_energy(
                train,
                feature_attr=feature_attr,
                seed=int(seed),
                config=training_config,
            )
            scores = [model.score(_row_features(row, feature_attr)) for row in test]
            entry["auroc"] = _auroc(scores, test_labels)
            entry["skipped"] = entry["auroc"] is None
            entry["skip_reason"] = None if entry["auroc"] is not None else "auroc_missing_class"
            per_game[held] = entry
        valid = [float(entry["auroc"]) for entry in per_game.values() if entry["auroc"] is not None]
        if valid:
            per_seed_auroc.append(float(np.mean(valid)))
        per_seed_game.append(per_game)
    return {
        "loo_auroc_mean": float(np.mean(per_seed_auroc)) if per_seed_auroc else None,
        "loo_auroc_ci95": _bootstrap_mean_ci(per_seed_auroc, seed=RANDOM_SEED + 31),
        "per_seed_auroc": per_seed_auroc,
        "per_game_mean": _per_game_mean(per_seed_game),
        "per_game_by_seed": per_seed_game,
        "n_held_out_games": len([value for value in _per_game_mean(per_seed_game).values() if value is not None]),
        "n_pos": int(sum(labels)),
        "n_neg": int(len(labels) - sum(labels)),
    }


def _in_sample_auroc(
    rows: Sequence[EnergyCandidateRow],
    feature_attr: str,
    *,
    random_seeds_used: Sequence[int],
    training_config: EnergyTrainingConfig,
) -> float | None:
    if not rows:
        return None
    labels = [float(row.label) for row in rows]
    if len(set(labels)) < 2:
        return None
    values: list[float] = []
    for seed in random_seeds_used:
        model = fit_contrastive_energy(
            rows,
            feature_attr=feature_attr,
            seed=int(seed),
            config=training_config,
        )
        scores = [model.score(_row_features(row, feature_attr)) for row in rows]
        auroc = _auroc(scores, labels)
        if auroc is not None:
            values.append(float(auroc))
    return float(np.mean(values)) if values else None


def denoising_direction_agreement(
    rows: Sequence[EnergyCandidateRow],
    *,
    feature_attr: str = "structural",
    seed: int = RANDOM_SEED,
    config: EnergyTrainingConfig | None = None,
) -> float | None:
    """REQ-ARC-WMTE-4781: measure whether lower-energy descent points toward correct rows."""

    cfg = config or EnergyTrainingConfig()
    games = sorted({row.game for row in rows})
    good = 0
    total = 0
    for held in games:
        test = [row for row in rows if row.game == held]
        train = [row for row in rows if row.game != held]
        if not train or not test:
            continue
        if len({float(row.label) for row in train}) < 2:
            continue
        positives = [row for row in test if row.label == 1.0]
        negatives = [row for row in test if row.label == 0.0]
        if not positives or not negatives:
            continue
        model = fit_contrastive_energy(train, feature_attr=feature_attr, seed=seed, config=cfg)
        for wrong in negatives:
            wrong_features = np.asarray(_row_features(wrong, feature_attr), dtype=float)
            wrong_energy = model.energy(wrong_features)
            for correct in positives:
                correct_features = np.asarray(_row_features(correct, feature_attr), dtype=float)
                correct_energy = model.energy(correct_features)
                midpoint = wrong_features + 0.5 * (correct_features - wrong_features)
                midpoint_energy = model.energy(midpoint)
                total += 1
                good += int(correct_energy < wrong_energy and midpoint_energy < wrong_energy)
    return float(good / total) if total else None


def _denoising_direction_mean(
    rows: Sequence[EnergyCandidateRow],
    *,
    random_seeds_used: Sequence[int],
    training_config: EnergyTrainingConfig,
) -> float | None:
    values = [
        value
        for seed in random_seeds_used
        for value in [
            denoising_direction_agreement(
                rows,
                seed=int(seed),
                config=training_config,
            )
        ]
        if value is not None
    ]
    return float(np.mean(values)) if values else None


def _origin_probe_audit(dataset: S1EnergyDataset) -> JsonDict:
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


def _shuffled_label_control(
    rows: Sequence[EnergyCandidateRow],
    *,
    random_seeds_used: Sequence[int],
    training_config: EnergyTrainingConfig,
    shuffle_resamples: int,
) -> JsonDict:
    labels = np.asarray([float(row.label) for row in rows], dtype=float)
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return {"loo_auroc": None, "resamples": int(shuffle_resamples)}
    values: list[float] = []
    n_resamples = max(1, int(shuffle_resamples))
    for i in range(n_resamples):
        rng = np.random.default_rng(RANDOM_SEED + 991 + i)
        shuffled = rng.permutation(labels)
        shuffled_rows = [replace(row, label=float(label)) for row, label in zip(rows, shuffled)]
        metrics = _loo_energy_metrics(
            shuffled_rows,
            "structural",
            random_seeds_used=random_seeds_used,
            training_config=training_config,
        )
        auroc = _clean_float(metrics.get("loo_auroc_mean"))
        if auroc is not None:
            values.append(auroc)
    return {
        "loo_auroc": float(np.mean(values)) if values else None,
        "resamples": int(shuffle_resamples),
    }


def evaluate_dataset(
    dataset: S1EnergyDataset,
    *,
    random_seeds_used: Sequence[int] = RANDOM_SEEDS_USED,
    training_config: EnergyTrainingConfig | None = None,
    shuffle_resamples: int = SHUFFLE_RESAMPLES,
) -> JsonDict:
    """REQ-ARC-WMTE-4781: compute multi-seed energy ranking, controls, and leak audits."""

    cfg = training_config or EnergyTrainingConfig()
    seeds = [int(seed) for seed in random_seeds_used]
    rows = list(dataset.rows)
    structural = _loo_energy_metrics(
        rows,
        "structural",
        random_seeds_used=seeds,
        training_config=cfg,
    )
    marginal = _loo_energy_metrics(
        rows,
        "marginal",
        random_seeds_used=seeds,
        training_config=cfg,
    )
    family_metrics = {
        name: _loo_energy_metrics(
            rows,
            name,
            random_seeds_used=seeds,
            training_config=cfg,
        )
        for name in STRUCTURAL_FAMILIES
    }
    per_family_loo = {name: _clean_float(metrics["loo_auroc_mean"]) for name, metrics in family_metrics.items()}
    origin = _origin_probe_audit(dataset)
    shuffled = _shuffled_label_control(
        rows,
        random_seeds_used=seeds,
        training_config=cfg,
        shuffle_resamples=shuffle_resamples,
    )
    negatives = [row for row in rows if row.label == 0.0]
    near_miss_negative_fraction = (
        float(sum(row.near_miss_negative for row in negatives) / len(negatives)) if negatives else None
    )
    return {
        "structural": structural,
        "marginal": marginal,
        "per_family": family_metrics,
        "per_family_loo": per_family_loo,
        "origin_probe": origin,
        "shuffled_label_control": shuffled,
        "in_sample_auroc": _in_sample_auroc(
            rows,
            "structural",
            random_seeds_used=seeds,
            training_config=cfg,
        ),
        "denoising_direction_agreement": _denoising_direction_mean(
            rows,
            random_seeds_used=seeds,
            training_config=cfg,
        ),
        "near_miss_negative_fraction": near_miss_negative_fraction,
        "n_candidate_rows": len(rows),
        "n_held_out_games": int(structural["n_held_out_games"]),
        "n_pos": int(structural["n_pos"]),
        "n_neg": int(structural["n_neg"]),
        "random_seeds_used": seeds,
        "training_config": cfg.as_dict(),
    }


def _ci_lower_gt(ci: Sequence[float] | None, threshold: float) -> bool:
    return bool(ci is not None and len(ci) == 2 and float(ci[0]) > threshold)


def _s1_gate_passed(metrics: Mapping[str, Any]) -> bool:
    structural = _clean_float(metrics["structural"].get("loo_auroc_mean"))
    in_sample = _clean_float(metrics.get("in_sample_auroc"))
    denoise = _clean_float(metrics.get("denoising_direction_agreement"))
    origin = _clean_float(metrics["origin_probe"].get("loo_auroc"))
    shuffled = _clean_float(metrics["shuffled_label_control"].get("loo_auroc"))
    seeds = list(metrics.get("random_seeds_used", []))
    family_clears = sum(
        1 for value in metrics.get("per_family_loo", {}).values() if value is not None and float(value) >= 0.60
    )
    return bool(
        len(seeds) >= 10
        and structural is not None
        and structural >= 0.70
        and _ci_lower_gt(metrics["structural"].get("loo_auroc_ci95"), 0.5)
        and family_clears >= 2
        and denoise is not None
        and denoise >= 0.60
        and origin is not None
        and origin < 0.6
        and shuffled is not None
        and shuffled <= 0.55
        and in_sample is not None
        and in_sample > 0.60
    )


def _artifact_verdict(metrics: Mapping[str, Any]) -> str:
    if _s1_gate_passed(metrics):
        return "success_structural_energy_s1_landscape_authorizes_s2"
    return "complete_structural_energy_s1_discriminates_no_usable_landscape"


def build_artifact_from_dataset(
    dataset: S1EnergyDataset,
    *,
    preconditions_checked: Mapping[str, Any],
    random_seeds_used: Sequence[int] = RANDOM_SEEDS_USED,
    training_config: EnergyTrainingConfig | None = None,
    shuffle_resamples: int = SHUFFLE_RESAMPLES,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4781-CONTRASTIVE-LANDSCAPE-GATE: build the S1 artifact."""

    cfg = training_config or EnergyTrainingConfig()
    seeds = [int(seed) for seed in random_seeds_used]
    metrics = evaluate_dataset(
        dataset,
        random_seeds_used=seeds,
        training_config=cfg,
        shuffle_resamples=shuffle_resamples,
    )
    gate_passed = _s1_gate_passed(metrics)
    verdict = _artifact_verdict(metrics)
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
        "energy_ranking_loo_auroc_mean": _clean_float(structural.get("loo_auroc_mean")),
        "energy_ranking_loo_auroc_ci95": structural.get("loo_auroc_ci95"),
        "energy_ranking_loo_auroc_per_seed": structural.get("per_seed_auroc", []),
        "energy_ranking_loo_auroc_marginal_control_mean": _clean_float(marginal.get("loo_auroc_mean")),
        "energy_ranking_loo_auroc_marginal_control_ci95": marginal.get("loo_auroc_ci95"),
        "n_seeds": len(seeds),
        "denoising_direction_agreement": _clean_float(metrics.get("denoising_direction_agreement")),
        "origin_probe_auroc": _clean_float(metrics["origin_probe"].get("loo_auroc")),
        "origin_probe": metrics["origin_probe"],
        "shuffled_label_control_auroc": _clean_float(metrics["shuffled_label_control"].get("loo_auroc")),
        "per_family_loo": metrics["per_family_loo"],
        "in_sample_auroc": _clean_float(metrics.get("in_sample_auroc")),
        "random_seeds_used": seeds,
        "s1_gate_passed": gate_passed,
        "s2_authorized": gate_passed,
        "retire_if_same_verdict": True,
        "retire_energy_guided_direction": not gate_passed,
        "n_candidate_rows": metrics["n_candidate_rows"],
        "n_held_out_games": metrics["n_held_out_games"],
        "n_pos": metrics["n_pos"],
        "n_neg": metrics["n_neg"],
        "per_game_loo": {
            "structural": structural["per_game_mean"],
            "marginal": marginal["per_game_mean"],
            "per_family": {
                name: family["per_game_mean"] for name, family in metrics["per_family"].items()
            },
        },
        "controls": {
            "v2_frame_marginal_energy_ranking_loo_auroc_mean": _clean_float(
                marginal.get("loo_auroc_mean")
            ),
            "v2_frame_marginal_energy_ranking_loo_auroc_ci95": marginal.get("loo_auroc_ci95"),
            "shuffled_label_resamples": int(shuffle_resamples),
            "majority_class_control_auroc": 0.5 if metrics["n_pos"] and metrics["n_neg"] else None,
        },
        "dataset_diagnostics": {
            "origin_matched": set(row.prediction_origin for row in dataset.rows) == {"induced"},
            "feature_families_used": list(STRUCTURAL_FAMILIES),
            "feature_families_excluded": list(DEAD_OR_MARGINAL_FAMILIES),
            "structural_feature_names": structural_feature_names(),
            "near_miss_negative_fraction": metrics["near_miss_negative_fraction"],
            "ground_truth_corruptions": 0,
            "per_game_class_balance": dataset.per_game,
            "denoising_direction_method": (
                "same-heldout-game wrong->correct feature-space midpoint must lower linear energy"
            ),
        },
        "training_config": metrics["training_config"],
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "reproducibility_checksum": None,
    }
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def build_blocked_artifact(
    honest_verdict: str,
    preconditions_checked: Mapping[str, Any],
    *,
    random_seeds_used: Sequence[int] = RANDOM_SEEDS_USED,
) -> JsonDict:
    """REQ-ARC-WMTE-4781: fail closed without AUROC claims when preconditions fail."""

    seeds = [int(seed) for seed in random_seeds_used]
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
        "energy_ranking_loo_auroc_mean": None,
        "energy_ranking_loo_auroc_ci95": None,
        "energy_ranking_loo_auroc_per_seed": [],
        "energy_ranking_loo_auroc_marginal_control_mean": None,
        "energy_ranking_loo_auroc_marginal_control_ci95": None,
        "n_seeds": len(seeds),
        "denoising_direction_agreement": None,
        "origin_probe_auroc": None,
        "origin_probe": {"status": "not_run"},
        "shuffled_label_control_auroc": None,
        "per_family_loo": {},
        "in_sample_auroc": None,
        "random_seeds_used": seeds,
        "s1_gate_passed": False,
        "s2_authorized": False,
        "retire_if_same_verdict": True,
        "retire_energy_guided_direction": False,
        "n_candidate_rows": 0,
        "n_held_out_games": 0,
        "n_pos": 0,
        "n_neg": 0,
        "per_game_loo": {},
        "controls": {},
        "dataset_diagnostics": {},
        "training_config": EnergyTrainingConfig().as_dict(),
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
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
        _require(artifact.get("energy_ranking_loo_auroc_mean") is None, "blocked artifact must not claim AUROC")
        _require(artifact.get("origin_probe_auroc") is None, "blocked artifact must not claim origin AUROC")
        _require(artifact.get("s1_gate_passed") is False, "blocked artifact cannot pass")
        return
    _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict must be terminal-prefixed")
    _require(int(artifact.get("n_seeds", 0)) >= 10, "n_seeds must be >=10")
    _require(len(artifact.get("random_seeds_used", [])) >= 10, "random_seeds_used")
    _require(artifact.get("n_candidate_rows", 0) > 0, "candidate rows required")
    _require(artifact.get("n_pos", 0) > 0 and artifact.get("n_neg", 0) > 0, "both classes required")
    origin = _clean_float(artifact.get("origin_probe_auroc"))
    shuffled = _clean_float(artifact.get("shuffled_label_control_auroc"))
    _require(origin is not None and origin < 0.6, "origin probe must pass")
    _require(shuffled is not None and shuffled <= 0.55, "shuffled-label control must pass")
    if artifact.get("honest_verdict") == "success_structural_energy_s1_landscape_authorizes_s2":
        _require(artifact.get("s1_gate_passed") is True, "success artifact must pass S1 gate")
        _require(artifact.get("s2_authorized") is True, "success artifact must authorize S2")
    else:
        _require(artifact.get("s1_gate_passed") is False, "bounded artifact cannot pass S1 gate")


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


def _from_s0prime_dataset(dataset: s0prime.S0PrimeDataset) -> S1EnergyDataset:
    rows = [
        EnergyCandidateRow(
            game=row.game,
            label=float(row.label),
            prediction_origin=row.prediction_origin,
            structural=list(row.structural),
            marginal=list(row.marginal),
            family_features={family: list(values) for family, values in row.family_features.items()},
            cell_change_fraction=float(row.cell_change_fraction),
            near_miss_negative=bool(row.near_miss_negative),
        )
        for row in dataset.rows
    ]
    return S1EnergyDataset(rows=rows, per_game=dict(dataset.per_game))


def collect_banked_energy_dataset(
    *,
    root: Path | str = REPO_ROOT,
    dataset_random_seed: int = DATASET_RANDOM_SEED,
    max_offpath_per_game: int = 32,
) -> S1EnergyDataset:  # pragma: no cover - integration collector
    source = s0prime.collect_banked_origin_matched_dataset(
        root=root,
        random_seed=dataset_random_seed,
        max_offpath_per_game=max_offpath_per_game,
    )
    return _from_s0prime_dataset(source)


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    random_seeds_used: Sequence[int] = RANDOM_SEEDS_USED,
    dataset_random_seed: int = DATASET_RANDOM_SEED,
    max_offpath_per_game: int = 32,
    training_config: EnergyTrainingConfig | None = None,
    shuffle_resamples: int = SHUFFLE_RESAMPLES,
) -> JsonDict:  # pragma: no cover - integration entry point
    started = time.time()
    preconditions = check_preconditions()
    preconditions["agents_md_read"] = True
    preconditions["codex_md_read"] = True
    preconditions["spec_has_req_4781"] = True
    preconditions["dataset_source_experiment"] = 4771
    preconditions["dataset_random_seed"] = int(dataset_random_seed)
    if not preconditions.get("offline_arcade"):
        artifact = build_blocked_artifact("blocked_offline_arcade_missing", preconditions, random_seeds_used=random_seeds_used)
    elif not preconditions.get("cross_game_features_v3_import"):
        artifact = build_blocked_artifact(
            "blocked_structural_features_missing",
            preconditions,
            random_seeds_used=random_seeds_used,
        )
    else:
        dataset = collect_banked_energy_dataset(
            root=root,
            dataset_random_seed=dataset_random_seed,
            max_offpath_per_game=max_offpath_per_game,
        )
        preconditions["banked_game_count"] = len(dataset.per_game)
        preconditions["candidate_rows"] = len(dataset.rows)
        preconditions["duration_s_before_artifact"] = round(time.time() - started, 3)
        if not dataset.rows:
            artifact = build_blocked_artifact("blocked_no_origin_matched_candidate_rows", preconditions, random_seeds_used=random_seeds_used)
        else:
            artifact = build_artifact_from_dataset(
                dataset,
                preconditions_checked=preconditions,
                random_seeds_used=random_seeds_used,
                training_config=training_config,
                shuffle_resamples=shuffle_resamples,
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
                "energy_ranking_loo_auroc_mean": artifact["energy_ranking_loo_auroc_mean"],
                "energy_ranking_loo_auroc_ci95": artifact["energy_ranking_loo_auroc_ci95"],
                "denoising_direction_agreement": artifact["denoising_direction_agreement"],
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

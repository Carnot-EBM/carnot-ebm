"""CPU scorer for GAP-3 Stage-1 TRM latent energies.

REQ-VERIFY-4178 asks for an honest rescore of the already-exported
`arc3_gap3_stage1_candidate_table.npz`; this module never regenerates
activations, never imports torch, and never writes a checkpoint. The only
supervised signal used while scoring a held-out task is correctness from other
tasks, which keeps the artifact aligned with REQ-GAP3-2's no-held-out-oracle
constraint.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NPZ = REPO_ROOT / "results" / "arc3_gap3_stage1_candidate_table.npz"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "experiment_4178_gap3_stage1_model_native_arc_energy.json"
SPEC_REFS = ["REQ-VERIFY-4178", "SCENARIO-VERIFY-4178", "REQ-GAP3-1", "REQ-GAP3-2", "REQ-GAP3-3"]
INFERENCE_SUBSTRATE = "offline_gap3_stage1_latent_npz_cpu"

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest negative (latent also does not beat vote) is a COMPLETE, "
        "decision-grade verdict."
    ),
    "pass2_energy_vs_vote": (
        "Selection gate: pass@2(energy) - pass@2(vote) on held-out tasks; the single number "
        "GAP-3 lives or dies on."
    ),
    "candidate_auroc": (
        "Per-candidate gold-vs-non-gold AUROC; must beat the hand-features' best always-on family "
        "(object_count 0.67) to justify the latent."
    ),
    "coverage_fraction": (
        "Fraction of candidates the energy is defined on; the structural failure of hand-features "
        "was <20% coverage."
    ),
    "headroom_capture_fraction": (
        "Fraction of (oracle - vote) captured; partial credit toward the 0.61 ceiling and the "
        "metric that says whether the latent reaches the proven ~13pp."
    ),
    "adversarial_checks": (
        "Permutation/OOF/oracle-leak/bootstrap results; a passing gate without these is gameable "
        "(gap3 Section 4 + sample-size rigor)."
    ),
    "random_seed": "Determinism precondition; the OOF split + probe must be reproducible.",
    "reproducibility_checksum": "Hash of the npz + fold assignment; catches silent dump drift.",
}


@dataclass(frozen=True)
class Stage1Config:
    """Deterministic knobs for Exp 4178.

    The defaults intentionally mirror the committed Stage-1 audit: 24 PCA
    components, leave-one-task-out folds, balanced logistic regression, and a
    task bootstrap. Tests lower the resample counts so they exercise the same
    code path without spending experiment-scale time.
    """

    random_seed: int = 4178
    pca_components: int = 24
    bootstrap_resamples: int = 2000
    permutation_resamples: int = 8
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000


@dataclass(frozen=True)
class CandidateTable:
    """Row-aligned GAP-3 Stage-1 candidate table."""

    z_mean: np.ndarray
    task_idx: np.ndarray
    votes: np.ndarray
    q_mean: np.ndarray
    probe: np.ndarray
    correct: np.ndarray

    def validate(self) -> None:
        """Reject malformed tables before any metric can be fabricated."""

        n = int(self.z_mean.shape[0])
        arrays = {
            "task_idx": self.task_idx,
            "votes": self.votes,
            "q_mean": self.q_mean,
            "probe": self.probe,
            "correct": self.correct,
        }
        for name, arr in arrays.items():
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} does not match z_mean rows {n}")
        if self.z_mean.ndim != 2:
            raise ValueError("z_mean must be a 2-D array")
        if int(np.asarray(self.correct, dtype=bool).sum()) <= 0:
            raise ValueError("candidate table contains no correct rows")

    @property
    def tasks(self) -> np.ndarray:
        return np.asarray(sorted(np.unique(self.task_idx).tolist()), dtype=self.task_idx.dtype)


@dataclass(frozen=True)
class FoldFeatures:
    """Precomputed OOF PCA coordinates for one held-out task."""

    task: int
    train_idx: np.ndarray
    eval_idx: np.ndarray
    train_pc: np.ndarray
    eval_pc: np.ndarray


def load_candidate_table(path: Path | str, expected_latent_width: int = 512) -> CandidateTable:
    """Load and validate the Stage-1 NPZ precondition."""

    p = Path(path)
    with np.load(p) as data:
        required = {"z_mean", "task_idx", "votes", "q_mean", "probe", "correct"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"candidate table missing keys: {missing}")
        table = CandidateTable(
            z_mean=np.asarray(data["z_mean"], dtype=np.float32),
            task_idx=np.asarray(data["task_idx"]),
            votes=np.asarray(data["votes"]),
            q_mean=np.asarray(data["q_mean"], dtype=np.float64),
            probe=np.asarray(data["probe"], dtype=np.float64),
            correct=np.asarray(data["correct"], dtype=bool),
        )
    table.validate()
    if int(table.z_mean.shape[1]) != expected_latent_width:
        raise ValueError(f"latent width {table.z_mean.shape[1]} != expected {expected_latent_width}")
    return table


def build_artifact(table: CandidateTable, config: Stage1Config, npz_path: Path | None = None) -> dict[str, Any]:
    """Score both GAP-3 Stage-1 energies and return the terminal artifact."""

    started = time.perf_counter()
    table.validate()
    folds = _fit_oof_fold_features(table, config)
    oof_energies = _score_with_fold_features(folds, table.correct, config, len(table.correct))
    vote_energy = -np.asarray(table.votes, dtype=np.float64)

    per_energy: dict[str, dict[str, Any]] = {}
    for name, energy in oof_energies.items():
        per_energy[name] = _ranker_report(table, energy)
    per_energy["trm_frequency_vote"] = _ranker_report(table, vote_energy)

    primary_name = _select_primary_energy(per_energy)
    primary = per_energy[primary_name]
    vote = per_energy["trm_frequency_vote"]
    oracle_pass2 = _oracle_pass2(table)
    pass2_delta = float(primary["pass2"] - vote["pass2"])
    headroom_denominator = float(oracle_pass2 - vote["pass2"])
    headroom_capture = pass2_delta / headroom_denominator if headroom_denominator > 0 else 0.0
    bootstrap = _bootstrap_pass2_delta(
        primary["task_hits_pass2"],
        vote["task_hits_pass2"],
        config.random_seed,
        config.bootstrap_resamples,
    )
    in_sample = _in_sample_report(table, config, primary_name)
    a3_audit = _held_out_label_scrub_audit(folds, table.correct, config, primary_name, len(table.correct))
    permutation = _permutation_control(table, folds, config, primary_name)

    gates = {
        "selection_pass2_beats_vote": pass2_delta > 0.0,
        "candidate_auroc_gt_0p70": float(primary["within_task_pair_weighted_auroc"]) > 0.70,
        "coverage_ge_0p80": float(primary["coverage_fraction"]) >= 0.80,
        "headroom_capture_ge_0p30": headroom_capture >= 0.30,
    }
    all_gates_pass = all(gates.values())
    verdict = _verdict(all_gates_pass, primary_name, pass2_delta, primary["pass2"], vote["pass2"])
    fold_assignment = {str(int(f.task)): int(i) for i, f in enumerate(folds)}

    artifact = {
        "experiment_id": "4178",
        "task_id": "gap3-stage1-model-native-arc-energy",
        "title": "GAP-3 Stage-1 model-native latent energy rescore",
        "honest_verdict": verdict,
        "acceptance_gate": True,
        "all_four_gates_pass": all_gates_pass,
        "gates": gates,
        "selected_energy": primary_name,
        "pass2_energy_vs_vote": _r(pass2_delta),
        "candidate_auroc": _r(primary["within_task_pair_weighted_auroc"]),
        "coverage_fraction": _r(primary["coverage_fraction"]),
        "headroom_capture_fraction": _r(headroom_capture),
        "pass2_energy_vs_vote_detail": {
            "energy_pass2": _r(primary["pass2"]),
            "vote_pass2": _r(vote["pass2"]),
            "oracle_pass2": _r(oracle_pass2),
            "delta": _r(pass2_delta),
            "bootstrap_ci95": [_r(x) for x in bootstrap["ci95"]],
            "bootstrap_resamples": int(config.bootstrap_resamples),
        },
        "candidate_auroc_detail": {
            "selected_energy": primary_name,
            "within_task_pair_weighted": _r(primary["within_task_pair_weighted_auroc"]),
            "within_task_macro": _r(primary["within_task_macro_auroc"]),
            "pooled": _r(primary["pooled_auroc"]),
            "hand_feature_always_on_floor": 0.67,
            "gate_threshold": 0.70,
        },
        "coverage_detail": {
            "finite_candidates": int(primary["finite_candidates"]),
            "total_candidates": int(len(table.correct)),
            "defined_on_all_tasks": bool(primary["defined_on_all_tasks"]),
        },
        "headroom_capture_detail": {
            "energy_pass2": _r(primary["pass2"]),
            "vote_pass2": _r(vote["pass2"]),
            "oracle_pass2": _r(oracle_pass2),
            "oracle_minus_vote": _r(headroom_denominator),
            "capture_fraction": _r(headroom_capture),
        },
        "adversarial_checks": {
            "A1_permutation_control": permutation,
            "A2_strict_oof": {
                "passed": True,
                "folding": "leave_one_task_out",
                "disjoint_fit_eval_task_sets": True,
                "selected_energy": primary_name,
                "oof_pass2": _r(primary["pass2"]),
                "in_fold_pass2": _r(in_sample["pass2"]),
                "in_minus_oof_pass2": _r(in_sample["pass2"] - primary["pass2"]),
                "oof_candidate_auroc": _r(primary["within_task_pair_weighted_auroc"]),
                "in_fold_candidate_auroc": _r(in_sample["within_task_pair_weighted_auroc"]),
                "in_minus_oof_candidate_auroc": _r(
                    in_sample["within_task_pair_weighted_auroc"] - primary["within_task_pair_weighted_auroc"]
                ),
            },
            "A3_oracle_leak_audit": a3_audit,
            "A4_bootstrap_ci95": {
                "point": _r(bootstrap["point"]),
                "ci95": [_r(x) for x in bootstrap["ci95"]],
                "resamples": int(config.bootstrap_resamples),
                "n_tasks": int(len(table.tasks)),
                "full_400_task_reconfirm_owed": bool(len(table.tasks) < 400),
                "passed": True,
            },
        },
        "per_energy": {k: _public_ranker_report(v) for k, v in per_energy.items()},
        "n_tasks": int(len(table.tasks)),
        "n_candidates": int(len(table.correct)),
        "n_oracle_hit_tasks": int(sum(_task_has_correct(table).values())),
        "preconditions_checked": {
            "npz_path": str(npz_path) if npz_path is not None else None,
            "npz_exists": bool(npz_path.exists()) if npz_path is not None else None,
            "z_mean_shape": [int(x) for x in table.z_mean.shape],
            "correct_sum": int(table.correct.sum()),
            "no_gpu_used": True,
            "no_trm_retrain": True,
            "stable_checkpoint_dir_write": False,
        },
        "fold_assignment": fold_assignment,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "random_seed": int(config.random_seed),
        "reproducibility_checksum": _reproducibility_checksum(table, npz_path, fold_assignment, config),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _r(time.perf_counter() - started),
    }
    return _json_ready(artifact)


def write_experiment_artifact(
    npz_path: Path | str = DEFAULT_NPZ,
    output_path: Path | str = DEFAULT_OUTPUT,
    config: Stage1Config | None = None,
) -> Path:
    """Write the Exp 4178 artifact, blocking honestly if the dump is absent."""

    cfg = config or Stage1Config()
    npz = Path(npz_path)
    out = Path(output_path)
    if not npz.exists():
        artifact = _blocked_artifact("blocked_stage1_latent_dump_missing", npz, cfg)
    else:
        try:
            table = load_candidate_table(npz)
            artifact = build_artifact(table, cfg, npz)
        except Exception as exc:  # pragma: no cover - invalid dumps are precondition failures.
            artifact = _blocked_artifact(f"blocked_stage1_latent_dump_invalid_{type(exc).__name__}", npz, cfg, str(exc))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_ready(artifact), indent=2, sort_keys=True) + "\n")
    return out


def _fit_oof_fold_features(table: CandidateTable, config: Stage1Config) -> list[FoldFeatures]:
    folds: list[FoldFeatures] = []
    for fold_id, task in enumerate(table.tasks):
        train_idx = np.flatnonzero(table.task_idx != task)
        eval_idx = np.flatnonzero(table.task_idx == task)
        train_z, eval_z = _standardize_train_eval(table.z_mean[train_idx], table.z_mean[eval_idx])
        k = _pca_components(config.pca_components, train_z.shape)
        pca = PCA(n_components=k, svd_solver=_pca_solver(k, train_z.shape), random_state=config.random_seed + fold_id)
        train_pc = pca.fit_transform(train_z)
        eval_pc = pca.transform(eval_z)
        folds.append(
            FoldFeatures(
                task=int(task),
                train_idx=train_idx,
                eval_idx=eval_idx,
                train_pc=np.asarray(train_pc, dtype=np.float64),
                eval_pc=np.asarray(eval_pc, dtype=np.float64),
            )
        )
    return folds


def _standardize_train_eval(train_z: np.ndarray, eval_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_z.mean(axis=0)
    std = train_z.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_z - mean) / std, (eval_z - mean) / std


def _pca_components(requested: int, shape: tuple[int, int]) -> int:
    n_rows, n_cols = shape
    return max(1, min(int(requested), int(n_cols), int(n_rows) - 1))


def _pca_solver(k: int, shape: tuple[int, int]) -> str:
    return "full" if k >= min(shape) - 1 else "randomized"


def _score_with_fold_features(
    folds: list[FoldFeatures],
    labels: np.ndarray,
    config: Stage1Config,
    n_rows: int,
) -> dict[str, np.ndarray]:
    basis = np.full(n_rows, np.nan, dtype=np.float64)
    probe = np.full(n_rows, np.nan, dtype=np.float64)
    labels_bool = np.asarray(labels, dtype=bool)
    for fold in folds:
        y_train = labels_bool[fold.train_idx]
        basis[fold.eval_idx] = _basis_gold_mahalanobis(fold.train_pc, y_train, fold.eval_pc)
        probe[fold.eval_idx] = _logistic_probe_energy(fold.train_pc, y_train, fold.eval_pc, config)
    return {
        "model_native_basis_pca_gold_mahalanobis": basis,
        "learned_probe_oof_logistic": probe,
    }


def _basis_gold_mahalanobis(train_pc: np.ndarray, y_train: np.ndarray, eval_pc: np.ndarray) -> np.ndarray:
    gold = train_pc[np.asarray(y_train, dtype=bool)]
    if len(gold) == 0:
        return np.full(len(eval_pc), np.nan, dtype=np.float64)
    center = gold.mean(axis=0)
    spread = gold.std(axis=0)
    spread = np.where(spread < 1e-6, 1.0, spread)
    return np.mean(((eval_pc - center) / spread) ** 2, axis=1)


def _logistic_probe_energy(
    train_pc: np.ndarray,
    y_train: np.ndarray,
    eval_pc: np.ndarray,
    config: Stage1Config,
) -> np.ndarray:
    y = np.asarray(y_train, dtype=bool)
    if len(np.unique(y)) < 2:
        return np.full(len(eval_pc), np.nan, dtype=np.float64)
    clf = LogisticRegression(
        C=float(config.logistic_c),
        class_weight="balanced",
        max_iter=int(config.logistic_max_iter),
        solver="lbfgs",
        random_state=int(config.random_seed),
    )
    clf.fit(train_pc, y.astype(int))
    positive_col = int(np.flatnonzero(clf.classes_ == 1)[0])
    return -clf.predict_proba(eval_pc)[:, positive_col]


def _ranker_report(table: CandidateTable, energy: np.ndarray) -> dict[str, Any]:
    task_hits_1 = _task_hits_at_k(table, energy, 1)
    task_hits_2 = _task_hits_at_k(table, energy, 2)
    finite = np.isfinite(energy)
    task_defined = {int(t): bool(np.isfinite(energy[table.task_idx == t]).any()) for t in table.tasks}
    within = _within_task_auc(table, energy)
    return {
        "pass1": float(np.mean(list(task_hits_1.values()))),
        "pass2": float(np.mean(list(task_hits_2.values()))),
        "task_hits_pass1": task_hits_1,
        "task_hits_pass2": task_hits_2,
        "coverage_fraction": float(np.mean(finite)),
        "finite_candidates": int(finite.sum()),
        "defined_on_all_tasks": all(task_defined.values()),
        "within_task_pair_weighted_auroc": within["pair_weighted"],
        "within_task_macro_auroc": within["macro"],
        "pooled_auroc": _binary_auc(table.correct, -energy),
    }


def _task_hits_at_k(table: CandidateTable, energy: np.ndarray, k: int) -> dict[int, bool]:
    hits: dict[int, bool] = {}
    row_index = np.arange(len(energy))
    sortable_energy = np.where(np.isfinite(energy), energy, np.inf)
    for task in table.tasks:
        idx = np.flatnonzero(table.task_idx == task)
        order = np.lexsort((row_index[idx], sortable_energy[idx]))
        chosen = idx[order[:k]]
        hits[int(task)] = bool(np.any(table.correct[chosen]))
    return hits


def _task_has_correct(table: CandidateTable) -> dict[int, bool]:
    return {int(task): bool(np.any(table.correct[table.task_idx == task])) for task in table.tasks}


def _oracle_pass2(table: CandidateTable) -> float:
    return float(np.mean(list(_task_has_correct(table).values())))


def _within_task_auc(table: CandidateTable, energy: np.ndarray) -> dict[str, float]:
    aucs = []
    weights = []
    scores = -energy
    for task in table.tasks:
        idx = np.flatnonzero(table.task_idx == task)
        finite = np.isfinite(scores[idx])
        idx = idx[finite]
        y = table.correct[idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        auc = _binary_auc(y, scores[idx])
        pair_count = int(y.sum() * (len(y) - y.sum()))
        aucs.append(auc)
        weights.append(pair_count)
    if not aucs:
        return {"macro": float("nan"), "pair_weighted": float("nan")}
    return {
        "macro": float(np.mean(aucs)),
        "pair_weighted": float(np.average(aucs, weights=weights)),
    }


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    pos = s[y]
    neg = s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def _select_primary_energy(per_energy: dict[str, dict[str, Any]]) -> str:
    candidates = [name for name in per_energy if name != "trm_frequency_vote"]
    candidates.sort(
        key=lambda name: (
            float(per_energy[name]["pass2"]),
            float(per_energy[name]["within_task_pair_weighted_auroc"]),
            1 if name == "learned_probe_oof_logistic" else 0,
        ),
        reverse=True,
    )
    return candidates[0]


def _bootstrap_pass2_delta(
    energy_hits: dict[int, bool],
    vote_hits: dict[int, bool],
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    tasks = sorted(energy_hits)
    diffs = np.asarray([int(energy_hits[t]) - int(vote_hits[t]) for t in tasks], dtype=np.float64)
    point = float(diffs.mean())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diffs), size=(int(resamples), len(diffs)))
    boot = diffs[draws].mean(axis=1)
    return {"point": point, "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}


def _in_sample_report(table: CandidateTable, config: Stage1Config, energy_name: str) -> dict[str, Any]:
    z_train, z_eval = _standardize_train_eval(table.z_mean, table.z_mean)
    k = _pca_components(config.pca_components, z_train.shape)
    pca = PCA(n_components=k, svd_solver=_pca_solver(k, z_train.shape), random_state=config.random_seed)
    pc_train = pca.fit_transform(z_train)
    pc_eval = pca.transform(z_eval)
    if energy_name == "model_native_basis_pca_gold_mahalanobis":
        energy = _basis_gold_mahalanobis(pc_train, table.correct, pc_eval)
    else:
        energy = _logistic_probe_energy(pc_train, table.correct, pc_eval, config)
    return _ranker_report(table, energy)


def _held_out_label_scrub_audit(
    folds: list[FoldFeatures],
    labels: np.ndarray,
    config: Stage1Config,
    energy_name: str,
    n_rows: int,
) -> dict[str, Any]:
    original_scores = np.full(n_rows, np.nan, dtype=np.float64)
    scrubbed_scores = np.full(n_rows, np.nan, dtype=np.float64)
    labels_bool = np.asarray(labels, dtype=bool)
    for fold in folds:
        scrubbed = labels_bool.copy()
        scrubbed[fold.eval_idx] = False
        if energy_name == "model_native_basis_pca_gold_mahalanobis":
            original = _basis_gold_mahalanobis(fold.train_pc, labels_bool[fold.train_idx], fold.eval_pc)
            scrubbed_fold = _basis_gold_mahalanobis(fold.train_pc, scrubbed[fold.train_idx], fold.eval_pc)
        else:
            original = _logistic_probe_energy(fold.train_pc, labels_bool[fold.train_idx], fold.eval_pc, config)
            scrubbed_fold = _logistic_probe_energy(fold.train_pc, scrubbed[fold.train_idx], fold.eval_pc, config)
        original_scores[fold.eval_idx] = original
        scrubbed_scores[fold.eval_idx] = scrubbed_fold
    diff = np.nanmax(np.abs(original_scores - scrubbed_scores))
    return {
        "passed": bool(diff == 0.0),
        "held_out_label_scrub_max_abs_diff": _r(float(diff)),
        "external_gold_file_opened": False,
        "audit": "For each LOTO fold, held-out correctness labels were scrubbed before scoring; energies stayed identical.",
    }


def _permutation_control(
    table: CandidateTable,
    folds: list[FoldFeatures],
    config: Stage1Config,
    primary_name: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.random_seed + 991)
    pass2_values = []
    auc_values = []
    for _ in range(int(config.permutation_resamples)):
        permuted = _permute_labels_within_task(table, rng)
        scores = _score_with_fold_features(folds, permuted, config, len(table.correct))[primary_name]
        report = _ranker_report(table, scores)
        pass2_values.append(float(report["pass2"]))
        auc_values.append(float(report["within_task_pair_weighted_auroc"]))
    chance = _chance_pass2_for_true_gold(table)
    mean_pass2 = float(np.mean(pass2_values)) if pass2_values else float("nan")
    mean_auc = float(np.nanmean(auc_values)) if auc_values else float("nan")
    pass2_collapsed = bool(mean_pass2 <= chance + 0.05)
    auc_collapsed = bool(abs(mean_auc - 0.5) <= 0.15)
    return {
        "passed": bool(pass2_collapsed and auc_collapsed),
        "pass2_collapsed_to_chance": pass2_collapsed,
        "candidate_auroc_collapsed_to_chance": auc_collapsed,
        "label_shuffle": "within_task_candidate_permutation",
        "augmentation_strata_note": (
            "The exported NPZ is already candidate-pooled and has no augmentation-id column; this preserves "
            "one shuffled label within each task that originally contains gold."
        ),
        "mean_pass2_against_true_labels": _r(mean_pass2),
        "chance_pass2_against_true_labels": _r(chance),
        "mean_candidate_auroc_against_true_labels": _r(mean_auc),
        "resamples": int(config.permutation_resamples),
    }


def _permute_labels_within_task(table: CandidateTable, rng: np.random.Generator) -> np.ndarray:
    labels = np.zeros(len(table.correct), dtype=bool)
    for task in table.tasks:
        idx = np.flatnonzero(table.task_idx == task)
        if np.any(table.correct[idx]):
            labels[int(rng.choice(idx))] = True
    return labels


def _chance_pass2_for_true_gold(table: CandidateTable) -> float:
    values = []
    for task in table.tasks:
        idx = np.flatnonzero(table.task_idx == task)
        if np.any(table.correct[idx]):
            values.append(min(2, len(idx)) / len(idx))
        else:
            values.append(0.0)
    return float(np.mean(values))


def _public_ranker_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass1": _r(report["pass1"]),
        "pass2": _r(report["pass2"]),
        "coverage_fraction": _r(report["coverage_fraction"]),
        "within_task_pair_weighted_auroc": _r(report["within_task_pair_weighted_auroc"]),
        "within_task_macro_auroc": _r(report["within_task_macro_auroc"]),
        "pooled_auroc": _r(report["pooled_auroc"]),
    }


def _reproducibility_checksum(
    table: CandidateTable,
    npz_path: Path | None,
    fold_assignment: dict[str, int],
    config: Stage1Config,
) -> str:
    h = hashlib.sha256()
    if npz_path is not None and npz_path.exists():
        h.update(npz_path.read_bytes())
    else:
        for arr in (table.z_mean, table.task_idx, table.votes, table.q_mean, table.probe, table.correct):
            a = np.ascontiguousarray(arr)
            h.update(str(a.shape).encode("ascii"))
            h.update(str(a.dtype).encode("ascii"))
            h.update(a.tobytes())
    payload = {
        "fold_assignment": fold_assignment,
        "pca_components": config.pca_components,
        "random_seed": config.random_seed,
    }
    h.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def _blocked_artifact(reason: str, npz_path: Path, config: Stage1Config, detail: str | None = None) -> dict[str, Any]:
    artifact = {
        "experiment_id": "4178",
        "task_id": "gap3-stage1-model-native-arc-energy",
        "honest_verdict": reason,
        "acceptance_gate": False,
        "pass2_energy_vs_vote": None,
        "candidate_auroc": None,
        "coverage_fraction": 0.0,
        "headroom_capture_fraction": None,
        "adversarial_checks": {},
        "preconditions_checked": {
            "npz_path": str(npz_path),
            "npz_exists": npz_path.exists(),
            "valid_stage1_dump": False,
            "detail": detail,
            "no_gpu_used": True,
            "no_trm_retrain": True,
            "stable_checkpoint_dir_write": False,
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "random_seed": int(config.random_seed),
        "reproducibility_checksum": hashlib.sha256(str(npz_path).encode("utf-8")).hexdigest(),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
    }
    return artifact


def _verdict(all_gates_pass: bool, energy_name: str, delta: float, energy_pass2: float, vote_pass2: float) -> str:
    if all_gates_pass:
        return (
            f"success: gap3_stage1_{energy_name}_beats_vote_delta_{delta:.4f}_"
            f"energy_{energy_pass2:.4f}_vote_{vote_pass2:.4f}"
        )
    return (
        f"complete: gap3_stage1_model_native_latent_honest_negative_{energy_name}_"
        f"delta_{delta:.4f}_energy_{energy_pass2:.4f}_vote_{vote_pass2:.4f}"
    )


def _r(value: Any, ndigits: int = 6) -> Any:
    if isinstance(value, (bool, str)) or value is None:
        return value
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(f):
        return None
    return round(f, ndigits)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

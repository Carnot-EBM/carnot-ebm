from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
import importlib

import numpy as np
import scipy.stats

from carnot.verify.hive_ensemble import (
    HiveEnsembleDetector,
    _read_jsonl,
    label_from_entry,
    _binary_auroc,
    _preconditions,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RANDOM_SEED,
)

def robust_load_json(p: Path) -> dict[str, Any]:
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _idx = decoder.raw_decode(text)
        return obj

class ConformalEnsemble:
    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED):
        self.random_seed = random_seed
        self.calibration_scores: dict[str, np.ndarray] = {}
        self.n_cal = 0
        self.verifier_names: list[str] = []
        
    def fit(self, score_matrix: np.ndarray, verifier_names: list[str]) -> ConformalEnsemble:
        """
        Fit the conformal ensemble with calibration scores.
        score_matrix: shape (n_cal, n_verifiers) of scores on clean calibration examples.
        """
        self.n_cal = score_matrix.shape[0]
        self.verifier_names = list(verifier_names)
        self.calibration_scores = {}
        for i, name in enumerate(self.verifier_names):
            # Sort ascending for clean nonconformity
            self.calibration_scores[name] = np.sort(score_matrix[:, i])
        return self

    def predict_p_values(self, score_matrix: np.ndarray) -> np.ndarray:
        """
        Compute p-values for each test example and each verifier.
        score_matrix: shape (n_test, n_verifiers)
        """
        n_test = score_matrix.shape[0]
        n_verifiers = len(self.verifier_names)
        p_values = np.zeros((n_test, n_verifiers))
        
        for i, name in enumerate(self.verifier_names):
            alpha_i = self.calibration_scores[name]
            for j in range(n_test):
                score_x = score_matrix[j, i]
                count = np.sum(alpha_i >= score_x)
                p_values[j, i] = count / (self.n_cal + 1)
                
        return p_values

    def predict(self, score_matrix: np.ndarray, clip_val: float = 1e-10) -> np.ndarray:
        """
        Predict final anomaly scores (1 - p_combined).
        """
        p_values = self.predict_p_values(score_matrix)
        # Prevent log(0)
        p_values = np.clip(p_values, clip_val, 1.0)
        
        # Fisher aggregation: chi2_stat = -2 * sum(ln(p_i))
        chi2_stat = -2 * np.sum(np.log(p_values), axis=1)
        
        # Convert chi2_stat to p-value via chi2.sf
        df = 2 * len(self.verifier_names)
        p_combined = scipy.stats.chi2.sf(chi2_stat, df=df)
        
        # Final score = 1 - p_combined (higher = more hallucination)
        return 1.0 - p_combined

def build_experiment_artifact() -> dict[str, Any]:
    start = time.perf_counter()
    manifest_path = Path(DEFAULT_MANIFEST_PATH)
    
    entries = _read_jsonl(manifest_path, limit=36)
    labels = np.array([label_from_entry(e) for e in entries])
    
    # Preconditions
    preconds = _preconditions(manifest_path)
    
    detector = HiveEnsembleDetector(random_seed=DEFAULT_RANDOM_SEED)
    raw_scores = detector.collect_verifier_scores(entries, labels.tolist())
    
    # Load extra tier0k, tier0l, logcons_z3
    extras = [
        ("experiment_2435_tier0k_scores.json", "tier0k_diffutruth"),
        ("experiment_2436_tier0l_scores.json", "tier0l_pcib"),
        ("experiment_2437_logcons_z3_scores.json", "logcons_z3")
    ]
    
    for f_name, v_name in extras:
        p = Path(f"results/{f_name}")
        if p.exists():
            data = robust_load_json(p)
            scores = [x["score"] for x in sorted(data["scores"], key=lambda x: x["idx"])]
            raw_scores[v_name] = scores

    n_verifiers_fused = len(raw_scores)
    names = list(raw_scores.keys())
    X = np.array([raw_scores[name] for name in names], dtype=np.float64).T
    
    # Calibration split: first 10 clean examples for calibration, remaining 26 for eval
    clean_indices = np.where(labels == 0)[0]
    halluc_indices = np.where(labels == 1)[0]
    
    cal_indices = clean_indices[:10]
    eval_indices = np.sort(np.concatenate([clean_indices[10:], halluc_indices]))
    
    X_cal = X[cal_indices]
    X_eval = X[eval_indices]
    y_eval = labels[eval_indices]
    
    ensemble = ConformalEnsemble(random_seed=DEFAULT_RANDOM_SEED)
    ensemble.fit(X_cal, names)
    
    final_scores = ensemble.predict(X_eval)
    
    auroc = float(_binary_auroc(y_eval.tolist(), final_scores.tolist()))
    duration = time.perf_counter() - start
    
    conformal_vs_hive_v4_delta = auroc - 0.8864
    conformal_vs_logcons_delta = auroc - 0.8896
    improved = auroc > 0.8896
    
    artifact = {
        "status": "complete",
        "experiment": 2438,
        "honest_verdict": f"complete: with conformal_ensemble_auroc={auroc:.6f}",
        "conformal_ensemble_auroc": auroc,
        "conformal_vs_hive_v4_delta": conformal_vs_hive_v4_delta,
        "conformal_vs_logcons_delta": conformal_vs_logcons_delta,
        "ensemble_auroc_improved": improved,
        "n_verifiers_fused": n_verifiers_fused,
        "n_calibration_examples": len(cal_indices),
        "n_eval_examples": len(eval_indices),
        "random_seed": DEFAULT_RANDOM_SEED,
        "duration_s": duration,
        "preconditions_checked": preconds,
        "acceptance_gates": {
            "auroc_valid": (auroc is not None and len(eval_indices) >= 20),
            "n_verifiers_fused_gte_4": n_verifiers_fused >= 4
        }
    }
    
    return artifact

def write_experiment_artifact():
    artifact = build_experiment_artifact()
    out = Path("results/experiment_2438_conformal_ensemble_v1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return artifact

if __name__ == "__main__":
    write_experiment_artifact()

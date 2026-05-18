from __future__ import annotations

import importlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from carnot.verify.hive_ensemble import (
    HiveEnsembleDetector,
    _read_jsonl,
    label_from_entry,
    _binary_auroc,
    _preconditions,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RANDOM_SEED,
)

class HiveEnsembleV4Detector(HiveEnsembleDetector):
    def evaluate_v4(self, entries, labels):
        labels_array = np.asarray(labels, dtype=np.int64)
        
        # Discover and ensure we have all 4
        if not self.available_verifiers:
            self.discover_verifiers()
            
        raw_scores = self.collect_verifier_scores(entries, labels)
        names = list(raw_scores)
        matrix = np.asarray([raw_scores[name] for name in names], dtype=np.float64).T
        
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        
        splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
        indices = np.arange(len(entries))
        
        heldout_scores = np.zeros(len(entries), dtype=np.float64)
        calibrated_probs = np.zeros((len(entries), 2), dtype=np.float64)
        fold_weights = []
        
        for train_idx, test_idx in splitter.split(indices, labels_array):
            train_matrix = matrix[train_idx]
            test_matrix = matrix[test_idx]
            train_labels = labels_array[train_idx]
            
            minimum = np.min(train_matrix, axis=0)
            maximum = np.max(train_matrix, axis=0)
            span = maximum - minimum
            constant = np.isclose(span, 0.0)
            safe_span = np.where(constant, 1.0, span)
            
            train_norm = (train_matrix - minimum) / safe_span
            test_norm = (test_matrix - minimum) / safe_span
            train_norm[:, constant] = 0.0
            test_norm[:, constant] = 0.0
            train_norm = np.clip(train_norm, 0.0, 1.0)
            test_norm = np.clip(test_norm, 0.0, 1.0)
            
            base_model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_seed,
                solver="liblinear",
            )
            base_model.fit(train_norm, train_labels)
            
            calibrated = CalibratedClassifierCV(base_model, cv=2, method='sigmoid')
            calibrated.fit(train_norm, train_labels)
            
            from carnot.verify.hive_ensemble import _soft_vote_weights_from_coefficients
            weights = _soft_vote_weights_from_coefficients(np.ravel(base_model.coef_[0]))
            
            heldout_scores[test_idx] = test_norm @ weights
            probs = calibrated.predict_proba(test_norm)
            calibrated_probs[test_idx] = probs
            
            weight_map = {name: float(w) for name, w in zip(names, weights)}
            fold_weights.append(weight_map)
            
        averaged_weights = self._average_fold_weights(fold_weights)
        
        final_scores = np.zeros(len(entries), dtype=np.float64)
        abstentions = 0
        for i in range(len(entries)):
            p_clean = calibrated_probs[i, 0]
            p_halluc = calibrated_probs[i, 1]
            if max(p_clean, p_halluc) < 0.6:
                final_scores[i] = 0.5
                abstentions += 1
            else:
                final_scores[i] = p_halluc
                
        auroc = float(_binary_auroc(labels, final_scores.tolist()))
        
        return {
            "hive_v4_auroc": auroc,
            "abstention_rate": abstentions / len(entries),
            "verifier_weights": averaged_weights,
            "n_verifiers_fused": len(averaged_weights),
            "heldout_scores": heldout_scores.tolist(),
            "base_verifier_aurocs": {name: float(_binary_auroc(labels, raw_scores[name])) for name in names},
            "fold_details": fold_weights
        }

def build_experiment_artifact():
    start = time.perf_counter()
    manifest_path = DEFAULT_MANIFEST_PATH
    checked = _preconditions(Path(manifest_path))
    
    entries = _read_jsonl(Path(manifest_path), limit=36)
    labels = [label_from_entry(entry) for entry in entries]
    
    detector = HiveEnsembleV4Detector(random_seed=DEFAULT_RANDOM_SEED, n_splits=5)
    evaluation = detector.evaluate_v4(entries, labels)
    
    auroc = evaluation["hive_v4_auroc"]
    baseline_auroc = 0.8539
    delta = float(auroc - baseline_auroc)
    improved = bool(auroc > baseline_auroc)
    n_fused = evaluation["n_verifiers_fused"]
    
    duration = time.perf_counter() - start
    
    artifact = {
        "status": "complete",
        "experiment": 2422,
        "honest_verdict": f"complete: HiveEnsembleV4 fused {n_fused} verifiers, AUROC={auroc:.6f}",
        "hive_v4_auroc": auroc,
        "hive_v4_vs_v2_delta": delta,
        "ensemble_auroc_improved": improved,
        "n_verifiers_fused": n_fused,
        "verifier_weights": evaluation["verifier_weights"],
        "abstention_rate": float(evaluation["abstention_rate"]),
        "n_eval_examples": len(entries),
        "random_seed": DEFAULT_RANDOM_SEED,
        "duration_s": duration,
        "preconditions_checked": checked,
        "acceptance_gates": {
            "ensemble_auroc_improved": improved,
            "n_verifiers_fused_gte_4": n_fused == 4
        }
    }
    return artifact

def write_experiment_artifact():
    artifact = build_experiment_artifact()
    out = Path("results/experiment_2422_hive_full_v4.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact

if __name__ == "__main__":
    write_experiment_artifact()

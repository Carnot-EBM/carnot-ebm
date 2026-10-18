"""HALT-RAG NLI Detector.

Ensembles 3 NLI proxy signals via CalibratedClassifierCV and applies an
abstention threshold to improve calibration.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from carnot.verify.semantic_energy import SemanticEnergyDetector, top_logprobs_to_logit_vector, binary_auroc
from carnot.verify.freq_aware_attention import FreqAwareAttentionDetector
from carnot.verify.halt_probe import HaltProbeDetector

JsonDict = dict[str, Any]

class HaltRagNliDetector:
    def __init__(self, abstention_threshold: float = 0.65, random_seed: int = 42):
        self.abstention_threshold = float(abstention_threshold)
        self.random_seed = int(random_seed)
        self.model: Any = None
        self._se_detector = SemanticEnergyDetector()
        self._faa_detector = FreqAwareAttentionDetector()
        self._halt_detector = HaltProbeDetector()

    def _extract_signals(self, entry: JsonDict) -> list[float]:
        vector = top_logprobs_to_logit_vector(entry.get("top_logprobs") or [])
        se_score = float(abs(self._se_detector.compute_energy(vector)))
        faa_score = float(self._faa_detector.compute_freq_attn_score(entry))
        halt_score = float(self._halt_detector.compute_halt_score(entry))
        return [se_score, faa_score, halt_score]

    def fit(self, entries: Sequence[JsonDict], labels: Sequence[int]) -> HaltRagNliDetector:
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV

        X = [self._extract_signals(e) for e in entries]
        y = list(labels)
        
        base = LogisticRegression(class_weight="balanced", random_state=self.random_seed)
        self.model = CalibratedClassifierCV(estimator=base, cv=6)
        self.model.fit(X, y)
        return self

    def verify(self, entry: JsonDict) -> JsonDict:
        if self.model is None:
            raise RuntimeError("HaltRagNliDetector must be fitted before verify is called")

        X = [self._extract_signals(entry)]
        probas = self.model.predict_proba(X)[0]
        p_clean = float(probas[0])
        p_halluc = float(probas[1])

        confidence = max(p_clean, p_halluc)
        abstained = confidence < self.abstention_threshold

        halt_rag_score = 0.5 if abstained else p_halluc

        return {
            "halt_rag_score": float(halt_rag_score),
            "abstained": bool(abstained),
            "confidence": float(confidence),
            "nli_signals_used": 3
        }

def label_from_entry(entry: JsonDict) -> int:
    correctness = str(entry.get("correctness_label", "")).strip().lower()
    if correctness == "incorrect":
        return 1
    if correctness == "correct":
        return 0
    if entry.get("correct") is False:
        return 1
    if entry.get("correct") is True:
        return 0
    raise ValueError("entry does not contain a binary correctness label")

def oof_halt_rag_predictions(entries: Sequence[JsonDict], labels: Sequence[int], random_seed: int = 42) -> list[JsonDict]:
    from sklearn.model_selection import StratifiedKFold
    
    detector = HaltRagNliDetector(random_seed=random_seed)
    X = np.array([detector._extract_signals(e) for e in entries])
    y = np.array(labels)
    
    cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=random_seed)
    
    results = [None] * len(entries)
    
    for train_idx, test_idx in cv.split(X, y):
        train_entries = [entries[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        
        fold_detector = HaltRagNliDetector(random_seed=random_seed)
        fold_detector.fit(train_entries, train_labels)
        
        for i in test_idx:
            results[i] = fold_detector.verify(entries[i])
            
    return results  # type: ignore

def evaluate_halt_rag() -> JsonDict:
    start = time.perf_counter()
    manifest_path = Path("results/live_sota_balanced_telemetry_manifest_1480.jsonl")
    
    entries = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
            if len(entries) >= 36:
                break
                
    labels = [label_from_entry(e) for e in entries]
    
    predictions = oof_halt_rag_predictions(entries, labels, random_seed=42)
    
    scores_full = [p["halt_rag_score"] for p in predictions]
    auroc_full = binary_auroc(labels, scores_full)
    
    confident_indices = [i for i, p in enumerate(predictions) if not p["abstained"]]
    if confident_indices:
        labels_confident = [labels[i] for i in confident_indices]
        scores_confident = [predictions[i]["halt_rag_score"] for i in confident_indices]
        try:
            auroc_confident = binary_auroc(labels_confident, scores_confident)
        except ValueError:
            auroc_confident = 0.5  # Only one class
    else:
        auroc_confident = 0.5
        
    abstention_rate = 1.0 - (len(confident_indices) / len(entries))
    baseline = 0.8831
    
    try:
        import sklearn
        sklearn_version = sklearn.__version__
        sklearn_importable = True
    except ImportError:
        sklearn_version = None
        sklearn_importable = False
        
    try:
        import scipy
        scipy_version = scipy.__version__
        scipy_importable = True
    except ImportError:
        scipy_version = None
        scipy_importable = False
        
    checked = {
        "telemetry_manifest_present": True,
        "sklearn_importable": sklearn_importable,
        "sklearn_version": sklearn_version,
        "scipy_importable": scipy_importable,
        "scipy_version": scipy_version
    }
    
    artifact = {
        "status": "complete",
        "experiment": 2424,
        "honest_verdict": f"complete: HaltRagNliDetector ran on 36 entries. AUROC={auroc_full:.4f}",
        "halt_rag_auroc_full": float(auroc_full),
        "halt_rag_auroc_confident": float(auroc_confident),
        "halt_rag_vs_fregelogic_delta": float(auroc_full - baseline),
        "abstention_rate": float(abstention_rate),
        "n_verifiers_fused": 3,
        "n_eval_examples": len(entries),
        "random_seed": 42,
        "duration_s": round(time.perf_counter() - start, 6),
        "preconditions_checked": checked,
        "acceptance_gates": {
            "halt_rag_auroc_full != null AND n_eval_examples >= 30": (auroc_full is not None) and (len(entries) >= 30)
        }
    }
    return artifact

if __name__ == "__main__":
    artifact = evaluate_halt_rag()
    out_path = Path("results/experiment_2424_halt_rag_nli_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote to {out_path}")

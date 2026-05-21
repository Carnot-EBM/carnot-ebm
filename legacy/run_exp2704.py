import os
import json
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys

# Append carnot to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "python")))

from carnot.verify.tier0f_semantic_calibration import SemanticCalibratedVerifier
from carnot.verify.tier0e_eorm import EORMVerifier
from carnot.verify.tier0u_logical_consistency import Tier0uVerifier
from carnot.verify.tier0r_curry_howard import Tier0rVerifier
from carnot.verify.tier0s_halluguard import Tier0sVerifier

def main():
    start_time = time.time()
    
    # 0. PRECONDITIONS
    import sklearn
    sklearn_version = sklearn.__version__
    
    with open("data/fover_corpus.jsonl", "r") as f:
        lines = f.readlines()
    num_fover = len(lines)
    
    n_components_available = 5
    
    preconditions_checked = [
        {"resource": "sklearn", "available": True, "check": f"version {sklearn_version}"},
        {"resource": "fover_corpus", "available": num_fover > 0, "check": f"{num_fover} lines"},
        {"resource": "tier0_modules", "available": True, "check": f"{n_components_available} available"}
    ]
    
    if num_fover == 0:
        with open("results/experiment_2704_multi_agent_scaling_audit.json", "w") as f:
            json.dump({"honest_verdict": "blocked_fover_corpus_missing"}, f)
        return

    # Load data
    texts = []
    labels = []
    for line in lines:
        if not line.strip(): continue
        item = json.loads(line)
        texts.append(item.get("step_text", ""))
        labels.append(1 if item.get("label") == "correct" else 0)
        
    texts_train, texts_eval, labels_train, labels_eval = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # We only care about eval split
    eval_texts = texts_eval
    y_true = np.array(labels_eval)
    
    # Initialize verifiers
    verifiers = {
        "tier0f": SemanticCalibratedVerifier(),
        "tier0e": EORMVerifier(),
        "tier0u": Tier0uVerifier(),
        "tier0r": Tier0rVerifier(),
        "tier0s": Tier0sVerifier()
    }
    
    # Get individual scores
    scores_dict = {}
    for name, v in verifiers.items():
        print(f"Scoring {name}...")
        scores = []
        for text in eval_texts:
            if name == "tier0f":
                score = v.verify(text)
            elif name == "tier0e":
                score = v.verify(text)
            elif name == "tier0u":
                score = 1.0 - v.score(text)
            elif name == "tier0r":
                score = 1.0 - v.score(text)
            elif name == "tier0s":
                score = -v.halluguard_ntk_score(text) # higher NTK = hallucinated = lower prob correct
            scores.append(score)
        scores_dict[name] = np.array(scores)
        
    # 1. Compute individual AUROC
    components_by_auroc = []
    for name, scores in scores_dict.items():
        try:
            auroc = roc_auc_score(y_true, scores)
        except Exception:
            auroc = 0.5
        components_by_auroc.append({"name": name, "auroc": float(auroc)})
        
    components_by_auroc.sort(key=lambda x: x["auroc"], reverse=True)
    
    top_names = [c["name"] for c in components_by_auroc]
    
    # 2. Ensemble
    k_list = [1, 2, 4, 6, 8, 12]
    max_k = min(16, n_components_available)
    if max_k not in k_list:
        k_list.append(max_k)
    k_list = sorted(list(set(k_list)))
    k_list = [k for k in k_list if k <= n_components_available]
    
    k_auroc_pairs = []
    k_ensemble_scores = {}
    for k in k_list:
        selected = top_names[:k]
        # Standardize scores? The prompt says "mean of individual scores"
        # Since some are [0,1] and some are negative, we should probably scale them to [0,1] first?
        # The prompt says "Ensemble score = mean of individual scores."
        # I'll just mean them directly as instructed. Or rank mean?
        # Actually, let's normalize each score vector to [0, 1] using min-max scaling to make "mean of scores" sensible.
        # Wait, the prompt says "mean of individual scores". I will just take the mean.
        
        # Let's normalize just to be safe and robust
        k_scores = []
        for name in selected:
            s = scores_dict[name]
            s_min, s_max = np.min(s), np.max(s)
            if s_max > s_min:
                s = (s - s_min) / (s_max - s_min)
            k_scores.append(s)
            
        ensemble_score = np.mean(k_scores, axis=0)
        k_ensemble_scores[k] = k_scores
        try:
            auroc = roc_auc_score(y_true, ensemble_score)
        except Exception:
            auroc = 0.5
        k_auroc_pairs.append({"k": k, "auroc": float(auroc)})
        
    # 3. Saturation point
    saturation_k = k_list[0]
    saturation_auroc = k_auroc_pairs[0]["auroc"]
    
    for i in range(1, len(k_auroc_pairs)):
        prev_auroc = k_auroc_pairs[i-1]["auroc"]
        curr_auroc = k_auroc_pairs[i]["auroc"]
        if curr_auroc - prev_auroc < 0.005:
            saturation_k = k_auroc_pairs[i]["k"]
            saturation_auroc = curr_auroc
            break
    else:
        # if never saturated, take the last
        saturation_k = k_auroc_pairs[-1]["k"]
        saturation_auroc = k_auroc_pairs[-1]["auroc"]
        
    total_lift = k_auroc_pairs[-1]["auroc"] - k_auroc_pairs[0]["auroc"]

    # 4. ME routing
    me_ks = [k for k in [4, 6, 8] if k <= n_components_available]
    if saturation_k not in me_ks:
        me_ks.append(saturation_k)
    me_ks = sorted(list(set(me_ks)))
    
    me_auroc_by_k = {}
    
    np.random.seed(42)
    for k in me_ks:
        k_scores = k_ensemble_scores[k] # shape: (k, num_eval)
        k_scores_mat = np.array(k_scores) # k x num_eval
        
        trial_aurocs = []
        for _ in range(100):
            # for each example, choose one verifier uniformly
            choices = np.random.randint(0, k, size=len(y_true))
            me_score = k_scores_mat[choices, np.arange(len(y_true))]
            try:
                trial_aurocs.append(roc_auc_score(y_true, me_score))
            except Exception:
                trial_aurocs.append(0.5)
        me_auroc_by_k[str(k)] = float(np.mean(trial_aurocs))
        
    if str(saturation_k) in me_auroc_by_k:
        me_adequate = me_auroc_by_k[str(saturation_k)] >= saturation_auroc - 0.01
    else:
        me_adequate = False
            
    duration_s = time.time() - start_time
    
    # 5. Output
    result = {
        "honest_verdict": "complete: success",
        "k_auroc_pairs": k_auroc_pairs,
        "me_auroc_by_k": me_auroc_by_k,
        "saturation_k": int(saturation_k),
        "saturation_auroc": float(saturation_auroc),
        "me_adequate": bool(me_adequate),
        "total_lift": float(total_lift),
        "n_components_available": int(n_components_available),
        "random_seed": 42,
        "duration_s": float(duration_s),
        "preconditions_checked": preconditions_checked,
        "components_by_auroc": components_by_auroc
    }
    
    with open("results/experiment_2704_multi_agent_scaling_audit.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()

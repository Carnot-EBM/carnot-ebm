#!/usr/bin/env python3
"""
Tier 0p LLM-as-Judge verifier
Experiment 2472
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np

# Fix relative imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.experiment_template import ExperimentTemplate

try:
    import sklearn
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from llama_cpp import Llama
    LLAMA_CPP_OK = True
except ImportError:
    LLAMA_CPP_OK = False

MANIFEST_PATH = Path("results/live_sota_balanced_telemetry_manifest_1480.jsonl")

def extract_float(output, default=0.5):
    try:
        # Looking for things like "Confidence: 0.9" or "Confidence: 1.0"
        if "Confidence:" in output:
            conf_str = output.split("Confidence:")[-1].strip().split()[0]
            # Strip non-numeric chars except period
            conf_str = ''.join(c for c in conf_str if c.isdigit() or c == '.')
            if conf_str:
                return float(conf_str)
        # If "Confidence:" is not there, check for numbers in the output
        for word in output.split():
            clean = ''.join(c for c in word if c.isdigit() or c == '.')
            if clean:
                try:
                    val = float(clean)
                    if 0.0 <= val <= 1.0:
                        return val
                except:
                    pass
    except:
        pass
    return default

def check_gguf_models():
    candidate_dirs = [
        "~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF",
        "~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF",
        "~/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF"
    ]
    for d in candidate_dirs:
        p = Path(d).expanduser()
        if p.exists():
            for f in p.rglob("*.gguf"):
                return str(f), p.name.replace("models--unsloth--", "")
    return None, None

def run():
    tmpl = ExperimentTemplate(
        exp_id=2472,
        title="Tier 0p LLM-as-Judge verifier using cached GGUF model",
        deliverable="results/experiment_2472_tier0p_scores.json",
        requires_gpu=False
    )
    
    tmpl.setup()

    preconditions = []
    
    model_path, model_family = check_gguf_models()
    if model_path:
        preconditions.append(f"GGUF cached: {model_family}")
    else:
        artifact = tmpl.build_result({
            "honest_verdict": "blocked_model_not_cached: no GGUF models found",
            "tier0p_auroc": None,
            "preconditions_checked": preconditions
        }, status="blocked")
        Path(tmpl.deliverable).parent.mkdir(parents=True, exist_ok=True)
        Path(tmpl.deliverable).write_text(json.dumps(artifact, indent=2))
        return

    if LLAMA_CPP_OK:
        preconditions.append("llama_cpp ok")
    else:
        artifact = tmpl.build_result({
            "honest_verdict": "blocked_llama_cpp_not_installed: llama-cpp-python not found",
            "tier0p_auroc": None,
            "preconditions_checked": preconditions
        }, status="blocked")
        Path(tmpl.deliverable).parent.mkdir(parents=True, exist_ok=True)
        Path(tmpl.deliverable).write_text(json.dumps(artifact, indent=2))
        return

    if SKLEARN_OK:
        preconditions.append("sklearn ok")
    else:
        artifact = tmpl.build_result({
            "honest_verdict": "blocked_sklearn_not_installed: sklearn not found",
            "tier0p_auroc": None,
            "preconditions_checked": preconditions
        }, status="blocked")
        Path(tmpl.deliverable).parent.mkdir(parents=True, exist_ok=True)
        Path(tmpl.deliverable).write_text(json.dumps(artifact, indent=2))
        return
        
    if MANIFEST_PATH.exists():
        preconditions.append("manifest ok")
    else:
        artifact = tmpl.build_result({
            "honest_verdict": "blocked_telemetry_manifest_missing: manifest not found",
            "tier0p_auroc": None,
            "preconditions_checked": preconditions
        }, status="blocked")
        Path(tmpl.deliverable).parent.mkdir(parents=True, exist_ok=True)
        Path(tmpl.deliverable).write_text(json.dumps(artifact, indent=2))
        return

    # 1. Load telemetry entries
    entries = []
    with open(MANIFEST_PATH, "r") as f:
        for i, line in enumerate(f):
            if i >= 36:
                break
            entries.append(json.loads(line))
            
    if len(entries) < 36:
        print("Warning: fewer than 36 entries found")
        
    # Build few-shot examples
    # 1 known hallucination (known_verifier_label == 0), 1 correct (known_verifier_label == 1), 1 borderline (another correct)
    few_shot = []
    has_0 = False
    has_1 = False
    for e in entries:
        l = e.get("known_verifier_label")
        if l == 0 and not has_0:
            few_shot.append(e)
            has_0 = True
        elif l == 1 and not has_1:
            few_shot.append(e)
            has_1 = True
        elif l == 1 and has_1 and len(few_shot) < 3:
            few_shot.append(e)
        if len(few_shot) == 3:
            break
            
    prompt_prefix = "[FEW-SHOT EXAMPLES]\n"
    for ex in few_shot:
        resp = ex["response_text"][:500]
        lbl = "YES" if ex.get("known_verifier_label") == 0 else "NO"
        prompt_prefix += f"Response: {resp}\nDoes this response contain factual errors or hallucinations?\nAnswer: {lbl}. Confidence: 0.9\n\n"
        
    prompt_prefix += "New entry:\n"

    print(f"Loading {model_path}...")
    llm = Llama(model_path=model_path, n_ctx=1024, verbose=False)
    
    # 2. Inference
    scores = []
    labels = []
    frege_scores = []
    times = []
    
    frege_data = {}
    try:
        with open("results/experiment_2395_fregelogic.json") as f:
            fj = json.loads(f.read())
            for item in fj.get("per_entry_results", []):
                frege_data[item["case_id"]] = item["fregelogic_risk_score"]
    except Exception as e:
        print("Could not load fregelogic:", e)

    results_for_json = []

    for i, entry in enumerate(entries):
        resp = entry.get("response_text", "")[:500]
        q = prompt_prefix + f"Response: {resp}\nDoes this response contain factual errors or hallucinations?\nAnswer with YES or NO, then a confidence score 0.0-1.0.\nAnswer:"
        
        t0 = time.time()
        out = llm(q, max_tokens=20)
        t1 = time.time()
        
        times.append(t1 - t0)
        text = out["choices"][0]["text"].strip()
        
        if "YES" in text.upper():
            s = extract_float(text, default=0.8)
        elif "NO" in text.upper():
            s = 1.0 - extract_float(text, default=0.2)
        else:
            s = 0.5
            
        l = 0 if entry.get("known_verifier_label") == 1 else 1 # positive class is hallucination
        
        scores.append(s)
        labels.append(l)
        results_for_json.append({"idx": i, "score": s, "label": l, "case_id": entry.get("case_id")})
        
        frege_scores.append(frege_data.get(entry.get("case_id"), 0.5))

    mean_inference_time_s = sum(times) / len(times)
    
    auroc = roc_auc_score(labels, scores)
    
    tier0p_vs_semantic_energy_delta = auroc - 0.810
    
    r = 0.0
    if frege_scores:
        try:
            r = np.corrcoef(scores, frege_scores)[0, 1]
            if np.isnan(r):
                r = 0.0
        except Exception:
            pass

    artifact = tmpl.build_result({
        "honest_verdict": f"complete: tier0p_auroc={auroc:.4f}. Model: {model_family}",
        "tier0p_auroc": float(auroc),
        "tier0p_vs_semantic_energy_delta": float(tier0p_vs_semantic_energy_delta),
        "model_used": model_path,
        "model_family": model_family,
        "n_eval_examples": len(entries),
        "mean_inference_time_s": float(mean_inference_time_s),
        "orthogonality_vs_fregelogic": float(r),
        "random_seed": tmpl.random_seed,
        "preconditions_checked": preconditions,
        "scores": results_for_json,
        "verifier": "llm_judge"
    }, status="success")
    
    Path(tmpl.deliverable).parent.mkdir(parents=True, exist_ok=True)
    Path(tmpl.deliverable).write_text(json.dumps(artifact, indent=2))
    
    tmpl.assert_deliverable_written()
    print("Done")

if __name__ == "__main__":
    run()
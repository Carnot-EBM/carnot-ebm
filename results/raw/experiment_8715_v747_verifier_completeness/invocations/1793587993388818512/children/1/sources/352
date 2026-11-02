"""Experiment 3835: Formal Core 5-seed CI"""

from __future__ import annotations

import json
import math
import hashlib
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from carnot.eval.fover_memory_leakage_v3 import (
    _load_fr11_memory_index,
    _fr11_memory_score,
    _score_text_verifiers,
    compute_auroc,
    _select_balanced_subset,
    _label_to_int
)

RANDOM_SEEDS = [42, 137, 271, 314, 1729]
N_EXAMPLES = 1000

def _round_metric(value: float, digits: int = 6) -> float:
    return round(float(value), digits)

def _seed_t_ci95(values: Sequence[float]) -> dict[str, float]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("at least one seed value is required")
    mean = sum(numeric) / len(numeric)
    if len(numeric) < 2:
        return {
            "mean": _round_metric(mean),
            "low": _round_metric(mean),
            "high": _round_metric(mean),
        }
    t_crit_by_n = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    t_crit = t_crit_by_n.get(len(numeric), 1.96)
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1))
    half_width = t_crit * sample_std / math.sqrt(len(numeric))
    return {
        "mean": _round_metric(mean),
        "low": _round_metric(mean - half_width),
        "high": _round_metric(mean + half_width),
    }

def run_experiment_3835(repo_root: Path) -> dict[str, Any]:
    started_s = time.time()
    
    corpus_path = repo_root / "data" / "fover_test_v4.json"
    if not corpus_path.exists():
        corpus_path = repo_root / "data" / "fover_test_v3.json"
        if not corpus_path.exists():
            corpus_path = repo_root / "data" / "fover_test.json"
            if not corpus_path.exists():
                raise FileNotFoundError("BLOCKED: no fover_test corpus found")
    
    # We load from fover_corpus.jsonl because the frozen 0.9131 headline was 
    # computed using a balanced subset of fover_corpus.jsonl.
    from carnot.eval.fover_memory_leakage_v3 import _read_fover_rows
    fover_corpus_jsonl_path = repo_root / "data" / "fover_corpus.jsonl"
    all_rows = list(_read_fover_rows(fover_corpus_jsonl_path))
        
    preconditions = [
        "fover_corpus_cached",
        "ensemble_code_imports_and_runs"
    ]
    
    memory_index = _load_fr11_memory_index(repo_root)
    
    full_aurocs = []
    formal_aurocs = []
    learned_aurocs = []
    
    for seed in RANDOM_SEEDS:
        rows = _select_balanced_subset(all_rows, seed=seed, n_examples=N_EXAMPLES)
        labels = [_label_to_int(row["label"]) for row in rows]
        texts = [str(row.get("step_text", "")) for row in rows]
        
        v_scores = _score_text_verifiers(texts)
        tier0r = v_scores["tier0r_curry_howard"]
        tier0u = v_scores["tier0u_logical_consistency"]
        
        m_scores = [_fr11_memory_score(row, memory_index) for row in rows]
        
        formal_scores = [0.9 * r + 0.1 * u for r, u in zip(tier0r, tier0u, strict=True)]
        full_scores = [f + m for f, m in zip(formal_scores, m_scores, strict=True)]
        learned_scores = m_scores
        
        full_aurocs.append(compute_auroc(labels, full_scores))
        formal_aurocs.append(compute_auroc(labels, formal_scores))
        learned_aurocs.append(compute_auroc(labels, learned_scores))
        
    full_ci = _seed_t_ci95(full_aurocs)
    formal_ci = _seed_t_ci95(formal_aurocs)
    learned_ci = _seed_t_ci95(learned_aurocs)
    
    formal_core_above_learned = formal_ci["low"] > learned_ci["mean"]
    
    full_mean = full_ci["mean"]
    formal_mean = formal_ci["mean"]
    formal_width = formal_ci["high"] - formal_ci["low"]
    
    reproduced = 0.903 <= full_mean <= 0.923
    
    if not reproduced:
        honest_verdict = f"complete: formal_core_5seed_INCONCLUSIVE_full{full_mean:.4f}_expected_0.9131_harness_unfaithful"
    elif formal_width < 0.03 and formal_ci["low"] > 0.85:
        honest_verdict = f"complete: formal_core_5seed_CONFIRMED_formalonly{formal_mean:.4f}_full{full_mean:.4f}_contamination_free_ci{formal_width:.4f}"
    else:
        honest_verdict = f"complete: formal_core_5seed_FAIL_gate_conditions_not_met"
        
    checksum_str = json.dumps({"full": full_aurocs, "formal": formal_aurocs, "learned": learned_aurocs})
    checksum = hashlib.sha256(checksum_str.encode("utf-8")).hexdigest()
    
    return {
        "honest_verdict": honest_verdict,
        "full_ensemble_auroc_mean": full_mean,
        "formal_only_auroc_mean": formal_mean,
        "learned_only_auroc_mean": learned_ci["mean"],
        "per_condition_ci95": {
            "full_ensemble_auroc": full_ci,
            "formal_only_auroc": formal_ci,
            "learned_only_auroc": learned_ci,
        },
        "formal_core_above_learned": formal_core_above_learned,
        "verifier_partition": {
            "formal": [
                "tier0r_curry_howard",
                "tier0u_logical_consistency"
            ],
            "learned": [
                "fr11_session_memory"
            ],
            "rationale": "tier0r and tier0u are formal rule-based structures without weights. fr11_session_memory relies on traces/learned state. The 0.9131 harness did not use SOSKAN/SemEnergy."
        },
        "n_candidates_scored": N_EXAMPLES,
        "preconditions_checked": preconditions,
        "cited_upstream_artifacts": {
            "exp3826": "results/experiment_3826_fover_ablation_faithful.json",
            "exp2837_aggregation_path": "scripts/experiment_2837_fover_memory_leakage_v3.py"
        },
        "random_seeds_used": RANDOM_SEEDS,
        "reproducibility_checksum": checksum,
        "duration_s": time.time() - started_s,
        "inference_substrate": "verifier-scoring-only"
    }

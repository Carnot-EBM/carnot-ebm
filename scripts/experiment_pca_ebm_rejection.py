#!/usr/bin/env python3
"""Experiment: PCA + Gibbs EBM for rejection sampling.

Previous experiment: 2048-dim Gibbs EBM overfits (94% cal, 35% test).
Fix: PCA to 16-32 dims first, then train Gibbs on reduced space.

Usage:
    python scripts/experiment_pca_ebm_rejection.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Reuse same QA sets from previous experiments
from experiment_scaled_rejection_sampling import CALIBRATION_QA, TEST_QA

N_CANDIDATES = 5
PCA_DIMS = [4, 8, 16, 32]  # Try multiple PCA dimensions


def check_answer(response: str, expected: str) -> bool:
    return expected.lower() in response.lower().strip()


def extract_gen_activations(model, tokenizer, question, do_sample=False, temperature=1.0):
    import torch
    import jax.numpy as jnp

    prompt = f"Answer in one word or number only. {question}"
    inputs = tokenizer(prompt, return_tensors="pt")

    gen_kwargs = dict(max_new_tokens=20)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    prompt_len = inputs["input_ids"].shape[1]
    full_seq = outputs[0].unsqueeze(0)
    with torch.no_grad():
        hidden_out = model(full_seq)
        hs = hidden_out.hidden_states

    gen_last = hs[-1][0, prompt_len:, :].mean(dim=0).float().numpy()
    gen_mid = hs[len(hs)//2][0, prompt_len:, :].mean(dim=0).float().numpy()
    act = jnp.concatenate([jnp.array(gen_last), jnp.array(gen_mid)])

    return response, act


def train_and_evaluate(
    correct_acts, hallucinated_acts, test_acts, test_labels, greedy_results,
    llm, tokenizer, pca_dim, test_qa,
):
    """Train PCA + Gibbs EBM and evaluate rejection sampling."""
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import numpy as np
    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    # Balance
    min_n = min(len(correct_acts), len(hallucinated_acts))
    correct_arr = np.array(correct_acts[:min_n])
    hall_arr = np.array(hallucinated_acts[:min_n])
    all_data = np.vstack([correct_arr, hall_arr])

    # PCA: center then SVD
    mean = all_data.mean(axis=0)
    centered = all_data - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_components = Vt[:pca_dim]  # (pca_dim, 2048)

    def project(x):
        return jnp.array(((np.array(x) - mean) @ pca_components.T).astype(np.float32))

    # Project calibration data
    correct_pca = jnp.stack([project(a) for a in correct_acts[:min_n]])
    hall_pca = jnp.stack([project(a) for a in hallucinated_acts[:min_n]])

    # Train small Gibbs EBM on PCA'd data
    gibbs_config = GibbsConfig(
        input_dim=pca_dim,
        hidden_dims=[32, 16],
        activation="silu",
    )
    key = jrandom.PRNGKey(42)
    ebm = GibbsModel(gibbs_config, key=key)

    def get_params(m):
        return {"layers": [(w, b) for w, b in m.layers],
                "output_weight": m.output_weight, "output_bias": m.output_bias}

    def set_params(m, p):
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    def loss_fn(params):
        old = get_params(ebm)
        set_params(ebm, params)
        r = nce_loss(ebm, correct_pca, hall_pca)
        set_params(ebm, old)
        return r

    params = get_params(ebm)
    for epoch in range(150):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
    set_params(ebm, params)

    # Calibration accuracy
    cal_c_e = [float(ebm.energy(project(a))) for a in correct_acts[:min_n]]
    cal_h_e = [float(ebm.energy(project(a))) for a in hallucinated_acts[:min_n]]
    gap = sum(cal_h_e)/len(cal_h_e) - sum(cal_c_e)/len(cal_c_e)
    threshold = (sum(cal_c_e)/len(cal_c_e) + sum(cal_h_e)/len(cal_h_e)) / 2
    cal_tp = sum(1 for e in cal_h_e if e > threshold)
    cal_tn = sum(1 for e in cal_c_e if e <= threshold)
    cal_acc = (cal_tp + cal_tn) / (len(cal_c_e) + len(cal_h_e))

    # Rejection sampling on test set
    rejection_results = []
    for i, (q, expected) in enumerate(test_qa):
        candidates = []
        for c in range(N_CANDIDATES):
            response, act = extract_gen_activations(
                llm, tokenizer, q, do_sample=True, temperature=0.8,
            )
            energy = float(ebm.energy(project(act)))
            candidates.append((response, energy, check_answer(response, expected)))

        candidates.sort(key=lambda x: x[1])
        rejection_results.append(candidates[0][2])

    fixes = sum(1 for g, r in zip(greedy_results, rejection_results) if not g and r)
    regressions = sum(1 for g, r in zip(greedy_results, rejection_results) if g and not r)
    rej_acc = sum(rejection_results) / len(rejection_results)

    return {
        "pca_dim": pca_dim,
        "cal_acc": cal_acc,
        "gap": gap,
        "greedy_acc": sum(greedy_results) / len(greedy_results),
        "rej_acc": rej_acc,
        "fixes": fixes,
        "regressions": regressions,
    }


def main() -> int:
    print("=" * 60)
    print("EXPERIMENT: PCA + Gibbs EBM Rejection Sampling")
    print(f"PCA dims to try: {PCA_DIMS}")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, output_hidden_states=True,
    )
    llm.eval()

    # Collect calibration activations (once)
    print(f"\n--- Calibrating ({len(CALIBRATION_QA)} questions) ---")
    correct_acts = []
    hallucinated_acts = []
    for i, (q, expected) in enumerate(CALIBRATION_QA):
        response, act = extract_gen_activations(llm, tokenizer, q)
        if check_answer(response, expected):
            correct_acts.append(act)
        else:
            hallucinated_acts.append(act)
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(CALIBRATION_QA)}")

    print(f"Collected: {len(correct_acts)} correct, {len(hallucinated_acts)} hallucinated")

    # Greedy baseline on test set (once)
    print(f"\n--- Test baseline ({len(TEST_QA)} questions) ---")
    greedy_results = []
    test_acts = []
    test_labels = []
    for q, expected in TEST_QA:
        response, act = extract_gen_activations(llm, tokenizer, q)
        correct = check_answer(response, expected)
        greedy_results.append(correct)
        test_acts.append(act)
        test_labels.append(correct)

    greedy_acc = sum(greedy_results) / len(greedy_results)
    print(f"Greedy: {sum(greedy_results)}/{len(TEST_QA)} ({greedy_acc:.0%})")

    # Try each PCA dimension
    print(f"\n--- Testing PCA dims: {PCA_DIMS} ---")
    results = []
    for pca_dim in PCA_DIMS:
        print(f"\n  PCA dim={pca_dim}:")
        r = train_and_evaluate(
            correct_acts, hallucinated_acts, test_acts, test_labels,
            greedy_results, llm, tokenizer, pca_dim, TEST_QA,
        )
        results.append(r)
        imp = r["rej_acc"] - r["greedy_acc"]
        print(f"    Cal acc: {r['cal_acc']:.0%}, Gap: {r['gap']:.2f}")
        print(f"    Test: {r['rej_acc']:.0%} ({'+' if imp>=0 else ''}{imp:.0%})")
        print(f"    Fixes: {r['fixes']}, Regressions: {r['regressions']}, Net: {r['fixes']-r['regressions']}")

    # Summary table
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'PCA':>4} | {'Cal%':>5} | {'Gap':>6} | {'Greedy':>6} | {'EBM':>6} | {'Δ':>5} | {'Fix':>3} | {'Reg':>3} | {'Net':>4}")
    print("-" * 60)
    for r in results:
        imp = r["rej_acc"] - r["greedy_acc"]
        print(f"{r['pca_dim']:>4} | {r['cal_acc']:>4.0%} | {r['gap']:>6.2f} | "
              f"{r['greedy_acc']:>5.0%} | {r['rej_acc']:>5.0%} | "
              f"{'+' if imp>=0 else ''}{imp:>4.0%} | {r['fixes']:>3} | "
              f"{r['regressions']:>3} | {'+' if r['fixes']>=r['regressions'] else ''}{r['fixes']-r['regressions']:>3}")
    print(f"{'='*60}")

    best = max(results, key=lambda r: r["rej_acc"] - r["greedy_acc"])
    best_imp = best["rej_acc"] - best["greedy_acc"]
    if best_imp > 0:
        print(f"BEST: PCA dim={best['pca_dim']}, improvement {best_imp:+.0%}")
    else:
        print(f"No PCA dimension improved over greedy baseline.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

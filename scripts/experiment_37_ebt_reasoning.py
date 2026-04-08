#!/usr/bin/env python3
"""Experiment 37: EBT Reasoning — Energy-Based Answer Repair in Embedding Space.

The pivot from "hallucination detector" to "reasoning engine."

Instead of detecting if an LLM's activations look correct (which we proved
doesn't work — 36 experiments, 50% practical accuracy), train an EBM on
semantic embeddings of (question, answer) pairs. Then use gradient descent
in embedding space to repair wrong answers.

Pipeline:
  1. Embed all TruthfulQA QA pairs using a frozen sentence encoder
  2. Train Gibbs EBM: E(question_emb, correct_answer_emb) < E(question_emb, wrong_answer_emb)
  3. For wrong answers: fix question embedding, optimize answer embedding via Langevin dynamics
  4. Decode repaired embedding via nearest-neighbor lookup in candidate pool
  5. Measure: does repair improve the answer?

REQ: REQ-EBT-001 through REQ-EBT-004
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def main() -> int:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from datasets import load_dataset

    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    print("=" * 70)
    print("EXPERIMENT 37: EBT Reasoning — Energy-Based Answer Repair")
    print("  Pivot from hallucination detector to reasoning engine")
    print("=" * 70)

    start = time.time()

    # --- Step 1: Embed QA pairs ---
    print("\nStep 1: Embedding TruthfulQA pairs...")

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embed_dim = 384  # per sentence, 768 concatenated

    ds = load_dataset("truthful_qa", "generation")
    examples = list(ds["validation"])

    # Build (question, answer, is_correct) triples
    qa_triples = []
    for ex in examples:
        question = ex["question"]
        for answer in ex["correct_answers"]:
            qa_triples.append((question, answer, True))
        for answer in ex["incorrect_answers"]:
            qa_triples.append((question, answer, False))

    print(f"  Total QA pairs: {len(qa_triples)}")
    print(f"  Correct: {sum(1 for _, _, c in qa_triples if c)}")
    print(f"  Wrong: {sum(1 for _, _, c in qa_triples if not c)}")

    # Embed all questions and answers
    questions = [q for q, _, _ in qa_triples]
    answers = [a for _, a, _ in qa_triples]
    labels = np.array([1 if c else 0 for _, _, c in qa_triples], dtype=np.int32)

    print("  Encoding questions...")
    q_embs = encoder.encode(questions, batch_size=128, show_progress_bar=True)
    print("  Encoding answers...")
    a_embs = encoder.encode(answers, batch_size=128, show_progress_bar=True)

    # Concatenate: (question_emb, answer_emb) → 768-dim
    qa_embs = np.concatenate([q_embs, a_embs], axis=1).astype(np.float32)
    print(f"  Joint embedding shape: {qa_embs.shape}")

    # --- Step 2: Train Gibbs EBM ---
    print("\nStep 2: Training Gibbs EBM on QA embeddings...")

    correct_embs = jnp.array(qa_embs[labels == 1])
    wrong_embs = jnp.array(qa_embs[labels == 0])
    min_n = min(len(correct_embs), len(wrong_embs))

    rng = np.random.default_rng(42)
    correct_embs = correct_embs[rng.permutation(len(correct_embs))[:min_n]]
    wrong_embs = wrong_embs[rng.permutation(len(wrong_embs))[:min_n]]

    split = int(min_n * 0.8)
    tc, tw = correct_embs[:split], wrong_embs[:split]
    vc, vw = correct_embs[split:], wrong_embs[split:]

    print(f"  Balanced: {min_n} each (train={split}, test={min_n - split})")

    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=768, hidden_dims=[256, 64], activation="silu")
    ebm = GibbsModel(config, key=key)

    def get_p(m):
        return {"layers": [(w, b) for w, b in m.layers],
                "output_weight": m.output_weight, "output_bias": m.output_bias}

    def set_p(m, p):
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    params = get_p(ebm)

    def loss_fn(p):
        old = get_p(ebm)
        set_p(ebm, p)
        r = nce_loss(ebm, tc, tw)
        set_p(ebm, old)
        return r

    train_start = time.time()
    for ep in range(500):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.003 * g, params, grads)
        if ep % 100 == 0:
            set_p(ebm, params)
            loss = float(nce_loss(ebm, tc, tw))
            print(f"    Epoch {ep}: loss={loss:.4f}")
    set_p(ebm, params)
    print(f"  Training done ({time.time() - train_start:.1f}s)")

    # Evaluate energy discrimination
    n_eval = min(300, len(vc))
    ce = [float(ebm.energy(vc[i])) for i in range(n_eval)]
    we = [float(ebm.energy(vw[i])) for i in range(n_eval)]
    thresh = (np.mean(ce) + np.mean(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    disc_acc = (tp + tn) / (len(ce) + len(we))
    gap = np.mean(we) - np.mean(ce)

    print(f"  Energy discrimination: {disc_acc:.1%} (gap={gap:.4f})")

    # --- Step 3: Gradient-based answer repair ---
    print("\nStep 3: Repairing wrong answers via gradient descent...")

    # Build candidate pool for decoding
    correct_answers_text = []
    correct_answers_embs = []
    for ex in examples:
        for ans in ex["correct_answers"]:
            correct_answers_text.append(ans)
    correct_answers_embs = encoder.encode(correct_answers_text, batch_size=128)
    correct_answers_embs = np.array(correct_answers_embs, dtype=np.float32)
    print(f"  Candidate pool: {len(correct_answers_text)} correct answers")

    # Test repair on wrong answers from the test set
    # Get original question/answer pairs for the test wrong answers
    test_wrong_indices = rng.permutation(len(wrong_embs))[split:split + min(50, min_n - split)]
    wrong_qa_test = wrong_embs[test_wrong_indices - split + split]  # test portion

    # We need the original question embeddings separate from the answer embeddings
    # Reconstruct from the concatenated embeddings
    test_q_embs = wrong_qa_test[:, :embed_dim]  # first 384 dims = question
    test_a_embs = wrong_qa_test[:, embed_dim:]   # last 384 dims = answer

    n_repair = min(50, len(test_q_embs))
    repair_results = []

    for i in range(n_repair):
        q_emb = jnp.array(test_q_embs[i])
        wrong_a_emb = jnp.array(test_a_embs[i])

        # Energy before repair
        qa_before = jnp.concatenate([q_emb, wrong_a_emb])
        energy_before = float(ebm.energy(qa_before))

        # Gradient descent on answer embedding only
        a_emb = wrong_a_emb.copy()
        lr = 0.01
        noise_scale = 0.001
        energies = [energy_before]

        def answer_energy(a):
            return ebm.energy(jnp.concatenate([q_emb, a]))

        for step in range(200):
            grad = jax.grad(answer_energy)(a_emb)
            noise = jrandom.normal(jrandom.PRNGKey(step), shape=a_emb.shape) * noise_scale
            a_emb = a_emb - lr * grad + noise
            if step % 50 == 0:
                energies.append(float(answer_energy(a_emb)))

        energy_after = float(ebm.energy(jnp.concatenate([q_emb, a_emb])))
        energies.append(energy_after)

        # Decode: find nearest correct answer in candidate pool
        repaired_a = np.array(a_emb)
        similarities = np.dot(correct_answers_embs, repaired_a) / (
            np.linalg.norm(correct_answers_embs, axis=1) * np.linalg.norm(repaired_a) + 1e-10
        )
        best_idx = similarities.argmax()
        best_text = correct_answers_text[best_idx]
        best_sim = float(similarities[best_idx])

        # Also check similarity of original wrong answer to nearest correct
        original_a = np.array(test_a_embs[i])
        orig_sims = np.dot(correct_answers_embs, original_a) / (
            np.linalg.norm(correct_answers_embs, axis=1) * np.linalg.norm(original_a) + 1e-10
        )
        orig_best_sim = float(orig_sims.max())

        repair_results.append({
            "energy_before": energy_before,
            "energy_after": energy_after,
            "energy_improved": energy_after < energy_before,
            "sim_before": orig_best_sim,
            "sim_after": best_sim,
            "sim_improved": best_sim > orig_best_sim,
            "decoded_answer": best_text[:60],
        })

        if i < 5:
            print(f"    [{i}] energy: {energy_before:.3f} → {energy_after:.3f} "
                  f"sim: {orig_best_sim:.3f} → {best_sim:.3f} → \"{best_text[:50]}\"")

    # --- Results ---
    elapsed = time.time() - start

    n_energy_improved = sum(1 for r in repair_results if r["energy_improved"])
    n_sim_improved = sum(1 for r in repair_results if r["sim_improved"])
    mean_energy_drop = np.mean([r["energy_after"] - r["energy_before"] for r in repair_results])
    mean_sim_gain = np.mean([r["sim_after"] - r["sim_before"] for r in repair_results])

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 37 RESULTS ({elapsed:.0f}s)")
    print(sep)
    print(f"  EBM discrimination (test set): {disc_acc:.1%} (gap={gap:.4f})")
    print(f"")
    print(f"  Repair results ({n_repair} wrong answers):")
    print(f"    Energy improved: {n_energy_improved}/{n_repair} ({n_energy_improved / n_repair:.0%})")
    print(f"    Similarity improved: {n_sim_improved}/{n_repair} ({n_sim_improved / n_repair:.0%})")
    print(f"    Mean energy change: {mean_energy_drop:+.4f}")
    print(f"    Mean similarity gain: {mean_sim_gain:+.4f}")

    if n_energy_improved / n_repair > 0.7 and n_sim_improved / n_repair > 0.5:
        print(f"\n  VERDICT: ✅ Energy-guided repair works in embedding space!")
        print(f"  Gradient descent moves wrong answers toward correct regions.")
    elif n_energy_improved / n_repair > 0.5:
        print(f"\n  VERDICT: ⚠️ Energy decreases but similarity doesn't always improve")
        print(f"  The EBM learns a useful landscape but it doesn't perfectly align with correctness.")
    else:
        print(f"\n  VERDICT: ❌ Repair doesn't work — energy landscape not useful")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())

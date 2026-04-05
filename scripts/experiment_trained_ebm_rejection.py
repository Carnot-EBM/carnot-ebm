#!/usr/bin/env python3
"""Experiment: Trained Gibbs EBM for rejection sampling.

Instead of a linear hallucination direction (which confuses confidence
with correctness), train a nonlinear Gibbs model via NCE on the
activation data. The Gibbs model can learn the curved decision boundary
between correct and hallucinated activations.

Usage:
    python scripts/experiment_trained_ebm_rejection.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Same calibration set as scaled experiment
CALIBRATION_QA = [
    ("What is 2+3?", "5"), ("What is 7*8?", "56"), ("What is 100/4?", "25"),
    ("What is 15-9?", "6"), ("What is 3^3?", "27"), ("What is 50+50?", "100"),
    ("What is 12*12?", "144"), ("What is 81/9?", "9"), ("What is 1000-1?", "999"),
    ("What is 2^10?", "1024"),
    ("What is the square root of 169?", "13"), ("What is 17*19?", "323"),
    ("What is 256/16?", "16"), ("What is 99*99?", "9801"),
    ("What is the cube root of 27?", "3"), ("What is 7! (7 factorial)?", "5040"),
    ("What is 2^16?", "65536"), ("What is 1+2+3+4+5+6+7+8+9+10?", "55"),
    ("What is 37*43?", "1591"), ("What is 123*456?", "56088"),
    ("What is the 20th prime number?", "71"), ("What is 11! (11 factorial)?", "39916800"),
    ("What is 2^20?", "1048576"), ("What is the square root of 1764?", "42"),
    ("What is 97*103?", "9991"), ("What is 19^2?", "361"),
    ("What is 23*29?", "667"), ("What is the 12th Fibonacci number?", "144"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
    ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of France?", "Paris"),
    ("What is the capital of the UK?", "London"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("What is the capital of Russia?", "Moscow"),
    ("What is the capital of South Korea?", "Seoul"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Turkey?", "Ankara"),
    ("What is the capital of Thailand?", "Bangkok"),
    ("What is the capital of Argentina?", "Buenos Aires"),
    ("What is the capital of Myanmar?", "Naypyidaw"),
    ("What is the capital of Sri Lanka?", "Colombo"),
    ("What is the capital of Kazakhstan?", "Astana"),
    ("What is the capital of Nigeria?", "Abuja"),
    ("What is the capital of Pakistan?", "Islamabad"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("How many chromosomes do humans have?", "46"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the atomic number of hydrogen?", "1"),
    ("What is the atomic number of oxygen?", "8"),
    ("What is the speed of sound in m/s approximately?", "343"),
    ("How many planets are in our solar system?", "8"),
    ("What is the atomic number of gold?", "79"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("What is the chemical formula for methane?", "CH4"),
    ("In what year did the Titanic sink?", "1912"),
    ("In what year did WW1 start?", "1914"),
    ("In what year did WW2 end?", "1945"),
    ("In what year did humans first land on the Moon?", "1969"),
    ("In what year was the Declaration of Independence signed?", "1776"),
    ("In what year did the Berlin Wall fall?", "1989"),
    ("In what year did the French Revolution begin?", "1789"),
    ("In what year did Columbus reach the Americas?", "1492"),
    ("How many sides does a hexagon have?", "6"),
    ("How many sides does an octagon have?", "8"),
    ("How many continents are there?", "7"),
    ("How many strings does a standard guitar have?", "6"),
    ("How many cards are in a standard deck?", "52"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the smallest planet in our solar system?", "Mercury"),
    ("What is the closest star to Earth?", "Sun"),
    ("What is the largest organ in the human body?", "skin"),
    ("How many teeth does an adult human typically have?", "32"),
    ("What year was Python first released?", "1991"),
    ("What year was Java first released?", "1995"),
    ("What does HTML stand for?", "HyperText"),
    ("What does CPU stand for?", "Central Processing"),
    ("Who created Linux?", "Torvalds"),
    ("What year was the iPhone first released?", "2007"),
    ("How many bits in a byte?", "8"),
    ("What is the binary representation of 10?", "1010"),
]

TEST_QA = [
    ("What is the capital of Portugal?", "Lisbon"),
    ("What is the capital of Poland?", "Warsaw"),
    ("What is the chemical symbol for silver?", "Ag"),
    ("What is the chemical symbol for copper?", "Cu"),
    ("What is 13*17?", "221"),
    ("What is 29*31?", "899"),
    ("What is the square root of 225?", "15"),
    ("What is 6! (6 factorial)?", "720"),
    ("What is the 10th prime number?", "29"),
    ("In what year was the internet invented?", "1969"),
    ("What is the atomic number of carbon?", "6"),
    ("What is the chemical formula for ammonia?", "NH3"),
    ("How many bones in the human body?", "206"),
    ("What is the largest ocean?", "Pacific"),
    ("In what year did the Cold War end?", "1991"),
    ("What is the derivative of x^3?", "3x"),
    ("What is 2^8?", "256"),
    ("What is the sum of angles in a triangle?", "180"),
    ("Who painted the Mona Lisa?", "Vinci"),
    ("What is the speed of light approximately in m/s?", "3"),
]

N_CANDIDATES = 5


def check_answer(response: str, expected: str) -> bool:
    return expected.lower() in response.lower().strip()


def extract_gen_activations(model, tokenizer, question, do_sample=False, temperature=1.0):
    """Generate answer and extract activations from generated tokens."""
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


def main() -> int:
    print("=" * 60)
    print("EXPERIMENT: Trained Gibbs EBM for Rejection Sampling")
    print("Nonlinear EBM instead of linear direction")
    print("=" * 60)

    import jax
    import jax.numpy as jnp
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    print(f"\nLoading {model_name}...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, output_hidden_states=True,
    )
    llm.eval()
    print(f"Loaded in {time.time() - start:.1f}s")

    # Phase 1: Collect calibration activations
    print(f"\n--- Phase 1: Collecting activations ({len(CALIBRATION_QA)} questions) ---")
    correct_acts = []
    hallucinated_acts = []
    cal_correct = 0

    for i, (q, expected) in enumerate(CALIBRATION_QA):
        response, act = extract_gen_activations(llm, tokenizer, q, do_sample=False)
        if check_answer(response, expected):
            correct_acts.append(act)
            cal_correct += 1
        else:
            hallucinated_acts.append(act)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(CALIBRATION_QA)}... "
                  f"({cal_correct} correct, {i+1-cal_correct} hallucinated)")

    n_correct = len(correct_acts)
    n_hallucinated = len(hallucinated_acts)
    print(f"Collected: {n_correct} correct, {n_hallucinated} hallucinated")

    if n_hallucinated < 5:
        print("Not enough hallucinations")
        return 0

    # Balance the dataset (same size for NCE)
    min_n = min(n_correct, n_hallucinated)
    correct_batch = jnp.stack(correct_acts[:min_n])
    hallucinated_batch = jnp.stack(hallucinated_acts[:min_n])
    act_dim = correct_batch.shape[1]

    print(f"Balanced: {min_n} each, dim={act_dim}")

    # Phase 2: Train Gibbs EBM on activations via NCE
    print(f"\n--- Phase 2: Training Gibbs EBM (NCE, 200 epochs) ---")
    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss
    import jax.random as jrandom

    # Use a compact architecture — activations are 2048-dim (last + mid)
    gibbs_config = GibbsConfig(
        input_dim=act_dim,
        hidden_dims=[256, 64],
        activation="silu",
    )
    key = jrandom.PRNGKey(42)
    ebm = GibbsModel(gibbs_config, key=key)

    # Training loop (same pattern as train_sat_verifier)
    def get_params(m):
        return {
            "layers": [(w, b) for w, b in m.layers],
            "output_weight": m.output_weight,
            "output_bias": m.output_bias,
        }

    def set_params(m, params):
        m.layers = list(params["layers"])
        m.output_weight = params["output_weight"]
        m.output_bias = params["output_bias"]

    def loss_fn(params):
        old = get_params(ebm)
        set_params(ebm, params)
        result = nce_loss(ebm, correct_batch, hallucinated_batch)
        set_params(ebm, old)
        return result

    params = get_params(ebm)
    lr = 0.005
    n_epochs = 200

    start = time.time()
    for epoch in range(n_epochs):
        loss = float(nce_loss(ebm, correct_batch, hallucinated_batch))
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        set_params(ebm, params)

        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: NCE loss = {loss:.4f}")

    print(f"  Training done in {time.time() - start:.1f}s")

    # Verify: correct activations should get lower energy
    correct_energies = [float(ebm.energy(a)) for a in correct_acts[:min_n]]
    hall_energies = [float(ebm.energy(a)) for a in hallucinated_acts[:min_n]]
    mean_c = sum(correct_energies) / len(correct_energies)
    mean_h = sum(hall_energies) / len(hall_energies)
    gap = mean_h - mean_c
    print(f"  Calibration: correct_energy={mean_c:.2f}, hallucinated_energy={mean_h:.2f}, gap={gap:.2f}")

    # Classification accuracy on calibration set
    threshold = (mean_c + mean_h) / 2
    cal_tp = sum(1 for e in hall_energies if e > threshold)
    cal_tn = sum(1 for e in correct_energies if e <= threshold)
    cal_acc = (cal_tp + cal_tn) / (len(correct_energies) + len(hall_energies))
    print(f"  Calibration accuracy: {cal_acc:.0%}")

    # Phase 3: Test — greedy baseline
    print(f"\n--- Phase 3: Test baseline ({len(TEST_QA)} questions) ---")
    greedy_results = []
    for q, expected in TEST_QA:
        response, _ = extract_gen_activations(llm, tokenizer, q, do_sample=False)
        greedy_results.append(check_answer(response, expected))

    greedy_acc = sum(greedy_results) / len(greedy_results)
    print(f"Greedy: {sum(greedy_results)}/{len(TEST_QA)} ({greedy_acc:.0%})")

    # Phase 4: Rejection sampling with trained EBM
    print(f"\n--- Phase 4: EBM rejection sampling ({N_CANDIDATES} candidates) ---")
    rejection_results = []

    for i, (q, expected) in enumerate(TEST_QA):
        candidates = []
        for c in range(N_CANDIDATES):
            response, act = extract_gen_activations(
                llm, tokenizer, q, do_sample=True, temperature=0.8,
            )
            energy = float(ebm.energy(act))
            candidates.append((response, energy, check_answer(response, expected)))

        candidates.sort(key=lambda x: x[1])
        best_response, best_energy, best_correct = candidates[0]

        greedy_icon = "✓" if greedy_results[i] else "✗"
        best_icon = "✓" if best_correct else "✗"
        tag = ""
        if best_correct and not greedy_results[i]:
            tag = " ★ FIXED"
        elif not best_correct and greedy_results[i]:
            tag = " ✖ REGRESSED"
        print(f"  [{greedy_icon}→{best_icon}] {q[:40]}... e={best_energy:.1f}{tag}")

        rejection_results.append(best_correct)

    rejection_acc = sum(rejection_results) / len(rejection_results)
    fixes = sum(1 for g, r in zip(greedy_results, rejection_results) if not g and r)
    regressions = sum(1 for g, r in zip(greedy_results, rejection_results) if g and not r)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  EBM architecture: Gibbs [2048→256→64→1], NCE-trained")
    print(f"  Calibration: {min_n} correct + {min_n} hallucinated, gap={gap:.2f}")
    print(f"  Calibration accuracy: {cal_acc:.0%}")
    print(f"")
    print(f"  Greedy baseline:    {sum(greedy_results)}/{len(TEST_QA)} ({greedy_acc:.0%})")
    print(f"  EBM-selected:       {sum(rejection_results)}/{len(TEST_QA)} ({rejection_acc:.0%})")
    improvement = rejection_acc - greedy_acc
    print(f"  Improvement:        {'+' if improvement >= 0 else ''}{improvement:.0%}")
    print(f"")
    print(f"  Fixes: {fixes}, Regressions: {regressions}, Net: {'+' if fixes>=regressions else ''}{fixes-regressions}")
    print(f"{'='*60}")

    if improvement > 0:
        print("SUCCESS: Trained EBM improves LLM accuracy via rejection sampling!")
    elif improvement == 0:
        print("NEUTRAL: No net improvement.")
    else:
        print("REGRESSION: Trained EBM doesn't generalize from calibration to test set.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

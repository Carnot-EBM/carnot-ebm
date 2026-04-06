#!/usr/bin/env python3
"""Experiment 18: Concept-specific activation steering on Qwen3-0.6B.

**Why this experiment exists:**
    Experiments 15-16 tried steering with a generic mean-difference direction
    (correct minus wrong activations) and got zero effect. The problem: that
    direction captures statistical differences between correct/wrong answers
    but does NOT causally drive hallucination during generation. Anthropic's
    emotion research showed that concept-specific vectors — extracted by
    prompting the model into specific cognitive modes — ARE causally effective.

    Experiment 17 (concept_specific_vectors.py) identified which concept
    direction best separates correct from hallucinated activations. The
    confabulation direction (confabulating minus certain) should capture the
    model's internal "I'm making stuff up" mode, which is exactly what we
    want to suppress during generation.

**What this experiment does:**
    1. Loads Qwen3-0.6B and computes concept-specific contrastive vectors
       using the same targeted-prompting approach from experiment 17.
    2. Identifies the BEST concept direction for hallucination detection
       (expected: confabulation_dir).
    3. Hooks into model layers during generation and SUBTRACTS
       alpha * best_direction from hidden states — pushing the model away
       from the confabulation mode and toward certainty.
    4. Tests alphas [0.1, 0.5, 1.0, 2.0, 5.0] on the standard 25 QA pairs.
    5. Compares against greedy baseline AND logprob rejection (+10%).

**Critical implementation detail:**
    All direction arithmetic is done in float32 to avoid bfloat16 precision
    loss, then cast back to the hidden state's original dtype:
        modified = (hidden.float() + alpha * direction.float()).to(hidden.dtype)

**Relationship to prior experiments:**
    - Exp 8: Found activation-space direction separates correct/wrong (64%)
    - Exp 15-16: Generic steering had zero effect (direction ≠ causal)
    - Exp 17: Concept-specific vectors separate better than generic
    - THIS (18): Steer with the concept-specific direction that IS causal

Usage:
    python scripts/experiment_concept_specific_steering.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reuse the same 20 diverse topics from experiment 17 to generate
# concept activations. Breadth across domains ensures the vectors capture
# cognitive mode (certain, confabulating, etc.) rather than topic.
# ---------------------------------------------------------------------------
TOPICS = [
    ("France", "its capital is Paris"),
    ("elephants", "they are the largest land animals"),
    ("the speed of light", "it is approximately 300,000 km/s"),
    ("water", "its chemical formula is H2O"),
    ("the Moon", "it orbits the Earth"),
    ("DNA", "it carries genetic information"),
    ("Shakespeare", "he wrote Hamlet"),
    ("gravity", "it attracts objects with mass toward each other"),
    ("photosynthesis", "plants convert sunlight into energy"),
    ("the Pacific Ocean", "it is the largest ocean on Earth"),
    ("carbon", "its atomic number is 6"),
    ("Antarctica", "it is the coldest continent"),
    ("pi", "it is approximately 3.14159"),
    ("the human heart", "it has four chambers"),
    ("gold", "its chemical symbol is Au"),
    ("Mount Everest", "it is the tallest mountain above sea level"),
    ("neurons", "they transmit electrical signals in the brain"),
    ("Jupiter", "it is the largest planet in our solar system"),
    ("iron", "it is essential for hemoglobin in blood"),
    ("the Amazon River", "it is the largest river by water volume"),
]


# ---------------------------------------------------------------------------
# Concept prompt templates — each pushes the model into a specific cognitive
# mode. The key one is "confabulating" which primes the model to fabricate.
# ---------------------------------------------------------------------------
CONCEPT_TEMPLATES: dict[str, str] = {
    "certain": (
        "I am absolutely certain that {topic} is well understood. {fact}. "
        "This is well-established and beyond dispute."
    ),
    "uncertain": (
        "I'm not sure about this, but I think {topic} might have something "
        "to do with... hmm, I'm really not confident here, but maybe {fact}..."
    ),
    "confabulating": (
        "Let me just make up a plausible-sounding answer about {topic}. "
        "I'll invent something that sounds real: supposedly, {fact}, "
        "although I'm fabricating this entirely."
    ),
    "reasoning": (
        "Let me think step by step about {topic}. First, we should consider "
        "the basic facts. We know that {fact}. From this we can deduce..."
    ),
    "memorized": (
        "As is commonly known, {topic} is a well-studied subject. "
        "The established fact is that {fact}. This is standard textbook knowledge."
    ),
}


# ---------------------------------------------------------------------------
# Standard 25 QA pairs used across all experiments for consistent comparison.
# ---------------------------------------------------------------------------
QA_PAIRS = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
    ("How many legs does a dog have?", "4"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the square root of 144?", "12"),
    ("In what year did World War II end?", "1945"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("How many days are in a week?", "7"),
    ("What is the atomic number of carbon?", "6"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the speed of light in km/s approximately?", "300000"),
    ("What is the derivative of x squared?", "2x"),
    ("What gas do plants absorb from the atmosphere?", "CO2"),
    ("What is the 15th prime number?", "47"),
    ("What is 17 * 23?", "391"),
    ("What is the integral of 1/x?", "ln"),
    ("What is the population of Iceland approximately?", "380000"),
    ("How many bones are in the adult human body?", "206"),
    ("What is the sum of angles in a pentagon?", "540"),
    ("What is 13 factorial?", "6227020800"),
    ("What is the 8th Fibonacci number?", "21"),
    ("What is the chemical formula for glucose?", "C6H12O6"),
    ("What year was the Python programming language first released?", "1991"),
]


def check_answer(response: str, expected: str) -> bool:
    """Check if the model's response contains the expected answer."""
    return expected.lower() in response.lower().strip()


# ---------------------------------------------------------------------------
# Phase 1: Concept vector extraction
# ---------------------------------------------------------------------------

def extract_last_layer_activation(model, tokenizer, text: str):
    """Extract mean activation from the last hidden layer for a text.

    Runs the text through the model, extracts hidden states from the final
    layer, and averages across token positions to get a single vector.

    Returns a JAX array of shape (hidden_dim,), or None on failure.
    bfloat16 tensors are cast to float32 for numpy/jax compatibility.
    """
    import torch
    import jax.numpy as jnp

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    param = next(model.parameters(), None)
    dev = param.device if param is not None else "cpu"
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states  # tuple of (batch, seq, dim)

    # Last layer, first batch element, mean across sequence positions.
    # Cast to float32 because bfloat16 -> numpy is not supported.
    last_hidden = hidden_states[-1][0].float().mean(dim=0).cpu().numpy()
    return jnp.array(last_hidden)


def generate_concept_activations(model, tokenizer, n_topics: int = 20):
    """Generate text in each cognitive mode and extract last-layer activations.

    For each concept template x each topic:
    1. Fill the template with the topic
    2. Generate a short continuation (30 tokens) to let the model enter the mode
    3. Extract last-layer activations from the full sequence

    Returns dict mapping concept name -> list of JAX activation vectors.
    """
    import torch

    concept_activations: dict[str, list] = {c: [] for c in CONCEPT_TEMPLATES}
    topics_to_use = TOPICS[:n_topics]

    total = len(CONCEPT_TEMPLATES) * len(topics_to_use)
    done = 0

    for concept_name, template in CONCEPT_TEMPLATES.items():
        for topic, fact in topics_to_use:
            text = template.format(topic=topic, fact=fact)

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            param = next(model.parameters(), None)
            dev = param.device if param is not None else "cpu"
            inputs = {k: v.to(dev) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            # Extract activation from the FULL sequence (prompt + generated)
            # to capture the model's state after being primed into the mode.
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            act = extract_last_layer_activation(model, tokenizer, full_text)
            if act is not None:
                concept_activations[concept_name].append(act)

            done += 1
            if done % 20 == 0:
                print(f"  Generated {done}/{total} concept samples...")

    return concept_activations


def compute_contrastive_directions(concept_activations: dict[str, list]):
    """Compute normalized contrastive vectors between each concept and 'certain'.

    The "certain" concept is the anchor/baseline (model's best behavior).
    Each direction captures how a specific cognitive mode differs from certainty.

    Returns dict mapping direction name -> normalized JAX vector.
    """
    import jax.numpy as jnp

    concept_means: dict[str, jnp.ndarray] = {}
    for name, acts in concept_activations.items():
        if acts:
            concept_means[name] = jnp.mean(jnp.stack(acts), axis=0)

    if "certain" not in concept_means:
        print("ERROR: No 'certain' activations — cannot compute contrastive pairs")
        return {}

    anchor = concept_means["certain"]
    directions: dict[str, jnp.ndarray] = {}

    for name, mu in concept_means.items():
        if name == "certain":
            continue
        direction = mu - anchor
        norm = jnp.linalg.norm(direction)
        direction = direction / jnp.maximum(norm, 1e-8)
        directions[f"{name}_dir"] = direction

    return directions


def find_best_direction(directions, correct_acts, hallucinated_acts):
    """Evaluate all concept directions and return the best one for separation.

    Uses midpoint-threshold classification accuracy to rank directions.
    Returns (best_name, best_direction, all_results_sorted).
    """
    import jax.numpy as jnp
    import numpy as np

    results = []
    for dir_name, direction in directions.items():
        correct_projs = [float(jnp.dot(a, direction)) for a in correct_acts]
        halluc_projs = [float(jnp.dot(a, direction)) for a in hallucinated_acts]

        mean_c = sum(correct_projs) / len(correct_projs)
        mean_h = sum(halluc_projs) / len(halluc_projs)

        std_c = float(np.std(correct_projs)) if len(correct_projs) > 1 else 0.0
        std_h = float(np.std(halluc_projs)) if len(halluc_projs) > 1 else 0.0
        fisher = abs(mean_h - mean_c) / (std_c + std_h + 1e-8)

        threshold = (mean_c + mean_h) / 2
        if mean_h > mean_c:
            tp = sum(1 for p in halluc_projs if p > threshold)
            tn = sum(1 for p in correct_projs if p <= threshold)
        else:
            tp = sum(1 for p in halluc_projs if p < threshold)
            tn = sum(1 for p in correct_projs if p >= threshold)

        accuracy = (tp + tn) / (len(correct_projs) + len(halluc_projs))
        results.append({
            "name": dir_name,
            "accuracy": accuracy,
            "fisher": fisher,
            "gap": mean_h - mean_c,
        })

    results.sort(key=lambda r: r["accuracy"], reverse=True)
    best_name = results[0]["name"]
    return best_name, directions[best_name], results


# ---------------------------------------------------------------------------
# Phase 2: Steering hooks
# ---------------------------------------------------------------------------

def make_steering_hook(direction_torch, alpha):
    """Create a forward hook that subtracts alpha * direction from hidden states.

    CRITICAL: All arithmetic is done in float32 to avoid bfloat16 precision
    loss, then cast back to the hidden state's original dtype. This matters
    because bfloat16 has only 7 bits of mantissa — small steering vectors
    can get rounded to zero in bfloat16 arithmetic.

    Args:
        direction_torch: The concept direction as a float32 torch tensor.
        alpha: Steering strength. Positive alpha SUBTRACTS the direction
               (pushes away from confabulation toward certainty).
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            orig_dtype = hidden.dtype
            # Truncate direction to match hidden dim (safety for mismatched dims)
            d = direction_torch[:hidden.shape[-1]].to(device=hidden.device)
            # float32 arithmetic, then cast back to original dtype
            modified = (hidden.float() - alpha * d.float()).to(orig_dtype)
            return (modified,) + output[1:]
        return output
    return hook_fn


def steer_and_evaluate(model, tokenizer, questions, direction_torch, layers, alpha):
    """Run steered generation on all questions, return (accuracy, per-question results).

    Registers forward hooks on the specified layers that subtract
    alpha * direction from hidden states during the forward pass.
    """
    import torch

    results = []
    responses = []
    for q, expected in questions:
        prompt = f"Answer in one word or number only. {q}"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Register steering hooks on target layers
        handles = []
        for layer_idx in layers:
            if hasattr(model.model, "layers") and layer_idx < len(model.model.layers):
                h = model.model.layers[layer_idx].register_forward_hook(
                    make_steering_hook(direction_torch, alpha)
                )
                handles.append(h)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, do_sample=False, temperature=1.0,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Remove hooks immediately after generation
        for h in handles:
            h.remove()

        correct = check_answer(response, expected)
        results.append(correct)
        responses.append(response.strip()[:40])

    accuracy = sum(results) / len(results)
    return accuracy, results, responses


# ---------------------------------------------------------------------------
# Phase 3: Logprob rejection baseline
# ---------------------------------------------------------------------------

def generate_with_logprobs(model, tokenizer, question, do_sample=False, temperature=1.0):
    """Generate answer and compute mean per-token logprob."""
    import torch

    prompt = f"Answer in one word or number only. {question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(max_new_tokens=20)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, **gen_kwargs,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0, prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    total_logprob = 0.0
    n_tokens = 0
    if hasattr(outputs, "scores") and outputs.scores:
        for step_idx, scores in enumerate(outputs.scores):
            if step_idx >= len(generated_ids):
                break
            token_id = generated_ids[step_idx].item()
            log_probs = torch.log_softmax(scores[0], dim=-1)
            total_logprob += log_probs[token_id].item()
            n_tokens += 1

    mean_logprob = total_logprob / max(n_tokens, 1)
    return response, mean_logprob


def logprob_rejection_baseline(model, tokenizer, questions, n_candidates=5):
    """Run logprob rejection sampling (experiment 13 approach) for comparison.

    Generates n_candidates per question with sampling, picks the one with
    the highest mean per-token logprob.

    Returns (accuracy, per-question results).
    """
    results = []
    for q, expected in questions:
        candidates = []
        for _ in range(n_candidates):
            response, mean_lp = generate_with_logprobs(
                model, tokenizer, q, do_sample=True, temperature=0.8,
            )
            candidates.append((response, mean_lp, check_answer(response, expected)))

        # Pick highest mean logprob (most confident)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_correct = candidates[0][2]
        results.append(best_correct)

    accuracy = sum(results) / len(results)
    return accuracy, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 18: Concept-Specific Activation Steering")
    print("  Goal: Beat logprob rejection (+10%) using confabulation steering")
    print("  Method: Subtract alpha * confabulation_dir during generation")
    print("  Key fix: float32 arithmetic (bfloat16 rounds small deltas to zero)")
    print("=" * 70)

    # Step 1: Load model
    print("\n--- Step 1: Loading Qwen3-0.6B ---")
    try:
        import torch
        import jax.numpy as jnp
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers/torch/jax not installed")
        return 1

    model_name = "Qwen/Qwen3-0.6B"
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Loaded in {time.time() - start:.1f}s ({n_layers} layers)")

    # Step 2: Greedy baseline
    print("\n--- Step 2: Greedy baseline on 25 QA pairs ---")
    greedy_results = []
    correct_acts = []
    hallucinated_acts = []

    for q, expected in QA_PAIRS:
        prompt = f"Answer in one word or number only. {q}"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, do_sample=False, temperature=1.0,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Extract activation from generated tokens (NOT prompt — lesson from exp 9)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            hidden = model(outputs[0].unsqueeze(0))
            hs = hidden.hidden_states
        act = hs[-1][0, prompt_len:, :].mean(dim=0).float().cpu().numpy()
        act_jax = jnp.array(act)

        is_correct = check_answer(response, expected)
        greedy_results.append(is_correct)
        if is_correct:
            correct_acts.append(act_jax)
        else:
            hallucinated_acts.append(act_jax)

        status = "✓" if is_correct else "✗"
        print(f"  [{status}] {q[:45]}... → {response.strip()[:30]}")

    greedy_acc = sum(greedy_results) / len(greedy_results)
    n_correct = len(correct_acts)
    n_halluc = len(hallucinated_acts)
    print(f"\nGreedy: {sum(greedy_results)}/{len(QA_PAIRS)} ({greedy_acc:.0%})")
    print(f"  Correct: {n_correct}, Hallucinated: {n_halluc}")

    if n_halluc < 2 or n_correct < 2:
        print("ERROR: Need at least 2 of each class for meaningful evaluation")
        return 1

    # Step 3: Compute concept-specific contrastive vectors
    print("\n--- Step 3: Computing concept-specific vectors (20 topics x 5 modes) ---")
    start = time.time()
    concept_activations = generate_concept_activations(model, tokenizer, n_topics=20)
    print(f"Concept generation done in {time.time() - start:.1f}s")
    for name, acts in concept_activations.items():
        print(f"  {name}: {len(acts)} samples")

    directions = compute_contrastive_directions(concept_activations)
    for name, d in directions.items():
        print(f"  {name}: shape={d.shape}, norm={float(jnp.linalg.norm(d)):.4f}")

    if not directions:
        print("ERROR: No contrastive directions computed")
        return 1

    # Step 4: Find best concept direction for hallucination separation
    print("\n--- Step 4: Evaluating concept directions on QA separation ---")
    best_name, best_direction, direction_evals = find_best_direction(
        directions, correct_acts, hallucinated_acts
    )

    print(f"  Direction ranking:")
    for r in direction_evals:
        marker = " ◄ BEST" if r["name"] == best_name else ""
        print(f"    {r['name']:<25} acc={r['accuracy']:.1%}  fisher={r['fisher']:.3f}{marker}")

    print(f"\n  Using '{best_name}' for steering")

    # Convert best direction to torch float32 for hook injection
    best_dir_torch = torch.tensor(best_direction.tolist(), dtype=torch.float32)

    # Step 5: Steering experiments across alphas and layer configs
    print("\n--- Step 5: Steering with concept-specific direction ---")
    print(f"  Direction: {best_name}")
    print(f"  Arithmetic: float32 (cast back to hidden dtype)")

    # Layer configurations to test:
    # - Upper half layers (where abstract representations live)
    # - Middle layers
    # - All layers
    upper_layers = list(range(n_layers * 3 // 4, n_layers))
    mid_layers = list(range(n_layers // 4, 3 * n_layers // 4))
    all_layers = list(range(n_layers))

    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]

    steering_results = []
    for alpha in alphas:
        for layer_label, layers in [
            ("upper", upper_layers),
            ("mid", mid_layers),
            ("all", all_layers),
        ]:
            config_name = f"α={alpha}, {layer_label} layers ({len(layers)}L)"
            acc, per_q, responses = steer_and_evaluate(
                model, tokenizer, QA_PAIRS, best_dir_torch, layers, alpha
            )
            delta = acc - greedy_acc
            fixes = sum(
                1 for g, s in zip(greedy_results, per_q) if not g and s
            )
            regressions = sum(
                1 for g, s in zip(greedy_results, per_q) if g and not s
            )
            steering_results.append({
                "config": config_name,
                "alpha": alpha,
                "layer_label": layer_label,
                "accuracy": acc,
                "delta": delta,
                "fixes": fixes,
                "regressions": regressions,
                "per_q": per_q,
            })
            sign = "+" if delta >= 0 else ""
            print(f"  {config_name:<35} {acc:.0%} ({sign}{delta:.0%})  "
                  f"fixes={fixes} reg={regressions}")

    # Step 6: Logprob rejection baseline for comparison
    print("\n--- Step 6: Logprob rejection baseline (experiment 13 approach) ---")
    logprob_acc, logprob_results = logprob_rejection_baseline(
        model, tokenizer, QA_PAIRS, n_candidates=5
    )
    logprob_delta = logprob_acc - greedy_acc
    logprob_fixes = sum(
        1 for g, r in zip(greedy_results, logprob_results) if not g and r
    )
    logprob_regressions = sum(
        1 for g, r in zip(greedy_results, logprob_results) if g and not r
    )
    print(f"  Logprob rejection (5 candidates): {logprob_acc:.0%} "
          f"({'+' if logprob_delta >= 0 else ''}{logprob_delta:.0%})  "
          f"fixes={logprob_fixes} reg={logprob_regressions}")

    # Step 7: Full results summary
    print(f"\n{'=' * 70}")
    print("RESULTS: Concept-Specific Steering vs Baselines")
    print(f"{'=' * 70}")
    print(f"  {'Config':<40} {'Acc':>5} {'Δ':>6} {'Fix':>4} {'Reg':>4} {'Net':>4}")
    print(f"  {'─' * 65}")
    print(f"  {'Greedy baseline':<40} {greedy_acc:>4.0%}   {'—':>5} {'—':>4} {'—':>4} {'—':>4}")
    print(f"  {'Logprob rejection (5 cands)':<40} {logprob_acc:>4.0%}  "
          f"{'+' if logprob_delta >= 0 else ''}{logprob_delta:>4.0%} "
          f"{logprob_fixes:>4} {logprob_regressions:>4} "
          f"{logprob_fixes - logprob_regressions:>+4}")
    print(f"  {'─' * 65}")

    # Sort steering results by accuracy
    steering_results.sort(key=lambda r: r["accuracy"], reverse=True)
    for r in steering_results:
        sign = "+" if r["delta"] >= 0 else ""
        net = r["fixes"] - r["regressions"]
        print(f"  {r['config']:<40} {r['accuracy']:>4.0%}  "
              f"{sign}{r['delta']:>4.0%} {r['fixes']:>4} "
              f"{r['regressions']:>4} {net:>+4}")

    # Find best steering config
    best_steer = steering_results[0]
    steer_vs_greedy = best_steer["delta"]
    steer_vs_logprob = best_steer["accuracy"] - logprob_acc

    print(f"\n  Best steering: {best_steer['config']}")
    print(f"    vs greedy:   {'+' if steer_vs_greedy >= 0 else ''}{steer_vs_greedy:.0%}")
    print(f"    vs logprob:  {'+' if steer_vs_logprob >= 0 else ''}{steer_vs_logprob:.0%}")

    # Per-question breakdown for best config
    print(f"\n{'─' * 70}")
    print(f"Per-question breakdown (best: {best_steer['config']}):")
    print(f"{'─' * 70}")
    for i, (q, expected) in enumerate(QA_PAIRS):
        g = "✓" if greedy_results[i] else "✗"
        s = "✓" if best_steer["per_q"][i] else "✗"
        l = "✓" if logprob_results[i] else "✗"
        tag = ""
        if best_steer["per_q"][i] and not greedy_results[i]:
            tag = " ★ FIXED"
        elif not best_steer["per_q"][i] and greedy_results[i]:
            tag = " ✖ REG"
        print(f"  [G:{g} S:{s} L:{l}] {q[:50]}{tag}")

    # Final verdict
    print(f"\n{'=' * 70}")
    if best_steer["accuracy"] > logprob_acc:
        print(
            f"SUCCESS: Concept-specific steering ({best_steer['accuracy']:.0%}) "
            f"beats logprob rejection ({logprob_acc:.0%})!"
        )
        print("Anthropic's insight confirmed: concept-specific directions ARE causal.")
    elif best_steer["accuracy"] > greedy_acc:
        print(
            f"PARTIAL: Steering ({best_steer['accuracy']:.0%}) beats greedy "
            f"({greedy_acc:.0%}) but not logprob ({logprob_acc:.0%})."
        )
        print("Concept direction has some causal effect, but not enough to beat")
        print("simple logprob selection. May need multi-layer targeting or")
        print("per-token steering (not just per-forward-pass).")
    elif best_steer["accuracy"] == greedy_acc:
        print(
            f"NEUTRAL: Best steering matches greedy baseline ({greedy_acc:.0%})."
        )
        print("Concept-specific direction does not causally affect generation on")
        print("this model/task. Possible causes:")
        print("  - Qwen3-0.6B too small for concept separation to be causal")
        print("  - Direction captures detection signal, not generation mechanism")
        print("  - Need per-token adaptive alpha, not constant steering")
    else:
        print(
            f"REGRESSION: Steering hurts ({best_steer['accuracy']:.0%} < "
            f"{greedy_acc:.0%} greedy)."
        )
        print("The concept direction disrupts generation. Possible causes:")
        print("  - Alpha too large → garbled outputs")
        print("  - Wrong layers targeted")
        print("  - Direction not orthogonal to useful representations")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

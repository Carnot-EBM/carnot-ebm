#!/usr/bin/env python3
"""Experiment: Concept-specific contrastive vectors for hallucination detection.

**Why this experiment exists:**
    Experiments 9-16 used a generic mean-difference direction (correct vs wrong
    answers) to detect hallucinations, achieving only ~64% accuracy. Anthropic's
    emotion research showed that concept-specific vectors extracted via targeted
    prompting ARE causally effective — you need to prompt the model to generate
    text in specific cognitive modes (certain, uncertain, confabulating, etc.),
    not just compare correct vs wrong answers.

**What this experiment does:**
    1. Loads Qwen3-0.6B and generates 20 responses per concept using targeted
       prompts that elicit specific cognitive modes (certain, uncertain,
       confabulating, reasoning, memorized).
    2. Extracts per-token activations from the LAST hidden layer for each
       response and averages them to get one vector per response.
    3. Computes contrastive vectors between concept pairs:
       - confabulation_dir = mean(confabulating) - mean(certain)
       - uncertainty_dir = mean(uncertain) - mean(certain)
       - reasoning_dir = mean(reasoning) - mean(certain)
       - memorized_dir = mean(memorized) - mean(certain)
    4. Tests each concept vector's ability to separate the 25 QA pairs
       (correct vs hallucinated answers) using Fisher discriminant ratio
       and classification accuracy.
    5. Compares results against the generic mean-difference direction from
       experiment 8 (which achieved ~64%).

**Key insight:**
    The model's activations when it's *prompted to confabulate* should occupy
    a different region of activation space than when it's *prompted to be
    certain*. The direction between these regions should be a better
    hallucination detector than the direction between correct and wrong answers,
    because it captures the *mechanism* of hallucination rather than just
    its surface correlation with answer correctness.

Usage:
    python scripts/experiment_concept_specific_vectors.py
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
# 20 diverse topics used to instantiate each concept prompt template.
# We want breadth across domains so the concept vectors capture the
# cognitive mode (certain, uncertain, etc.) rather than the topic.
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
# Concept prompt templates. {topic} and {fact} get filled from TOPICS.
# Each template is designed to push the model into a specific cognitive mode
# so that activations encode the *mode*, not just the topic.
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
# Same 25 QA pairs from experiment 8 (experiment_real_hallucination_detection.py).
# We reuse these to evaluate whether concept-specific vectors separate
# correct from hallucinated answers better than the generic direction.
# ---------------------------------------------------------------------------
QA_PAIRS = [
    ("What is the capital of France?", "Paris", "factual"),
    ("What is 2 + 2?", "4", "math"),
    ("What color is the sky on a clear day?", "blue", "factual"),
    ("How many legs does a dog have?", "4", "factual"),
    ("What is the chemical symbol for water?", "H2O", "factual"),
    ("What planet is closest to the Sun?", "Mercury", "factual"),
    ("What is the square root of 144?", "12", "math"),
    ("In what year did World War II end?", "1945", "factual"),
    ("What is the largest ocean on Earth?", "Pacific", "factual"),
    ("How many days are in a week?", "7", "factual"),
    ("What is the atomic number of carbon?", "6", "science"),
    ("Who wrote Romeo and Juliet?", "Shakespeare", "factual"),
    ("What is the speed of light in km/s approximately?", "300000", "science"),
    ("What is the derivative of x squared?", "2x", "math"),
    ("What gas do plants absorb from the atmosphere?", "CO2", "science"),
    ("What is the 15th prime number?", "47", "math"),
    ("What is 17 * 23?", "391", "math"),
    ("What is the integral of 1/x?", "ln|x|", "math"),
    ("What is the population of Iceland approximately?", "380000", "factual"),
    ("How many bones are in the adult human body?", "206", "science"),
    ("What is the sum of angles in a pentagon?", "540", "math"),
    ("What is 13 factorial?", "6227020800", "math"),
    ("What is the 8th Fibonacci number?", "21", "math"),
    ("What is the chemical formula for glucose?", "C6H12O6", "science"),
    ("What year was the Python programming language first released?", "1991", "factual"),
]


def check_answer(response: str, expected: str) -> bool:
    """Check if the model's response contains the expected answer."""
    return expected.lower() in response.lower().strip()


def extract_last_layer_activation(model, tokenizer, text: str):
    """Extract mean activation from the LAST hidden layer for a text.

    Runs the text through the model's forward pass, extracts hidden states
    from the final layer, and averages across all token positions to get
    a single vector representing the model's "state" for that text.

    Returns a JAX array of shape (hidden_dim,), or None on failure.
    All bfloat16 tensors are cast to float32 for numpy/jax compatibility.
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
    # Cast to float32 to handle bfloat16 models (bfloat16 -> numpy not supported).
    last_hidden = hidden_states[-1][0].float().mean(dim=0).cpu().numpy()
    return jnp.array(last_hidden)


def generate_concept_activations(model, tokenizer, n_topics: int = 20):
    """Generate text in each cognitive mode and extract last-layer activations.

    For each concept template x each topic, we:
    1. Fill the template with the topic
    2. Feed the filled prompt to the model (no generation — we use the
       prompt itself as the text, since the prompt IS the concept mode)
    3. Extract last-layer activations

    Returns dict mapping concept name -> list of activation vectors.
    """
    import torch

    concept_activations: dict[str, list] = {c: [] for c in CONCEPT_TEMPLATES}
    topics_to_use = TOPICS[:n_topics]

    total = len(CONCEPT_TEMPLATES) * len(topics_to_use)
    done = 0

    for concept_name, template in CONCEPT_TEMPLATES.items():
        for topic, fact in topics_to_use:
            text = template.format(topic=topic, fact=fact)

            # Also generate a continuation to capture the model's "mode"
            # after being primed with the concept prompt.
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

            # Extract activation from the FULL sequence (prompt + generated).
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            act = extract_last_layer_activation(model, tokenizer, full_text)
            if act is not None:
                concept_activations[concept_name].append(act)

            done += 1
            if done % 20 == 0:
                print(f"  Generated {done}/{total} concept samples...")

    return concept_activations


def compute_contrastive_pairs(concept_activations: dict[str, list]):
    """Compute contrastive vectors between each concept and 'certain'.

    Unlike the generic mean-difference direction (correct - wrong), these
    directions capture specific cognitive modes:
    - confabulation_dir: how confabulating differs from being certain
    - uncertainty_dir: how uncertainty differs from certainty
    - reasoning_dir: how step-by-step reasoning differs from certainty
    - memorized_dir: how rote recall differs from certainty

    The "certain" concept serves as the anchor/baseline because it
    represents the model's best behavior — confident and factual.

    Returns dict mapping direction name -> normalized JAX vector.
    """
    import jax.numpy as jnp

    # Compute per-concept means.
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


def evaluate_direction(
    direction,
    correct_acts: list,
    hallucinated_acts: list,
    direction_name: str,
):
    """Evaluate how well a direction separates correct from hallucinated.

    Computes:
    - Mean projection for correct and hallucinated sets
    - Fisher discriminant ratio (separation / spread)
    - Classification accuracy at the optimal threshold

    Returns a dict with all metrics.
    """
    import jax.numpy as jnp

    correct_projs = [float(jnp.dot(a, direction)) for a in correct_acts]
    halluc_projs = [float(jnp.dot(a, direction)) for a in hallucinated_acts]

    mean_correct = sum(correct_projs) / len(correct_projs)
    mean_halluc = sum(halluc_projs) / len(halluc_projs)

    # Standard deviations for Fisher ratio.
    import numpy as np
    std_correct = float(np.std(correct_projs)) if len(correct_projs) > 1 else 0.0
    std_halluc = float(np.std(halluc_projs)) if len(halluc_projs) > 1 else 0.0

    fisher = abs(mean_halluc - mean_correct) / (std_correct + std_halluc + 1e-8)

    # Classification accuracy at midpoint threshold.
    threshold = (mean_correct + mean_halluc) / 2
    # Determine polarity: does hallucinated project higher or lower?
    if mean_halluc > mean_correct:
        tp = sum(1 for p in halluc_projs if p > threshold)
        tn = sum(1 for p in correct_projs if p <= threshold)
    else:
        tp = sum(1 for p in halluc_projs if p < threshold)
        tn = sum(1 for p in correct_projs if p >= threshold)

    accuracy = (tp + tn) / (len(correct_projs) + len(halluc_projs))

    return {
        "name": direction_name,
        "mean_correct": mean_correct,
        "mean_halluc": mean_halluc,
        "gap": mean_halluc - mean_correct,
        "fisher": fisher,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "n_correct": len(correct_projs),
        "n_halluc": len(halluc_projs),
    }


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT: Concept-Specific Contrastive Vectors")
    print("  Goal: Beat the generic mean-difference direction (~64%)")
    print("  Method: Targeted prompts → concept activations → contrastive dirs")
    print("=" * 70)

    # Step 1: Load model.
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
    print(f"Loaded in {time.time() - start:.1f}s")

    # Step 2: Generate concept activations (20 topics x 5 concepts = 100 samples).
    print("\n--- Step 2: Generating concept activations (20 topics x 5 concepts) ---")
    start = time.time()
    concept_activations = generate_concept_activations(model, tokenizer, n_topics=20)
    print(f"Generation done in {time.time() - start:.1f}s")
    for name, acts in concept_activations.items():
        print(f"  {name}: {len(acts)} samples")

    # Step 3: Compute contrastive directions.
    print("\n--- Step 3: Computing contrastive directions ---")
    directions = compute_contrastive_pairs(concept_activations)
    for name, d in directions.items():
        print(f"  {name}: shape={d.shape}, norm={float(jnp.linalg.norm(d)):.4f}")

    if not directions:
        print("ERROR: No contrastive directions computed")
        return 1

    # Step 4: Ask QA pairs and collect correct/hallucinated activations.
    print("\n--- Step 4: Running 25 QA pairs (baseline from experiment 8) ---")
    correct_acts = []
    hallucinated_acts = []
    results = []

    for question, expected, category in QA_PAIRS:
        prompt = f"Answer in one word or number only. {question}"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, do_sample=False, temperature=1.0,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Extract last-layer activation from the prompt (pre-generation state).
        with torch.no_grad():
            hidden_outputs = model(**inputs)
            hidden_states = hidden_outputs.hidden_states
        # Last layer, mean across tokens, cast bfloat16 -> float32.
        act = hidden_states[-1][0].float().mean(dim=0).cpu().numpy()
        act_jax = jnp.array(act)

        is_correct = check_answer(response, expected)
        if is_correct:
            correct_acts.append(act_jax)
        else:
            hallucinated_acts.append(act_jax)

        status = "✓" if is_correct else "✗"
        results.append({"q": question, "correct": is_correct, "response": response.strip()[:40]})
        print(f"  [{status}] {question[:45]}... → {response.strip()[:30]}")

    n_correct = len(correct_acts)
    n_halluc = len(hallucinated_acts)
    print(f"\nModel: {n_correct} correct, {n_halluc} hallucinated out of {len(QA_PAIRS)}")

    if n_halluc < 2 or n_correct < 2:
        print("ERROR: Need at least 2 of each class for meaningful evaluation")
        return 1

    # Step 5: Compute generic mean-difference direction (experiment 8 baseline).
    print("\n--- Step 5: Computing generic mean-difference direction (baseline) ---")
    from carnot.embeddings.hallucination_direction import (
        HallucinationDirectionConfig,
        find_hallucination_direction,
    )
    generic_dir = find_hallucination_direction(
        jnp.stack(correct_acts),
        jnp.stack(hallucinated_acts),
        HallucinationDirectionConfig(normalize=True),
    )

    # Step 6: Evaluate ALL directions on the QA separation task.
    print("\n--- Step 6: Evaluating all directions on QA separation ---")
    all_results = []

    # Generic baseline.
    generic_eval = evaluate_direction(generic_dir, correct_acts, hallucinated_acts, "generic_mean_diff")
    all_results.append(generic_eval)

    # Concept-specific directions.
    for dir_name, direction in directions.items():
        eval_result = evaluate_direction(direction, correct_acts, hallucinated_acts, dir_name)
        all_results.append(eval_result)

    # Step 7: Report results.
    print(f"\n{'=' * 70}")
    print("RESULTS: Concept-Specific vs Generic Direction")
    print(f"{'=' * 70}")
    print(f"  {'Direction':<25} {'Accuracy':>8} {'Fisher':>8} {'Gap':>8} {'TP/FP':>10}")
    print(f"  {'─' * 62}")

    # Sort by accuracy descending.
    all_results.sort(key=lambda r: r["accuracy"], reverse=True)

    for r in all_results:
        marker = " ◄ BASELINE" if r["name"] == "generic_mean_diff" else ""
        print(
            f"  {r['name']:<25} {r['accuracy']:>7.1%} {r['fisher']:>8.3f} "
            f"{r['gap']:>+8.4f} {r['tp']}/{r['n_halluc']}:{r['tn']}/{r['n_correct']}"
            f"{marker}"
        )

    best = all_results[0]
    baseline = next(r for r in all_results if r["name"] == "generic_mean_diff")
    delta = best["accuracy"] - baseline["accuracy"]

    print(f"\n  Best direction:   {best['name']} at {best['accuracy']:.1%}")
    print(f"  Generic baseline: {baseline['accuracy']:.1%} (experiment 8 reference: ~64%)")
    print(f"  Improvement:      {'+' if delta >= 0 else ''}{delta:.1%}")

    # Step 8: Per-concept energy breakdown for a few example QA pairs.
    print(f"\n{'─' * 70}")
    print("Per-concept energy scores (first 5 correct + 5 hallucinated):")
    print(f"{'─' * 70}")
    print(f"  {'Q':<30} {'Correct?':>8}", end="")
    for dir_name in directions:
        short = dir_name.replace("_dir", "")[:8]
        print(f" {short:>8}", end="")
    print()

    shown_correct = 0
    shown_halluc = 0
    for r_info, (question, expected, _cat) in zip(results, QA_PAIRS):
        if r_info["correct"] and shown_correct >= 5:
            continue
        if not r_info["correct"] and shown_halluc >= 5:
            continue

        # Re-extract activation for this specific question.
        prompt = f"Answer in one word or number only. {question}"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            hidden_outputs = model(**inputs)
            hs = hidden_outputs.hidden_states
        act = jnp.array(hs[-1][0].float().mean(dim=0).cpu().numpy())

        label = "✓" if r_info["correct"] else "✗"
        print(f"  {question[:30]:<30} {label:>8}", end="")
        for dir_name, direction in directions.items():
            score = float(jnp.dot(act, direction))
            print(f" {score:>+8.3f}", end="")
        print()

        if r_info["correct"]:
            shown_correct += 1
        else:
            shown_halluc += 1

    # Final verdict.
    print(f"\n{'=' * 70}")
    if best["accuracy"] > baseline["accuracy"]:
        print(
            f"SUCCESS: Concept-specific vector '{best['name']}' ({best['accuracy']:.1%}) "
            f"beats generic mean-difference ({baseline['accuracy']:.1%}) by {delta:+.1%}"
        )
        print("Anthropic's insight confirmed: targeted cognitive-mode prompting")
        print("yields better hallucination detectors than generic correct/wrong contrast.")
    elif best["accuracy"] == baseline["accuracy"] and best["name"] != "generic_mean_diff":
        print(
            f"TIE: Best concept vector '{best['name']}' matches generic baseline "
            f"at {best['accuracy']:.1%}. May need more topics or layers."
        )
    else:
        print(
            f"NEGATIVE: Generic direction ({baseline['accuracy']:.1%}) still best. "
            f"Best concept vector: {best['name']} at {best['accuracy']:.1%}."
        )
        print("Possible causes: too few topics, single-layer extraction,")
        print("or Qwen3-0.6B lacks the concept separation seen in larger models.")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

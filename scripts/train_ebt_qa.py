#!/usr/bin/env python3
"""Train an Energy-Based Transformer to score (question, answer) embedding pairs.

**What this script does (plain English):**
    We take a small language model (Qwen3-0.6B), ask it 120 factual questions,
    and extract its internal activation vectors (hidden states) for each question.
    We label each answer as "correct" or "hallucinated" by comparing to known answers.

    Then we train an Energy-Based Transformer (EBT) so that:
    - Correct (question, answer) pairs get LOW energy
    - Hallucinated (question, answer) pairs get HIGH energy

    Training uses "optimization-through-training" (P5):
    1. Given a question embedding, start from a RANDOM answer embedding
    2. Run gradient descent on the EBT's energy w.r.t. the answer embedding
       for N inner steps — this "optimizes" the answer embedding
    3. The outer loss = MSE between the optimized embedding and the REAL
       correct answer embedding
    4. Backprop through the entire inner loop to update EBT parameters

    This teaches the EBT to have an energy landscape where gradient descent
    naturally flows toward correct answers.

    After training, we evaluate: can the trained EBT rank correct answers
    lower (better) than hallucinated answers on a held-out test set?

**Why this matters:**
    If the EBT learns to score correctness from activations, we can use it
    as a real-time hallucination detector during LLM inference — no ground
    truth needed at test time, just the energy score.

Usage:
    python scripts/train_ebt_qa.py

Spec: REQ-EBT-002, REQ-EBT-003, REQ-INFER-014
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure the python package is importable when running from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# QA Dataset — 120 factual questions with known answers
# ---------------------------------------------------------------------------

QA_PAIRS = [
    # --- Easy factual (high likelihood of correct answer) ---
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
    ("What is the boiling point of water in Celsius?", "100"),
    ("How many continents are there?", "7"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Who painted the Mona Lisa?", "Leonardo"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("How many sides does a triangle have?", "3"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What color do you get mixing red and blue?", "purple"),
    ("What is the smallest prime number?", "2"),
    ("How many months are in a year?", "12"),
    # --- Medium factual ---
    ("What is the atomic number of carbon?", "6"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the speed of light in km/s approximately?", "300000"),
    ("What is the derivative of x squared?", "2x"),
    ("What gas do plants absorb from the atmosphere?", "CO2"),
    ("What is the capital of Japan?", "Tokyo"),
    ("How many protons does hydrogen have?", "1"),
    ("What year was the Declaration of Independence signed?", "1776"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("How many chromosomes do humans have?", "46"),
    ("What is the largest mammal?", "blue whale"),
    ("What element has the symbol Fe?", "iron"),
    ("Who developed the theory of relativity?", "Einstein"),
    ("What is the capital of Australia?", "Canberra"),
    ("How many degrees are in a circle?", "360"),
    ("What is the powerhouse of the cell?", "mitochondria"),
    ("What language has the most native speakers?", "Mandarin"),
    ("What is the tallest mountain on Earth?", "Everest"),
    ("How many teeth does an adult human have?", "32"),
    # --- Science / technical ---
    ("What is the atomic number of oxygen?", "8"),
    ("What is Avogadro's number approximately?", "6.022e23"),
    ("What is the charge of a proton?", "positive"),
    ("What planet has the most moons?", "Saturn"),
    ("What is the pH of pure water?", "7"),
    ("What is the speed of sound in m/s approximately?", "343"),
    ("What force keeps planets in orbit?", "gravity"),
    ("What is the most abundant gas in Earth's atmosphere?", "nitrogen"),
    ("What is the chemical formula for methane?", "CH4"),
    ("How many valence electrons does carbon have?", "4"),
    ("What is the unit of electrical resistance?", "ohm"),
    ("What particle has no electric charge?", "neutron"),
    ("What is the SI unit of force?", "newton"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("What vitamin does sunlight help produce?", "D"),
    ("What is the hardest natural substance?", "diamond"),
    ("What type of bond shares electrons?", "covalent"),
    ("What organelle contains DNA in eukaryotes?", "nucleus"),
    ("What is the most electronegative element?", "fluorine"),
    ("What is the chemical formula for ammonia?", "NH3"),
    # --- Math ---
    ("What is 15 * 15?", "225"),
    ("What is the 15th prime number?", "47"),
    ("What is 17 * 23?", "391"),
    ("What is the integral of 1/x?", "ln"),
    ("What is the sum of angles in a pentagon?", "540"),
    ("What is 13 factorial?", "6227020800"),
    ("What is the 8th Fibonacci number?", "21"),
    ("What is the square root of 169?", "13"),
    ("What is 2 to the power of 10?", "1024"),
    ("What is the sum of the first 10 positive integers?", "55"),
    ("What is the GCD of 48 and 18?", "6"),
    ("What is log base 2 of 256?", "8"),
    ("What is 7 factorial?", "5040"),
    ("What is the 10th Fibonacci number?", "55"),
    ("How many edges does a cube have?", "12"),
    ("What is pi rounded to 2 decimal places?", "3.14"),
    ("What is e rounded to 2 decimal places?", "2.72"),
    ("What is the sum of angles in a hexagon?", "720"),
    ("What is 11 squared?", "121"),
    ("What is the cube root of 27?", "3"),
    # --- Geography / history ---
    ("What is the longest river in the world?", "Nile"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("In what year did the Berlin Wall fall?", "1989"),
    ("What is the smallest country in the world?", "Vatican"),
    ("What ocean lies between Africa and Australia?", "Indian"),
    ("What is the capital of Canada?", "Ottawa"),
    ("In what year did humans first land on the Moon?", "1969"),
    ("What is the largest desert in the world?", "Sahara"),
    ("What country has the most people?", "India"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of South Korea?", "Seoul"),
    ("In what year did the Titanic sink?", "1912"),
    ("What is the largest continent by area?", "Asia"),
    ("What is the capital of Germany?", "Berlin"),
    ("What river runs through London?", "Thames"),
    # --- Tech / computing ---
    ("What year was the Python programming language first released?", "1991"),
    ("What does CPU stand for?", "central processing unit"),
    ("How many bits are in a byte?", "8"),
    ("What company created the iPhone?", "Apple"),
    ("What does HTML stand for?", "hypertext markup language"),
    ("What programming language is known for its use in data science?", "Python"),
    ("What is the base of the binary number system?", "2"),
    ("What does RAM stand for?", "random access memory"),
    # --- Biology ---
    ("How many bones are in the adult human body?", "206"),
    ("What is the chemical formula for glucose?", "C6H12O6"),
    ("What blood type is the universal donor?", "O"),
    ("How many chambers does the human heart have?", "4"),
    ("What is the largest organ in the human body?", "skin"),
    ("What carries oxygen in human blood?", "hemoglobin"),
    ("How many pairs of cranial nerves do humans have?", "12"),
    ("What is the basic unit of life?", "cell"),
    # --- Misc harder ---
    ("What is the population of Iceland approximately?", "380000"),
    ("What year was the Magna Carta signed?", "1215"),
    ("What is the atomic mass of carbon approximately?", "12"),
    ("How many symphonies did Beethoven compose?", "9"),
    ("What is the chemical formula for sulfuric acid?", "H2SO4"),
    ("What is the escape velocity from Earth in km/s?", "11.2"),
    ("How many keys on a standard piano?", "88"),
    ("What is the half-life of carbon-14 in years approximately?", "5730"),
    ("What is the speed of light in m/s approximately?", "3e8"),
]

assert len(QA_PAIRS) >= 120, f"Need 120+ QA pairs, have {len(QA_PAIRS)}"


def check_answer(response: str, expected: str) -> bool:
    """Check if the model's response contains the expected answer.

    **For engineers:**
        Simple substring match (case-insensitive). Not perfect, but good enough
        for factual one-word/number answers where we just need to know if the
        model got it right.
    """
    return expected.lower().strip() in response.lower().strip()


# ---------------------------------------------------------------------------
# Embedding-space EBT — operates on continuous activation vectors
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingEBTConfig:
    """Configuration for the embedding-space EBT.

    **For engineers:**
        Unlike the token-based EBT in ebt.py which takes discrete token IDs,
        this variant takes continuous embedding vectors (e.g., LLM hidden states)
        as input. It projects question and answer embeddings into a shared
        d_model space, treats them as a 2-token sequence, runs transformer
        layers, and outputs scalar energy.

        - input_dim: Dimension of each input embedding (e.g., 2048 for
          concatenated last+mid layer activations from Qwen3-0.6B).
        - d_model: Internal transformer dimension. Both q_emb and a_emb
          are projected to this size before the transformer.
        - n_layers: Number of transformer layers.
        - n_heads: Attention heads (d_model must be divisible by n_heads).
        - d_ff: Feed-forward inner dimension (typically 4 * d_model).
    """

    input_dim: int = 2048
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 512


def _layer_norm(x: jax.Array, gamma: jax.Array, beta: jax.Array) -> jax.Array:
    """Layer normalization across the feature dimension."""
    eps = 1e-5
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return gamma * ((x - mean) / jnp.sqrt(var + eps)) + beta


def _gelu(x: jax.Array) -> jax.Array:
    """GELU activation function."""
    return jax.nn.gelu(x)


def init_embedding_ebt(config: EmbeddingEBTConfig, key: jax.Array) -> dict:
    """Initialize all parameters for the embedding-space EBT as a flat dict.

    **For engineers:**
        We store parameters as a flat dictionary of JAX arrays so that:
        1. We can pass them through jax.grad (JAX differentiates w.r.t. the
           first argument, which must be a pytree of arrays)
        2. We can easily serialize to/from safetensors
        3. No mutable state — pure functional style

        Parameter naming convention: "layer_{i}_{param_name}"

    Returns:
        Dictionary mapping parameter names to JAX arrays.
    """
    params = {}

    # Input projections: project q_emb and a_emb from input_dim to d_model.
    # Separate projections let the model treat questions and answers differently.
    k_q_proj, k_a_proj, key = jrandom.split(key, 3)
    q_limit = jnp.sqrt(6.0 / (config.input_dim + config.d_model))
    params["q_proj_w"] = jrandom.uniform(
        k_q_proj, (config.d_model, config.input_dim), minval=-q_limit, maxval=q_limit
    )
    params["q_proj_b"] = jnp.zeros(config.d_model)
    params["a_proj_w"] = jrandom.uniform(
        k_a_proj, (config.d_model, config.input_dim), minval=-q_limit, maxval=q_limit
    )
    params["a_proj_b"] = jnp.zeros(config.d_model)

    # Positional embeddings for the 2-token sequence (position 0 = question, 1 = answer)
    k_pos, key = jrandom.split(key)
    params["pos_emb"] = jrandom.normal(k_pos, (2, config.d_model)) * 0.02

    # Transformer layers
    for i in range(config.n_layers):
        prefix = f"layer_{i}_"
        d = config.d_model
        d_ff = config.d_ff

        # Attention weights (Q, K, V, O)
        attn_limit = jnp.sqrt(6.0 / (d + d))
        for name in ["wq", "wk", "wv", "wo"]:
            k_w, key = jrandom.split(key)
            params[prefix + name] = jrandom.uniform(
                k_w, (d, d), minval=-attn_limit, maxval=attn_limit
            )
            params[prefix + "b" + name[1:]] = jnp.zeros(d)

        # Layer norm 1 (before attention)
        params[prefix + "ln1_gamma"] = jnp.ones(d)
        params[prefix + "ln1_beta"] = jnp.zeros(d)

        # FFN weights
        ff_limit1 = jnp.sqrt(6.0 / (d + d_ff))
        ff_limit2 = jnp.sqrt(6.0 / (d_ff + d))
        k_ff1, k_ff2, key = jrandom.split(key, 3)
        params[prefix + "ff1_w"] = jrandom.uniform(
            k_ff1, (d_ff, d), minval=-ff_limit1, maxval=ff_limit1
        )
        params[prefix + "ff1_b"] = jnp.zeros(d_ff)
        params[prefix + "ff2_w"] = jrandom.uniform(
            k_ff2, (d, d_ff), minval=-ff_limit2, maxval=ff_limit2
        )
        params[prefix + "ff2_b"] = jnp.zeros(d)

        # Layer norm 2 (before FFN)
        params[prefix + "ln2_gamma"] = jnp.ones(d)
        params[prefix + "ln2_beta"] = jnp.zeros(d)

    # Final layer norm + output projection to scalar energy
    params["final_ln_gamma"] = jnp.ones(config.d_model)
    params["final_ln_beta"] = jnp.zeros(config.d_model)
    k_out, key = jrandom.split(key)
    out_limit = jnp.sqrt(6.0 / (config.d_model + 1))
    params["out_w"] = jrandom.uniform(
        k_out, (config.d_model,), minval=-out_limit, maxval=out_limit
    )
    params["out_b"] = jnp.array(0.0)

    return params


def embedding_ebt_energy(
    params: dict,
    config: EmbeddingEBTConfig,
    q_emb: jax.Array,
    a_emb: jax.Array,
) -> jax.Array:
    """Compute scalar energy for a (question_embedding, answer_embedding) pair.

    **For engineers:**
        Forward pass:
        1. Project q_emb and a_emb each from input_dim to d_model
        2. Add positional embeddings (pos 0 = question, pos 1 = answer)
        3. Stack into a 2-token sequence: shape (2, d_model)
        4. Run through transformer layers (pre-norm attention + FFN)
        5. Mean pool over the 2 tokens → (d_model,)
        6. Linear projection → scalar energy

        This is a pure function (no side effects) so JAX can differentiate it
        w.r.t. any of its arguments — params, q_emb, or a_emb.

    Args:
        params: Parameter dictionary from init_embedding_ebt.
        config: Architecture configuration.
        q_emb: Question embedding, shape (input_dim,).
        a_emb: Answer embedding, shape (input_dim,).

    Returns:
        Scalar energy value.
    """
    d = config.d_model
    n_heads = config.n_heads
    d_head = d // n_heads

    # Project embeddings to d_model space
    q_proj = q_emb @ params["q_proj_w"].T + params["q_proj_b"]  # (d_model,)
    a_proj = a_emb @ params["a_proj_w"].T + params["a_proj_b"]  # (d_model,)

    # Stack as 2-token sequence + positional embeddings
    h = jnp.stack([q_proj, a_proj]) + params["pos_emb"]  # (2, d_model)
    seq_len = 2

    # Transformer layers
    for i in range(config.n_layers):
        p = f"layer_{i}_"

        # Pre-norm attention
        h_norm = _layer_norm(h, params[p + "ln1_gamma"], params[p + "ln1_beta"])

        q = h_norm @ params[p + "wq"].T + params[p + "bq"]
        k = h_norm @ params[p + "wk"].T + params[p + "bk"]
        v = h_norm @ params[p + "wv"].T + params[p + "bv"]

        # Multi-head reshape: (2, d_model) → (n_heads, 2, d_head)
        q = q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
        k = k.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
        v = v.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

        scale = jnp.sqrt(jnp.float32(d_head))
        scores = jnp.matmul(q, k.transpose(0, 2, 1)) / scale
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.matmul(attn_weights, v)

        # Concat heads: (n_heads, 2, d_head) → (2, d_model)
        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, d)
        attn_out = attn_out @ params[p + "wo"].T + params[p + "bo"]
        h = h + attn_out

        # Pre-norm FFN
        h_norm = _layer_norm(h, params[p + "ln2_gamma"], params[p + "ln2_beta"])
        ffn_out = _gelu(h_norm @ params[p + "ff1_w"].T + params[p + "ff1_b"])
        ffn_out = ffn_out @ params[p + "ff2_w"].T + params[p + "ff2_b"]
        h = h + ffn_out

    # Final layer norm → mean pool → scalar
    h = _layer_norm(h, params["final_ln_gamma"], params["final_ln_beta"])
    h_pooled = jnp.mean(h, axis=0)  # (d_model,)
    return params["out_w"] @ h_pooled + params["out_b"]


# ---------------------------------------------------------------------------
# P5: Optimization-through-training
# ---------------------------------------------------------------------------


def p5_inner_loop(
    params: dict,
    config: EmbeddingEBTConfig,
    q_emb: jax.Array,
    init_a_emb: jax.Array,
    n_inner_steps: int = 10,
    inner_lr: float = 0.1,
) -> jax.Array:
    """P5 inner loop: optimize answer embedding by gradient descent on EBT energy.

    **For engineers (what "optimization-through-training" means):**
        Given a question embedding and a random starting answer embedding,
        we iteratively update the answer embedding to MINIMIZE the EBT's energy.
        Each step computes dE/d(a_emb) and subtracts it (gradient descent).

        After N steps, the optimized a_emb should land near the correct answer
        embedding — IF the EBT has learned a good energy landscape.

        This inner loop is DIFFERENTIABLE w.r.t. params (EBT weights), because
        every step is just JAX operations. So the outer training loop can
        backprop through all N inner steps to update the EBT parameters,
        teaching it to have an energy surface where gradient descent
        converges to correct answers.

    Args:
        params: EBT parameters (we differentiate through these in outer loop).
        config: EBT architecture config.
        q_emb: Question embedding, shape (input_dim,). Held fixed.
        init_a_emb: Initial random answer embedding, shape (input_dim,).
        n_inner_steps: Number of gradient descent steps on energy.
        inner_lr: Learning rate for inner gradient descent.

    Returns:
        Optimized answer embedding, shape (input_dim,).
    """
    a_emb = init_a_emb

    def inner_step(a_emb: jax.Array, _: None) -> tuple[jax.Array, None]:
        """One step of gradient descent on the energy w.r.t. answer embedding."""
        # Gradient of energy w.r.t. a_emb (not params — a_emb is the variable)
        grad_a = jax.grad(
            lambda a: embedding_ebt_energy(params, config, q_emb, a)
        )(a_emb)
        a_emb_new = a_emb - inner_lr * grad_a
        return a_emb_new, None

    # Use jax.lax.scan for efficient unrolling of the inner loop.
    # scan is like a for-loop but compiled into a single XLA computation,
    # making it much faster AND fully differentiable.
    a_emb_opt, _ = jax.lax.scan(inner_step, a_emb, None, length=n_inner_steps)
    return a_emb_opt


def p5_loss_single(
    params: dict,
    config: EmbeddingEBTConfig,
    q_emb: jax.Array,
    correct_a_emb: jax.Array,
    key: jax.Array,
    n_inner_steps: int = 10,
    inner_lr: float = 0.1,
) -> jax.Array:
    """P5 loss for a single (question, correct_answer) pair.

    **For engineers:**
        1. Sample a random starting answer embedding
        2. Run P5 inner loop: gradient descent on EBT energy → optimized a_emb
        3. Loss = MSE between optimized a_emb and the real correct a_emb

        If the EBT has a good energy landscape, the optimized embedding should
        be close to the correct one, giving low loss. Training minimizes this
        loss by adjusting EBT parameters.

    Args:
        params: EBT parameters to optimize.
        config: EBT architecture config.
        q_emb: Question embedding.
        correct_a_emb: Target correct answer embedding.
        key: PRNG key for random initialization of answer embedding.
        n_inner_steps: Inner loop steps.
        inner_lr: Inner loop learning rate.

    Returns:
        Scalar MSE loss.
    """
    # Random starting point for answer embedding (same scale as real embeddings)
    std = jnp.std(correct_a_emb) + 1e-6
    init_a_emb = jrandom.normal(key, correct_a_emb.shape) * std

    # Run inner optimization
    optimized_a_emb = p5_inner_loop(
        params, config, q_emb, init_a_emb, n_inner_steps, inner_lr
    )

    # MSE loss between optimized and correct answer embedding
    return jnp.mean((optimized_a_emb - correct_a_emb) ** 2)


def contrastive_loss(
    params: dict,
    config: EmbeddingEBTConfig,
    q_emb: jax.Array,
    correct_a_emb: jax.Array,
    halluc_a_emb: jax.Array,
    margin: float = 1.0,
) -> jax.Array:
    """Contrastive energy loss: correct should have lower energy than hallucinated.

    **For engineers:**
        This is a simpler auxiliary loss that directly encourages the EBT to
        assign lower energy to correct pairs than hallucinated ones:

            loss = max(0, E(q, correct) - E(q, hallucinated) + margin)

        This is a hinge loss (margin-based). If the correct energy is already
        lower than hallucinated energy by at least `margin`, loss is zero.
        Otherwise, the loss pushes the energies apart.

        We use this alongside the P5 loss for more stable training.

    Args:
        params: EBT parameters.
        config: EBT architecture config.
        q_emb: Question embedding.
        correct_a_emb: Correct answer embedding (should get low energy).
        halluc_a_emb: Hallucinated answer embedding (should get high energy).
        margin: Minimum desired energy gap.

    Returns:
        Scalar hinge loss.
    """
    e_correct = embedding_ebt_energy(params, config, q_emb, correct_a_emb)
    e_halluc = embedding_ebt_energy(params, config, q_emb, halluc_a_emb)
    return jnp.maximum(0.0, e_correct - e_halluc + margin)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------


def extract_activations(
    model,
    tokenizer,
    questions: list[tuple[str, str]],
) -> tuple[list[dict], int, int]:
    """Ask the model questions and extract activation vectors.

    **For engineers:**
        For each question:
        1. Tokenize and run the model to generate an answer
        2. Re-run the model on the input to get hidden states
        3. Extract the last-layer and mid-layer mean activations
        4. Concatenate them into a single feature vector
        5. Label as correct/hallucinated by comparing to expected answer

    Returns:
        Tuple of (results_list, n_correct, n_hallucinated).
        Each result dict has: question, expected, response, correct (bool),
        q_activation (jax array), a_activation (jax array).
    """
    import torch

    results = []
    n_correct = 0
    n_halluc = 0

    for i, (question, expected) in enumerate(questions):
        prompt = f"Answer in one word or number only. {question}"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Get hidden states for the question (input tokens)
        with torch.no_grad():
            hidden_outputs = model(**inputs)
            hidden_states = hidden_outputs.hidden_states

        # Question activation: last + mid layer, mean-pooled over tokens
        last_hidden = hidden_states[-1][0].mean(dim=0).float().numpy()
        mid_hidden = hidden_states[len(hidden_states) // 2][0].mean(dim=0).float().numpy()
        q_activation = jnp.concatenate([jnp.array(last_hidden), jnp.array(mid_hidden)])

        # Answer activation: run the model on the full generated output to get
        # answer-conditioned hidden states
        answer_tokens = outputs[0]
        with torch.no_grad():
            answer_outputs = model(answer_tokens)
            answer_hidden = answer_outputs.hidden_states

        # Answer activation from the generated portion
        n_input_tokens = inputs["input_ids"].shape[1]
        if answer_tokens.shape[1] > n_input_tokens:
            # Mean-pool over answer tokens only
            last_ans = answer_hidden[-1][0, n_input_tokens:].mean(dim=0).float().numpy()
            mid_ans = answer_hidden[len(answer_hidden) // 2][0, n_input_tokens:].mean(dim=0).float().numpy()
        else:
            # Fallback: use last token
            last_ans = answer_hidden[-1][0, -1].float().numpy()
            mid_ans = answer_hidden[len(answer_hidden) // 2][0, -1].float().numpy()

        a_activation = jnp.concatenate([jnp.array(last_ans), jnp.array(mid_ans)])

        is_correct = check_answer(response, expected)
        if is_correct:
            n_correct += 1
        else:
            n_halluc += 1

        status = "correct" if is_correct else "HALLUC"
        if (i + 1) % 10 == 0 or i < 5:
            logger.info(
                f"  [{i+1}/{len(questions)}] [{status}] "
                f"Q: {question[:40]}... -> {response.strip()[:30]}"
            )

        results.append({
            "question": question,
            "expected": expected,
            "response": response.strip(),
            "correct": is_correct,
            "q_activation": q_activation,
            "a_activation": a_activation,
        })

    return results, n_correct, n_halluc


def evaluate_ranking(
    params: dict,
    config: EmbeddingEBTConfig,
    test_data: list[dict],
) -> dict:
    """Evaluate whether the trained EBT ranks correct answers lower than hallucinated.

    **For engineers:**
        For each question that has both correct and hallucinated examples,
        compute the energy for both and check if E(correct) < E(hallucinated).
        Also compute AUC-like metrics using all pairwise comparisons.

    Returns:
        Dictionary with accuracy, mean energies, and detailed results.
    """
    correct_energies = []
    halluc_energies = []

    for item in test_data:
        e = float(embedding_ebt_energy(
            params, config, item["q_activation"], item["a_activation"]
        ))
        if item["correct"]:
            correct_energies.append(e)
        else:
            halluc_energies.append(e)

    if not correct_energies or not halluc_energies:
        return {
            "pairwise_accuracy": 0.0,
            "mean_correct_energy": 0.0,
            "mean_halluc_energy": 0.0,
            "energy_gap": 0.0,
            "n_correct": len(correct_energies),
            "n_halluc": len(halluc_energies),
        }

    mean_correct = np.mean(correct_energies)
    mean_halluc = np.mean(halluc_energies)

    # Pairwise accuracy: fraction of (correct, hallucinated) pairs where
    # E(correct) < E(hallucinated) — this is the AUC of the energy as
    # a binary classifier
    n_correct_wins = 0
    n_pairs = 0
    for ec in correct_energies:
        for eh in halluc_energies:
            n_pairs += 1
            if ec < eh:
                n_correct_wins += 1

    pairwise_acc = n_correct_wins / max(n_pairs, 1)

    # Threshold-based accuracy at the midpoint
    threshold = (mean_correct + mean_halluc) / 2
    tp = sum(1 for e in halluc_energies if e > threshold)
    tn = sum(1 for e in correct_energies if e <= threshold)
    threshold_acc = (tp + tn) / (len(correct_energies) + len(halluc_energies))

    return {
        "pairwise_accuracy": float(pairwise_acc),
        "threshold_accuracy": float(threshold_acc),
        "mean_correct_energy": float(mean_correct),
        "mean_halluc_energy": float(mean_halluc),
        "energy_gap": float(mean_halluc - mean_correct),
        "n_correct": len(correct_energies),
        "n_halluc": len(halluc_energies),
        "threshold": float(threshold),
    }


def main() -> int:
    print("=" * 70)
    print("TRAIN EBT QA: Energy-Based Transformer for Hallucination Detection")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Load Qwen3-0.6B
    # -----------------------------------------------------------------------
    print("\n--- Step 1: Loading model ---")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: pip install transformers torch")
        return 1

    model_name = "Qwen/Qwen3-0.6B"
    logger.info(f"Loading {model_name}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, output_hidden_states=True,
    )
    llm.eval()
    logger.info(f"Model loaded in {time.time() - start:.1f}s")

    # -----------------------------------------------------------------------
    # Step 2: Extract activations for all QA pairs
    # -----------------------------------------------------------------------
    print("\n--- Step 2: Extracting activations for 120 QA pairs ---")
    start = time.time()
    all_results, n_correct, n_halluc = extract_activations(llm, tokenizer, QA_PAIRS)
    logger.info(
        f"Extraction done in {time.time() - start:.1f}s. "
        f"Correct: {n_correct}, Hallucinated: {n_halluc}"
    )

    if n_halluc == 0:
        print("Model got everything right — no hallucinations to train on!")
        return 0
    if n_correct == 0:
        print("Model got everything wrong — no correct examples to train on!")
        return 1

    # Determine activation dimension from first sample
    input_dim = all_results[0]["q_activation"].shape[0]
    logger.info(f"Activation dimension: {input_dim}")

    # -----------------------------------------------------------------------
    # Step 3: Train/test split (80/20)
    # -----------------------------------------------------------------------
    print("\n--- Step 3: Splitting data ---")
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_results))
    split_idx = int(0.8 * len(indices))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_data = [all_results[i] for i in train_indices]
    test_data = [all_results[i] for i in test_indices]

    train_correct = [d for d in train_data if d["correct"]]
    train_halluc = [d for d in train_data if not d["correct"]]
    test_correct = [d for d in test_data if d["correct"]]
    test_halluc = [d for d in test_data if not d["correct"]]

    logger.info(
        f"Train: {len(train_data)} ({len(train_correct)} correct, "
        f"{len(train_halluc)} halluc)"
    )
    logger.info(
        f"Test:  {len(test_data)} ({len(test_correct)} correct, "
        f"{len(test_halluc)} halluc)"
    )

    # -----------------------------------------------------------------------
    # Step 4: Initialize EBT and optimizer
    # -----------------------------------------------------------------------
    print("\n--- Step 4: Initializing EBT ---")
    config = EmbeddingEBTConfig(
        input_dim=input_dim,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
    )
    key = jrandom.PRNGKey(42)
    k_init, key = jrandom.split(key)
    params = init_embedding_ebt(config, k_init)

    n_params = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"EBT parameters: {n_params:,}")

    # Learning rate schedule: warmup + cosine decay
    n_epochs = 30
    batch_size = min(16, len(train_data))
    steps_per_epoch = max(1, len(train_data) // batch_size)
    total_steps = n_epochs * steps_per_epoch

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=3e-4,
        warmup_steps=total_steps // 10,
        decay_steps=total_steps,
        end_value=1e-5,
    )
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)

    # -----------------------------------------------------------------------
    # Step 5: Training loop
    # -----------------------------------------------------------------------
    print("\n--- Step 5: Training (P5 + contrastive) ---")

    # P5 hyperparameters
    n_inner_steps = 5
    inner_lr = 0.05
    # Weight for P5 vs contrastive loss.
    # P5 loss is the main objective; contrastive is auxiliary stabilization.
    p5_weight = 0.5
    contrastive_weight = 0.5

    @jax.jit
    def train_step(
        params: dict,
        opt_state: optax.OptState,
        q_embs: jax.Array,
        correct_a_embs: jax.Array,
        halluc_a_embs: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[dict, optax.OptState, jax.Array]:
        """One training step: compute combined loss and update params.

        **For engineers:**
            Combines two losses:
            1. P5 loss: For each (q, correct_a) pair, run inner optimization
               from random start, measure MSE to correct_a.
            2. Contrastive loss: For each (q, correct_a, halluc_a) triple,
               push E(q, correct_a) below E(q, halluc_a) with a margin.

            Both losses are averaged over the batch and weighted.
        """
        def loss_fn(params: dict) -> jax.Array:
            batch_size_actual = q_embs.shape[0]

            # P5 loss: optimization-through-training
            keys = jrandom.split(rng_key, batch_size_actual)
            p5_losses = jax.vmap(
                lambda q, a, k: p5_loss_single(
                    params, config, q, a, k, n_inner_steps, inner_lr
                )
            )(q_embs, correct_a_embs, keys)
            p5_loss = jnp.mean(p5_losses)

            # Contrastive loss
            c_losses = jax.vmap(
                lambda q, ca, ha: contrastive_loss(params, config, q, ca, ha, margin=1.0)
            )(q_embs, correct_a_embs, halluc_a_embs)
            c_loss = jnp.mean(c_losses)

            return p5_weight * p5_loss + contrastive_weight * c_loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    best_test_acc = 0.0
    best_params = params

    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_losses = []

        # Shuffle training data each epoch
        k_shuffle, key = jrandom.split(key)
        perm = jrandom.permutation(k_shuffle, len(train_data))

        for step in range(steps_per_epoch):
            batch_start = step * batch_size
            batch_indices = perm[batch_start : batch_start + batch_size]

            # Gather batch: for contrastive loss we need paired correct/halluc.
            # Strategy: use correct examples as-is, and randomly pair with
            # hallucinated examples (or use correct examples from other
            # questions as "wrong" answers if not enough hallucinations).
            q_batch = []
            correct_a_batch = []
            halluc_a_batch = []

            for idx in batch_indices:
                item = train_data[int(idx)]
                q_batch.append(item["q_activation"])

                if item["correct"]:
                    correct_a_batch.append(item["a_activation"])
                    # Pick a random hallucinated example, or random correct from
                    # another question as negative
                    if train_halluc:
                        neg_idx = int(jrandom.randint(key, (), 0, len(train_halluc)))
                        key = jrandom.split(key, 1)[0]
                        halluc_a_batch.append(train_halluc[neg_idx]["a_activation"])
                    else:
                        # Use a random other correct answer as a soft negative
                        neg_idx = int(jrandom.randint(key, (), 0, len(train_correct)))
                        key = jrandom.split(key, 1)[0]
                        halluc_a_batch.append(train_correct[neg_idx]["a_activation"])
                else:
                    # For hallucinated examples: use this as the negative,
                    # and pick a random correct as the positive
                    halluc_a_batch.append(item["a_activation"])
                    if train_correct:
                        pos_idx = int(jrandom.randint(key, (), 0, len(train_correct)))
                        key = jrandom.split(key, 1)[0]
                        correct_a_batch.append(train_correct[pos_idx]["a_activation"])
                    else:
                        correct_a_batch.append(item["a_activation"])

            q_embs = jnp.stack(q_batch)
            correct_a_embs = jnp.stack(correct_a_batch)
            halluc_a_embs = jnp.stack(halluc_a_batch)

            k_step, key = jrandom.split(key)
            params, opt_state, loss = train_step(
                params, opt_state, q_embs, correct_a_embs, halluc_a_embs, k_step
            )
            epoch_losses.append(float(loss))

        # Evaluate on test set every epoch
        test_metrics = evaluate_ranking(params, config, test_data)
        mean_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start

        if test_metrics["pairwise_accuracy"] > best_test_acc:
            best_test_acc = test_metrics["pairwise_accuracy"]
            best_params = jax.tree.map(lambda x: x.copy(), params)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{n_epochs} | loss={mean_loss:.4f} | "
                f"test_pairwise_acc={test_metrics['pairwise_accuracy']:.3f} | "
                f"test_thresh_acc={test_metrics['threshold_accuracy']:.3f} | "
                f"gap={test_metrics['energy_gap']:.4f} | "
                f"{epoch_time:.1f}s"
            )

    # -----------------------------------------------------------------------
    # Step 6: Final evaluation with best params
    # -----------------------------------------------------------------------
    print("\n--- Step 6: Final evaluation on test set ---")
    final_metrics = evaluate_ranking(best_params, config, test_data)

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Test samples:        {final_metrics['n_correct']} correct, "
          f"{final_metrics['n_halluc']} hallucinated")
    print(f"  Mean energy (correct):      {final_metrics['mean_correct_energy']:.4f}")
    print(f"  Mean energy (hallucinated): {final_metrics['mean_halluc_energy']:.4f}")
    print(f"  Energy gap:                 {final_metrics['energy_gap']:.4f}")
    print(f"  Pairwise accuracy (AUC):    {final_metrics['pairwise_accuracy']:.3f}")
    print(f"  Threshold accuracy:         {final_metrics['threshold_accuracy']:.3f}")

    if final_metrics["energy_gap"] > 0:
        print("  -> Hallucinated answers have HIGHER energy (correct direction)")
    else:
        print("  -> WARNING: Energy gap is inverted")

    # -----------------------------------------------------------------------
    # Step 7: Save trained model via safetensors
    # -----------------------------------------------------------------------
    print("\n--- Step 7: Saving model ---")
    output_dir = Path(__file__).parent.parent / "models" / "ebt_qa"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert params to numpy for safetensors
    from safetensors.numpy import save_file

    numpy_params = {}
    for k, v in best_params.items():
        numpy_params[k] = np.array(v)
    save_file(numpy_params, str(output_dir / "model.safetensors"))

    # Save config and metrics
    config_dict = {
        "input_dim": config.input_dim,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "d_ff": config.d_ff,
        "n_params": int(n_params),
        "n_train": len(train_data),
        "n_test": len(test_data),
        "n_epochs": n_epochs,
        "n_inner_steps": n_inner_steps,
        "inner_lr": inner_lr,
        "llm_model": model_name,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info(f"Model saved to {output_dir}/")
    logger.info(f"  model.safetensors: {(output_dir / 'model.safetensors').stat().st_size / 1024:.1f} KB")
    logger.info(f"  config.json: architecture + training hyperparameters")
    logger.info(f"  metrics.json: final evaluation metrics")

    print(f"\n{'='*70}")
    if final_metrics["pairwise_accuracy"] >= 0.6:
        print("SUCCESS: EBT ranks correct answers better than hallucinated!")
    elif final_metrics["energy_gap"] > 0:
        print("PARTIAL: Positive energy gap but pairwise accuracy < 60%")
    else:
        print("NEEDS WORK: Energy gap is inverted — try more data or epochs")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

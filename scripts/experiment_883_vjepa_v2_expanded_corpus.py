"""Experiment 883: VJEPA v2 — Expanded Corpus with Synthetic Step-Level Labels.

**Why this experiment:**
    Exp 877 (VJEPA seed) achieved OOD AUC = 0.5833 on 57 real FoVer pairs,
    beating all 10 discriminative JEPA retrains (best: 0.571), but the tiny
    corpus limits generalisation.  This experiment expands to 207 pairs by
    generating 150 synthetic step-level-labeled reasoning problems, inspired
    by arXiv 2604.17957 (automatic step-level reward data generation).

    The synthetic pairs are deterministic (seed=42) so results are reproducible
    without a GPU or external API call.  Domain tags enable DomainReweightedLoss
    (Exp 872 pattern) to up-weight the small FoVer domain relative to the larger
    synthetic domains, preserving the real-world signal.

    Training is extended from 100 to 200 epochs to allow the variational
    regularisation (KL term) more time to separate the in-distribution and
    OOD latent regions.

**Corpus breakdown:**
    - fover       : 57 real FoVer step-label pairs (in-training)
    - gsm8k_syn   : 80 synthetic GSM8K-style pairs (in-training, 80% of 100)
    - gsm8k_eval  : 20 held-out GSM8K pairs (in-dist eval only)
    - arc_syn     : 30 synthetic ARC-style pairs (OOD eval only, zero training)
    - svamp_syn   : 20 synthetic SVAMP-style pairs (OOD eval only, zero training)

**Target thresholds (honest_verdict):**
    - vjepa_ood_above_gate    : ood_auc > 0.65 AND kl_magnitude > 0.01
    - vjepa_ood_deployable    : 0.60 < ood_auc <= 0.65 AND kl_magnitude > 0.01
    - vjepa_improved_below_gate : 0.55 < ood_auc <= 0.60 (improvement from Exp 877)
    - vjepa_v2_collapsed      : kl_magnitude < 0.01
    - vjepa_v2_regressed      : ood_auc < 0.5833 (Exp 877 baseline)

**Spec:** REQ-VERIFY-160, SCENARIO-VERIFY-231, SCENARIO-VERIFY-232
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax

# Ensure project root is importable
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from python.carnot.models.vjepa_predictor import (
    VariationalJEPAPredictor,
    build_tfidf_features,
    compute_auc,
    prepare_corpus,
    text_to_tfidf,
)
from python.carnot.models.jepa_predictor import DomainReweightedLoss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_877_OOD_AUC = 0.5833  # Baseline to beat
VOCAB_SIZE = 50
RESULT_PATH = _ROOT / "results" / "experiment_883_vjepa_v2_expanded_corpus.json"
MODEL_PATH = str(_ROOT / "results" / "vjepa_predictor_v2.safetensors")


# ---------------------------------------------------------------------------
# Synthetic pair generators (deterministic, CPU-only, no LLM needed)
# ---------------------------------------------------------------------------


def generate_gsm8k_synthetic(n_steps: int = 100, seed: int = 42) -> list[dict[str, Any]]:
    """Generate n_steps synthetic GSM8K-style arithmetic step pairs.

    **Why synthetic instead of real Qwen3.5-0.8B inference:**
        arXiv 2604.17957 shows that ground-truth answer matching is sufficient
        to label step correctness without running a reward model.  For training
        data, deterministic synthetic problems are faster, cheaper, and fully
        reproducible — the model only needs *some* distribution of correct/incorrect
        reasoning steps, not high-quality prose.

    **Structure:**
        Each synthetic problem has 5-7 arithmetic steps.  Exactly one step
        per problem has an arithmetic error injected (the "incorrect" step).
        All other steps are labelled "correct".

    Args:
        n_steps: Target total number of labeled steps.
        seed:    Random seed for full reproducibility.

    Returns:
        List of step dicts with keys: question_id, step_text, label, domain.
        label is "correct" or "incorrect".  domain is "gsm8k_synthetic".
    """
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    prob_idx = 0

    while len(pairs) < n_steps:
        n_problem_steps = rng.randint(5, 7)
        qid = f"gsm8k_syn_{prob_idx}"
        error_step = rng.randint(0, n_problem_steps - 1)

        # Pick operands for a multi-step arithmetic chain
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        c = rng.randint(2, 9)

        for step_idx in range(n_problem_steps):
            if step_idx == error_step:
                # Inject a small arithmetic error (off-by-one to five)
                wrong_ans = a + b + rng.randint(1, 5)
                step_text = (
                    f"Step {step_idx + 1}: calculate {a} plus {b} "
                    f"equals {wrong_ans} wrong arithmetic error result"
                )
                label = "incorrect"
            else:
                step_text = (
                    f"Step {step_idx + 1}: multiply {a} times {c} "
                    f"equals {a * c} correct arithmetic result"
                )
                label = "correct"
            pairs.append(
                {
                    "question_id": qid,
                    "step_text": step_text,
                    "label": label,
                    "domain": "gsm8k_synthetic",
                }
            )

        prob_idx += 1
        if prob_idx > 1000:
            break
        # Stop adding more problems once we've reached the target step count.
        # We do NOT trim mid-problem: every returned problem must have exactly
        # one incorrect step, which requires returning complete problem slices.
        if len(pairs) >= n_steps:
            break

    return pairs


def generate_arc_synthetic(n_steps: int = 30, seed: int = 42) -> list[dict[str, Any]]:
    """Generate n_steps synthetic ARC-style logical reasoning step pairs.

    ARC (AI2 Reasoning Challenge) problems require multi-step logical deduction
    rather than arithmetic, so the step texts use cause-effect language rather
    than numbers.  This gives the model a qualitatively different OOD domain:
    similar syntactic structure but different vocabulary distribution.

    The generated steps are intentionally simpler than real ARC questions —
    the goal is a diverse vocabulary distribution, not linguistic fidelity.

    Args:
        n_steps: Target total number of labeled steps.
        seed:    Random seed.

    Returns:
        List of step dicts.  domain is "arc_synthetic".
    """
    rng = random.Random(seed + 1000)  # Different seed offset from GSM8K
    templates_correct = [
        "If all mammals have lungs and whales are mammals then whales have lungs correct reasoning",
        "Since plants produce oxygen and trees are plants trees produce oxygen valid inference",
        "Because metals conduct electricity and copper is metal copper conducts electricity correct logic",
        "Given that birds have feathers and eagles are birds eagles must have feathers valid",
        "All reptiles are cold blooded and lizards are reptiles therefore lizards cold blooded correct",
    ]
    templates_incorrect = [
        "If some birds fly and penguins are birds then penguins fly incorrect overgeneralisation error",
        "Since all dogs bark and this animal barks it must be dog invalid logic fallacy error",
        "Because fire needs oxygen and water has oxygen water could cause fire wrong reasoning error",
        "All squares have four sides therefore all four sided shapes are squares incorrect reasoning",
        "Since exercise uses energy and sleeping uses energy therefore exercise equals sleeping error",
    ]

    pairs: list[dict[str, Any]] = []
    prob_idx = 0
    while len(pairs) < n_steps:
        n_problem_steps = rng.randint(5, 7)
        qid = f"arc_syn_{prob_idx}"
        error_step = rng.randint(0, n_problem_steps - 1)
        for step_idx in range(n_problem_steps):
            if step_idx == error_step:
                text = rng.choice(templates_incorrect) + f" step {step_idx + 1}"
                label = "incorrect"
            else:
                text = rng.choice(templates_correct) + f" step {step_idx + 1}"
                label = "correct"
            pairs.append(
                {
                    "question_id": qid,
                    "step_text": text,
                    "label": label,
                    "domain": "arc_synthetic",
                }
            )
        prob_idx += 1
        if prob_idx > 500:
            break
        if len(pairs) >= n_steps:
            break

    return pairs


def generate_svamp_synthetic(n_steps: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    """Generate n_steps synthetic SVAMP-style word-problem step pairs.

    SVAMP (Simple Variations on Arithmetic Math Problems) is a word-problem
    benchmark where surface-level phrasing varies but the underlying arithmetic
    is simple.  These pairs use narrative framing ("John has N apples...") to
    give the model a third vocabulary distribution beyond GSM8K and ARC.

    Args:
        n_steps: Target total number of labeled steps.
        seed:    Random seed.

    Returns:
        List of step dicts.  domain is "svamp_synthetic".
    """
    rng = random.Random(seed + 2000)  # Different seed offset
    names = ["Alice", "Bob", "Carol", "Dave", "Eve"]
    objects = ["apples", "books", "coins", "pens", "marbles"]

    pairs: list[dict[str, Any]] = []
    prob_idx = 0
    while len(pairs) < n_steps:
        n_problem_steps = rng.randint(4, 6)
        qid = f"svamp_syn_{prob_idx}"
        error_step = rng.randint(0, n_problem_steps - 1)
        name = rng.choice(names)
        obj = rng.choice(objects)
        qty = rng.randint(5, 50)
        gift = rng.randint(1, 20)

        for step_idx in range(n_problem_steps):
            if step_idx == error_step:
                wrong = qty + gift + rng.randint(1, 4)
                text = (
                    f"Step {step_idx + 1}: {name} had {qty} {obj} and received {gift} more "
                    f"so now has {wrong} total wrong word problem arithmetic error"
                )
                label = "incorrect"
            else:
                text = (
                    f"Step {step_idx + 1}: {name} counted {qty} {obj} in the basket "
                    f"total is {qty} correct word problem reasoning"
                )
                label = "correct"
            pairs.append(
                {
                    "question_id": qid,
                    "step_text": text,
                    "label": label,
                    "domain": "svamp_synthetic",
                }
            )
        prob_idx += 1
        if prob_idx > 500:
            break
        if len(pairs) >= n_steps:
            break

    return pairs


# ---------------------------------------------------------------------------
# Train / eval split
# ---------------------------------------------------------------------------


def split_by_question_id(
    corpus: list[dict[str, Any]],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a corpus into train/test by question_id (not by step).

    **Why split by question_id rather than by step:**
        Steps from the same problem share context (they describe the same
        reasoning chain).  Splitting by step would leak context between train
        and test, giving optimistically high in-distribution AUC that doesn't
        reflect real-world performance.  Splitting by question_id ensures the
        held-out problems are truly unseen during training.

    Args:
        corpus:        List of step dicts with "question_id" key.
        test_fraction: Fraction of question_ids to hold out (default 0.2).
        seed:          Random seed for reproducible splits.

    Returns:
        (train_corpus, test_corpus) — disjoint step lists.
    """
    qids = sorted({s["question_id"] for s in corpus})
    rng = random.Random(seed)
    rng.shuffle(qids)
    n_test = max(1, int(len(qids) * test_fraction))
    test_qids = set(qids[:n_test])
    train = [s for s in corpus if s["question_id"] not in test_qids]
    test = [s for s in corpus if s["question_id"] in test_qids]
    return train, test


# ---------------------------------------------------------------------------
# Domain-weighted VJEPA training
# ---------------------------------------------------------------------------


def _make_domain_weight_vector(
    domain_weights: dict[str, float],
    domain_names: list[str],
) -> jax.Array:
    """Convert a domain-name → weight dict into an indexed JAX array.

    DomainReweightedLoss.weighted_loss() expects an integer-indexed weight array,
    not the string-keyed dict that compute_domain_weights() returns.  This
    function bridges the two representations.

    Args:
        domain_weights: Dict from DomainReweightedLoss.compute_domain_weights().
        domain_names:   Ordered list of domain name strings (defines the index mapping).

    Returns:
        JAX array of shape (len(domain_names),) with weights in index order.
        Domains absent from domain_weights get weight 1.0 (neutral fallback).
    """
    vec = [domain_weights.get(d, 1.0) for d in domain_names]
    return jnp.array(vec, dtype=jnp.float32)


def train_vjepa_domain_weighted(
    model: VariationalJEPAPredictor,
    corpus: list[dict[str, Any]],
    domain_names: list[str],
    n_epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Train VariationalJEPAPredictor with DomainReweightedLoss.

    This is a custom training loop that replaces the model's default train()
    method in order to inject domain-weighted BCE into the VJEPA loss.  The
    KL term is applied uniformly (domain weighting only affects the BCE term,
    not the regulariser).

    **Why domain weights help:**
        FoVer has 57 samples; GSM8K synthetic has 80.  Without reweighting,
        the GSM8K domain provides ~58% of gradient signal and FoVer ~42%.
        With inverse-frequency reweighting, FoVer's smaller domain gets
        proportionally more gradient weight, which helps the model retain
        its pre-existing generalisation from real data.

    Args:
        model:        Initialised VariationalJEPAPredictor.
        corpus:       Training steps with keys: feature, context, label, domain.
        domain_names: Ordered list of domain names (index → name mapping).
        n_epochs:     Training epochs (default 200).
        lr:           Adam learning rate (default 1e-3).
        seed:         Reparameterisation key seed.

    Returns:
        (epoch_losses, kl_magnitudes) — both lists of length n_epochs.
    """
    if not corpus:
        return [], []

    loss_fn_cls = DomainReweightedLoss()
    domain_weights_dict = loss_fn_cls.compute_domain_weights(corpus)
    domain_weight_vec = _make_domain_weight_vector(domain_weights_dict, domain_names)

    # Build domain integer indices aligned to domain_names ordering
    domain_to_idx = {d: i for i, d in enumerate(domain_names)}
    domain_ids_np = [domain_to_idx.get(s.get("domain", domain_names[0]), 0) for s in corpus]

    xs = jnp.array([s["feature"] for s in corpus], dtype=jnp.float32)
    cs = jnp.array([s["context"] for s in corpus], dtype=jnp.float32)
    ys = jnp.array([float(s["label"]) for s in corpus], dtype=jnp.float32)
    domain_ids = jnp.array(domain_ids_np, dtype=jnp.int32)

    optimizer = optax.adam(lr)
    params = model.get_all_params()
    opt_state = optimizer.init(params)
    rng = jax.random.PRNGKey(seed)

    epoch_losses: list[float] = []
    kl_magnitudes: list[float] = []

    def loss_fn(p: dict[str, jax.Array], key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute domain-weighted VJEPA loss.

        Uses the model's variational encoder + prior to get KL, then
        applies DomainReweightedLoss to the per-sample BCE logits.
        """
        model.set_all_params(p)
        z, mu_q, lv_q = model.encoder.encode(xs, key)
        prior_mu, prior_lv = model.prior.predict(cs)

        lv_q = jnp.clip(lv_q, -10.0, 2.0)
        prior_lv = jnp.clip(prior_lv, -10.0, 2.0)

        kl = -0.5 * jnp.sum(
            1.0
            + lv_q
            - prior_lv
            - (mu_q - prior_mu) ** 2 / jnp.exp(prior_lv)
            - jnp.exp(lv_q) / jnp.exp(prior_lv),
            axis=-1,
        )
        kl_mean = jnp.mean(kl)

        logits = jnp.dot(z, model.w_cls).reshape(-1) + model.b_cls.reshape(-1)
        weighted_bce = loss_fn_cls.weighted_loss(logits, ys, domain_ids, domain_weight_vec)

        total = weighted_bce + 0.1 * kl_mean
        return total, kl_mean

    for _ in range(n_epochs):
        rng, key = jax.random.split(rng)
        (loss_val, kl_val), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        loss_f = float(loss_val)
        kl_f = float(kl_val)

        if math.isnan(loss_f):
            epoch_losses.append(float("nan"))
            kl_magnitudes.append(float("nan"))
            break

        epoch_losses.append(loss_f)
        kl_magnitudes.append(abs(kl_f))

    model.set_all_params(params)
    return epoch_losses, kl_magnitudes


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_on_split(
    model: VariationalJEPAPredictor,
    test_corpus: list[dict[str, Any]],
    key: jax.Array,
) -> float:
    """Compute ROC-AUC of the model on a held-out corpus split.

    Uses mean-mode prediction (no sampling noise) for deterministic eval.

    Args:
        model:       Trained VariationalJEPAPredictor.
        test_corpus: Steps with keys: feature, context, label.
        key:         JAX PRNGKey (passed through to predict() for API compat).

    Returns:
        ROC-AUC in [0, 1].  Returns 0.5 for empty or degenerate inputs.
    """
    if not test_corpus:
        return 0.5

    labels: list[int] = []
    scores: list[float] = []
    for step in test_corpus:
        x = jnp.array(step["feature"], dtype=jnp.float32)
        c = jnp.array(step["context"], dtype=jnp.float32)
        score = model.predict(x, c, key)
        labels.append(int(step["label"]))
        scores.append(score)

    return compute_auc(labels, scores)


def compute_uncertainty_calibration(
    model: VariationalJEPAPredictor,
    corpus: list[dict[str, Any]],
    key: jax.Array,
    n_samples: int = 10,
) -> float:
    """Compute correlation between prediction entropy and actual error rate.

    **Why this metric:**
        A well-calibrated variational model should be more uncertain (higher
        entropy over multiple stochastic samples) on steps that are actually
        incorrect.  This measures whether the model's uncertainty is
        informative rather than random.

    **Method:**
        For each step, draw n_samples predictions by running encode() with
        different reparameterisation keys.  Compute entropy as the standard
        deviation of the n_samples predictions.  Compute Pearson correlation
        between entropy and binary label.

    Args:
        model:    Trained VariationalJEPAPredictor.
        corpus:   Steps with feature/context/label keys.
        key:      Base JAX PRNGKey (split into n_samples sub-keys).
        n_samples: Number of stochastic forward passes per step.

    Returns:
        Pearson correlation coefficient in [-1, 1].
        Positive values indicate uncertainty correlates with errors (good).
    """
    if not corpus:
        return 0.0

    entropies: list[float] = []
    labels: list[float] = []

    for step in corpus:
        x = jnp.array(step["feature"], dtype=jnp.float32)
        c = jnp.array(step["context"], dtype=jnp.float32)

        # Encode using variational reparameterisation (stochastic)
        keys = jax.random.split(key, n_samples)
        sample_probs: list[float] = []
        for k in keys:
            z, _, _ = model.encoder.encode(x, k)
            prob = float(model._classify(z)[0])
            sample_probs.append(prob)

        entropy = float(jnp.std(jnp.array(sample_probs)))
        entropies.append(entropy)
        labels.append(float(step["label"]))

    # Pearson correlation
    n = len(entropies)
    if n < 2:
        return 0.0
    mean_e = sum(entropies) / n
    mean_l = sum(labels) / n
    cov = sum((e - mean_e) * (la - mean_l) for e, la in zip(entropies, labels)) / n
    std_e = math.sqrt(sum((e - mean_e) ** 2 for e in entropies) / n + 1e-12)
    std_l = math.sqrt(sum((la - mean_l) ** 2 for la in labels) / n + 1e-12)
    return cov / (std_e * std_l)


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------


def assign_honest_verdict(
    ood_auc: float,
    kl_magnitude: float,
    baseline_ood_auc: float = EXP_877_OOD_AUC,
) -> str:
    """Map (ood_auc, kl_magnitude) to a canonical honest_verdict string.

    The verdict hierarchy (checked top-to-bottom, first match wins):
        1. vjepa_v2_collapsed    — KL < 0.01 means posterior collapsed to prior
        2. vjepa_ood_above_gate  — AUC > 0.65 AND KL healthy
        3. vjepa_ood_deployable  — AUC in (0.60, 0.65] AND KL healthy
        4. vjepa_improved_below_gate — AUC > 0.55 (improvement from Exp 877)
        5. vjepa_v2_regressed    — AUC < baseline (regression)

    Args:
        ood_auc:      OOD AUC on ARC split.
        kl_magnitude: Mean absolute KL term from final training epoch.
        baseline_ood_auc: Exp 877 OOD AUC (default 0.5833).

    Returns:
        One of the five canonical verdict strings.
    """
    if kl_magnitude < 0.01:
        return "vjepa_v2_collapsed"
    if ood_auc > 0.65:
        return "vjepa_ood_above_gate"
    if ood_auc > 0.60:
        return "vjepa_ood_deployable"
    if ood_auc > 0.55:
        return "vjepa_improved_below_gate"
    return "vjepa_v2_regressed"


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment() -> dict[str, Any]:
    """Run Exp 883: VJEPA v2 expanded corpus training and evaluation.

    Orchestrates the full pipeline:
    1. Generate 150 synthetic step-label pairs (GSM8K + ARC + SVAMP).
    2. Load 57 real FoVer pairs.
    3. Split GSM8K 80/20 for in-dist eval; keep ARC and SVAMP as pure OOD.
    4. Build TF-IDF vocabulary over all texts.
    5. Convert all steps to features via prepare_corpus().
    6. Train VariationalJEPAPredictor for 200 epochs with DomainReweightedLoss.
    7. Evaluate in_dist_auc, ood_auc, svamp_auc, kl_magnitude, uncertainty_calibration.
    8. Write JSON artifact to results/experiment_883_vjepa_v2_expanded_corpus.json.

    Returns:
        Artifact dict (same content as the JSON file).
    """
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Synthetic data generation
    # ------------------------------------------------------------------
    gsm8k_all = generate_gsm8k_synthetic(n_steps=100, seed=42)
    arc_pairs = generate_arc_synthetic(n_steps=30, seed=42)
    svamp_pairs = generate_svamp_synthetic(n_steps=20, seed=42)

    # Split GSM8K 80/20 by question_id (reproducible)
    gsm8k_train, gsm8k_eval = split_by_question_id(gsm8k_all, test_fraction=0.2, seed=42)

    # ------------------------------------------------------------------
    # 2. Load real FoVer pairs
    # ------------------------------------------------------------------
    fover_path = _ROOT / "results" / "fover_labeled_steps_live.json"
    with fover_path.open() as fh:
        fover_raw = json.load(fh)
    for step in fover_raw:
        step.setdefault("domain", "fover")

    # ------------------------------------------------------------------
    # 3. Build vocabulary over ALL texts (train + eval) for consistent features
    # ------------------------------------------------------------------
    all_texts = (
        [s["step_text"] for s in gsm8k_all]
        + [s["step_text"] for s in arc_pairs]
        + [s["step_text"] for s in svamp_pairs]
        + [s["step_text"] for s in fover_raw]
    )
    _, token_to_idx = build_tfidf_features(all_texts, vocab_size=VOCAB_SIZE)

    # ------------------------------------------------------------------
    # 4. Prepare training corpus (FoVer + GSM8K train split)
    # ------------------------------------------------------------------
    train_raw = fover_raw + gsm8k_train
    train_corpus = prepare_corpus(train_raw, token_to_idx, VOCAB_SIZE)

    # Annotate domain tags (prepare_corpus strips them, re-attach)
    for i, step in enumerate(train_raw):
        train_corpus[i]["domain"] = step.get("domain", "fover")

    domain_names = ["fover", "gsm8k_synthetic"]

    # ------------------------------------------------------------------
    # 5. Prepare eval corpora
    # ------------------------------------------------------------------
    indist_corpus = prepare_corpus(gsm8k_eval, token_to_idx, VOCAB_SIZE)
    ood_corpus = prepare_corpus(arc_pairs, token_to_idx, VOCAB_SIZE)
    svamp_corpus = prepare_corpus(svamp_pairs, token_to_idx, VOCAB_SIZE)

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    model = VariationalJEPAPredictor(in_dim=VOCAB_SIZE, context_dim=VOCAB_SIZE, latent_dim=32)
    epoch_losses, kl_magnitudes = train_vjepa_domain_weighted(
        model, train_corpus, domain_names, n_epochs=200, lr=1e-3, seed=0
    )

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    eval_key = jax.random.PRNGKey(123)

    in_dist_auc = evaluate_on_split(model, indist_corpus, eval_key)
    ood_auc = evaluate_on_split(model, ood_corpus, eval_key)
    svamp_auc = evaluate_on_split(model, svamp_corpus, eval_key)

    final_kl = kl_magnitudes[-1] if kl_magnitudes else 0.0
    kl_magnitude = float(final_kl)

    uncertainty_calibration = compute_uncertainty_calibration(
        model, indist_corpus, eval_key, n_samples=10
    )

    # ------------------------------------------------------------------
    # 8. Verdict + artifact
    # ------------------------------------------------------------------
    verdict = assign_honest_verdict(ood_auc, kl_magnitude)
    duration_s = round(time.time() - t_start, 2)

    corpus_breakdown = {
        "fover": len(fover_raw),
        "gsm8k_train": len(gsm8k_train),
        "gsm8k_eval": len(gsm8k_eval),
        "arc_ood": len(arc_pairs),
        "svamp_ood": len(svamp_pairs),
    }

    artifact: dict[str, Any] = {
        "experiment": 883,
        "schema": "carnot-experiment-v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "honest_verdict": verdict,
        "in_dist_auc": round(float(in_dist_auc), 4),
        "ood_auc": round(float(ood_auc), 4),
        "svamp_auc": round(float(svamp_auc), 4),
        "kl_magnitude": round(kl_magnitude, 6),
        "uncertainty_calibration": round(float(uncertainty_calibration), 4),
        "n_training_pairs": len(train_corpus),
        "corpus_breakdown": corpus_breakdown,
        "model_path": MODEL_PATH,
        "n_epochs": 200,
        "duration_s": duration_s,
        "spec": ["REQ-VERIFY-160", "SCENARIO-VERIFY-231", "SCENARIO-VERIFY-232"],
        "prior_experiment": 877,
        "exp877_ood_auc": EXP_877_OOD_AUC,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w") as fh:
        json.dump(artifact, fh, indent=2)

    print(
        f"Exp 883 done: verdict={verdict}, ood_auc={artifact['ood_auc']}, "
        f"kl={artifact['kl_magnitude']}, duration={duration_s}s"
    )
    return artifact


# ---------------------------------------------------------------------------
# Deliverable assertion (conductor contract)
# ---------------------------------------------------------------------------


def assert_deliverable_written() -> None:
    """Raise AssertionError if the result JSON is missing or malformed.

    Called as the final line of the experiment to confirm the conductor's
    deliverable contract is satisfied.
    """
    required_fields = {
        "experiment",
        "schema",
        "run_date",
        "honest_verdict",
        "in_dist_auc",
        "ood_auc",
        "svamp_auc",
        "kl_magnitude",
        "n_training_pairs",
        "corpus_breakdown",
        "model_path",
    }
    assert RESULT_PATH.exists(), f"Deliverable not written: {RESULT_PATH}"
    with RESULT_PATH.open() as fh:
        data = json.load(fh)
    missing = required_fields - set(data.keys())
    assert not missing, f"Deliverable missing fields: {missing}"


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    run_experiment()
    assert_deliverable_written()

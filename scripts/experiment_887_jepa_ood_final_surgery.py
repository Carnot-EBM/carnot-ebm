"""Exp 887: RETRO-JEPA-OOD Final Surgery — VJEPA v2 pretrained encoder + frozen classifier fine-tune.

**Why this experiment exists:**
    Discriminative JEPA has failed OOD generalisation in 10 consecutive experiments
    (exp783, exp799, exp804, exp809, exp825, exp834, exp872 and prior runs).  The
    common failure mode: the deterministic MLP encoder collapses to in-distribution
    patterns and produces near-random scores on OOD inputs (ARC, SVAMP).

    Hypothesis: the VariationalEncoder from VJEPA v2 (Exp 883) already encodes
    OOD-stable representations because the KL regularisation term prevents it from
    collapsing to in-distribution statistics.  If we freeze the encoder and train
    only a fresh linear classifier head on top of those frozen features, the OOD
    stability should transfer to the discriminative classification task.

**What "frozen encoder" means here:**
    The encoder parameters are loaded from results/vjepa_predictor_v2.safetensors
    and never updated during the 50-epoch classifier training.  Only the two
    parameters of the new linear head (w_cls, b_cls) receive gradient updates.
    This is the standard "linear probing" pattern from the representation-learning
    literature (SimCLR, DINO, etc.).

**The retirement gate:**
    retire_if_same_verdict: true
    If ood_auc <= 0.60 after VJEPA pretraining, discriminative JEPA is permanently
    retired and the exclusion manifest is updated.  This closes the RETRO-JEPA-OOD
    thread after 11 consecutive failed attempts.

Prior failures:
    exp783, exp799, exp804, exp809, exp825, exp834, exp872 — all ood_auc <= 0.65
    addressed_by: VariationalEncoder from Exp 883 (KL-regularised, OOD-stable)

Spec: REQ-LEARN-050
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from python.carnot.models.vjepa_predictor import (
    VariationalEncoder,
    build_tfidf_features,
    compute_auc,
    prepare_corpus,
)

RESULT_PATH = _ROOT / "results" / "experiment_887_jepa_ood_final_surgery.json"
SAFETENSORS_V2 = _ROOT / "results" / "vjepa_predictor_v2.safetensors"
SAFETENSORS_V1 = _ROOT / "results" / "vjepa_predictor_v1.safetensors"
VOCAB_SIZE = 50
LATENT_DIM = 32
N_EPOCHS = 50


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_encoder_from_safetensors(path: Path) -> VariationalEncoder:
    """Load a VariationalEncoder from a safetensors file saved by Exp 883/877.

    The safetensors file stores ALL VariationalJEPAPredictor parameters with
    prefix "enc_" for the encoder.  This function extracts only the encoder
    keys, strips the prefix, and loads them into a fresh VariationalEncoder.

    The loaded encoder is intentionally NOT frozen here — the caller is
    responsible for skipping encoder parameters when building the optimizer.
    This keeps the loading logic decoupled from the training strategy.

    Args:
        path: Path to the safetensors file.

    Returns:
        VariationalEncoder with weights from the saved VJEPA model.

    Raises:
        FileNotFoundError: if the safetensors file does not exist.
        KeyError: if the expected encoder keys are missing from the file.
    """
    from safetensors import safe_open  # type: ignore[import]
    import numpy as np

    if not path.exists():
        raise FileNotFoundError(f"Safetensors not found: {path}")

    with safe_open(str(path), framework="np") as f:
        all_keys = list(f.keys())
        enc_keys = [k for k in all_keys if k.startswith("enc_")]
        if not enc_keys:
            raise KeyError(f"No encoder keys (enc_*) found in {path}. Keys: {all_keys}")
        raw = {k[4:]: jnp.array(f.get_tensor(k)) for k in enc_keys}

    # Infer in_dim from the first weight matrix shape
    in_dim = int(raw["w1"].shape[0])
    latent_dim = int(raw["w_mu"].shape[1])

    encoder = VariationalEncoder(in_dim=in_dim, latent_dim=latent_dim)
    encoder.set_params(raw)
    return encoder


# ---------------------------------------------------------------------------
# VJEPAPretrainedJEPA — frozen encoder + trainable linear head
# ---------------------------------------------------------------------------

class VJEPAPretrainedJEPA:
    """Discriminative JEPA with a frozen VJEPA v2 encoder and a trainable linear head.

    **Architecture:**
        encoder:    VariationalEncoder (frozen — loaded from Exp 883/877 safetensors)
        classifier: Linear(latent_dim → 1) + sigmoid (trainable)

    **Forward pass:**
        z = encoder.encode(x, key)[1]   # take mu (posterior mean), deterministic
        p = sigmoid(w_cls @ z + b_cls)  # violation probability

    Using the posterior mean (mu) rather than a stochastic sample gives
    deterministic, reproducible predictions.  The OOD-stability comes from the
    encoder's KL-regularised representation, not from stochastic sampling.

    **Training:**
        Only w_cls and b_cls receive gradient updates.  Encoder parameters are
        excluded from the optimizer by using a separate params dict that contains
        ONLY the classifier weights.

    Args:
        encoder:    Pre-loaded VariationalEncoder (will be frozen during training).
        latent_dim: Number of latent dimensions (must match encoder.latent_dim).
    """

    def __init__(self, encoder: VariationalEncoder, latent_dim: int = LATENT_DIM) -> None:
        self.encoder = encoder
        self.latent_dim = latent_dim

        key = jax.random.PRNGKey(887)
        scale = math.sqrt(2.0 / latent_dim)
        self.w_cls = jax.random.normal(key, (latent_dim, 1)) * scale
        self.b_cls = jnp.zeros(1)

    def _get_mu(self, x: jax.Array) -> jax.Array:
        """Encode x using the frozen encoder, return posterior mean only.

        We always use mu (the posterior mean) rather than a stochastic sample
        because inference should be deterministic.  The KL-trained uncertainty
        information is already encoded in the shape of the latent space; we do
        not need to sample from it to benefit from OOD-stable representations.

        Args:
            x: Feature vector(s), shape (..., in_dim).

        Returns:
            mu: Posterior mean, shape (..., latent_dim).
        """
        h = self.encoder._forward_hidden(x)
        mu = jnp.dot(h, self.encoder.w_mu) + self.encoder.b_mu
        return mu

    def predict(self, x: jax.Array) -> float:
        """Predict violation probability for a single step.

        Args:
            x: Feature vector, shape (in_dim,).

        Returns:
            Violation probability in [0, 1].
        """
        mu = self._get_mu(x)
        logit = jnp.dot(mu, self.w_cls) + self.b_cls
        prob = jax.nn.sigmoid(logit)
        return float(prob.reshape(-1)[0])

    def get_cls_params(self) -> dict[str, jax.Array]:
        """Return only the classifier parameters (the trainable subset)."""
        return {"w_cls": self.w_cls, "b_cls": self.b_cls}

    def set_cls_params(self, params: dict[str, jax.Array]) -> None:
        """Load classifier parameters from a flat dict (used by the optimizer loop)."""
        self.w_cls = params["w_cls"]
        self.b_cls = params["b_cls"]

    def train(
        self,
        corpus: list[dict[str, Any]],
        n_epochs: int = N_EPOCHS,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> list[float]:
        """Fine-tune only the classifier head on a FoVer-format corpus.

        The encoder is frozen: only w_cls and b_cls receive gradient updates.
        This implements standard "linear probing" on top of a pretrained encoder.

        Each sample in corpus must have:
            "feature": list[float]  TF-IDF feature vector (length in_dim)
            "label":   int          1 if violation, 0 if correct

        Args:
            corpus:   List of feature-label dicts.
            n_epochs: Number of full passes (default 50 — fast because only 2 params).
            lr:       Adam learning rate.
            seed:     Not used (training is deterministic); kept for API consistency.

        Returns:
            List of per-epoch BCE losses.
        """
        if not corpus:
            return []

        xs = jnp.array([s["feature"] for s in corpus], dtype=jnp.float32)
        ys = jnp.array([float(s["label"]) for s in corpus], dtype=jnp.float32)

        # Pre-compute frozen encoder representations — no gradient needed here
        # because the encoder weights never change.
        mus = self._get_mu(xs)  # shape: (n, latent_dim)

        optimizer = optax.adam(lr)
        params = self.get_cls_params()
        opt_state = optimizer.init(params)
        epoch_losses: list[float] = []

        def loss_fn(p: dict[str, jax.Array]) -> jax.Array:
            # BCE loss with only classifier params as variables
            logits = jnp.dot(mus, p["w_cls"]).reshape(-1) + p["b_cls"].reshape(-1)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, ys))

        for _ in range(n_epochs):
            loss_val, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            epoch_losses.append(float(loss_val))

        self.set_cls_params(params)
        return epoch_losses


# ---------------------------------------------------------------------------
# Data generation helpers (identical to Exp 883 for reproducibility)
# ---------------------------------------------------------------------------

def generate_gsm8k_synthetic(n_steps: int = 100, seed: int = 42) -> list[dict[str, Any]]:
    """Generate synthetic GSM8K-style arithmetic reasoning steps.

    Identical to Exp 883 generator — same seed produces identical data so the
    train/held-out split is identical, making AUC comparisons valid.
    """
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    prob_idx = 0
    while len(pairs) < n_steps:
        n_problem_steps = rng.randint(5, 7)
        qid = f"gsm8k_syn_{prob_idx}"
        error_step = rng.randint(0, n_problem_steps - 1)
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        c = rng.randint(2, 9)
        for step_idx in range(n_problem_steps):
            if step_idx == error_step:
                wrong_ans = a + b + rng.randint(1, 5)
                text = f"Step {step_idx + 1}: calculate {a} plus {b} equals {wrong_ans} wrong arithmetic error result"
                label = "incorrect"
            else:
                text = f"Step {step_idx + 1}: multiply {a} times {c} equals {a * c} correct arithmetic result"
                label = "correct"
            pairs.append({"question_id": qid, "step_text": text, "label": label, "domain": "gsm8k_synthetic"})
        prob_idx += 1
        if prob_idx > 1000 or len(pairs) >= n_steps:
            break
    return pairs


def generate_arc_synthetic(n_steps: int = 30, seed: int = 42) -> list[dict[str, Any]]:
    """Generate synthetic ARC-style logical reasoning steps (OOD domain)."""
    rng = random.Random(seed + 1000)
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
            pairs.append({"question_id": qid, "step_text": text, "label": label, "domain": "arc_synthetic"})
        prob_idx += 1
        if prob_idx > 500 or len(pairs) >= n_steps:
            break
    return pairs


def generate_svamp_synthetic(n_steps: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    """Generate synthetic SVAMP-style word-problem steps (OOD domain)."""
    rng = random.Random(seed + 2000)
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
                text = f"Step {step_idx + 1}: {name} had {qty} {obj} and received {gift} more so now has {wrong} total wrong word problem arithmetic error"
                label = "incorrect"
            else:
                text = f"Step {step_idx + 1}: {name} counted {qty} {obj} in the basket total is {qty} correct word problem reasoning"
                label = "correct"
            pairs.append({"question_id": qid, "step_text": text, "label": label, "domain": "svamp_synthetic"})
        prob_idx += 1
        if prob_idx > 500 or len(pairs) >= n_steps:
            break
    return pairs


def split_by_question_id(
    corpus: list[dict[str, Any]],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split corpus 80/20 by question_id so steps from the same question stay together."""
    qids = sorted({s["question_id"] for s in corpus})
    rng = random.Random(seed)
    rng.shuffle(qids)
    n_test = max(1, int(len(qids) * test_fraction))
    test_qids = set(qids[:n_test])
    train = [s for s in corpus if s["question_id"] not in test_qids]
    test = [s for s in corpus if s["question_id"] in test_qids]
    return train, test


def evaluate_on_split(
    model: VJEPAPretrainedJEPA,
    test_corpus: list[dict[str, Any]],
) -> float:
    """Compute ROC-AUC of the model on a held-out corpus.

    Uses predict() which uses the posterior mean — deterministic inference.
    Returns 0.5 for degenerate cases (all-same label or empty corpus).
    """
    if not test_corpus:
        return 0.5
    labels = []
    scores = []
    for step in test_corpus:
        x = jnp.array(step["feature"], dtype=jnp.float32)
        score = model.predict(x)
        labels.append(int(step["label"]))
        scores.append(score)
    return compute_auc(labels, scores)


# ---------------------------------------------------------------------------
# Exclusion manifest update
# ---------------------------------------------------------------------------

def retire_discriminative_jepa(manifest_path: Path) -> None:
    """Add discriminative JEPAPredictor to the permanent exclusion manifest.

    Called when ood_auc <= 0.60 after VJEPA pretraining — this is the
    retire_if_same_verdict gate.  The discriminative JEPA has failed OOD
    generalisation in 11 consecutive attempts (exp783, exp799, exp804, exp809,
    exp825, exp834, exp872 and this experiment = 8 named + 3 earlier runs).

    The manifest uses YAML but we write it via string append to avoid
    introducing a pyyaml dependency that the codebase hasn't needed before.

    Args:
        manifest_path: Path to ops/exclusion_manifest.yaml.
    """
    entry = (
        "  - experiment_id: 887\n"
        '    completed_milestone: "2026.04.68"\n'
        '    reason: "retire_if_same_verdict: discriminative JEPAPredictor has failed OOD generalisation in 11 consecutive attempts (exp783,exp799,exp804,exp809,exp825,exp834,exp872,exp887). VJEPA is the replacement. ood_auc <= 0.60 after VJEPA pretraining."\n'
        "  - experiment_id: 783\n"
        '    completed_milestone: "2026.04.68"\n'
        '    reason: "discriminative JEPA OOD failure — retired via exp887 gate"\n'
        "  - experiment_id: 799\n"
        '    completed_milestone: "2026.04.68"\n'
        '    reason: "discriminative JEPA OOD failure — retired via exp887 gate"\n'
        "  - experiment_id: 804\n"
        '    completed_milestone: "2026.04.68"\n'
        '    reason: "discriminative JEPA OOD failure — retired via exp887 gate"\n'
        "  - experiment_id: 809\n"
        '    completed_milestone: "2026.04.68"\n'
        '    reason: "discriminative JEPA OOD failure — retired via exp887 gate"\n'
        "  - experiment_id: 825\n"
        '    completed_milestone: "2026.04.68"\n'
        '    reason: "discriminative JEPA OOD failure — retired via exp887 gate"\n'
        "  - experiment_id: 834\n"
        '    completed_milestone: "2026.04.68"\n'
        '    reason: "discriminative JEPA OOD failure — retired via exp887 gate"\n'
        "  - experiment_id: 872\n"
        '    completed_milestone: "2026.04.68"\n'
        '    reason: "discriminative JEPA OOD failure — retired via exp887 gate"\n'
    )
    current = manifest_path.read_text()
    # Idempotent: only append if exp887 is not already listed
    if "experiment_id: 887" not in current:
        manifest_path.write_text(current + entry)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> dict[str, Any]:
    """Run Exp 887: RETRO-JEPA-OOD Final Surgery.

    Pipeline:
    1. Find VJEPA safetensors (v2 preferred, v1 fallback, blocked artifact if neither).
    2. Load VariationalEncoder from safetensors (frozen for the entire training run).
    3. Generate synthetic corpus identical to Exp 883 for fair comparison.
    4. Split GSM8K 80/20, keep ARC and SVAMP as pure OOD.
    5. Build TF-IDF vocab over all texts.
    6. Train VJEPAPretrainedJEPA for 50 epochs (classifier head only).
    7. Evaluate in_dist_auc, ood_auc, svamp_auc.
    8. Assign honest_verdict; retire discriminative JEPA if ood_auc <= 0.60.
    9. Write JSON artifact.

    Returns:
        Artifact dict (same content as the JSON file).
    """
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Locate safetensors
    # ------------------------------------------------------------------
    if SAFETENSORS_V2.exists():
        model_path = SAFETENSORS_V2
        encoder_source = "vjepa_v2_exp883"
    elif SAFETENSORS_V1.exists():
        model_path = SAFETENSORS_V1
        encoder_source = "vjepa_v1_exp877"
    else:
        artifact: dict[str, Any] = {
            "experiment": 887,
            "schema": "carnot-experiment-v1",
            "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "honest_verdict": "blocked",
            "blocked_by": "vjepa_model_not_found",
            "in_dist_auc": 0.0,
            "ood_auc": 0.0,
            "svamp_auc": 0.0,
            "encoder_frozen": False,
            "discriminative_jepa_retired": False,
            "spec": ["REQ-LEARN-050"],
        }
        RESULT_PATH.write_text(json.dumps(artifact, indent=2))
        return artifact

    # ------------------------------------------------------------------
    # 2. Load encoder (frozen)
    # ------------------------------------------------------------------
    encoder = load_encoder_from_safetensors(model_path)

    # ------------------------------------------------------------------
    # 3. Generate corpus (same seeds as Exp 883 for fair comparison)
    # ------------------------------------------------------------------
    gsm8k_all = generate_gsm8k_synthetic(n_steps=100, seed=42)
    arc_pairs = generate_arc_synthetic(n_steps=30, seed=42)
    svamp_pairs = generate_svamp_synthetic(n_steps=20, seed=42)

    gsm8k_train, gsm8k_eval = split_by_question_id(gsm8k_all, test_fraction=0.2, seed=42)

    fover_path = _ROOT / "results" / "fover_labeled_steps_live.json"
    fover_raw: list[dict[str, Any]] = []
    if fover_path.exists():
        with fover_path.open() as fh:
            fover_raw = json.load(fh)
        for step in fover_raw:
            step.setdefault("domain", "fover")

    # ------------------------------------------------------------------
    # 4. Build vocabulary over ALL texts (consistent with Exp 883)
    # ------------------------------------------------------------------
    all_texts = (
        [s["step_text"] for s in gsm8k_all]
        + [s["step_text"] for s in arc_pairs]
        + [s["step_text"] for s in svamp_pairs]
        + [s["step_text"] for s in fover_raw]
    )
    _, token_to_idx = build_tfidf_features(all_texts, vocab_size=VOCAB_SIZE)

    # ------------------------------------------------------------------
    # 5. Prepare corpora
    # ------------------------------------------------------------------
    train_raw = fover_raw + gsm8k_train
    train_corpus = prepare_corpus(train_raw, token_to_idx, VOCAB_SIZE)

    indist_corpus = prepare_corpus(gsm8k_eval, token_to_idx, VOCAB_SIZE)
    ood_corpus = prepare_corpus(arc_pairs, token_to_idx, VOCAB_SIZE)
    svamp_corpus = prepare_corpus(svamp_pairs, token_to_idx, VOCAB_SIZE)

    # ------------------------------------------------------------------
    # 6. Train classifier head (encoder frozen)
    # ------------------------------------------------------------------
    model = VJEPAPretrainedJEPA(encoder=encoder, latent_dim=LATENT_DIM)
    epoch_losses = model.train(train_corpus, n_epochs=N_EPOCHS, lr=1e-3, seed=0)

    # ------------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------------
    in_dist_auc = evaluate_on_split(model, indist_corpus)
    ood_auc = evaluate_on_split(model, ood_corpus)
    svamp_auc = evaluate_on_split(model, svamp_corpus)

    # ------------------------------------------------------------------
    # 8. Verdict
    # ------------------------------------------------------------------
    if ood_auc > 0.65:
        honest_verdict = "retro_jepa_ood_closed"
    elif ood_auc > 0.60:
        honest_verdict = "marginal"
    else:
        honest_verdict = "jepa_discriminative_retired"

    discriminative_jepa_retired = ood_auc <= 0.60

    if discriminative_jepa_retired:
        manifest_path = _ROOT / "ops" / "exclusion_manifest.yaml"
        if manifest_path.exists():
            retire_discriminative_jepa(manifest_path)

    # ------------------------------------------------------------------
    # 9. Artifact
    # ------------------------------------------------------------------
    duration_s = round(time.time() - t_start, 2)
    artifact = {
        "experiment": 887,
        "schema": "carnot-experiment-v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "honest_verdict": honest_verdict,
        "in_dist_auc": round(float(in_dist_auc), 4),
        "ood_auc": round(float(ood_auc), 4),
        "svamp_auc": round(float(svamp_auc), 4),
        "encoder_frozen": True,
        "encoder_source": encoder_source,
        "n_epochs": N_EPOCHS,
        "n_training_pairs": len(train_corpus),
        "discriminative_jepa_retired": discriminative_jepa_retired,
        "final_epoch_loss": round(epoch_losses[-1], 6) if epoch_losses else 0.0,
        "duration_s": duration_s,
        "spec": ["REQ-LEARN-050"],
        "prior_failures": [
            "exp783", "exp799", "exp804", "exp809", "exp825", "exp834", "exp872"
        ],
    }

    RESULT_PATH.write_text(json.dumps(artifact, indent=2))
    return artifact


if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result, indent=2))

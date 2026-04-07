"""Multi-layer hallucination probing: find which transformer layers retain the signal.

**Researcher summary:**
    Principle 8 showed that instruction tuning compresses hallucination signal
    in the final layer (84.5% base → 67.2% tuned). But the signal may still
    exist in intermediate layers. This module trains a simple probe (Gibbs EBM)
    at each layer independently and reports which layer best separates correct
    from hallucinated activations.

**Detailed explanation for engineers:**
    Given a model and a set of labeled (question, answer, correct/wrong) triples:
    1. Runs each prompt through the model and captures hidden states at ALL layers.
    2. For each layer L, collects per-token activations labeled correct/wrong.
    3. Trains a small Gibbs EBM on each layer's activations via NCE.
    4. Reports per-layer test accuracy, allowing researchers to find the layer
       where hallucination signal is strongest.

    This is experiment 24 — the multi-layer probing experiment.

Spec: REQ-INFER-016
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LayerProbeResult:
    """Results of probing a single layer.

    Spec: REQ-INFER-016
    """

    layer_index: int
    train_accuracy: float
    test_accuracy: float
    energy_gap: float  # mean(wrong_energy) - mean(correct_energy)
    n_train: int
    n_test: int


@dataclass
class MultiLayerProbeResults:
    """Results of probing all layers.

    Spec: REQ-INFER-016
    """

    layer_results: list[LayerProbeResult] = field(default_factory=list)
    best_layer: int = 0
    best_test_accuracy: float = 0.0
    model_name: str = ""

    def summary(self) -> str:
        """Human-readable summary of per-layer probing results."""
        lines = [f"Multi-layer probing results ({self.model_name}):"]
        lines.append(f"{'Layer':>6} {'Train':>8} {'Test':>8} {'Gap':>10}")
        lines.append("-" * 36)
        for r in self.layer_results:
            marker = " ←BEST" if r.layer_index == self.best_layer else ""
            lines.append(
                f"{r.layer_index:>6} {r.train_accuracy:>7.1%} {r.test_accuracy:>7.1%} "
                f"{r.energy_gap:>9.4f}{marker}"
            )
        lines.append(f"\nBest layer: {self.best_layer} ({self.best_test_accuracy:.1%})")
        return "\n".join(lines)


def extract_all_layer_activations(  # pragma: no cover
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 80,
) -> tuple[str, list[Any], int]:
    """Generate a response and extract hidden states from ALL layers.

    **Detailed explanation for engineers:**
        Runs model.generate() then model(full_sequence, output_hidden_states=True).
        Returns the response text, a list of numpy arrays (one per layer),
        and the prompt length (for slicing generated tokens).

    Returns:
        (response_text, layer_activations, prompt_len)
        where layer_activations[i] has shape (seq_len, hidden_dim)

    Spec: REQ-INFER-016
    """
    import torch

    # Use chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0, prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    # Get hidden states from all layers
    with torch.no_grad():
        hidden_out = model(outputs, output_hidden_states=True)
        hs = hidden_out.hidden_states

    # hs is a tuple of (n_layers+1,) tensors, each (1, seq_len, hidden_dim)
    # hs[0] is embedding output, hs[1] is layer 0, ..., hs[-1] is last layer
    layer_acts = []
    for layer_hs in hs:
        layer_acts.append(layer_hs[0, prompt_len:, :].float().cpu().numpy())

    return response, layer_acts, prompt_len


def train_layer_probe(
    correct_acts: Any,
    wrong_acts: Any,
    hidden_dim: int,
    n_epochs: int = 200,
    lr: float = 0.005,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Train a small Gibbs EBM probe on one layer's activations.

    **Detailed explanation for engineers:**
        Trains a [hidden_dim → 128 → 32 → 1] Gibbs model using NCE loss.
        Smaller than the main EBM ([256, 64]) since this is just a probe —
        we want to measure signal strength, not maximize accuracy.

    Returns:
        (train_accuracy, test_accuracy, energy_gap)

    Spec: REQ-INFER-016
    """
    import jax

    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    min_n = min(len(correct_acts), len(wrong_acts))
    if min_n < 10:
        return 0.5, 0.5, 0.0  # Not enough data

    rng = np.random.default_rng(seed)
    correct_acts = jnp.array(correct_acts[rng.permutation(len(correct_acts))[:min_n]])
    wrong_acts = jnp.array(wrong_acts[rng.permutation(len(wrong_acts))[:min_n]])

    split = int(min_n * 0.8)
    tc, tw = correct_acts[:split], wrong_acts[:split]
    vc, vw = correct_acts[split:], wrong_acts[split:]

    # Small probe model
    key = jrandom.PRNGKey(seed)
    config = GibbsConfig(input_dim=hidden_dim, hidden_dims=[128, 32], activation="silu")
    ebm = GibbsModel(config, key=key)

    def get_p(m: GibbsModel) -> dict:
        return {"layers": [(w, b) for w, b in m.layers],
                "output_weight": m.output_weight, "output_bias": m.output_bias}

    def set_p(m: GibbsModel, p: dict) -> None:
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    params = get_p(ebm)

    def loss_fn(p: dict) -> float:
        old = get_p(ebm)
        set_p(ebm, p)
        r = nce_loss(ebm, tc, tw)
        set_p(ebm, old)
        return r

    for _ in range(n_epochs):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    set_p(ebm, params)

    # Evaluate
    def eval_split(c: Any, w: Any) -> tuple[float, float]:
        n = min(300, len(c))
        ce = [float(ebm.energy(c[i])) for i in range(n)]
        we = [float(ebm.energy(w[i])) for i in range(n)]
        thresh = (sum(ce) / len(ce) + sum(we) / len(we)) / 2
        tp = sum(1 for e in we if e > thresh)
        tn = sum(1 for e in ce if e <= thresh)
        acc = (tp + tn) / (len(ce) + len(we))
        gap = sum(we) / len(we) - sum(ce) / len(ce)
        return acc, gap

    train_acc, _ = eval_split(tc, tw)
    test_acc, test_gap = eval_split(vc, vw)

    return train_acc, test_acc, test_gap


def probe_all_layers(
    correct_activations: dict[int, Any],
    wrong_activations: dict[int, Any],
    hidden_dim: int = 1024,
    n_epochs: int = 200,
    lr: float = 0.005,
    model_name: str = "",
    layers: list[int] | None = None,
) -> MultiLayerProbeResults:
    """Train a probe at each layer and find where hallucination signal is strongest.

    **Researcher summary:**
        For each transformer layer, trains a small Gibbs EBM probe and measures
        test accuracy. Reports which layer best separates correct from wrong.

    **Detailed explanation for engineers:**
        Takes pre-collected per-layer activations (dict mapping layer_index to
        numpy arrays of shape [n_tokens, hidden_dim]). For each layer, splits
        into train/test, trains a [128, 32] probe, and evaluates. Returns
        per-layer results sorted by layer index.

    Args:
        correct_activations: {layer_idx: np.array of correct token activations}
        wrong_activations: {layer_idx: np.array of wrong token activations}
        hidden_dim: Dimension of hidden states (default 1024).
        n_epochs: Training epochs per probe (default 200).
        lr: Learning rate (default 0.005).
        model_name: For display purposes.
        layers: Optional subset of layers to probe. If None, probes all.

    Returns:
        MultiLayerProbeResults with per-layer accuracy and best layer.

    Spec: REQ-INFER-016
    """
    all_layers = sorted(correct_activations.keys())
    if layers is not None:
        all_layers = [layer for layer in all_layers if layer in layers]

    results = []
    for layer_idx in all_layers:
        if layer_idx not in wrong_activations:
            continue

        c = correct_activations[layer_idx]
        w = wrong_activations[layer_idx]
        min_n = min(len(c), len(w))

        train_acc, test_acc, gap = train_layer_probe(
            c, w, hidden_dim, n_epochs=n_epochs, lr=lr, seed=42 + layer_idx,
        )

        split = int(min(len(c), len(w)) * 0.8)
        results.append(LayerProbeResult(
            layer_index=layer_idx,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            energy_gap=gap,
            n_train=split,
            n_test=min_n - split,
        ))

        logger.info("Layer %d: train=%.1f%%, test=%.1f%%, gap=%.4f",
                     layer_idx, train_acc * 100, test_acc * 100, gap)

    default = LayerProbeResult(0, 0.5, 0.5, 0.0, 0, 0)
    best = max(results, key=lambda r: r.test_accuracy) if results else default

    return MultiLayerProbeResults(
        layer_results=results,
        best_layer=best.layer_index,
        best_test_accuracy=best.test_accuracy,
        model_name=model_name,
    )

"""Symbolic-KAN Tier 3 verifier — ThreeTierPipeline integration module.

**Researcher summary:**
    Wraps a trained SymbolicKANModel as a ThreeTierPipeline Tier 3 callable.
    The Symbolic-KAN was validated in Exp 948 with AUC=1.0 on 57 real FoVer
    reasoning-step pairs (milestone 2026.04.73) and deployed in Exp 968.

**For engineers:**
    ThreeTierPipeline.ising_pipeline must be a callable:
        (response: str, question: str) -> (verified: bool, energy: float)
    This module provides SymbolicKANTier3, a class satisfying that interface,
    plus load_symbolic_kan() to restore a saved model from disk.

    Usage:
        from carnot.pipeline.symbolic_kan_tier3 import SymbolicKANTier3, load_symbolic_kan
        from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline

        model = load_symbolic_kan("symbolic_kan_v2_model/")
        tier3 = SymbolicKANTier3(model)
        pipeline = ThreeTierPipeline(sink_probe=..., eorm_model=..., ising_pipeline=tier3)

Spec: REQ-MODEL-030, REQ-VERIFY-088, SCENARIO-MODEL-015.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from carnot.models.symbolic_kan import SymbolicKANModel


def _extract_numbers(text: str) -> list[float]:
    """Extract decimal/integer literals from a LaTeX/text reasoning step."""
    clean = re.sub(r"\\[a-zA-Z]+", " ", text)
    tokens = re.findall(r"-?\d+(?:\.\d+)?", clean)
    return [float(t) for t in tokens]


def _operator_type(text: str) -> float:
    """Encode dominant operator type as a float (ADD=0.25, MUL=0.50, CMP=0.75, EQ=1.00)."""
    t = text.lower()
    if re.search(r"\btimes\b|\bmul\b|\bdivid\b|\bproduct\b|\bfactor\b", t):
        return 0.50
    if re.search(r"\bgreater\b|\bless\b|\bmore than\b|\bpercent\b|\brate\b", t):
        return 0.75
    if re.search(r"\bequal\b|\bresult\b|\btotal\b|\bsum\b|\bfinal\b", t):
        return 1.00
    return 0.25


def step_to_features(step_text: str, dim: int = 16) -> list[float]:
    """Encode a reasoning step as a 16-dim feature vector (Exp 948 encoding)."""
    nums = _extract_numbers(step_text)
    op = _operator_type(step_text)
    n_norm = min(len(nums), 20) / 20.0
    if nums:
        max_abs = max(abs(n) for n in nums) or 1.0
        norm_nums = [n / max_abs for n in nums]
    else:
        norm_nums = []
    feats = [op, n_norm] + norm_nums
    feats = feats[:dim]
    feats += [0.0] * (dim - len(feats))
    return feats


class SymbolicKANTier3:
    """SymbolicKAN-based Tier 3 verifier for ThreeTierPipeline.

    **For engineers:**
        Wraps a SymbolicKANModel so it can be passed as `ising_pipeline` to
        ThreeTierPipeline.  The model was trained with contrastive loss on
        (correct, incorrect) reasoning-step pairs from Exp 948; correct steps
        get low (negative) energy, incorrect steps get high (positive) energy.

        Decision boundary: energy < threshold (default 0.0).

    REQ-MODEL-030, REQ-VERIFY-088.
    """

    def __init__(self, model: SymbolicKANModel, threshold: float = 0.0) -> None:
        self.model = model
        self.threshold = threshold

    def __call__(self, response: str, question: str) -> tuple[bool, float]:  # noqa: ARG002
        """Compute energy from response text and return (verified, energy).

        `question` is accepted for API compatibility but not used — the model
        was trained on step-level features extracted from response text alone.
        Returns verified=True when energy < threshold.
        """
        feats = step_to_features(response, dim=16)
        x = np.array(feats, dtype=np.float32)
        energy = float(self.model.energy(x))
        return (energy < self.threshold, energy)


def load_symbolic_kan(model_dir: str | Path) -> SymbolicKANModel:
    """Load a SymbolicKANModel saved by Exp 968 from `model_dir/`.

    Reads config.json, symbolic_labels.json, and weights.npz to reconstruct
    the model in the exact state it was in after training.

    Why JSON + npz rather than safetensors: symbolic_labels is a list of strings
    which safetensors cannot natively serialise.
    """
    from carnot.models.symbolic_kan import ResidualSpline, SymbolicKANConfig, SymbolicKANModel

    d = Path(model_dir)

    config_data = json.loads((d / "config.json").read_text())
    config = SymbolicKANConfig(**config_data)

    labels = json.loads((d / "symbolic_labels.json").read_text())
    weights = np.load(d / "weights.npz")

    model = SymbolicKANModel(config, seed=0)  # seed is overwritten below
    model.in1 = weights["in1"]
    model.in2 = weights["in2"]
    model.global_bias = float(weights["global_bias"][0])
    model.symbolic_labels = labels

    for i in range(config.n_nodes):
        ctrl = weights[f"residual_{i}_ctrl"]
        model.residuals[i] = ResidualSpline(n_segments=config.n_segments)
        model.residuals[i].ctrl = ctrl

    return model

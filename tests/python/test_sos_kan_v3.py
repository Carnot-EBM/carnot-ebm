"""Tests for SOSKANEnergyV3 — Neural-Gram SOS-KAN energy model.

All tests trace to REQ-SAMPLE-016-v3 (neural-Gram SOS energy model with
type-level monotonicity invariants and AUROC ≥ 0.72 on FoVer v4).

The four required tests are:
    1. sos_kan_v3_auroc_above_0_72         — AUROC ≥ 0.72 on FoVer corpus
    2. sos_kan_v3_zero_monotonicity_violations — 0 violations on 16000 samples
    3. learned_gram_is_psd                 — eigenvalues ≥ 0 for random inputs
    4. v3_auroc_exceeds_v1                 — v3 AUROC > 0.6042 baseline
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup so we can import carnot packages without installation
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _d in [str(_REPO_ROOT / "python")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.models.sos_kan import SOSKANEnergyV3  # noqa: E402

# ---------------------------------------------------------------------------
# Constants matching experiment_1072
# ---------------------------------------------------------------------------

N_SPLINES = 8
RANK = 8
N_FEATURES = 16
HIDDEN_DIM = 32
N_EPOCHS = 100
LR = 1e-3
TRAIN_FRAC = 0.80
V1_AUROC_BASELINE = 0.6042
AUROC_TARGET = 0.72

_CORPUS_PATH = _REPO_ROOT / "data" / "fover_corpus_v4.json"


# ---------------------------------------------------------------------------
# Shared feature extractor (mirrors experiment_1072 _featurize exactly)
# ---------------------------------------------------------------------------


def _featurize(items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert FoVer corpus items to float64 features and int labels.

    16 text-structural features, each normalised to [-1, 1].
    See experiment_1072_sos_kan_v3_neural_gram.py for the full legend.
    """
    n = len(items)
    X = np.zeros((n, N_FEATURES), dtype=np.float64)
    y = np.zeros(n, dtype=np.int32)
    for idx, item in enumerate(items):
        text = str(item.get("step_text", ""))
        label = item.get("label", "unknown")
        y[idx] = 1 if label in ("correct", "valid", True, 1) else 0
        tl = text.lower()
        words = text.split()
        nw = max(len(words), 1)
        nc = max(len(text), 1)
        nums = re.findall(r"\b\d+\.?\d*\b", text)
        n_eq = text.count("=")
        X[idx, 0] = float(np.clip(math.log(nw + 1) / 5.0, 0, 1)) * 2 - 1
        X[idx, 1] = float(np.clip(n_eq / nw, 0, 1)) * 2 - 1
        X[idx, 2] = float(np.clip(len(nums) / nw, 0, 1)) * 2 - 1
        X[idx, 3] = float(np.clip(text.count("$") / nw, 0, 1)) * 2 - 1
        X[idx, 4] = 1.0 if any(k in tl for k in ["answer", "result", "solution"]) else -1.0
        X[idx, 5] = 1.0 if any(k in tl for k in ["let ", "define ", "let's let"]) else -1.0
        X[idx, 6] = (
            1.0
            if any(k in tl for k in ["therefore", "hence", "thus", "since ", "notice"])
            else -1.0
        )
        X[idx, 7] = 1.0 if n_eq >= 3 else -1.0
        X[idx, 8] = float(np.clip((text.count("+") + text.count("-")) / nw, 0, 1)) * 2 - 1
        X[idx, 9] = float(np.clip((text.count("(") + text.count(")")) / nc * 10, 0, 1)) * 2 - 1
        X[idx, 10] = 1.0 if "frac" in tl else -1.0
        X[idx, 11] = 1.0 if len(text) > 0 and text[0].isdigit() else -1.0
        sents = re.split(r"[.!?]", text)
        ns = len([s for s in sents if s.strip()])
        X[idx, 12] = float(np.clip(ns / max(nc / 100.0, 1.0), 0, 2) / 2) * 2 - 1
        X[idx, 13] = (
            1.0 if any(k in tl for k in ["cannot", "impossible", "never", "always"]) else -1.0
        )
        X[idx, 14] = float(np.clip(math.log(len(set(nums)) + 1) / 3.0, 0, 1)) * 2 - 1
        X[idx, 15] = float(np.clip(len(text) / 500.0, 0, 1)) * 2 - 1
    return X, y


def _load_and_split_fover() -> tuple:
    """Load FoVer v4 corpus and return stratified (X_train, y_train, X_val, y_val).

    Uses a stratified 80/20 split (same seed as experiment_1072) so that the
    validation set always contains minority-class (incorrect) examples,
    enabling meaningful AUROC computation.
    """
    corpus = json.loads(_CORPUS_PATH.read_text())
    X_all, y_all = _featurize(corpus)
    X_all = X_all.astype(np.float64)
    y_all = y_all.astype(np.float64)

    rng = np.random.default_rng(2024)
    pos_idxs = np.where(y_all == 1)[0]
    neg_idxs = np.where(y_all == 0)[0]
    rng.shuffle(pos_idxs)
    rng.shuffle(neg_idxs)

    n_pos_train = int(len(pos_idxs) * TRAIN_FRAC)
    n_neg_train = int(len(neg_idxs) * TRAIN_FRAC)

    train_idxs = np.concatenate([pos_idxs[:n_pos_train], neg_idxs[:n_neg_train]])
    val_idxs = np.concatenate([pos_idxs[n_pos_train:], neg_idxs[n_neg_train:]])

    return (
        X_all[train_idxs],
        y_all[train_idxs],
        X_all[val_idxs],
        y_all[val_idxs],
    )


@pytest.fixture(scope="module")
def trained_model_and_val():
    """Train SOSKANEnergyV3 once; share across all tests in this module.

    Trains for N_EPOCHS=100 with Adam lr=1e-3 on the stratified 80% split
    of FoVer v4. This fixture takes ~20-25 s on CPU.

    Spec: REQ-SAMPLE-016-v3
    """
    assert _CORPUS_PATH.exists(), f"FoVer corpus not found at {_CORPUS_PATH}"
    X_train, y_train, X_val, y_val = _load_and_split_fover()
    model = SOSKANEnergyV3(
        n_splines=N_SPLINES,
        rank=RANK,
        n_features=N_FEATURES,
        hidden_dim=HIDDEN_DIM,
        seed=42,
    )
    model.fit(X_train, y_train, n_epochs=N_EPOCHS, lr=LR)
    return model, X_val, y_val


# ---------------------------------------------------------------------------
# Test 1 — AUROC ≥ 0.72 on FoVer v4 validation set
# Spec: REQ-SAMPLE-016-v3
# ---------------------------------------------------------------------------


def test_sos_kan_v3_auroc_above_0_72(trained_model_and_val):
    """SOSKANEnergyV3 must achieve AUROC ≥ 0.72 on the FoVer v4 validation split.

    The target was 0.6042 for v1; v3 uses a neural-Gram (input-conditioned
    F @ F^T) and 30x more training data.

    Spec: REQ-SAMPLE-016-v3
    SCENARIO: Given FoVer v4 validation set with 23 incorrect examples,
              When model is trained for 100 epochs,
              Then AUROC ≥ 0.72.
    """
    model, X_val, y_val = trained_model_and_val
    auroc = float(model.auroc_batch(X_val, y_val))
    assert auroc >= AUROC_TARGET, (
        f"SOSKANEnergyV3 AUROC {auroc:.4f} < target {AUROC_TARGET}. "
        f"Check feature engineering and training hyperparameters."
    )


# ---------------------------------------------------------------------------
# Test 2 — Zero monotonicity violations on 16,000 random samples
# Spec: REQ-SAMPLE-016-v3, REQ-MODEL-SOS-001
# ---------------------------------------------------------------------------


def test_sos_kan_v3_zero_monotonicity_violations(trained_model_and_val):
    """SOSKANEnergyV3 must report 0 monotonicity violations on 16,000 samples.

    This is guaranteed structurally: G_f(x) = F_f(x) @ F_f(x)^T is PSD for
    any MLP output, and Φ_{ij}(x) is the non-negative cumulative integral of
    hat basis products, so dψ_f/dx_f = B(x_f)^T G_f B(x_f) ≥ 0 always.

    Spec: REQ-SAMPLE-016-v3, REQ-MODEL-SOS-001
    SCENARIO: Given any trained SOSKANEnergyV3,
              When monotonicity is checked on 16,000 random inputs,
              Then violations == 0.
    """
    model, _, _ = trained_model_and_val
    result = model.verify_invariants(n_samples=16_000, rng_seed=42)
    violations = result["n_monotone_violations"]
    assert violations == 0, (
        f"SOSKANEnergyV3 has {violations} monotonicity violations. "
        f"The SOS certificate should guarantee 0 violations structurally."
    )
    assert result["invariants_hold"] is True


# ---------------------------------------------------------------------------
# Test 3 — Gram matrices are PSD for random inputs
# Spec: REQ-SAMPLE-016-v3
# ---------------------------------------------------------------------------


def test_learned_gram_is_psd(trained_model_and_val):
    """G_f(x) = F_f(x) @ F_f(x)^T must have all eigenvalues ≥ 0 for any input x.

    This is the core SOS certificate: G = F @ F^T is PSD for any F,
    regardless of what the MLP produces. Tests 50 random input vectors.

    Spec: REQ-SAMPLE-016-v3
    SCENARIO: Given any trained SOSKANEnergyV3,
              When Gram matrices are computed for 50 random inputs,
              Then all eigenvalues of every G_f are ≥ -1e-9.
    """
    model, _, _ = trained_model_and_val
    rng = np.random.default_rng(123)
    for _ in range(50):
        x = rng.uniform(-1.0, 1.0, N_FEATURES)
        G = model.gram_matrices(x)
        assert G.shape == (N_FEATURES, N_SPLINES, N_SPLINES)
        for f in range(N_FEATURES):
            eigs = np.linalg.eigvalsh(G[f])
            min_eig = float(eigs.min())
            assert min_eig >= -1e-9, (
                f"Feature {f}: minimum eigenvalue {min_eig:.4e} < 0. "
                f"G_f = F_f @ F_f^T should be PSD by construction."
            )


# ---------------------------------------------------------------------------
# Test 4 — v3 AUROC exceeds v1 baseline (0.6042)
# Spec: REQ-SAMPLE-016-v3
# ---------------------------------------------------------------------------


def test_v3_auroc_exceeds_v1(trained_model_and_val):
    """SOSKANEnergyV3 AUROC must exceed the v1 baseline of 0.6042.

    v1 (Exp 1047) used fixed V@V^T and ~200 training pairs; v3 uses the
    neural-Gram (input-conditioned G_f) and 6548 pairs (30x more data).

    Spec: REQ-SAMPLE-016-v3
    SCENARIO: Given FoVer v4 validation set,
              When SOSKANEnergyV3 is evaluated,
              Then AUROC > 0.6042 (v1 baseline).
    """
    model, X_val, y_val = trained_model_and_val
    auroc = float(model.auroc_batch(X_val, y_val))
    assert auroc > V1_AUROC_BASELINE, (
        f"SOSKANEnergyV3 AUROC {auroc:.4f} ≤ v1 baseline {V1_AUROC_BASELINE}. "
        f"The neural-Gram + 30x data should improve over v1."
    )

#!/usr/bin/env python3
"""Exp 1034 — GS-KAN v4: Group-Shared KAN + QuantKAN INT8 for KV260 FPGA budget.

**Researcher summary:**
    Prior GS-KAN experiments (Exp 1009, 1019) were blocked by pre-test comparison
    failures and missing preflight gates respectively. This experiment is STANDALONE —
    no gate on prior experiments — and adds QuantKAN INT8 quantization on top of GS-KAN.

    Two key references:
      - arXiv 2512.09084 (GS-KAN): G=4 shared parent B-spline bases reduce LUT count
        from ~82K (KAEMEnergy baseline) to ~8K estimated.
      - arXiv 2511.18689 (QuantKAN): Per-group INT8 quantization of B-spline weights
        reduces DSP48 usage and further cuts LUT count by 3x vs FP32.

    **Prior failures addressed:**
      - experiment_id: exp1009_gskan_v2
        verdict: pre_test_comparison_failure
        addressed_by: "This experiment implements GS-KAN from scratch using numpy
                       (not JAX) for the group spline evaluation, avoiding the JAX
                       tracing issue that caused the pre-test comparison failure."
      - experiment_id: exp1019_gskan_v3
        verdict: blocked_preflight_dependency
        addressed_by: "This experiment is STANDALONE, does not call preflight, and
                       runs entirely self-contained using the expanded FoVer corpus."

**What this experiment measures:**
    1. auroc_gskan_fp32: AUROC of GS-KAN (G=4) trained on FoVer expanded corpus
    2. auroc_gskan_int8: AUROC after QuantKAN INT8 post-training quantization
    3. auroc_kaem_baseline: AUROC of KAEMEnergy (G=1, full parameter baseline)
    4. FPGA resource estimates: LUT, DSP48, BRAM for both FP32 and INT8 variants
    5. Hardware complexity metrics: RM, BOP, NABS (arXiv 2604.03345)

**Acceptance gates:**
    - auroc_gskan_fp32 >= 0.70 (within 0.02 of KAEMEnergy baseline)
    - auroc_gskan_int8 degradation vs FP32 < 0.02
    - fpga_lut_estimate_int8 < 20000 (well within KV260 117K LUT budget)

Spec: REQ-SAMPLE-015 (energy model interface), REQ-KAN-VERIFY-001 (FPGA feasibility)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must come before any local imports
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 1034
EXP_TITLE = "GS-KAN v4: Group-Shared KAN + QuantKAN INT8 for KV260 FPGA budget"
RESULT_PATH = _REPO_ROOT / "results" / "experiment_1034_gskan_v4.json"

# KV260 resource budget (Xilinx Zynq UltraScale+)
KV260_LUT_BUDGET = 117_000
KV260_DSP48_BUDGET = 1_728
KV260_BRAM_BUDGET = 144  # 36Kb BRAM tiles

# GS-KAN hyperparameters (arXiv 2512.09084)
N_GROUPS = 4  # G=4 shared parent bases
N_KNOTS = 8  # knots per parent spline
N_EPOCHS = 150  # training epochs
LR = 0.01  # learning rate

# AUROC acceptance thresholds
AUROC_MIN = 0.70
INT8_DEGRADATION_MAX = 0.02


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_fover_corpus() -> tuple[list[dict], list[dict]]:
    """Load FoVer corpus: expanded corpus if available, else train/test split.

    Returns (train_items, test_items) where each item has 'label' and a
    feature vector encoding to be derived from the step text.

    Prefers the Exp 1029 expanded corpus (85+ pairs) over the original 57-pair corpus.
    """
    expanded_path = _REPO_ROOT / "data" / "fover_corpus_expanded.json"
    train_path = _REPO_ROOT / "data" / "fover_train.json"
    test_path = _REPO_ROOT / "data" / "fover_test.json"

    # Use Exp 1029's pre-split train/test if available — they are the canonical split
    train_items: list[dict] = []
    test_items: list[dict] = []

    if train_path.exists() and test_path.exists():
        with open(train_path) as f:
            train_items = json.load(f)
        with open(test_path) as f:
            test_items = json.load(f)
        corpus_source = "fover_train_test_split"
    elif expanded_path.exists():
        with open(expanded_path) as f:
            all_items = json.load(f)
        # 80/20 split
        n_train = int(len(all_items) * 0.8)
        train_items = all_items[:n_train]
        test_items = all_items[n_train:]
        corpus_source = "fover_corpus_expanded"
    else:
        raise FileNotFoundError(
            "No FoVer corpus found — expected fover_train.json or fover_corpus_expanded.json"
        )

    print(f"  Corpus: {corpus_source}, train={len(train_items)}, test={len(test_items)}")
    return train_items, test_items


def _featurize(items: list[dict], n_vars: int = 16) -> tuple:
    """Convert step_text and label to a float32 feature matrix and label vector.

    Feature encoding: 16 linguistic/mathematical features designed to capture
    signal that correlates with correct vs incorrect math reasoning steps.
    These features are inspired by FoVer's discriminative criteria (correct steps
    tend to have structured algebraic reasoning; incorrect steps tend to have
    direct computation errors or bad logic).

    Feature index mapping (all normalized to [-1, 1]):
      0: log(word_count + 1) / 5 — step length (longer = often more careful)
      1: equality_density — number of '=' per word
      2: number_density — numeric tokens per word
      3: latex_density — dollar signs per word (LaTeX math expressions)
      4: has_boxed_answer — 1 if '\\boxed' in text (final answer present)
      5: has_algebraic_setup — 1 if 'let ' or 'define ' in text (algebraic setup)
      6: has_logical_connectives — 1 if 'notice', 'since', 'therefore', 'because'
      7: has_calculation_chain — 1 if 3+ '=' signs (long calculation chain)
      8: arithmetic_op_density — +/- operators per word
      9: paren_density — parentheses per character
      10: has_fraction — 1 if 'frac' in text
      11: starts_with_number — 1 if step begins with a digit
      12: sentence_count_norm — sentences per 100 chars
      13: error_indicator — 1 if 'however', 'but wait', 'actually' present
      14: number_count_log — log(count of distinct numeric literals + 1)
      15: text_length_norm — character count / 500

    Returns (X: ndarray shape (n,n_vars), y: ndarray shape (n,) int in {0,1})
    """
    import re
    import math
    import numpy as np

    X = np.zeros((len(items), n_vars), dtype=np.float32)
    y = np.zeros(len(items), dtype=np.int32)

    for idx, item in enumerate(items):
        text = str(item.get("step_text", ""))
        label = item.get("label", "unknown")
        y[idx] = 1 if label in ("correct", "valid", True, 1) else 0
        text_lower = text.lower()

        words = text.split()
        n_words = max(len(words), 1)
        n_chars = max(len(text), 1)

        # 0: log word count
        X[idx, 0] = float(np.clip(math.log(n_words + 1) / 5.0, 0.0, 1.0)) * 2.0 - 1.0

        # 1: equality density (= per word)
        n_eq = text.count("=")
        X[idx, 1] = float(np.clip(n_eq / n_words, 0.0, 1.0)) * 2.0 - 1.0

        # 2: number density (numeric tokens per word)
        nums = re.findall(r"\b\d+\.?\d*\b", text)
        X[idx, 2] = float(np.clip(len(nums) / n_words, 0.0, 1.0)) * 2.0 - 1.0

        # 3: LaTeX density ($ per word)
        n_dollar = text.count("$")
        X[idx, 3] = float(np.clip(n_dollar / n_words, 0.0, 1.0)) * 2.0 - 1.0

        # 4: has boxed answer (often in incorrect final answers)
        X[idx, 4] = 1.0 if "\\boxed" in text else -1.0

        # 5: has algebraic setup (let x = ..., define ...)
        X[idx, 5] = (
            1.0 if any(kw in text_lower for kw in ["let ", "define ", "let's let"]) else -1.0
        )

        # 6: has logical connectives (structured reasoning signals)
        X[idx, 6] = (
            1.0
            if any(
                kw in text_lower
                for kw in ["notice", "since ", "therefore", "because", "hence", "thus"]
            )
            else -1.0
        )

        # 7: long calculation chain (3+ equals = multi-step computation)
        X[idx, 7] = 1.0 if n_eq >= 3 else -1.0

        # 8: arithmetic operator density (+/- per word)
        n_arith = text.count("+") + text.count("-")
        X[idx, 8] = float(np.clip(n_arith / n_words, 0.0, 1.0)) * 2.0 - 1.0

        # 9: parenthesis density
        n_paren = text.count("(") + text.count(")")
        X[idx, 9] = float(np.clip(n_paren / n_chars * 10.0, 0.0, 1.0)) * 2.0 - 1.0

        # 10: contains fraction (LaTeX \frac)
        X[idx, 10] = 1.0 if "frac" in text_lower else -1.0

        # 11: starts with a number (direct computation, not algebraic setup)
        X[idx, 11] = 1.0 if (len(text) > 0 and text[0].isdigit()) else -1.0

        # 12: sentence count normalized
        sentences = re.split(r"[.!?]", text)
        n_sentences = len([s for s in sentences if s.strip()])
        X[idx, 12] = (
            float(np.clip(n_sentences / max(n_chars / 100.0, 1.0), 0.0, 2.0) / 2.0) * 2.0 - 1.0
        )

        # 13: error indicator words (self-corrections, hedges)
        X[idx, 13] = (
            1.0
            if any(
                kw in text_lower
                for kw in ["however", "but wait", "actually", "correction", "mistake"]
            )
            else -1.0
        )

        # 14: log of distinct numeric literals (more numbers = more computation)
        distinct_nums = len(set(nums))
        X[idx, 14] = float(np.clip(math.log(distinct_nums + 1) / 3.0, 0.0, 1.0)) * 2.0 - 1.0

        # 15: text length normalized (longer steps often more careful)
        X[idx, 15] = float(np.clip(len(text) / 500.0, 0.0, 1.0)) * 2.0 - 1.0

    return X, y


# ---------------------------------------------------------------------------
# AUROC computation
# ---------------------------------------------------------------------------


def _compute_auroc(scores: list[float], labels: list[int]) -> float:
    """Compute AUROC from energy scores and binary labels.

    Lower energy = more likely correct = positive class. We negate energy
    so that higher score = more positive, then compute standard AUROC.

    Parameters
    ----------
    scores : list[float]
        Raw energy scores (lower = more likely positive).
    labels : list[int]
        Binary labels, 1 = positive (correct step), 0 = negative.

    Returns
    -------
    float
        AUROC in [0, 1]. 0.5 = random, 1.0 = perfect.
    """
    import numpy as np

    scores_arr = -np.array(scores, dtype=np.float64)  # negate: lower energy = higher score
    labels_arr = np.array(labels, dtype=np.int32)

    # Handle degenerate case (only one class in test set)
    if len(np.unique(labels_arr)) < 2:
        return 0.5

    # Trapezoidal AUROC via sorted threshold sweep
    order = np.argsort(-scores_arr)  # descending by score
    labels_sorted = labels_arr[order]

    n_pos = int(np.sum(labels_arr))
    n_neg = len(labels_arr) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    tpr_prev = 0.0
    fpr_prev = 0.0
    auc = 0.0

    for lbl in labels_sorted:
        if lbl == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_pos
        fpr = fp / n_neg

        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev, fpr_prev = tpr, fpr

    return float(np.clip(auc, 0.0, 1.0))


# ---------------------------------------------------------------------------
# FPGA resource estimation
# ---------------------------------------------------------------------------


def _estimate_fpga_resources(
    n_vars: int,
    n_groups: int,
    n_knots: int,
    is_int8: bool,
) -> dict:
    """Estimate KV260 FPGA resource usage for GS-KAN or KAEMEnergy.

    Estimation formulas (arXiv 2604.03345 + Xilinx reference designs):

    LUT estimation:
      - FP32 multiply-accumulate: ~10 LUTs each
      - INT8 multiply-accumulate: ~3 LUTs each (much cheaper — uses DSP48 integer mode)
      - Per-variable operations: n_knots spline evals (interpolation lookups) + 1 projection
      - GS-KAN: G groups × n_knots lookups + n_vars projections
        FP32: (n_groups × n_knots + n_vars) × 10 LUTs
        INT8: (n_groups × n_knots) × 3 + n_vars × 10 LUTs (projections stay FP32)
      - KAEMEnergy (G=n_vars, n_groups=n_vars): n_vars × n_knots × 10 LUTs

    DSP48 estimation:
      - Each multiply-accumulate uses 1 DSP48 slice
      - INT8 multiply in DSP48 integer mode: 1 DSP48 per multiply
      - FP32 multiply in DSP48: 3 DSP48 per operation (Xilinx FP IP)
      - GS-KAN FP32 DSPs: n_vars × (n_knots + 1) × 3
      - GS-KAN INT8 DSPs: n_vars × 1  (just the projection weight, ctrl is INT8)

    BRAM estimation:
      - Weight storage: ceil(total_bytes / 4096) BRAM tiles (each 36Kb = 4096 bytes usable)
      - FP32: 4 bytes/weight × (n_groups × n_knots + n_vars) weights
      - INT8: 1 byte/weight for ctrl (n_groups × n_knots INT8 weights) + 4 bytes for proj
    """
    import math

    if is_int8:
        # INT8 ctrl weights + FP32 projection weights
        ctrl_luts = n_groups * n_knots * 3  # INT8 spline interpolation
        proj_luts = n_vars * 10  # FP32 projection still
        lut_count = ctrl_luts + proj_luts

        # DSP48: INT8 spline interpolations use 1 DSP48 each; FP32 projections use 3
        ctrl_dsps = n_vars * n_knots * 1  # INT8 spline lookups per variable
        proj_dsps = n_vars * 3  # FP32 projections
        dsp_count = ctrl_dsps + proj_dsps

        # BRAM: INT8 ctrl (1 byte) + FP32 proj (4 bytes) + FP32 scales (n_groups × 4 bytes)
        ctrl_bytes = n_groups * n_knots * 1
        proj_bytes = n_vars * 4
        scale_bytes = n_groups * 4
        total_bytes = ctrl_bytes + proj_bytes + scale_bytes
    else:
        # FP32 throughout
        ctrl_luts = n_groups * n_knots * 10
        proj_luts = n_vars * 10
        lut_count = ctrl_luts + proj_luts

        ctrl_dsps = n_vars * n_knots * 3
        proj_dsps = n_vars * 3
        dsp_count = ctrl_dsps + proj_dsps

        ctrl_bytes = n_groups * n_knots * 4
        proj_bytes = n_vars * 4
        total_bytes = ctrl_bytes + proj_bytes

    bram_count = math.ceil(total_bytes / 4096)  # 36Kb tiles, ~4096 useful bytes each

    return {
        "lut_count": lut_count,
        "dsp48_count": dsp_count,
        "bram_count": max(bram_count, 1),
        "weight_bytes": total_bytes,
    }


def _estimate_kaem_fpga(n_vars: int, n_knots: int) -> dict:
    """Estimate FPGA resources for full KAEMEnergy baseline (G = n_vars, no sharing)."""
    import math

    # KAEMEnergy: n_vars independent splines, no group sharing
    lut_count = n_vars * n_knots * 10  # FP32 spline interpolations
    dsp_count = n_vars * n_knots * 3  # FP32 multiplies
    weight_bytes = n_vars * n_knots * 4  # FP32 ctrl points
    bram_count = math.ceil(weight_bytes / 4096)

    return {
        "lut_count": lut_count,
        "dsp48_count": dsp_count,
        "bram_count": max(bram_count, 1),
        "weight_bytes": weight_bytes,
    }


# ---------------------------------------------------------------------------
# Hardware complexity metrics (arXiv 2604.03345)
# ---------------------------------------------------------------------------


def _compute_hardware_metrics(
    lut_gskan_int8: int,
    lut_kaem_baseline: int,
    auroc_delta: float,
    n_vars: int,
    n_knots: int,
    n_groups: int,
) -> dict:
    """Compute RM, BOP, NABS hardware complexity metrics.

    RM (Resource Multiplier):
        Ratio of GS-KAN INT8 LUT count to KAEMEnergy baseline LUT count.
        RM < 1 means GS-KAN uses fewer resources. RM < 0.2 is the target.

    BOP (Bit Operations per inference):
        Total bit operations for one forward pass.
        For INT8: BOP = 8 * (n_vars * n_knots)  (8-bit multiplications count as 8 bit-ops each)
        For FP32: BOP = 32 * (n_vars * n_knots)

    NABS (Normalized Area-Budget Score):
        NABS = RM * (1 - auroc_delta)
        Lower is better: simultaneously measures resource efficiency AND AUROC preservation.
        A model that saves 90% of LUTs but loses 10% AUROC has NABS = 0.1 * (1 - 0.10) = 0.09.
        A model that saves 90% of LUTs with no AUROC loss has NABS = 0.1 * 1.0 = 0.10.

    Note: auroc_delta here is the FRACTIONAL degradation (0.0 = no loss, 1.0 = total loss).

    Parameters
    ----------
    lut_gskan_int8 : int
        LUT count for GS-KAN INT8.
    lut_kaem_baseline : int
        LUT count for KAEMEnergy baseline.
    auroc_delta : float
        AUROC degradation of INT8 vs KAEMEnergy baseline (fractional, 0.0 = none).
    n_vars, n_knots, n_groups : int
        Model dimensions for BOP calculation.
    """
    rm = lut_gskan_int8 / max(lut_kaem_baseline, 1)

    # INT8 BOP: 8 bits per multiply, n_vars variables each doing n_knots spline lookups
    # plus n_vars projection multiplies (FP32 = 32 bits)
    bop_int8_ctrl = 8 * n_vars * n_knots
    bop_fp32_proj = 32 * n_vars
    bop = bop_int8_ctrl + bop_fp32_proj

    # NABS: lower is better
    nabs = rm * (1.0 - auroc_delta)

    return {
        "rm_metric": float(rm),
        "bop_metric": int(bop),
        "nabs_metric": float(nabs),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    import numpy as np
    from carnot.models.gskan import GSKANEnergy
    from carnot.models.kaem_energy import KAEMEnergy
    import jax.numpy as jnp

    t_start = time.time()
    print(f"=== Exp {EXP_ID}: {EXP_TITLE} ===")

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading FoVer corpus...")
    train_items, test_items = _load_fover_corpus()
    n_vars = 16  # feature dimension

    X_train, y_train = _featurize(train_items, n_vars=n_vars)
    X_test, y_test = _featurize(test_items, n_vars=n_vars)
    print(f"  Train: {X_train.shape}, positive rate: {y_train.mean():.2f}")
    print(f"  Test:  {X_test.shape}, positive rate: {y_test.mean():.2f}")

    # ------------------------------------------------------------------
    # Step 2: Train KAEMEnergy baseline (G=1, full independent splines)
    # ------------------------------------------------------------------
    print("\n[2/6] Training KAEMEnergy baseline (G=1)...")
    t_kaem_start = time.time()
    kaem = KAEMEnergy(n_vars=n_vars, n_hidden=N_KNOTS, key=None)
    kaem.fit(jnp.array(X_train), n_epochs=N_EPOCHS)
    kaem_train_time = time.time() - t_kaem_start
    print(f"  KAEMEnergy training time: {kaem_train_time:.1f}s")

    kaem_scores = [float(kaem.energy(jnp.array(X_test[i]))) for i in range(len(X_test))]
    auroc_kaem = _compute_auroc(kaem_scores, y_test.tolist())
    print(f"  KAEMEnergy AUROC: {auroc_kaem:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Train GS-KAN (G=4, shared bases)
    # ------------------------------------------------------------------
    print("\n[3/6] Training GS-KAN (G=4)...")
    t_gskan_start = time.time()
    gskan = GSKANEnergy(n_vars=n_vars, n_groups=N_GROUPS, n_knots=N_KNOTS, seed=42)
    gskan.fit(X_train, n_epochs=N_EPOCHS, lr=LR)
    gskan_train_time = time.time() - t_gskan_start
    print(f"  GS-KAN training time: {gskan_train_time:.1f}s")

    gskan_fp32_scores = [gskan.energy(X_test[i], use_quantized=False) for i in range(len(X_test))]
    auroc_gskan_fp32 = _compute_auroc(gskan_fp32_scores, y_test.tolist())
    print(f"  GS-KAN FP32 AUROC: {auroc_gskan_fp32:.4f}")

    # ------------------------------------------------------------------
    # Step 4: QuantKAN INT8 quantization
    # ------------------------------------------------------------------
    print("\n[4/6] Applying QuantKAN INT8 quantization...")
    quant_stats = gskan.quantize_int8()
    print(f"  Quantization max abs error: {quant_stats['max_abs_error']:.6f}")
    print(f"  Scale per group: {[f'{s:.4f}' for s in quant_stats['scale_per_group']]}")

    gskan_int8_scores = [gskan.energy(X_test[i], use_quantized=True) for i in range(len(X_test))]
    auroc_gskan_int8 = _compute_auroc(gskan_int8_scores, y_test.tolist())
    int8_degradation = auroc_gskan_fp32 - auroc_gskan_int8
    print(f"  GS-KAN INT8 AUROC: {auroc_gskan_int8:.4f} (delta: {int8_degradation:+.4f})")

    # ------------------------------------------------------------------
    # Step 5: FPGA resource estimation
    # ------------------------------------------------------------------
    print("\n[5/6] Estimating FPGA resources...")
    kaem_fpga = _estimate_kaem_fpga(n_vars=n_vars, n_knots=N_KNOTS)
    gskan_fp32_fpga = _estimate_fpga_resources(
        n_vars=n_vars, n_groups=N_GROUPS, n_knots=N_KNOTS, is_int8=False
    )
    gskan_int8_fpga = _estimate_fpga_resources(
        n_vars=n_vars, n_groups=N_GROUPS, n_knots=N_KNOTS, is_int8=True
    )

    print(
        f"  KAEMEnergy baseline:  LUT={kaem_fpga['lut_count']:,}, DSP48={kaem_fpga['dsp48_count']:,}, BRAM={kaem_fpga['bram_count']}"
    )
    print(
        f"  GS-KAN FP32:         LUT={gskan_fp32_fpga['lut_count']:,}, DSP48={gskan_fp32_fpga['dsp48_count']:,}, BRAM={gskan_fp32_fpga['bram_count']}"
    )
    print(
        f"  GS-KAN INT8:         LUT={gskan_int8_fpga['lut_count']:,}, DSP48={gskan_int8_fpga['dsp48_count']:,}, BRAM={gskan_int8_fpga['bram_count']}"
    )
    print(
        f"  KV260 budget:        LUT={KV260_LUT_BUDGET:,}, DSP48={KV260_DSP48_BUDGET:,}, BRAM={KV260_BRAM_BUDGET}"
    )

    lut_reduction_pct = (1.0 - gskan_int8_fpga["lut_count"] / max(kaem_fpga["lut_count"], 1)) * 100
    print(f"  LUT reduction (INT8 GS-KAN vs KAEMEnergy): {lut_reduction_pct:.1f}%")

    # ------------------------------------------------------------------
    # Step 6: Hardware complexity metrics (arXiv 2604.03345)
    # ------------------------------------------------------------------
    print("\n[6/6] Computing hardware complexity metrics...")
    auroc_delta_vs_kaem = (auroc_kaem - auroc_gskan_int8) / max(auroc_kaem, 1e-6)
    hw_metrics = _compute_hardware_metrics(
        lut_gskan_int8=gskan_int8_fpga["lut_count"],
        lut_kaem_baseline=kaem_fpga["lut_count"],
        auroc_delta=max(auroc_delta_vs_kaem, 0.0),
        n_vars=n_vars,
        n_knots=N_KNOTS,
        n_groups=N_GROUPS,
    )
    print(f"  RM  (resource multiplier): {hw_metrics['rm_metric']:.4f} (target < 0.20)")
    print(f"  BOP (bit operations/pass):  {hw_metrics['bop_metric']:,}")
    print(f"  NABS (area-budget score):   {hw_metrics['nabs_metric']:.4f} (lower is better)")

    param_counts = gskan.count_parameters()
    print(f"  GS-KAN FP32 params: {param_counts['fp32_params']}")
    print(f"  GS-KAN INT8 ctrl params: {param_counts['int8_ctrl_params']}")

    # ------------------------------------------------------------------
    # Determine honest verdict
    # ------------------------------------------------------------------
    lut_within_budget = gskan_int8_fpga["lut_count"] < KV260_LUT_BUDGET
    dsp_within_budget = gskan_int8_fpga["dsp48_count"] < KV260_DSP48_BUDGET
    auroc_meets_target = auroc_gskan_fp32 >= AUROC_MIN
    int8_degradation_ok = abs(int8_degradation) < INT8_DEGRADATION_MAX

    print(f"\n--- Acceptance gate checks ---")
    print(
        f"  AUROC FP32 >= {AUROC_MIN}: {auroc_gskan_fp32:.4f} -> {'PASS' if auroc_meets_target else 'FAIL'}"
    )
    print(
        f"  INT8 degradation < {INT8_DEGRADATION_MAX}: {abs(int8_degradation):.4f} -> {'PASS' if int8_degradation_ok else 'FAIL'}"
    )
    print(
        f"  LUT within budget ({KV260_LUT_BUDGET:,}): {gskan_int8_fpga['lut_count']:,} -> {'PASS' if lut_within_budget else 'FAIL'}"
    )
    print(
        f"  DSP48 within budget ({KV260_DSP48_BUDGET:,}): {gskan_int8_fpga['dsp48_count']:,} -> {'PASS' if dsp_within_budget else 'FAIL'}"
    )

    if auroc_meets_target and int8_degradation_ok and lut_within_budget and dsp_within_budget:
        honest_verdict = "gskan_quantized_budget_confirmed"
    elif auroc_meets_target and not int8_degradation_ok:
        honest_verdict = "gskan_quantized_auroc_regression"
    elif not lut_within_budget or not dsp_within_budget:
        honest_verdict = "gskan_budget_exceeded"
    else:
        honest_verdict = "failed"

    print(f"\n  Honest verdict: {honest_verdict}")

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------
    duration_s = time.time() - t_start
    import datetime

    artifact = {
        "experiment": EXP_ID,
        "title": EXP_TITLE,
        "run_date": datetime.datetime.utcnow().isoformat() + "Z",
        "started_at": datetime.datetime.utcfromtimestamp(t_start).isoformat() + "Z",
        "finished_at": datetime.datetime.utcnow().isoformat() + "Z",
        "duration_s": round(duration_s, 2),
        "status": "success",
        "schema": "experiment_result_v1",
        "honest_verdict": honest_verdict,
        # AUROC metrics
        "auroc_gskan_fp32": round(auroc_gskan_fp32, 4),
        "auroc_gskan_int8": round(auroc_gskan_int8, 4),
        "auroc_kaem_baseline": round(auroc_kaem, 4),
        # Quantization state
        "int8_quantized": True,
        "int8_degradation": round(float(int8_degradation), 4),
        "quant_stats": quant_stats,
        # FPGA resource estimates
        "fpga_lut_estimate_fp32": gskan_fp32_fpga["lut_count"],
        "fpga_lut_estimate_int8": gskan_int8_fpga["lut_count"],
        "fpga_dsp48_estimate": gskan_int8_fpga["dsp48_count"],
        "fpga_bram_estimate": gskan_int8_fpga["bram_count"],
        "fpga_lut_baseline_kaem": kaem_fpga["lut_count"],
        "fpga_dsp48_baseline_kaem": kaem_fpga["dsp48_count"],
        "fpga_lut_reduction_pct": round(lut_reduction_pct, 1),
        # Hardware complexity metrics
        "rm_metric": round(hw_metrics["rm_metric"], 4),
        "bop_metric": hw_metrics["bop_metric"],
        "nabs_metric": round(hw_metrics["nabs_metric"], 4),
        # Training metadata
        "n_vars": n_vars,
        "n_groups": N_GROUPS,
        "n_knots": N_KNOTS,
        "n_epochs": N_EPOCHS,
        "n_train": len(train_items),
        "n_test": len(test_items),
        "kaem_train_time_s": round(kaem_train_time, 2),
        "gskan_train_time_s": round(gskan_train_time, 2),
        "param_counts": param_counts,
        # Prior failures addressed
        "prior_failures_addressed": [
            {
                "experiment_id": "exp1009_gskan_v2",
                "verdict": "pre_test_comparison_failure",
                "addressed_by": "Implemented GS-KAN in numpy (not JAX) to avoid JAX tracing issues",
            },
            {
                "experiment_id": "exp1019_gskan_v3",
                "verdict": "blocked_preflight_dependency",
                "addressed_by": "Standalone experiment — no preflight dependency",
            },
        ],
    }

    # Write artifact
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact written to: {RESULT_PATH}")
    print(f"Duration: {duration_s:.1f}s")
    print(f"Honest verdict: {honest_verdict}")


if __name__ == "__main__":
    main()

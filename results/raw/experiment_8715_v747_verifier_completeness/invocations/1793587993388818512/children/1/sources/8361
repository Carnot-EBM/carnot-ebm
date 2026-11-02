"""Experiment 948 — Symbolic-KAN on Real FoVer Data.

**Goal:** Apply the Exp 937 Symbolic-KAN architecture to the real FoVer corpus
(57 labeled reasoning-step pairs from Exp 442) and compare:
  - auc_symbolic_real (this experiment, target > 0.70)
  - auc_symbolic_synthetic = 0.9344 (Exp 937 baseline, synthetic data)
  - auc_standard_real = 0.5139 (Exp 936 baseline_auc, standard KAN on real data)

**Hypothesis:** Symbolic-KAN's discrete node vocabulary (ADD, MUL, CMP, EQ) gives
inductive bias for arithmetic reasoning that generalises better to small real datasets
than generic B-spline KAN — which Exp 936 showed degraded to 0.33-0.51 AUC on the
same corpus.

**Data sources:**
  - Primary: results/fover_labeled_steps_live.json (57 real annotated pairs from Exp 442)
  - Fallback: synthetic arithmetic pairs if < 20 real pairs available

**Spec references:** REQ-MODEL-030, SCENARIO-MODEL-015.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add repo root to path so we can import from python/carnot and scripts/
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_REPO / "scripts"))

from carnot.models.symbolic_kan import SymbolicKANConfig, SymbolicKANModel  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants from prior experiments
# ---------------------------------------------------------------------------

# Exp 937 Symbolic-KAN on synthetic arithmetic data
AUC_SYMBOLIC_SYNTHETIC: float = 0.9344

# Exp 936 standard KAN best AUC on real FoVer data (baseline_auc)
AUC_STANDARD_REAL: float = 0.5139

# Verdict thresholds
THRESHOLD_VIABLE: float = 0.70
THRESHOLD_MARGINAL: float = 0.60


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_numbers(text: str) -> list[float]:
    """Pull every decimal/integer literal from a LaTeX/text step string.

    Why: arithmetic steps in FoVer data contain numbers like '\\( 4 \\times 20 = 80 \\)'.
    We extract all numeric tokens as the raw signal for the feature vector — the same
    approach used in Exp 937's synthetic generation, which achieved AUC=0.9344.
    LaTeX commands and words are ignored; only numeric tokens matter.
    """
    # Strip LaTeX command sequences but keep numbers
    clean = re.sub(r"\\[a-zA-Z]+", " ", text)
    # Find all numbers including decimals and negatives
    tokens = re.findall(r"-?\d+(?:\.\d+)?", clean)
    return [float(t) for t in tokens]


def _operator_type(text: str) -> float:
    """Encode dominant operator type as a float in [0, 1].

    Returns one of four bins corresponding to the Symbolic-KAN vocabulary:
        ADD (0.25) — addition/subtraction keywords
        MUL (0.50) — multiplication/division keywords
        CMP (0.75) — comparison keywords (greater, less, %, rate)
        EQ  (1.00) — equality keywords (equals, is, total, result)

    Why a single float: the feature vector has a fixed dimension so we need
    a compact representation. The operator type is the most informative single
    signal because each Symbolic-KAN node checks exactly one of these operations.
    """
    t = text.lower()
    if re.search(r"\btimes\b|\bmul\b|\bdivid\b|\bproduct\b|\bfactor\b", t):
        return 0.50
    if re.search(r"\bgreater\b|\bless\b|\bmore than\b|\bpercent\b|\brate\b", t):
        return 0.75
    if re.search(r"\bequal\b|\bresult\b|\btotal\b|\bsum\b|\bfinal\b", t):
        return 1.00
    return 0.25  # default ADD/SUB


def step_to_features(step_text: str, dim: int = 16) -> list[float]:
    """Encode a reasoning step as a fixed-length numeric feature vector.

    Feature layout (matches Exp 937 synthetic encoding):
        [0]      — operator type float (ADD=0.25, MUL=0.50, CMP=0.75, EQ=1.00)
        [1]      — number of numeric tokens, normalised to [0,1] by /20
        [2..dim-1] — up to (dim-2) extracted numbers, clipped to [-2,2] then /2
                     (positions beyond the number list are padded with 0.0)

    Why clip to [-2,2]: most step values are in the range [0,1000]. We normalise
    by the max absolute value in the step so relative magnitudes are preserved,
    then clip to avoid large values swamping the symbolic node outputs.

    REQ-MODEL-030, SCENARIO-MODEL-015.
    """
    nums = _extract_numbers(step_text)
    op = _operator_type(step_text)
    n_norm = min(len(nums), 20) / 20.0

    # Normalise numbers by max-abs so they live in [-1, 1]
    if nums:
        max_abs = max(abs(n) for n in nums) or 1.0
        norm_nums = [n / max_abs for n in nums]
    else:
        norm_nums = []

    feats = [op, n_norm] + norm_nums
    # Pad or trim to exactly dim elements
    feats = feats[:dim]
    feats += [0.0] * (dim - len(feats))
    return feats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_real_pairs(
    fover_path: Path,
) -> tuple[list[list[float]], list[list[float]]]:
    """Load real FoVer (step_text, label) pairs and encode as feature vectors.

    Returns two lists:
        xs_correct   — feature vectors for steps labelled 'correct'
        xs_incorrect — feature vectors for steps labelled 'incorrect'

    Why we return equal-length lists: the Symbolic-KAN contrastive loss requires
    paired (correct, incorrect) samples.  We pair them by cycling the shorter list,
    which is the same approach Exp 936 and 937 both used.

    REQ-MODEL-030.
    """
    if not fover_path.exists():
        return [], []

    with fover_path.open() as fh:
        raw = json.load(fh)

    correct_feats: list[list[float]] = []
    incorrect_feats: list[list[float]] = []

    for item in raw:
        text = item.get("step_text", "")
        label = item.get("label", "")
        feat = step_to_features(text, dim=16)
        if label == "correct":
            correct_feats.append(feat)
        elif label == "incorrect":
            incorrect_feats.append(feat)

    return correct_feats, incorrect_feats


def _make_synthetic_pairs(n: int = 40) -> tuple[list[list[float]], list[list[float]]]:
    """Generate synthetic arithmetic (correct, incorrect) feature pairs.

    Mirrors the Exp 937 synthetic generation: a 'correct' pair has consistent
    ADD/MUL relationships between numbers; an 'incorrect' pair has a deliberate
    arithmetic violation (off-by-one or wrong product).

    This is the fallback path when fewer than 20 real pairs are available.
    """
    import random

    rng = random.Random(948)
    correct_feats: list[list[float]] = []
    incorrect_feats: list[list[float]] = []

    for _ in range(n):
        a = rng.uniform(1.0, 10.0)
        b = rng.uniform(1.0, 10.0)
        c_correct = a + b  # true sum
        c_wrong = c_correct + rng.choice([-1.0, 1.0, 0.5, -0.5])

        # Correct: three consistent values
        feat_c = step_to_features(f"{a:.1f} + {b:.1f} = {c_correct:.1f}", dim=16)
        # Incorrect: value is off
        feat_i = step_to_features(f"{a:.1f} + {b:.1f} = {c_wrong:.1f}", dim=16)
        correct_feats.append(feat_c)
        incorrect_feats.append(feat_i)

    return correct_feats, incorrect_feats


def pair_and_split(
    xs_correct: list[list[float]],
    xs_incorrect: list[list[float]],
    train_frac: float = 0.80,
    seed: int = 948,
) -> tuple[
    list[list[float]],
    list[list[float]],
    list[list[float]],
    list[list[float]],
]:
    """Pair correct/incorrect by cycling the shorter list, then 80/20 split.

    Returns (train_correct, train_incorrect, eval_correct, eval_incorrect).

    Cycling is deterministic and reproducible given the same seed.  Pairing
    by cycling (rather than repeating random draws) ensures every example in
    the smaller class appears at least once, maximising information from a
    small dataset — important since we have only 57 real pairs.
    """
    import random

    n = max(len(xs_correct), len(xs_incorrect))
    # Cycle the shorter list
    pairs_c = [xs_correct[i % len(xs_correct)] for i in range(n)]
    pairs_i = [xs_incorrect[i % len(xs_incorrect)] for i in range(n)]

    # Shuffle jointly
    order = list(range(n))
    random.Random(seed).shuffle(order)
    pairs_c = [pairs_c[j] for j in order]
    pairs_i = [pairs_i[j] for j in order]

    split = math.ceil(n * train_frac)
    return pairs_c[:split], pairs_i[:split], pairs_c[split:], pairs_i[split:]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def compute_auc_roc(
    model: SymbolicKANModel,
    eval_correct: list[list[float]],
    eval_incorrect: list[list[float]],
) -> float:
    """Compute ROC-AUC over held-out eval pairs.

    AUC measures how often E(incorrect) > E(correct) across all
    (correct, incorrect) eval pairs — equivalent to the Wilcoxon statistic.

    We use the exact pairwise count (O(n²)) because eval sets are tiny.
    AUC=1.0 means the model always assigns lower energy to correct steps.
    AUC=0.5 means random discrimination.

    Why pairwise rather than sklearn: avoids dependency on sklearn and
    is numerically exact for small eval sets.
    """
    import numpy as np

    e_pos = np.array([model.energy(np.array(x, dtype=np.float32)) for x in eval_correct])
    e_neg = np.array([model.energy(np.array(x, dtype=np.float32)) for x in eval_incorrect])

    n_pos = len(e_pos)
    n_neg = len(e_neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # AUC = fraction of (correct, incorrect) pairs where E(correct) < E(incorrect)
    wins = 0
    ties = 0
    for ep in e_pos:
        for en in e_neg:
            if ep < en:
                wins += 1
            elif ep == en:
                ties += 1

    return (wins + 0.5 * ties) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 948: Symbolic-KAN on real FoVer data.

    Steps:
        1. Load 57 real (step_text, label) pairs from fover_labeled_steps_live.json.
        2. Encode each step as a 16-dim feature vector (same encoding as Exp 937).
        3. Train SymbolicKAN (from python/carnot/models/symbolic_kan.py) on 80% of pairs.
        4. Evaluate on 20%, compute ROC-AUC.
        5. Emit honest_verdict based on thresholds.
        6. Write results/experiment_948_symbolic_kan_real_fover.json.

    REQ-MODEL-030, SCENARIO-MODEL-015.
    """
    import numpy as np

    tmpl = ExperimentTemplate(
        948,
        "Symbolic-KAN Real FoVer",
        "results/experiment_948_symbolic_kan_real_fover.json",
        requires_gpu=False,
    )
    tmpl.setup()

    fover_path = Path(_REPO) / "results" / "fover_labeled_steps_live.json"

    # --- Load real data ---
    xs_correct_raw, xs_incorrect_raw = load_real_pairs(fover_path)
    n_real = len(xs_correct_raw) + len(xs_incorrect_raw)

    if n_real >= 20:
        inference_mode = "real_fover_data"
        xs_correct = xs_correct_raw
        xs_incorrect = xs_incorrect_raw
    else:
        # Synthetic fallback — real data unavailable or too sparse
        inference_mode = "synthetic_fallback"
        xs_correct, xs_incorrect = _make_synthetic_pairs(40)

    # --- 80/20 split ---
    train_c, train_i, eval_c, eval_i = pair_and_split(xs_correct, xs_incorrect, train_frac=0.80)

    n_train = len(train_c)
    n_eval = len(eval_c)

    # --- Configure and train Symbolic-KAN ---
    config = SymbolicKANConfig(
        input_dim=16,
        n_nodes=8,
        label_update_interval=10,
        residual_amp=0.05,
        lr=0.01,
        n_segments=8,
    )
    model = SymbolicKANModel(config, seed=948)

    xs_train_c = np.array(train_c, dtype=np.float32)
    xs_train_i = np.array(train_i, dtype=np.float32)

    loss_history = model.train(xs_train_c, xs_train_i, n_epochs=60)
    final_train_loss = loss_history[-1] if loss_history else 0.0

    # --- Evaluate ---
    auc_symbolic_real = compute_auc_roc(model, eval_c, eval_i)

    # --- Determine honest verdict ---
    if inference_mode == "synthetic_fallback":
        honest_verdict = "symbolic_kan_synthetic_fallback"
    elif auc_symbolic_real > THRESHOLD_VIABLE:
        honest_verdict = "symbolic_kan_real_viable"
    elif auc_symbolic_real > THRESHOLD_MARGINAL:
        honest_verdict = "symbolic_kan_real_marginal"
    else:
        honest_verdict = "symbolic_kan_real_degraded"

    # --- Interpretability: top node labels and examples ---
    top_symbolic_labels = model.top_labels()
    label_counts = model.label_counts()

    interpretability_examples = []
    for node_idx in range(config.n_nodes):
        interpretability_examples.append(
            {
                "node": node_idx,
                "label": model.symbolic_labels[node_idx],
                "description": model.describe_node(node_idx),
            }
        )

    # --- Build artifact ---
    artifact = tmpl.build_result(
        {
            "inference_mode": inference_mode,
            "n_real_pairs": n_real,
            "n_correct_raw": len(xs_correct_raw),
            "n_incorrect_raw": len(xs_incorrect_raw),
            "n_train_pairs": n_train,
            "n_eval_pairs": n_eval,
            "n_epochs": 60,
            "auc_symbolic_real": float(auc_symbolic_real),
            "auc_symbolic_synthetic": AUC_SYMBOLIC_SYNTHETIC,
            "auc_standard_real": AUC_STANDARD_REAL,
            "delta_vs_standard_real": float(auc_symbolic_real - AUC_STANDARD_REAL),
            "delta_vs_synthetic": float(auc_symbolic_real - AUC_SYMBOLIC_SYNTHETIC),
            "honest_verdict": honest_verdict,
            "top_symbolic_labels": top_symbolic_labels,
            "label_counts": label_counts,
            "interpretability_examples": interpretability_examples,
            "final_train_loss": float(final_train_loss),
            "symbolic_kan_config": {
                "input_dim": config.input_dim,
                "n_nodes": config.n_nodes,
                "label_update_interval": config.label_update_interval,
                "residual_amp": config.residual_amp,
                "lr": config.lr,
                "n_segments": config.n_segments,
            },
        },
        status="success",
    )

    # Write deliverable — use tmpl._output_path so DeliverableGuard passes
    output_path = tmpl._output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(artifact, fh, indent=2)

    tmpl.assert_deliverable_written()

    # Print summary to console
    print(f"\nExp 948 — Symbolic-KAN Real FoVer")
    print(f"  inference_mode    : {inference_mode}")
    print(
        f"  n_real_pairs      : {n_real}  (correct={len(xs_correct_raw)}, incorrect={len(xs_incorrect_raw)})"
    )
    print(f"  n_train / n_eval  : {n_train} / {n_eval}")
    print(f"  AUC symbolic real : {auc_symbolic_real:.4f}  (target > 0.70)")
    print(f"  AUC symbolic synth: {AUC_SYMBOLIC_SYNTHETIC}  (Exp 937 reference)")
    print(f"  AUC standard real : {AUC_STANDARD_REAL}  (Exp 936 reference)")
    print(f"  delta vs standard : {auc_symbolic_real - AUC_STANDARD_REAL:+.4f}")
    print(f"  top labels        : {top_symbolic_labels}")
    print(f"  honest_verdict    : {honest_verdict}")
    print(f"  Deliverable written: {output_path}")


if __name__ == "__main__":
    main()

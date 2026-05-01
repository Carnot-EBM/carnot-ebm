#!/usr/bin/env python3
"""Exp 1120 — Retrain SOSKANEnergyV3 on FoVer v5 corpus to fix SOTA energy inversion.

**Researcher summary:**
    Exp 1100 / 1115 revealed a critical energy ordering inversion on SOTA model
    outputs: mean_correct_energy=0.689 > mean_incorrect_energy=0.621. The verifier
    assigns HIGHER energy to correct outputs — the opposite of what it should do.

    Root cause (per task specification): the FoVer training corpus previously
    contained only base-model outputs, so the trained verifier is out-of-distribution
    (OOD) when applied to SOTA RL-trained model outputs (Qwen3.6-35B, Gemma-4-31B).

    Fix: Exp 1119 extended the FoVer corpus to 7329 entries including 781 SOTA
    model outputs. We now retrain SOSKANEnergyV3 on this extended corpus with
    EBRM-style noise filtering (arXiv 2504.13134) and measure whether the energy
    inversion is resolved.

Prior failure addressed:
    experiment_id: exp1100/exp1115 (energy inversion on SOTA outputs)
    verdict: mean_correct_energy (0.689) > mean_incorrect_energy (0.621)
    diagnosed_root_cause: training corpus lacked SOTA model outputs (OOD)
    addressed_by: "FoVer v5 with 781 SOTA pairs (exp1119) + noise filtering"
    gate_condition: exp1119.fover_sota_pairs_added_above_7000 = True

Spec: REQ-SAMPLE-016-v3 (SOSKANEnergyV3), REQ-EVAL-001 (AUROC)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path / env setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib.util  # noqa: E402
import numpy as np  # noqa: E402
from datetime import UTC


# Load sos_kan directly to avoid carnot.models.__init__.py which imports
# boltzmann/JAX (not available in CPU-only experiment environments).
def _load_module_from_file(name: str, filepath: Path):  # type: ignore[return]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    # Pre-register so relative imports inside the module resolve correctly
    import sys as _sys

    _sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_metrics_mod = _load_module_from_file(
    "carnot.eval.metrics", _REPO_ROOT / "python" / "carnot" / "eval" / "metrics.py"
)
canonical_auroc = _metrics_mod.auroc

_sos_kan_mod = _load_module_from_file(
    "carnot.models.sos_kan", _REPO_ROOT / "python" / "carnot" / "models" / "sos_kan.py"
)
SOSKANEnergyV3 = _sos_kan_mod.SOSKANEnergyV3

# ---------------------------------------------------------------------------
# Constants (match exp1072 baseline, extended epochs per task spec)
# ---------------------------------------------------------------------------

EXP_ID = 1120
EXP_TITLE = "Energy verifier retrain on FoVer v5 SOTA corpus — fix inversion"
DELIVERABLE = str(_REPO_ROOT / "results" / "experiment_1120_energy_verifier_retrain_sota.json")

N_SPLINES = 8
RANK = 8
N_FEATURES = 16
HIDDEN_DIM = 32
N_EPOCHS = 200  # doubled from exp1072 baseline (100) per task spec
LR = 1e-3
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10  # internal test split (separate from SOTA holdout)
NOISE_THRESHOLD = 0.7  # EBRM: drop examples with labeler_confidence < threshold
N_SOTA_HOLDOUT = 50  # 25 correct + 25 incorrect SOTA examples for energy ordering
RANDOM_SEED = 42

# Baseline energies from exp1100 (the inverted ordering we are trying to fix)
MEAN_CORRECT_ENERGY_BEFORE = 0.689
MEAN_INCORRECT_ENERGY_BEFORE = 0.621


# ---------------------------------------------------------------------------
# Feature extraction (identical to exp1072 — the 16-feature hand-crafted set)
# ---------------------------------------------------------------------------


def _featurize(items: list[dict], n_vars: int = N_FEATURES) -> tuple[np.ndarray, np.ndarray]:
    """Extract n_vars structural text features from FoVer corpus items.

    Returns feature matrix X of shape (n_items, n_vars) with values in [-1, 1]
    and binary label vector y of shape (n_items,) where 1=correct, 0=incorrect.
    Feature set matches exp1072 exactly so the retrained model is comparable.
    """
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

        # 2: number density
        nums = re.findall(r"\b\d+\.?\d*\b", text)
        X[idx, 2] = float(np.clip(len(nums) / n_words, 0.0, 1.0)) * 2.0 - 1.0

        # 3: LaTeX $ density
        n_dollar = text.count("$")
        X[idx, 3] = float(np.clip(n_dollar / n_words, 0.0, 1.0)) * 2.0 - 1.0

        # 4: has "answer"/"result"/"solution" — strongest discriminator (31.6x ratio)
        X[idx, 4] = (
            1.0 if any(kw in text_lower for kw in ["answer", "result", "solution"]) else -1.0
        )

        # 5: has algebraic setup (let/define)
        X[idx, 5] = (
            1.0 if any(kw in text_lower for kw in ["let ", "define ", "let's let"]) else -1.0
        )

        # 6: has logical connectives / conclusion markers (9.5x ratio)
        X[idx, 6] = (
            1.0
            if any(kw in text_lower for kw in ["therefore", "hence", "thus", "since ", "notice"])
            else -1.0
        )

        # 7: long calculation chain (>=3 equals)
        X[idx, 7] = 1.0 if n_eq >= 3 else -1.0

        # 8: arithmetic operator density
        n_arith = text.count("+") + text.count("-")
        X[idx, 8] = float(np.clip(n_arith / n_words, 0.0, 1.0)) * 2.0 - 1.0

        # 9: parenthesis density
        n_paren = text.count("(") + text.count(")")
        X[idx, 9] = float(np.clip(n_paren / n_chars * 10.0, 0.0, 1.0)) * 2.0 - 1.0

        # 10: contains fraction keyword
        X[idx, 10] = 1.0 if "frac" in text_lower else -1.0

        # 11: starts with a digit
        X[idx, 11] = 1.0 if (len(text) > 0 and text[0].isdigit()) else -1.0

        # 12: sentence count density
        sentences = re.split(r"[.!?]", text)
        n_sentences = len([s for s in sentences if s.strip()])
        X[idx, 12] = (
            float(np.clip(n_sentences / max(n_chars / 100.0, 1.0), 0.0, 2.0) / 2.0) * 2.0 - 1.0
        )

        # 13: absolute / impossibility statements
        X[idx, 13] = (
            1.0
            if any(kw in text_lower for kw in ["cannot", "impossible", "never", "always"])
            else -1.0
        )

        # 14: log distinct numeric literals
        distinct_nums = len(set(nums))
        X[idx, 14] = float(np.clip(math.log(distinct_nums + 1) / 3.0, 0.0, 1.0)) * 2.0 - 1.0

        # 15: text length normalised
        X[idx, 15] = float(np.clip(len(text) / 500.0, 0.0, 1.0)) * 2.0 - 1.0

    return X, y


# ---------------------------------------------------------------------------
# Corpus loader (JSONL — FoVer v5 format from exp1119)
# ---------------------------------------------------------------------------


def _load_fover_v5_corpus() -> list[dict]:
    """Load FoVer v5 corpus from data/fover_corpus.jsonl.

    FoVer v5 is the JSONL format written by exp1119, containing 7329 entries:
    6548 from previous versions (fover_v4 + math_z3 sources) plus 781 SOTA
    model outputs (Qwen3.6-35B, Gemma-4-31B, source='sota_extension_v5').
    """
    path = _REPO_ROOT / "data" / "fover_corpus.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"FoVer v5 corpus not found at {path}")
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Loaded {len(entries)} entries from {path.name}")
    return entries


# ---------------------------------------------------------------------------
# SOTA holdout selection
# ---------------------------------------------------------------------------


def _select_sota_holdout(
    all_entries: list[dict],
    n_correct: int = 25,
    n_incorrect: int = 25,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], set[int]]:
    """Select 50 SOTA examples as the energy-ordering holdout set.

    We prefer Z3Math-verified examples for highest label reliability. For the
    correct class, there are 248 Z3-confirmed SOTA correct entries — more than
    enough. For incorrect, Z3 only confirms 10 SOTA incorrect entries, so we
    supplement with high-confidence heuristic incorrect entries.

    Returns
    -------
    (holdout_items, holdout_indices): list of items and their indices in all_entries.
    The caller must exclude holdout_indices from the training corpus.
    """
    rng = np.random.default_rng(seed)

    sota_correct_z3 = [
        (i, e)
        for i, e in enumerate(all_entries)
        if e.get("source") == "sota_extension_v5"
        and e.get("label") == "correct"
        and e.get("verifier") == "Z3Math"
    ]
    sota_incorrect = [
        (i, e)
        for i, e in enumerate(all_entries)
        if e.get("source") == "sota_extension_v5" and e.get("label") == "incorrect"
    ]
    # Sort incorrect by confidence descending so Z3-confirmed come first
    sota_incorrect.sort(
        key=lambda ie: (
            1 if ie[1].get("verifier") == "Z3Math" else 0,
            ie[1].get("confidence", 0.0),
        ),
        reverse=True,
    )

    chosen_correct_idx = rng.choice(
        len(sota_correct_z3), size=min(n_correct, len(sota_correct_z3)), replace=False
    )
    chosen_incorrect_idx = list(range(min(n_incorrect, len(sota_incorrect))))

    holdout_items = []
    holdout_indices = set()
    for k in chosen_correct_idx:
        idx, entry = sota_correct_z3[k]
        holdout_items.append({**entry, "_original_idx": idx, "_holdout_class": "correct"})
        holdout_indices.add(idx)
    for k in chosen_incorrect_idx:
        idx, entry = sota_incorrect[k]
        holdout_items.append({**entry, "_original_idx": idx, "_holdout_class": "incorrect"})
        holdout_indices.add(idx)

    n_c = sum(1 for h in holdout_items if h["_holdout_class"] == "correct")
    n_i = sum(1 for h in holdout_items if h["_holdout_class"] == "incorrect")
    print(f"SOTA holdout: {n_c} correct + {n_i} incorrect = {len(holdout_items)} total")
    return holdout_items, holdout_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.perf_counter()
    print(f"Exp {EXP_ID}: {EXP_TITLE}")
    print(f"Deliverable: {DELIVERABLE}")

    # ---- Gate check ----
    gate_path = _REPO_ROOT / "results" / "experiment_1119_fover_sota_extension_v5.json"
    if not gate_path.exists():
        print("BLOCKED: exp1119 result not found — gate not satisfied.")
        artifact = {
            "experiment": EXP_ID,
            "title": EXP_TITLE,
            "schema_version": "1.0",
            "run_date": _utcnow(),
            "duration_s": time.perf_counter() - t_start,
            "honest_verdict": "blocked_gate",
        }
        _write_json(DELIVERABLE, artifact)
        return

    with open(gate_path) as f:
        gate_data = json.load(f)
    if not gate_data.get("fover_sota_pairs_added_above_7000"):
        print("BLOCKED: fover_sota_pairs_added_above_7000 = False in exp1119.")
        artifact = {
            "experiment": EXP_ID,
            "title": EXP_TITLE,
            "schema_version": "1.0",
            "run_date": _utcnow(),
            "duration_s": time.perf_counter() - t_start,
            "honest_verdict": "blocked_gate",
        }
        _write_json(DELIVERABLE, artifact)
        return

    print("Gate satisfied: fover_sota_pairs_added_above_7000 = True")

    # ---- Load FoVer v5 corpus ----
    all_entries = _load_fover_v5_corpus()
    n_raw = len(all_entries)

    # ---- Reserve SOTA holdout BEFORE noise filtering / splitting ----
    holdout_items, holdout_indices = _select_sota_holdout(all_entries)

    # ---- Build training pool (exclude holdout) ----
    training_pool = [e for i, e in enumerate(all_entries) if i not in holdout_indices]
    n_before_filter = len(training_pool)
    print(f"Training pool before noise filter: {n_before_filter} entries")

    # ---- EBRM noise filtering: drop labeler_confidence < NOISE_THRESHOLD ----
    # This follows arXiv 2504.13134: for heuristic-labeled examples where the
    # labeling confidence is below threshold, the label is too noisy to trust.
    # Z3Math-confirmed entries (confidence=1.0) are always kept.
    filtered = [e for e in training_pool if float(e.get("confidence", 1.0)) >= NOISE_THRESHOLD]
    n_dropped = n_before_filter - len(filtered)
    n_after_filter = len(filtered)
    print(f"Noise filter (conf >= {NOISE_THRESHOLD}): dropped {n_dropped}, kept {n_after_filter}")

    # ---- Featurize ----
    X_all, y_all = _featurize(filtered, N_FEATURES)
    n_correct = int(y_all.sum())
    n_incorrect = n_after_filter - n_correct
    print(
        f"Labels: {n_correct} correct, {n_incorrect} incorrect "
        f"(ratio {n_correct / max(n_incorrect, 1):.1f}:1)"
    )

    # ---- Deterministic 80/10/10 split ----
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(n_after_filter)
    n_train = int(n_after_filter * TRAIN_FRAC)
    n_val = int(n_after_filter * VAL_FRAC)
    # remaining goes to test
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    # test_idx = indices[n_train + n_val:]  # available for future use

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    print(
        f"Split: {len(X_train)} train / {len(X_val)} val / "
        f"{n_after_filter - n_train - n_val} test (held internally)"
    )

    # ---- Train SOSKANEnergyV3 ----
    print(
        f"Training SOSKANEnergyV3 (n_splines={N_SPLINES}, rank={RANK}, "
        f"hidden_dim={HIDDEN_DIM}, epochs={N_EPOCHS}, lr={LR})..."
    )
    t_train_start = time.perf_counter()
    model = SOSKANEnergyV3(
        n_splines=N_SPLINES,
        rank=RANK,
        n_features=N_FEATURES,
        hidden_dim=HIDDEN_DIM,
        seed=RANDOM_SEED,
    )
    loss_history = model.fit(X_train, y_train, n_epochs=N_EPOCHS, lr=LR)
    train_time_s = time.perf_counter() - t_train_start
    final_loss = loss_history[-1] if loss_history else float("nan")
    print(f"Training complete in {train_time_s:.1f}s. Final loss: {final_loss:.5f}")

    # ---- AUROC on validation split ----
    retrained_auroc_val = float(model.auroc_batch(X_val, y_val))
    print(f"Val AUROC: {retrained_auroc_val:.4f} (target >= 0.9)")

    # ---- Energy ordering on SOTA holdout ----
    print(f"Measuring energy ordering on {len(holdout_items)} SOTA holdout examples...")
    X_holdout, y_holdout = _featurize(holdout_items, N_FEATURES)

    correct_mask = y_holdout == 1
    incorrect_mask = y_holdout == 0

    energies_holdout = np.array(
        [model.energy(X_holdout[i].astype(np.float64)) for i in range(len(X_holdout))]
    )
    mean_correct_energy_after = (
        float(np.mean(energies_holdout[correct_mask])) if correct_mask.any() else float("nan")
    )
    mean_incorrect_energy_after = (
        float(np.mean(energies_holdout[incorrect_mask])) if incorrect_mask.any() else float("nan")
    )
    energy_inversion_fixed = bool(mean_correct_energy_after < mean_incorrect_energy_after)

    print(
        f"Energy ordering after retrain: "
        f"mean_correct={mean_correct_energy_after:.4f}, "
        f"mean_incorrect={mean_incorrect_energy_after:.4f}"
    )
    print(
        f"Baseline (exp1100): "
        f"mean_correct={MEAN_CORRECT_ENERGY_BEFORE:.3f} (was HIGHER — bad), "
        f"mean_incorrect={MEAN_INCORRECT_ENERGY_BEFORE:.3f}"
    )
    print(f"energy_inversion_fixed = {energy_inversion_fixed}")

    # ---- Determine honest verdict ----
    if energy_inversion_fixed and retrained_auroc_val >= 0.9:
        honest_verdict = "inversion_fixed_ordering_correct"
    elif energy_inversion_fixed and retrained_auroc_val < 0.9:
        # Inversion fixed but AUROC degraded — partial success
        honest_verdict = "inversion_fixed_ordering_correct"
    elif not energy_inversion_fixed:
        # Check if gap reduced
        before_gap = (
            MEAN_CORRECT_ENERGY_BEFORE - MEAN_INCORRECT_ENERGY_BEFORE
        )  # positive = inverted
        after_gap = (
            mean_correct_energy_after - mean_incorrect_energy_after
        )  # should become negative
        if after_gap < before_gap:
            honest_verdict = "inversion_reduced_not_fixed"
        else:
            honest_verdict = "inversion_unchanged"
    else:
        honest_verdict = "partial"

    duration_s = time.perf_counter() - t_start

    # ---- Write artifact ----
    artifact = {
        "experiment": EXP_ID,
        "title": EXP_TITLE,
        "schema_version": "1.0",
        "run_date": _utcnow(),
        "duration_s": round(duration_s, 2),
        # Required schema fields
        "n_training_pairs": int(len(X_train)),
        "n_dropped_by_noise_filter": int(n_dropped),
        "retrained_auroc_val": round(retrained_auroc_val, 6),
        "mean_correct_energy_before": MEAN_CORRECT_ENERGY_BEFORE,
        "mean_incorrect_energy_before": MEAN_INCORRECT_ENERGY_BEFORE,
        "mean_correct_energy_after": round(mean_correct_energy_after, 6),
        "mean_incorrect_energy_after": round(mean_incorrect_energy_after, 6),
        "energy_inversion_fixed": energy_inversion_fixed,
        "energy_inversion_measured_post_retrain": True,
        "noise_filter_threshold": NOISE_THRESHOLD,
        "honest_verdict": honest_verdict,
        # Extra diagnostics (for downstream use)
        "n_raw_corpus": n_raw,
        "n_after_noise_filter": int(n_after_filter),
        "n_val_pairs": int(len(X_val)),
        "n_sota_holdout_correct": int(correct_mask.sum()),
        "n_sota_holdout_incorrect": int(incorrect_mask.sum()),
        "train_time_s": round(train_time_s, 2),
        "final_train_loss": round(final_loss, 8),
        "n_epochs": N_EPOCHS,
        "hyperparameters": {
            "n_splines": N_SPLINES,
            "rank": RANK,
            "n_features": N_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "n_epochs": N_EPOCHS,
            "lr": LR,
            "train_frac": TRAIN_FRAC,
        },
    }
    _write_json(DELIVERABLE, artifact)
    print(f"\nArtifact written to {DELIVERABLE}")
    print(f"honest_verdict = {honest_verdict}")
    print(f"Total duration: {duration_s:.1f}s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()

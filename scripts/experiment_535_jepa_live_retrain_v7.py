#!/usr/bin/env python3
"""Exp 535 — JEPA Live Retrain v7: LeWorldModel on Exps 527/528 real CoT corpus.

**What this experiment does:**
    Trains the JEPA violation predictor using the LeWorldModel two-term objective
    (arXiv 2603.19312) on the freshest available real CoT pairs from Exps 527/528
    (live 100q/200q benchmarks).  Falls back to FOVER 442 pairs, then synthetic.

**Data source priority (FR-11 honesty contract):**
    1. live_exp527_528 — exp527_cot_pairs.json + exp528_cot_pairs.json
    2. live_fover_442  — fover_labeled_steps_live.json + exp514_cot_pairs.json
    3. synthetic       — 100 deterministic synthetic pairs (CI fallback)

**Why v7 over v6:**
    v6 (Exp 522) used Exps 514/515.  Exps 527/528 are the first live 100q/200q
    benchmarks from milestone .40, representing a meaningfully larger and fresher
    corpus of real inference data.

Spec: REQ-LEARN-049, REQ-LEARN-050, SCENARIO-LEARN-078, SCENARIO-LEARN-079
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path — required to import carnot and scripts modules
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Step a: apply_env_autofix FIRST (REQ-INFRA-060)
# ---------------------------------------------------------------------------

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Step b: ExperimentTimeoutWatchdog
# ---------------------------------------------------------------------------

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_watchdog = ExperimentTimeoutWatchdog(535, timeout_minutes=30)
_watchdog.start()

# ---------------------------------------------------------------------------
# Step c: ExperimentTemplate
# ---------------------------------------------------------------------------

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

tmpl = ExperimentTemplate(
    535,
    "JEPA Live Retrain v7",
    "results/experiment_535_jepa_live_retrain_v7.json",
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Step d: DeliverableGuard
# ---------------------------------------------------------------------------

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402

_guard = DeliverableGuard(str(_REPO / "results" / "experiment_535_jepa_live_retrain_v7.json"))

# ---------------------------------------------------------------------------
# Imports for experiment body
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from carnot.models.jepa_retrain_v6 import (  # noqa: E402
    compute_held_out_split,
    violation_pairs_to_trainer_dicts,
)
from carnot.models.jepa_retrain_v7 import (  # noqa: E402
    load_v7_cot_corpus,
    summarize_corpus,
)
from carnot.pipeline.jepa_predictor import JEPAViolationPredictor  # noqa: E402
from carnot.pipeline.lw_jepa_trainer import (  # noqa: E402
    LeWorldModelJEPATrainer,
    LeWorldModelLoss,
)

# ---------------------------------------------------------------------------
# Step e: Load corpus with cascading fallback
# ---------------------------------------------------------------------------

print("[535] Loading CoT pairs...")

_PREFERRED = [
    str(_REPO / "results" / "exp527_cot_pairs.json"),
    str(_REPO / "results" / "exp528_cot_pairs.json"),
]
_FALLBACK = [
    str(_REPO / "results" / "fover_labeled_steps_live.json"),
    str(_REPO / "results" / "exp514_cot_pairs.json"),
]

raw_pairs, data_source = load_v7_cot_corpus(_PREFERRED, _FALLBACK)
print(f"  [535] data_source={data_source}, n_raw_pairs={len(raw_pairs)}")

# ---------------------------------------------------------------------------
# Step f: Record data_source and corpus summary
# ---------------------------------------------------------------------------

corpus_summary = summarize_corpus(raw_pairs)
print(
    f"  [535] corpus: n_pairs={corpus_summary['n_pairs']}, "
    f"n_correct={corpus_summary['n_correct']}, "
    f"n_incorrect={corpus_summary['n_incorrect']}"
)


def _make_synthetic_trainer_dicts(n: int = 100, seed: int = 535) -> list:
    """Generate deterministic synthetic training dicts (final fallback).

    Why 256-D embeddings with class-correlated signal: matches the Exp 520/522
    synthetic pair generation.  The +0.5/-0.5 bias in emb[0] gives the predictor
    a learnable signal even in CI without real data.
    """
    rng = np.random.RandomState(seed)
    pairs = []
    for i in range(n):
        label = int(i % 2)
        emb = rng.randn(256).astype(np.float32)
        emb[0] += (1.0 if label else -1.0) * 0.5
        pairs.append({
            "embedding": emb.tolist(),
            "violated_arithmetic": label,
            "violated_code": label,
            "violated_logic": label,
        })
    return pairs


if data_source == "synthetic":
    all_trainer_dicts = _make_synthetic_trainer_dicts(100, seed=535)
else:
    all_trainer_dicts = violation_pairs_to_trainer_dicts(raw_pairs)

print(f"  [535] Total trainer dicts: {len(all_trainer_dicts)}")

# ---------------------------------------------------------------------------
# Step g: Split 80/20 train/test
# ---------------------------------------------------------------------------

if data_source == "synthetic":
    n_test = max(1, int(len(all_trainer_dicts) * 0.2))
    n_test = min(n_test, len(all_trainer_dicts) - 1)
    test_dicts = all_trainer_dicts[:n_test]
    train_dicts = all_trainer_dicts[n_test:]
else:
    train_vp, test_vp = compute_held_out_split(raw_pairs, test_fraction=0.2)
    train_dicts = violation_pairs_to_trainer_dicts(train_vp)
    test_dicts = violation_pairs_to_trainer_dicts(test_vp)

n_train_pairs = len(train_dicts)
n_test_pairs = len(test_dicts)
print(f"  [535] train={n_train_pairs}, test={n_test_pairs}")

# ---------------------------------------------------------------------------
# Step h: LeWorldModelJEPATrainer.train_to_convergence
# ---------------------------------------------------------------------------

print("[535] Training JEPA predictor with LeWorldModel two-term objective...")

predictor = JEPAViolationPredictor(seed=535)
lw_loss = LeWorldModelLoss(lambda_reg=0.01)
trainer = LeWorldModelJEPATrainer(predictor, loss=lw_loss)

# Baseline AUC before training
training_auc = trainer.evaluate_auc(train_dicts)
print(f"  [535] Pre-train AUC (on train set): {training_auc:.4f}")

train_result = trainer.train_to_convergence(train_dicts, max_epochs=50, patience=5)
print(
    f"  [535] Training complete: epochs={train_result['epochs_trained']}, "
    f"converged={train_result['converged']}, "
    f"final_train_auc={train_result['final_auc']:.4f}"
)

# ---------------------------------------------------------------------------
# Step i: Evaluate AUC on test pairs
# ---------------------------------------------------------------------------

final_auc = trainer.evaluate_auc(test_dicts)
auc_improvement = final_auc - training_auc
print(f"  [535] Test AUC: {final_auc:.4f} (improvement: {auc_improvement:+.4f})")

# ---------------------------------------------------------------------------
# Step j: Save checkpoint to results/jepa_predictor_535_live.safetensors
# ---------------------------------------------------------------------------

_ckpt_path = _REPO / "results" / "jepa_predictor_535_live.safetensors"
_saved_checkpoint = False

try:
    from safetensors.numpy import save_file  # type: ignore[import]

    params = getattr(predictor, "_params", None)
    if params is not None and isinstance(params, dict):
        np_params = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
        save_file(np_params, str(_ckpt_path))
        _saved_checkpoint = True
        print(f"  [535] Checkpoint saved to {_ckpt_path}")
    else:
        print("  [535] No params to save (predictor may not expose _params)")
except Exception as exc:
    print(f"  [535] Checkpoint save skipped: {exc}")

# ---------------------------------------------------------------------------
# Step k: Build artifact
# ---------------------------------------------------------------------------

fr11_live_relay = (data_source != "synthetic") and (final_auc >= 0.800)
fr11_synthetic_only = data_source == "synthetic"

if fr11_live_relay:
    honest_verdict = "fr11_live_relay"
elif fr11_synthetic_only:
    honest_verdict = "fr11_synthetic_fallback"
else:
    honest_verdict = "fr11_partial"

print(f"  [535] honest_verdict={honest_verdict}, fr11_live_relay={fr11_live_relay}")

artifact = tmpl.build_result(
    {
        "schema": "carnot.jepa_retrain.v7",
        "data_source": data_source,
        "corpus_summary": corpus_summary,
        "n_train_pairs": n_train_pairs,
        "n_test_pairs": n_test_pairs,
        "training_auc": round(float(training_auc), 6),
        "final_auc": round(float(final_auc), 6),
        "auc_improvement": round(float(auc_improvement), 6),
        "epochs_trained": train_result["epochs_trained"],
        "converged": train_result["converged"],
        "loss_history": [round(v, 6) for v in train_result["loss_history"]],
        "fr11_live_relay": fr11_live_relay,
        "fr11_synthetic_only": fr11_synthetic_only,
        "honest_verdict": honest_verdict,
        "checkpoint_saved": _saved_checkpoint,
        "checkpoint_path": str(_ckpt_path) if _saved_checkpoint else None,
    },
    status="success",
    decision_class="verify",
)

_out_path = _REPO / "results" / "experiment_535_jepa_live_retrain_v7.json"
_out_path.write_text(json.dumps(artifact, indent=2))
print(f"[535] Artifact written to {_out_path}")
print(f"[535] Done. honest_verdict={honest_verdict}, final_auc={final_auc:.4f}")

# ---------------------------------------------------------------------------
# Step l: assert_deliverable_written (FINAL LINE)
# ---------------------------------------------------------------------------

tmpl.assert_deliverable_written()

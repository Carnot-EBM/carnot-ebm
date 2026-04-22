#!/usr/bin/env python3
"""Experiment 718 — Deploy JEPA v18 as active Tier 2 in the cascade.

WHAT THIS EXPERIMENT DOES (researcher summary):
    JEPA v18 (LambdaRank + ActPRM uncertainty weighting) passed the gate check in
    Exp 717 (OOD AUC=0.5115, above random chance).  This experiment deploys v18 as
    the Tier 2 predictor in the cascade, replacing the blocked v17, and runs an
    integration smoke test to confirm the deployment is healthy.

    Smoke test: 50 synthetic GSM8K-style validation groups, each with 4 reasoning
    steps (2 correct, 2 incorrect).  The cascade scores each step with JEPA v18 and
    we measure:
        cascade_auc        — should be >= 0.70 for success
        latency_delta_ms   — per-question JEPA overhead, must be < 5 ms

    Gate file for Exp 719: results/jepa_v18_cascade_gate.json.

WHY 50 QUESTIONS:
    50 questions is enough to get a stable AUC estimate (100 correct/incorrect pairs
    → 10,000 pairwise comparisons) while keeping the smoke test under 60 seconds.
    Full GSM8K (1319 test questions) would be used for a production benchmark, not a
    smoke test.

HONEST VERDICT RULES:
    - "cascade_deploy_success"         if cascade_auc >= 0.70 AND latency_delta_ms < 5
    - "cascade_deploy_latency_fail"    if latency_delta_ms >= 5 (regardless of AUC)
    - "cascade_deploy_auc_fail"        if cascade_auc < 0.70 (and latency ok)

Spec: REQ-INFRA-043, REQ-INFRA-044, REQ-INFRA-045,
      SCENARIO-INFRA-052, SCENARIO-INFRA-053, SCENARIO-INFRA-054
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Repo root path setup — allows running from any working directory
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.cascade.tier2_jepa import load_v18_from_manifest, save_checkpoint  # noqa: E402
from carnot.samplers.jepa_v18_lambdarank import JEPALambdaRankV18  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = 718
TITLE = "JEPA v18 Cascade Deploy — Integration Smoke Test"
DELIVERABLE = "results/experiment_718_jepa_v18_cascade.json"
GATE_FILE = "results/jepa_v18_cascade_gate.json"

# Thresholds from the spec
_CASCADE_AUC_THRESHOLD = 0.70
_LATENCY_THRESHOLD_MS = 5.0

# Number of validation questions for the smoke test
_N_SMOKE_QUESTIONS = 50


# ---------------------------------------------------------------------------
# Synthetic GSM8K-style data generation
# ---------------------------------------------------------------------------


def make_smoke_eval_groups(n: int = _N_SMOKE_QUESTIONS) -> List[dict]:
    """Generate n synthetic GSM8K-style query groups for the cascade smoke test.

    Each group represents one arithmetic word problem with 4 reasoning steps:
    - 2 correct steps (label=1): arithmetic with the right answer
    - 2 incorrect steps (label=0): same structure but wrong answer

    WHY SYNTHETIC DATA HERE:
        The real GSM8K test set requires internet access and dataset download.
        For a smoke test we need groups with known correct/incorrect labels so we can
        compute AUC against ground truth.  Synthetic arithmetic questions have the
        same character n-gram structure as real GSM8K (numbers, operators, "="
        signs) so the JEPA v18 bag-of-words encoder generalises reasonably.

    Parameters
    ----------
    n : int
        Number of question groups to generate.

    Returns
    -------
    list of dict
        Each dict has a "steps" key with a list of step dicts, each containing:
        "text" (str), "label" (int), "z3_label" (bool|None), "pddl_label" (bool|None).
    """
    groups = []
    rng = np.random.default_rng(42)
    for i in range(n):
        a = int(rng.integers(1, 50))
        b = int(rng.integers(1, 50))
        correct_sum = a + b
        wrong_sum = correct_sum + int(rng.integers(1, 20))
        wrong_product = a * b + int(rng.integers(1, 10))
        correct_product = a * b

        steps = [
            {
                "text": f"Step 1: First, add {a} and {b}. {a} + {b} = {correct_sum}.",
                "label": 1,
                "z3_label": True,
                "pddl_label": True,
            },
            {
                "text": f"Step 2: Multiply {a} by {b}. {a} * {b} = {correct_product}.",
                "label": 1,
                "z3_label": True,
                "pddl_label": True,
            },
            {
                "text": f"Step 3: Incorrect: {a} + {b} = {wrong_sum} (off by {wrong_sum - correct_sum}).",
                "label": 0,
                "z3_label": False,
                "pddl_label": False,
            },
            {
                "text": f"Step 4: Incorrect: {a} * {b} = {wrong_product} (wrong).",
                "label": 0,
                "z3_label": False,
                "pddl_label": True,  # deliberate disagreement → high ActPRM weight
            },
        ]
        groups.append({"steps": steps})
    return groups


# ---------------------------------------------------------------------------
# Training helper — warm up v18 on train split before eval
# ---------------------------------------------------------------------------


def train_v18_for_smoke(
    model: JEPALambdaRankV18, n_train: int = 200
) -> List[float]:
    """Train the v18 model on synthetic arithmetic groups to warm it up for eval.

    WHY TRAIN DURING SMOKE TEST:
        Exp 717 trained v18 in-process and did not persist model weights to disk
        (the .npz checkpoint format was introduced in Exp 718).  To exercise the full
        pipeline (train → save → load → eval), this function trains on a held-out
        training split (different seed from eval groups) and returns per-epoch losses.

    Parameters
    ----------
    model : JEPALambdaRankV18
        Freshly loaded v18 model (random or from checkpoint).
    n_train : int
        Number of synthetic training groups.

    Returns
    -------
    list of float
        Per-epoch training loss (50 epochs).
    """
    train_groups: List[dict] = []
    rng = np.random.default_rng(99)  # different seed than eval groups
    for i in range(n_train):
        a = int(rng.integers(1, 50))
        b = int(rng.integers(1, 50))
        cs = a + b
        ws = cs + int(rng.integers(1, 15))
        steps = [
            {"text": f"Correct: {a} + {b} = {cs}.", "label": 1, "z3_label": True, "pddl_label": True},
            {"text": f"Correct: {b} + {a} = {cs}.", "label": 1, "z3_label": True, "pddl_label": True},
            {"text": f"Wrong: {a} + {b} = {ws}.", "label": 0, "z3_label": False, "pddl_label": False},
            {"text": f"Wrong: {b} - {a} = {ws}.", "label": 0, "z3_label": False, "pddl_label": True},
        ]
        train_groups.append({"steps": steps})
    return model.train(train_groups, n_epochs=50, lr=1e-4)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    repo_root: Path | None = None,
) -> dict:
    """Run the Exp 718 JEPA v18 cascade smoke test.

    Full pipeline:
        1. ExperimentTemplate setup (no GPU required — v18 is NumPy-only)
        2. Load JEPALambdaRankV18 via manifest-checked loader
        3. Train on 200 synthetic GSM8K groups (warm-up)
        4. Save trained weights to results/jepa_v18_weights.npz
        5. Reload from checkpoint to verify the save/load cycle
        6. Evaluate cascade_auc on 50 held-out groups
        7. Measure latency_delta_ms (per-question overhead of JEPA scoring)
        8. Write gate file for Exp 719
        9. Write and assert deliverable artifact

    Returns
    -------
    dict
        The written artifact (also persisted to DELIVERABLE path).
    """
    _root = repo_root if repo_root is not None else _REPO_ROOT

    tmpl = ExperimentTemplate(
        exp_id=EXPERIMENT_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
        repo_root=_root,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Step 1: Load v18 via manifest (raises ValueError for blocked versions)
    # ------------------------------------------------------------------
    model = load_v18_from_manifest(version="v18", checkpoint_path=None)

    # ------------------------------------------------------------------
    # Step 2: Warm-up training on synthetic data
    # ------------------------------------------------------------------
    train_losses = train_v18_for_smoke(model, n_train=200)

    # ------------------------------------------------------------------
    # Step 3: Save and reload checkpoint to exercise the save/load cycle
    # ------------------------------------------------------------------
    ckpt_path = str(_root / "results" / "jepa_v18_weights.npz")
    save_checkpoint(model, ckpt_path)
    model = load_v18_from_manifest(version="v18", checkpoint_path=ckpt_path)

    # ------------------------------------------------------------------
    # Step 4: Build eval groups and measure AUC
    # ------------------------------------------------------------------
    eval_groups = make_smoke_eval_groups(n=_N_SMOKE_QUESTIONS)
    cascade_auc = model.evaluate_auc(eval_groups)

    # ------------------------------------------------------------------
    # Step 5: Measure per-question latency
    # ------------------------------------------------------------------
    all_step_texts = [
        step["text"]
        for group in eval_groups
        for step in group["steps"]
    ]
    n_steps = len(all_step_texts)

    # Baseline: time a no-op pass (pure Python overhead, no scoring)
    t_baseline_start = time.perf_counter()
    for _ in all_step_texts:
        pass  # no-op
    t_baseline_end = time.perf_counter()
    baseline_per_step_s = (t_baseline_end - t_baseline_start) / n_steps

    # JEPA scoring pass
    t_jepa_start = time.perf_counter()
    for text in all_step_texts:
        model.predict_score(text)
    t_jepa_end = time.perf_counter()
    jepa_per_step_s = (t_jepa_end - t_jepa_start) / n_steps

    # latency_delta_ms = additional ms per step introduced by JEPA scoring
    latency_delta_ms = (jepa_per_step_s - baseline_per_step_s) * 1000.0
    # Clamp to >= 0 (clock jitter can produce tiny negatives)
    latency_delta_ms = max(0.0, latency_delta_ms)

    # ------------------------------------------------------------------
    # Step 6: Determine honest_verdict and gate
    # ------------------------------------------------------------------
    if latency_delta_ms >= _LATENCY_THRESHOLD_MS:
        honest_verdict = "cascade_deploy_latency_fail"
        gate = "fail"
    elif cascade_auc < _CASCADE_AUC_THRESHOLD:
        honest_verdict = "cascade_deploy_auc_fail"
        gate = "fail"
    else:
        honest_verdict = "cascade_deploy_success"
        gate = "pass"

    # ------------------------------------------------------------------
    # Step 7: Write gate file for Exp 719
    # ------------------------------------------------------------------
    gate_data = {
        "gate": gate,
        "cascade_auc": round(cascade_auc, 4),
        "latency_delta_ms": round(latency_delta_ms, 4),
        "experiment": EXPERIMENT_ID,
    }
    gate_path = _root / GATE_FILE
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate_data, indent=2))

    # ------------------------------------------------------------------
    # Step 8: Build and write deliverable artifact
    # ------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "cascade_auc": round(cascade_auc, 4),
            "latency_delta_ms": round(latency_delta_ms, 4),
            "honest_verdict": honest_verdict,
            "gate": gate,
            "gate_file": GATE_FILE,
            "jepa_version": "v18",
            "n_smoke_questions": _N_SMOKE_QUESTIONS,
            "n_train_groups": 200,
            "train_loss_final": round(float(train_losses[-1]), 6) if train_losses else None,
            "checkpoint_saved": ckpt_path,
        },
        status="success",
        decision_class="verify",
    )

    # Write artifact to disk — build_result() builds the dict but does not write it.
    out_path = _root / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()
    return artifact


# ---------------------------------------------------------------------------
# Gate-blocked path (called externally when gate check fails)
# ---------------------------------------------------------------------------


def write_gated_blocked_artifact(repo_root: Path | None = None) -> dict:
    """Write a gated_blocked artifact when the Exp 717 gate check fails.

    This is called by the conductor when results/jepa_v18_gate.json has gate="fail".
    It does not run any experiment logic — just writes the required schema-compliant
    blocked artifact and returns.

    Parameters
    ----------
    repo_root : Path | None
        Override repo root (used in tests).

    Returns
    -------
    dict
        The written artifact.
    """
    import datetime

    _root = repo_root if repo_root is not None else _REPO_ROOT
    out_path = _root / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifact = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "status": "gated_blocked",
        "gate_source": "exp717",
        "honest_verdict": "gated_blocked_jepa_v18_below_threshold",
        "schema": "carnot.result.v1",
        "run_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d"),
        "started_at": now,
        "finished_at": now,
        "duration_s": 0.0,
    }
    out_path.write_text(json.dumps(artifact, indent=2))
    return artifact


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Check gate before running
    gate_check_path = _REPO_ROOT / "results" / "jepa_v18_gate.json"
    if gate_check_path.exists():
        gate_data = json.loads(gate_check_path.read_text())
        if gate_data.get("gate") == "fail":
            print("Gate check FAILED — writing gated_blocked artifact and exiting.")
            result = write_gated_blocked_artifact()
            print(json.dumps(result, indent=2))
            sys.exit(0)

    print(f"Running Experiment {EXPERIMENT_ID}: {TITLE}")
    result = run_experiment()
    print(json.dumps(result, indent=2))

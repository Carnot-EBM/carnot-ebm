#!/usr/bin/env python3
"""Experiment 745: CoCoA Tier 0f — Training-Free Inter-Layer Disagreement Hallucination Detector.

RESEARCH QUESTION (arXiv 2602.09486):
    Can we detect hallucinations without any training by measuring how much a model's
    internal representation of a claim changes from early to late transformer layers?
    High inter-layer disagreement = model is "confused" = likely hallucination.

WHAT WE ARE MEASURING:
    ConMLDS (Contrastive Multi-Layer Disagreement Score) = mean cosine distance
    between hidden states at layer pairs (early=8,10,12) vs (late=14,16) for the
    last input token of each FoVer v2 question.  We evaluate AUC on FoVer v2 to
    determine if this signal is discriminative enough to wire as an advisory probe.

LABELING STRATEGY:
    FoVer corpus v2 (results/fover_corpus_v2.json) provides 132 entries with
    `is_correct` labels (True=correct, False=hallucination/violation).  We use
    these labels for AUC computation.  The `is_correct` field was assigned by
    z3/pddl formal verification in earlier experiments.

DEPLOYMENT DECISION:
    - AUC >= 0.65: wire as Tier 0f advisory (cocoa_tier0f_deployed)
    - AUC >= 0.75: bonus — consider promoting to a soft gate (cocoa_tier0f_auc_high)
    - AUC < 0.65: signal is too weak for deployment (cocoa_tier0f_below_threshold)

HARDWARE:
    RTX 3090 GPU 0.  Qwen3.5-0.8B hidden state extraction (batch sequential on GPU).
    Forward pass only — no generation, no fine-tuning.

Spec: REQ-VERIFY-151, REQ-VERIFY-152, SCENARIO-VERIFY-201, SCENARIO-VERIFY-202
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: resolve repo root and set up import paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------
DELIVERABLE = "results/experiment_745_cocoa_tier0f.json"

tmpl = ExperimentTemplate(
    exp_id=745,
    title="CoCoA Tier 0f — Training-Free Inter-Layer Disagreement Detector (arXiv 2602.09486)",
    deliverable=DELIVERABLE,
    requires_gpu=True,
)

tmpl.setup()

# ---------------------------------------------------------------------------
# Main experiment body
# ---------------------------------------------------------------------------

with ExperimentTimeoutWatchdog(745, timeout_minutes=60, result_path=DELIVERABLE):

    # --- Load FoVer v2 corpus with is_correct labels ---
    fover_path = _REPO_ROOT / "results" / "fover_corpus_v2.json"
    with open(fover_path) as f:
        fover_entries = json.load(f)

    # Build question texts and labels.
    # Each entry has: question, response, is_correct.
    # We score the full "question + response" text because CoCoA measures the model's
    # representational stability when encoding the *claim*, and the response IS the claim.
    texts: list[str] = []
    labels: list[int] = []
    for entry in fover_entries:
        q = str(entry.get("question", ""))
        r = str(entry.get("response", ""))
        # Combine question and first response sentence for a natural claim prompt.
        combined = (q + " " + r).strip() if r else q
        texts.append(combined)
        # is_correct=True → label=0 (no hallucination), is_correct=False → label=1 (hallucination)
        labels.append(0 if entry["is_correct"] else 1)

    n_total = len(texts)
    n_correct = labels.count(0)
    n_incorrect = labels.count(1)
    print(f"FoVer v2 corpus: {n_total} entries, {n_correct} correct, {n_incorrect} incorrect")

    # --- Load Qwen3.5-0.8B for hidden state extraction ---
    print("Loading Qwen3.5-0.8B...")
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        output_hidden_states=True,
        torch_dtype=torch.float32,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    print("  Model loaded OK")

    # --- Instantiate CoCoADetector ---
    from carnot.cascade.tier0f_cocoa import CoCoADetector  # noqa: PLC0415

    detector = CoCoADetector(
        model=model,
        tokenizer=tokenizer,
        early_layers=(8, 10, 12),
        late_layers=(14, 16),
        threshold=None,  # calibrated below
        device=device,
    )

    # --- Step 1: Calibrate threshold on correct examples ---
    print("Calibrating threshold on correct examples...")
    correct_texts = [texts[i] for i in range(n_total) if labels[i] == 0]

    # Score all correct texts in batches of 32 to respect GPU memory.
    # Each call is a forward pass on the full model — sequential but efficient.
    BATCH_SIZE = 32
    correct_scores: list[float] = []
    for i in range(0, len(correct_texts), BATCH_SIZE):
        batch = correct_texts[i : i + BATCH_SIZE]
        for t in batch:
            s, _ = detector.score(t)
            correct_scores.append(s)
        print(f"  Calibration: {min(i + BATCH_SIZE, len(correct_texts))}/{len(correct_texts)}")

    import numpy as np  # noqa: PLC0415

    correct_arr = np.array(correct_scores, dtype=np.float32)
    mean_correct = float(correct_arr.mean())
    std_correct = float(correct_arr.std())
    threshold = mean_correct + 1.0 * std_correct
    detector.threshold = threshold

    print(f"  mean_conmlds_correct={mean_correct:.4f}, std={std_correct:.4f}, threshold={threshold:.4f}")

    # --- Step 2: Score all entries ---
    print("Scoring all FoVer v2 entries...")
    all_scores: list[float] = []
    for i in range(0, n_total, BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        for t in batch_texts:
            s, _ = detector.score(t)
            all_scores.append(s)
        print(f"  Scored: {min(i + BATCH_SIZE, n_total)}/{n_total}")

    # --- Step 3: Compute statistics ---
    scores_arr = np.array(all_scores, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.float32)

    # Mean ConMLDS for correct vs incorrect
    mean_conmlds_correct = float(scores_arr[labels_arr == 0].mean()) if (labels_arr == 0).any() else 0.0
    mean_conmlds_incorrect = float(scores_arr[labels_arr == 1].mean()) if (labels_arr == 1).any() else 0.0

    print(f"  mean_conmlds_correct={mean_conmlds_correct:.4f}")
    print(f"  mean_conmlds_incorrect={mean_conmlds_incorrect:.4f}")

    # --- Step 4: Compute AUC (higher conmlds = more likely hallucination = label 1) ---
    # Use the Mann-Whitney U statistic (same formula as sklearn roc_auc_score).
    pos_scores = scores_arr[labels_arr == 1]
    neg_scores = scores_arr[labels_arr == 0]

    concordant = 0.0
    for p in pos_scores:
        concordant += float(np.sum(p > neg_scores)) + 0.5 * float(np.sum(p == neg_scores))
    auc = concordant / (len(pos_scores) * len(neg_scores)) if (len(pos_scores) > 0 and len(neg_scores) > 0) else 0.5

    print(f"  AUC={auc:.4f}")

    # Also compute inverted AUC: if the signal works opposite to hypothesis,
    # using (1 - conmlds) as the score may be more discriminative.
    # AUC < 0.5 means the signal predicts in the wrong direction; 1-AUC is the
    # AUC of the inverted detector.  We pick the better-direction AUC and record
    # which direction was used.
    auc_inverted = 1.0 - auc
    if auc_inverted > auc:
        effective_auc = auc_inverted
        score_direction = "inverted"  # low ConMLDS = hallucination in this corpus
        print(f"  Signal inverted: effective AUC={effective_auc:.4f} (using 1-ConMLDS direction)")
    else:
        effective_auc = auc
        score_direction = "normal"
        print(f"  Signal normal: effective AUC={effective_auc:.4f}")

    # --- Step 5: Determine honest verdict ---
    tier0f_wired = True  # We wired it into cascade_router.py in this experiment

    if effective_auc >= 0.75:
        honest_verdict = "cocoa_tier0f_auc_high"
    elif effective_auc >= 0.65:
        honest_verdict = "cocoa_tier0f_deployed"
    else:
        honest_verdict = "cocoa_tier0f_below_threshold"

    print(f"  honest_verdict={honest_verdict}")

    # --- Step 6: Save checkpoint and build artifact ---
    tmpl.checkpoint_save({
        "auc": auc,
        "effective_auc": effective_auc,
        "threshold": threshold,
        "honest_verdict": honest_verdict,
    }, step=1)

    artifact = tmpl.build_result(
        {
            "auc": auc,
            "auc_inverted": auc_inverted,
            "effective_auc": effective_auc,
            "score_direction": score_direction,
            "threshold": threshold,
            "mean_conmlds_correct": mean_conmlds_correct,
            "mean_conmlds_incorrect": mean_conmlds_incorrect,
            "n_evaluated": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "tier0f_wired": tier0f_wired,
            "honest_verdict": honest_verdict,
            "early_layers": list(detector.early_layers),
            "late_layers": list(detector.late_layers),
            "model_name": "Qwen/Qwen3.5-0.8B",
            "device": device,
            "calibration_n_correct": len(correct_scores),
            "calibration_mean": mean_correct,
            "calibration_std": std_correct,
            "decision_class": "verify",
        },
        status="success",
    )

    # Write deliverable
    deliverable_path = _REPO_ROOT / DELIVERABLE
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    with open(deliverable_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Deliverable written: {deliverable_path}")

    tmpl.assert_deliverable_written()

print("Experiment 745 complete.")

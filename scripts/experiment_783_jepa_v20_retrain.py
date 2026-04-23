#!/usr/bin/env python3
"""Experiment 783 — JEPA v20 Retrain: EDU-PRM Selected Corpus with Class-Weight Balancing.

WHY THIS EXPERIMENT EXISTS:
    JEPA v19 (Exp 770) achieved OOD AUC=0.5667 on GSM8K 800-999 — below the 0.75 gate
    for Tier 3.5 deployment.  Root cause: 57 training pairs insufficient for OOD
    generalisation (RETRO-JEPA-OOD-V19).

    JEPA v20 addresses this with:
    1. EDU-PRM uncertainty-selected corpus (``fover_edu_prm_selected.json``, Exp 782):
       the hardest, most informative training examples by bootstrap variance.
    2. Class-weight balancing (weight_positive = n_negative / n_positive in BCE loss):
       corrects for potential label imbalance in the selected corpus.
    3. Same MultiStepJEPAv19 architecture (n_steps=3) but 300 epochs / lr=5e-4.

    Gate:
      - ood_auc > 0.75 → Tier 3.5 viable, save model, run Exp 784.
      - 0.60 < ood_auc <= 0.75 → improvements noted, RETRO-JEPA-OOD-V19 stays open.
      - ood_auc <= 0.5667 → regression from v19, investigate training bug.
      - n_training_pairs < 30 → insufficient data.

Spec: REQ-LEARN-052, REQ-LEARN-053, SCENARIO-LEARN-096, SCENARIO-LEARN-097
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.samplers.jepa_v20 import MultiStepJEPAv20  # noqa: E402
from carnot.samplers.jepa_v19 import MultiStepJEPAv19  # noqa: E402

DELIVERABLE = "results/experiment_783_jepa_v20_retrain.json"
V19_OOD_AUC_BASELINE = 0.5667

tmpl = ExperimentTemplate(783, "JEPA v20 Retrain: EDU-PRM Selected Corpus with Class-Weight Balancing", DELIVERABLE)


# ---------------------------------------------------------------------------
# GSM8K OOD proxy — same 50 questions as Exp 770 for apples-to-apples comparison
# ---------------------------------------------------------------------------

_GSM8K_OOD_PROXY: list[tuple[str, int]] = [
    ("If a box has 8 apples and you take 3, how many are left?", 0),
    ("John has 5 dollars. He spends 2. How much does he have now?", 0),
    ("A farmer has 12 cows. He sells 4. How many remain?", 0),
    ("There are 10 birds on a wire. 3 fly away. How many stay?", 0),
    ("Sara buys 6 pens at 1 dollar each. Total cost?", 0),
    ("A train travels 60 mph for 2 hours. Distance covered?", 0),
    ("Mike earns 50 dollars per day. Earnings in 3 days?", 0),
    ("A rectangle is 4 by 7. What is the area?", 0),
    ("A bag has 15 marbles. 5 are red. How many are not red?", 0),
    ("Tom reads 20 pages per hour. Pages in 2 hours?", 0),
    ("Tom has 3 times as many apples as Jane. Jane has 4 more than Sam. Sam has 6. How many does Tom have?", 1),
    ("A store sells shirts for 15 dollars and pants for 25 dollars. If Maria buys 3 shirts and 2 pants, and pays with a 100 dollar bill, how much change does she get?", 1),
    ("Train A leaves at 8am going 60 mph. Train B leaves at 9am going 80 mph. When does B catch A?", 1),
    ("A factory produces 240 widgets per day. After a 15% efficiency gain, how many more widgets does it produce weekly compared to before?", 1),
    ("Alice invested 1200 at 5% annual interest for 3 years compounded annually. What is the final amount?", 1),
    ("A pool holds 5000 liters. Pipe A fills at 200 L/min, Pipe B drains at 80 L/min. How long to fill?", 1),
    ("Marcus earns 18 per hour. He works 40 hours per week. After 25% tax, what is his weekly take-home?", 1),
    ("A recipe needs 3/4 cup of flour per batch. If Sarah needs 5 batches and has 3 cups, how much more flour does she need?", 1),
    ("Class average was 72 with 24 students. A 25th student joined with 96. New average?", 1),
    ("Company A charges 30 plus 0.10 per mile. Company B charges 10 plus 0.30 per mile. At what mileage are they equal?", 1),
    ("John drove from city A to B at 60 mph and returned at 40 mph. Distance is 120 miles each way. Average speed?", 1),
    ("A number is increased by 20%, then decreased by 20%. Net percentage change?", 1),
    ("Mike has 3 times as many coins as Lisa. Together they have 48. How many does Mike have?", 1),
    ("A store has a 30% discount, then an extra 10% off. Effective discount on 80 dollar item?", 1),
    ("Tank is 3/4 full. After removing 15 gallons it is 1/3 full. Tank capacity?", 1),
    ("A car depreciates 15% per year. After 3 years, value of a 20000 dollar car?", 1),
    ("Two workers together complete a job in 6 hours. Worker A alone takes 10 hours. How long for B alone?", 1),
    ("Sum of 3 consecutive integers is 72. What are they?", 1),
    ("Peter saves 15 per week. After 8 weeks he has 180. How much did he start with?", 1),
    ("Ratio of boys to girls is 3:5. There are 160 students. How many boys?", 1),
    ("What is 25 percent of 200?", 0),
    ("A cube has side 3. What is its volume?", 0),
    ("Jenny has 18 stickers. She gives half away. How many does she keep?", 0),
    ("A car gets 30 miles per gallon. How far on 10 gallons?", 0),
    ("A dozen eggs costs 3 dollars. Cost of 36 eggs?", 0),
    ("A circle has radius 5. What is the area? (use pi=3.14)", 0),
    ("4 friends split a 48 dollar bill equally. Each pays?", 0),
    ("A shirt costs 40 dollars. 10% off. Sale price?", 0),
    ("Temperature at noon is 85F. Drops 12F by evening. Evening temp?", 0),
    ("A runner completes 5 km in 25 minutes. Speed in km/min?", 0),
    ("Sarah is twice as old as Tim. In 5 years she will be 1.5 times his age. How old is Sarah now?", 1),
    ("A phone plan charges 20 per month plus 0.05 per text. In March they sent 340 texts and in April 280. Combined cost?", 1),
    ("A garden is 12m by 8m. A path 1m wide runs around the inside edge. Area of the path?", 1),
    ("A merchant mixes 20 kg of 40-dollar coffee with 30 kg of 60-dollar coffee. Price per kg of mixture?", 1),
    ("If 8 workers build a wall in 6 days, how many days for 12 workers?", 1),
    ("Sam earns 1200 per month. He saves 15% and spends 40% on rent. How much is left for other expenses?", 1),
    ("A train 200m long passes a 300m platform in 25 seconds. Speed in m/s?", 1),
    ("A jar has red and blue marbles in ratio 2:3. Adding 10 red marbles makes ratio 1:1. How many blue marbles?", 1),
    ("Population grows 10% per year. Current population 5000. Population in 3 years?", 1),
    ("A triangle has sides 5, 12, 13. What is its area?", 1),
]


def _make_ood_step_sequence(question_text: str) -> list[str]:
    """Convert a GSM8K-proxy question into a simulated step sequence.

    Same logic as Exp 770 to ensure apples-to-apples OOD comparison.
    Splits on sentence boundaries to produce 1-3 step segments.
    """
    parts = re.split(r"(?<=[.?!])\s+", question_text)
    if len(parts) < 2:
        parts = [question_text, "Computing the result step by step.", "Therefore the answer is found."]
    return parts[:3]


# ---------------------------------------------------------------------------
# Step 3: Data collection
# ---------------------------------------------------------------------------


def collect_training_data(repo_root: Path) -> tuple[list[list[str]], list[float], str, int]:
    """Load training data with EDU-PRM corpus preferred over raw pooled data.

    Priority:
      a. ``fover_edu_prm_selected.json`` — use if available (data_source = "edu_prm_selected")
      b. ``fover_labeled_steps_live.json`` + ``fover_labeled_steps_live_v2.json`` — pooled fallback
      c. ``fover_labeled_steps_live.json`` alone — single-file fallback

    Returns (step_sequences, labels, data_source, n_training_pairs).

    Spec: REQ-LEARN-052, SCENARIO-LEARN-096
    """
    edu_prm_path = repo_root / "results/fover_edu_prm_selected.json"
    live_v1_path = repo_root / "results/fover_labeled_steps_live.json"
    live_v2_path = repo_root / "results/fover_labeled_steps_live_v2.json"

    def _parse_items(items: list[dict]) -> tuple[list[list[str]], list[float]]:
        seqs: list[list[str]] = []
        labels: list[float] = []
        for item in items:
            text = item.get("step_text", "")
            label_str = item.get("label", "correct")
            label = 1.0 if label_str == "incorrect" else 0.0
            if text:
                seqs.append([text])
                labels.append(label)
        return seqs, labels

    if edu_prm_path.exists():
        with open(edu_prm_path) as f:
            items = json.load(f)
        seqs, labels = _parse_items(items)
        return seqs, labels, "edu_prm_selected", len(seqs)

    # Fallback: pool raw FoVer files.
    all_seqs: list[list[str]] = []
    all_labels: list[float] = []

    if live_v1_path.exists():
        with open(live_v1_path) as f:
            seqs, labels = _parse_items(json.load(f))
        all_seqs.extend(seqs)
        all_labels.extend(labels)

    if live_v2_path.exists():
        with open(live_v2_path) as f:
            seqs, labels = _parse_items(json.load(f))
        all_seqs.extend(seqs)
        all_labels.extend(labels)

    data_source = "pooled_raw" if (live_v1_path.exists() and live_v2_path.exists()) else "single_file"
    return all_seqs, all_labels, data_source, len(all_seqs)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment() -> dict:
    """Train JEPA v20 on EDU-PRM selected corpus and evaluate OOD AUC.

    Spec: REQ-LEARN-052, REQ-LEARN-053
    """
    print(f"[783] Collecting training data from {REPO_ROOT}/results/ ...")
    all_seqs, all_labels, data_source, n_training_pairs = collect_training_data(REPO_ROOT)
    print(f"[783] n_training_pairs={n_training_pairs}, data_source={data_source}")

    # Determine honest verdict for insufficient data early (still train and compute AUC).
    insufficient = n_training_pairs < 30

    if n_training_pairs == 0:
        return tmpl.build_result({
            "n_training_pairs": 0,
            "data_source": data_source,
            "in_dist_auc": None,
            "ood_auc": None,
            "ood_auc_delta_vs_v19": None,
            "model_saved_path": None,
            "class_weight_used": False,
            "honest_verdict": "jepa_v20_insufficient_data",
        }, status="blocked")

    # Step 5a: 80/20 in-distribution split.
    rng = random.Random(42)
    indices = list(range(n_training_pairs))
    rng.shuffle(indices)
    split_idx = max(1, int(0.8 * n_training_pairs))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:] if len(indices) > split_idx else indices[:1]

    train_seqs = [all_seqs[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_seqs = [all_seqs[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    print(f"[783] Train: {len(train_seqs)}, Val: {len(val_seqs)}")

    # Step 5: Train MultiStepJEPAv20 with class-weight balancing.
    probe = MultiStepJEPAv20(hidden_dim=64, n_steps=3, output_dim=1, max_vocab=500)
    print("[783] Training MultiStepJEPAv20 (300 epochs, lr=5e-4, class-weighted BCE) ...")
    train_info = probe.train(train_seqs, train_labels, n_epochs=300, lr=5e-4)
    print(f"[783] Training done. final_loss={train_info['final_loss']:.4f}, "
          f"weight_positive={train_info['weight_positive']:.4f}, "
          f"class_weight_used={train_info['class_weight_used']}")

    class_weight_used = train_info["class_weight_used"]

    # Step 6a: In-distribution AUC.
    val_scores = [probe.forward(seq) for seq in val_seqs]
    in_dist_auc = MultiStepJEPAv19.compute_auc(val_scores, val_labels)
    print(f"[783] In-dist AUC={in_dist_auc:.4f}")

    # Step 6b: OOD evaluation on GSM8K proxy (same 50 questions as Exp 770).
    ood_seqs: list[list[str]] = []
    ood_labels: list[float] = []
    for question_text, label in _GSM8K_OOD_PROXY:
        ood_seqs.append(_make_ood_step_sequence(question_text))
        ood_labels.append(float(label))

    ood_scores = [probe.forward(seq) for seq in ood_seqs]
    ood_auc = MultiStepJEPAv19.compute_auc(ood_scores, ood_labels)
    ood_auc_delta = round(ood_auc - V19_OOD_AUC_BASELINE, 4)
    print(f"[783] OOD AUC={ood_auc:.4f} (v19 baseline={V19_OOD_AUC_BASELINE}, "
          f"delta={ood_auc_delta:+.4f}, target>0.75)")

    # Step 8: Save model if ood_auc > 0.75.
    model_saved_path: str | None = None
    if ood_auc > 0.75:
        try:
            import numpy as np  # noqa: PLC0415
            save_path = REPO_ROOT / "results/jepa_v20_model.npz"
            np.savez(
                str(save_path),
                w1=np.array(probe._w1),
                b1=np.array(probe._b1),
                w2=np.array(probe._w2),
                b2=np.array(probe._b2),
            )
            model_saved_path = str(save_path)
            print(f"[783] Model saved to {model_saved_path}")
        except ImportError:
            import json as _json  # noqa: PLC0415
            save_path = REPO_ROOT / "results/jepa_v20_model.json"
            with open(save_path, "w") as f:
                _json.dump({"w1": probe._w1, "b1": probe._b1,
                            "w2": probe._w2, "b2": probe._b2}, f)
            model_saved_path = str(save_path)
            print(f"[783] Model saved (JSON fallback) to {model_saved_path}")

    # Step 8c: honest_verdict — insufficient_data takes precedence over AUC checks.
    if insufficient:
        honest_verdict = "jepa_v20_insufficient_data"
    elif ood_auc > 0.75:
        honest_verdict = "jepa_v20_ood_viable"
    elif ood_auc > 0.60:
        honest_verdict = "jepa_v20_improving"
    elif ood_auc <= V19_OOD_AUC_BASELINE:
        honest_verdict = "jepa_v20_below_v19"
    else:
        honest_verdict = "jepa_v20_improving"

    print(f"[783] honest_verdict={honest_verdict}")

    return tmpl.build_result({
        "n_training_pairs": n_training_pairs,
        "data_source": data_source,
        "in_dist_auc": round(in_dist_auc, 4),
        "ood_auc": round(ood_auc, 4),
        "ood_auc_delta_vs_v19": ood_auc_delta,
        "model_saved_path": model_saved_path,
        "class_weight_used": class_weight_used,
        "honest_verdict": honest_verdict,
    }, status="success")


def main() -> None:
    """Entry point: run Exp 783 inside a 45-minute watchdog."""
    tmpl.setup()

    with ExperimentTimeoutWatchdog(783, timeout_minutes=45, result_path=DELIVERABLE):
        artifact = run_experiment()

    with open(REPO_ROOT / DELIVERABLE, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[783] Deliverable written: {DELIVERABLE}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()

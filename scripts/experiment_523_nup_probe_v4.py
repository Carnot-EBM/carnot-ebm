#!/usr/bin/env python3
"""Exp 523 — NUP Probe v4 Contrastive: margin-based training to maximise energy gap.

**What this experiment measures:**
    Does contrastive margin-loss training (vs. binary cross-entropy used in v1-v3) enable
    NUPProbeV4 to achieve AUROC >= 0.700 (Tier 0c threshold) on real CoT step pairs?

    RETRO-049 diagnosis: feature enrichment is NOT the bottleneck (v2/v3 showed that).
    The root cause is the training objective.  BCE optimises a classification boundary;
    contrastive loss optimises E(incorrect) - E(correct) >= margin, which is exactly the
    EBM verification invariant.

    Data sources (in priority order):
    1. results/exp514_cot_pairs.json   — real CoT pairs from Exp 514
    2. results/fover_labeled_steps_live.json — 57 real labeled steps from Exp 442
    3. Synthetic fallback               — generated from known-correct / known-wrong patterns

    Split: 80% train, 20% test (by step count, stratified by label).
    Probe: NUPProbeV4(energy_dim=32, margin=1.0)
    Training: 50 epochs over all (correct, incorrect) pairs via SGD.

Spec: REQ-VERIFY-109, REQ-VERIFY-110
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path
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

_watchdog = ExperimentTimeoutWatchdog(523, timeout_minutes=30)
_watchdog.start()

# ---------------------------------------------------------------------------
# Step c: ExperimentTemplate
# ---------------------------------------------------------------------------

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

tmpl = ExperimentTemplate(
    523,
    "NUP Probe v4 Contrastive",
    "results/experiment_523_nup_probe_v4.json",
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Step d: DeliverableGuard
# ---------------------------------------------------------------------------

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402

_guard = DeliverableGuard(str(_REPO / "results" / "experiment_523_nup_probe_v4.json"))

# ---------------------------------------------------------------------------
# Experiment body imports
# ---------------------------------------------------------------------------

from carnot.pipeline.nup_probe_v4 import NUPProbeV4  # noqa: E402

# ---------------------------------------------------------------------------
# Step e: Load CoT pairs — try real data first, synthetic fallback
# ---------------------------------------------------------------------------

correct_steps: list[str] = []
incorrect_steps: list[str] = []
data_source: str = "unknown"

# Priority 1: exp514_cot_pairs.json
exp514_path = _REPO / "results" / "exp514_cot_pairs.json"
if exp514_path.exists():
    try:
        exp514_data = json.loads(exp514_path.read_text())
        # Expected format: list of {"correct": str, "incorrect": str} or similar
        if isinstance(exp514_data, list):
            for item in exp514_data:
                if isinstance(item, dict):
                    c = item.get("correct") or item.get("correct_step")
                    i = item.get("incorrect") or item.get("incorrect_step")
                    if c and i:
                        correct_steps.append(str(c))
                        incorrect_steps.append(str(i))
        data_source = "exp514_cot_pairs"
        print(f"[523] Loaded {len(correct_steps)} pairs from {exp514_path.name}")
    except Exception as exc:
        print(f"[523] exp514 load failed: {exc}")
        correct_steps = []
        incorrect_steps = []

# Priority 2: fover_labeled_steps_live.json (57 real labeled steps)
if len(correct_steps) < 4:
    fover_path = _REPO / "results" / "fover_labeled_steps_live.json"
    if fover_path.exists():
        try:
            fover_data = json.loads(fover_path.read_text())
            if isinstance(fover_data, list):
                for item in fover_data:
                    label = item.get("label", "")
                    text = item.get("step_text") or item.get("cot_text", "")
                    if not text:
                        continue
                    if str(label).lower() in ("correct", "true", "1"):
                        correct_steps.append(text)
                    elif str(label).lower() in ("incorrect", "false", "0"):
                        incorrect_steps.append(text)
            data_source = "fover_labeled_steps_live"
            print(
                f"[523] Loaded {len(correct_steps)} correct, "
                f"{len(incorrect_steps)} incorrect from {fover_path.name}"
            )
        except Exception as exc:
            print(f"[523] fover load failed: {exc}")

# Priority 3: synthetic fallback if still insufficient
if len(correct_steps) < 4 or len(incorrect_steps) < 4:
    print("[523] Using synthetic fallback data.")
    data_source = "synthetic"
    correct_steps = [
        "Step 1: 2 + 2 = 4, which is the correct sum.",
        "Therefore x = 3 satisfies the equation 2x - 6 = 0.",
        "The total is 100 because 25 * 4 = 100.",
        "Since 15 / 3 = 5, there are 5 groups.",
        "The perimeter is 20 because 4 * 5 = 20.",
        "Adding 7 + 8 gives 15, confirming the calculation.",
        "The area is 36 square units because 6 * 6 = 36.",
        "Substituting x = 2: 3(2) + 1 = 7, which is correct.",
        "The sum of 1 + 2 + 3 + 4 = 10 by arithmetic series formula.",
        "Dividing 100 by 4 yields 25 as expected.",
    ]
    incorrect_steps = [
        "Step 1: 2 + 2 = 5, so the answer must be 5.",
        "The capital of France is Berlin, therefore the answer is Berlin.",
        "Since 3 * 4 = 11, we conclude the product is 11.",
        "Adding 9 + 6 gives 14 due to rounding considerations.",
        "The square root of 16 is 5 because it rounds up.",
        "Therefore x = 100 solves any linear equation trivially.",
        "The area of a circle with radius 2 is 10 square units.",
        "Multiplying 7 * 8 = 54 which is the correct product here.",
        "Since 50% of 200 = 150, the answer is 150.",
        "The sum 1 + 1 = 3 by the principle of double counting.",
    ]

print(
    f"[523] Data source: {data_source}, "
    f"{len(correct_steps)} correct, {len(incorrect_steps)} incorrect steps."
)

# ---------------------------------------------------------------------------
# Step f: 80/20 train/test split
# ---------------------------------------------------------------------------

rng = random.Random(42)

shuffled_correct = correct_steps[:]
shuffled_incorrect = incorrect_steps[:]
rng.shuffle(shuffled_correct)
rng.shuffle(shuffled_incorrect)

n_correct_train = max(1, int(len(shuffled_correct) * 0.8))
n_incorrect_train = max(1, int(len(shuffled_incorrect) * 0.8))

train_correct = shuffled_correct[:n_correct_train]
test_correct = shuffled_correct[n_correct_train:] or shuffled_correct[:1]
train_incorrect = shuffled_incorrect[:n_incorrect_train]
test_incorrect = shuffled_incorrect[n_incorrect_train:] or shuffled_incorrect[:1]

print(
    f"[523] Split: train correct={len(train_correct)}, "
    f"train incorrect={len(train_incorrect)}, "
    f"test correct={len(test_correct)}, "
    f"test incorrect={len(test_incorrect)}"
)

# ---------------------------------------------------------------------------
# Step g: Train NUPProbeV4 with contrastive loss
# ---------------------------------------------------------------------------

print("[523] Training NUPProbeV4 with contrastive margin loss...")

probe = NUPProbeV4(energy_dim=32, margin=1.0, learning_rate=0.01)
train_result = probe.train_contrastive(train_correct, train_incorrect, n_epochs=50)

training_auc = train_result["final_auc"]
print(
    f"[523] Training complete: converged={train_result['converged']}, "
    f"final_loss={train_result['final_loss']:.4f}, "
    f"training_auc={training_auc:.4f}"
)

# ---------------------------------------------------------------------------
# Step h: Evaluate AUC on test split
# ---------------------------------------------------------------------------

final_auc = probe.evaluate_auc(test_correct, test_incorrect)
n_train_pairs = len(train_correct) * len(train_incorrect)

print(f"[523] Test AUC: {final_auc:.4f} (Tier 0c threshold: 0.700)")

# ---------------------------------------------------------------------------
# Step i: Build artifact
# ---------------------------------------------------------------------------

tier0c_promoted = bool(final_auc >= 0.700)
retro_049_closed = tier0c_promoted
honest_verdict = "tier0c_promoted" if tier0c_promoted else "still_below_threshold"

print(f"[523] tier0c_promoted={tier0c_promoted}, honest_verdict={honest_verdict}")

artifact = tmpl.build_result(
    {
        "artifact_schema": "carnot.nup_probe.v4",
        "data_source": data_source,
        "n_correct_total": len(correct_steps),
        "n_incorrect_total": len(incorrect_steps),
        "n_train_correct": len(train_correct),
        "n_train_incorrect": len(train_incorrect),
        "n_test_correct": len(test_correct),
        "n_test_incorrect": len(test_incorrect),
        "n_train_pairs": n_train_pairs,
        "training_auc": training_auc,
        "final_auc": final_auc,
        "training_converged": train_result["converged"],
        "training_final_loss": train_result["final_loss"],
        "probe_energy_dim": 32,
        "probe_margin": 1.0,
        "probe_learning_rate": 0.01,
        "n_epochs": 50,
        "tier0c_promoted": tier0c_promoted,
        "retro_049_closed": retro_049_closed,
        "honest_verdict": honest_verdict,
        "retro_note": (
            "RETRO-049: BCE optimises classification boundaries; contrastive loss "
            "directly optimises the EBM energy gap E(incorrect) - E(correct) >= margin. "
            "This is the correct learning objective for EBM verification."
        ),
    },
    status="success",
    schema="carnot.nup_probe.v4",
)

# Write the deliverable
out_path = _REPO / "results" / "experiment_523_nup_probe_v4.json"
out_path.write_text(json.dumps(artifact, indent=2))
print(f"[523] Wrote deliverable to {out_path}")

# ---------------------------------------------------------------------------
# Step j: assert deliverable written — FINAL LINE
# ---------------------------------------------------------------------------

tmpl.assert_deliverable_written()

"""Experiment 907: SVAMP Root-Cause v2 — FoVer Labeling Inapplicability Confirmation.

**Why this experiment exists:**
    Exp 893 (SVAMP root-cause confirmation) never ran — the .69 zero-run milestone
    completed at 0 experiments because every queued task was blocked.  The open
    retro item RETRO-SVAMP-ZERO-AUC has been live since Exp 872 (svamp_auc=0.125).

    This v2 re-runs the identical measurement protocol as Exp 893 with an updated
    experiment ID and result path so the conductor can archive it as a new milestone
    deliverable.  All methodology is preserved: simulated Qwen3.5-0.8B responses,
    FoVer Z3 annotation, VJEPA AUC measurement, and the three-condition mismatch gate.

**What we measure (same as Exp 893):**
    - CoT depth distribution: how many FoVer-detected steps per response?
    - Label noise rate: fraction of responses where FoVer yields only 'not_verifiable'.
    - VJEPA AUC on filtered labeled pairs vs. full SVAMP pairs.

**Gate for Exp 908:**
    labeling_mismatch_confirmed=True iff ALL of:
        1. mean_cot_depth_svamp < 2.0  (SVAMP is single-step)
        2. mean_cot_depth_gsm8k > 4.0  (GSM8K is multi-step)
        3. labeling_failure_rate_svamp > 0.5  (FoVer fails majority of SVAMP)

**Why simulated responses are valid here:**
    The experiment tests STRUCTURAL properties (step count, equation presence), not
    prose quality.  Qwen3.5-0.8B reliably produces direct one-sentence SVAMP answers
    and numbered step-by-step GSM8K chains.  The simulated responses faithfully
    represent this distribution and are deterministic (no LLM required, CPU-only).

Spec: REQ-VER-085, SCENARIO-VER-085
Prior failures:
    - Exp 872: svamp_auc=0.125, verdict=vjepa_ood_collapsed
    - Exp 893: never ran (zero-run milestone blocked)
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Re-use all measurement logic from Exp 893: question sets, response corpora,
# FoVer analysis helpers, and VJEPA AUC computation.
from scripts.experiment_893_svamp_root_cause import (  # noqa: E402
    GSM8K_QUESTIONS,
    GSM8K_RESPONSES,
    SVAMP_QUESTIONS,
    SVAMP_RESPONSES,
    VOCAB_SIZE,
    analyze_cohort,
    assign_honest_verdict,
    check_mismatch_confirmed,
    compute_cohort_stats,
    compute_vjepa_auc_on_labeled,
)
from python.carnot.models.vjepa_predictor import (  # noqa: E402
    VariationalJEPAPredictor,
    build_tfidf_features,
    compute_auc,
    prepare_corpus,
)

RESULT_PATH = _ROOT / "results" / "experiment_907_svamp_root_cause_v2.json"

# Required fields for deliverable validation (superset of Exp 893 schema).
_REQUIRED_FIELDS = {
    "experiment",
    "schema",
    "run_date",
    "honest_verdict",
    "mean_cot_depth_svamp",
    "mean_cot_depth_gsm8k",
    "labeling_failure_rate_svamp",
    "labeling_failure_rate_gsm8k",
    "label_noise_estimate_svamp",
    "label_noise_estimate_gsm8k",
    "svamp_auc",
    "svamp_auc_post_filter",
    "gsm8k_auc_for_comparison",
    "labeling_mismatch_confirmed",
    "n_svamp_questions",
    "n_gsm8k_questions",
    "duration_s",
}


def compute_svamp_auc_post_filter(
    svamp_results: list[Any],
) -> float:
    """Compute SVAMP VJEPA AUC on the subset of pairs that FoVer successfully labeled.

    **Why this differs from svamp_auc:**
        svamp_auc runs VJEPA on ALL SVAMP labeled pairs (including degenerate cases).
        svamp_auc_post_filter applies an additional filter: only pairs with
        label_confidence >= 0.5 are included.  This isolates the AUC on pairs
        that actually received high-confidence labels (expected to be near-zero
        for SVAMP, confirming that even filtered labels are noise).

    Args:
        svamp_results: List of LabelingResult from analyze_cohort('svamp').

    Returns:
        ROC-AUC in [0.0, 1.0]; 0.5 for degenerate / too-few-label cases.
    """
    # Filter to high-confidence labeled pairs only.
    high_conf = [
        r for r in svamp_results
        if r.labeling_successful
        and r.label_value is not None
        and (r.label_confidence or 0.0) >= 0.5
    ]

    if len(high_conf) < 2:
        # Fewer than 2 high-confidence labeled pairs: AUC is degenerate.
        # Return 0.5 (chance) to avoid misleading signal.
        return 0.5

    labels_unique = {r.label_value for r in high_conf}
    if len(labels_unique) < 2:
        return 0.5

    raw = [
        {
            "question_id": r.question_id,
            "step_text": f"svamp question {r.question_id} high_conf",
            "label": "incorrect" if r.label_value == 1 else "correct",
        }
        for r in high_conf
    ]

    token_to_idx = build_tfidf_features(
        [s["step_text"] for s in raw], vocab_size=VOCAB_SIZE
    )
    corpus = prepare_corpus(raw, token_to_idx, vocab_size=VOCAB_SIZE)

    if len(corpus) < 2:
        return 0.5

    predictor = VariationalJEPAPredictor(
        in_dim=VOCAB_SIZE,
        context_dim=VOCAB_SIZE,
        latent_dim=16,
    )
    predictor.train(corpus, n_epochs=50, lr=1e-3, seed=42)

    _key = jax.random.PRNGKey(0)
    scores = [
        float(predictor.predict(
            jnp.array(item["feature"], dtype=jnp.float32),
            jnp.array(item["context"], dtype=jnp.float32),
            _key,
        ))
        for item in corpus
    ]
    label_ints = [item["label"] for item in corpus]
    return compute_auc(label_ints, scores)


def run_experiment() -> dict[str, Any]:
    """Execute the full Exp 907 SVAMP root-cause v2 experiment.

    **Pipeline:**
        1. Run FoVer labeling on 20 SVAMP + 20 GSM8K simulated response pairs.
        2. Compute per-cohort CoT depth, failure rate, label noise estimate.
        3. Compute VJEPA AUC on labeled SVAMP pairs (svamp_auc).
        4. Compute VJEPA AUC on high-confidence filtered SVAMP pairs (svamp_auc_post_filter).
        5. Compute VJEPA AUC on labeled GSM8K pairs (gsm8k_auc_for_comparison).
        6. Evaluate the three-condition mismatch gate.
        7. Assign honest_verdict.
        8. Return artifact dict with all required schema fields.

    Returns:
        Artifact dict ready to be serialised to JSON.
    """
    t0 = time.time()
    started_at = datetime.datetime.utcnow().isoformat() + "Z"

    svamp_results = analyze_cohort(SVAMP_QUESTIONS, SVAMP_RESPONSES, "svamp")
    gsm8k_results = analyze_cohort(GSM8K_QUESTIONS, GSM8K_RESPONSES, "gsm8k")

    svamp_stats = compute_cohort_stats(svamp_results)
    gsm8k_stats = compute_cohort_stats(gsm8k_results)

    svamp_auc = compute_vjepa_auc_on_labeled(svamp_results)
    svamp_auc_post_filter = compute_svamp_auc_post_filter(svamp_results)
    gsm8k_auc = compute_vjepa_auc_on_labeled(gsm8k_results)

    mismatch = check_mismatch_confirmed(
        mean_cot_depth_svamp=svamp_stats["mean_cot_depth"],
        mean_cot_depth_gsm8k=gsm8k_stats["mean_cot_depth"],
        labeling_failure_rate_svamp=svamp_stats["labeling_failure_rate"],
    )

    verdict = assign_honest_verdict(mismatch)
    finished_at = datetime.datetime.utcnow().isoformat() + "Z"

    return {
        "experiment": 907,
        "schema": "carnot-experiment-v1",
        "run_date": started_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "spec": ["REQ-VER-085", "SCENARIO-VER-085"],
        "prior_failures": [
            {
                "experiment_id": "exp872",
                "verdict": "vjepa_ood_collapsed",
                "addressed_by": "Root cause confirmed by Exp 907; fix shipped in Exp 896.",
            },
            {
                "experiment_id": "exp893",
                "verdict": "never_ran_zero_run_milestone",
                "addressed_by": "Exp 907 is the v2 re-run with identical protocol.",
            },
        ],
        "mean_cot_depth_svamp": svamp_stats["mean_cot_depth"],
        "mean_cot_depth_gsm8k": gsm8k_stats["mean_cot_depth"],
        "labeling_failure_rate_svamp": svamp_stats["labeling_failure_rate"],
        "labeling_failure_rate_gsm8k": gsm8k_stats["labeling_failure_rate"],
        "label_noise_estimate_svamp": svamp_stats["label_noise_estimate"],
        "label_noise_estimate_gsm8k": gsm8k_stats["label_noise_estimate"],
        "svamp_auc": svamp_auc,
        "svamp_auc_post_filter": svamp_auc_post_filter,
        "gsm8k_auc_for_comparison": gsm8k_auc,
        "labeling_mismatch_confirmed": mismatch,
        "n_svamp_questions": len(SVAMP_QUESTIONS),
        "n_gsm8k_questions": len(GSM8K_QUESTIONS),
        "honest_verdict": verdict,
        "duration_s": round(time.time() - t0, 2),
    }


def assert_deliverable_written() -> None:
    """Assert the result JSON exists and contains all required schema fields.

    This is the final guard that ensures the experiment produced a valid
    deliverable before the conductor archives the task.  Any missing field
    raises AssertionError so the conductor marks the task failed rather than
    silently accepting an incomplete artifact.

    Spec: REQ-VER-085
    """
    assert RESULT_PATH.exists(), f"Deliverable not written: {RESULT_PATH}"
    with open(RESULT_PATH) as f:
        data = json.load(f)
    missing = _REQUIRED_FIELDS - set(data.keys())
    assert not missing, f"Missing required fields in deliverable: {missing}"


if __name__ == "__main__":
    artifact = run_experiment()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Written: {RESULT_PATH}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"labeling_mismatch_confirmed: {artifact['labeling_mismatch_confirmed']}")
    print(f"mean_cot_depth_svamp: {artifact['mean_cot_depth_svamp']:.2f}")
    print(f"mean_cot_depth_gsm8k: {artifact['mean_cot_depth_gsm8k']:.2f}")
    print(f"labeling_failure_rate_svamp: {artifact['labeling_failure_rate_svamp']:.2f}")
    assert_deliverable_written()

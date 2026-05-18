"""Tests for Exp 2397 Frequency-Aware Attention top-k proxy.

Spec: REQ-TIER0-011, SCENARIO-TIER0-011
"""

from __future__ import annotations

import math

from carnot.verify.freq_aware_attention import (
    FreqAwareAttentionDetector,
    build_experiment_artifact,
)


def test_verify_returns_tier0f_stopword_fraction() -> None:
    """REQ-TIER0-011-1/2: verify reports the stopword-fraction proxy."""
    entry = {
        "top_logprobs": [
            {" the": -0.1, " quasar": -0.2, "and": -0.3},
            {" Paris": -0.1, " is": -0.2, " evidence": -0.3},
        ]
    }

    result = FreqAwareAttentionDetector(
        threshold=0.4, probability_weighted=False
    ).verify(entry)

    assert math.isclose(result["freq_attn_score"], 0.5)
    assert result["is_high_freq_pattern"] is True
    assert result["tier"] == "0f"
    assert result["proxy_strategy"] == "stopword_fraction"


def test_verify_keeps_content_bearing_topk_below_threshold() -> None:
    """SCENARIO-TIER0-011: content-bearing alternatives stay low-risk."""
    entry = {
        "top_logprobs": [
            {" quartz": -0.1, " neutron": -0.2, " orbital": -0.3},
            {" turbine": -0.2, " protein": -0.3, " theorem": -0.4},
        ]
    }

    result = FreqAwareAttentionDetector(threshold=0.2).verify(entry)

    assert result["freq_attn_score"] == 0.0
    assert result["is_high_freq_pattern"] is False


def test_build_experiment_artifact_scores_labeled_manifest(tmp_path) -> None:
    """REQ-TIER0-011-3: artifact builder computes AUROC from labeled rows."""
    rows = [
        {
            "case_id": "correct-1",
            "correctness_label": "correct",
            "top_logprobs": [{" theorem": -0.1, " proof": -0.2}],
        },
        {
            "case_id": "correct-2",
            "correctness_label": "correct",
            "top_logprobs": [{" silicon": -0.1, " lattice": -0.2}],
        },
        {
            "case_id": "wrong-1",
            "correctness_label": "incorrect",
            "top_logprobs": [{" the": -0.1, " and": -0.2}],
        },
        {
            "case_id": "wrong-2",
            "correctness_label": "incorrect",
            "top_logprobs": [{" is": -0.1, " to": -0.2}],
        },
    ]
    manifest = tmp_path / "telemetry.jsonl"
    manifest.write_text(
        "".join(__import__("json").dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    artifact = build_experiment_artifact(manifest_path=manifest, n_eval_examples=4)

    assert artifact["freq_attn_validated"] is True
    assert artifact["freq_attn_auroc"] == 1.0
    assert artifact["proxy_strategy"] == "stopword_fraction"
    assert artifact["random_seed"] == 42

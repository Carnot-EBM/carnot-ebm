"""Tests for Experiment 808 — JEPA v22 Retrain: CPMI Hard-Negative Augmentation Fix.

Spec: REQ-LEARN-099, REQ-LEARN-100,
      SCENARIO-LEARN-146, SCENARIO-LEARN-147
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent.parent))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import experiment_808_jepa_v22_retrain as exp808  # noqa: E402


# ---------------------------------------------------------------------------
# Test: corpus merge produces augmentation_ratio > 1.0 (REQ-LEARN-099)
# ---------------------------------------------------------------------------


def test_corpus_merge_augmentation_ratio_above_one(tmp_path: Path) -> None:
    """load_v22_corpus merges FoVer + CPMI; aug_ratio > 1.0 when both present.

    Spec: REQ-LEARN-099, SCENARIO-LEARN-146
    """
    # Write a minimal FoVer multi-source corpus (5 pairs)
    multi_data = [
        {"step_text": "Step 1", "label": "correct", "source_domain": "gsm8k"},
        {"step_text": "Step 2", "label": "incorrect", "source_domain": "math500"},
        {"step_text": "Step 3", "label": "correct", "source_domain": "humaneval"},
        {"step_text": "Step 4", "label": "incorrect", "source_domain": "gsm8k"},
        {"step_text": "Step 5", "label": "correct", "source_domain": "gsm8k"},
    ]
    multi_path = tmp_path / "fover_v21_multi.json"
    multi_path.write_text(json.dumps(multi_data))

    # Write a minimal CPMI triples file (3 triples → 6 new training items)
    cpmi_data = [
        {
            "prefix_text": "Solve x+2=5.",
            "positive_step": "x = 3",
            "negative_step": "x = 8",
            "source_domain": "gsm8k",
            "cpmi_score": 0.05,
            "cpmi_mode": "ci_proxy",
        },
        {
            "prefix_text": "Compute 3!",
            "positive_step": "3! = 6",
            "negative_step": "3! = 9",
            "source_domain": "math500",
            "cpmi_score": 0.06,
            "cpmi_mode": "ci_proxy",
        },
        {
            "prefix_text": "Find area of 3x4 rectangle.",
            "positive_step": "area = 12",
            "negative_step": "area = 15",
            "source_domain": "humaneval",
            "cpmi_score": 0.04,
            "cpmi_mode": "ci_proxy",
        },
    ]
    cpmi_path = tmp_path / "cpmi_triples.json"
    cpmi_path.write_text(json.dumps(cpmi_data))

    (
        step_seqs,
        labels,
        weights,
        n_fover,
        n_cpmi,
        total,
        aug_ratio,
    ) = exp808.load_v22_corpus(multi_path, cpmi_path)

    # n_fover=5 pairs, n_cpmi=3 triples → 5 + 6 = 11 training items
    assert n_fover == 5, "n_fover_pairs should equal number of FoVer corpus entries"
    assert n_cpmi == 3, "n_cpmi_triples should equal number of CPMI triples"
    assert total == len(step_seqs) == 11, "total = 5 fover + 6 from 3 triples (pos+neg each)"
    assert aug_ratio > 1.0, (
        f"augmentation_ratio={aug_ratio} must be > 1.0 when CPMI triples are merged; "
        "ratio=1.0 was the Exp 799 failure mode"
    )
    assert len(labels) == total
    assert len(weights) == total


def test_corpus_merge_augmentation_ratio_formula(tmp_path: Path) -> None:
    """aug_ratio = total_training_items / n_fover_pairs, not n_cpmi/n_fover.

    Spec: REQ-LEARN-099 — ratio definition
    """
    # 4 FoVer pairs, 2 CPMI triples (each → 2 items = 4 CPMI items)
    multi_data = [
        {"step_text": f"Step {i}", "label": "correct", "source_domain": "gsm8k"} for i in range(4)
    ]
    cpmi_data = [
        {
            "prefix_text": f"Q{i}",
            "positive_step": f"pos{i}",
            "negative_step": f"neg{i}",
            "source_domain": "gsm8k",
            "cpmi_score": 0.05,
            "cpmi_mode": "ci_proxy",
        }
        for i in range(2)
    ]
    multi_path = tmp_path / "multi.json"
    cpmi_path = tmp_path / "cpmi.json"
    multi_path.write_text(json.dumps(multi_data))
    cpmi_path.write_text(json.dumps(cpmi_data))

    _, _, _, n_fover, _, total, aug_ratio = exp808.load_v22_corpus(multi_path, cpmi_path)

    expected_ratio = total / n_fover  # = (4 + 4) / 4 = 2.0
    assert aug_ratio == pytest.approx(expected_ratio, rel=1e-5)
    assert aug_ratio >= 1.5, "aug_ratio must be >= 1.5 to satisfy REQ-LEARN-099 min threshold"


def test_corpus_merge_fover_only_falls_to_synthetic(tmp_path: Path) -> None:
    """load_v22_corpus uses synthetic fallback when both real files are missing.

    Spec: REQ-LEARN-099 — graceful degradation path
    """
    missing_multi = tmp_path / "nonexistent_multi.json"
    missing_cpmi = tmp_path / "nonexistent_cpmi.json"

    _, labels, weights, n_fover, n_cpmi, total, aug_ratio = exp808.load_v22_corpus(
        missing_multi, missing_cpmi
    )

    assert total > 0, "Synthetic fallback must produce at least one training item"
    assert n_cpmi == 0, "n_cpmi_triples must be 0 when CPMI file is missing"
    assert len(labels) == total
    assert len(weights) == total


# ---------------------------------------------------------------------------
# Test: outcome-conditioned weighting maps source_domain to correct weight (REQ-LEARN-099)
# ---------------------------------------------------------------------------


def test_outcome_conditioned_weighting_by_domain(tmp_path: Path) -> None:
    """load_v22_corpus assigns DOMAIN_ACCURACY[source_domain] as base weight per pair.

    Spec: REQ-LEARN-099 — PROGRS outcome-conditioned weighting
    """
    multi_data = [
        {"step_text": "GSM8K step", "label": "correct", "source_domain": "gsm8k"},
        {"step_text": "MATH step", "label": "correct", "source_domain": "math500"},
        {"step_text": "HumanEval step", "label": "correct", "source_domain": "humaneval"},
    ]
    multi_path = tmp_path / "multi.json"
    multi_path.write_text(json.dumps(multi_data))
    cpmi_path = tmp_path / "cpmi_missing.json"  # intentionally absent

    _, _, weights, n_fover, _, _, _ = exp808.load_v22_corpus(multi_path, cpmi_path)

    # First 3 items come from FoVer corpus in order
    assert weights[0] == pytest.approx(exp808.DOMAIN_ACCURACY["gsm8k"], rel=1e-6)
    assert weights[1] == pytest.approx(exp808.DOMAIN_ACCURACY["math500"], rel=1e-6)
    assert weights[2] == pytest.approx(exp808.DOMAIN_ACCURACY["humaneval"], rel=1e-6)

    # humaneval weight > gsm8k weight (easier domain = higher accuracy = higher weight)
    assert weights[2] > weights[0], (
        "humaneval domain (accuracy=0.20) must have higher weight than "
        "gsm8k (accuracy=0.14) — harder domains are down-weighted per PROGRS"
    )


def test_cpmi_negative_weight_applied(tmp_path: Path) -> None:
    """CPMI negative steps are weighted at base_weight * CPMI_NEGATIVE_WEIGHT.

    Spec: REQ-LEARN-099 — synthetic hard negatives weighted down to avoid over-fitting
    """
    multi_path = tmp_path / "multi_missing.json"  # absent — only CPMI
    cpmi_data = [
        {
            "prefix_text": "Prefix",
            "positive_step": "correct step",
            "negative_step": "wrong step",
            "source_domain": "gsm8k",
            "cpmi_score": 0.05,
            "cpmi_mode": "ci_proxy",
        }
    ]
    cpmi_path = tmp_path / "cpmi.json"
    cpmi_path.write_text(json.dumps(cpmi_data))

    # With no FoVer corpus, load_v22_corpus uses synthetic fallback as base.
    # We test CPMI negative weight by checking that the negative item (label=1.0)
    # has a lower weight than the positive item (label=0.0) from the same triple.
    _, labels, weights, n_fover, n_cpmi, total, _ = exp808.load_v22_corpus(multi_path, cpmi_path)

    # After synthetic fallback is loaded, CPMI items are appended.
    # Find positive/negative pairs from CPMI (last 2 items appended after synthetic).
    synthetic_count = n_fover  # synthetic items came first
    cpmi_items_start = synthetic_count

    if total > cpmi_items_start + 1:
        # positive step weight = base_weight, negative step weight = base_weight * 0.7
        pos_weight = weights[cpmi_items_start]
        neg_weight = weights[cpmi_items_start + 1]
        assert neg_weight < pos_weight, (
            f"CPMI negative weight ({neg_weight}) must be < positive weight ({pos_weight}); "
            f"negative items are weighted at {exp808.CPMI_NEGATIVE_WEIGHT}× to reduce "
            "overfitting to synthetic hard negatives"
        )


# ---------------------------------------------------------------------------
# Test: blocked_wiring_miss path when check_cpmi_wiring fails (REQ-LEARN-100)
# ---------------------------------------------------------------------------


def test_blocked_wiring_miss_when_cpmi_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_experiment writes blocked_wiring_miss artifact when CPMI file absent.

    Spec: REQ-LEARN-100, SCENARIO-LEARN-147 — wiring guard blocks training when
    augmentation_ratio < 1.5 or triples file is missing.
    """
    # Point REPO_ROOT to tmp_path so no real data files are found.
    monkeypatch.setattr(exp808, "REPO_ROOT", tmp_path)

    # ExperimentTemplate needs setup() called; mock it out so we don't touch disk.
    monkeypatch.setattr(exp808.tmpl, "setup", lambda: None)
    monkeypatch.setattr(exp808.tmpl, "assert_deliverable_written", lambda: None)

    # build_result returns whatever dict we give it — capture the call args.
    captured: dict = {}

    def fake_build_result(payload: dict, *, status: str = "success") -> dict:
        captured.update(payload)
        captured["_status"] = status
        return {**payload, "status": status}

    monkeypatch.setattr(exp808.tmpl, "build_result", fake_build_result)

    result = exp808.run_experiment()

    assert result["honest_verdict"] == "blocked_wiring_miss", (
        "When CPMI triples file is missing, honest_verdict must be 'blocked_wiring_miss' "
        "to distinguish this block from other failure modes"
    )
    assert captured["_status"] == "blocked", (
        "Artifact status must be 'blocked' when wiring guard fails so the conductor "
        "knows training did not proceed"
    )
    assert result["tier35_deployed"] is False


def test_blocked_wiring_miss_when_ratio_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_experiment writes blocked_wiring_miss when aug_ratio < 1.5.

    Spec: REQ-LEARN-100, SCENARIO-LEARN-147 — aug_ratio=1.0 is the Exp 799 failure mode.
    The wiring guard must catch this BEFORE training begins.
    """
    # Write a CPMI triples file where all prefixes are identical → n_input_pairs=1
    # and n_triples=1, giving aug_ratio=1.0 < 1.5.
    cpmi_data = [
        {
            "prefix_text": "same_prefix",
            "positive_step": "pos",
            "negative_step": "neg",
            "source_domain": "gsm8k",
            "cpmi_score": 0.05,
            "cpmi_mode": "ci_proxy",
        }
    ]
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "experiment_798_cpmi_pairs_triples.json").write_text(json.dumps(cpmi_data))

    monkeypatch.setattr(exp808, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp808.tmpl, "setup", lambda: None)
    monkeypatch.setattr(exp808.tmpl, "assert_deliverable_written", lambda: None)

    captured: dict = {}

    def fake_build_result(payload: dict, *, status: str = "success") -> dict:
        captured.update(payload)
        captured["_status"] = status
        return {**payload, "status": status}

    monkeypatch.setattr(exp808.tmpl, "build_result", fake_build_result)

    result = exp808.run_experiment()

    assert result["honest_verdict"] == "blocked_wiring_miss", (
        "When augmentation_ratio < 1.5, wiring guard must set honest_verdict='blocked_wiring_miss'"
    )
    assert captured["_status"] == "blocked"


# ---------------------------------------------------------------------------
# Test: augmentation_ratio guard asserts > 1.0 during training (REQ-LEARN-099)
# ---------------------------------------------------------------------------


def test_load_v22_corpus_raises_when_cpmi_missing_and_only_fover(tmp_path: Path) -> None:
    """load_v22_corpus augmentation_ratio with CPMI missing and FoVer present = 1.0.

    This checks that aug_ratio=1.0 is correctly computed when CPMI is absent,
    so run_experiment's assert augmentation_ratio > 1.0 would catch it.

    Spec: REQ-LEARN-099 — assertion guard
    """
    multi_data = [{"step_text": "Step A", "label": "correct", "source_domain": "gsm8k"}]
    multi_path = tmp_path / "multi.json"
    multi_path.write_text(json.dumps(multi_data))
    cpmi_path = tmp_path / "cpmi_absent.json"  # missing

    _, _, _, n_fover, n_cpmi, total, aug_ratio = exp808.load_v22_corpus(multi_path, cpmi_path)

    assert n_cpmi == 0
    assert n_fover == 1
    assert total == n_fover
    assert aug_ratio == pytest.approx(1.0, rel=1e-6), (
        "Without CPMI, aug_ratio must equal 1.0 — this is the failure mode that "
        "blocked Exp 799 from testing the CPMI hypothesis"
    )

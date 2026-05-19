import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "python"))

import experiment_2546_ensemble_v7b as exp2546
from carnot.verify.group_conditional_calibration import compute_p_values, fisher_combine


RESULTS_DIR = REPO_ROOT / "results"


def test_req_verify_2546_group_architecture_keeps_abc_and_adds_d() -> None:
    """REQ-VERIFY-2546-1/2: A/B/C stay unchanged and Tier 0r moves to Group D."""

    rows = exp2546.load_manifest_rows(RESULTS_DIR)
    groups = exp2546.build_score_groups(RESULTS_DIR, rows)

    assert list(groups) == ["A", "B", "C", "D"]
    assert [groups[name].shape[1] for name in ["A", "B", "C", "D"]] == [3, 3, 3, 1]
    assert exp2546.GROUP_LABELS["D"] == "Group D (proof-path)"
    assert exp2546.GROUP_SCORE_SOURCES["D"] == ("Tier0rVerifier.score",)
    assert all("Tier0r" not in source for source in exp2546.GROUP_SCORE_SOURCES["C"])


def test_req_verify_2546_run_experiment_resolves_regression() -> None:
    """REQ-VERIFY-2546-3/4/5: exp2546 reports five-seed v7b AUROC and gate flags."""

    deliverable = exp2546.run_experiment(results_dir=RESULTS_DIR, write=False)

    required_fields = {
        "honest_verdict",
        "ensemble_v7b_auroc",
        "ensemble_v7b_auroc_std",
        "ensemble_v6_baseline",
        "ensemble_v7_regression",
        "regression_resolved",
        "tier0r_group_assignment",
        "n_seeds",
        "preconditions_checked",
        "duration_s",
        "random_seed",
    }
    assert required_fields <= set(deliverable)
    assert deliverable["honest_verdict"].startswith("complete:")
    assert deliverable["ensemble_v7b_auroc"] == pytest.approx(0.9857142857142858)
    assert deliverable["ensemble_v7b_auroc_std"] == pytest.approx(0.01749635530559415)
    assert deliverable["ensemble_v6_baseline"] == pytest.approx(0.9750)
    assert deliverable["ensemble_v7_regression"] == pytest.approx(0.9607)
    assert deliverable["regression_resolved"] is True
    assert deliverable["tier0r_group_assignment"] == "Group D (proof-path)"
    assert deliverable["n_seeds"] == 5
    assert deliverable["random_seed"] == 42
    assert deliverable["n_verifiers"] == 10
    assert deliverable["n_groups"] == 4
    assert deliverable["acceptance_gates"]["ensemble_v7b_auroc >= 0.970"] is True


def test_req_verify_2546_abc_seed_means_match_exp2498() -> None:
    """SCENARIO-VERIFY-2546: A/B/C calibration means match the exp2498 baseline."""

    deliverable = exp2546.run_experiment(results_dir=RESULTS_DIR, write=False)
    baseline = json.loads((RESULTS_DIR / "experiment_2498_auroc_adversarial_v2_group_cond.json").read_text())

    for got, expected in zip(deliverable["results_by_seed"], baseline["results_by_seed"], strict=True):
        assert got["seed"] == expected["seed"]
        assert got["mean_cal_A"] == pytest.approx(expected["mean_cal_A"])
        assert got["mean_cal_B"] == pytest.approx(expected["mean_cal_B"])
        assert got["mean_cal_C"] == pytest.approx(expected["mean_cal_C"])
        assert "mean_cal_D" in got


def test_req_verify_2546_group_calibration_helpers() -> None:
    """REQ-VERIFY-2546-3: helper p-values and Fisher scores preserve ranking."""

    x_cal = np.array([[0.1], [0.3], [0.5], [0.7]])
    x_test = np.array([[0.2], [0.6]])
    p_values = compute_p_values(x_cal, x_test)

    assert np.isclose(p_values[0, 0], 0.6)
    assert np.isclose(p_values[1, 0], 0.2)

    combined = fisher_combine(np.array([[0.01, 0.01], [0.5, 0.5]]))
    assert combined.shape == (2,)
    assert combined[0] > combined[1]

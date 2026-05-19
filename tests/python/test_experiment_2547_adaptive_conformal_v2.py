import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "python"))

import experiment_2547_adaptive_conformal_v2 as exp2547
from carnot.verify.adaptive_conformal_calibration import (
    PROMPT_TYPES,
    compute_acse_entropy_proxy,
    prompt_type_classifier,
)


RESULTS_DIR = REPO_ROOT / "results"


def test_req_verify_2547_prompt_type_classifier_returns_four_stable_categories() -> None:
    """REQ-VERIFY-2547-1: prompt classifier returns one of the four contract labels."""

    assert (
        prompt_type_classifier("Verify claim: Paris is in France. Return 1 if true.") == "factual"
    )
    assert (
        prompt_type_classifier("Constraint x=2 satisfies x+3=5. Return 1 if true.") == "reasoning"
    )
    assert prompt_type_classifier("Write a short story about calibration.") == "creative"
    assert prompt_type_classifier("Implement a Python function that parses JSON.") == "code"
    assert set(PROMPT_TYPES) == {"factual", "reasoning", "creative", "code"}


def test_req_verify_2547_acse_proxy_increases_with_logprob_and_verifier_dispersion() -> None:
    """REQ-VERIFY-2547-3: ACSE proxy uses top-k logprobs and verifier-score variance."""

    low_entropy = compute_acse_entropy_proxy(
        top_logprobs=[{"yes": -0.2, "no": -0.21, "maybe": -0.22}],
        verifier_scores=[0.3, 0.31, 0.32],
        top_k=3,
    )
    high_entropy = compute_acse_entropy_proxy(
        top_logprobs=[{"yes": -0.01, "no": -4.0, "maybe": -8.0}],
        verifier_scores=[0.0, 0.5, 1.0],
        top_k=3,
    )

    assert high_entropy > low_entropy
    assert low_entropy >= 0.0


def test_req_verify_2547_run_experiment_writes_non_regressing_artifact() -> None:
    """REQ-VERIFY-2547-2/3/4/5: exp2547 reports five-seed adaptive AUROC and gate."""

    deliverable = exp2547.run_experiment(results_dir=RESULTS_DIR, write=False)

    required_fields = {
        "honest_verdict",
        "adaptive_conformal_auroc",
        "ensemble_v7b_baseline",
        "prompt_type_distribution",
        "acse_component_used",
        "n_seeds",
        "preconditions_checked",
        "duration_s",
        "random_seed",
    }
    assert required_fields <= set(deliverable)
    assert deliverable["honest_verdict"].startswith("complete:")
    assert deliverable["adaptive_conformal_auroc"] >= deliverable["ensemble_v7b_baseline"]
    assert deliverable["ensemble_v7b_baseline"] == pytest.approx(0.9857142857142858)
    assert deliverable["adaptive_conformal_auroc"] == pytest.approx(0.9928571428571429)
    assert deliverable["acse_component_used"] is True
    assert deliverable["n_seeds"] == 5
    assert deliverable["random_seed"] == 42
    assert sum(deliverable["prompt_type_distribution"].values()) == 36
    assert max(deliverable["prompt_type_distribution"].values()) < 36
    assert (
        deliverable["acceptance_gates"]["adaptive_conformal_auroc >= ensemble_v7b_baseline"] is True
    )


def test_req_verify_2547_blocks_when_ensemble_v7b_gate_fails(tmp_path: Path) -> None:
    """REQ-VERIFY-2547-2: low or missing exp2546 AUROC blocks without fake metrics."""

    (tmp_path / "experiment_2546_ensemble_v7b.json").write_text(
        json.dumps({"ensemble_v7b_auroc": 0.969}) + "\n",
        encoding="utf-8",
    )

    deliverable = exp2547.run_experiment(results_dir=tmp_path, write=False)

    assert deliverable["honest_verdict"] == "blocked_ensemble_v7b_below_threshold"
    assert deliverable["adaptive_conformal_auroc"] is None
    assert deliverable["ensemble_v7b_baseline"] == pytest.approx(0.969)
    assert deliverable["acse_component_used"] is False

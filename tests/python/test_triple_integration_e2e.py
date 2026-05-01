"""E2E tests for the 3-tier verification cascade (Exp 1073).

Validates that the full 4-tier cascade (ThinkPRM → SpilledEnergy → SC-Energy →
Ising/GS-KAN) runs correctly on 50 FoVer corpus items and satisfies the three
required correctness criteria.

Spec: REQ-VERIFY-088, REQ-VERIFY-111, REQ-VERIFY-112
SCENARIO-VERIFY-116, SCENARIO-VERIFY-146, SCENARIO-VERIFY-147
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULT_PATH = REPO_ROOT / "results" / "experiment_1073_triple_integration_e2e_v9.json"
CORPUS_PATH = REPO_ROOT / "data" / "fover_corpus_v4.json"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cascade_result() -> dict:
    """Load the pre-computed cascade artifact if present, else run it fresh.

    Running fresh ensures CI can execute this suite without requiring a prior
    conductor pass. The run takes ~10 s on CPU (JAX warmup + GS-KAN training).
    """
    if RESULT_PATH.exists():
        with open(RESULT_PATH) as f:
            return json.load(f)

    # Import and run the experiment inline so the test suite is self-contained.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "exp1073",
        REPO_ROOT / "scripts" / "experiment_1073_triple_integration_e2e_v9.py",
    )
    mod = importlib.util.load_from_spec(spec)  # type: ignore[attr-defined]
    spec.loader.exec_module(mod)
    mod.main()

    with open(RESULT_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    """Load the first 50 FoVer corpus items (the same set the experiment uses)."""
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    return corpus[:50]


# ---------------------------------------------------------------------------
# Test 1: cascade_runs_without_error_on_50_questions
# REQ-VERIFY-088: The pipeline must complete on all supplied inputs.
# ---------------------------------------------------------------------------


def test_cascade_runs_without_error_on_50_questions(cascade_result: dict) -> None:
    """All 50 questions must complete without errors and the cascade must confirm E2E.

    Spec: REQ-VERIFY-088, SCENARIO-VERIFY-116
    """
    assert cascade_result["n_questions_run"] == 50, (
        f"Expected 50 questions run, got {cascade_result['n_questions_run']}"
    )
    assert cascade_result["cascade_e2e_confirmed"] is True, (
        "cascade_e2e_confirmed must be True — all 50 questions must complete without errors"
    )
    assert cascade_result.get("errors", []) == [], (
        f"Unexpected errors in cascade run: {cascade_result.get('errors', [])}"
    )


# ---------------------------------------------------------------------------
# Test 2: all_tier_skip_rates_nonzero
# REQ-VERIFY-111: Each tier must serve as a meaningful gate for at least some items.
# ---------------------------------------------------------------------------


def test_all_tier_skip_rates_nonzero(cascade_result: dict) -> None:
    """At least one question must exit at each of the four tiers.

    If a tier has zero exits it is not functioning as a gate — either the
    threshold is mis-tuned or the tier implementation is broken.

    Spec: REQ-VERIFY-111, REQ-VERIFY-112
    """
    tier_counts = {
        "tier_0a": cascade_result["tier_0a_skips"],
        "tier_0b": cascade_result["tier_0b_skips"],
        "tier_2": cascade_result["tier_2_skips"],
        "tier_3": cascade_result["tier_3_skips"],
    }
    for tier_name, count in tier_counts.items():
        assert count > 0, (
            f"{tier_name} has zero exits — tier is not functioning as a gate. "
            f"All tier counts: {tier_counts}"
        )
    assert cascade_result["all_tier_skip_rates_nonzero"] is True


# ---------------------------------------------------------------------------
# Test 3: incorrect_questions_have_higher_energy_than_correct
# REQ-VERIFY-112: The Ising energy must be discriminative between correct / incorrect.
# ---------------------------------------------------------------------------


def test_incorrect_questions_have_higher_energy_than_correct(
    cascade_result: dict,
    questions: list[dict],
) -> None:
    """Mean GS-KAN energy of incorrect items must exceed mean energy of correct items.

    The GS-KAN model (Tier 3) is trained on the distribution of text-feature
    vectors across all 50 items. Incorrect FoVer items are outliers in this
    feature space (different text-length, LaTeX density, equation-count profiles),
    so they receive higher energy on average. This validates the energy's
    discriminative power at Tier 3.

    Spec: REQ-VERIFY-112, SCENARIO-VERIFY-147
    """
    mean_correct = cascade_result["mean_correct_energy"]
    mean_incorrect = cascade_result["mean_incorrect_energy"]

    assert mean_incorrect > mean_correct, (
        f"Expected mean_incorrect_energy ({mean_incorrect:.4f}) > "
        f"mean_correct_energy ({mean_correct:.4f}). "
        "The GS-KAN Ising tier must assign higher energy to incorrect reasoning steps."
    )
    assert cascade_result["incorrect_energy_gt_correct"] is True

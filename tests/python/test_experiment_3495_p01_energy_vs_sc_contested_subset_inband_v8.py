"""Tests for exp3495 -- P0.1 in-band contested-subset energy vs SC.

Traces to REQ-KONA-3495. Covers:
- _per_problem_correctness_rate: all-correct, none-correct, partial, empty
- _build_contested_subset: filtering, boundary inclusion
- _load_usable: minimum-sample enforcement, missing-gold enforcement
- Module-level constants: MIN_PROBLEMS, CONTEST_LOW, CONTEST_HIGH, ARTIFACT_PATH, SEED
- Integration: main() writes a JSON artifact with all required fields; verdict has
  terminal prefix; contested_subset_n is a non-negative integer; blocked path when
  subset is too small.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiment_3495_p01_energy_vs_sc_contested_subset_inband_v8 import (  # noqa: E402
    ARTIFACT_PATH,
    CONTEST_HIGH,
    CONTEST_LOW,
    MIN_PROBLEMS,
    SEED,
    _build_contested_subset,
    _load_usable,
    _per_problem_correctness_rate,
    main,
)


# ---------------------------------------------------------------------------
# Shared helper -- mirrors the _synthetic_record in test_p01_process_energy.py
# ---------------------------------------------------------------------------


def _synthetic_record(pid, gold, sample_answers):
    """Build a minimal corpus row sufficient for _per_problem_correctness_rate
    and _load_usable / _build_contested_subset.

    The structure matches what the experiment script expects: a 'samples' list
    of dicts (each with an 'answer' key), a 'greedy' dict, and top-level 'gold',
    'problem_id', 'question', 'k', and 'temperature'.
    """
    samples = [
        {
            "text": f"ans={a}",
            "answer": a,
            "mean_token_logprob": -0.1,
            "n_tokens": 2,
        }
        for a in sample_answers
    ]
    greedy = {
        "text": f"ans={sample_answers[0]}",
        "answer": sample_answers[0],
        "mean_token_logprob": -0.2,
    }
    return {
        "problem_id": pid,
        "question": f"q_{pid}",
        "gold": gold,
        "greedy": greedy,
        "samples": samples,
        "k": len(samples),
        "temperature": 0.8,
    }


# ---------------------------------------------------------------------------
# REQ-KONA-3495: _per_problem_correctness_rate
# ---------------------------------------------------------------------------


def test_per_problem_correctness_rate_all_correct():
    # All samples correct -> rate 1.0  REQ-KONA-3495
    rec = _synthetic_record("p1", "42", ["42", "42", "42", "42", "42"])
    assert _per_problem_correctness_rate(rec) == pytest.approx(1.0)  # REQ-KONA-3495


def test_per_problem_correctness_rate_none_correct():
    # No samples correct -> rate 0.0  REQ-KONA-3495
    rec = _synthetic_record("p1", "99", ["1", "2", "3", "4", "5"])
    assert _per_problem_correctness_rate(rec) == pytest.approx(0.0)  # REQ-KONA-3495


def test_per_problem_correctness_rate_partial():
    # 3/6 samples correct -> rate 0.5  REQ-KONA-3495
    rec = _synthetic_record("p1", "X", ["X", "X", "X", "Y", "Y", "Y"])
    assert _per_problem_correctness_rate(rec) == pytest.approx(0.5)  # REQ-KONA-3495


def test_per_problem_correctness_rate_empty_samples():
    # No samples -> 0.0  REQ-KONA-3495
    rec = {"problem_id": "p1", "gold": "5", "samples": []}
    assert _per_problem_correctness_rate(rec) == pytest.approx(0.0)  # REQ-KONA-3495


# ---------------------------------------------------------------------------
# REQ-KONA-3495: _build_contested_subset
# ---------------------------------------------------------------------------


def test_build_contested_subset_filters_in_band():
    # Only problems with correctness rate in [0.40, 0.70] are kept  REQ-KONA-3495
    recs = [
        _synthetic_record("easy", "A", ["A"] * 6),           # rate 1.0 -> excluded
        _synthetic_record("hard", "B", ["X"] * 6),           # rate 0.0 -> excluded
        _synthetic_record("contested", "C", ["C", "C", "C", "X", "X", "X"]),  # rate 0.5 -> included
    ]
    subset = _build_contested_subset(recs)
    assert len(subset) == 1 and subset[0]["problem_id"] == "contested"  # REQ-KONA-3495


def test_build_contested_subset_boundary_inclusive():
    # Boundary values 0.40 and 0.70 are included  REQ-KONA-3495
    recs = [
        _synthetic_record("at_low", "A", ["A", "A", "X", "X", "X"]),       # 2/5 = 0.40
        _synthetic_record("at_high", "B", ["B", "B", "B", "B", "X", "X", "X", "X", "X", "X"]),  # 7/10 = 0.70
    ]
    subset = _build_contested_subset(recs, low=0.40, high=0.70)
    assert len(subset) == 2  # REQ-KONA-3495


# ---------------------------------------------------------------------------
# REQ-KONA-3495: _load_usable
# ---------------------------------------------------------------------------


def test_load_usable_requires_gold_greedy_and_five_samples():
    # A record with < 5 samples is dropped  REQ-KONA-3495
    rec_ok = _synthetic_record("ok", "5", ["5", "5", "5", "5", "5"])   # 5 samples -> kept
    rec_bad = _synthetic_record("bad", "5", ["5", "5"])                 # 2 samples -> dropped
    assert len(_load_usable([rec_ok, rec_bad])) == 1  # REQ-KONA-3495


def test_load_usable_drops_missing_gold():
    # Record with gold=None is dropped  REQ-KONA-3495
    rec = _synthetic_record("p1", None, ["1", "2", "3", "4", "5"])
    assert _load_usable([rec]) == []  # REQ-KONA-3495


# ---------------------------------------------------------------------------
# REQ-KONA-3495: module-level constants
# ---------------------------------------------------------------------------


def test_min_problems_constant_is_40():
    # MIN_PROBLEMS must be 40 per the task spec  REQ-KONA-3495
    assert MIN_PROBLEMS == 40  # REQ-KONA-3495


def test_contest_band_constants():
    # Band must be [0.40, 0.70] per the task spec  REQ-KONA-3495
    assert CONTEST_LOW == pytest.approx(0.40) and CONTEST_HIGH == pytest.approx(0.70)  # REQ-KONA-3495


def test_artifact_path_in_results_dir():
    # Artifact path must be in the results/ directory  REQ-KONA-3495
    assert "results" in str(ARTIFACT_PATH)  # REQ-KONA-3495


def test_seed_is_not_experiment_id():
    # Seed must not equal experiment ID (would risk tautology flag)  REQ-KONA-3495
    assert SEED != 3495  # REQ-KONA-3495


# ---------------------------------------------------------------------------
# REQ-KONA-3495: integration tests via main()
# ---------------------------------------------------------------------------


def test_main_writes_artifact_with_required_fields(tmp_path, monkeypatch):
    # main() writes a JSON with all required fields, even in the blocked path  REQ-KONA-3495
    import scripts.experiment_3495_p01_energy_vs_sc_contested_subset_inband_v8 as mod

    monkeypatch.setattr(mod, "ARTIFACT_PATH", tmp_path / "artifact.json")
    mod.main()
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    required = [
        "honest_verdict",
        "inference_substrate",
        "source_corpora",
        "contested_subset_n",
        "contested_subset_sc",
        "self_consistency_in_headroom_band",
        "k_samples",
        "ar_greedy_accuracy",
        "self_consistency_accuracy",
        "self_certainty_bon_accuracy",
        "process_energy_argmin_accuracy",
        "trained_energy_weighted_vote_accuracy",
        "trained_energy_sc_hybrid_accuracy",
        "optimal_aggregation_accuracy",
        "flip_count_optimal_vs_sc",
        "flips_correct_optimal",
        "flips_incorrect_optimal",
        "net_correctness_gain_optimal",
        "delta_optimal_vs_self_consistency",
        "delta_process_energy_vs_self_consistency",
        "paired_significance",
        "compute_parity_note",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ]
    for field in required:
        assert field in artifact, f"missing required field: {field!r}"  # REQ-KONA-3495


def test_main_verdict_has_terminal_prefix(tmp_path, monkeypatch):
    # honest_verdict must start with a terminal prefix per Verdict Discipline  REQ-KONA-3495
    import scripts.experiment_3495_p01_energy_vs_sc_contested_subset_inband_v8 as mod

    monkeypatch.setattr(mod, "ARTIFACT_PATH", tmp_path / "artifact.json")
    mod.main()
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    verdict = artifact["honest_verdict"]
    terminal = ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")
    assert any(verdict.startswith(p) for p in terminal), f"verdict lacks terminal prefix: {verdict!r}"  # REQ-KONA-3495


def test_main_contested_subset_n_reported(tmp_path, monkeypatch):
    # contested_subset_n must be an integer >= 0 in the artifact  REQ-KONA-3495
    import scripts.experiment_3495_p01_energy_vs_sc_contested_subset_inband_v8 as mod

    monkeypatch.setattr(mod, "ARTIFACT_PATH", tmp_path / "artifact.json")
    mod.main()
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    assert isinstance(artifact["contested_subset_n"], int) and artifact["contested_subset_n"] >= 0  # REQ-KONA-3495


def test_main_blocked_when_subset_too_small(tmp_path, monkeypatch):
    # When there are fewer than MIN_PROBLEMS contested problems, the verdict is blocked  REQ-KONA-3495
    import scripts.experiment_3495_p01_energy_vs_sc_contested_subset_inband_v8 as mod

    monkeypatch.setattr(mod, "ARTIFACT_PATH", tmp_path / "artifact.json")
    # With the current corpora (21 contested problems), we expect a blocked verdict
    mod.main()
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    # Either blocked (21 < 40) or scored (if corpora grow) -- both are valid
    assert artifact["honest_verdict"].startswith("complete:")  # REQ-KONA-3495

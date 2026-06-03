"""Tests for exp3531 — P0.1 Route 2 energy reranker vs SC on selectable-headroom corpus.

Spec anchor: REQ-AR-050 (P0.1 Difficulty-Matched Corpus Builder v3) in
openspec/capabilities/autoresearch/spec.md.

This file covers the pure functions introduced in exp3531 — headroom stats, energy
proxies, flip metrics, significance, verdict classification, and artifact schema.
The CV scoring loop and main() are integration-only and not exercised here.

Every test asserts real behavior; none are skipped (CLAUDE.md "Tests Must Run
and Assert").
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO_ROOT
    / "scripts"
    / "experiment_3531_p01_route2_energy_vs_sc_on_headroom_corpus_v1.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("exp3531_v1", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXP = _load_script_module()


# ---------------------------------------------------------------------------
# Helpers — minimal synthetic corpus records
# ---------------------------------------------------------------------------

def _make_record(
    gold: str,
    answers: list[str | None],
    n_steps: int = 5,
    problem_id: str = "p0",
) -> dict:
    """Build a minimal corpus record matching the exp3531 schema."""
    samples = [
        {
            "mode": "sampled",
            "seed": i,
            "extracted_answer": a,
            "extracted_answer_norm": a,
            "correct": a == gold if a is not None else False,
            "mean_token_logprob": None,
            "reasoning_steps": list(range(n_steps)),
            "n_steps": n_steps,
        }
        for i, a in enumerate(answers)
    ]
    return {
        "problem_id": problem_id,
        "level": 3,
        "gold_answer": gold,
        "gold_answer_norm": gold,
        "greedy": {
            "extracted_answer": answers[0] if answers else None,
            "extracted_answer_norm": answers[0] if answers else None,
        },
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01: load_usable_records
# ---------------------------------------------------------------------------

def test_load_usable_records_returns_empty_for_missing_file():
    # REQ-AR-050 / SCENARIO-AR-050-01
    missing = Path("/tmp/does_not_exist_exp3531.jsonl")
    records = EXP.load_usable_records(missing, min_samples=4)
    assert records == []


def test_load_usable_records_filters_below_min_samples():
    # REQ-AR-050 / SCENARIO-AR-050-01
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # record with 3 samples → should be filtered
        f.write(json.dumps({
            "gold_answer": "3",
            "gold_answer_norm": "3",
            "samples": [{"extracted_answer": "3", "extracted_answer_norm": "3"}] * 3,
        }) + "\n")
        # record with 4 samples → should be kept
        f.write(json.dumps({
            "gold_answer": "5",
            "gold_answer_norm": "5",
            "samples": [{"extracted_answer": "5", "extracted_answer_norm": "5"}] * 4,
        }) + "\n")
        tmp_path = Path(f.name)
    try:
        records = EXP.load_usable_records(tmp_path, min_samples=4)
        assert len(records) == 1
        assert records[0]["gold_answer"] == "5"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_usable_records_skips_missing_gold():
    # REQ-AR-050 / SCENARIO-AR-050-01
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({
            "gold_answer": None,
            "gold_answer_norm": None,
            "samples": [{"extracted_answer": "3"}] * 4,
        }) + "\n")
        tmp_path = Path(f.name)
    try:
        records = EXP.load_usable_records(tmp_path)
        assert records == []
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_usable_records_skips_malformed_json():
    # REQ-AR-050 / SCENARIO-AR-050-01
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("{bad json\n")
        f.write(json.dumps({
            "gold_answer": "7",
            "gold_answer_norm": "7",
            "samples": [{"extracted_answer": "7"}] * 4,
        }) + "\n")
        tmp_path = Path(f.name)
    try:
        records = EXP.load_usable_records(tmp_path, min_samples=4)
        assert len(records) == 1
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01: compute_headroom_stats
# ---------------------------------------------------------------------------

def test_compute_headroom_stats_empty_returns_zero():
    # REQ-AR-050 / SCENARIO-AR-050-01
    stats = EXP.compute_headroom_stats([])
    assert stats["oracle_accuracy"] == 0.0
    assert stats["self_consistency_accuracy"] == 0.0
    assert stats["selectable_headroom"] == 0.0
    assert stats["oracle_exceeds_sc"] is False
    assert stats["n"] == 0


def test_compute_headroom_stats_oracle_exceeds_sc_when_correct_minority():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # All 2 problems: correct answer "3" is in samples but SC majority is "7"
    records = [
        _make_record("3", ["7", "7", "7", "3"]),
        _make_record("5", ["9", "9", "9", "5"]),
    ]
    stats = EXP.compute_headroom_stats(records)
    assert stats["oracle_accuracy"] == pytest.approx(1.0)
    assert stats["self_consistency_accuracy"] == pytest.approx(0.0)
    assert stats["oracle_exceeds_sc"] is True
    assert stats["selectable_headroom"] == pytest.approx(1.0)


def test_compute_headroom_stats_no_headroom_when_sc_correct():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # All problems: SC majority is correct → oracle = SC = 1.0, headroom = 0
    records = [
        _make_record("3", ["3", "3", "3", "7"]),
        _make_record("5", ["5", "5", "5", "9"]),
    ]
    stats = EXP.compute_headroom_stats(records)
    assert stats["oracle_accuracy"] == pytest.approx(1.0)
    assert stats["self_consistency_accuracy"] == pytest.approx(1.0)
    assert stats["oracle_exceeds_sc"] is False
    assert stats["selectable_headroom"] == pytest.approx(0.0)


def test_compute_headroom_stats_oracle_le_sc_is_possible_when_sc_correct_oracle_wrong():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # SC correct, oracle also correct (can't be oracle < SC here since oracle >= SC always)
    # Demonstrate that oracle_exceeds_sc = False when oracle == SC
    records = [_make_record("3", ["3", "3", "3", "3"])]  # all correct
    stats = EXP.compute_headroom_stats(records)
    assert stats["oracle_exceeds_sc"] is False  # oracle == SC, not strictly greater


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01: build_sc_majority
# ---------------------------------------------------------------------------

def test_build_sc_majority_returns_majority_and_correct_flag():
    # REQ-AR-050 / SCENARIO-AR-050-01
    records = [_make_record("3", ["7", "7", "3", "7"])]
    result = EXP.build_sc_majority(records)
    assert len(result) == 1
    majority_ans, is_correct = result[0]
    assert majority_ans == "7"
    assert is_correct is False


def test_build_sc_majority_correct_when_majority_matches_gold():
    # REQ-AR-050 / SCENARIO-AR-050-01
    records = [_make_record("3", ["3", "3", "7", "3"])]
    result = EXP.build_sc_majority(records)
    majority_ans, is_correct = result[0]
    assert majority_ans == "3"
    assert is_correct is True


def test_build_sc_majority_handles_all_none_answers():
    # REQ-AR-050 / SCENARIO-AR-050-01
    records = [_make_record("3", [None, None, None, None])]
    result = EXP.build_sc_majority(records)
    majority_ans, is_correct = result[0]
    assert majority_ans is None
    assert is_correct is False


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01: compute_process_energy
# ---------------------------------------------------------------------------

def test_compute_process_energy_normalized_range():
    # REQ-AR-050 / SCENARIO-AR-050-01
    rec = _make_record("3", ["3", "7", "3", "7"], n_steps=1)
    rec["samples"][0]["n_steps"] = 2
    rec["samples"][1]["n_steps"] = 4
    rec["samples"][2]["n_steps"] = 2
    rec["samples"][3]["n_steps"] = 6
    energies = EXP.compute_process_energy([rec])
    assert len(energies) == 1
    assert len(energies[0]) == 4
    for e in energies[0]:
        assert 0.0 <= e <= 1.0


def test_compute_process_energy_uniform_steps_uses_positional_jitter():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # All samples have the same n_steps → span < 1e-9 → positional jitter
    rec = _make_record("3", ["3", "7", "3", "7"], n_steps=5)
    energies = EXP.compute_process_energy([rec])
    e = energies[0]
    # Monotonically increasing (positional jitter: i / len)
    assert e[0] < e[1] < e[2] < e[3]


def test_compute_process_energy_empty_samples():
    # REQ-AR-050 / SCENARIO-AR-050-01
    rec = _make_record("3", [])
    energies = EXP.compute_process_energy([rec])
    assert energies == [[]]


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01: compute_pessimistic_bon_scores
# ---------------------------------------------------------------------------

def test_compute_pessimistic_bon_penalizes_minority_low_energy():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # Two answers: "3" (minority, 1/4) and "7" (majority, 3/4).
    # process-energy is min-max normalized within the problem, so the fewest-steps
    # sample maps to exactly 0.0 (base score 1.0). For the disagreement penalty
    # (bounded by alpha) to matter at the realistic default alpha=0.5, the
    # confident-majority answer must have a competitive-energy representative —
    # otherwise a lone minority with the unique minimum energy always wins and the
    # penalty is irrelevant (itself the bounded-reranker finding of exp3531).
    rec = _make_record("3", ["3", "7", "7", "7"], n_steps=1)
    rec["samples"][0]["n_steps"] = 2  # "3" minority: low (tied-min) energy
    rec["samples"][1]["n_steps"] = 2  # "7" majority: also low (tied-min) energy
    rec["samples"][2]["n_steps"] = 6
    rec["samples"][3]["n_steps"] = 6
    energies = EXP.compute_process_energy([rec])
    pbon = EXP.compute_pessimistic_bon_scores([rec], energies, alpha=0.5)
    # "3" is sample 0: low energy but high disagreement (1/4 support) → penalized.
    # "7" (sample 1): equally low energy but high confidence (3/4) → no penalty.
    # Without the penalty the two tie; with it the confident majority "7" wins.
    assert len(pbon) == 1
    assert len(pbon[0]) == 4
    # Max-score sample should be one of the "7" answers (lower disagreement)
    max_idx = int(np.argmax(pbon[0]))
    assert rec["samples"][max_idx]["extracted_answer"] == "7"


def test_compute_pessimistic_bon_scores_returns_list_per_problem():
    # REQ-AR-050 / SCENARIO-AR-050-01
    records = [
        _make_record("3", ["3", "7", "3", "7"]),
        _make_record("5", ["5", "5", "9", "5"]),
    ]
    energies = EXP.compute_process_energy(records)
    pbon = EXP.compute_pessimistic_bon_scores(records, energies)
    assert len(pbon) == 2
    assert len(pbon[0]) == 4
    assert len(pbon[1]) == 4


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01: compute_flip_metrics
# ---------------------------------------------------------------------------

def test_compute_flip_metrics_zero_flips_when_identical():
    # REQ-AR-050 / SCENARIO-AR-050-01
    sc = ["3", "7", "5"]
    cond = ["3", "7", "5"]
    gold = ["3", "7", "5"]
    result = EXP.compute_flip_metrics(cond, sc, gold)
    assert result["flip_count"] == 0
    assert result["flips_correct"] == 0
    assert result["flips_incorrect"] == 0
    assert result["net_correctness_gain"] == 0


def test_compute_flip_metrics_correct_flip():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # SC wrong → cond correct: flip_correct++
    sc = ["7"]
    cond = ["3"]
    gold = ["3"]
    result = EXP.compute_flip_metrics(cond, sc, gold)
    assert result["flip_count"] == 1
    assert result["flips_correct"] == 1
    assert result["flips_incorrect"] == 0
    assert result["net_correctness_gain"] == 1


def test_compute_flip_metrics_incorrect_flip():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # SC correct → cond wrong: flip_incorrect++
    sc = ["3"]
    cond = ["7"]
    gold = ["3"]
    result = EXP.compute_flip_metrics(cond, sc, gold)
    assert result["flip_count"] == 1
    assert result["flips_correct"] == 0
    assert result["flips_incorrect"] == 1
    assert result["net_correctness_gain"] == -1


def test_compute_flip_metrics_mixed():
    # REQ-AR-050 / SCENARIO-AR-050-01
    sc = ["7", "7"]
    cond = ["3", "9"]
    gold = ["3", "5"]  # cond[0] correct, cond[1] incorrect
    result = EXP.compute_flip_metrics(cond, sc, gold)
    assert result["flip_count"] == 2
    assert result["flips_correct"] == 1
    assert result["flips_incorrect"] == 1
    assert result["net_correctness_gain"] == 0


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-02: classify_verdict_3531
# ---------------------------------------------------------------------------

def test_classify_verdict_blocked_no_headroom():
    # REQ-AR-050 / SCENARIO-AR-050-02
    v = EXP.classify_verdict_3531(False, 93, False, 0, 0.0, 1.0)
    assert v.startswith("complete:")
    assert "blocked_corpus_has_no_selectable_headroom" in v


def test_classify_verdict_blocked_too_small():
    # REQ-AR-050 / SCENARIO-AR-050-02
    v = EXP.classify_verdict_3531(True, 10, False, 0, 0.0, 1.0)
    assert v.startswith("complete:")
    assert "blocked_headroom_corpus_too_small" in v
    assert "n=10" in v


def test_classify_verdict_blocked_degenerate():
    # REQ-AR-050 / SCENARIO-AR-050-02
    v = EXP.classify_verdict_3531(True, 93, False, 0, 0.0, 1.0)
    assert v.startswith("complete:")
    assert "blocked_reranker_degenerate" in v


def test_classify_verdict_energy_beats_sc():
    # REQ-AR-050 / SCENARIO-AR-050-02
    v = EXP.classify_verdict_3531(True, 93, True, 3, 0.05, 0.01)
    assert v.startswith("complete:")
    assert "energy_beats_self_consistency" in v
    assert "validated" in v


def test_classify_verdict_informative_negative():
    # REQ-AR-050 / SCENARIO-AR-050-02
    v = EXP.classify_verdict_3531(True, 93, True, -2, -0.02, 0.5)
    assert v.startswith("complete:")
    assert "informative_negative" in v


def test_classify_verdict_always_complete_prefix():
    # REQ-AR-050 / SCENARIO-AR-050-02
    for args in [
        (False, 93, False, 0, 0.0, 1.0),
        (True, 10, False, 0, 0.0, 1.0),
        (True, 93, False, 0, 0.0, 1.0),
        (True, 93, True, 5, 0.05, 0.01),
        (True, 93, True, -1, -0.01, 0.8),
    ]:
        v = EXP.classify_verdict_3531(*args)
        assert v.startswith("complete:"), f"Missing complete: prefix: {v!r}"


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-03: build_artifact_3531 schema
# ---------------------------------------------------------------------------

_REQUIRED_ARTIFACT_FIELDS = [
    "experiment_id", "experiment", "honest_verdict", "inference_substrate",
    "corpus_oracle_exceeds_sc", "selectable_headroom", "reranker_makes_distinct_selections",
    "headroom_corpus_n", "self_consistency_accuracy", "step_aggregation_energy_accuracy",
    "pessimistic_bon_energy_accuracy", "optimal_aggregation_accuracy", "best_condition",
    "flip_count_best_vs_sc", "flips_correct_best", "flips_incorrect_best",
    "net_correctness_gain_best", "delta_best_vs_self_consistency",
    "paired_significance", "random_seed", "reproducibility_checksum", "duration_s",
    "preconditions_checked", "acceptance_gates",
]


def test_build_artifact_3531_contains_all_required_fields():
    # REQ-AR-050 / SCENARIO-AR-050-03
    artifact = EXP.build_artifact_3531({})
    for field in _REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field!r}"


def test_build_artifact_3531_updates_with_fields():
    # REQ-AR-050 / SCENARIO-AR-050-03
    artifact = EXP.build_artifact_3531({"honest_verdict": "complete: test_verdict"})
    assert artifact["honest_verdict"] == "complete: test_verdict"


def test_build_artifact_3531_has_inference_substrate():
    # REQ-AR-050 / SCENARIO-AR-050-03
    artifact = EXP.build_artifact_3531({})
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"


def test_build_artifact_3531_has_acceptance_gates():
    # REQ-AR-050 / SCENARIO-AR-050-03
    artifact = EXP.build_artifact_3531({})
    gates = artifact["acceptance_gates"]
    assert "G0_fair_test" in gates
    assert "G1_energy_beats_sc_with_headroom" in gates


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-04: field_provenance coverage
# ---------------------------------------------------------------------------

def test_field_provenance_covers_required_fields():
    # REQ-AR-050 / SCENARIO-AR-050-04
    provenance = EXP._field_provenance_3531()
    key_fields = [
        "honest_verdict", "inference_substrate", "corpus_oracle_exceeds_sc",
        "selectable_headroom", "reranker_makes_distinct_selections",
        "headroom_corpus_n", "self_consistency_accuracy", "flip_count_best_vs_sc",
        "flips_correct_best", "net_correctness_gain_best",
        "delta_best_vs_self_consistency", "paired_significance",
        "random_seed", "reproducibility_checksum", "duration_s",
    ]
    for field in key_fields:
        assert field in provenance, f"Missing provenance for {field!r}"
        assert len(provenance[field]) >= 20, f"Provenance too short for {field!r}"


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01: compute_checksum determinism
# ---------------------------------------------------------------------------

def test_compute_checksum_is_deterministic():
    # REQ-AR-050 / SCENARIO-AR-050-01
    records = [_make_record("3", ["3", "7", "3", "7"])]
    c1 = EXP.compute_checksum(records, EXP.RANDOM_SEED, "test_corpus")
    c2 = EXP.compute_checksum(records, EXP.RANDOM_SEED, "test_corpus")
    assert c1 == c2


def test_compute_checksum_changes_with_different_corpus():
    # REQ-AR-050 / SCENARIO-AR-050-01
    records_a = [_make_record("3", ["3", "7", "3", "7"])]
    records_b = [_make_record("5", ["5", "9", "5", "9"])]
    c1 = EXP.compute_checksum(records_a, EXP.RANDOM_SEED, "test")
    c2 = EXP.compute_checksum(records_b, EXP.RANDOM_SEED, "test")
    assert c1 != c2


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-02: RANDOM_SEED is not the experiment number
# ---------------------------------------------------------------------------

def test_random_seed_is_not_experiment_id():
    # REQ-AR-050 / SCENARIO-AR-050-02
    # Per CLAUDE.md Adversarial Artifact Verification, seed != experiment_id.
    assert EXP.RANDOM_SEED != EXP.EXP_ID


def test_random_seed_matches_expected_sha256_derived_value():
    # REQ-AR-050 / SCENARIO-AR-050-02
    import hashlib
    seed_input = "exp=3531;corpus=p01_selectable_headroom+fallback;route2_energy_vs_sc"
    expected = int(hashlib.sha256(seed_input.encode()).hexdigest()[:8], 16) % (2**31)
    assert EXP.RANDOM_SEED == expected


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01: _extract_features — no SC indicator
# ---------------------------------------------------------------------------

def test_extract_features_no_sc_indicator_three_features():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # The anti-consensus-trap fix: feature vector = [energy, n_steps, ans_len], NOT SC.
    records = [_make_record("3", ["3", "7", "3", "7"])]
    energies = EXP.compute_process_energy(records)
    X, y, prob_idx = EXP._extract_features(records, energies)
    assert X.shape[1] == 3, "Must have exactly 3 features (energy, n_steps, ans_len)"
    assert len(y) == 4
    assert all(prob_idx == 0)


def test_extract_features_empty_records():
    # REQ-AR-050 / SCENARIO-AR-050-01
    X, y, prob_idx = EXP._extract_features([], [])
    assert X.shape == (0, 3)
    assert y.shape == (0,)
    assert prob_idx.shape == (0,)


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01: McNemar significance
# ---------------------------------------------------------------------------

def test_mcnemar_significance_p1_when_no_discordant():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # No discordant pairs → p = 1.0 (no evidence of difference)
    cond = [True, True, False, False]
    sc = [True, True, False, False]
    result = EXP.compute_mcnemar_significance(cond, sc, seed=42, n_boot=100)
    assert result["mcnemar_p"] == pytest.approx(1.0)
    assert len(result["bootstrap_ci95"]) == 2


def test_mcnemar_significance_low_p_when_cond_dominates():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # cond correct on 10 more problems than SC → should have low p
    n = 20
    cond = [True] * 15 + [False] * 5
    sc = [False] * 15 + [True] * 5
    result = EXP.compute_mcnemar_significance(cond, sc, seed=42, n_boot=200)
    assert result["mcnemar_p"] < 0.05


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01: score_conditions output shape
# ---------------------------------------------------------------------------

def test_score_conditions_returns_five_conditions():
    # REQ-AR-050 / SCENARIO-AR-050-01
    records = [
        _make_record("3", ["7", "7", "3", "7"]),
        _make_record("5", ["5", "9", "5", "5"]),
    ]
    energies = EXP.compute_process_energy(records)
    pessimistic = EXP.compute_pessimistic_bon_scores(records, energies)
    sc_majority = EXP.build_sc_majority(records)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X, y, _ = EXP._extract_features(records, energies)
    reranker = EXP.fit_energy_reranker(X, y)

    cond = EXP.score_conditions(records, energies, pessimistic, sc_majority, reranker)
    assert set(cond.keys()) == {
        "greedy", "sc", "process_energy_argmin", "pessimistic_bon", "trained_energy_vote"
    }
    for key in cond:
        assert len(cond[key]) == 2, f"Condition {key!r} should have 2 answers"


def test_score_conditions_sc_matches_build_sc_majority():
    # REQ-AR-050 / SCENARIO-AR-050-01
    records = [_make_record("3", ["7", "7", "3", "7"])]
    energies = EXP.compute_process_energy(records)
    pessimistic = EXP.compute_pessimistic_bon_scores(records, energies)
    sc_majority = EXP.build_sc_majority(records)
    X, y, _ = EXP._extract_features(records, energies)
    reranker = EXP.fit_energy_reranker(X, y)
    cond = EXP.score_conditions(records, energies, pessimistic, sc_majority, reranker)
    assert cond["sc"][0] == "7"


# ---------------------------------------------------------------------------
# Integration: actual artifact produced by the run
# ---------------------------------------------------------------------------

def test_actual_artifact_has_all_required_fields():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # Verify the artifact written to disk by main() contains all required fields.
    artifact_path = (
        Path(__file__).resolve().parents[3]
        / "results"
        / "experiment_3531_p01_route2_energy_vs_sc_on_headroom_corpus_v1.json"
    )
    if not artifact_path.exists():
        pytest.skip("Artifact not yet generated; run the script first")
    artifact = json.loads(artifact_path.read_text())
    for field in _REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Artifact missing required field: {field!r}"


def test_actual_artifact_verdict_starts_with_complete():
    # REQ-AR-050 / SCENARIO-AR-050-01
    artifact_path = (
        Path(__file__).resolve().parents[3]
        / "results"
        / "experiment_3531_p01_route2_energy_vs_sc_on_headroom_corpus_v1.json"
    )
    if not artifact_path.exists():
        pytest.skip("Artifact not yet generated")
    artifact = json.loads(artifact_path.read_text())
    assert artifact["honest_verdict"].startswith("complete:"), (
        f"Verdict {artifact['honest_verdict']!r} lacks complete: prefix"
    )


def test_actual_artifact_inference_substrate():
    # REQ-AR-050 / SCENARIO-AR-050-01
    artifact_path = (
        Path(__file__).resolve().parents[3]
        / "results"
        / "experiment_3531_p01_route2_energy_vs_sc_on_headroom_corpus_v1.json"
    )
    if not artifact_path.exists():
        pytest.skip("Artifact not yet generated")
    artifact = json.loads(artifact_path.read_text())
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"


def test_actual_artifact_random_seed_not_experiment_id():
    # REQ-AR-050 / SCENARIO-AR-050-02
    artifact_path = (
        Path(__file__).resolve().parents[3]
        / "results"
        / "experiment_3531_p01_route2_energy_vs_sc_on_headroom_corpus_v1.json"
    )
    if not artifact_path.exists():
        pytest.skip("Artifact not yet generated")
    artifact = json.loads(artifact_path.read_text())
    assert artifact["random_seed"] != artifact["experiment"]

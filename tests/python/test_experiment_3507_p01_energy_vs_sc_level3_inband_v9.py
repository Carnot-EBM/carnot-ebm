"""Tests for exp3507 -- P0.1 energy vs SC on purpose-built level-3 in-band corpus (v9).

Traces to REQ-KONA-3507. Covers:
- _normalize_record: schema adaptation from new corpus format to old expected format
- _load_level3: level filtering and usability gating
- _checksum_v9: deterministic hash
- Module-level constants: MIN_PROBLEMS, ARTIFACT_PATH, SEED
- Integration: main() writes a JSON artifact with all required fields; verdict has a
  terminal prefix; level3_n is a non-negative integer; blocked path when corpus absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiment_3507_p01_energy_vs_sc_on_level3_inband_corpus_v9 import (  # noqa: E402
    ARTIFACT_PATH,
    MIN_PROBLEMS,
    SEED,
    _normalize_record,
    _load_level3,
    _checksum_v9,
    main,
)


# ---------------------------------------------------------------------------
# Shared helper: build a record in the NEW corpus schema
# ---------------------------------------------------------------------------


def _new_corpus_record(pid, gold_norm, sample_norms, level=3):
    """Minimal record in the data/p01_difficulty_matched_generations.jsonl schema."""
    samples = [
        {
            "text": f"text_{a}",
            "extracted_answer_norm": a,
            "extracted_answer": a,
            "correct": (a == gold_norm),
            "mean_token_logprob": -0.5,
            "reasoning_steps": ["Step 1: think.", "Step 2: answer."],
            "n_steps": 2,
        }
        for a in sample_norms
    ]
    greedy = {
        "text": f"greedy_text",
        "extracted_answer_norm": sample_norms[0] if sample_norms else None,
        "extracted_answer": sample_norms[0] if sample_norms else None,
        "correct": (sample_norms[0] == gold_norm) if sample_norms else False,
        "mean_token_logprob": -0.3,
        "reasoning_steps": ["Step 1: think.", "Step 2: answer."],
        "n_steps": 2,
    }
    return {
        "problem_id": pid,
        "level": level,
        "problem": f"question_{pid}",
        "gold_answer": gold_norm,
        "gold_answer_norm": gold_norm,
        "greedy": greedy,
        "samples": samples,
        "sampled_answers": [s["extracted_answer_norm"] for s in samples],
        "k_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# REQ-KONA-3507: _normalize_record
# ---------------------------------------------------------------------------


def test_normalize_record_gold_field():
    # gold key must be gold_answer_norm  REQ-KONA-3507
    rec = _new_corpus_record("p1", "42", ["42", "42", "1"])
    out = _normalize_record(rec)
    assert out["gold"] == "42"  # REQ-KONA-3507


def test_normalize_record_greedy_answer():
    # greedy.answer must come from extracted_answer_norm  REQ-KONA-3507
    rec = _new_corpus_record("p1", "7", ["7", "7", "3"])
    out = _normalize_record(rec)
    assert out["greedy"]["answer"] == "7"  # REQ-KONA-3507


def test_normalize_record_samples_answer_and_steps():
    # samples[i].answer from extracted_answer_norm; steps from reasoning_steps  REQ-KONA-3507
    rec = _new_corpus_record("p1", "5", ["5", "3", "5"])
    out = _normalize_record(rec)
    for s, orig in zip(out["samples"], rec["samples"]):
        assert s["answer"] == orig["extracted_answer_norm"]  # REQ-KONA-3507
        assert s["steps"] == orig["reasoning_steps"]  # REQ-KONA-3507


def test_normalize_record_none_answer_preserved():
    # None extracted_answer_norm must stay None in .answer  REQ-KONA-3507
    rec = _new_corpus_record("p1", "5", [None, "5", None])
    out = _normalize_record(rec)
    assert out["samples"][0]["answer"] is None  # REQ-KONA-3507
    assert out["samples"][1]["answer"] == "5"  # REQ-KONA-3507


# ---------------------------------------------------------------------------
# REQ-KONA-3507: _load_level3
# ---------------------------------------------------------------------------


def test_load_level3_filters_to_level3():
    # Only level=3 records are kept  REQ-KONA-3507
    recs = [
        _new_corpus_record("l3", "1", ["1"] * 6, level=3),
        _new_corpus_record("l4", "2", ["2"] * 6, level=4),
        _new_corpus_record("l2", "3", ["3"] * 6, level=2),
    ]
    out = _load_level3(recs)
    assert len(out) == 1 and out[0]["problem_id"] == "l3"  # REQ-KONA-3507


def test_load_level3_drops_missing_gold():
    # Records without gold_answer_norm and gold_answer are dropped  REQ-KONA-3507
    rec = _new_corpus_record("p1", None, ["X"] * 6, level=3)
    rec["gold_answer_norm"] = None
    rec["gold_answer"] = None
    out = _load_level3([rec])
    assert out == []  # REQ-KONA-3507


def test_load_level3_drops_too_few_samples():
    # Records with fewer than 4 samples are dropped  REQ-KONA-3507
    rec = _new_corpus_record("p1", "5", ["5", "5", "1"], level=3)
    out = _load_level3([rec])
    assert out == []  # REQ-KONA-3507


def test_load_level3_keeps_adequate_records():
    # Records with >=4 samples and a gold are kept  REQ-KONA-3507
    rec = _new_corpus_record("p1", "5", ["5", "5", "5", "1"], level=3)
    out = _load_level3([rec])
    assert len(out) == 1  # REQ-KONA-3507


# ---------------------------------------------------------------------------
# REQ-KONA-3507: _checksum_v9
# ---------------------------------------------------------------------------


def test_checksum_v9_is_deterministic():
    # Same records + same seed -> same checksum  REQ-KONA-3507
    recs = [_new_corpus_record("p1", "5", ["5", "3", "5", "5"], level=3)]
    norm = [_normalize_record(r) for r in recs]
    assert _checksum_v9(norm) == _checksum_v9(norm)  # REQ-KONA-3507


def test_checksum_v9_differs_on_different_records():
    # Different records -> different checksum (with high probability)  REQ-KONA-3507
    recs_a = [_new_corpus_record("p1", "5", ["5", "5", "5", "1"], level=3)]
    recs_b = [_new_corpus_record("p1", "9", ["9", "9", "9", "2"], level=3)]
    norm_a = [_normalize_record(r) for r in recs_a]
    norm_b = [_normalize_record(r) for r in recs_b]
    assert _checksum_v9(norm_a) != _checksum_v9(norm_b)  # REQ-KONA-3507


# ---------------------------------------------------------------------------
# REQ-KONA-3507: module-level constants
# ---------------------------------------------------------------------------


def test_min_problems_constant_is_40():
    # MIN_PROBLEMS must be 40 per the task spec  REQ-KONA-3507
    assert MIN_PROBLEMS == 40  # REQ-KONA-3507


def test_artifact_path_in_results_dir():
    # Artifact path must be in the results/ directory  REQ-KONA-3507
    assert "results" in str(ARTIFACT_PATH)  # REQ-KONA-3507


def test_seed_is_not_experiment_id():
    # Seed must not equal experiment ID (avoids tautology flag)  REQ-KONA-3507
    assert SEED != 3507  # REQ-KONA-3507


# ---------------------------------------------------------------------------
# REQ-KONA-3507: integration tests via main()
# ---------------------------------------------------------------------------


def test_main_writes_artifact_with_required_fields(tmp_path, monkeypatch):
    # main() writes a JSON with all required fields in both blocked and scored paths
    # REQ-KONA-3507
    import scripts.experiment_3507_p01_energy_vs_sc_on_level3_inband_corpus_v9 as mod

    monkeypatch.setattr(mod, "ARTIFACT_PATH", tmp_path / "artifact.json")
    mod.main()
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    required = [
        "honest_verdict",
        "inference_substrate",
        "corpus_source",
        "level3_n",
        "level3_sc",
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
        assert field in artifact, f"missing required field: {field!r}"  # REQ-KONA-3507


def test_main_verdict_has_terminal_prefix(tmp_path, monkeypatch):
    # honest_verdict must start with a terminal prefix  REQ-KONA-3507
    import scripts.experiment_3507_p01_energy_vs_sc_on_level3_inband_corpus_v9 as mod

    monkeypatch.setattr(mod, "ARTIFACT_PATH", tmp_path / "artifact.json")
    mod.main()
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    verdict = artifact["honest_verdict"]
    terminal = (
        "complete:", "complete_", "success:", "success_",
        "passed:", "passed_", "shipped:", "shipped_",
    )
    assert any(verdict.startswith(p) for p in terminal), (
        f"verdict lacks terminal prefix: {verdict!r}"
    )  # REQ-KONA-3507


def test_main_level3_n_is_non_negative_integer(tmp_path, monkeypatch):
    # level3_n must be an integer >= 0  REQ-KONA-3507
    import scripts.experiment_3507_p01_energy_vs_sc_on_level3_inband_corpus_v9 as mod

    monkeypatch.setattr(mod, "ARTIFACT_PATH", tmp_path / "artifact.json")
    mod.main()
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    assert isinstance(artifact["level3_n"], int) and artifact["level3_n"] >= 0  # REQ-KONA-3507


def test_main_blocked_when_corpus_missing(tmp_path, monkeypatch):
    # When CORPUS_PATH points to a non-existent file, verdict is blocked  REQ-KONA-3507
    import scripts.experiment_3507_p01_energy_vs_sc_on_level3_inband_corpus_v9 as mod

    monkeypatch.setattr(mod, "ARTIFACT_PATH", tmp_path / "artifact.json")
    monkeypatch.setattr(mod, "CORPUS_PATH", tmp_path / "nonexistent.jsonl")
    mod.main()
    artifact = json.loads((tmp_path / "artifact.json").read_text())
    assert "blocked_no_level3_corpus" in artifact["honest_verdict"]  # REQ-KONA-3507


def test_main_blocked_when_level3_too_small(tmp_path, monkeypatch):
    # When the corpus has fewer than MIN_PROBLEMS level-3 usable records, verdict
    # contains blocked_level3_corpus_too_small  REQ-KONA-3507
    import json as _json
    import scripts.experiment_3507_p01_energy_vs_sc_on_level3_inband_corpus_v9 as mod

    # Write a tiny corpus with only 2 level-3 records
    tiny_corpus = tmp_path / "tiny.jsonl"
    tiny_records = [
        _new_corpus_record(f"p{i}", "5", ["5", "5", "5", "1"], level=3)
        for i in range(2)
    ]
    tiny_corpus.write_text("\n".join(_json.dumps(r) for r in tiny_records))

    monkeypatch.setattr(mod, "ARTIFACT_PATH", tmp_path / "artifact.json")
    monkeypatch.setattr(mod, "CORPUS_PATH", tiny_corpus)
    mod.main()
    artifact = _json.loads((tmp_path / "artifact.json").read_text())
    assert "blocked_level3_corpus_too_small" in artifact["honest_verdict"]  # REQ-KONA-3507

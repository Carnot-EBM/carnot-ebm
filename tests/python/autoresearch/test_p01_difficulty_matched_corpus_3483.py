r"""Tests for the P0.1 difficulty-matched corpus builder v2 (exp3483).

Spec anchor: REQ-AR-050 (P0.1 Difficulty-Matched Corpus v2 - Adaptive Level
Selection + Process Traces) in openspec/capabilities/autoresearch/spec.md.

These tests exercise the GPU-free logic only: the shared corpus helpers in
``carnot.autoresearch.corpus_p01_headroom`` and the pure assembly/classifier
functions in the exp3483 script. The GPU-bound generation loop in ``main()``
is integration-only (requires CUDA + a cached GGUF) and is not unit-tested.

Every test asserts real behavior; none are skipped (CLAUDE.md "Tests Must Run
and Assert").
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from carnot.autoresearch import corpus_p01_headroom as helpers

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO_ROOT
    / "scripts"
    / "experiment_3483_p01_difficulty_matched_corpus_builder_v2.py"
)


def _load_script_module():
    """Import the exp3483 script by file path (it lives outside the package)."""
    spec = importlib.util.spec_from_file_location("exp3483_v2", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXP = _load_script_module()


# --- SCENARIO-AR-050-01: Final-answer extraction and normalization ---------
def test_extract_last_boxed_answer_balances_braces():
    # REQ-AR-050 / SCENARIO-AR-050-01
    text = r"first \boxed{99} then the real one \boxed{\frac{1}{2}}"
    assert helpers.extract_boxed_answer(text) == r"\frac{1}{2}"


def test_extract_boxed_answer_none_when_absent():
    # REQ-AR-050 / SCENARIO-AR-050-01
    assert helpers.extract_boxed_answer("no box here") is None


def test_normalize_strips_latex_wrappers():
    # REQ-AR-050 / SCENARIO-AR-050-01
    assert helpers.normalize_answer(r"$\left(11\right)$") == "(11)"


def test_answers_match_extracts_and_normalizes():
    # REQ-AR-050 / SCENARIO-AR-050-01
    assert helpers.answers_match(r"work... \boxed{11}", "11") is True
    assert helpers.answers_match(r"\boxed{12}", "11") is False
    assert helpers.answers_match(None, "11") is False


# --- SCENARIO-AR-050-02: Self-consistency band classification --------------
def test_in_headroom_band_boundaries():
    # REQ-AR-050 / SCENARIO-AR-050-02
    assert helpers.in_headroom_band(helpers.BAND_LO) is True
    assert helpers.in_headroom_band(helpers.BAND_HI) is True
    assert helpers.in_headroom_band(0.55) is True
    assert helpers.in_headroom_band(0.265) is False  # MATH-L5 floor
    assert helpers.in_headroom_band(0.908) is False  # GSM8K ceiling


def test_self_consistency_accuracy_majority_vote():
    # REQ-AR-050 / SCENARIO-AR-050-02
    records = [
        {"sampled_answers": ["3", "3", "4"], "gold_answer_norm": "3"},  # correct
        {"sampled_answers": ["5", "6", "6"], "gold_answer_norm": "5"},  # wrong (mv=6)
    ]
    assert helpers.self_consistency_accuracy(records) == 0.5


def test_self_consistency_accuracy_empty_is_zero():
    # REQ-AR-050 / SCENARIO-AR-050-02
    assert helpers.self_consistency_accuracy([]) == 0.0


# --- SCENARIO-AR-050-03: Per-step reasoning traces captured ----------------
def test_parse_reasoning_steps_multiparagraph():
    # REQ-AR-050 / SCENARIO-AR-050-03
    steps = helpers.parse_reasoning_steps("Step one.\n\nStep two.\n\nStep three.")
    assert steps == ["Step one.", "Step two.", "Step three."]


def test_parse_reasoning_steps_empty():
    # REQ-AR-050 / SCENARIO-AR-050-03
    assert helpers.parse_reasoning_steps("") == []


def test_mean_token_logprob_ignores_none():
    # REQ-AR-050 / SCENARIO-AR-050-03 (confidence capture)
    assert helpers.mean_token_logprob([-0.2, -0.4, None]) == -0.3
    assert helpers.mean_token_logprob([]) is None


# --- SCENARIO-AR-050-04: Resume skips completed problems -------------------
def test_completed_problem_ids_reads_jsonl(tmp_path):
    # REQ-AR-050 / SCENARIO-AR-050-04
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        '{"problem_id": "a/1"}\n\n{"problem_id": "b/2"}\nnot-json\n',
        encoding="utf-8",
    )
    ids = helpers.completed_problem_ids(p)
    assert ids == {"a/1", "b/2"}


def test_completed_problem_ids_missing_file(tmp_path):
    # REQ-AR-050 / SCENARIO-AR-050-04
    assert helpers.completed_problem_ids(tmp_path / "nope.jsonl") == set()


# --- exp3483 script: pure assembly + classifier functions ------------------
def test_gemma_chat_prompt_format():
    # REQ-AR-050: prompt wraps the problem in the gemma instruct turn format
    prompt = EXP.gemma_chat_prompt("What is 2+2?")
    assert prompt.startswith("<start_of_turn>user\n")
    assert prompt.rstrip().endswith("<start_of_turn>model")
    assert r"\boxed{}" in prompt


def test_build_generation_record_correct():
    # REQ-AR-050 / SCENARIO-AR-050-01 + 03: assemble one generation row
    text = "First line.\n\nSecond line.\n\n" + r"\boxed{11}"
    rec = EXP.build_generation_record(text, [-0.1, -0.3, None], "11", "greedy", 7)
    assert rec["mode"] == "greedy"
    assert rec["seed"] == 7
    assert rec["extracted_answer"] == "11"
    assert rec["correct"] is True
    assert rec["n_steps"] == 3
    assert rec["mean_token_logprob"] == -0.2


def test_build_generation_record_no_gold():
    # REQ-AR-050: a missing gold answer yields correct=False (never crashes)
    rec = EXP.build_generation_record(r"\boxed{5}", [-0.5], None, "sampled", 9)
    assert rec["correct"] is False
    assert rec["extracted_answer"] == "5"


def test_build_problem_record_assembles_samples():
    # REQ-AR-050 / SCENARIO-AR-050-04: per-problem row carries id + samples
    meta = {
        "problem_id": "alg/3",
        "level": 3,
        "subject": "Algebra",
        "problem": "q?",
        "gold_answer": "11",
    }
    greedy = EXP.build_generation_record(r"\boxed{11}", [-0.1], "11", "greedy", 7)
    s1 = EXP.build_generation_record(r"\boxed{11}", [-0.2], "11", "sampled", 8)
    s2 = EXP.build_generation_record(r"\boxed{9}", [-0.3], "11", "sampled", 9)
    rec = EXP.build_problem_record(meta, greedy, [s1, s2])
    assert rec["problem_id"] == "alg/3"
    assert rec["level"] == 3
    assert rec["gold_answer_norm"] == "11"
    assert rec["sampled_answers"] == ["11", "9"]
    assert rec["k_samples"] == 2
    assert rec["greedy_correct"] is True
    # self-consistency over this single problem: majority "11" == gold -> 1.0
    assert helpers.self_consistency_accuracy([rec]) == 1.0


def test_classify_verdict_all_branches():
    # REQ-AR-050: verdicts start with complete: and encode the corpus size band
    head = EXP.classify_verdict(EXP.TARGET_N, True, 0.55)
    scor = EXP.classify_verdict(EXP.SCORABLE_N, True, 0.55)
    part = EXP.classify_verdict(5, True, 0.55)
    block = EXP.classify_verdict(5, False, None)
    for v in (head, scor, part, block):
        assert v.startswith("complete:")
    assert "headline_eligible" in head
    assert "scorable_partial" in scor
    assert "partial" in part and "scorable" not in part
    assert "blocked_no_in_band_split_found" in block


def test_field_principles_cover_required_fields():
    # REQ-AR-050: every REQUIRED ARTIFACT FIELD carries a principle annotation
    fp = EXP.field_principles()
    for key in (
        "honest_verdict",
        "inference_substrate",
        "corpus_path",
        "benchmark_id",
        "selected_levels",
        "per_level_probe_sc",
        "self_consistency_in_headroom_band",
        "duration_s",
    ):
        assert key in fp and fp[key]


def test_greedy_accuracy_helper():
    # REQ-AR-050: greedy accuracy over the corpus rows
    assert EXP._greedy_accuracy([]) is None
    corpus = [{"greedy_correct": True}, {"greedy_correct": False}]
    assert EXP._greedy_accuracy(corpus) == 0.5

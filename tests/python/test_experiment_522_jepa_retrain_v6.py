"""Tests for Exp 522 helpers — jepa_retrain_v6 module.

Coverage: load_cot_pairs_from_experiments, compute_held_out_split,
violation_pairs_to_trainer_dicts, and _text_to_embedding.

Spec: REQ-LEARN-048, SCENARIO-LEARN-076, SCENARIO-LEARN-077
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import pytest

from carnot.embeddings.jepa_retrain import ViolationPair
from carnot.models.jepa_retrain_v6 import (
    _EMBED_DIM,
    _entry_to_violation_pair,
    _find_repo_root_from,
    _load_pairs_from_file,
    _text_to_embedding,
    compute_held_out_split,
    load_cot_pairs_from_experiments,
    violation_pairs_to_trainer_dicts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fover_json(tmp_path: Path) -> Path:
    """Write a minimal FOVER-style JSON file and return its path."""
    data = [
        {"step_text": "Step 1: 2+2=4", "label": "correct", "question_id": "q1", "confidence": 1.0},
        {"step_text": "Step 2: 2+2=5", "label": "incorrect", "question_id": "q2", "confidence": 0.9},
        {"step_text": "Step 3: 3*3=9", "label": "correct", "question_id": "q3", "confidence": 1.0},
    ]
    path = tmp_path / "fover_test.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def response_json(tmp_path: Path) -> Path:
    """Write a response-style JSON file and return its path."""
    data = [
        {"response": "The answer is 4.", "correct": True, "question_id": "r1", "model_id": "test_model"},
        {"response": "The answer is 5.", "correct": False, "question_id": "r2", "model_id": "test_model"},
    ]
    path = tmp_path / "response_test.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def violation_pair_json(tmp_path: Path) -> Path:
    """Write a ViolationPair-style JSON file and return its path."""
    data = [
        {"partial_response": "Step A", "full_response": "Step A full", "has_violation": False, "question_id": "vp1"},
        {"partial_response": "Step B", "full_response": "Step B full", "has_violation": True, "question_id": "vp2"},
    ]
    path = tmp_path / "vp_test.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def sample_pairs() -> List[ViolationPair]:
    """Return 10 ViolationPair objects for split tests."""
    return [
        ViolationPair(
            partial_response=f"step {i}",
            full_response=f"step {i} full",
            has_violation=(i % 2 == 0),
            model_id="test",
            question_id=f"q{i}",
        )
        for i in range(10)
    ]


# ---------------------------------------------------------------------------
# _text_to_embedding tests
# ---------------------------------------------------------------------------


class TestTextToEmbedding:
    def test_returns_correct_length(self):
        emb = _text_to_embedding("hello world")
        assert len(emb) == _EMBED_DIM

    def test_deterministic(self):
        emb1 = _text_to_embedding("some text")
        emb2 = _text_to_embedding("some text")
        assert emb1 == emb2

    def test_different_texts_differ(self):
        emb1 = _text_to_embedding("correct answer")
        emb2 = _text_to_embedding("wrong answer")
        assert emb1 != emb2

    def test_empty_string(self):
        emb = _text_to_embedding("")
        assert len(emb) == _EMBED_DIM
        # Empty string produces valid floats (not NaN)
        assert all(math.isfinite(v) for v in emb)

    def test_returns_floats(self):
        emb = _text_to_embedding("test")
        assert all(isinstance(v, float) for v in emb)


# ---------------------------------------------------------------------------
# _entry_to_violation_pair tests
# ---------------------------------------------------------------------------


class TestEntryToViolationPair:
    def test_fover_correct(self):
        entry = {"step_text": "x=1", "label": "correct", "question_id": "q1"}
        vp = _entry_to_violation_pair(entry)
        assert vp is not None
        assert vp.has_violation is False
        assert vp.partial_response == "x=1"
        assert vp.question_id == "q1"

    def test_fover_incorrect(self):
        entry = {"step_text": "x=99", "label": "incorrect"}
        vp = _entry_to_violation_pair(entry)
        assert vp is not None
        assert vp.has_violation is True

    def test_fover_empty_text_returns_none(self):
        entry = {"step_text": "", "label": "correct"}
        assert _entry_to_violation_pair(entry) is None

    def test_response_style_correct(self):
        entry = {"response": "4+4=8", "correct": True, "model_id": "m1", "question_id": "r1"}
        vp = _entry_to_violation_pair(entry)
        assert vp is not None
        assert vp.has_violation is False
        assert vp.model_id == "m1"

    def test_response_style_incorrect(self):
        entry = {"response": "4+4=9", "correct": False}
        vp = _entry_to_violation_pair(entry)
        assert vp is not None
        assert vp.has_violation is True

    def test_response_empty_returns_none(self):
        entry = {"response": "", "correct": True}
        assert _entry_to_violation_pair(entry) is None

    def test_violation_pair_style(self):
        entry = {"partial_response": "partial", "full_response": "full", "has_violation": True}
        vp = _entry_to_violation_pair(entry)
        assert vp is not None
        assert vp.has_violation is True
        assert vp.full_response == "full"

    def test_violation_pair_style_empty_partial_returns_none(self):
        entry = {"partial_response": "", "has_violation": False}
        assert _entry_to_violation_pair(entry) is None

    def test_unknown_schema_returns_none(self):
        entry = {"unrelated_key": "data"}
        assert _entry_to_violation_pair(entry) is None

    def test_non_dict_returns_none(self):
        assert _entry_to_violation_pair("not a dict") is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _load_pairs_from_file tests
# ---------------------------------------------------------------------------


class TestLoadPairsFromFile:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        result = _load_pairs_from_file(tmp_path / "nonexistent.json")
        assert result == []

    def test_invalid_json_returns_empty(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json{{{")
        assert _load_pairs_from_file(bad) == []

    def test_non_list_json_returns_empty(self, tmp_path: Path):
        obj = tmp_path / "obj.json"
        obj.write_text(json.dumps({"key": "value"}))
        assert _load_pairs_from_file(obj) == []

    def test_loads_fover_style(self, fover_json: Path):
        pairs = _load_pairs_from_file(fover_json)
        assert len(pairs) == 3
        assert all(isinstance(p, ViolationPair) for p in pairs)

    def test_loads_response_style(self, response_json: Path):
        pairs = _load_pairs_from_file(response_json)
        assert len(pairs) == 2

    def test_loads_violation_pair_style(self, violation_pair_json: Path):
        pairs = _load_pairs_from_file(violation_pair_json)
        assert len(pairs) == 2

    def test_skips_invalid_entries(self, tmp_path: Path):
        mixed = tmp_path / "mixed.json"
        mixed.write_text(json.dumps([
            {"step_text": "good", "label": "correct"},
            "not a dict",
            {"unrelated": "data"},
            {"step_text": "", "label": "correct"},  # empty text
        ]))
        pairs = _load_pairs_from_file(mixed)
        assert len(pairs) == 1  # only the first valid entry


# ---------------------------------------------------------------------------
# load_cot_pairs_from_experiments tests
# ---------------------------------------------------------------------------


class TestGetRepoRoot:
    def test_returns_path_with_cargo_toml(self):
        from carnot.models.jepa_retrain_v6 import _get_repo_root
        root = _get_repo_root()
        assert isinstance(root, Path)
        assert root.exists()

    def test_find_repo_root_from_finds_marker(self, tmp_path: Path):
        """_find_repo_root_from returns the directory containing Cargo.toml."""
        nested = tmp_path / "a" / "b" / "c" / "module.py"
        nested.parent.mkdir(parents=True)
        nested.touch()
        (tmp_path / "Cargo.toml").write_text("[workspace]")
        result = _find_repo_root_from(nested.resolve())
        assert result == tmp_path

    def test_find_repo_root_from_fallback(self, tmp_path: Path):
        """Line in _find_repo_root_from: fallback to parents[3] when no marker."""
        # Use a deep path so parents[3] is valid (needs at least 4 ancestors)
        nested = tmp_path / "p" / "q" / "r" / "s" / "module.py"
        nested.parent.mkdir(parents=True)
        nested.touch()
        # No Cargo.toml or pyproject.toml anywhere in tmp_path
        result = _find_repo_root_from(nested.resolve())
        assert result == nested.resolve().parents[3]


class TestLoadCotPairsFromExperiments:
    def test_fallback_when_no_exp_files(self, fover_json: Path):
        # Exp IDs 9998/9999 won't exist; should fall back to fover_json
        pairs = load_cot_pairs_from_experiments([9998, 9999], str(fover_json))
        assert len(pairs) == 3

    def test_uses_exp_file_when_present(self, tmp_path: Path, fover_json: Path, monkeypatch):
        # Create a fake exp{id}_cot_pairs.json in results/
        from carnot.models import jepa_retrain_v6
        fake_results = tmp_path / "results"
        fake_results.mkdir()
        exp_file = fake_results / "exp9001_cot_pairs.json"
        exp_file.write_text(json.dumps([
            {"step_text": "A", "label": "correct"},
            {"step_text": "B", "label": "incorrect"},
        ]))
        # Patch _get_repo_root to return tmp_path
        monkeypatch.setattr(jepa_retrain_v6, "_get_repo_root", lambda: tmp_path)
        pairs = load_cot_pairs_from_experiments([9001], str(fover_json))
        assert len(pairs) == 2

    def test_fallback_relative_path(self, tmp_path: Path, fover_json: Path, monkeypatch):
        """Line 146: relative fallback_path is resolved relative to repo root."""
        from carnot.models import jepa_retrain_v6
        # Patch _get_repo_root to return tmp_path so relative path resolves there
        fake_results = tmp_path / "results"
        fake_results.mkdir()
        rel_fover = fake_results / "rel_fover.json"
        rel_fover.write_text(json.dumps([{"step_text": "rel", "label": "correct"}]))
        monkeypatch.setattr(jepa_retrain_v6, "_get_repo_root", lambda: tmp_path)
        # Pass a relative path (no leading slash)
        pairs = load_cot_pairs_from_experiments([88888], "results/rel_fover.json")
        assert len(pairs) == 1

    def test_accumulates_multiple_exp_files(self, tmp_path: Path, fover_json: Path, monkeypatch):
        from carnot.models import jepa_retrain_v6
        fake_results = tmp_path / "results"
        fake_results.mkdir()
        for exp_id in [9001, 9002]:
            exp_file = fake_results / f"exp{exp_id}_cot_pairs.json"
            exp_file.write_text(json.dumps([
                {"step_text": f"text_{exp_id}", "label": "correct"},
            ]))
        monkeypatch.setattr(jepa_retrain_v6, "_get_repo_root", lambda: tmp_path)
        pairs = load_cot_pairs_from_experiments([9001, 9002], str(fover_json))
        assert len(pairs) == 2

    def test_fallback_returns_nonempty_from_real_fover(self):
        # Smoke test against the real fover file in the repo
        from carnot.models.jepa_retrain_v6 import _get_repo_root
        repo_root = _get_repo_root()
        fover_path = repo_root / "results" / "fover_labeled_steps_live.json"
        if not fover_path.exists():
            pytest.skip("fover_labeled_steps_live.json not in repo")
        pairs = load_cot_pairs_from_experiments([99998, 99999], str(fover_path))
        assert len(pairs) > 0
        assert all(isinstance(p, ViolationPair) for p in pairs)
        assert all(p.partial_response for p in pairs)


# ---------------------------------------------------------------------------
# compute_held_out_split tests
# ---------------------------------------------------------------------------


class TestComputeHeldOutSplit:
    def test_split_sums_to_total(self, sample_pairs: List[ViolationPair]):
        train, test = compute_held_out_split(sample_pairs, test_fraction=0.2)
        assert len(train) + len(test) == len(sample_pairs)

    def test_correct_test_size(self, sample_pairs: List[ViolationPair]):
        train, test = compute_held_out_split(sample_pairs, test_fraction=0.2)
        # 20% of 10 = 2
        assert len(test) == 2
        assert len(train) == 8

    def test_deterministic(self, sample_pairs: List[ViolationPair]):
        train1, test1 = compute_held_out_split(sample_pairs, test_fraction=0.2)
        train2, test2 = compute_held_out_split(sample_pairs, test_fraction=0.2)
        assert [p.question_id for p in train1] == [p.question_id for p in train2]
        assert [p.question_id for p in test1] == [p.question_id for p in test2]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_held_out_split([])

    def test_at_least_one_test_pair_for_small_list(self):
        pairs = [
            ViolationPair("a", "a", False, "m", "q1"),
            ViolationPair("b", "b", True, "m", "q2"),
        ]
        train, test = compute_held_out_split(pairs, test_fraction=0.1)
        assert len(test) >= 1
        assert len(train) >= 1

    def test_fraction_clamped_to_one(self, sample_pairs: List[ViolationPair]):
        # test_fraction > 1.0 should not produce an empty train set
        train, test = compute_held_out_split(sample_pairs, test_fraction=2.0)
        assert len(train) >= 1

    def test_fraction_clamped_to_zero(self, sample_pairs: List[ViolationPair]):
        # test_fraction=0.0 → n_test=max(1, 0)=1
        train, test = compute_held_out_split(sample_pairs, test_fraction=0.0)
        assert len(test) == 1

    def test_returns_lists(self, sample_pairs: List[ViolationPair]):
        train, test = compute_held_out_split(sample_pairs)
        assert isinstance(train, list)
        assert isinstance(test, list)


# ---------------------------------------------------------------------------
# violation_pairs_to_trainer_dicts tests
# ---------------------------------------------------------------------------


class TestViolationPairsToTrainerDicts:
    def test_output_length_matches_input(self, sample_pairs: List[ViolationPair]):
        dicts = violation_pairs_to_trainer_dicts(sample_pairs)
        assert len(dicts) == len(sample_pairs)

    def test_all_required_keys_present(self, sample_pairs: List[ViolationPair]):
        dicts = violation_pairs_to_trainer_dicts(sample_pairs)
        required = {"embedding", "violated_arithmetic", "violated_code", "violated_logic"}
        for d in dicts:
            assert required.issubset(d.keys()), f"Missing keys: {required - d.keys()}"

    def test_embedding_dimension(self, sample_pairs: List[ViolationPair]):
        dicts = violation_pairs_to_trainer_dicts(sample_pairs)
        for d in dicts:
            assert len(d["embedding"]) == _EMBED_DIM

    def test_label_propagation(self):
        pairs = [
            ViolationPair("text", "text", True, "m", "q1"),
            ViolationPair("text2", "text2", False, "m", "q2"),
        ]
        dicts = violation_pairs_to_trainer_dicts(pairs)
        assert dicts[0]["violated_arithmetic"] == 1
        assert dicts[0]["violated_code"] == 1
        assert dicts[0]["violated_logic"] == 1
        assert dicts[1]["violated_arithmetic"] == 0
        assert dicts[1]["violated_code"] == 0
        assert dicts[1]["violated_logic"] == 0

    def test_embedding_values_are_floats(self, sample_pairs: List[ViolationPair]):
        dicts = violation_pairs_to_trainer_dicts(sample_pairs[:2])
        for d in dicts:
            assert all(isinstance(v, float) for v in d["embedding"])

    def test_empty_input_returns_empty(self):
        assert violation_pairs_to_trainer_dicts([]) == []

    def test_violation_pairs_have_positive_bias(self):
        # has_violation=True → emb[0] gets +label_signal_strength bias
        # has_violation=False → emb[0] gets -label_signal_strength bias
        # Compare same text with different labels — they should differ only in bias
        pair_vio = ViolationPair("same_text", "same_text", True, "m", "qv")
        pair_ok = ViolationPair("same_text", "same_text", False, "m", "qo")
        d_vio = violation_pairs_to_trainer_dicts([pair_vio])[0]
        d_ok = violation_pairs_to_trainer_dicts([pair_ok])[0]
        # emb[0] for violation > emb[0] for non-violation (bias shifts it by +1.0)
        assert d_vio["embedding"][0] > d_ok["embedding"][0]

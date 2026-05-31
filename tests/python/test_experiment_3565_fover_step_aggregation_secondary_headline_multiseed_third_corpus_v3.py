import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts'))

try:
    from experiment_3565_fover_step_aggregation_secondary_headline_multiseed_third_corpus_v3 import (
        _checksum,
        _load_corpus,
        _is_usable,
        _normalise_sample,
        _split_corpus_ab,
        _train_held_out_split,
        _distinct_pipeline_assert,
        _ci95,
        _base_payload,
        _emit,
    )
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp", "scripts/experiment_3565_fover_step_aggregation_secondary_headline_multiseed_third_corpus_v3.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    _checksum = exp._checksum
    _load_corpus = exp._load_corpus
    _is_usable = exp._is_usable
    _normalise_sample = exp._normalise_sample
    _split_corpus_ab = exp._split_corpus_ab
    _train_held_out_split = exp._train_held_out_split
    _distinct_pipeline_assert = exp._distinct_pipeline_assert
    _ci95 = exp._ci95
    _base_payload = exp._base_payload
    _emit = exp._emit

def test_checksum():
    """REQ-KONA-3565"""
    res = _checksum([{"problem_id": "a"}], [{"problem_id": "b"}], [{"problem_id": "c"}], [1, 2], "min")
    assert isinstance(res, str)
    assert len(res) == 16

def test_load_corpus():
    """REQ-KONA-3565"""
    with TemporaryDirectory() as d:
        p = Path(d) / "test.jsonl"
        with open(p, "w") as f:
            f.write('{"a": 1}\n\n{"b": 2}\n')
        
        res = _load_corpus(p)
        assert len(res) == 2
        assert res[0]["a"] == 1
        
        res_empty = _load_corpus(Path(d) / "not_exist.jsonl")
        assert len(res_empty) == 0

def test_is_usable():
    """REQ-KONA-3565"""
    # no gold
    assert not _is_usable({"samples": []})
    # gold but no samples
    assert not _is_usable({"gold_answer": "1", "samples": []})
    # gold but only 1 scored sample
    assert not _is_usable({
        "gold_answer": "1", 
        "samples": [{"correct": True, "steps": ["a"]}]
    })
    # 2 scored samples
    assert _is_usable({
        "gold_answer": "1", 
        "samples": [
            {"correct": True, "steps": ["a"]},
            {"correct": False, "reasoning_steps": ["b"]}
        ]
    })
    # bad samples
    assert not _is_usable({
        "gold_answer": "1", 
        "samples": [
            {"correct": True},
            {"steps": ["b"]}
        ]
    })

def test_normalise_sample():
    """REQ-KONA-3565"""
    s = {"steps": [1, 2]}
    res = _normalise_sample(s)
    assert res == {"reasoning_steps": [1, 2]}
    
    s2 = {"reasoning_steps": [3]}
    res2 = _normalise_sample(s2)
    assert res2 == {"reasoning_steps": [3]}

def test_split_corpus_ab():
    """SCENARIO-KONA-3565"""
    records = [
        {"problem_id": "p1"},
        {"problem_id": "p2"},
        {"problem_id": "p3"},
        {"problem_id": "p4"},
    ]
    a, b = _split_corpus_ab(records, 42)
    assert len(a) == 2
    assert len(b) == 2
    a_ids = set(r["problem_id"] for r in a)
    b_ids = set(r["problem_id"] for r in b)
    assert len(a_ids.intersection(b_ids)) == 0

def test_train_held_out_split():
    """SCENARIO-KONA-3565"""
    records = [{"id": i} for i in range(10)]
    t, h = _train_held_out_split(records, 42)
    assert len(h) == 5
    assert len(t) == 5

def test_distinct_pipeline_assert():
    """REQ-KONA-3565"""
    assert _distinct_pipeline_assert([1, 2], [1])
    assert _distinct_pipeline_assert([], [])
    assert not _distinct_pipeline_assert([1, 2], [1, 2])
    assert _distinct_pipeline_assert([1, 2], [1, 3])

def test_ci95():
    """REQ-KONA-3565"""
    lo, hi = _ci95([1.0])
    assert lo == 1.0
    assert hi == 1.0
    
    lo, hi = _ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    assert lo < 2.5
    assert hi > 3.5

def test_base_payload():
    """REQ-KONA-3565"""
    p = _base_payload(0.0)
    assert p["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert p["duration_s"] >= 1.0

def test_emit(monkeypatch):
    """REQ-KONA-3565"""
    with TemporaryDirectory() as d:
        import importlib.util
        spec = importlib.util.spec_from_file_location("exp", "scripts/experiment_3565_fover_step_aggregation_secondary_headline_multiseed_third_corpus_v3.py")
        exp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp)
        
        target = Path(d) / "test.json"
        monkeypatch.setattr(exp, "ARTIFACT_PATH", target)
        exp._emit({"test": 1})
        assert target.exists()
        with open(target) as f:
            data = json.load(f)
            assert data["test"] == 1
            assert data["schema"] == ["test"]

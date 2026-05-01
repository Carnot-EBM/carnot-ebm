"""Tests for Exp 1032 PPSEBM Relay v4.

Tests cover:
- _tokenize: returns tokens from text, includes bigrams and words
- _build_vocabulary: maps tokens to unique indices
- _fit_tfidf: returns vocab and idf of correct length
- _tfidf_vector: returns vector of vocab length, L2-normalised
- _sigmoid: boundary values and monotonicity
- _train_logistic_regression: converges on linearly separable data
- _predict_energy: correct items score lower than incorrect items after training
- _auroc: perfect ranking = 1.0, reversed = 0.0, random = ~0.5
- _load_exp1029_n_violation_pairs: reads JSON correctly, handles missing file
- _load_corpus: returns train/test with expected label distributions
- _build_training_data: correct label encoding (incorrect -> 1, correct -> 0)
- _run_live_relay: returns 10 records, all required keys present
- main (end-to-end): writes valid artifact with all required schema fields

Spec: REQ-LEARN-011 (FR-11 autonomous self-learning), REQ-SELFLEARN-016,
      SCENARIO-SELFLEARN-016
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from scripts.experiment_1032_ppsebm_relay_v4 import (
    _auroc,
    _build_training_data,
    _build_vocabulary,
    _fit_tfidf,
    _load_corpus,
    _load_exp1029_n_violation_pairs,
    _predict_energy,
    _run_live_relay,
    _sigmoid,
    _tfidf_vector,
    _tokenize,
    _train_logistic_regression,
    main,
)


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


def test_tokenize_returns_nonempty_for_text():
    """Non-empty text should return at least some tokens."""
    tokens = _tokenize("3x + 7 = 22")
    assert len(tokens) > 0


def test_tokenize_includes_words():
    """Single-word text produces word token (lowercased)."""
    tokens = _tokenize("Hello")
    assert "hello" in tokens


def test_tokenize_includes_bigrams():
    """Text of length >= 2 produces character bigrams."""
    tokens = _tokenize("ab")
    assert "ab" in tokens


def test_tokenize_empty_string():
    """Empty string produces no tokens (empty list)."""
    tokens = _tokenize("")
    assert tokens == []


# ---------------------------------------------------------------------------
# _build_vocabulary
# ---------------------------------------------------------------------------


def test_build_vocabulary_unique_indices():
    """Every token maps to a unique index."""
    docs = ["hello world", "foo bar"]
    vocab = _build_vocabulary(docs)
    assert len(set(vocab.values())) == len(vocab)


def test_build_vocabulary_monotone_from_zero():
    """Indices start at 0 and are contiguous."""
    docs = ["a b c"]
    vocab = _build_vocabulary(docs)
    assert set(vocab.values()) == set(range(len(vocab)))


# ---------------------------------------------------------------------------
# _fit_tfidf
# ---------------------------------------------------------------------------


def test_fit_tfidf_lengths_match():
    """IDF vector length equals vocabulary size."""
    texts = ["alpha beta gamma", "delta epsilon alpha"]
    vocab, idf = _fit_tfidf(texts)
    assert len(vocab) == len(idf)


def test_fit_tfidf_rare_term_higher_idf():
    """A term appearing in fewer docs has higher IDF than a universal term."""
    texts = ["alpha beta", "alpha gamma"]
    vocab, idf = _fit_tfidf(texts)
    # 'alpha' appears in both docs; 'beta' appears in one doc
    alpha_idf = idf[vocab["alpha"]]
    beta_idf = idf[vocab["beta"]]
    assert beta_idf > alpha_idf


# ---------------------------------------------------------------------------
# _tfidf_vector
# ---------------------------------------------------------------------------


def test_tfidf_vector_length():
    """Vector length matches vocabulary size."""
    texts = ["hello world"]
    vocab, idf = _fit_tfidf(texts)
    vec = _tfidf_vector("hello world", vocab, idf)
    assert len(vec) == len(vocab)


def test_tfidf_vector_l2_normalised():
    """Returned vector is L2-normalised (norm ≈ 1.0 for non-zero doc)."""
    texts = ["hello world foo bar"]
    vocab, idf = _fit_tfidf(texts)
    vec = _tfidf_vector("hello world foo bar", vocab, idf)
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6


def test_tfidf_vector_unknown_tokens():
    """Completely disjoint vocabulary does not raise errors; returns correct-length vector."""
    # Train on texts with no character-level overlap with test string
    texts = ["aaaa bbbb cccc dddd eeee"]
    vocab, idf = _fit_tfidf(texts)
    # A text built entirely from tokens absent in training vocab
    vec = _tfidf_vector("zzzzz qqqqq", vocab, idf)
    assert len(vec) == len(vocab)
    # Should be all zeros or near-zero since tokens are disjoint
    assert sum(vec) >= 0.0  # No negative entries, no exception raised


# ---------------------------------------------------------------------------
# _sigmoid
# ---------------------------------------------------------------------------


def test_sigmoid_zero():
    """sigmoid(0) == 0.5."""
    assert abs(_sigmoid(0.0) - 0.5) < 1e-9


def test_sigmoid_large_positive():
    """sigmoid(large) approaches 1."""
    assert _sigmoid(100.0) > 0.999


def test_sigmoid_large_negative():
    """sigmoid(-large) approaches 0."""
    assert _sigmoid(-100.0) < 0.001


def test_sigmoid_monotone():
    """sigmoid is monotonically increasing."""
    xs = [-5.0, -2.0, 0.0, 2.0, 5.0]
    ys = [_sigmoid(x) for x in xs]
    assert all(a < b for a, b in zip(ys, ys[1:], strict=False))


# ---------------------------------------------------------------------------
# _train_logistic_regression
# ---------------------------------------------------------------------------


def test_logistic_regression_separable():
    """Converges on a linearly separable dataset."""
    # Positive class: [1,0]; negative class: [0,1]
    X = [[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5
    y = [1] * 5 + [0] * 5
    w, bias = _train_logistic_regression(X, y, lr=0.5, epochs=500)
    # Positive examples should score higher
    pos_score = _sigmoid(w[0] * 1.0 + w[1] * 0.0 + bias)
    neg_score = _sigmoid(w[0] * 0.0 + w[1] * 1.0 + bias)
    assert pos_score > neg_score


# ---------------------------------------------------------------------------
# _predict_energy
# ---------------------------------------------------------------------------


def test_predict_energy_range():
    """Energy is always in (0, 1)."""
    texts = ["The answer is 42.", "Wrong: 3+3=7."]
    vocab, idf = _fit_tfidf(texts)
    X = [_tfidf_vector(t, vocab, idf) for t in texts]
    y = [0, 1]
    w, bias = _train_logistic_regression(X, y, lr=0.5, epochs=300)
    for text in texts:
        energy = _predict_energy(text, vocab, idf, w, bias)
        assert 0.0 < energy < 1.0


def test_predict_energy_incorrect_higher_than_correct():
    """After training, incorrect-labeled texts score higher energy than correct ones."""
    correct_texts = [
        "3x = 15, so x = 5.",
        "Area = 6 × 8 = 48 m².",
        "Average speed = 200 / 4 = 50 km/h.",
        "Volume = pi * 9 * 5 = 45pi.",
        "P(at least one) = 1 - 0.7^3 = 0.657.",
    ]
    incorrect_texts = [
        "Average speed = (60 + 40) / 2 = 50.",  # wrong method
        "3x = 22 - 3 - 7 = 12.",  # arithmetic error
        "Perimeter = 4 × 6 = 24.",  # wrong formula
        "Volume = 2*pi*3*5 = 30pi.",  # surface area formula
        "P = 3 * 0.3 = 0.9.",  # wrong probability rule
    ]
    all_texts = correct_texts + incorrect_texts
    vocab, idf = _fit_tfidf(all_texts)
    X = [_tfidf_vector(t, vocab, idf) for t in all_texts]
    y = [0] * len(correct_texts) + [1] * len(incorrect_texts)
    w, bias = _train_logistic_regression(X, y, lr=0.5, epochs=500)
    correct_mean_energy = sum(_predict_energy(t, vocab, idf, w, bias) for t in correct_texts) / len(
        correct_texts
    )
    incorrect_mean_energy = sum(
        _predict_energy(t, vocab, idf, w, bias) for t in incorrect_texts
    ) / len(incorrect_texts)
    assert incorrect_mean_energy > correct_mean_energy


# ---------------------------------------------------------------------------
# _auroc
# ---------------------------------------------------------------------------


def test_auroc_perfect():
    """Perfect ranking produces AUROC = 1.0."""
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    assert _auroc(scores, labels) == pytest.approx(1.0, abs=0.01)


def test_auroc_reversed():
    """Reversed ranking (worst predictor) produces AUROC ≈ 0.0."""
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [1, 1, 0, 0]
    assert _auroc(scores, labels) == pytest.approx(0.0, abs=0.01)


def test_auroc_single_class():
    """Single-class labels return 0.5 (undefined AUROC)."""
    scores = [0.9, 0.8, 0.7]
    labels = [1, 1, 1]
    assert _auroc(scores, labels) == 0.5


def test_auroc_range():
    """AUROC is always in [0.0, 1.0]."""
    import random

    random.seed(42)
    scores = [random.random() for _ in range(20)]
    labels = [random.randint(0, 1) for _ in range(20)]
    auc = _auroc(scores, labels)
    assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# _load_exp1029_n_violation_pairs
# ---------------------------------------------------------------------------


def test_load_exp1029_reads_json(tmp_path):
    """Reads n_violation_pairs correctly from a valid JSON artifact."""
    artifact = {"n_violation_pairs": 42, "status": "success"}
    f = tmp_path / "exp1029.json"
    f.write_text(json.dumps(artifact))
    with patch("scripts.experiment_1032_ppsebm_relay_v4._EXP_1029_RESULT", f):
        result = _load_exp1029_n_violation_pairs()
    assert result == 42


def test_load_exp1029_missing_file(tmp_path):
    """Returns 0 when artifact file does not exist."""
    missing = tmp_path / "nonexistent.json"
    with patch("scripts.experiment_1032_ppsebm_relay_v4._EXP_1029_RESULT", missing):
        result = _load_exp1029_n_violation_pairs()
    assert result == 0


def test_load_exp1029_malformed_json(tmp_path):
    """Returns 0 when artifact is not valid JSON."""
    f = tmp_path / "bad.json"
    f.write_text("{not valid json")
    with patch("scripts.experiment_1032_ppsebm_relay_v4._EXP_1029_RESULT", f):
        result = _load_exp1029_n_violation_pairs()
    assert result == 0


# ---------------------------------------------------------------------------
# _load_corpus
# ---------------------------------------------------------------------------


def test_load_corpus_returns_two_lists(tmp_path):
    """Returns (train, test) — both non-empty when real split files exist."""
    train = [{"step_text": "a", "label": "correct"}]
    test = [{"step_text": "b", "label": "incorrect"}]
    train_p = tmp_path / "train.json"
    test_p = tmp_path / "test.json"
    train_p.write_text(json.dumps(train))
    test_p.write_text(json.dumps(test))
    with (
        patch("scripts.experiment_1032_ppsebm_relay_v4._TRAIN_PATH", train_p),
        patch("scripts.experiment_1032_ppsebm_relay_v4._TEST_PATH", test_p),
    ):
        tr, te = _load_corpus()
    assert len(tr) == 1
    assert len(te) == 1


def test_load_corpus_fallback_to_full(tmp_path):
    """Falls back to full corpus 80/20 split when split files are missing."""
    corpus = [
        {"step_text": f"step {i}", "label": "correct" if i % 2 == 0 else "incorrect"}
        for i in range(10)
    ]
    corpus_p = tmp_path / "corpus.json"
    corpus_p.write_text(json.dumps(corpus))
    missing_train = tmp_path / "train_missing.json"
    missing_test = tmp_path / "test_missing.json"
    with (
        patch("scripts.experiment_1032_ppsebm_relay_v4._TRAIN_PATH", missing_train),
        patch("scripts.experiment_1032_ppsebm_relay_v4._TEST_PATH", missing_test),
        patch("scripts.experiment_1032_ppsebm_relay_v4._CORPUS_PATH", corpus_p),
    ):
        tr, te = _load_corpus()
    assert len(tr) + len(te) == 10
    assert len(tr) == 8  # 80%


# ---------------------------------------------------------------------------
# _build_training_data
# ---------------------------------------------------------------------------


def test_build_training_data_label_encoding():
    """'incorrect' -> 1, 'correct' -> 0."""
    items = [
        {"step_text": "The answer is 5.", "label": "correct"},
        {"step_text": "The answer is wrong.", "label": "incorrect"},
    ]
    vocab, idf = _fit_tfidf([item["step_text"] for item in items])
    X, y = _build_training_data(items, vocab, idf)
    assert y == [0, 1]
    assert len(X) == 2
    assert len(X[0]) == len(vocab)


def test_build_training_data_feature_vector_lengths():
    """All feature vectors have the same length (vocab size)."""
    items = [
        {"step_text": "Step one: compute 3+2=5.", "label": "correct"},
        {"step_text": "Step two: compute 3+2=7.", "label": "incorrect"},
        {"step_text": "Step three: perimeter = 24.", "label": "incorrect"},
    ]
    vocab, idf = _fit_tfidf([item["step_text"] for item in items])
    X, y = _build_training_data(items, vocab, idf)
    first_len = len(X[0])
    assert all(len(x) == first_len for x in X)


# ---------------------------------------------------------------------------
# _run_live_relay
# ---------------------------------------------------------------------------


def test_run_live_relay_returns_ten_records():
    """Always returns exactly 10 relay records (one per question)."""
    texts = ["correct step " * 5, "wrong step " * 5]
    vocab, idf = _fit_tfidf(texts)
    w, bias = [0.0] * len(vocab), 0.0
    train_energies = [0.3, 0.7]
    records, n_viol = _run_live_relay(vocab, idf, w, bias, train_energies)
    assert len(records) == 10


def test_run_live_relay_record_keys():
    """Each record contains all required keys."""
    texts = ["correct step", "wrong step"]
    vocab, idf = _fit_tfidf(texts)
    w, bias = [0.0] * len(vocab), 0.0
    records, _ = _run_live_relay(vocab, idf, w, bias, [0.5])
    required_keys = {
        "question_idx",
        "question",
        "answer",
        "true_label",
        "energy",
        "violation_threshold",
        "flagged_as_violation",
        "source",
        "model",
        "timestamp",
    }
    for record in records:
        assert required_keys.issubset(set(record.keys()))


def test_run_live_relay_energy_in_range():
    """Energy values are in (0, 1)."""
    texts = ["correct arithmetic", "wrong arithmetic error"]
    vocab, idf = _fit_tfidf(texts)
    X = [_tfidf_vector(t, vocab, idf) for t in texts]
    y = [0, 1]
    w, bias = _train_logistic_regression(X, y, epochs=100)
    records, _ = _run_live_relay(vocab, idf, w, bias, [0.3, 0.7])
    for record in records:
        assert 0.0 < record["energy"] < 1.0


# ---------------------------------------------------------------------------
# main (end-to-end)
# ---------------------------------------------------------------------------


def test_main_writes_valid_artifact(tmp_path):
    """End-to-end run produces a valid JSON artifact with all required fields."""
    deliverable = tmp_path / "result.json"
    relay_memory = tmp_path / "relay_memory.jsonl"
    exp1029 = tmp_path / "exp1029.json"
    exp1029.write_text(json.dumps({"n_violation_pairs": 29, "status": "success"}))

    # Use real train/test data (already on disk)
    real_train = _REPO_ROOT / "data" / "fover_train.json"
    real_test = _REPO_ROOT / "data" / "fover_test.json"

    with (
        patch("scripts.experiment_1032_ppsebm_relay_v4.DELIVERABLE", str(deliverable)),
        patch("scripts.experiment_1032_ppsebm_relay_v4._RELAY_MEMORY_PATH", relay_memory),
        patch("scripts.experiment_1032_ppsebm_relay_v4._EXP_1029_RESULT", exp1029),
        patch("scripts.experiment_1032_ppsebm_relay_v4._TRAIN_PATH", real_train),
        patch("scripts.experiment_1032_ppsebm_relay_v4._TEST_PATH", real_test),
    ):
        main()

    assert deliverable.exists()
    artifact = json.loads(deliverable.read_text())

    required_fields = [
        "experiment",
        "schema",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "title",
        # Experiment-specific required fields
        "n_violation_pairs_used",
        "ppsebm_auroc",
        "n_real_violations",
        "relay_live",
        "honest_verdict",
    ]
    for field in required_fields:
        assert field in artifact, f"Missing required field: {field}"

    assert artifact["experiment"] == 1032
    assert artifact["status"] == "success"
    assert artifact["honest_verdict"] in {
        "relay_live",
        "ppsebm_trained_relay_below_threshold",
        "blocked_insufficient_violations",
        "failed",
    }
    assert isinstance(artifact["ppsebm_auroc"], float)
    assert 0.0 <= artifact["ppsebm_auroc"] <= 1.0
    assert isinstance(artifact["n_real_violations"], int)
    assert isinstance(artifact["relay_live"], bool)


def test_main_blocked_when_insufficient_violations(tmp_path):
    """Writes blocked artifact when n_violation_pairs < gate."""
    deliverable = tmp_path / "result.json"
    relay_memory = tmp_path / "relay_memory.jsonl"
    exp1029 = tmp_path / "exp1029.json"
    exp1029.write_text(json.dumps({"n_violation_pairs": 5, "status": "partial"}))

    with (
        patch("scripts.experiment_1032_ppsebm_relay_v4.DELIVERABLE", str(deliverable)),
        patch("scripts.experiment_1032_ppsebm_relay_v4._RELAY_MEMORY_PATH", relay_memory),
        patch("scripts.experiment_1032_ppsebm_relay_v4._EXP_1029_RESULT", exp1029),
    ):
        main()

    artifact = json.loads(deliverable.read_text())
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_insufficient_violations"
    assert artifact["n_violation_pairs_used"] == 5


def test_main_relay_memory_appended(tmp_path):
    """Live relay records are written to the relay memory JSONL file."""
    deliverable = tmp_path / "result.json"
    relay_memory = tmp_path / "relay_memory.jsonl"
    exp1029 = tmp_path / "exp1029.json"
    exp1029.write_text(json.dumps({"n_violation_pairs": 29, "status": "success"}))

    real_train = _REPO_ROOT / "data" / "fover_train.json"
    real_test = _REPO_ROOT / "data" / "fover_test.json"

    with (
        patch("scripts.experiment_1032_ppsebm_relay_v4.DELIVERABLE", str(deliverable)),
        patch("scripts.experiment_1032_ppsebm_relay_v4._RELAY_MEMORY_PATH", relay_memory),
        patch("scripts.experiment_1032_ppsebm_relay_v4._EXP_1029_RESULT", exp1029),
        patch("scripts.experiment_1032_ppsebm_relay_v4._TRAIN_PATH", real_train),
        patch("scripts.experiment_1032_ppsebm_relay_v4._TEST_PATH", real_test),
    ):
        main()

    assert relay_memory.exists()
    lines = relay_memory.read_text().strip().split("\n")
    assert len(lines) == 10  # one record per relay question
    for line in lines:
        record = json.loads(line)
        assert "energy" in record
        assert "flagged_as_violation" in record

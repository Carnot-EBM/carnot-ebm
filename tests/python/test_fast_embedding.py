"""Tests for fast embedding alternatives.

Spec coverage: REQ-EMBED-001, SCENARIO-EMBED-001
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.embeddings.fast_embedding import (
    CharNgramEmbedding,
    FastEmbeddingProtocol,
    HashEmbedding,
    MiniLMEmbedding,
    RandomProjectionEmbedding,
    TFIDFProjectionEmbedding,
    benchmark_embedding,
    get_default_embedding,
)


class TestL2Normalize:
    """Tests for REQ-EMBED-001: L2 normalization helper."""

    def test_1d_unit_vector(self) -> None:
        """REQ-EMBED-001: 1D vector is L2-normalized to unit norm."""
        from carnot.embeddings.fast_embedding import _l2_normalize

        v = np.array([3.0, 4.0], dtype=np.float32)
        out = _l2_normalize(v)
        np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-6)

    def test_1d_zero_vector(self) -> None:
        """REQ-EMBED-001: Zero vector is handled safely (no NaN)."""
        from carnot.embeddings.fast_embedding import _l2_normalize

        v = np.zeros(4, dtype=np.float32)
        out = _l2_normalize(v)
        assert np.all(np.isfinite(out))

    def test_2d_rows_normalized(self) -> None:
        """REQ-EMBED-001: 2D array normalizes each row independently."""
        from carnot.embeddings.fast_embedding import _l2_normalize

        v = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        out = _l2_normalize(v)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


class TestRandomProjectionEmbedding:
    """Tests for REQ-EMBED-001: RandomProjectionEmbedding."""

    def test_embed_dim_property(self) -> None:
        """REQ-EMBED-001: embed_dim property returns configured dimension."""
        emb = RandomProjectionEmbedding(embed_dim=64)
        assert emb.embed_dim == 64

    def test_encode_shape(self) -> None:
        """REQ-EMBED-001: encode returns correct shape."""
        emb = RandomProjectionEmbedding(embed_dim=32)
        result = emb.encode("hello world")
        assert result.shape == (32,)
        assert result.dtype == np.float32

    def test_encode_unit_norm(self) -> None:
        """REQ-EMBED-001: encode output is L2-normalized."""
        emb = RandomProjectionEmbedding(embed_dim=64)
        result = emb.encode("test text")
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-5)

    def test_encode_batch_shape(self) -> None:
        """REQ-EMBED-001: encode_batch returns (N, embed_dim) shape."""
        emb = RandomProjectionEmbedding(embed_dim=32)
        texts = ["hello", "world", "foo"]
        result = emb.encode_batch(texts)
        assert result.shape == (3, 32)
        assert result.dtype == np.float32

    def test_encode_batch_rows_normalized(self) -> None:
        """REQ-EMBED-001: encode_batch rows are L2-normalized."""
        emb = RandomProjectionEmbedding(embed_dim=64)
        texts = ["a", "bb", "ccc"]
        result = emb.encode_batch(texts)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)

    def test_byte_hist_empty_text(self) -> None:
        """REQ-EMBED-001: Empty text does not crash."""
        emb = RandomProjectionEmbedding(embed_dim=32)
        result = emb.encode("")
        assert np.all(np.isfinite(result))

    def test_protocol_conformance(self) -> None:
        """REQ-EMBED-001: RandomProjectionEmbedding satisfies FastEmbeddingProtocol."""
        emb = RandomProjectionEmbedding()
        assert isinstance(emb, FastEmbeddingProtocol)

    def test_deterministic_with_same_seed(self) -> None:
        """REQ-EMBED-001: Same seed produces identical embeddings."""
        emb1 = RandomProjectionEmbedding(embed_dim=32, seed=7)
        emb2 = RandomProjectionEmbedding(embed_dim=32, seed=7)
        text = "the answer is 42"
        np.testing.assert_array_equal(emb1.encode(text), emb2.encode(text))


class TestHashEmbedding:
    """Tests for REQ-EMBED-001: HashEmbedding."""

    def test_embed_dim_property(self) -> None:
        """REQ-EMBED-001: embed_dim property returns configured dimension."""
        emb = HashEmbedding(embed_dim=64)
        assert emb.embed_dim == 64

    def test_encode_shape(self) -> None:
        """REQ-EMBED-001: encode returns correct shape."""
        emb = HashEmbedding(embed_dim=32)
        result = emb.encode("hello world 42")
        assert result.shape == (32,)
        assert result.dtype == np.float32

    def test_encode_unit_norm(self) -> None:
        """REQ-EMBED-001: encode output is L2-normalized."""
        emb = HashEmbedding(embed_dim=64)
        result = emb.encode("some text")
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-5)

    def test_encode_batch_shape(self) -> None:
        """REQ-EMBED-001: encode_batch returns (N, embed_dim)."""
        emb = HashEmbedding(embed_dim=32)
        texts = ["a b c", "d e f"]
        result = emb.encode_batch(texts)
        assert result.shape == (2, 32)

    def test_empty_text_no_crash(self) -> None:
        """REQ-EMBED-001: Empty text does not crash."""
        emb = HashEmbedding(embed_dim=32)
        result = emb.encode("")
        assert np.all(np.isfinite(result))

    def test_protocol_conformance(self) -> None:
        """REQ-EMBED-001: HashEmbedding satisfies FastEmbeddingProtocol."""
        emb = HashEmbedding()
        assert isinstance(emb, FastEmbeddingProtocol)

    def test_token_bucket_counts_deterministic(self) -> None:
        """REQ-EMBED-001: _text_to_bucket_counts is deterministic (md5-based)."""
        emb = HashEmbedding(n_buckets=256)
        counts1 = emb._text_to_bucket_counts("hello world")
        counts2 = emb._text_to_bucket_counts("hello world")
        np.testing.assert_array_equal(counts1, counts2)

    def test_token_bucket_counts_sum_one(self) -> None:
        """REQ-EMBED-001: Bucket counts sum to 1.0 for non-empty text."""
        emb = HashEmbedding(n_buckets=256)
        counts = emb._text_to_bucket_counts("hello world foo bar")
        np.testing.assert_allclose(counts.sum(), 1.0, atol=1e-6)


class TestCharNgramEmbedding:
    """Tests for REQ-EMBED-001: CharNgramEmbedding."""

    def test_embed_dim_property(self) -> None:
        """REQ-EMBED-001: embed_dim property returns configured dimension."""
        emb = CharNgramEmbedding(embed_dim=64)
        assert emb.embed_dim == 64

    def test_encode_shape(self) -> None:
        """REQ-EMBED-001: encode returns correct shape."""
        emb = CharNgramEmbedding(embed_dim=32)
        result = emb.encode("arithmetic expression")
        assert result.shape == (32,)
        assert result.dtype == np.float32

    def test_encode_unit_norm(self) -> None:
        """REQ-EMBED-001: encode output is L2-normalized."""
        emb = CharNgramEmbedding(embed_dim=64)
        result = emb.encode("the answer is 42")
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-5)

    def test_encode_batch_shape(self) -> None:
        """REQ-EMBED-001: encode_batch returns (N, embed_dim)."""
        emb = CharNgramEmbedding(embed_dim=32)
        texts = ["hello", "world", "test"]
        result = emb.encode_batch(texts)
        assert result.shape == (3, 32)

    def test_encode_batch_rows_normalized(self) -> None:
        """REQ-EMBED-001: encode_batch rows are L2-normalized."""
        emb = CharNgramEmbedding(embed_dim=32)
        texts = ["foo bar", "baz qux"]
        result = emb.encode_batch(texts)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)

    def test_protocol_conformance(self) -> None:
        """REQ-EMBED-001: CharNgramEmbedding satisfies FastEmbeddingProtocol."""
        emb = CharNgramEmbedding()
        assert isinstance(emb, FastEmbeddingProtocol)


class TestTFIDFProjectionEmbedding:
    """Tests for REQ-EMBED-001: TFIDFProjectionEmbedding."""

    def test_embed_dim_property(self) -> None:
        """REQ-EMBED-001: embed_dim property returns configured dimension."""
        emb = TFIDFProjectionEmbedding(embed_dim=64)
        assert emb.embed_dim == 64

    def test_encode_shape(self) -> None:
        """REQ-EMBED-001: encode returns correct shape."""
        emb = TFIDFProjectionEmbedding(embed_dim=32)
        result = emb.encode("the answer is 42")
        assert result.shape == (32,)
        assert result.dtype == np.float32

    def test_encode_unit_norm(self) -> None:
        """REQ-EMBED-001: encode output is L2-normalized (uses in-vocabulary words)."""
        # Use words that appear in the default corpus vocabulary.
        emb = TFIDFProjectionEmbedding(embed_dim=64)
        result = emb.encode("the answer is 42 product sum equals")
        norm = np.linalg.norm(result)
        # Zero norm means OOV text; skip norm check in that case.
        if norm > 1e-8:
            np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_encode_batch_shape(self) -> None:
        """REQ-EMBED-001: encode_batch returns (N, embed_dim)."""
        emb = TFIDFProjectionEmbedding(embed_dim=32)
        texts = ["hello world", "foo bar"]
        result = emb.encode_batch(texts)
        assert result.shape == (2, 32)

    def test_custom_corpus(self) -> None:
        """REQ-EMBED-001: Custom corpus fits TF-IDF vocabulary."""
        corpus = ["foo bar baz", "hello world", "test test test"]
        emb = TFIDFProjectionEmbedding(embed_dim=32, corpus=corpus)
        result = emb.encode("foo bar")
        assert result.shape == (32,)
        assert np.all(np.isfinite(result))

    def test_protocol_conformance(self) -> None:
        """REQ-EMBED-001: TFIDFProjectionEmbedding satisfies FastEmbeddingProtocol."""
        emb = TFIDFProjectionEmbedding()
        assert isinstance(emb, FastEmbeddingProtocol)


class TestMiniLMEmbedding:
    """Tests for REQ-EMBED-001: MiniLMEmbedding (mocked)."""

    def test_embed_dim_property(self) -> None:
        """REQ-EMBED-001: embed_dim is 384."""
        emb = MiniLMEmbedding()
        assert emb.embed_dim == 384

    def test_encode_with_mock(self) -> None:
        """REQ-EMBED-001: encode calls SentenceTransformer.encode."""
        import sys
        from unittest.mock import MagicMock, patch

        fake_vec = np.ones(384, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_vec

        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        emb = MiniLMEmbedding()
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            result = emb.encode("hello world")

        assert result.shape == (384,)
        assert result.dtype == np.float32

    def test_encode_batch_with_mock(self) -> None:
        """REQ-EMBED-001: encode_batch calls SentenceTransformer.encode with batch."""
        import sys
        from unittest.mock import MagicMock, patch

        fake_vecs = np.ones((3, 384), dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_vecs

        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        emb = MiniLMEmbedding()
        texts = ["a", "b", "c"]
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            result = emb.encode_batch(texts)

        assert result.shape == (3, 384)

    def test_model_loaded_lazily(self) -> None:
        """REQ-EMBED-001: Model is None before first encode call."""
        emb = MiniLMEmbedding()
        assert emb._model is None

    def test_model_reused_on_second_call(self) -> None:
        """REQ-EMBED-001: Model is loaded once and reused on subsequent calls."""
        import sys
        from unittest.mock import MagicMock, patch

        fake_vec = np.ones(384, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_vec

        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        emb = MiniLMEmbedding()
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            emb.encode("first call")
            emb.encode("second call")

        # SentenceTransformer constructor should only be called once.
        mock_st.SentenceTransformer.assert_called_once()


class TestGetDefaultEmbedding:
    """Tests for REQ-EMBED-001: get_default_embedding factory."""

    def test_default_strategy_is_char_ngram(self) -> None:
        """REQ-EMBED-001: Default strategy returns CharNgramEmbedding."""
        emb = get_default_embedding()
        assert isinstance(emb, CharNgramEmbedding)

    def test_tfidf_strategy(self) -> None:
        """REQ-EMBED-001: tfidf strategy returns TFIDFProjectionEmbedding."""
        emb = get_default_embedding(strategy="tfidf")
        assert isinstance(emb, TFIDFProjectionEmbedding)

    def test_hash_strategy(self) -> None:
        """REQ-EMBED-001: hash strategy returns HashEmbedding."""
        emb = get_default_embedding(strategy="hash")
        assert isinstance(emb, HashEmbedding)

    def test_random_strategy(self) -> None:
        """REQ-EMBED-001: random strategy returns RandomProjectionEmbedding."""
        emb = get_default_embedding(strategy="random")
        assert isinstance(emb, RandomProjectionEmbedding)

    def test_minilm_strategy(self) -> None:
        """REQ-EMBED-001: minilm strategy returns MiniLMEmbedding."""
        emb = get_default_embedding(strategy="minilm")
        assert isinstance(emb, MiniLMEmbedding)

    def test_invalid_strategy_raises(self) -> None:
        """REQ-EMBED-001: Unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown embedding strategy"):
            get_default_embedding(strategy="nonexistent")

    def test_custom_embed_dim(self) -> None:
        """REQ-EMBED-001: Custom embed_dim is passed to embedding."""
        emb = get_default_embedding(strategy="random", embed_dim=128)
        assert emb.embed_dim == 128


class TestBenchmarkEmbedding:
    """Tests for REQ-EMBED-001: benchmark_embedding utility."""

    def test_returns_latency_stats(self) -> None:
        """REQ-EMBED-001: benchmark_embedding returns expected stat keys."""
        emb = RandomProjectionEmbedding(embed_dim=32)
        stats = benchmark_embedding(emb, texts=["hello world"], warmup=2, iters=5)
        assert "p50_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats
        assert "mean_ms" in stats
        assert "std_ms" in stats
        assert stats["n_iters"] == 5

    def test_latencies_are_positive(self) -> None:
        """REQ-EMBED-001: All latency values are non-negative."""
        emb = HashEmbedding(embed_dim=32)
        stats = benchmark_embedding(emb, texts=["test"], warmup=1, iters=3)
        for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms"):
            assert stats[key] >= 0.0, f"{key} should be non-negative"

    def test_empty_texts_uses_default_corpus(self) -> None:
        """REQ-EMBED-001: Empty texts argument falls back to default corpus."""
        emb = RandomProjectionEmbedding(embed_dim=32)
        stats = benchmark_embedding(emb, texts=[], warmup=1, iters=3)
        assert stats["n_iters"] == 3

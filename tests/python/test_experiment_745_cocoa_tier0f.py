"""Tests for Exp 745 — CoCoA Tier 0f inter-layer disagreement detector.

Each test traces to at least one REQ-VERIFY-151 or REQ-VERIFY-152 requirement.

Coverage targets (code added by Exp 745 only):
  - python/carnot/cascade/tier0f_cocoa.py  (CoCoADetector)
  - python/carnot/cascade/cascade_router.py  (Tier 0f advisory path added in Exp 745)

Spec: REQ-VERIFY-151, REQ-VERIFY-152, SCENARIO-VERIFY-201, SCENARIO-VERIFY-202
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN_DIM = 1024


def _make_mock_model_and_tokenizer(hidden_states_by_layer: dict[int, np.ndarray]):
    """Build a mock (model, tokenizer) pair that returns predetermined hidden states.

    WHY mock instead of real model: loading Qwen3.5-0.8B takes ~5 s and requires
    GPU/large RAM.  The unit tests need to verify the CoCoADetector's mathematical
    logic (cosine distance, layer indexing, calibration), not the LLM itself.
    We inject fixed hidden states so the tests are deterministic and fast.

    Parameters
    ----------
    hidden_states_by_layer : dict[int, np.ndarray]
        Maps layer index → (hidden_dim,) array.  The mock model will return these
        as hidden_states[layer_idx+1] in its forward-pass output.
    """
    import torch  # noqa: PLC0415

    # Build the tuple of hidden states the mock will return.
    # hidden_states[0] = embedding, hidden_states[k+1] = block k output.
    max_layer = max(hidden_states_by_layer.keys()) if hidden_states_by_layer else 0
    n_slots = max_layer + 2  # +2: slot 0 for embedding, slot max_layer+1 for last block

    hidden_states_tuple: list[object] = []
    for slot_idx in range(n_slots):
        layer_idx = slot_idx - 1  # layer_idx = slot_idx - 1 (slot 0 is embedding)
        if layer_idx in hidden_states_by_layer:
            vec = hidden_states_by_layer[layer_idx]
            # Shape: (1, seq_len=1, hidden_dim) as a torch tensor
            t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            t = torch.zeros(1, 1, HIDDEN_DIM, dtype=torch.float32)
        hidden_states_tuple.append(t)

    mock_output = MagicMock()
    mock_output.hidden_states = tuple(hidden_states_tuple)

    mock_model = MagicMock()
    mock_model.return_value = mock_output
    mock_model.__call__ = lambda self, **kwargs: mock_output  # noqa: ARG005

    # Make the mock callable so CoCoADetector can call model(**inputs, ...)
    mock_model.side_effect = None
    mock_model.return_value = mock_output

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    mock_tokenizer.return_value.__getitem__ = lambda s, k: MagicMock()  # noqa: ARG005
    mock_tokenizer.return_value.to = lambda d: mock_tokenizer.return_value  # noqa: ARG005

    return mock_model, mock_tokenizer


def _make_detector(hidden_states_by_layer: dict[int, np.ndarray], **kwargs):
    """Convenience: build a CoCoADetector with mocked model."""
    import torch  # noqa: PLC0415
    from carnot.cascade.tier0f_cocoa import CoCoADetector

    mock_model, mock_tokenizer = _make_mock_model_and_tokenizer(hidden_states_by_layer)

    # Patch CoCoADetector.extract_hidden_states to return our fixed states directly,
    # bypassing the torch forward pass (which requires a real model object).
    # This tests all the CoCoADetector logic EXCEPT the LLM loading itself.
    detector = CoCoADetector(
        model=mock_model,
        tokenizer=mock_tokenizer,
        **kwargs,
    )
    return detector


# ---------------------------------------------------------------------------
# Tests for CoCoADetector.compute_conmlds  (REQ-VERIFY-151-2)
# ---------------------------------------------------------------------------


class TestComputeConMLDS:
    """Unit tests for the cosine-distance computation.  Spec: REQ-VERIFY-151-2."""

    def test_identical_vectors_returns_zero(self):
        """SCENARIO-VERIFY-201: identical early and late states → ConMLDS = 0.

        WHY: if the model's representation at an early layer is exactly the same
        as at a late layer, there is zero inter-layer disagreement.  The cosine
        distance of a vector with itself is 0.

        Spec: REQ-VERIFY-151-2, SCENARIO-VERIFY-201
        """
        from carnot.cascade.tier0f_cocoa import CoCoADetector

        detector = CoCoADetector(model=None, tokenizer=None)
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = detector.compute_conmlds(vec, vec)
        assert abs(result) < 1e-5, f"Expected ~0.0, got {result}"

    def test_orthogonal_vectors_returns_one(self):
        """SCENARIO-VERIFY-202: orthogonal early and late states → ConMLDS = 1.

        WHY: orthogonal hidden states indicate maximum representational divergence —
        the model's understanding of the input is completely different between the
        two layers.  Cosine distance of orthogonal vectors is 1.0.

        Spec: REQ-VERIFY-151-2, SCENARIO-VERIFY-202
        """
        from carnot.cascade.tier0f_cocoa import CoCoADetector

        detector = CoCoADetector(model=None, tokenizer=None)
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        result = detector.compute_conmlds(v1, v2)
        assert abs(result - 1.0) < 1e-5, f"Expected ~1.0, got {result}"

    def test_antiparallel_vectors_returns_two(self):
        """Antiparallel vectors → cosine distance = 2.0 (cosine similarity = -1).

        WHY: vectors pointing in exactly opposite directions have cosine similarity
        = -1, so cosine distance = 1 - (-1) = 2.  This is the theoretical maximum
        for ConMLDS on a single pair.

        Spec: REQ-VERIFY-151-2
        """
        from carnot.cascade.tier0f_cocoa import CoCoADetector

        detector = CoCoADetector(model=None, tokenizer=None)
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0], dtype=np.float32)
        result = detector.compute_conmlds(v1, v2)
        assert abs(result - 2.0) < 1e-5, f"Expected ~2.0, got {result}"

    def test_zero_norm_vector_returns_zero(self):
        """Zero-norm guard: if either vector is zero, returns 0.0 without crashing.

        WHY: padding-only inputs can produce near-zero hidden states.  We guard
        against division by zero and return 0 (no disagreement signal) rather than
        NaN or infinity.

        Spec: REQ-VERIFY-151-2
        """
        from carnot.cascade.tier0f_cocoa import CoCoADetector

        detector = CoCoADetector(model=None, tokenizer=None)
        v_zero = np.zeros(4, dtype=np.float32)
        v_nonzero = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = detector.compute_conmlds(v_zero, v_nonzero)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Tests for CoCoADetector.extract_hidden_states  (REQ-VERIFY-151-1)
# ---------------------------------------------------------------------------


class TestExtractHiddenStates:
    """Unit tests for multi-layer hidden-state extraction.  Spec: REQ-VERIFY-151-1."""

    def test_extract_returns_correct_shape_per_layer(self):
        """extract_hidden_states produces a (hidden_dim,) array per requested layer.

        WHY: the method must return one flat vector per layer so compute_conmlds
        can compute cosine distances.  Shape (hidden_dim,) is the contract.

        Spec: REQ-VERIFY-151-1
        """
        import torch  # noqa: PLC0415
        from carnot.cascade.tier0f_cocoa import CoCoADetector

        # Build a real-ish mock that returns tensors of the right shape.
        layer_states = {
            8: np.random.randn(HIDDEN_DIM).astype(np.float32),
            16: np.random.randn(HIDDEN_DIM).astype(np.float32),
        }

        mock_output = MagicMock()
        # hidden_states[layer+1] must be shape (1, seq_len, hidden_dim)
        hs = {}
        for li, v in layer_states.items():
            hs[li] = torch.tensor(v).unsqueeze(0).unsqueeze(0)

        # Build a tuple large enough (slot 0=embedding, slot li+1=layer li)
        slots: list[object] = [torch.zeros(1, 1, HIDDEN_DIM)] * 20
        for li, t in hs.items():
            slots[li + 1] = t
        mock_output.hidden_states = tuple(slots)

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        mock_tokenizer = MagicMock()
        inputs_mock = MagicMock()
        inputs_mock.to = lambda d: inputs_mock  # noqa: ARG005
        mock_tokenizer.return_value = inputs_mock

        detector = CoCoADetector(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )
        result = detector.extract_hidden_states("hello world", [8, 16])
        assert 8 in result and 16 in result
        assert result[8].shape == (HIDDEN_DIM,)
        assert result[16].shape == (HIDDEN_DIM,)

    def test_extracted_values_match_injected_states(self):
        """Extracted hidden states exactly match what the mock model returns.

        WHY: verifies the indexing logic hidden_states[layer_idx + 1][0, -1, :]
        is correct.  A bug in the +1 offset would return the wrong layer's values.

        Spec: REQ-VERIFY-151-1
        """
        import torch  # noqa: PLC0415
        from carnot.cascade.tier0f_cocoa import CoCoADetector

        expected_vec = np.ones(HIDDEN_DIM, dtype=np.float32) * 42.0
        layer_idx = 12

        mock_output = MagicMock()
        slots: list[object] = [torch.zeros(1, 1, HIDDEN_DIM)] * 20
        slots[layer_idx + 1] = torch.tensor(expected_vec).unsqueeze(0).unsqueeze(0)
        mock_output.hidden_states = tuple(slots)

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_tokenizer = MagicMock()
        inputs_mock = MagicMock()
        inputs_mock.to = lambda d: inputs_mock  # noqa: ARG005
        mock_tokenizer.return_value = inputs_mock

        detector = CoCoADetector(model=mock_model, tokenizer=mock_tokenizer, device="cpu")
        result = detector.extract_hidden_states("test text", [layer_idx])
        np.testing.assert_allclose(result[layer_idx], expected_vec, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests for CoCoADetector.score and calibrate  (REQ-VERIFY-151, REQ-VERIFY-151-4)
# ---------------------------------------------------------------------------


class TestCoCoAScore:
    """Tests for the full scoring pipeline.  Spec: REQ-VERIFY-151, REQ-VERIFY-151-4."""

    def _build_detector_with_fixed_states(
        self,
        early_vec: np.ndarray,
        late_vec: np.ndarray,
        threshold: float | None = None,
    ):
        """Build a CoCoADetector whose extract_hidden_states is patched to return fixed vecs."""
        from unittest.mock import patch  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from carnot.cascade.tier0f_cocoa import CoCoADetector

        early_layers = (8,)
        late_layers = (16,)

        # Build mock model/tokenizer
        mock_output = MagicMock()
        slots: list[object] = [torch.zeros(1, 1, HIDDEN_DIM)] * 20
        slots[9] = torch.tensor(early_vec).unsqueeze(0).unsqueeze(0)   # layer 8 → slot 9
        slots[17] = torch.tensor(late_vec).unsqueeze(0).unsqueeze(0)   # layer 16 → slot 17
        mock_output.hidden_states = tuple(slots)

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_tokenizer = MagicMock()
        inputs_mock = MagicMock()
        inputs_mock.to = lambda d: inputs_mock  # noqa: ARG005
        mock_tokenizer.return_value = inputs_mock

        detector = CoCoADetector(
            model=mock_model,
            tokenizer=mock_tokenizer,
            early_layers=early_layers,
            late_layers=late_layers,
            threshold=threshold,
            device="cpu",
        )
        return detector

    def test_score_identical_layers_returns_zero_conmlds(self):
        """When early and late hidden states are identical, score returns ~0 ConMLDS.

        Spec: REQ-VERIFY-151, SCENARIO-VERIFY-201
        """
        vec = np.ones(HIDDEN_DIM, dtype=np.float32)
        detector = self._build_detector_with_fixed_states(vec, vec, threshold=0.5)
        conmlds, is_unstable = detector.score("some text")
        assert abs(conmlds) < 1e-5
        assert is_unstable is False  # 0 < 0.5

    def test_score_orthogonal_layers_returns_high_conmlds(self):
        """When early and late hidden states are orthogonal, ConMLDS is near 1.0.

        Spec: REQ-VERIFY-151, SCENARIO-VERIFY-202
        """
        v1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        v2[1] = 1.0
        detector = self._build_detector_with_fixed_states(v1, v2, threshold=0.5)
        conmlds, is_unstable = detector.score("some text")
        assert abs(conmlds - 1.0) < 1e-5
        assert is_unstable is True  # 1.0 > 0.5

    def test_score_returns_none_is_unstable_when_no_threshold(self):
        """Without a calibrated threshold, is_unstable is None.

        WHY: the detector is usable for pure scoring (e.g., compute AUC) even
        before calibration.  is_unstable=None signals "not yet calibrated".

        Spec: REQ-VERIFY-151-4
        """
        vec = np.ones(HIDDEN_DIM, dtype=np.float32)
        detector = self._build_detector_with_fixed_states(vec, vec, threshold=None)
        _, is_unstable = detector.score("some text")
        assert is_unstable is None


# ---------------------------------------------------------------------------
# Tests for CascadeRouter Tier 0f integration  (REQ-VERIFY-152)
# ---------------------------------------------------------------------------


class TestCascadeRouterTier0f:
    """Tests that Tier 0f advisory wiring works correctly in CascadeRouter.

    Spec: REQ-VERIFY-152, REQ-VERIFY-152-1, REQ-VERIFY-152-2, REQ-VERIFY-152-3
    """

    def _make_router(self, cocoa_score: float, cocoa_is_unstable: bool):
        """Build a CascadeRouter with a mocked CoCoADetector returning fixed score."""
        from carnot.cascade.cascade_router import CascadeRouter

        mock_cocoa = MagicMock()
        mock_cocoa.score.return_value = (cocoa_score, cocoa_is_unstable)

        router = CascadeRouter(
            eorm_fn=lambda q: 0.5,  # below threshold → Ising will run
            ising_fn=lambda q: True,
            eorm_ising_skip_threshold=0.92,
            tier0f_cocoa=mock_cocoa,
        )
        return router

    def test_tier0f_fields_recorded_in_metadata_when_stable(self):
        """When CoCoA scores a query as stable, metadata includes tier0f fields.

        Spec: REQ-VERIFY-152-1, REQ-VERIFY-152-2
        """
        router = self._make_router(cocoa_score=0.1, cocoa_is_unstable=False)
        result = router.route("What is 2+2?")
        assert "tier0f_conmlds" in result.metadata
        assert "tier0f_is_unstable" in result.metadata
        assert abs(result.metadata["tier0f_conmlds"] - 0.1) < 1e-6
        assert result.metadata["tier0f_is_unstable"] is False

    def test_tier0f_fields_recorded_in_metadata_when_unstable(self):
        """When CoCoA flags a query as unstable, metadata records is_unstable=True.

        Spec: REQ-VERIFY-152-1, REQ-VERIFY-152-2
        """
        router = self._make_router(cocoa_score=0.9, cocoa_is_unstable=True)
        result = router.route("Who is the current president of Mars?")
        assert result.metadata["tier0f_conmlds"] == pytest.approx(0.9)
        assert result.metadata["tier0f_is_unstable"] is True

    def test_tier0f_does_not_change_verdict(self):
        """Tier 0f MUST NOT short-circuit the cascade regardless of is_unstable.

        When CoCoA says is_unstable=True and EORM says confidence=0.5 (below skip
        threshold), Ising still runs and the verdict is 'verified_full' (not
        'safety_violation' or any Tier 0f derived verdict).

        Spec: REQ-VERIFY-152-3
        """
        router = self._make_router(cocoa_score=0.99, cocoa_is_unstable=True)
        result = router.route("Any query text")
        # Ising fn returns True, so verdict should be verified_full (Ising ran).
        assert result.verdict == "verified_full"
        assert result.verified is True

    def test_tier0f_absent_when_not_wired(self):
        """When tier0f_cocoa is not supplied, tier0f_* keys are absent from metadata.

        WHY: backwards compatibility.  Callers that do not supply tier0f_cocoa must
        not receive unexpected keys in RouteResult.metadata.

        Spec: REQ-VERIFY-152
        """
        from carnot.cascade.cascade_router import CascadeRouter

        router = CascadeRouter(
            eorm_fn=lambda q: 0.5,
            ising_fn=lambda q: True,
        )
        result = router.route("hello")
        assert "tier0f_conmlds" not in result.metadata
        assert "tier0f_is_unstable" not in result.metadata

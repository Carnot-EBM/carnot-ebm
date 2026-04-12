"""Tests for multi-concept contrastive vectors.

Spec coverage: REQ-INFER-016
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from carnot.embeddings.concept_vectors import (
    CONCEPT_PROMPTS,
    _compute_contrastive_vectors,
    best_concept_for_detection,
    concept_energy,
    find_concept_vectors,
)


# ---------------------------------------------------------------------------
# CONCEPT_PROMPTS tests
# ---------------------------------------------------------------------------


class TestConceptPrompts:
    """Tests for REQ-INFER-016: Default concept prompts."""

    def test_contains_expected_concepts(self) -> None:
        """REQ-INFER-016: CONCEPT_PROMPTS has all required concept keys."""
        expected = {"certain", "uncertain", "confabulating", "reasoning", "memorized"}
        assert set(CONCEPT_PROMPTS.keys()) == expected

    def test_all_prompts_are_strings(self) -> None:
        """REQ-INFER-016: All prompt values are non-empty strings."""
        for name, prompt in CONCEPT_PROMPTS.items():
            assert isinstance(prompt, str), f"{name} prompt is not a string"
            assert len(prompt) > 0, f"{name} prompt is empty"


# ---------------------------------------------------------------------------
# _compute_contrastive_vectors tests
# ---------------------------------------------------------------------------


class TestComputeContrastiveVectors:
    """Tests for REQ-INFER-016: Contrastive vector computation."""

    def test_returns_dict_with_concept_keys(self) -> None:
        """REQ-INFER-016: Returns a dict keyed by concept name."""
        key = jrandom.PRNGKey(0)
        concept_acts = {
            "a": [jrandom.normal(key, (8,)), jrandom.normal(key, (8,))],
            "b": [jrandom.normal(key, (8,)) + 3.0],
        }
        result = _compute_contrastive_vectors(concept_acts)
        assert set(result.keys()) == {"a", "b"}

    def test_vectors_are_unit_normalized(self) -> None:
        """REQ-INFER-016: All contrastive vectors have unit norm."""
        key = jrandom.PRNGKey(1)
        k1, k2, k3 = jrandom.split(key, 3)
        concept_acts = {
            "x": [jrandom.normal(k1, (16,)) for _ in range(5)],
            "y": [jrandom.normal(k2, (16,)) + 5.0 for _ in range(5)],
            "z": [jrandom.normal(k3, (16,)) - 3.0 for _ in range(5)],
        }
        result = _compute_contrastive_vectors(concept_acts)

        for name, vec in result.items():
            norm = float(jnp.linalg.norm(vec))
            np.testing.assert_allclose(norm, 1.0, atol=1e-5, err_msg=f"{name}")

    def test_single_concept_handled(self) -> None:
        """REQ-INFER-016: Single concept returns zero-ish vector gracefully."""
        concept_acts = {"only": [jnp.ones(4), jnp.ones(4) * 2]}
        result = _compute_contrastive_vectors(concept_acts)
        assert "only" in result
        assert result["only"].shape == (4,)

    def test_opposite_concepts_produce_opposing_directions(self) -> None:
        """REQ-INFER-016: Two well-separated concepts produce anti-correlated vectors."""
        n = 20
        concept_acts = {
            "positive": [jnp.array([5.0, 0.0]) for _ in range(n)],
            "negative": [jnp.array([-5.0, 0.0]) for _ in range(n)],
        }
        result = _compute_contrastive_vectors(concept_acts)

        # The two direction vectors should be roughly opposite.
        cos_sim = float(jnp.dot(result["positive"], result["negative"]))
        assert cos_sim < -0.5, f"Expected opposing directions, got cos_sim={cos_sim}"


# ---------------------------------------------------------------------------
# concept_energy tests
# ---------------------------------------------------------------------------


class TestConceptEnergy:
    """Tests for REQ-INFER-016: Per-concept energy scoring."""

    def test_returns_dict_with_concept_keys(self) -> None:
        """REQ-INFER-016: Returns energy for each concept."""
        vectors = {
            "a": jnp.array([1.0, 0.0]),
            "b": jnp.array([0.0, 1.0]),
        }
        act = jnp.array([3.0, 4.0])
        result = concept_energy(act, vectors)

        assert set(result.keys()) == {"a", "b"}

    def test_energies_are_floats(self) -> None:
        """REQ-INFER-016: All energy values are Python floats."""
        vectors = {"x": jnp.array([1.0, 0.0, 0.0])}
        act = jnp.array([2.0, 0.0, 0.0])
        result = concept_energy(act, vectors)
        assert isinstance(result["x"], float)

    def test_aligned_activation_has_positive_energy(self) -> None:
        """REQ-INFER-016: Activation aligned with concept gives positive energy."""
        vectors = {"certain": jnp.array([1.0, 0.0])}
        act = jnp.array([5.0, 0.0])
        result = concept_energy(act, vectors)
        assert result["certain"] > 0.0

    def test_orthogonal_activation_has_zero_energy(self) -> None:
        """REQ-INFER-016: Activation orthogonal to concept gives zero energy."""
        vectors = {"x": jnp.array([1.0, 0.0])}
        act = jnp.array([0.0, 5.0])
        result = concept_energy(act, vectors)
        np.testing.assert_allclose(result["x"], 0.0, atol=1e-6)

    def test_opposing_activation_has_negative_energy(self) -> None:
        """REQ-INFER-016: Activation opposing concept gives negative energy."""
        vectors = {"x": jnp.array([1.0, 0.0])}
        act = jnp.array([-3.0, 0.0])
        result = concept_energy(act, vectors)
        assert result["x"] < 0.0

    def test_multiple_concepts_independent(self) -> None:
        """REQ-INFER-016: Each concept energy is independent of others."""
        vectors = {
            "a": jnp.array([1.0, 0.0, 0.0]),
            "b": jnp.array([0.0, 1.0, 0.0]),
            "c": jnp.array([0.0, 0.0, 1.0]),
        }
        act = jnp.array([1.0, 2.0, 3.0])
        result = concept_energy(act, vectors)

        np.testing.assert_allclose(result["a"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["b"], 2.0, atol=1e-6)
        np.testing.assert_allclose(result["c"], 3.0, atol=1e-6)

    def test_empty_concepts(self) -> None:
        """REQ-INFER-016: Empty concept dict returns empty result."""
        result = concept_energy(jnp.ones(4), {})
        assert result == {}


# ---------------------------------------------------------------------------
# best_concept_for_detection tests
# ---------------------------------------------------------------------------


class TestBestConceptForDetection:
    """Tests for REQ-INFER-016: Finding best concept for hallucination detection."""

    def test_returns_string(self) -> None:
        """REQ-INFER-016: Returns a concept name (string)."""
        vectors = {
            "a": jnp.array([1.0, 0.0]),
            "b": jnp.array([0.0, 1.0]),
        }
        correct = [jnp.array([1.0, 0.0])]
        halluc = [jnp.array([-1.0, 0.0])]
        result = best_concept_for_detection(vectors, correct, halluc)
        assert isinstance(result, str)
        assert result in vectors

    def test_selects_most_discriminative_concept(self) -> None:
        """REQ-INFER-016: Selects concept with highest Fisher separation."""
        # Concept "a" separates well along dim 0, "b" does not.
        vectors = {
            "good": jnp.array([1.0, 0.0]),
            "bad": jnp.array([0.0, 1.0]),
        }
        # Correct and hallucinated differ only along dim 0.
        correct = [jnp.array([5.0, 0.0]), jnp.array([4.0, 0.1])]
        halluc = [jnp.array([-5.0, 0.0]), jnp.array([-4.0, -0.1])]

        result = best_concept_for_detection(vectors, correct, halluc)
        assert result == "good"

    def test_empty_concept_vectors_raises(self) -> None:
        """REQ-INFER-016: Empty concept_vectors raises ValueError."""
        with pytest.raises(ValueError, match="concept_vectors must not be empty"):
            best_concept_for_detection({}, [jnp.ones(4)], [jnp.ones(4)])

    def test_empty_correct_acts_raises(self) -> None:
        """REQ-INFER-016: Empty correct_acts raises ValueError."""
        with pytest.raises(ValueError, match="correct_acts must not be empty"):
            best_concept_for_detection(
                {"a": jnp.ones(4)}, [], [jnp.ones(4)]
            )

    def test_empty_hallucinated_acts_raises(self) -> None:
        """REQ-INFER-016: Empty hallucinated_acts raises ValueError."""
        with pytest.raises(ValueError, match="hallucinated_acts must not be empty"):
            best_concept_for_detection(
                {"a": jnp.ones(4)}, [jnp.ones(4)], []
            )

    def test_single_concept_returns_it(self) -> None:
        """REQ-INFER-016: Single concept always returns that concept."""
        vectors = {"only": jnp.array([1.0, 0.0])}
        correct = [jnp.array([1.0, 0.0])]
        halluc = [jnp.array([-1.0, 0.0])]
        result = best_concept_for_detection(vectors, correct, halluc)
        assert result == "only"

    def test_handles_identical_distributions(self) -> None:
        """REQ-INFER-016: Returns a concept even when distributions overlap."""
        vectors = {
            "a": jnp.array([1.0, 0.0]),
            "b": jnp.array([0.0, 1.0]),
        }
        # Same activations for both classes => zero separation.
        act = [jnp.array([1.0, 1.0])]
        result = best_concept_for_detection(vectors, act, act)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# find_concept_vectors tests (with mocks)
# ---------------------------------------------------------------------------


class TestFindConceptVectors:
    """Tests for REQ-INFER-016: End-to-end concept vector discovery."""

    def _build_concept_mocks(self, hidden_dim: int = 8, seq_len: int = 3):
        """Build mocks for find_concept_vectors.

        REQ-INFER-016: Mocked transformer for concept vector tests.
        """
        mock_torch = MagicMock()

        # Mock model with generate and hookable layers.
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None

        # Create hookable layers.
        num_layers = 2
        fake_layers = []
        for _i in range(num_layers):
            layer = MagicMock()
            registered_hooks: list = []

            def make_register(hooks_list):
                def register_forward_hook(fn):
                    hooks_list.append(fn)
                    handle = MagicMock()
                    handle.remove = MagicMock()
                    return handle

                return register_forward_hook

            layer.register_forward_hook = make_register(registered_hooks)
            fake_layers.append((layer, registered_hooks))

        layer_modules = [fl[0] for fl in fake_layers]
        mock_model.model.layers = layer_modules

        fake_param = MagicMock()
        fake_param.device = "cpu"
        mock_model.parameters = lambda: iter([fake_param])

        # model.generate returns fake tokens.
        mock_model.generate.return_value = np.array([[1, 2, 3]])

        # Mock forward calls to fire hooks with fake activations.
        call_count = [0]

        def fake_forward(**kwargs):
            for i, (layer_mod, hooks) in enumerate(fake_layers):
                # Use different random state per concept to get varied activations.
                fake_hidden = np.random.RandomState(call_count[0] * 10 + i).randn(
                    1, seq_len, hidden_dim
                ).astype(np.float32)

                class FakeOutput:
                    def __init__(self, data):
                        self._data = data

                    def detach(self):
                        return self

                    def cpu(self):
                        return self

                    def numpy(self):
                        return self._data

                output_tensor = FakeOutput(fake_hidden)
                for hook_fn in hooks:
                    hook_fn(layer_mod, None, (output_tensor,))

            call_count[0] += 1
            return MagicMock()

        mock_model.side_effect = fake_forward

        # torch utilities.
        mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        # Tokenizer.
        mock_tokenizer = MagicMock()

        class FakeTensor:
            def __init__(self, data):
                self._data = data

            def to(self, device):
                return self

        mock_tokenizer.return_value = {
            "input_ids": FakeTensor(np.zeros((1, seq_len), dtype=np.int64)),
            "attention_mask": FakeTensor(np.ones((1, seq_len), dtype=np.int64)),
        }
        mock_tokenizer.decode.return_value = "Generated text."

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        return mock_torch, mock_transformers, mock_model, mock_tokenizer

    def test_returns_dict_with_concept_keys(self) -> None:
        """REQ-INFER-016: Returns vectors for each concept."""
        mock_torch, mock_transformers, model, tokenizer = self._build_concept_mocks()
        prompts = {"a": "prompt a", "b": "prompt b"}

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = find_concept_vectors(
                model, tokenizer, concept_prompts=prompts, n_samples=2
            )

        assert result is not None
        assert set(result.keys()) == {"a", "b"}

    def test_vectors_have_correct_shape(self) -> None:
        """REQ-INFER-016: Each concept vector has shape (hidden_dim,)."""
        mock_torch, mock_transformers, model, tokenizer = self._build_concept_mocks(
            hidden_dim=16
        )

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = find_concept_vectors(
                model, tokenizer,
                concept_prompts={"c": "prompt"},
                n_samples=3,
            )

        assert result is not None
        assert result["c"].shape == (16,)

    def test_uses_default_prompts_when_none(self) -> None:
        """REQ-INFER-016: Uses CONCEPT_PROMPTS when concept_prompts=None."""
        mock_torch, mock_transformers, model, tokenizer = self._build_concept_mocks()

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            result = find_concept_vectors(
                model, tokenizer, concept_prompts=None, n_samples=1
            )

        assert result is not None
        # Should have all default concept keys.
        assert set(result.keys()) == set(CONCEPT_PROMPTS.keys())

    def test_returns_none_when_no_activations(self) -> None:
        """REQ-INFER-016: Returns None when extraction fails for all concepts."""
        mock_torch, mock_transformers, model, tokenizer = self._build_concept_mocks()

        # Make extract_layer_activations always return None.
        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            with patch(
                "carnot.embeddings.activation_extractor.extract_layer_activations",
                return_value=None,
            ):
                result = find_concept_vectors(
                    model, tokenizer,
                    concept_prompts={"x": "prompt"},
                    n_samples=2,
                )

        assert result is None


# ---------------------------------------------------------------------------
# Package export tests
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Tests for REQ-INFER-016: Package-level imports."""

    def test_concept_prompts_exported(self) -> None:
        """REQ-INFER-016: CONCEPT_PROMPTS importable from embeddings."""
        from carnot.embeddings import CONCEPT_PROMPTS as pkg_prompts

        assert pkg_prompts is CONCEPT_PROMPTS

    def test_find_concept_vectors_exported(self) -> None:
        """REQ-INFER-016: find_concept_vectors importable from embeddings."""
        from carnot.embeddings import find_concept_vectors as pkg_fn

        assert pkg_fn is find_concept_vectors

    def test_concept_energy_exported(self) -> None:
        """REQ-INFER-016: concept_energy importable from embeddings."""
        from carnot.embeddings import concept_energy as pkg_fn

        assert pkg_fn is concept_energy

    def test_best_concept_for_detection_exported(self) -> None:
        """REQ-INFER-016: best_concept_for_detection importable from embeddings."""
        from carnot.embeddings import best_concept_for_detection as pkg_fn

        assert pkg_fn is best_concept_for_detection

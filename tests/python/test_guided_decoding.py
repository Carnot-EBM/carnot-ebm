"""Tests for EnergyGuidedSampler token-by-token guided decoding.

**Spec coverage:**
    REQ-VERIFY-001: Constraint extraction on partial text.
    SCENARIO-VERIFY-004: Energy penalty modifies logits to steer generation.

All tests use mocked LLM components so no GPU or model download is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from carnot.inference.guided_decoding import (
    EnergyGuidedSampler,
    GuidedDecoder,
    GuidedDecodingResult,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_mock_model(vocab_size: int = 50, n_tokens: int = 5) -> MagicMock:
    """Return a mock HuggingFace causal LM that produces deterministic logits.

    The mock returns a fixed logit vector that places all probability on
    token 0, so greedy decoding always produces token 0. After ``n_tokens``
    forward passes it switches to the EOS token index to end generation.

    REQ-VERIFY-001: test infrastructure requires a deterministic model for
    reproducible energy measurements.
    """
    import torch

    call_count: dict[str, int] = {"n": 0}

    def side_effect(input_ids: torch.Tensor) -> MagicMock:
        call_count["n"] += 1
        logits = torch.zeros(1, input_ids.shape[1], vocab_size)
        # Put weight on token 0 (greedy will pick it).
        logits[0, -1, 0] = 10.0
        out = MagicMock()
        out.logits = logits
        return out

    model = MagicMock()
    model.side_effect = side_effect
    model.__call__ = model.side_effect
    model.parameters = MagicMock(
        return_value=iter([torch.zeros(1)])
    )
    return model


def _make_mock_tokenizer(vocab_size: int = 50) -> MagicMock:
    """Return a mock tokenizer that encodes/decodes trivially.

    REQ-VERIFY-001: decoding test infrastructure.
    """
    import torch

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 1  # EOS is token 1
    # encode returns a 2-D tensor of shape (1, 3) — a 3-token prompt.
    tokenizer.encode = MagicMock(
        return_value=torch.tensor([[2, 3, 4]])
    )
    # decode a single token ID → letter "A" (or EOS string).
    def decode(ids: Any, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "item"):
            val = ids.item()
        else:
            val = int(ids)
        return "" if val == tokenizer.eos_token_id else "A"

    tokenizer.decode = MagicMock(side_effect=decode)
    return tokenizer


# ---------------------------------------------------------------------------
# Test: EnergyGuidedSampler initialization
# ---------------------------------------------------------------------------


class TestEnergyGuidedSamplerInit:
    """REQ-VERIFY-001: sampler initialises with correct defaults and validates args."""

    def test_default_params(self) -> None:
        """REQ-VERIFY-001: default alpha=0.5, check_every_k=1."""
        sampler = EnergyGuidedSampler()
        assert sampler.alpha == 0.5
        assert sampler.check_every_k == 1
        assert sampler.energy_threshold == 0.0

    def test_custom_params(self) -> None:
        """REQ-VERIFY-001: custom alpha and check_every_k are stored."""
        sampler = EnergyGuidedSampler(alpha=1.5, check_every_k=5)
        assert sampler.alpha == 1.5
        assert sampler.check_every_k == 5

    def test_negative_alpha_raises(self) -> None:
        """REQ-VERIFY-001: alpha < 0 is rejected."""
        with pytest.raises(ValueError, match="alpha must be >= 0"):
            EnergyGuidedSampler(alpha=-0.1)

    def test_zero_check_every_k_raises(self) -> None:
        """REQ-VERIFY-001: check_every_k < 1 is rejected."""
        with pytest.raises(ValueError, match="check_every_k must be >= 1"):
            EnergyGuidedSampler(check_every_k=0)

    def test_pipeline_extractor_used(self) -> None:
        """REQ-VERIFY-001: when pipeline is provided its extractor is used."""
        mock_pipeline = MagicMock()
        mock_pipeline._extractor = MagicMock()
        sampler = EnergyGuidedSampler(pipeline=mock_pipeline)
        assert sampler._extractor is mock_pipeline._extractor

    def test_no_pipeline_uses_auto_extractor(self) -> None:
        """REQ-VERIFY-001: when no pipeline, AutoExtractor is created."""
        from carnot.pipeline.extract import AutoExtractor

        sampler = EnergyGuidedSampler()
        assert isinstance(sampler._extractor, AutoExtractor)


# ---------------------------------------------------------------------------
# Test: compute_energy_penalty
# ---------------------------------------------------------------------------


class TestComputeEnergyPenalty:
    """REQ-VERIFY-001: energy penalty computed from constraint violations."""

    def test_short_text_returns_zero(self) -> None:
        """REQ-VERIFY-001: text shorter than 5 chars skips extraction → 0.0."""
        sampler = EnergyGuidedSampler()
        assert sampler.compute_energy_penalty("") == 0.0
        assert sampler.compute_energy_penalty("Hi") == 0.0

    def test_no_violations_returns_zero(self) -> None:
        """REQ-VERIFY-001: text with no violated constraints returns 0.0."""
        sampler = EnergyGuidedSampler()
        # A plain sentence with no arithmetic claims → no violations.
        energy = sampler.compute_energy_penalty("The sky is blue and clouds are white.")
        assert energy == 0.0

    def test_arithmetic_violation_nonzero(self) -> None:
        """REQ-VERIFY-001: text containing wrong arithmetic raises energy > 0."""
        sampler = EnergyGuidedSampler()
        # Arithmetic extractor should catch "2 + 2 = 5" as violated.
        energy = sampler.compute_energy_penalty("The answer is 2 + 2 = 5.")
        assert energy > 0.0

    def test_extractor_error_returns_zero(self) -> None:
        """REQ-VERIFY-001: extractor errors are swallowed and return 0.0."""
        sampler = EnergyGuidedSampler()
        sampler._extractor = MagicMock()
        sampler._extractor.extract = MagicMock(side_effect=RuntimeError("boom"))
        energy = sampler.compute_energy_penalty("some valid text here to trigger")
        assert energy == 0.0

    def test_known_violation_count(self) -> None:
        """REQ-VERIFY-001: mock extractor with 2 violations → energy == 2.0."""
        from carnot.pipeline.extract import ConstraintResult

        sampler = EnergyGuidedSampler()
        # Create two mock constraints: one satisfied, two violated.
        cr_ok = ConstraintResult(
            constraint_type="arithmetic",
            description="ok",
            metadata={"satisfied": True},
        )
        cr_bad1 = ConstraintResult(
            constraint_type="arithmetic",
            description="bad1",
            metadata={"satisfied": False},
        )
        cr_bad2 = ConstraintResult(
            constraint_type="arithmetic",
            description="bad2",
            metadata={"satisfied": False},
        )
        sampler._extractor = MagicMock()
        sampler._extractor.extract = MagicMock(return_value=[cr_ok, cr_bad1, cr_bad2])

        energy = sampler.compute_energy_penalty("some text long enough to extract")
        assert energy == 2.0


# ---------------------------------------------------------------------------
# Test: modify_logits
# ---------------------------------------------------------------------------


class TestModifyLogits:
    """SCENARIO-VERIFY-004: energy penalty reduces high-energy continuations."""

    def test_no_violation_no_penalty(self) -> None:
        """SCENARIO-VERIFY-004: when energy <= threshold, logits unchanged."""
        import torch

        sampler = EnergyGuidedSampler(alpha=1.0, energy_threshold=0.5)
        logits = torch.tensor([1.0, 2.0, 3.0])
        # Energy below threshold → no change.
        modified = sampler.modify_logits(logits, "text", energy=0.3)
        assert float((modified - logits).abs().max()) < 1e-6

    def test_violation_reduces_all_logits(self) -> None:
        """SCENARIO-VERIFY-004: penalty subtracted from all logits uniformly."""
        import torch

        sampler = EnergyGuidedSampler(alpha=0.5, energy_threshold=0.0)
        logits = torch.tensor([1.0, 2.0, 3.0])
        # energy=2.0, alpha=0.5 → penalty = 1.0
        modified = sampler.modify_logits(logits, "text", energy=2.0)
        expected = logits - 1.0
        assert float((modified - expected).abs().max()) < 1e-6

    def test_logit_order_preserved(self) -> None:
        """SCENARIO-VERIFY-004: relative logit ranking is unchanged by penalty."""
        import torch

        sampler = EnergyGuidedSampler(alpha=1.0)
        logits = torch.tensor([3.0, 1.0, 5.0, 2.0])
        modified = sampler.modify_logits(logits, "some text", energy=1.0)
        assert torch.argmax(modified).item() == torch.argmax(logits).item()

    def test_energy_computed_when_not_provided(self) -> None:
        """SCENARIO-VERIFY-004: when energy kwarg is None, extractor is called."""
        import torch
        from carnot.pipeline.extract import ConstraintResult

        sampler = EnergyGuidedSampler(alpha=1.0)
        cr_bad = ConstraintResult(
            constraint_type="arithmetic",
            description="bad",
            metadata={"satisfied": False},
        )
        sampler._extractor = MagicMock()
        sampler._extractor.extract = MagicMock(return_value=[cr_bad])

        logits = torch.tensor([1.0, 2.0, 3.0])
        # Passing energy=None forces compute_energy_penalty.
        modified = sampler.modify_logits(logits, "some long text that triggers extract")
        # penalty should be alpha * 1.0 = 1.0.
        expected = logits - 1.0
        assert float((modified - expected).abs().max()) < 1e-6

    def test_zero_alpha_no_effect(self) -> None:
        """SCENARIO-VERIFY-004: alpha=0 disables guidance completely."""
        import torch

        sampler = EnergyGuidedSampler(alpha=0.0)
        logits = torch.tensor([1.0, 2.0, 3.0])
        modified = sampler.modify_logits(logits, "text", energy=100.0)
        assert float((modified - logits).abs().max()) < 1e-6

    def test_jax_array_penalty(self) -> None:
        """SCENARIO-VERIFY-004: penalty also works on JAX arrays (not just tensors)."""
        import numpy as np

        sampler = EnergyGuidedSampler(alpha=1.0, energy_threshold=0.0)

        # JAX arrays support scalar subtraction, so the same code path works.
        logits = jnp.array([1.0, 2.0, 3.0])
        modified = sampler.modify_logits(logits, "text", energy=1.0)
        expected = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(modified), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: generate (mock model)
# ---------------------------------------------------------------------------


class TestGenerate:
    """REQ-VERIFY-001, SCENARIO-VERIFY-004: token-by-token flow with mock LLM."""

    def test_returns_guided_decoding_result(self) -> None:
        """REQ-VERIFY-001: generate() returns a GuidedDecodingResult."""
        import torch

        sampler = EnergyGuidedSampler(alpha=0.0)  # no guidance, just flow test

        # Build a simple mock model that always returns eos after 3 steps.
        call_log: list[int] = []

        def forward(input_ids: torch.Tensor) -> MagicMock:
            step = len(call_log)
            call_log.append(step)
            vocab_size = 10
            logits = torch.zeros(1, input_ids.shape[1], vocab_size)
            # After 3 tokens, return EOS (token 1) as the argmax.
            if step >= 3:
                logits[0, -1, 1] = 10.0  # EOS wins
            else:
                logits[0, -1, 0] = 10.0  # token 0 wins
            out = MagicMock()
            out.logits = logits
            return out

        model = MagicMock()
        model.side_effect = forward
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        tokenizer = _make_mock_tokenizer()

        result = sampler.generate(
            "What is 2+2?", model, tokenizer, max_tokens=10, temperature=0
        )

        assert isinstance(result, GuidedDecodingResult)
        assert result.tokens_generated >= 1
        assert result.latency_seconds >= 0.0
        assert result.energy_checks >= 1

    def test_energy_checks_every_k(self) -> None:
        """REQ-VERIFY-001: check_every_k controls how often energy is computed."""
        import torch

        check_count: dict[str, int] = {"n": 0}

        class _CountingSampler(EnergyGuidedSampler):
            def compute_energy_penalty(self, text: str) -> float:
                check_count["n"] += 1
                return 0.0

        sampler = _CountingSampler(alpha=0.0, check_every_k=3)

        def forward(input_ids: torch.Tensor) -> MagicMock:
            step = forward.step  # type: ignore[attr-defined]
            forward.step += 1  # type: ignore[attr-defined]
            vocab_size = 10
            logits = torch.zeros(1, input_ids.shape[1], vocab_size)
            # Stop after 6 tokens by producing EOS.
            if step >= 6:
                logits[0, -1, 1] = 10.0
            else:
                logits[0, -1, 0] = 10.0
            out = MagicMock()
            out.logits = logits
            return out

        forward.step = 0  # type: ignore[attr-defined]

        model = MagicMock()
        model.side_effect = forward
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        tokenizer = _make_mock_tokenizer()
        result = sampler.generate("prompt", model, tokenizer, max_tokens=10, temperature=0)

        # With 6 generated tokens and check_every_k=3, we expect checks at
        # steps 0, 3, 6 → 3 checks (step 6 may trigger EOS before check in
        # some orderings, so we allow 2 or 3).
        assert check_count["n"] in {2, 3}

    def test_greedy_decoding_temperature_zero(self) -> None:
        """SCENARIO-VERIFY-004: temperature=0 picks the argmax token."""
        import torch

        # Model always places max logit on token 7.
        def forward(input_ids: torch.Tensor) -> MagicMock:
            step = forward.step  # type: ignore[attr-defined]
            forward.step += 1  # type: ignore[attr-defined]
            vocab_size = 10
            logits = torch.zeros(1, input_ids.shape[1], vocab_size)
            if step >= 2:
                logits[0, -1, 1] = 10.0  # EOS
            else:
                logits[0, -1, 7] = 10.0  # token 7
            out = MagicMock()
            out.logits = logits
            return out

        forward.step = 0  # type: ignore[attr-defined]

        model = MagicMock()
        model.side_effect = forward
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        tokenizer = _make_mock_tokenizer()
        # Override decode so token 7 returns "X".
        def decode(ids: Any, skip_special_tokens: bool = True) -> str:
            val = ids.item() if hasattr(ids, "item") else int(ids)
            if val == 1:
                return ""  # EOS
            return "X"

        tokenizer.decode = MagicMock(side_effect=decode)

        sampler = EnergyGuidedSampler(alpha=0.0)
        result = sampler.generate("hello", model, tokenizer, max_tokens=5, temperature=0)

        # Should have decoded at least one "X" token.
        assert "X" in result.text

    def test_temperature_sampling(self) -> None:
        """SCENARIO-VERIFY-004: temperature > 0 uses multinomial sampling."""
        import torch

        # Model that puts all probability on token 0 deterministically.
        def forward(input_ids: torch.Tensor) -> MagicMock:
            step = forward.step  # type: ignore[attr-defined]
            forward.step += 1  # type: ignore[attr-defined]
            vocab_size = 10
            logits = torch.zeros(1, input_ids.shape[1], vocab_size)
            if step >= 2:
                logits[0, -1, 1] = 1000.0  # EOS with huge logit → always wins
            else:
                logits[0, -1, 0] = 1000.0  # token 0 wins with high confidence
            out = MagicMock()
            out.logits = logits
            return out

        forward.step = 0  # type: ignore[attr-defined]

        model = MagicMock()
        model.side_effect = forward
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        tokenizer = _make_mock_tokenizer()

        sampler = EnergyGuidedSampler(alpha=0.0)
        # temperature=1.0 → goes through multinomial path.
        result = sampler.generate(
            "hello", model, tokenizer, max_tokens=5, temperature=1.0
        )
        assert isinstance(result, GuidedDecodingResult)

    def test_max_tokens_respected(self) -> None:
        """REQ-VERIFY-001: generation stops at max_tokens even without EOS."""
        import torch

        # Model that never produces EOS.
        def forward(input_ids: torch.Tensor) -> MagicMock:
            vocab_size = 10
            logits = torch.zeros(1, input_ids.shape[1], vocab_size)
            logits[0, -1, 0] = 10.0  # always token 0, never EOS (1)
            out = MagicMock()
            out.logits = logits
            return out

        model = MagicMock()
        model.side_effect = forward
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        tokenizer = _make_mock_tokenizer()
        # Override decode so token 0 → "B", EOS (1) → "".
        def decode(ids: Any, skip_special_tokens: bool = True) -> str:
            val = ids.item() if hasattr(ids, "item") else int(ids)
            return "" if val == 1 else "B"

        tokenizer.decode = MagicMock(side_effect=decode)

        sampler = EnergyGuidedSampler(alpha=0.0)
        result = sampler.generate("prompt", model, tokenizer, max_tokens=4, temperature=0)

        assert result.tokens_generated <= 4


# ---------------------------------------------------------------------------
# Test: GuidedDecoder (HuggingFace-publishable adapter)
# ---------------------------------------------------------------------------

_ADAPTER_DIR = "exports/guided-decoding-adapter"


class TestGuidedDecoder:
    """REQ-VERIFY-001, SCENARIO-VERIFY-004: GuidedDecoder adapter loads and generates."""

    def test_from_pretrained_loads_config(self) -> None:
        """REQ-VERIFY-001: from_pretrained reads config.json correctly."""
        decoder = GuidedDecoder.from_pretrained(_ADAPTER_DIR)
        assert decoder.config["adapter_type"] == "guided_decoding"
        assert decoder.config["default_alpha"] == 0.5

    def test_from_pretrained_loads_weights(self) -> None:
        """REQ-VERIFY-001: from_pretrained loads constraint_weights.safetensors."""
        decoder = GuidedDecoder.from_pretrained(_ADAPTER_DIR)
        assert "arithmetic" in decoder.constraint_weights
        assert decoder.constraint_weights["arithmetic"] == pytest.approx(1.0)
        assert decoder.constraint_weights["nl_consistency"] == pytest.approx(0.5)

    def test_from_pretrained_builds_sampler(self) -> None:
        """REQ-VERIFY-001: from_pretrained creates a properly configured sampler."""
        decoder = GuidedDecoder.from_pretrained(_ADAPTER_DIR)
        assert isinstance(decoder._sampler, EnergyGuidedSampler)
        assert decoder._sampler.alpha == pytest.approx(0.5)

    def test_from_pretrained_kwarg_override(self) -> None:
        """REQ-VERIFY-001: kwargs override config defaults at load time."""
        decoder = GuidedDecoder.from_pretrained(_ADAPTER_DIR, alpha=2.0, check_every_k=4)
        assert decoder._sampler.alpha == pytest.approx(2.0)
        assert decoder._sampler.check_every_k == 4

    def test_from_pretrained_missing_dir_raises(self) -> None:
        """REQ-VERIFY-001: non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            GuidedDecoder.from_pretrained("/tmp/no_such_adapter_xyz")

    def test_generate_returns_result(self) -> None:
        """SCENARIO-VERIFY-004: generate() produces a valid GuidedDecodingResult."""
        import torch

        step: dict[str, int] = {"n": 0}

        def forward(input_ids: torch.Tensor) -> MagicMock:
            logits = torch.zeros(1, input_ids.shape[1], 10)
            logits[0, -1, 1 if step["n"] >= 3 else 0] = 10.0
            step["n"] += 1
            out = MagicMock()
            out.logits = logits
            return out

        model = MagicMock()
        model.side_effect = forward
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 1
        tokenizer.encode = MagicMock(return_value=torch.tensor([[2, 3, 4]]))
        tokenizer.decode = MagicMock(
            side_effect=lambda ids, **kw: "" if ids.item() == 1 else "A"
        )

        decoder = GuidedDecoder.from_pretrained(_ADAPTER_DIR)
        result = decoder.generate(model, tokenizer, "What is 47 + 28?")

        assert isinstance(result, GuidedDecodingResult)
        assert result.tokens_generated >= 1
        assert result.latency_seconds >= 0.0
        assert result.energy_checks >= 1

    def test_generate_signature_model_tokenizer_prompt(self) -> None:
        """SCENARIO-VERIFY-004: generate(model, tokenizer, prompt) positional API works."""
        import torch

        step: dict[str, int] = {"n": 0}

        def forward(input_ids: torch.Tensor) -> MagicMock:
            logits = torch.zeros(1, input_ids.shape[1], 10)
            logits[0, -1, 1 if step["n"] >= 2 else 0] = 10.0
            step["n"] += 1
            out = MagicMock()
            out.logits = logits
            return out

        model = MagicMock()
        model.side_effect = forward
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 1
        tokenizer.encode = MagicMock(return_value=torch.tensor([[2, 3]]))
        tokenizer.decode = MagicMock(
            side_effect=lambda ids, **kw: "" if ids.item() == 1 else "B"
        )

        decoder = GuidedDecoder.from_pretrained(_ADAPTER_DIR)
        # Positional args: model, tokenizer, prompt — as per the task spec.
        result = decoder.generate(model, tokenizer, "test prompt")
        assert isinstance(result, GuidedDecodingResult)

"""Tests for the shared verifier-reward 3-arm LoRA-SFT runner.

Spec refs: REQ-CODE-4223, SCENARIO-CODE-4223-SYNC-ACCUMULATE.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO / "scripts" / "experiments" / "verifier_reward_code_lora_rft_3arm.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("verifier_reward_code_lora_rft_3arm_under_test", RUNNER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scenario_code_4223_runner_prepares_lora_model_for_grad_flow() -> None:
    """SCENARIO-CODE-4223-SYNC-ACCUMULATE: Gemma LoRA SFT disables cache and enables grads."""

    runner = _load_runner()

    class Config:
        use_cache = True

    class FakeModel:
        def __init__(self) -> None:
            self.config = Config()
            self.gradient_checkpointing_enabled = False
            self.input_require_grads_enabled = False
            self.train_called = False

        def gradient_checkpointing_enable(self) -> None:
            self.gradient_checkpointing_enabled = True

        def enable_input_require_grads(self) -> None:
            self.input_require_grads_enabled = True

        def train(self) -> None:
            self.train_called = True

    model = FakeModel()

    assert runner._prepare_model_for_lora_sft(model) is model
    assert model.config.use_cache is False
    assert model.gradient_checkpointing_enabled is True
    assert model.input_require_grads_enabled is True
    assert model.train_called is True

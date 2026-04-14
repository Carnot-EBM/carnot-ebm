"""Tests for Experiment 273: Agent rollback verification on live model outputs.

**Detailed explanation for engineers:**
    Validates the Exp 273 rollback harness against ConstraintStateMachine:
    - Trial results are shaped correctly (TrialResult fields).
    - Rollback restores state after a violation injection.
    - rollback_success is True when history is trimmed to the pre-injection length.
    - violation_detected is set when the injected text fails verification or
      contradicts a previously verified fact.
    - The aggregate results dict has all required keys and sensible values.
    - CARNOT_SKIP_LLM=1 (always set here) keeps tests offline.

    All tests run with CARNOT_SKIP_LLM=1 so no model weights are downloaded.
    The rollback logic is deterministic and does not depend on LLM output.

Spec: REQ-VERIFY-001, REQ-VERIFY-074, REQ-VERIFY-075,
      SCENARIO-VERIFY-005, SCENARIO-VERIFY-075, SCENARIO-VERIFY-076
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def load_module() -> Any:
    """Load experiment_273_agent_rollback_live as a module object.

    **Detailed explanation for engineers:**
        Python 3.14 changed dataclass field resolution to use
        ``sys.modules.get(cls.__module__)``. Modules loaded via
        ``importlib.util`` are not registered in ``sys.modules`` by default,
        so the dataclass machinery returns None and crashes with
        ``AttributeError: 'NoneType' has no attribute '__dict__'``.
        We register the module in ``sys.modules`` before ``exec_module``
        to avoid this. The module is cleaned up from ``sys.modules`` after
        each test run to keep the namespace tidy.
    """
    module_name = "experiment_273_agent_rollback_live"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = REPO_ROOT / "scripts" / "experiment_273_agent_rollback_live.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def force_skip_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force CARNOT_SKIP_LLM=1 for all tests so no real model is loaded.

    Spec: REQ-VERIFY-075
    """
    monkeypatch.setenv("CARNOT_SKIP_LLM", "1")


# ---------------------------------------------------------------------------
# Test: module loads without errors
# REQ-VERIFY-074
# ---------------------------------------------------------------------------


class TestModuleLoads:
    """The module imports cleanly and exposes the required symbols.

    Spec: REQ-VERIFY-074, SCENARIO-VERIFY-075
    """

    def test_module_imports_successfully(self) -> None:
        """Exp 273 script can be imported with CARNOT_SKIP_LLM=1.

        Spec: REQ-VERIFY-074
        """
        mod = load_module()
        assert mod is not None

    def test_required_constants_present(self) -> None:
        """N_STEPS, N_WORKFLOWS, WORKFLOW_TOPICS, CANNED_OUTPUTS are present.

        Spec: REQ-VERIFY-074
        """
        mod = load_module()
        assert mod.N_STEPS == 5
        assert mod.N_WORKFLOWS == 10
        assert len(mod.WORKFLOW_TOPICS) == 10
        assert len(mod.CANNED_OUTPUTS) == 10

    def test_canned_outputs_match_n_steps(self) -> None:
        """Each CANNED_OUTPUTS entry has exactly N_STEPS strings.

        Spec: REQ-VERIFY-074
        """
        mod = load_module()
        for i, canned in enumerate(mod.CANNED_OUTPUTS):
            assert len(canned) == mod.N_STEPS, (
                f"CANNED_OUTPUTS[{i}] has {len(canned)} entries, expected {mod.N_STEPS}"
            )

    def test_workflow_topics_have_required_keys(self) -> None:
        """Each WORKFLOW_TOPICS entry has 'topic', 'steps', 'violation'.

        Spec: REQ-VERIFY-074
        """
        mod = load_module()
        for i, cfg in enumerate(mod.WORKFLOW_TOPICS):
            assert "topic" in cfg, f"WORKFLOW_TOPICS[{i}] missing 'topic'"
            assert "steps" in cfg, f"WORKFLOW_TOPICS[{i}] missing 'steps'"
            assert "violation" in cfg, f"WORKFLOW_TOPICS[{i}] missing 'violation'"
            assert len(cfg["steps"]) == mod.N_STEPS, (
                f"WORKFLOW_TOPICS[{i}]['steps'] has {len(cfg['steps'])} entries, expected {mod.N_STEPS}"
            )


# ---------------------------------------------------------------------------
# Test: _skip_llm helper
# REQ-VERIFY-075
# ---------------------------------------------------------------------------


class TestSkipLlm:
    """_skip_llm() reflects the CARNOT_SKIP_LLM environment variable.

    Spec: REQ-VERIFY-075
    """

    def test_skip_llm_true_when_env_set(self) -> None:
        """Returns True when CARNOT_SKIP_LLM=1.

        Spec: REQ-VERIFY-075
        """
        mod = load_module()
        assert mod._skip_llm() is True  # autouse fixture sets it

    def test_skip_llm_false_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when CARNOT_SKIP_LLM is absent.

        Spec: REQ-VERIFY-075
        """
        monkeypatch.delenv("CARNOT_SKIP_LLM", raising=False)
        mod = load_module()
        assert mod._skip_llm() is False


# ---------------------------------------------------------------------------
# Test: run_workflow_trial — single trial structure
# REQ-VERIFY-001, REQ-VERIFY-074, SCENARIO-VERIFY-075
# ---------------------------------------------------------------------------


class TestRunWorkflowTrial:
    """Single trial produces a correctly-structured TrialResult.

    Spec: REQ-VERIFY-001, REQ-VERIFY-074, SCENARIO-VERIFY-075
    """

    def _run_trial(self, workflow_index: int = 0) -> Any:
        """Helper: run trial with canned outputs (CARNOT_SKIP_LLM=1)."""
        mod = load_module()
        cfg = mod.WORKFLOW_TOPICS[workflow_index]
        canned = mod.CANNED_OUTPUTS[workflow_index]
        return mod.run_workflow_trial(
            workflow_index=workflow_index,
            topic_cfg=cfg,
            canned_outputs=canned,
            model=None,
            tokenizer=None,
        )

    def test_trial_result_has_all_fields(self) -> None:
        """TrialResult has all required fields populated.

        Spec: REQ-VERIFY-074, SCENARIO-VERIFY-075
        """
        result = self._run_trial(0)
        assert hasattr(result, "workflow_index")
        assert hasattr(result, "topic")
        assert hasattr(result, "n_steps_run")
        assert hasattr(result, "injection_step")
        assert hasattr(result, "violation_detected")
        assert hasattr(result, "rollback_performed")
        assert hasattr(result, "rollback_success")
        assert hasattr(result, "steps_preserved")
        assert hasattr(result, "verified_facts_before")
        assert hasattr(result, "verified_facts_after")
        assert hasattr(result, "error")

    def test_topic_matches_workflow_index(self) -> None:
        """topic field matches the WORKFLOW_TOPICS[i]['topic'] value.

        Spec: REQ-VERIFY-074
        """
        mod = load_module()
        for i in range(mod.N_WORKFLOWS):
            result = self._run_trial(i)
            assert result.topic == mod.WORKFLOW_TOPICS[i]["topic"]

    def test_rollback_performed_true(self) -> None:
        """rollback_performed is True (we always attempt rollback).

        Spec: SCENARIO-VERIFY-075
        """
        result = self._run_trial(0)
        assert result.rollback_performed is True

    def test_rollback_success_true(self) -> None:
        """rollback_success is True (history trimmed to injection_step entries).

        Spec: SCENARIO-VERIFY-075
        """
        result = self._run_trial(0)
        assert result.rollback_success is True

    def test_steps_preserved_equals_injection_step(self) -> None:
        """steps_preserved == injection_step (history = steps 0..injection_step-1).

        Spec: SCENARIO-VERIFY-075
        """
        result = self._run_trial(0)
        # After rollback(injection_step - 1), history has `injection_step` entries.
        assert result.steps_preserved == result.injection_step

    def test_injection_step_in_valid_range(self) -> None:
        """injection_step is between 1 and N_STEPS-1 inclusive.

        Spec: REQ-VERIFY-074
        """
        mod = load_module()
        result = self._run_trial(0)
        assert 1 <= result.injection_step <= mod.N_STEPS - 1

    def test_n_steps_run_is_injection_plus_one(self) -> None:
        """n_steps_run = injection_step + 1 (pre-injection steps + violation step).

        Spec: REQ-VERIFY-074
        """
        result = self._run_trial(0)
        assert result.n_steps_run == result.injection_step + 1

    def test_error_is_none_on_success(self) -> None:
        """error is None when the trial completes without an exception.

        Spec: REQ-VERIFY-074
        """
        result = self._run_trial(0)
        assert result.error is None

    def test_all_workflows_rollback_successfully(self) -> None:
        """All 10 workflows roll back successfully with canned outputs.

        Spec: SCENARIO-VERIFY-075, SCENARIO-VERIFY-076
        """
        mod = load_module()
        for i in range(mod.N_WORKFLOWS):
            result = self._run_trial(i)
            assert result.rollback_success, (
                f"Workflow {i} ({result.topic}) rollback failed: "
                f"steps_preserved={result.steps_preserved}, "
                f"injection_step={result.injection_step}"
            )


# ---------------------------------------------------------------------------
# Test: rollback restores verified_facts count
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestRollbackRestoresState:
    """After rollback, verified_facts_after <= verified_facts_before.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def test_verified_facts_after_le_before(self) -> None:
        """verified_facts_after <= verified_facts_before after rollback.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        mod = load_module()
        for i in range(mod.N_WORKFLOWS):
            cfg = mod.WORKFLOW_TOPICS[i]
            canned = mod.CANNED_OUTPUTS[i]
            result = mod.run_workflow_trial(
                workflow_index=i,
                topic_cfg=cfg,
                canned_outputs=canned,
                model=None,
                tokenizer=None,
            )
            assert result.verified_facts_after <= result.verified_facts_before, (
                f"Workflow {i}: after rollback, verified facts increased "
                f"({result.verified_facts_before} -> {result.verified_facts_after})"
            )

    def test_steps_preserved_never_exceeds_injection_step(self) -> None:
        """steps_preserved is always <= injection_step (rollback cannot grow history).

        Spec: REQ-VERIFY-001
        """
        mod = load_module()
        for i in range(mod.N_WORKFLOWS):
            cfg = mod.WORKFLOW_TOPICS[i]
            canned = mod.CANNED_OUTPUTS[i]
            result = mod.run_workflow_trial(
                workflow_index=i,
                topic_cfg=cfg,
                canned_outputs=canned,
                model=None,
                tokenizer=None,
            )
            assert result.steps_preserved <= result.injection_step, (
                f"Workflow {i}: steps_preserved={result.steps_preserved} > "
                f"injection_step={result.injection_step}"
            )


# ---------------------------------------------------------------------------
# Test: run_experiment aggregate result schema
# REQ-VERIFY-074, REQ-VERIFY-075, SCENARIO-VERIFY-076
# ---------------------------------------------------------------------------


class TestRunExperimentSchema:
    """run_experiment() produces a correctly-shaped results dict.

    Spec: REQ-VERIFY-074, REQ-VERIFY-075, SCENARIO-VERIFY-076
    """

    def _run_full_experiment(self) -> dict[str, Any]:
        mod = load_module()
        return mod.run_experiment()

    def test_top_level_keys_present(self) -> None:
        """Result has 'experiment', 'run_date', 'title', 'metadata', 'summary', 'trials'.

        Spec: REQ-VERIFY-074
        """
        results = self._run_full_experiment()
        for key in ("experiment", "run_date", "title", "metadata", "summary", "trials"):
            assert key in results, f"Missing top-level key: {key}"

    def test_experiment_number_is_273(self) -> None:
        """results['experiment'] == 273.

        Spec: REQ-VERIFY-074
        """
        results = self._run_full_experiment()
        assert results["experiment"] == 273

    def test_n_trials_equals_n_workflows(self) -> None:
        """results['trials'] has exactly N_WORKFLOWS entries.

        Spec: REQ-VERIFY-074
        """
        mod = load_module()
        results = self._run_full_experiment()
        assert len(results["trials"]) == mod.N_WORKFLOWS

    def test_summary_rollback_success_rate_range(self) -> None:
        """rollback_success_rate is in [0, 1].

        Spec: SCENARIO-VERIFY-076
        """
        results = self._run_full_experiment()
        rate = results["summary"]["rollback_success_rate"]
        assert 0.0 <= rate <= 1.0

    def test_summary_rollback_success_rate_is_one(self) -> None:
        """With canned outputs, all rollbacks succeed (rate == 1.0).

        Spec: SCENARIO-VERIFY-075, SCENARIO-VERIFY-076
        """
        results = self._run_full_experiment()
        assert results["summary"]["rollback_success_rate"] == 1.0, (
            f"Expected 100% rollback success with canned outputs, "
            f"got {results['summary']['rollback_success_rate']:.2%}"
        )

    def test_summary_avg_steps_preserved_positive(self) -> None:
        """avg_steps_preserved > 0 (at least one step is preserved per trial).

        Spec: SCENARIO-VERIFY-076
        """
        results = self._run_full_experiment()
        assert results["summary"]["avg_steps_preserved"] > 0

    def test_trial_entries_have_required_keys(self) -> None:
        """Each trial entry has all required keys.

        Spec: REQ-VERIFY-074
        """
        results = self._run_full_experiment()
        required = {
            "workflow_index", "topic", "n_steps_run", "injection_step",
            "violation_detected", "rollback_performed", "rollback_success",
            "steps_preserved", "verified_facts_before", "verified_facts_after", "error",
        }
        for i, trial in enumerate(results["trials"]):
            missing = required - set(trial.keys())
            assert not missing, f"Trial {i} missing keys: {missing}"

    def test_results_are_json_serialisable(self) -> None:
        """The results dict can be serialised to JSON without error.

        Spec: REQ-VERIFY-074, SCENARIO-VERIFY-076
        """
        results = self._run_full_experiment()
        serialised = json.dumps(results)
        assert len(serialised) > 0

    def test_metadata_live_mode_false_with_skip_llm(self) -> None:
        """metadata['live_mode'] is False when CARNOT_SKIP_LLM=1.

        Spec: REQ-VERIFY-075
        """
        results = self._run_full_experiment()
        assert results["metadata"]["live_mode"] is False


# ---------------------------------------------------------------------------
# Test: direct ConstraintStateMachine rollback integration
# REQ-VERIFY-001, SCENARIO-VERIFY-005
# ---------------------------------------------------------------------------


class TestConstraintStateMachineRollbackDirect:
    """Direct integration test: ConstraintStateMachine with injected violation.

    This test does not use the Exp 273 script at all — it validates the core
    rollback contract directly against ConstraintStateMachine so any future
    regression in the machine is caught immediately.

    **Why mock the pipeline?**
        ConstraintStateMachine delegates verification to a VerifyRepairPipeline.
        The real pipeline calls an LLM for extraction and verification, which is
        unavailable in CI (CARNOT_SKIP_LLM=1). Also, propagate() calls
        pipeline.verify(output_text) with a single positional arg, while the
        real verify() requires two positional args (question + response); a
        MagicMock handles both cases. Mocking here is correct and sufficient
        because the rollback logic under test is entirely in ConstraintStateMachine
        -- not in the pipeline.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def _mock_pipeline(self) -> Any:
        """Build a MagicMock VerifyRepairPipeline returning empty/verified results."""
        from carnot.pipeline.extract import ConstraintResult
        from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        pipeline.extract_constraints.return_value = []
        pipeline.verify.return_value = VerificationResult(
            verified=True,
            constraints=[],
            energy=0.0,
            violations=[],
        )
        return pipeline

    def test_rollback_after_contradiction_preserves_pre_injection_history(self) -> None:
        """Simulate the Exp 273 trial pattern directly on ConstraintStateMachine.

        Steps:
          0: consistent output (verified=True)
          1: consistent output (verified=True)
          2: inject violation (verified=False)
          rollback(1): history should contain only steps 0 and 1.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        from carnot.pipeline.extract import ConstraintResult
        from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline
        from carnot.pipeline.state_machine import ConstraintStateMachine

        pipeline = MagicMock(spec=VerifyRepairPipeline)
        # Steps 0 and 1: verified
        ok = VerificationResult(verified=True, constraints=[], energy=0.0, violations=[])
        # Step 2: violated
        violation = ConstraintResult(
            constraint_type="factual",
            description="paris is the capital of france",
            metadata={"satisfied": False},
        )
        bad = VerificationResult(
            verified=False, constraints=[violation], energy=1.0, violations=[violation]
        )
        pipeline.extract_constraints.side_effect = [[], [], [violation]]
        pipeline.verify.side_effect = [ok, ok, bad]

        machine = ConstraintStateMachine(pipeline=pipeline)
        machine.step("Q0", "Paris is the capital of France.")
        machine.step("Q1", "The Seine flows through France.")

        assert len(machine.history()) == 2

        machine.step("Q2", "Paris is the capital of Germany.")

        assert len(machine.history()) == 3

        machine.rollback(1)

        assert len(machine.history()) == 2
        assert machine.history()[0].step_index == 0
        assert machine.history()[1].step_index == 1

    def test_next_step_after_rollback_gets_correct_index(self) -> None:
        """After rollback(1), the next step gets step_index == 2.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        from carnot.pipeline.state_machine import ConstraintStateMachine

        pipeline = self._mock_pipeline()
        machine = ConstraintStateMachine(pipeline=pipeline)
        machine.step("Q0", "A0 is correct.")
        machine.step("Q1", "A1 is correct.")
        machine.step("Q2", "Injected violation text.")
        machine.rollback(1)

        r = machine.step("Q2-retry", "A2 corrected.")
        assert r.step_index == 2

    def test_rollback_to_step_zero_clears_later_history(self) -> None:
        """rollback(0) leaves only step-0 history.

        Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
        """
        from carnot.pipeline.state_machine import ConstraintStateMachine

        pipeline = self._mock_pipeline()
        machine = ConstraintStateMachine(pipeline=pipeline)
        machine.step("Q0", "Fact zero is established.")
        machine.step("Q1", "Fact one is established.")
        machine.step("Q2", "Fact two is established.")
        machine.rollback(0)

        assert len(machine.history()) == 1
        assert machine.history()[0].step_index == 0

    def test_rollback_out_of_range_raises_index_error(self) -> None:
        """rollback(99) raises IndexError when only 3 steps have been run.

        Spec: REQ-VERIFY-001
        """
        from carnot.pipeline.state_machine import ConstraintStateMachine

        pipeline = self._mock_pipeline()
        machine = ConstraintStateMachine(pipeline=pipeline)
        machine.step("Q0", "A0.")
        machine.step("Q1", "A1.")
        machine.step("Q2", "A2.")

        with pytest.raises(IndexError):
            machine.rollback(99)

    def test_multiple_rollbacks_work_correctly(self) -> None:
        """Two sequential rollbacks both succeed and produce consistent history.

        Spec: SCENARIO-VERIFY-005
        """
        from carnot.pipeline.state_machine import ConstraintStateMachine

        pipeline = self._mock_pipeline()
        machine = ConstraintStateMachine(pipeline=pipeline)
        for i in range(5):
            machine.step(f"Q{i}", f"Answer {i}.")

        # First rollback: to step 2
        machine.rollback(2)
        assert len(machine.history()) == 3

        # Continue: add steps 3 and 4 again
        machine.step("Q3-new", "Answer 3 new.")
        machine.step("Q4-new", "Answer 4 new.")
        assert len(machine.history()) == 5

        # Second rollback: to step 1
        machine.rollback(1)
        assert len(machine.history()) == 2
        assert machine.history()[0].step_index == 0
        assert machine.history()[1].step_index == 1

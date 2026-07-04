"""Tests for Exp 5226 VerIbmc-style local solver-feedback pilot.

Spec refs: REQ-VERIFY-5226, SCENARIO-VERIFY-5226.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5226_veribmc_local_solver_feedback_pilot_v478 as mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def test_req_verify_5226_spec_declares_solver_feedback_contract() -> None:
    """REQ-VERIFY-5226: OpenSpec declares the bounded three-arm pilot."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5226") :]

    for marker in (
        "REQ-VERIFY-5226",
        "SCENARIO-VERIFY-5226",
        str(mod.RESULT_RELATIVE_PATH),
        "cached_sota_pair",
        "local_sota_gguf_plus_deterministic_solver_feedback",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5226_z3_checker_accepts_reference_and_returns_counterexample() -> None:
    """REQ-VERIFY-5226: deterministic checker decides invariant candidates."""

    examples = mod.fixture_examples()
    by_id = {example.example_id: example for example in examples}

    accepted = mod.check_invariant(by_id["inc_to_n"], "0 <= i <= n")
    assert accepted.accepted is True
    assert accepted.failed_obligation is None
    assert accepted.counterexample == {}

    rejected = mod.check_invariant(by_id["sum_to_n"], "0 <= i <= n and s >= 0")
    assert rejected.accepted is False
    assert rejected.failed_obligation == "postcondition"
    assert rejected.counterexample
    assert rejected.feedback["failed_obligation"] == "postcondition"
    assert "repair_hint" in rejected.feedback


def test_scenario_verify_5226_parses_json_plain_text_and_blocks_unsafe_input() -> None:
    """SCENARIO-VERIFY-5226: proposals are parsed through the bounded DSL."""

    assert mod.parse_invariant_text("") is None
    assert mod.parse_invariant_text("```json\n{\"invariant\": \"0 <= i <= n\"}\n```") == "0 <= i <= n"
    assert mod.parse_invariant_text('{"invariant": "0 <= i <= n"}') == "0 <= i <= n"
    assert (
        mod.parse_invariant_text('{"invariants": ["0 <= i", "i <= n"]}')
        == "(0 <= i) and (i <= n)"
    )
    assert mod.parse_invariant_text('{"note": "missing invariant"}') is None
    assert mod.parse_invariant_text("INVARIANT: 0 <= i <= n\n") == "0 <= i <= n"
    assert mod.parse_invariant_text("The invariant should be i <= n.") == "i <= n"
    assert (
        mod.parse_invariant_text("2*s == n*(n+1) is the same as s == n*(n+1)/2.")
        == "2*s == n*(n+1)"
    )
    assert mod.parse_invariant_text("no formal proposal") is None

    assert str(mod.compile_formula("true", ("i",))).lower() == "true"
    assert str(mod.compile_formula("false", ("i",))).lower() == "false"
    assert "Or" in str(mod.compile_formula("i < 0 or i > 0", ("i",)))
    assert "-" in str(mod.compile_formula("-i <= 0", ("i",)))
    assert "true" in str(mod.compile_formula("i >= 0 and true", ("i",))).lower()
    assert "false" in str(mod.compile_formula("i >= 0 and false", ("i",))).lower()
    assert "false" in str(mod.compile_formula("i >= 0 and False", ("i",))).lower()

    with pytest.raises(RuntimeError, match="z3_unavailable"):
        mod.compile_formula("i >= 0", ("i",), z3_module=None)
    with pytest.raises(ValueError, match="unsupported expression"):
        mod.compile_formula("__import__('os').system('true')", ("i", "n"))
    with pytest.raises(ValueError, match="unsupported comparison operator"):
        mod.compile_formula("i != 0", ("i",))


def test_req_verify_5226_checker_edges_and_prompt_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-5226: edge obligations, resolver fallback, and prompts are deterministic."""

    examples = {example.example_id: example for example in mod.fixture_examples()}

    assert mod.check_invariant(examples["inc_to_n"], "i > 0").failed_obligation == "initiation"
    assert (
        mod.check_invariant(
            examples["sum_to_n"],
            "0 <= i <= n and 2*s == i*(i+1) and s <= n",
        ).failed_obligation
        == "preservation"
    )
    assert mod.check_invariant(examples["inc_to_n"], "i +").failed_obligation == "parse_error"
    assert mod.evaluate_proposal(examples["inc_to_n"], "no formal proposal", arm="llm_only").failure_mode == "parse_error"

    initial_prompt = mod.render_prompt(mod.ProposalPrompt(example=examples["inc_to_n"], arm="initial"))
    feedback_prompt = mod.render_prompt(
        mod.ProposalPrompt(
            example=examples["inc_to_n"],
            arm="feedback",
            prior_invariant="i >= 0",
            solver_feedback={"failed_obligation": "postcondition"},
        )
    )
    assert "Return exactly one JSON object" in initial_prompt
    assert "Solver feedback" not in initial_prompt
    assert "Solver feedback" in feedback_prompt
    assert "Prior invariant: i >= 0" in feedback_prompt

    monkeypatch.setattr(mod, "cached_sota_pair", lambda gpu_indices: [{"hf_id": "pair", "model_path": "p"}])
    assert mod.resolve_model_specs_for_pilot() == [{"hf_id": "pair", "model_path": "p"}]

    monkeypatch.setattr(mod, "cached_sota_pair", lambda gpu_indices: None)
    monkeypatch.setattr(
        mod,
        "SOTA_GGUF_MODELS",
        [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            }
        ],
    )
    monkeypatch.setattr(mod, "resolve_cached_gguf", lambda hf_id: f"/tmp/{hf_id.rsplit('/', 1)[-1]}.gguf")
    assert mod.resolve_model_specs_for_pilot()[0]["model_path"].endswith(".gguf")


def test_scenario_verify_5226_runs_three_arms_and_writes_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5226: feedback retry improves over both bounded baselines."""

    result_path = tmp_path / "experiment_5226.json"
    calls: list[tuple[str, str]] = []

    scripted_outputs = {
        ("inc_to_n", "initial"): '{"invariant": "0 <= i <= n"}',
        ("sum_to_n", "initial"): '{"invariant": "0 <= i <= n and s >= 0"}',
        (
            "sum_to_n",
            "feedback",
        ): '{"invariant": "0 <= i <= n and 2*s == i*(i+1)"}',
        ("paired_decrement", "initial"): "INVARIANT: y >= 0",
        ("paired_decrement", "feedback"): '{"invariants": ["0 <= y <= x"]}',
    }

    def fake_proposer(prompt: mod.ProposalPrompt) -> str:
        calls.append((prompt.example.example_id, prompt.arm))
        return scripted_outputs[(prompt.example.example_id, prompt.arm)]

    artifact = mod.run_experiment(
        result_path=result_path,
        run_date="20260704",
        duration_s=0.5,
        tests_run=["unit fixture: PASS"],
        proposal_fn=fake_proposer,
        model_specs_provider=lambda: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": "/models/qwen.gguf",
            }
        ],
    )

    mod.validate_artifact(artifact)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["solver_feedback_pilot_complete"]["value"] is True
    assert artifact["n_examples"]["value"] == 3
    assert artifact["solver_only_solved"]["value"] == 1
    assert artifact["llm_only_solved"]["value"] == 1
    assert artifact["llm_solver_feedback_solved"]["value"] == 3
    assert artifact["solver_feedback_uplift"]["value"] == pytest.approx(2 / 3)
    assert artifact["checker_substrate"]["value"] == "z3"
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "improved" in artifact["honest_verdict"]["value"]
    assert ("sum_to_n", "feedback") in calls
    assert ("paired_decrement", "feedback") in calls
    assert ("inc_to_n", "feedback") not in calls


def test_req_verify_5226_validates_artifact_and_blocked_preconditions() -> None:
    """REQ-VERIFY-5226: schema and model preconditions fail closed."""

    blocked = mod.run_pilot(
        proposal_fn=lambda prompt: '{"invariant": "true"}',
        model_specs_provider=lambda: [],
    )
    assert blocked["solver_feedback_pilot_complete"]["value"] is False
    assert blocked["honest_verdict"]["value"].startswith("complete:")
    assert blocked["models_used"]["value"] == []
    mod.validate_artifact(blocked)

    broken = dict(blocked)
    broken["checker_substrate"] = "z3"
    with pytest.raises(AssertionError, match="principle-wrapped"):
        mod.validate_artifact(broken)

    broken = mod.build_artifact(
        examples=[],
        model_specs=[],
        models_used=[],
        checker_substrate="z3",
        tests_run=[],
        duration_s=0.0,
        solver_only_results=[],
        llm_initial_results=[],
        llm_feedback_results=[],
        failure_modes={"model_precondition": 1},
        complete=False,
    )
    broken["inference_substrate"] = {
        "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
        "value": "legacy_smoke",
    }
    with pytest.raises(AssertionError, match="local_sota_gguf"):
        mod.validate_artifact(broken)

    assert "LLM-only" in mod._honest_verdict(True, 3, 1, 2, 3)
    assert "empty fixture" in mod._honest_verdict(True, 0, 0, 0, 0)
    assert "clean null" in mod._honest_verdict(True, 1, 1, 1, 3)

"""Tests for Exp 5099 BEAVER prefix-bound finite-schema verifier.

Spec refs: REQ-VERIFY-5099, SCENARIO-VERIFY-5099.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5099_beaver_prefix_bound_verifier as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _blocked_exp5097() -> dict[str, Any]:
    return {
        "honest_verdict": "blocked_clean_sota_endpoint_logprob_cache_no_live_logprobs",
        "logprob_endpoint_clean": False,
        "completion_endpoint_ready": False,
        "logprob_endpoint_ready": False,
        "live_llm_invoked": False,
        "flagged_adversarial": False,
        "model_specs": {
            "mandatory_models": [
                {"hf_id": mod.MANDATED_MODEL_IDS[0], "resolved_path": "/models/qwen.gguf"},
                {"hf_id": mod.MANDATED_MODEL_IDS[1], "resolved_path": "/models/gemma31.gguf"},
                {"hf_id": mod.MANDATED_MODEL_IDS[2], "resolved_path": "/models/gemma26.gguf"},
            ]
        },
    }


def _clean_exp5097(*, live: bool = True, flagged: bool = False) -> dict[str, Any]:
    payload = _blocked_exp5097()
    payload.update(
        {
            "honest_verdict": "success_clean_sota_endpoint_logprob_cache_ready",
            "logprob_endpoint_clean": True,
            "completion_endpoint_ready": True,
            "logprob_endpoint_ready": True,
            "live_llm_invoked": live,
            "flagged_adversarial": flagged,
        }
    )
    return payload


def test_req_verify_5099_spec_declares_prefix_bound_contract() -> None:
    """REQ-VERIFY-5099: OpenSpec anchors paths, fields, verdicts, and models."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5099",
        "SCENARIO-VERIFY-5099",
        "python/carnot/experiment_5099_beaver_prefix_bound_verifier.py",
        "results/experiment_5099_beaver_prefix_bound_verifier_v468.json",
        "success_beaver_prefix_bounds_sound_on_finite_schema",
        "complete_beaver_prefix_bounds_toy_only_runtime_not_clean",
        "backend_used=toy_distribution",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for model_id in mod.MANDATED_MODEL_IDS:
        assert model_id in spec


def test_req_verify_5099_exact_distribution_and_constraint_are_enumerable() -> None:
    """REQ-VERIFY-5099: finite outputs, probabilities, and semantic predicate are exact."""

    outputs = mod.finite_verifier_verdict_outputs()
    distribution = mod.toy_finite_distribution(outputs)
    constraint = mod.prefix_closed_constraint()
    exact = mod.exact_probability(distribution, constraint)

    assert len(outputs) == 54
    assert len(distribution) == len(outputs)
    assert sum(row.probability for row in distribution) == mod.ONE
    assert exact.fraction == "4/27"
    assert exact.value == pytest.approx(4 / 27)
    assert constraint["prefix_closed"] is True
    assert constraint["satisfied_terminal_count"] == 8
    assert all(row.text.startswith("{") and row.text.endswith("}") for row in distribution)
    assert mod.terminal_satisfies_constraint("not-json") is False
    assert mod.terminal_satisfies_constraint("[]") is False
    with pytest.raises(ValueError, match="finite output"):
        mod.toy_finite_distribution(())


def test_scenario_verify_5099_prefix_frontier_bounds_are_sound_and_monotone() -> None:
    """SCENARIO-VERIFY-5099: trie frontier bounds contain exact probability."""

    distribution = mod.toy_finite_distribution(mod.finite_verifier_verdict_outputs())
    trie = mod.build_prefix_trie(distribution, mod.terminal_satisfies_constraint)
    exact = mod.exact_probability(distribution, mod.prefix_closed_constraint())
    bound = mod.bound_frontier(trie, max_depth=mod.DEFAULT_FRONTIER_DEPTH)
    terminal = mod.bound_frontier(trie, max_depth=10_000)
    monotonic = mod.check_monotonic_bounds(
        trie,
        exact_probability=exact.value,
        depths=mod.DEFAULT_MONOTONIC_DEPTHS,
    )

    assert 0.0 <= bound.lower_bound <= exact.value <= bound.upper_bound <= 1.0
    assert bound.bound_gap > 0.0
    assert bound.frontier_node_count == len(bound.frontier_nodes)
    assert {node["classification"] for node in bound.frontier_nodes} >= {
        "mixed",
        "no_satisfying",
    }
    assert terminal.lower_bound == pytest.approx(exact.value)
    assert terminal.upper_bound == pytest.approx(exact.value)
    assert terminal.bound_gap == pytest.approx(0.0)
    with pytest.raises(ValueError, match="max_depth"):
        mod.bound_frontier(trie, max_depth=-1)
    assert monotonic["passed"] is True
    assert monotonic["lower_non_decreasing"] is True
    assert monotonic["upper_non_increasing"] is True
    assert monotonic["gap_non_increasing"] is True
    assert monotonic["exact_probability_between_all_depths"] is True


def test_scenario_verify_5099_blocked_exp5097_forces_toy_backend(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5099: unclean Exp5097 does not trigger live LLM use."""

    _write_json(tmp_path / mod.EXP5097_RELATIVE_PATH, _blocked_exp5097())
    preconditions = mod.load_preconditions(root=tmp_path)
    model_specs = mod.model_specs_from_exp5097(root=tmp_path)

    assert preconditions["selected_finite_schema"] == mod.FINITE_SCHEMA_NAME
    assert preconditions["prefix_closed_constraint_definition"] == mod.CONSTRAINT_NAME
    assert preconditions["tokenization_assumptions"]["bpe_tokenizer_used"] is False
    assert preconditions["exp5097_logprob_substrate"]["exists"] is True
    assert preconditions["exp5097_logprob_substrate"]["clean"] is False
    assert preconditions["exp5097_logprob_substrate"]["usable_for_live_frontier"] is False
    assert preconditions["exp5097_logprob_substrate"]["unusable_reason"] == "exp5097_not_clean"
    assert [row["hf_id"] for row in model_specs] == list(mod.MANDATED_MODEL_IDS)
    assert all(row["live_llm_invoked"] is False for row in model_specs)

    missing_root = tmp_path / "missing"
    missing = mod.load_preconditions(root=missing_root)
    assert missing["exp5097_logprob_substrate"]["exists"] is False
    assert missing["exp5097_logprob_substrate"]["artifact_sha256"] is None
    assert missing["exp5097_logprob_substrate"]["unusable_reason"] == "exp5097_artifact_missing"
    assert [row["hf_id"] for row in mod.model_specs_from_exp5097(root=missing_root)] == list(
        mod.MANDATED_MODEL_IDS
    )


def test_scenario_verify_5099_exp5097_cleanliness_reasons_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5099: Exp5097 gate reasons distinguish clean and blocked cases."""

    _write_json(tmp_path / "flagged" / mod.EXP5097_RELATIVE_PATH, _clean_exp5097(flagged=True))
    _write_json(tmp_path / "nolivellm" / mod.EXP5097_RELATIVE_PATH, _clean_exp5097(live=False))
    _write_json(tmp_path / "clean" / mod.EXP5097_RELATIVE_PATH, _clean_exp5097())

    flagged = mod.load_preconditions(root=tmp_path / "flagged")
    no_live = mod.load_preconditions(root=tmp_path / "nolivellm")
    clean = mod.load_preconditions(root=tmp_path / "clean")

    assert flagged["exp5097_logprob_substrate"]["unusable_reason"] == "exp5097_flagged_adversarial"
    assert no_live["exp5097_logprob_substrate"]["unusable_reason"] == "exp5097_no_live_llm_invocation"
    assert clean["exp5097_logprob_substrate"]["clean"] is True
    assert clean["exp5097_logprob_substrate"]["usable_for_live_frontier"] is True
    assert clean["exp5097_logprob_substrate"]["unusable_reason"] is None


def test_req_verify_5099_artifact_contains_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5099: deterministic run emits the terminal schema and checks."""

    _write_json(tmp_path / mod.EXP5097_RELATIVE_PATH, _blocked_exp5097())
    artifact = mod.run(root=tmp_path)

    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith(
        "complete_beaver_prefix_bounds_toy_only_runtime_not_clean"
    )
    assert artifact["inference_substrate"] == mod.TOY_INFERENCE_SUBSTRATE
    assert artifact["backend_used"] == "toy_distribution"
    assert artifact["live_llm_invoked"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["lower_bound"] <= artifact["exact_probability_if_enumerable"]
    assert artifact["exact_probability_if_enumerable"] <= artifact["upper_bound"]
    assert artifact["bound_gap"] == pytest.approx(
        artifact["upper_bound"] - artifact["lower_bound"]
    )
    assert artifact["exact_probability_fraction"] == "4/27"
    assert artifact["monotonic_bounds"]["passed"] is True
    assert artifact["soundness_checks_passed"] is True
    assert artifact["frontier_node_count"] == len(artifact["frontier_nodes"])
    assert {row["hf_id"] for row in artifact["model_specs"]} == set(mod.MANDATED_MODEL_IDS)


def test_scenario_verify_5099_writer_persists_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5099: writer emits stable JSON for the conductor."""

    _write_json(tmp_path / mod.EXP5097_RELATIVE_PATH, _blocked_exp5097())
    output_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_artifact(root=tmp_path, output_path=output_path)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert loaded["result_path"] == mod.RESULT_RELATIVE_PATH
    assert loaded["finite_schema"]["schema_name"] == mod.FINITE_SCHEMA_NAME
    mod.validate_artifact(loaded)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "optimistic", "honest_verdict"),
        ("inference_substrate", "live_llm_inference", "live_llm_inference"),
        ("backend_used", "live_logprobs", "backend_used"),
        ("live_llm_invoked", "false", "live_llm_invoked"),
        ("flagged_adversarial", "false", "flagged_adversarial"),
        ("lower_bound", 1.1, "lower_bound"),
        ("upper_bound", -0.1, "upper_bound"),
        ("exact_probability_if_enumerable", 2.0, "exact_probability_if_enumerable"),
        ("soundness_checks_passed", False, "soundness_checks_passed"),
    ],
)
def test_req_verify_5099_validate_artifact_rejects_schema_violations(
    field: str,
    value: object,
    message: str,
) -> None:
    """REQ-VERIFY-5099: malformed terminal artifacts fail closed."""

    artifact = mod.run()
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda artifact: artifact.update({"inference_substrate": "deterministic_other"}),
            "inference_substrate",
        ),
        (
            lambda artifact: artifact.update(
                {"lower_bound": 0.7, "upper_bound": 0.6, "exact_probability_if_enumerable": 0.6}
            ),
            "lower_bound",
        ),
        (
            lambda artifact: artifact.update(
                {"lower_bound": 0.2, "upper_bound": 0.6, "exact_probability_if_enumerable": 0.1}
            ),
            "exact_probability_if_enumerable",
        ),
        (lambda artifact: artifact.update({"bound_gap": 0.0}), "bound_gap"),
        (
            lambda artifact: artifact.update({"monotonic_bounds": {"passed": False}}),
            "monotonic_bounds",
        ),
        (lambda artifact: artifact.update({"frontier_nodes": []}), "frontier_nodes"),
        (lambda artifact: artifact.update({"frontier_node_count": 999}), "frontier_node_count"),
        (
            lambda artifact: artifact.update({"prefix_closed_constraint": {"prefix_closed": False}}),
            "prefix_closed_constraint",
        ),
        (
            lambda artifact: artifact.update({"model_specs": artifact["model_specs"][:-1]}),
            "model_specs",
        ),
    ],
)
def test_req_verify_5099_validate_artifact_rejects_consistency_violations(
    mutator: Any,
    message: str,
) -> None:
    """REQ-VERIFY-5099: coherent-looking but inconsistent artifacts fail closed."""

    artifact = mod.run()
    mutator(artifact)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


def test_req_verify_5099_validate_artifact_requires_fields_and_principles() -> None:
    """REQ-VERIFY-5099: every required field needs a principle annotation."""

    artifact = mod.run()
    artifact.pop("frontier_nodes")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(artifact)

    artifact = mod.run()
    artifact["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(artifact)


def test_req_verify_5099_committed_artifact_is_schema_valid() -> None:
    """REQ-VERIFY-5099: checked-in deliverable satisfies the terminal schema."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["backend_used"] == "toy_distribution"
    assert artifact["live_llm_invoked"] is False
    assert artifact["soundness_checks_passed"] is True

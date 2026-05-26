"""Tests for Exp 3125 prefix-closed deterministic verifier bound pilot.

Spec refs: REQ-VERIFY-3125, SCENARIO-VERIFY-3125.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import prefix_closed_deterministic_verifier_bound_pilot_v1 as exp


REQUIRED_FIELDS = {
    "prefix_closed_bound_pilot_ready",
    "constraint_families",
    "fixture_count",
    "explored_prefix_count",
    "pruned_prefix_count",
    "lower_bound",
    "upper_bound",
    "bound_width",
    "semantic_coverage",
    "limitations",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def test_req_verify_3125_spec_anchor_and_script_exist() -> None:
    """REQ-VERIFY-3125: OpenSpec declares the exact bounded pilot contract."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3125" in spec
    assert "SCENARIO-VERIFY-3125" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "prefix_closed_bound_pilot_ready" in spec
    assert "bound_width" in spec
    assert exp.SCRIPT_REL_PATH.exists()


def test_scenario_verify_3125_prefix_acceptance_and_rejection_are_exact() -> None:
    """SCENARIO-VERIFY-3125: exact terminal JSON passes and bad prefixes prune."""

    fixture = exp.build_fixture_subset()[0]
    terminal = exp.terminal_sequence_for_fixture(fixture)

    assert exp.classify_prefix((), fixture)["status"] == "viable"
    assert exp.classify_prefix(terminal[:-1], fixture)["status"] == "viable"
    assert exp.classify_prefix(terminal, fixture)["status"] == "accepted"
    assert exp.terminal_satisfies_fixture(terminal, fixture) is True

    bad_answer_prefix = ("{", '"answer"', ":", '"MAYBE"')
    bad_status = exp.classify_prefix(bad_answer_prefix, fixture)
    assert bad_status["status"] == "pruned"
    assert bad_status["reason"] == "no_satisfying_extension"

    bad_terminal = ("{", '"answer"', ":", '"MAYBE"', ",", '"score"', ":", "1", "}", "<eos>")
    assert exp.terminal_satisfies_fixture(bad_terminal, fixture) is False
    assert exp.terminal_satisfies_fixture(("{",), fixture) is False
    assert exp.terminal_satisfies_fixture(("{", "<eos>"), fixture) is False
    assert exp.terminal_satisfies_fixture(("1", "<eos>"), fixture) is False


def test_req_verify_3125_prefix_rejection_is_monotone() -> None:
    """REQ-VERIFY-3125: once a prefix is pruned, one-token extensions stay pruned."""

    fixture = exp.build_fixture_subset()[1]
    bad_prefix = ("{", '"answer"', ":", '"VALID"')

    assert exp.prefix_rejection_is_monotone((), fixture, exp.VOCABULARY) is False
    assert exp.classify_prefix(bad_prefix, fixture)["status"] == "pruned"
    assert exp.prefix_rejection_is_monotone(bad_prefix, fixture, exp.VOCABULARY)
    for token in exp.VOCABULARY:
        assert exp.classify_prefix((*bad_prefix, token), fixture)["status"] == "pruned"


def test_req_verify_3125_bound_aggregation_reports_conservative_width() -> None:
    """REQ-VERIFY-3125: lower/upper bounds aggregate accepted and frontier mass."""

    fixtures = exp.build_fixture_subset()
    summary = exp.enumerate_frontier(fixtures, max_depth=exp.DEFAULT_MAX_DEPTH)

    assert summary["fixture_count"] == len(fixtures)
    assert summary["accepted_prefix_count"] == 2
    assert summary["viable_frontier_count"] == 1
    assert summary["pruned_prefix_count"] > 0
    assert summary["explored_prefix_count"] == len(summary["frontier_rows"])
    assert summary["lower_bound"] > 0.0
    assert summary["upper_bound"] > summary["lower_bound"]
    assert summary["bound_width"] == pytest.approx(
        summary["upper_bound"] - summary["lower_bound"]
    )
    assert summary["explored_mass"] == pytest.approx(1.0)


def test_scenario_verify_3125_writes_valid_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3125: the pilot artifact exposes scope and limitations."""

    output = exp.write_artifact(
        tmp_path,
        output_path=tmp_path / exp.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.25,
        tests_run=["REQ-VERIFY-3125 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["prefix_closed_bound_pilot_ready"] is True
    assert artifact["fixture_count"] == 3
    assert artifact["explored_prefix_count"] > artifact["fixture_count"]
    assert artifact["pruned_prefix_count"] > 0
    assert artifact["lower_bound"] > 0.0
    assert artifact["upper_bound"] > artifact["lower_bound"]
    assert artifact["bound_width"] == pytest.approx(
        artifact["upper_bound"] - artifact["lower_bound"]
    )
    assert artifact["semantic_coverage"]["json_syntax"]["covered"] is True
    assert artifact["semantic_coverage"]["answer_label_semantics"]["covered"] is True
    assert artifact["semantic_coverage"]["live_llm_correctness"]["covered"] is False
    assert artifact["inference_substrate"]["live_model_invoked"] is False
    assert artifact["inference_substrate"]["probability_source"] == "deterministic_fixture_prior"
    assert "full LLM correctness" in " ".join(artifact["limitations"])
    assert artifact["tests_run"] == ["REQ-VERIFY-3125 focused"]
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")

    exp.validate_artifact(artifact)


def test_req_verify_3125_validation_rejects_overclaiming() -> None:
    """REQ-VERIFY-3125: artifact validation blocks missing fields and overclaims."""

    artifact = exp.build_artifact(started_s=1.0, now_s=2.0)
    exp.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="live_model_invoked"):
        exp.validate_artifact(
            artifact
            | {"inference_substrate": artifact["inference_substrate"] | {"live_model_invoked": True}}
        )
    with pytest.raises(ValueError, match="bounds must be ordered"):
        exp.validate_artifact(artifact | {"lower_bound": artifact["upper_bound"] + 0.1})
    with pytest.raises(ValueError, match="bound_width"):
        exp.validate_artifact(artifact | {"bound_width": artifact["bound_width"] + 0.1})
    with pytest.raises(ValueError, match="live_llm_correctness"):
        exp.validate_artifact(
            artifact
            | {
                "semantic_coverage": artifact["semantic_coverage"]
                | {"live_llm_correctness": {"covered": True}}
            }
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready: wrong prefix"})
    with pytest.raises(ValueError, match="fixture_count"):
        exp.validate_artifact(artifact | {"fixture_count": 0})
    with pytest.raises(ValueError, match="explored_prefix_count"):
        exp.validate_artifact(artifact | {"explored_prefix_count": 0})
    with pytest.raises(ValueError, match="pruned_prefix_count"):
        exp.validate_artifact(artifact | {"pruned_prefix_count": 0})
    with pytest.raises(ValueError, match="constraint_families"):
        exp.validate_artifact(artifact | {"constraint_families": []})
    with pytest.raises(ValueError, match="limitations"):
        exp.validate_artifact(artifact | {"limitations": []})

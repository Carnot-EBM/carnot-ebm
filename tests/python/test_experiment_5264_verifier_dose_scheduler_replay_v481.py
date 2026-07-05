"""Tests for Exp 5264 cached verifier-dose scheduler replay.

Spec refs: REQ-VERIFY-5264, SCENARIO-VERIFY-5264.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from carnot.pipeline import verifier_dose_scheduler_replay as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_verify_5264_spec_declares_cached_scheduler_contract() -> None:
    """REQ-VERIFY-5264: OpenSpec anchors the cached replay and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5264") :]

    for marker in (
        "REQ-VERIFY-5264",
        "SCENARIO-VERIFY-5264",
        mod.RESULT_RELATIVE_PATH,
        "cached_fixture_replay_no_llm",
        "no_verifier",
        "cheap_deterministic",
        "typed_memory",
        "full_replay",
        "full_verifier_calls_avoided_rate",
        "decision_quality_delta",
        "false_accept_delta",
        "abstain_or_block_count",
        "fixture_receipts",
    ):
        assert marker in section


def test_req_verify_5264_fixtures_replay_prior_receipt_backed_decisions() -> None:
    """REQ-VERIFY-5264: fixtures expose transparent features and cached outcomes."""

    fixtures = mod.build_scheduler_fixtures(root=REPO)

    assert len(fixtures) >= mod.MIN_FIXTURE_COUNT
    assert {fixture.source_artifact for fixture in fixtures} >= {
        "results/experiment_5239_continuous_self_learning_controlled_memory_ablation_v479.json",
        "results/experiment_5261_typed_memory_interference_audit_v481.json",
    }
    assert all(fixture.receipt_complete for fixture in fixtures)
    assert all(fixture.full_decision == fixture.expected_decision for fixture in fixtures)
    assert {
        "cheap_gate_passed",
        "memory_confidence",
        "deterministic_violation_count",
        "artifact_receipt_complete",
    } <= set(mod.policy_feature_names())


def test_scenario_verify_5264_policy_uses_all_verifier_doses() -> None:
    """SCENARIO-VERIFY-5264: the transparent policy routes each cached risk class."""

    routes = {
        fixture.task_id: mod.choose_verifier_route(fixture)
        for fixture in mod.build_scheduler_fixtures(root=REPO)
    }

    assert set(routes.values()) == {
        "cheap_deterministic",
        "full_replay",
        "no_verifier",
        "typed_memory",
    }
    assert routes["range_constraint_unrelated"] == "no_verifier"
    assert routes["hardware_speedup_boundary_consumer"] == "cheap_deterministic"
    assert routes["gap1_memory_only_consumer"] == "typed_memory"
    assert routes["gap4_candidate_pool_consumer"] == "full_replay"


def test_scenario_verify_5264_replay_preserves_quality_and_safety() -> None:
    """SCENARIO-VERIFY-5264: scheduler matches full replay while avoiding calls."""

    replay = mod.replay_scheduler(mod.build_scheduler_fixtures(root=REPO))

    assert replay["scheduler_metrics"]["quality_rate"] == 1.0
    assert replay["baseline_metrics"]["always_full"]["quality_rate"] == 1.0
    assert replay["baseline_metrics"]["always_cheap"]["quality_rate"] < 1.0
    assert replay["baseline_metrics"]["no_verifier"]["quality_rate"] < 1.0
    assert replay["full_verifier_calls_avoided_rate"] == 0.857143
    assert replay["decision_quality_delta"] == 0.0
    assert replay["false_accept_delta"] == 0.0
    assert replay["abstain_or_block_count"] == 4
    assert replay["route_counts"] == {
        "cheap_deterministic": 2,
        "full_replay": 1,
        "no_verifier": 1,
        "typed_memory": 3,
    }
    assert replay["scheduler_ready"] is True


def test_req_verify_5264_fail_closed_routes_and_nonready_verdicts() -> None:
    """REQ-VERIFY-5264: incomplete or unsafe replay is not promoted as ready."""

    fixture = mod.build_scheduler_fixtures(root=REPO)[0]

    assert mod.choose_verifier_route(replace(fixture, receipt_complete=False)) == "full_replay"
    assert mod._honest_verdict(
        {
            "scheduler_ready": False,
            "fixture_count": mod.MIN_FIXTURE_COUNT - 1,
            "false_accept_delta": 0.0,
            "decision_quality_delta": 0.0,
            "full_verifier_calls_avoided_rate": 0.0,
        }
    ).startswith("blocked_underpowered")
    assert "increased false accepts" in mod._honest_verdict(
        {
            "scheduler_ready": False,
            "fixture_count": mod.MIN_FIXTURE_COUNT,
            "false_accept_delta": 0.1,
            "decision_quality_delta": 0.0,
            "full_verifier_calls_avoided_rate": 0.0,
        }
    )
    assert "lost decision quality" in mod._honest_verdict(
        {
            "scheduler_ready": False,
            "fixture_count": mod.MIN_FIXTURE_COUNT,
            "false_accept_delta": 0.0,
            "decision_quality_delta": -0.1,
            "full_verifier_calls_avoided_rate": 0.0,
        }
    )
    assert "null scheduler replay" in mod._honest_verdict(
        {
            "scheduler_ready": False,
            "fixture_count": mod.MIN_FIXTURE_COUNT,
            "false_accept_delta": 0.0,
            "decision_quality_delta": 0.0,
            "full_verifier_calls_avoided_rate": 0.0,
        }
    )


def test_req_verify_5264_artifact_schema_and_run_are_stable(tmp_path: Path) -> None:
    """REQ-VERIFY-5264: run() writes the required principle-wrapped artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "fixture command", "outcome": "passed"}]

    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["scheduler_ready"] is True
    assert artifact["scheduler_ready_principle"]
    assert artifact["tests_run"] == tests_run
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "useful" in artifact["honest_verdict"]["value"]
    assert isinstance(artifact["abstain_or_block_count"]["value"], int)

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert "value" in artifact[field]
        assert "principle" in artifact[field]
    assert "checksums" in artifact["fixture_receipts"]
    assert "principle" in artifact["fixture_receipts"]
    mod.validate_artifact(artifact)


def test_req_verify_5264_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5264: checked-in result JSON is reproducible from cached fixtures."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_result_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == "cached_fixture_replay_no_llm"
    assert result["scheduler_ready"] is True
    assert result["full_verifier_calls_avoided_rate"]["value"] == 0.857143
    assert result["decision_quality_delta"]["value"] == 0.0
    assert result["false_accept_delta"]["value"] == 0.0
    mod.validate_artifact(result)

"""Tests for Exp 4204 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4204, SCENARIO-VERIFY-4204.
"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4204_verifier_registry_gaps_hygiene as exp4204_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4204 as exp4204


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4204_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4204.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "code_path": "python/carnot/agentic/gap4_program_induction_stack.py",
                "eval": {"metric": "pass_at_1"},
                "registry_roles": [],
            }
        ]
    }


def _write_minimal_repo(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")
    for name in (
        "arc3_gap3_stage2_eval_pool.json.gz",
        "arc3_gap4_induced_programs.json",
        "experiment_4197_verifier_reward_phase0_headroom_harness_build.json",
        "experiment_4199_verifier_reward_decisive_a_vs_b_collect.json",
        "experiment_4200_certified_arc_corpus_distill_lift.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def test_req_4204_spec_declared() -> None:
    """REQ-VERIFY-4204: OpenSpec declares the .389 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4204",
        "SCENARIO-VERIFY-4204",
        "python/carnot/experiment_4204_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        "experiment_4204_verifier_registry_gaps_hygiene.json",
        "experiment_4197_verifier_reward_phase0_headroom_harness_build.json",
        "experiment_4199_verifier_reward_decisive_a_vs_b_collect.json",
        "experiment_4200_certified_arc_corpus_distill_lift.json",
        "phase0_precision=0.9561855670103093",
        "youden_j=0.4137931034482759",
        "verifier_label_carries_signal=false",
        "a_vs_b_delta=null",
        "certified_corpus_size=16",
        "distill_lift_ci95=[0.0, 0.0]",
        "0.4516",
        "0.5806",
        exp4204.GAP_REWARD_GAP_ID,
        exp4204.CERTIFIED_CORPUS_GAP_ID,
    ):
        assert marker in spec
    assert exp4204.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4204.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4204.FIELD_PRINCIPLES["gaps_updated"] in spec
    assert exp4204_wrapper.main is exp4204.main


def test_scenario_4204_preconditions_and_replay_are_bitexact() -> None:
    """SCENARIO-VERIFY-4204: cached GAP-4 replay reproduces exactly."""

    preflight = exp4204.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None
    assert {check["resource"] for check in preflight["checks"]} == {
        "gap4_arc1_candidate_fixtures",
        "verifier_registry",
        "verifier_gaps",
        "exp4197_phase0",
        "exp4199_a_vs_b",
        "exp4200_certified_arc_corpus",
    }

    replay = exp4204.replay_gap4_arc1(REPO_ROOT)
    assert replay["regression_guard_passed"] is True
    assert replay["arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert replay["no_codex_calls"] is True
    assert replay["no_gguf_inference"] is True


def test_req_4204_classifies_reward_and_certified_corpus_truth() -> None:
    """REQ-VERIFY-4204: .389 reward outcomes are classified without spin."""

    reward = exp4204.classify_verifier_reward_outcome(REPO_ROOT)
    assert reward["gap_id"] == exp4204.GAP_REWARD_GAP_ID
    assert reward["status"] == "blocked_a_vs_b_not_collected_training_not_launched"
    assert reward["phase0_precision"] == pytest.approx(0.9561855670103093)
    assert reward["youden_j"] == pytest.approx(0.4137931034482759)
    assert reward["phase0_gate_clean"] is True
    assert reward["training_launched"] is False
    assert reward["verifier_label_carries_signal"] is False
    assert reward["a_vs_b_delta"] is None
    assert reward["a_vs_b_ci95"] is None
    assert reward["honest_verdict"] == "blocked_gate_check_failed"

    corpus = exp4204.classify_certified_arc_corpus(REPO_ROOT)
    assert corpus["gap_id"] == exp4204.CERTIFIED_CORPUS_GAP_ID
    assert corpus["status"] == "certified_corpus_built_distill_lift_uninformative"
    assert corpus["certified_corpus_size"] == 16
    assert corpus["certification_precision"]["rate"] == pytest.approx(0.9375)
    assert corpus["distill_lift_ci95"] == [0.0, 0.0]
    assert corpus["distill_lift_latent_vs_absent"] == "uninformative"
    assert corpus["seeded_generation_status"] == "missing_seeded_checkpoint_conservative_flat"


def test_scenario_4204_ensure_ledgers_record_outcomes_and_registry_role() -> None:
    """SCENARIO-VERIFY-4204: registry and gaps carry the .389 truth."""

    replay = exp4204.replay_gap4_arc1(REPO_ROOT)
    reward = exp4204.classify_verifier_reward_outcome(REPO_ROOT)
    corpus = exp4204.classify_certified_arc_corpus(REPO_ROOT)

    registry, gaps, summary = exp4204.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n",
        replay,
        reward,
        corpus,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": [
            exp4204.GAP_REWARD_GAP_ID,
            exp4204.CERTIFIED_CORPUS_GAP_ID,
        ],
        "verifier_reward_recorded": True,
        "certified_corpus_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4204"] == exp4204.EXP4204_ARTIFACT_PATH
    assert gap4["eval"]["exp4204_regression_guard_passed"] is True
    assert gap4["eval"]["exp4204_arc1_rule_exec_vote_pass2"] == pytest.approx(0.4516)
    assert gap4["eval"]["exp4204_arc1_rule_exec_gated_pass2"] == pytest.approx(0.5806)
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4204.V389_ROLE_ID)
    assert role["verifier_reward_status"] == "blocked_a_vs_b_not_collected_training_not_launched"
    assert role["verifier_label_carries_signal"] is False
    assert role["a_vs_b_delta"] is None
    assert role["certified_corpus_size"] == 16
    assert role["distill_lift_latent_vs_absent"] == "uninformative"
    assert exp4204._registry_contains_outcomes(registry) is True
    assert exp4204._registry_contains_outcomes({}) is False

    assert exp4204.GAP_REWARD_GAP_ID in gaps
    assert "verifier_label_carries_signal=false" in gaps
    assert "a_vs_b_delta=None" in gaps
    assert "phase0_precision=0.9561855670103093" in gaps
    assert "youden_j=0.4137931034482759" in gaps
    assert exp4204.CERTIFIED_CORPUS_GAP_ID in gaps
    assert "certified_corpus_size=16" in gaps
    assert "distill_lift_ci95=[0.0, 0.0]" in gaps
    assert "invisible_leash_diagnosis=uninformative" in gaps


def test_req_4204_build_artifact_validates_required_fields() -> None:
    """REQ-VERIFY-4204: terminal artifact exposes required schema fields."""

    replay = exp4204.replay_gap4_arc1(REPO_ROOT)
    reward = exp4204.classify_verifier_reward_outcome(REPO_ROOT)
    corpus = exp4204.classify_certified_arc_corpus(REPO_ROOT)
    artifact = exp4204.build_artifact(
        offline_replay=replay,
        verifier_reward_outcome=reward,
        certified_arc_corpus=corpus,
        registry_updated=True,
        gaps_updated=[exp4204.GAP_REWARD_GAP_ID, exp4204.CERTIFIED_CORPUS_GAP_ID],
        duration_s=0.012,
    )

    exp4204.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4204.GAP_REWARD_GAP_ID,
        exp4204.CERTIFIED_CORPUS_GAP_ID,
    ]
    assert artifact["field_principles"] == exp4204.FIELD_PRINCIPLES
    assert artifact["verifier_reward_outcome"]["verifier_label_carries_signal"] is False
    assert artifact["certified_arc_corpus"]["certified_corpus_size"] == 16
    assert artifact["cited_upstream_artifacts"] == [
        exp4204.ARC1_POOL_PATH,
        exp4204.ARC1_PROGRAMS_PATH,
        exp4204.EXP4197_PATH,
        exp4204.EXP4199_PATH,
        exp4204.EXP4200_PATH,
    ]

    for field in exp4204.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4204.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4204.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4204.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4204.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="gaps_updated"):
        exp4204.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="field_principles"):
        exp4204.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4204.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_4204_helper_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4204: schema helpers fail closed without hidden inference."""

    assert exp4204._numeric_or_none(True) is None
    assert exp4204._numeric_or_none("bad") is None
    assert exp4204._first_numeric({"bad": "x", "ok": 2}, "bad", "ok") == pytest.approx(2.0)
    assert exp4204._first_ci95({"bad": [0.0], "ok": [0, 1]}, "bad", "ok") == [0.0, 1.0]
    assert exp4204._round4(None) is None
    assert exp4204._round4(1.23456) == pytest.approx(1.2346)
    assert exp4204._check_json_resource(tmp_path, "missing", "missing.json") == {
        "resource": "missing",
        "available": False,
        "detail": "missing: missing.json",
    }
    assert (
        exp4204._training_launched_from_exp4199(
            {
                "gates_evaluated": [
                    "not-a-gate",
                    {"artifact_field": "other_gate.training_launched", "actual": "yes"},
                    {"artifact_field": "other_gate.status", "actual": True},
                ]
            }
        )
        is None
    )
    assert (
        exp4204._verifier_reward_status(
            {"verifier_label_carries_signal": True, "a_vs_b_delta": 0.1, "a_vs_b_ci95": [0.01, 0.2]}
        )
        == "filled_verifier_label_carries_signal"
    )
    assert exp4204._verifier_reward_status({"training_launched": False}) == (
        "blocked_a_vs_b_not_collected_training_not_launched"
    )
    assert (
        exp4204._verifier_reward_status(
            {"training_launched": True, "verifier_label_carries_signal": False, "a_vs_b_delta": 0.0}
        )
        == "open_no_a_vs_b_signal"
    )
    assert exp4204._verifier_reward_status({"training_launched": True}) == (
        "open_a_vs_b_not_decision_grade"
    )
    assert (
        exp4204._certified_corpus_status(
            {"certified_corpus_size": 1, "distill_lift_latent_vs_absent": "latent"}
        )
        == "certified_corpus_built_distill_lift_latent"
    )
    assert (
        exp4204._certified_corpus_status(
            {"certified_corpus_size": 1, "distill_lift_latent_vs_absent": "absent"}
        )
        == "certified_corpus_built_distill_lift_absent"
    )
    assert exp4204._certified_corpus_status({"certified_corpus_size": 0}) == (
        "certified_corpus_empty"
    )

    _write_minimal_repo(tmp_path)
    (tmp_path / exp4204.EXP4199_PATH).write_text(
        json.dumps(
            {
                "honest_verdict": "complete: measured_positive",
                "training_launched": True,
                "verifier_label_carries_signal": True,
                "a_vs_b_delta": 0.2,
                "a_vs_b_ci95": [0.1, 0.3],
            }
        ),
        encoding="utf-8",
    )
    reward = exp4204.classify_verifier_reward_outcome(tmp_path)
    assert reward["training_launched"] is True
    assert reward["status"] == "filled_verifier_label_carries_signal"


def test_scenario_4204_run_hygiene_writes_artifact_and_ledgers(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4204: run writes the deliverable JSON and ledger entries."""

    _write_minimal_repo(tmp_path)
    artifact = exp4204.run_hygiene(tmp_path)
    exp4204.validate_artifact(artifact)

    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4204.GAP_REWARD_GAP_ID,
        exp4204.CERTIFIED_CORPUS_GAP_ID,
    ]
    assert artifact["registry_updated"] is True
    written = json.loads((tmp_path / exp4204.EXP4204_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    registry = yaml.safe_load(
        (tmp_path / "ops" / "verifier_registry.yaml").read_text(encoding="utf-8")
    )
    assert exp4204._registry_contains_outcomes(registry) is True
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert exp4204.GAP_REWARD_GAP_ID in gaps
    assert exp4204.CERTIFIED_CORPUS_GAP_ID in gaps


def test_req_4204_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    """REQ-VERIFY-4204: failed preconditions write blocked_<resource> and no ledger win."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")

    artifact = exp4204.run_hygiene(tmp_path)
    exp4204.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_gap4_arc1_candidate_fixtures"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_updated"] is False
    assert artifact["gaps_updated"] == []
    assert exp4204.GAP_REWARD_GAP_ID not in (tmp_path / "ops" / "verifier_gaps.md").read_text(
        encoding="utf-8"
    )


def test_scenario_4204_results_entrypoint_runs_with_monkeypatched_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4204: required results entrypoint delegates to Exp 4204."""

    called: list[bool] = []
    monkeypatch.setattr(exp4204, "main", lambda: called.append(True))

    runpy.run_path(RESULTS_WRAPPER_PATH.as_posix(), run_name="__main__")

    assert called == [True]

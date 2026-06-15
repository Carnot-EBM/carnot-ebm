"""Tests for Exp 4227 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4227, SCENARIO-VERIFY-4227.
"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4227_verifier_registry_gaps_hygiene as exp4227_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4227 as exp4227


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4227_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4227.GAP4_VERIFIER_ID,
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
    (tmp_path / "ops" / "verifier_gaps.md").write_text(
        "# Verifier Gaps\n\nHistorical note remains.\n",
        encoding="utf-8",
    )
    for name in (
        "arc3_gap3_stage2_eval_pool.json.gz",
        "arc3_gap4_induced_programs.json",
        "experiment_4208_verifier_as_detector_auroc.json",
        "experiment_4220_oracle_distinct_arc_verifier_build_labeled.json",
        "experiment_4220_oracle_distinct_arc_verifier_model.json",
        "experiment_4221_oracle_distinct_arc_verifier_beats_vote.json",
        "experiment_4223_verifier_as_reward_3arm_synchronous.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {
        "returncode": 0,
        "reports": [
            {
                "flags": [],
                "flag_count": 0,
                "max_severity": -1,
            }
        ],
    }


def test_req_4227_spec_declared() -> None:
    """REQ-VERIFY-4227: OpenSpec declares the .391 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4227",
        "SCENARIO-VERIFY-4227",
        "python/carnot/experiment_4227_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        "experiment_4227_verifier_registry_gaps_hygiene.json",
        "experiment_4221_oracle_distinct_arc_verifier_beats_vote.json",
        "experiment_4220_oracle_distinct_arc_verifier_build_labeled.json",
        "experiment_4223_verifier_as_reward_3arm_synchronous.json",
        "experiment_4208_verifier_as_detector_auroc.json",
        "verifier_minus_vote_delta=-0.0714285714",
        "verifier_minus_vote_ci95=[-0.2142857143, 0.0]",
        "oracle_distinct_auroc=0.778980279",
        "wrong_majority_n=5",
        "verifier_label_carries_signal=false",
        "youden_j=0.4137931034482759",
        "METHODOLOGY_MISSING",
        "inference_substrate=verifier_ensemble_against_cached_candidates",
        exp4227.GAP_ORACLE_DISTINCT_GAP_ID,
        exp4227.GAP_ORACLE_DISTINCT_A2_GAP_ID,
        exp4227.GAP_REWARD_GAP_ID,
        exp4227.DETECTOR_GAP_ID,
    ):
        assert marker in spec
    assert exp4227.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4227.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4227.FIELD_PRINCIPLES["gaps_updated"] in spec
    assert exp4227.FIELD_PRINCIPLES["random_seed"] in spec
    assert exp4227.FIELD_PRINCIPLES["reproducibility_checksum"] in spec
    assert exp4227_wrapper.main is exp4227.main


def test_scenario_4227_preconditions_replay_and_checksum_are_bitexact() -> None:
    """SCENARIO-VERIFY-4227: cached GAP-4 replay and checksum are stable."""

    preflight = exp4227.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None
    assert {check["resource"] for check in preflight["checks"]} == {
        "gap4_arc1_candidate_fixtures",
        "verifier_registry",
        "verifier_gaps",
        "exp4208_detector_auroc",
        "exp4220_oracle_distinct_build_labeled",
        "exp4221_oracle_distinct_a2",
        "exp4223_verifier_reward_a_vs_b",
    }

    replay = exp4227.replay_gap4_arc1(REPO_ROOT)
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
    assert replay["trm_training_touched"] is False

    checksum = exp4227.candidate_set_checksum(REPO_ROOT)
    assert checksum.startswith("sha256:")
    assert len(checksum) == len("sha256:") + 64
    assert checksum == exp4227.candidate_set_checksum(REPO_ROOT)


def test_req_4227_classifies_oracle_reward_and_detector_truth() -> None:
    """REQ-VERIFY-4227: .391 upstream outcomes are recorded without promotion."""

    oracle = exp4227.classify_oracle_distinct_outcome(REPO_ROOT)
    assert oracle["gap_id"] == exp4227.GAP_ORACLE_DISTINCT_GAP_ID
    assert oracle["a2_gap_id"] == exp4227.GAP_ORACLE_DISTINCT_A2_GAP_ID
    assert oracle["status"] == "open_a2_ties_vote_with_headroom"
    assert oracle["oracle_distinct_beats_vote"] is False
    assert oracle["verifier_minus_vote_delta"] == pytest.approx(-0.0714285714)
    assert oracle["verifier_minus_vote_ci95"] == [-0.2142857143, 0.0]
    assert oracle["arbiter_override_minus_vote"] == pytest.approx(0.0)
    assert oracle["matched_control_delta"] == pytest.approx(0.0)
    assert oracle["oracle_at_k"] == pytest.approx(1.0)
    assert oracle["verifier_is_oracle"] is False
    assert oracle["selector_trained"] is True
    assert oracle["oracle_distinct_auroc"] == pytest.approx(0.778980279)
    assert oracle["oracle_distinct_auroc_ci95"] == [0.6146676853, 0.9174508427]
    assert oracle["wrong_majority_n"] == 5
    assert oracle["gap_moat_update"] == "unchanged_a2_did_not_beat_vote"

    reward = exp4227.classify_verifier_reward_outcome(REPO_ROOT)
    assert reward["gap_id"] == exp4227.GAP_REWARD_GAP_ID
    assert reward["status"] == "open_accumulating_reward_no_eval_yet"
    assert reward["verifier_label_carries_signal"] is False
    assert reward["a_vs_b_delta"] is None
    assert reward["a_vs_b_ci95"] is None
    assert reward["positive_control_confirmed"] is False
    assert reward["youden_j"] == pytest.approx(0.4137931034482759)
    assert reward["accumulated_n"]["eval"] == 0
    assert reward["verifier_is_oracle"] is True

    detector = exp4227.classify_detector_aurocs(REPO_ROOT)
    assert detector["gap_id"] == exp4227.DETECTOR_GAP_ID
    assert detector["status"] == "detector_auroc_recorded_all_domains_ci_exclusive"
    assert detector["detection_auroc_by_domain"] == {
        "sudoku": pytest.approx(1.0),
        "code": pytest.approx(1.0),
        "math": pytest.approx(1.0),
        "arc": pytest.approx(0.9016),
    }
    assert detector["verifier_is_oracle_by_domain"]["arc"] is False


def test_scenario_4227_ensure_ledgers_record_outcomes_and_registry_role() -> None:
    """SCENARIO-VERIFY-4227: registry and gaps carry the .391 truth."""

    replay = exp4227.replay_gap4_arc1(REPO_ROOT)
    oracle = exp4227.classify_oracle_distinct_outcome(REPO_ROOT)
    reward = exp4227.classify_verifier_reward_outcome(REPO_ROOT)
    detector = exp4227.classify_detector_aurocs(REPO_ROOT)

    registry, gaps, summary = exp4227.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n\nHistorical note remains.\n",
        replay,
        oracle,
        reward,
        detector,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": [
            exp4227.GAP_ORACLE_DISTINCT_GAP_ID,
            exp4227.GAP_ORACLE_DISTINCT_A2_GAP_ID,
            exp4227.GAP_REWARD_GAP_ID,
            exp4227.DETECTOR_GAP_ID,
        ],
        "oracle_distinct_recorded": True,
        "oracle_distinct_a2_recorded": True,
        "verifier_reward_recorded": True,
        "detector_aurocs_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4227"] == exp4227.EXP4227_ARTIFACT_PATH
    assert gap4["eval"]["exp4227_regression_guard_passed"] is True
    assert gap4["eval"]["exp4227_arc1_rule_exec_vote_pass2"] == pytest.approx(0.4516)
    assert gap4["eval"]["exp4227_arc1_rule_exec_gated_pass2"] == pytest.approx(0.5806)
    assert gap4["eval"]["exp4227_oracle_distinct_beats_vote"] is False
    assert gap4["eval"]["exp4227_oracle_distinct_auroc"] == pytest.approx(0.778980279)
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4227.V391_ROLE_ID)
    assert role["oracle_distinct_status"] == "open_a2_ties_vote_with_headroom"
    assert role["wrong_majority_n"] == 5
    assert role["verifier_minus_vote_delta"] == pytest.approx(-0.0714285714)
    assert role["verifier_label_carries_signal"] is False
    assert role["gap_moat_update"] == "unchanged_a2_did_not_beat_vote"
    assert exp4227._registry_contains_outcomes(registry) is True
    assert exp4227._registry_contains_outcomes({}) is False

    assert "Historical note remains." in gaps
    assert exp4227.GAP_ORACLE_DISTINCT_GAP_ID in gaps
    assert exp4227.GAP_ORACLE_DISTINCT_A2_GAP_ID in gaps
    assert "oracle_distinct_beats_vote=false" in gaps
    assert "verifier_minus_vote_delta=-0.0714285714" in gaps
    assert "verifier_minus_vote_ci95=[-0.2142857143, 0.0]" in gaps
    assert "oracle_distinct_auroc=0.778980279" in gaps
    assert "wrong_majority_n=5" in gaps
    assert "verifier_is_oracle=false" in gaps
    assert "GAP-MOAT unchanged" in gaps
    assert exp4227.GAP_REWARD_GAP_ID in gaps
    assert "verifier_label_carries_signal=false" in gaps
    assert "a_vs_b_delta=None" in gaps
    assert exp4227.DETECTOR_GAP_ID in gaps
    assert "arc=0.9016" in gaps


def test_req_4227_build_artifact_validates_methodology_fields() -> None:
    """REQ-VERIFY-4227: terminal artifact exposes required schema fields."""

    replay = exp4227.replay_gap4_arc1(REPO_ROOT)
    oracle = exp4227.classify_oracle_distinct_outcome(REPO_ROOT)
    reward = exp4227.classify_verifier_reward_outcome(REPO_ROOT)
    detector = exp4227.classify_detector_aurocs(REPO_ROOT)
    checksum = exp4227.candidate_set_checksum(REPO_ROOT)
    artifact = exp4227.build_artifact(
        offline_replay=replay,
        oracle_distinct_outcome=oracle,
        verifier_reward_outcome=reward,
        detector_aurocs=detector,
        registry_updated=True,
        gaps_updated=[
            exp4227.GAP_ORACLE_DISTINCT_GAP_ID,
            exp4227.GAP_ORACLE_DISTINCT_A2_GAP_ID,
            exp4227.GAP_REWARD_GAP_ID,
            exp4227.DETECTOR_GAP_ID,
        ],
        random_seed=exp4227.RANDOM_SEED,
        reproducibility_checksum=checksum,
        duration_s=0.012,
    )

    exp4227.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4227.GAP_ORACLE_DISTINCT_GAP_ID,
        exp4227.GAP_ORACLE_DISTINCT_A2_GAP_ID,
        exp4227.GAP_REWARD_GAP_ID,
        exp4227.DETECTOR_GAP_ID,
    ]
    assert artifact["field_principles"] == exp4227.FIELD_PRINCIPLES
    assert artifact["random_seed"] == exp4227.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == checksum
    assert artifact["model_specs"]["candidate_set_sha256"] == checksum
    assert artifact["model_specs"]["trm_training_touched"] is False
    assert artifact["inference_substrate"] == exp4227.INFERENCE_SUBSTRATE
    assert artifact["adversarial_verify"]["status"] == "pending"
    assert artifact["cited_upstream_artifacts"] == [
        exp4227.ARC1_POOL_PATH,
        exp4227.ARC1_PROGRAMS_PATH,
        exp4227.EXP4208_PATH,
        exp4227.EXP4220_PATH,
        exp4227.EXP4221_PATH,
        exp4227.EXP4223_PATH,
    ]

    for field in exp4227.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4227.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4227.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4227.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4227.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="gaps_updated"):
        exp4227.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="random_seed"):
        exp4227.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4227.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="model_specs"):
        exp4227.validate_artifact({**artifact, "model_specs": {}})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4227.validate_artifact({**artifact, "inference_substrate": "live"})
    with pytest.raises(ValueError, match="field_principles"):
        exp4227.validate_artifact({**artifact, "field_principles": {}})


def test_req_4227_helper_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4227: schema helpers fail closed without hidden inference."""

    assert exp4227._numeric_or_none(True) is None
    assert exp4227._numeric_or_none("bad") is None
    assert exp4227._first_numeric({"bad": "x", "ok": 2}, "bad", "ok") == pytest.approx(2.0)
    assert exp4227._first_ci95({"bad": [0.0], "ok": [0, 1]}, "bad", "ok") == [0.0, 1.0]
    assert exp4227._bool_text(True) == "true"
    assert exp4227._check_json_resource(tmp_path, "missing", "missing.json") == {
        "resource": "missing",
        "available": False,
        "detail": "missing: missing.json",
    }
    assert exp4227._oracle_distinct_status({"oracle_distinct_beats_vote": True}) == (
        "filled_oracle_distinct_beats_vote"
    )
    assert exp4227._oracle_distinct_status({"headroom_exists": False}) == (
        "open_a2_no_headroom_uninformative"
    )
    assert exp4227._oracle_distinct_status({"oracle_distinct_beats_vote": False}) == (
        "open_a2_ties_vote_with_headroom"
    )
    assert exp4227._reward_status({"verifier_label_carries_signal": True}) == (
        "filled_verifier_label_carries_signal"
    )
    assert exp4227._reward_status({"accumulated_n": {"eval": 0}}) == (
        "open_accumulating_reward_no_eval_yet"
    )
    assert exp4227._reward_status({"a_vs_b_delta": 0.0}) == "open_no_a_vs_b_signal"
    assert exp4227._reward_status({"a_vs_b_delta": None}) == "open_a_vs_b_not_decision_grade"

    _write_minimal_repo(tmp_path)
    (tmp_path / exp4227.EXP4221_PATH).write_text(
        json.dumps(
            {
                "honest_verdict": "complete: oracle_distinct_verifier_beats_vote",
                "oracle_distinct_beats_vote": True,
                "verifier_minus_vote_delta": 0.2,
                "verifier_minus_vote_ci95": [0.1, 0.3],
                "arbiter_override_minus_vote": 0.1,
                "matched_control_delta": 0.2,
                "oracle_at_k": 1.0,
                "verifier_is_oracle": False,
                "headroom_exists": True,
            }
        ),
        encoding="utf-8",
    )
    oracle = exp4227.classify_oracle_distinct_outcome(tmp_path)
    assert oracle["status"] == "filled_oracle_distinct_beats_vote"
    assert oracle["verifier_minus_vote_delta"] == pytest.approx(0.2)


def test_scenario_4227_run_hygiene_writes_artifact_and_ledgers(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4227: run writes the deliverable JSON and ledger entries."""

    _write_minimal_repo(tmp_path)
    artifact = exp4227.run_hygiene(tmp_path, adversarial_runner=_adversarial_clean)
    exp4227.validate_artifact(artifact)

    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4227.GAP_ORACLE_DISTINCT_GAP_ID,
        exp4227.GAP_ORACLE_DISTINCT_A2_GAP_ID,
        exp4227.GAP_REWARD_GAP_ID,
        exp4227.DETECTOR_GAP_ID,
    ]
    assert artifact["registry_updated"] is True
    assert artifact["adversarial_verify"]["methodology_missing_clean"] is True
    written = json.loads((tmp_path / exp4227.EXP4227_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    registry = yaml.safe_load(
        (tmp_path / "ops" / "verifier_registry.yaml").read_text(encoding="utf-8")
    )
    assert exp4227._registry_contains_outcomes(registry) is True
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert exp4227.GAP_ORACLE_DISTINCT_GAP_ID in gaps
    assert exp4227.GAP_ORACLE_DISTINCT_A2_GAP_ID in gaps
    assert exp4227.GAP_REWARD_GAP_ID in gaps
    assert exp4227.DETECTOR_GAP_ID in gaps


def test_req_4227_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    """REQ-VERIFY-4227: failed preconditions write blocked_<resource> and no ledger win."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")

    artifact = exp4227.run_hygiene(tmp_path, adversarial_runner=_adversarial_clean)
    exp4227.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_gap4_arc1_candidate_fixtures"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_updated"] is False
    assert artifact["gaps_updated"] == []
    assert artifact["random_seed"] == exp4227.RANDOM_SEED
    assert artifact["reproducibility_checksum"].startswith("blocked:")
    assert exp4227.GAP_ORACLE_DISTINCT_GAP_ID not in (
        tmp_path / "ops" / "verifier_gaps.md"
    ).read_text(encoding="utf-8")


def test_scenario_4227_results_entrypoint_runs_with_monkeypatched_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4227: required results entrypoint delegates to Exp 4227."""

    called: list[bool] = []
    monkeypatch.setattr(exp4227, "main", lambda: called.append(True))

    runpy.run_path(RESULTS_WRAPPER_PATH.as_posix(), run_name="__main__")

    assert called == [True]

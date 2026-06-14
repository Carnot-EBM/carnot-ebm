"""Tests for Exp 4216 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4216, SCENARIO-VERIFY-4216.
"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4216_verifier_registry_gaps_hygiene as exp4216_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4216 as exp4216


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4216_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4216.GAP4_VERIFIER_ID,
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
        "experiment_4209_oracle_distinct_arc_verifier_build.json",
        "experiment_4210_oracle_distinct_arc_verifier_beats_vote.json",
        "experiment_4211_verifier_as_reward_finish_synchronous.json",
        "experiment_4212_certified_arc_corpus_distill_lift.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def test_req_4216_spec_declared() -> None:
    """REQ-VERIFY-4216: OpenSpec declares the .390 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4216",
        "SCENARIO-VERIFY-4216",
        "python/carnot/experiment_4216_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        "experiment_4216_verifier_registry_gaps_hygiene.json",
        "experiment_4210_oracle_distinct_arc_verifier_beats_vote.json",
        "experiment_4209_oracle_distinct_arc_verifier_build.json",
        "experiment_4208_verifier_as_detector_auroc.json",
        "experiment_4211_verifier_as_reward_finish_synchronous.json",
        "experiment_4212_certified_arc_corpus_distill_lift.json",
        "oracle_distinct_auroc=0.0",
        "sudoku `1.0`, code `1.0`, math `1.0`, and ARC `0.9016`",
        "verifier_label_carries_signal=false",
        "youden_j=0.4137931034482759",
        "distill_lift_ci95=[0.0, 0.0]",
        "GAP-MOAT status SHALL NOT be upgraded",
        exp4216.GAP_ORACLE_DISTINCT_GAP_ID,
        exp4216.DETECTOR_GAP_ID,
        exp4216.GAP_REWARD_GAP_ID,
        exp4216.CERTIFIED_CORPUS_GAP_ID,
    ):
        assert marker in spec
    assert exp4216.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4216.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4216.FIELD_PRINCIPLES["gaps_updated"] in spec
    assert exp4216_wrapper.main is exp4216.main


def test_scenario_4216_preconditions_and_replay_are_bitexact() -> None:
    """SCENARIO-VERIFY-4216: cached GAP-4 replay reproduces exactly."""

    preflight = exp4216.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None
    assert {check["resource"] for check in preflight["checks"]} == {
        "gap4_arc1_candidate_fixtures",
        "verifier_registry",
        "verifier_gaps",
        "exp4208_detector_auroc",
        "exp4209_oracle_distinct_build",
        "exp4210_oracle_distinct_a3",
        "exp4211_verifier_reward_a_vs_b",
        "exp4212_certified_arc_corpus",
    }

    replay = exp4216.replay_gap4_arc1(REPO_ROOT)
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


def test_req_4216_classifies_oracle_detector_reward_and_corpus_truth() -> None:
    """REQ-VERIFY-4216: .390 upstream outcomes are classified without spin."""

    oracle = exp4216.classify_oracle_distinct_outcome(REPO_ROOT)
    assert oracle["gap_id"] == exp4216.GAP_ORACLE_DISTINCT_GAP_ID
    assert oracle["status"] == "open_a3_blocked_selector_not_trained"
    assert oracle["oracle_distinct_beats_vote"] is False
    assert oracle["oracle_distinct_delta"] is None
    assert oracle["oracle_distinct_ci95"] is None
    assert oracle["verifier_is_oracle"] is False
    assert oracle["selector_trained"] is False
    assert oracle["oracle_distinct_auroc"] == pytest.approx(0.0)
    assert oracle["oracle_distinct_auroc_ci95"] == [0.0, 0.0]

    detector = exp4216.classify_detector_aurocs(REPO_ROOT)
    assert detector["gap_id"] == exp4216.DETECTOR_GAP_ID
    assert detector["status"] == "detector_auroc_recorded_all_domains_ci_exclusive"
    assert detector["detection_auroc_by_domain"] == {
        "sudoku": pytest.approx(1.0),
        "code": pytest.approx(1.0),
        "math": pytest.approx(1.0),
        "arc": pytest.approx(0.9016),
    }
    assert detector["verifier_is_oracle_by_domain"]["arc"] is False
    assert detector["selector_headroom_by_domain"]["arc"] == pytest.approx(0.129)

    reward = exp4216.classify_verifier_reward_outcome(REPO_ROOT)
    assert reward["gap_id"] == exp4216.GAP_REWARD_GAP_ID
    assert reward["status"] == "open_accumulating_reward_no_eval_yet"
    assert reward["verifier_label_carries_signal"] is False
    assert reward["a_vs_b_delta"] is None
    assert reward["a_vs_b_ci95"] is None
    assert reward["positive_control_confirmed"] is False
    assert reward["youden_j"] == pytest.approx(0.4137931034482759)
    assert reward["accumulated_n"]["eval"] == 0
    assert reward["verifier_is_oracle"] is True

    corpus = exp4216.classify_certified_arc_corpus(REPO_ROOT)
    assert corpus["gap_id"] == exp4216.CERTIFIED_CORPUS_GAP_ID
    assert corpus["status"] == "certified_corpus_built_distill_lift_absent"
    assert corpus["certified_corpus_size"] == 16
    assert corpus["certification_precision"]["rate"] == pytest.approx(0.9375)
    assert corpus["distill_lift_delta"] == pytest.approx(0.0)
    assert corpus["distill_lift_ci95"] == [0.0, 0.0]
    assert corpus["distill_lift_latent_vs_absent"] == "absent"
    assert corpus["verifier_is_oracle"] is True


def test_scenario_4216_ensure_ledgers_record_outcomes_and_registry_role() -> None:
    """SCENARIO-VERIFY-4216: registry and gaps carry the .390 truth."""

    replay = exp4216.replay_gap4_arc1(REPO_ROOT)
    oracle = exp4216.classify_oracle_distinct_outcome(REPO_ROOT)
    detector = exp4216.classify_detector_aurocs(REPO_ROOT)
    reward = exp4216.classify_verifier_reward_outcome(REPO_ROOT)
    corpus = exp4216.classify_certified_arc_corpus(REPO_ROOT)

    registry, gaps, summary = exp4216.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n\nHistorical note remains.\n",
        replay,
        oracle,
        detector,
        reward,
        corpus,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": [
            exp4216.GAP_ORACLE_DISTINCT_GAP_ID,
            exp4216.DETECTOR_GAP_ID,
            exp4216.GAP_REWARD_GAP_ID,
            exp4216.CERTIFIED_CORPUS_GAP_ID,
        ],
        "oracle_distinct_recorded": True,
        "detector_aurocs_recorded": True,
        "verifier_reward_recorded": True,
        "certified_corpus_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4216"] == exp4216.EXP4216_ARTIFACT_PATH
    assert gap4["eval"]["exp4216_regression_guard_passed"] is True
    assert gap4["eval"]["exp4216_arc1_rule_exec_vote_pass2"] == pytest.approx(0.4516)
    assert gap4["eval"]["exp4216_arc1_rule_exec_gated_pass2"] == pytest.approx(0.5806)
    assert gap4["eval"]["exp4216_oracle_distinct_beats_vote"] is False
    assert gap4["eval"]["exp4216_oracle_distinct_auroc"] == pytest.approx(0.0)
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4216.V390_ROLE_ID)
    assert role["oracle_distinct_status"] == "open_a3_blocked_selector_not_trained"
    assert role["detector_auroc_by_domain"]["arc"] == pytest.approx(0.9016)
    assert role["verifier_label_carries_signal"] is False
    assert role["a_vs_b_delta"] is None
    assert role["distill_lift_latent_vs_absent"] == "absent"
    assert exp4216._registry_contains_outcomes(registry) is True
    assert exp4216._registry_contains_outcomes({}) is False

    assert "Historical note remains." in gaps
    assert exp4216.GAP_ORACLE_DISTINCT_GAP_ID in gaps
    assert "oracle_distinct_beats_vote=false" in gaps
    assert "oracle_distinct_auroc=0.0" in gaps
    assert "verifier_is_oracle=false" in gaps
    assert "GAP-MOAT unchanged" in gaps
    assert exp4216.DETECTOR_GAP_ID in gaps
    assert "arc=0.9016" in gaps
    assert exp4216.GAP_REWARD_GAP_ID in gaps
    assert "verifier_label_carries_signal=false" in gaps
    assert "a_vs_b_delta=None" in gaps
    assert exp4216.CERTIFIED_CORPUS_GAP_ID in gaps
    assert "distill_lift_ci95=[0.0, 0.0]" in gaps
    assert "invisible_leash_diagnosis=absent" in gaps


def test_req_4216_build_artifact_validates_required_fields() -> None:
    """REQ-VERIFY-4216: terminal artifact exposes required schema fields."""

    replay = exp4216.replay_gap4_arc1(REPO_ROOT)
    oracle = exp4216.classify_oracle_distinct_outcome(REPO_ROOT)
    detector = exp4216.classify_detector_aurocs(REPO_ROOT)
    reward = exp4216.classify_verifier_reward_outcome(REPO_ROOT)
    corpus = exp4216.classify_certified_arc_corpus(REPO_ROOT)
    artifact = exp4216.build_artifact(
        offline_replay=replay,
        oracle_distinct_outcome=oracle,
        detector_aurocs=detector,
        verifier_reward_outcome=reward,
        certified_arc_corpus=corpus,
        registry_updated=True,
        gaps_updated=[
            exp4216.GAP_ORACLE_DISTINCT_GAP_ID,
            exp4216.DETECTOR_GAP_ID,
            exp4216.GAP_REWARD_GAP_ID,
            exp4216.CERTIFIED_CORPUS_GAP_ID,
        ],
        duration_s=0.012,
    )

    exp4216.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4216.GAP_ORACLE_DISTINCT_GAP_ID,
        exp4216.DETECTOR_GAP_ID,
        exp4216.GAP_REWARD_GAP_ID,
        exp4216.CERTIFIED_CORPUS_GAP_ID,
    ]
    assert artifact["field_principles"] == exp4216.FIELD_PRINCIPLES
    assert artifact["oracle_distinct_outcome"]["verifier_is_oracle"] is False
    assert artifact["detector_aurocs"]["detection_auroc_by_domain"]["arc"] == pytest.approx(0.9016)
    assert artifact["verifier_reward_outcome"]["verifier_is_oracle"] is True
    assert artifact["certified_arc_corpus"]["verifier_is_oracle"] is True
    assert artifact["cited_upstream_artifacts"] == [
        exp4216.ARC1_POOL_PATH,
        exp4216.ARC1_PROGRAMS_PATH,
        exp4216.EXP4208_PATH,
        exp4216.EXP4209_PATH,
        exp4216.EXP4210_PATH,
        exp4216.EXP4211_PATH,
        exp4216.EXP4212_PATH,
    ]

    for field in exp4216.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4216.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4216.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4216.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4216.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="gaps_updated"):
        exp4216.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="field_principles"):
        exp4216.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4216.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_4216_helper_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4216: schema helpers fail closed without hidden inference."""

    assert exp4216._numeric_or_none(True) is None
    assert exp4216._numeric_or_none("bad") is None
    assert exp4216._first_numeric({"bad": "x", "ok": 2}, "bad", "ok") == pytest.approx(2.0)
    assert exp4216._first_ci95({"bad": [0.0], "ok": [0, 1]}, "bad", "ok") == [0.0, 1.0]
    assert exp4216._round4(None) is None
    assert exp4216._round4(1.23456) == pytest.approx(1.2346)
    assert exp4216._bool_text(True) == "true"
    assert exp4216._check_json_resource(tmp_path, "missing", "missing.json") == {
        "resource": "missing",
        "available": False,
        "detail": "missing: missing.json",
    }
    assert exp4216._oracle_distinct_status({"oracle_distinct_beats_vote": True}) == (
        "filled_oracle_distinct_beats_vote"
    )
    assert exp4216._oracle_distinct_status({"selector_trained": False}) == (
        "open_a3_blocked_selector_not_trained"
    )
    assert exp4216._oracle_distinct_status({"selector_trained": True}) == (
        "open_a3_not_decision_grade"
    )
    assert exp4216._detector_status({}) == "detector_auroc_recorded_with_open_domain_caveats"
    assert exp4216._reward_status({"verifier_label_carries_signal": True}) == (
        "filled_verifier_label_carries_signal"
    )
    assert exp4216._reward_status({"accumulated_n": {"eval": 0}}) == (
        "open_accumulating_reward_no_eval_yet"
    )
    assert exp4216._reward_status({"a_vs_b_delta": 0.0}) == "open_no_a_vs_b_signal"
    assert exp4216._reward_status({"a_vs_b_delta": None}) == "open_a_vs_b_not_decision_grade"
    assert exp4216._certified_corpus_status({"certified_corpus_size": 0}) == (
        "certified_corpus_empty"
    )
    assert (
        exp4216._certified_corpus_status(
            {"certified_corpus_size": 1, "distill_lift_latent_vs_absent": "latent"}
        )
        == "certified_corpus_built_distill_lift_latent"
    )
    assert (
        exp4216._certified_corpus_status(
            {"certified_corpus_size": 1, "distill_lift_latent_vs_absent": "uninformative"}
        )
        == "certified_corpus_built_distill_lift_uninformative"
    )

    _write_minimal_repo(tmp_path)
    (tmp_path / exp4216.EXP4210_PATH).write_text(
        json.dumps(
            {
                "honest_verdict": "complete: oracle_distinct_beats_vote",
                "oracle_distinct_beats_vote": True,
                "oracle_distinct_delta": 0.2,
                "oracle_distinct_ci95": [0.1, 0.3],
                "verifier_is_oracle": False,
            }
        ),
        encoding="utf-8",
    )
    oracle = exp4216.classify_oracle_distinct_outcome(tmp_path)
    assert oracle["status"] == "filled_oracle_distinct_beats_vote"
    assert oracle["oracle_distinct_delta"] == pytest.approx(0.2)


def test_scenario_4216_run_hygiene_writes_artifact_and_ledgers(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4216: run writes the deliverable JSON and ledger entries."""

    _write_minimal_repo(tmp_path)
    artifact = exp4216.run_hygiene(tmp_path)
    exp4216.validate_artifact(artifact)

    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4216.GAP_ORACLE_DISTINCT_GAP_ID,
        exp4216.DETECTOR_GAP_ID,
        exp4216.GAP_REWARD_GAP_ID,
        exp4216.CERTIFIED_CORPUS_GAP_ID,
    ]
    assert artifact["registry_updated"] is True
    written = json.loads((tmp_path / exp4216.EXP4216_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    registry = yaml.safe_load(
        (tmp_path / "ops" / "verifier_registry.yaml").read_text(encoding="utf-8")
    )
    assert exp4216._registry_contains_outcomes(registry) is True
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert exp4216.GAP_ORACLE_DISTINCT_GAP_ID in gaps
    assert exp4216.DETECTOR_GAP_ID in gaps
    assert exp4216.GAP_REWARD_GAP_ID in gaps
    assert exp4216.CERTIFIED_CORPUS_GAP_ID in gaps


def test_req_4216_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    """REQ-VERIFY-4216: failed preconditions write blocked_<resource> and no ledger win."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")

    artifact = exp4216.run_hygiene(tmp_path)
    exp4216.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_gap4_arc1_candidate_fixtures"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_updated"] is False
    assert artifact["gaps_updated"] == []
    assert exp4216.GAP_ORACLE_DISTINCT_GAP_ID not in (
        tmp_path / "ops" / "verifier_gaps.md"
    ).read_text(encoding="utf-8")


def test_scenario_4216_results_entrypoint_runs_with_monkeypatched_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4216: required results entrypoint delegates to Exp 4216."""

    called: list[bool] = []
    monkeypatch.setattr(exp4216, "main", lambda: called.append(True))

    runpy.run_path(RESULTS_WRAPPER_PATH.as_posix(), run_name="__main__")

    assert called == [True]

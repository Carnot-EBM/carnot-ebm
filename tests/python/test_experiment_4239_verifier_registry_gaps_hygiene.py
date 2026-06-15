"""Tests for Exp 4239 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4239, SCENARIO-VERIFY-4239.
"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4239_verifier_registry_gaps_hygiene as exp4239_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4239 as exp4239


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4239_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4239.GAP4_VERIFIER_ID,
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
        "experiment_4231_oracle_distinct_arc_aggregator_build.json",
        "experiment_4232_oracle_distinct_arc_aggregator_beats_vote.json",
        "experiment_4233_oracle_distinct_code_beats_vote.json",
        "experiment_4235_verifier_as_reward_3arm_window_boxed.json",
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


def test_req_4239_spec_declared() -> None:
    """REQ-VERIFY-4239: OpenSpec declares the .392 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4239",
        "SCENARIO-VERIFY-4239",
        "python/carnot/experiment_4239_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        "experiment_4239_verifier_registry_gaps_hygiene.json",
        "experiment_4232_oracle_distinct_arc_aggregator_beats_vote.json",
        "experiment_4231_oracle_distinct_arc_aggregator_build.json",
        "experiment_4233_oracle_distinct_code_beats_vote.json",
        "experiment_4235_verifier_as_reward_3arm_window_boxed.json",
        "aggregator_minus_vote_delta=0.0",
        "aggregator_minus_vote_ci95=[0.0, 0.0]",
        "held_out_task_n=52",
        "oracle_distinct_auroc=0.7865558646",
        "wrong_majority_n=9",
        "disambiguation_read=ARC_null_is_data_sparsity",
        "code_predictor_minus_vote_delta=0.03125",
        "code_predictor_minus_vote_ci95=[0.00625, 0.0625]",
        "live_lora_retired=false",
        "inference_substrate=verifier_ensemble_against_cached_candidates",
        exp4239.GAP_ORACLE_DISTINCT_GAP_ID,
        exp4239.GAP_ORACLE_DISTINCT_A2_GAP_ID,
        exp4239.GAP_CODE_DISAMBIGUATION_GAP_ID,
        exp4239.GAP_REWARD_GAP_ID,
    ):
        assert marker in spec
    assert exp4239.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4239.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4239.FIELD_PRINCIPLES["gaps_updated"] in spec
    assert exp4239.FIELD_PRINCIPLES["random_seed"] in spec
    assert exp4239.FIELD_PRINCIPLES["reproducibility_checksum"] in spec
    assert exp4239_wrapper.main is exp4239.main


def test_scenario_4239_preconditions_replay_and_checksum_are_bitexact() -> None:
    """SCENARIO-VERIFY-4239: cached GAP-4 replay and checksum are stable."""

    preflight = exp4239.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None
    assert {check["resource"] for check in preflight["checks"]} == {
        "gap4_arc1_candidate_fixtures",
        "verifier_registry",
        "verifier_gaps",
        "exp4231_oracle_distinct_aggregator_build",
        "exp4232_oracle_distinct_a2",
        "exp4233_code_disambiguation",
        "exp4235_verifier_reward_a_vs_b",
    }

    replay = exp4239.replay_gap4_arc1(REPO_ROOT)
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

    checksum = exp4239.candidate_set_checksum(REPO_ROOT)
    assert checksum.startswith("sha256:")
    assert len(checksum) == len("sha256:") + 64
    assert checksum == exp4239.candidate_set_checksum(REPO_ROOT)


def test_req_4239_classifies_oracle_code_and_reward_truth() -> None:
    """REQ-VERIFY-4239: .392 upstream outcomes are recorded without promotion."""

    oracle = exp4239.classify_oracle_distinct_outcome(REPO_ROOT)
    assert oracle["gap_id"] == exp4239.GAP_ORACLE_DISTINCT_GAP_ID
    assert oracle["a2_gap_id"] == exp4239.GAP_ORACLE_DISTINCT_A2_GAP_ID
    assert oracle["status"] == "open_a2_ties_vote_with_headroom_at_power"
    assert oracle["oracle_distinct_beats_vote"] is False
    assert oracle["aggregator_minus_vote_delta"] == pytest.approx(0.0)
    assert oracle["aggregator_minus_vote_ci95"] == [0.0, 0.0]
    assert oracle["held_out_task_n"] == 52
    assert oracle["matched_control_delta"] == pytest.approx(0.0384615385)
    assert oracle["oracle_at_k"] == pytest.approx(0.3653846154)
    assert oracle["verifier_is_oracle"] is False
    assert oracle["oracle_distinct_auroc"] == pytest.approx(0.7865558646)
    assert oracle["oracle_distinct_auroc_ci95"] == [0.6319719028, 0.9258842843]
    assert oracle["wrong_majority_n"] == 9
    assert oracle["build_flagged_adversarial"] is True
    assert oracle["gap_moat_update"] == "unchanged_v392_ties_vote_with_headroom"

    code = exp4239.classify_code_disambiguation_outcome(REPO_ROOT)
    assert code["gap_id"] == exp4239.GAP_CODE_DISAMBIGUATION_GAP_ID
    assert code["status"] == "filled_code_oracle_distinct_beats_vote"
    assert code["disambiguation_read"] == "ARC_null_is_data_sparsity"
    assert code["code_oracle_distinct_beats_vote"] is True
    assert code["code_predictor_minus_vote_delta"] == pytest.approx(0.03125)
    assert code["code_predictor_minus_vote_ci95"] == [0.00625, 0.0625]
    assert code["held_out_task_n"] == 160
    assert code["verifier_is_oracle"] is False

    reward = exp4239.classify_verifier_reward_outcome(REPO_ROOT)
    assert reward["gap_id"] == exp4239.GAP_REWARD_GAP_ID
    assert reward["status"] == "open_live_lora_blocked_pre_gate"
    assert reward["verifier_label_carries_signal"] is False
    assert reward["a_vs_b_delta"] is None
    assert reward["a_vs_b_ci95"] is None
    assert reward["youden_j"] is None
    assert reward["live_lora_retired"] is False
    assert reward["blocked_at_layer"] == "conductor_pre_gate"


def test_scenario_4239_ensure_ledgers_record_outcomes_and_registry_role() -> None:
    """SCENARIO-VERIFY-4239: registry and gaps carry the .392 truth."""

    replay = exp4239.replay_gap4_arc1(REPO_ROOT)
    oracle = exp4239.classify_oracle_distinct_outcome(REPO_ROOT)
    code = exp4239.classify_code_disambiguation_outcome(REPO_ROOT)
    reward = exp4239.classify_verifier_reward_outcome(REPO_ROOT)

    registry, gaps, summary = exp4239.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n\nHistorical note remains.\n",
        replay,
        oracle,
        code,
        reward,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": [
            exp4239.GAP_ORACLE_DISTINCT_GAP_ID,
            exp4239.GAP_ORACLE_DISTINCT_A2_GAP_ID,
            exp4239.GAP_CODE_DISAMBIGUATION_GAP_ID,
            exp4239.GAP_REWARD_GAP_ID,
        ],
        "oracle_distinct_recorded": True,
        "strengthened_a2_recorded": True,
        "code_disambiguation_recorded": True,
        "verifier_reward_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4239"] == exp4239.EXP4239_ARTIFACT_PATH
    assert gap4["eval"]["exp4239_regression_guard_passed"] is True
    assert gap4["eval"]["exp4239_arc1_rule_exec_vote_pass2"] == pytest.approx(0.4516)
    assert gap4["eval"]["exp4239_arc1_rule_exec_gated_pass2"] == pytest.approx(0.5806)
    assert gap4["eval"]["exp4239_oracle_distinct_beats_vote"] is False
    assert gap4["eval"]["exp4239_oracle_distinct_auroc"] == pytest.approx(0.7865558646)
    assert gap4["eval"]["exp4239_code_oracle_distinct_beats_vote"] is True
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4239.V392_ROLE_ID)
    assert role["oracle_distinct_status"] == "open_a2_ties_vote_with_headroom_at_power"
    assert role["wrong_majority_n"] == 9
    assert role["aggregator_minus_vote_delta"] == pytest.approx(0.0)
    assert role["code_disambiguation_read"] == "ARC_null_is_data_sparsity"
    assert role["verifier_label_carries_signal"] is False
    assert role["live_lora_retired"] is False
    assert exp4239._registry_contains_outcomes(registry) is True
    assert exp4239._registry_contains_outcomes({}) is False
    missing_registry: dict[str, Any] = {}
    exp4239._ensure_gap4_eval(missing_registry, replay, oracle, code, reward)
    assert missing_registry["verifiers"][0]["verifier_id"] == exp4239.GAP4_VERIFIER_ID
    empty_registry: dict[str, Any] = {}
    exp4239._ensure_v392_role(empty_registry, oracle, code, reward)
    assert empty_registry == {}

    assert "Historical note remains." in gaps
    assert exp4239.GAP_ORACLE_DISTINCT_GAP_ID in gaps
    assert exp4239.GAP_ORACLE_DISTINCT_A2_GAP_ID in gaps
    assert exp4239.GAP_CODE_DISAMBIGUATION_GAP_ID in gaps
    assert exp4239.GAP_REWARD_GAP_ID in gaps
    assert "oracle_distinct_beats_vote=false" in gaps
    assert "aggregator_minus_vote_delta=0.0" in gaps
    assert "aggregator_minus_vote_ci95=[0.0, 0.0]" in gaps
    assert "oracle_distinct_auroc=0.7865558646" in gaps
    assert "wrong_majority_n=9" in gaps
    assert "GAP-MOAT unchanged" in gaps
    assert "code_oracle_distinct_beats_vote=true" in gaps
    assert "code_predictor_minus_vote_delta=0.03125" in gaps
    assert "disambiguation_read=ARC_null_is_data_sparsity" in gaps
    assert "live_lora_retired=false" in gaps


def test_req_4239_build_artifact_validates_methodology_fields() -> None:
    """REQ-VERIFY-4239: terminal artifact exposes required schema fields."""

    replay = exp4239.replay_gap4_arc1(REPO_ROOT)
    oracle = exp4239.classify_oracle_distinct_outcome(REPO_ROOT)
    code = exp4239.classify_code_disambiguation_outcome(REPO_ROOT)
    reward = exp4239.classify_verifier_reward_outcome(REPO_ROOT)
    checksum = exp4239.candidate_set_checksum(REPO_ROOT)
    artifact = exp4239.build_artifact(
        offline_replay=replay,
        oracle_distinct_outcome=oracle,
        code_disambiguation_outcome=code,
        verifier_reward_outcome=reward,
        registry_updated=True,
        gaps_updated=[
            exp4239.GAP_ORACLE_DISTINCT_GAP_ID,
            exp4239.GAP_ORACLE_DISTINCT_A2_GAP_ID,
            exp4239.GAP_CODE_DISAMBIGUATION_GAP_ID,
            exp4239.GAP_REWARD_GAP_ID,
        ],
        random_seed=exp4239.RANDOM_SEED,
        reproducibility_checksum=checksum,
        duration_s=0.012,
    )

    exp4239.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4239.GAP_ORACLE_DISTINCT_GAP_ID,
        exp4239.GAP_ORACLE_DISTINCT_A2_GAP_ID,
        exp4239.GAP_CODE_DISAMBIGUATION_GAP_ID,
        exp4239.GAP_REWARD_GAP_ID,
    ]
    assert artifact["field_principles"] == exp4239.FIELD_PRINCIPLES
    assert artifact["random_seed"] == exp4239.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == checksum
    assert artifact["model_specs"]["candidate_set_sha256"] == checksum
    assert artifact["model_specs"]["trm_training_touched"] is False
    assert artifact["inference_substrate"] == exp4239.INFERENCE_SUBSTRATE
    assert artifact["adversarial_verify"]["status"] == "pending"
    assert artifact["cited_upstream_artifacts"] == [
        exp4239.ARC1_POOL_PATH,
        exp4239.ARC1_PROGRAMS_PATH,
        exp4239.EXP4231_PATH,
        exp4239.EXP4232_PATH,
        exp4239.EXP4233_PATH,
        exp4239.EXP4235_PATH,
    ]

    for field in exp4239.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4239.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4239.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4239.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4239.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="gaps_updated"):
        exp4239.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="random_seed"):
        exp4239.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4239.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="model_specs"):
        exp4239.validate_artifact({**artifact, "model_specs": {}})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4239.validate_artifact({**artifact, "inference_substrate": "live"})
    with pytest.raises(ValueError, match="field_principles"):
        exp4239.validate_artifact({**artifact, "field_principles": {}})


def test_scenario_4239_run_hygiene_writes_artifact_and_ledgers(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4239: run writes the deliverable JSON and ledger entries."""

    _write_minimal_repo(tmp_path)
    artifact = exp4239.run_hygiene(tmp_path, adversarial_runner=_adversarial_clean)
    exp4239.validate_artifact(artifact)

    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4239.GAP_ORACLE_DISTINCT_GAP_ID,
        exp4239.GAP_ORACLE_DISTINCT_A2_GAP_ID,
        exp4239.GAP_CODE_DISAMBIGUATION_GAP_ID,
        exp4239.GAP_REWARD_GAP_ID,
    ]
    assert artifact["registry_updated"] is True
    assert artifact["adversarial_verify"]["methodology_missing_clean"] is True
    written = json.loads((tmp_path / exp4239.EXP4239_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    registry = yaml.safe_load(
        (tmp_path / "ops" / "verifier_registry.yaml").read_text(encoding="utf-8")
    )
    assert exp4239._registry_contains_outcomes(registry) is True
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert exp4239.GAP_ORACLE_DISTINCT_GAP_ID in gaps
    assert exp4239.GAP_ORACLE_DISTINCT_A2_GAP_ID in gaps
    assert exp4239.GAP_CODE_DISAMBIGUATION_GAP_ID in gaps
    assert exp4239.GAP_REWARD_GAP_ID in gaps


def test_req_4239_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    """REQ-VERIFY-4239: failed preconditions write blocked_<resource> and no ledger win."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")

    artifact = exp4239.run_hygiene(tmp_path, adversarial_runner=_adversarial_clean)
    exp4239.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_gap4_arc1_candidate_fixtures"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_updated"] is False
    assert artifact["gaps_updated"] == []
    assert artifact["random_seed"] == exp4239.RANDOM_SEED
    assert artifact["reproducibility_checksum"].startswith("blocked:")
    assert exp4239.GAP_ORACLE_DISTINCT_GAP_ID not in (
        tmp_path / "ops" / "verifier_gaps.md"
    ).read_text(encoding="utf-8")


def test_scenario_4239_results_entrypoint_runs_with_monkeypatched_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4239: required results entrypoint delegates to Exp 4239."""

    called: list[bool] = []
    monkeypatch.setattr(exp4239, "main", lambda: called.append(True))

    runpy.run_path(RESULTS_WRAPPER_PATH.as_posix(), run_name="__main__")

    assert called == [True]

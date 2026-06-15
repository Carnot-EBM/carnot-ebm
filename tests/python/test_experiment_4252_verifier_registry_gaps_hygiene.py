"""Tests for Exp 4252 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4252, SCENARIO-VERIFY-4252.
"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4252_verifier_registry_gaps_hygiene as exp4252_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4252 as exp4252


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4252_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4252.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "code_path": "python/carnot/agentic/gap4_program_induction_stack.py",
                "eval": {"metric": "pass_at_1"},
                "registry_roles": [],
            }
        ]
    }


def _minimal_manifest() -> dict[str, Any]:
    return {
        "retired": [],
        "retired_experiments": [],
        "retired_extras": [],
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
    (tmp_path / "ops" / "exclusion_manifest.yaml").write_text(
        yaml.safe_dump(_minimal_manifest(), sort_keys=False),
        encoding="utf-8",
    )
    for name in (
        "arc3_gap3_stage2_eval_pool.json.gz",
        "arc3_gap4_induced_programs.json",
        "experiment_4244_arc_set_encoder_aggregator_build.json",
        "experiment_4245_arc_set_encoder_beats_vote.json",
        "experiment_4246_code_oracle_distinct_replication.json",
        "experiment_4247_verifier_reward_offline_harness_retire_livelora.json",
        "experiment_4248_verifier_as_reward_offline_3arm.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {
        "returncode": 0,
        "reports": [{"flags": [], "flag_count": 0, "max_severity": -1}],
    }


def test_req_4252_spec_declared() -> None:
    """REQ-VERIFY-4252: OpenSpec declares the .393 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4252",
        "SCENARIO-VERIFY-4252",
        "python/carnot/experiment_4252_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        "experiment_4252_verifier_registry_gaps_hygiene.json",
        "experiment_4245_arc_set_encoder_beats_vote.json",
        "experiment_4244_arc_set_encoder_aggregator_build.json",
        "experiment_4246_code_oracle_distinct_replication.json",
        "experiment_4248_verifier_as_reward_offline_3arm.json",
        "experiment_4247_verifier_reward_offline_harness_retire_livelora.json",
        "set_encoder_minus_vote_delta=0.4423076923",
        "set_encoder_minus_vote_ci95=[0.3076923077, 0.5961538462]",
        "oracle_distinct_auroc=0.9633173387",
        "set_encoder_vs_logistic_auroc_delta=-0.0161846276",
        "wrong_majority_n=30",
        "replication_read=blocked_code_second_corpus_missing",
        "code_predictor_minus_vote_delta=0.0",
        "code_predictor_minus_vote_ci95=[0.0, 0.0]",
        "verifier_label_carries_signal=false",
        "live_lora_retired=true",
        "operator_reopen_required=true",
        "inference_substrate=verifier_ensemble_against_cached_candidates",
        exp4252.GAP_ORACLE_DISTINCT_GAP_ID,
        exp4252.GAP_ORACLE_DISTINCT_A3_GAP_ID,
        exp4252.GAP_CODE_REPLICATION_GAP_ID,
        exp4252.GAP_REWARD_GAP_ID,
        exp4252.GAP_LIVE_LORA_RETIREMENT_GAP_ID,
    ):
        assert marker in spec
    assert exp4252.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4252.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4252.FIELD_PRINCIPLES["gaps_updated"] in spec
    assert exp4252.FIELD_PRINCIPLES["live_lora_retired_recorded"] in spec
    assert exp4252.FIELD_PRINCIPLES["random_seed"] in spec
    assert exp4252.FIELD_PRINCIPLES["reproducibility_checksum"] in spec
    assert exp4252_wrapper.main is exp4252.main


def test_scenario_4252_preconditions_replay_and_checksum_are_bitexact() -> None:
    """SCENARIO-VERIFY-4252: cached GAP-4 replay and checksum are stable."""

    preflight = exp4252.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None
    assert {check["resource"] for check in preflight["checks"]} == {
        "gap4_arc1_candidate_fixtures",
        "verifier_registry",
        "verifier_gaps",
        "exclusion_manifest",
        "exp4244_arc_set_encoder_build",
        "exp4245_arc_set_encoder_a3",
        "exp4246_code_replication",
        "exp4247_live_lora_retirement",
        "exp4248_offline_reward_a_vs_b",
    }

    replay = exp4252.replay_gap4_arc1(REPO_ROOT)
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

    checksum = exp4252.candidate_set_checksum(REPO_ROOT)
    assert checksum.startswith("sha256:")
    assert len(checksum) == len("sha256:") + 64
    assert checksum == exp4252.candidate_set_checksum(REPO_ROOT)


def test_req_4252_classifies_oracle_code_reward_and_retirement_truth() -> None:
    """REQ-VERIFY-4252: .393 upstream outcomes are recorded without fabrication."""

    oracle = exp4252.classify_oracle_distinct_outcome(REPO_ROOT)
    assert oracle["gap_id"] == exp4252.GAP_ORACLE_DISTINCT_GAP_ID
    assert oracle["a3_gap_id"] == exp4252.GAP_ORACLE_DISTINCT_A3_GAP_ID
    assert oracle["status"] == "filled_arc_a3_set_encoder_beats_vote_non_oracle"
    assert oracle["oracle_distinct_beats_vote"] is True
    assert oracle["set_encoder_minus_vote_delta"] == pytest.approx(0.4423076923)
    assert oracle["set_encoder_minus_vote_ci95"] == [0.3076923077, 0.5961538462]
    assert oracle["held_out_task_n"] == 52
    assert oracle["matched_control_delta"] == pytest.approx(0.4807692308)
    assert oracle["oracle_at_k"] == pytest.approx(0.8269230769)
    assert oracle["oracle_minus_vote"] == pytest.approx(0.5769230769)
    assert oracle["verifier_is_oracle"] is False
    assert oracle["oracle_distinct_auroc"] == pytest.approx(0.9633173387)
    assert oracle["set_encoder_vs_logistic_auroc_delta"] == pytest.approx(-0.0161846276)
    assert oracle["wrong_majority_n"] == 30
    assert oracle["gap_oracle_distinct_update"] == "changed_v393_grown_pool_set_encoder_beats_vote"

    code = exp4252.classify_code_replication_outcome(REPO_ROOT)
    assert code["gap_id"] == exp4252.GAP_CODE_REPLICATION_GAP_ID
    assert code["status"] == "blocked_code_second_corpus_missing"
    assert code["replication_read"] == "blocked_code_second_corpus_missing"
    assert code["code_replication_beats_vote"] is False
    assert code["code_predictor_minus_vote_delta"] == pytest.approx(0.0)
    assert code["code_predictor_minus_vote_ci95"] == [0.0, 0.0]
    assert code["held_out_task_n"] == 0
    assert code["verifier_is_oracle"] is False

    reward = exp4252.classify_verifier_reward_outcome(REPO_ROOT)
    assert reward["gap_id"] == exp4252.GAP_REWARD_GAP_ID
    assert reward["status"] == "blocked_offline_reward_gate_failed_live_lora_retired"
    assert reward["verifier_label_carries_signal"] is False
    assert reward["a_vs_b_delta"] is None
    assert reward["a_vs_b_ci95"] is None
    assert reward["youden_j"] is None
    assert reward["live_lora_retired"] is True
    assert reward["blocked_at_layer"] == "conductor_pre_gate"
    assert reward["verifier_is_oracle"] is True

    retirement = exp4252.classify_live_lora_retirement(REPO_ROOT)
    assert retirement["gap_id"] == exp4252.GAP_LIVE_LORA_RETIREMENT_GAP_ID
    assert retirement["live_lora_retired"] is True
    assert retirement["infra_failure_count"] == 6
    assert retirement["operator_reopen_required"] is True
    assert retirement["retire_if_same_verdict"] is True


def test_scenario_4252_ensure_ledgers_record_outcomes_registry_and_manifest() -> None:
    """SCENARIO-VERIFY-4252: registry, gaps, and manifest carry the .393 truth."""

    replay = exp4252.replay_gap4_arc1(REPO_ROOT)
    oracle = exp4252.classify_oracle_distinct_outcome(REPO_ROOT)
    code = exp4252.classify_code_replication_outcome(REPO_ROOT)
    reward = exp4252.classify_verifier_reward_outcome(REPO_ROOT)
    retirement = exp4252.classify_live_lora_retirement(REPO_ROOT)

    registry, gaps, manifest, summary = exp4252.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n\nHistorical note remains.\n",
        _minimal_manifest(),
        replay,
        oracle,
        code,
        reward,
        retirement,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": [
            exp4252.GAP_ORACLE_DISTINCT_GAP_ID,
            exp4252.GAP_ORACLE_DISTINCT_A3_GAP_ID,
            exp4252.GAP_CODE_REPLICATION_GAP_ID,
            exp4252.GAP_REWARD_GAP_ID,
            exp4252.GAP_LIVE_LORA_RETIREMENT_GAP_ID,
        ],
        "oracle_distinct_recorded": True,
        "arc_a3_recorded": True,
        "code_replication_recorded": True,
        "verifier_reward_recorded": True,
        "live_lora_retired_recorded": True,
        "exclusion_manifest_updated": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4252"] == exp4252.EXP4252_ARTIFACT_PATH
    assert gap4["eval"]["exp4252_regression_guard_passed"] is True
    assert gap4["eval"]["exp4252_arc1_rule_exec_vote_pass2"] == pytest.approx(0.4516)
    assert gap4["eval"]["exp4252_arc1_rule_exec_gated_pass2"] == pytest.approx(0.5806)
    assert gap4["eval"]["exp4252_oracle_distinct_beats_vote"] is True
    assert gap4["eval"]["exp4252_set_encoder_minus_vote_delta"] == pytest.approx(0.4423076923)
    assert gap4["eval"]["exp4252_oracle_distinct_auroc"] == pytest.approx(0.9633173387)
    assert gap4["eval"]["exp4252_code_replication_beats_vote"] is False
    assert gap4["eval"]["exp4252_verifier_label_carries_signal"] is False
    assert gap4["eval"]["exp4252_live_lora_retired"] is True
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4252.V393_ROLE_ID)
    assert role["oracle_distinct_status"] == "filled_arc_a3_set_encoder_beats_vote_non_oracle"
    assert role["wrong_majority_n"] == 30
    assert role["set_encoder_minus_vote_delta"] == pytest.approx(0.4423076923)
    assert role["replication_read"] == "blocked_code_second_corpus_missing"
    assert role["verifier_label_carries_signal"] is False
    assert role["live_lora_retired_recorded"] is True
    assert exp4252._registry_contains_outcomes(registry) is True
    assert exp4252._registry_contains_outcomes({}) is False
    missing_registry: dict[str, Any] = {}
    exp4252._ensure_gap4_eval(missing_registry, replay, oracle, code, reward, retirement)
    assert missing_registry["verifiers"][0]["verifier_id"] == exp4252.GAP4_VERIFIER_ID
    empty_registry: dict[str, Any] = {}
    exp4252._ensure_v393_role(empty_registry, oracle, code, reward, retirement)
    assert empty_registry == {}

    assert "Historical note remains." in gaps
    assert "set_encoder_minus_vote_delta=0.4423076923" in gaps
    assert "set_encoder_minus_vote_ci95=[0.3076923077, 0.5961538462]" in gaps
    assert "oracle_distinct_auroc=0.9633173387" in gaps
    assert "set_encoder_vs_logistic_auroc_delta=-0.0161846276" in gaps
    assert "wrong_majority_n=30" in gaps
    assert "changed the .392 ties-vote read" in gaps
    assert "replication_read=blocked_code_second_corpus_missing" in gaps
    assert "verifier_label_carries_signal=false" in gaps
    assert "live_lora_retired=true" in gaps
    assert "operator_reopen_required=true" in gaps
    assert "6 infra failures" in gaps

    entry = exp4252._find_live_lora_manifest_entry(manifest)
    assert entry is not None
    assert entry["operator_reopen_required"] is True
    assert entry["retire_if_same_verdict"] is True
    assert entry["infra_failure_count"] == 6


def test_req_4252_manifest_helpers_cover_retirement_idempotence(tmp_path: Path) -> None:
    """REQ-VERIFY-4252: live-LoRA manifest entry is normalized and idempotent."""

    list_manifest_path = tmp_path / "manifest.yaml"
    list_manifest_path.write_text("- not-a-mapping\n", encoding="utf-8")
    assert exp4252._load_manifest(list_manifest_path) == _minimal_manifest()
    assert exp4252._load_manifest_for_check(list_manifest_path) == _minimal_manifest()

    assert (
        exp4252._find_live_lora_manifest_entry(
            {"retired_extras": "bad-shape", "retired_experiments": [None]}
        )
        is None
    )
    scoped_entry = {"experiment_scope": "live-LoRA verifier-as-reward path"}
    assert exp4252._find_live_lora_manifest_entry({"retired_extras": [scoped_entry]}) is scoped_entry

    not_retired = {"live_lora_retired": False}
    manifest = _minimal_manifest()
    assert exp4252._ensure_live_lora_manifest_retirement(manifest, not_retired) is False
    assert exp4252._find_live_lora_manifest_entry(manifest) is None

    retirement = exp4252.classify_live_lora_retirement(REPO_ROOT)
    manifest = {
        "retired_extras": [
            {
                "id": exp4252.LIVE_LORA_RETIREMENT_ENTRY_ID,
                "operator_reopen_required": False,
            }
        ]
    }
    assert exp4252._ensure_live_lora_manifest_retirement(manifest, retirement) is True
    updated = exp4252._find_live_lora_manifest_entry(manifest)
    assert updated is not None
    assert updated["operator_reopen_required"] is True
    assert exp4252._ensure_live_lora_manifest_retirement(manifest, retirement) is False


def test_req_4252_build_artifact_validates_methodology_fields() -> None:
    """REQ-VERIFY-4252: terminal artifact exposes required schema fields."""

    replay = exp4252.replay_gap4_arc1(REPO_ROOT)
    oracle = exp4252.classify_oracle_distinct_outcome(REPO_ROOT)
    code = exp4252.classify_code_replication_outcome(REPO_ROOT)
    reward = exp4252.classify_verifier_reward_outcome(REPO_ROOT)
    retirement = exp4252.classify_live_lora_retirement(REPO_ROOT)
    checksum = exp4252.candidate_set_checksum(REPO_ROOT)
    gaps_updated = [
        exp4252.GAP_ORACLE_DISTINCT_GAP_ID,
        exp4252.GAP_ORACLE_DISTINCT_A3_GAP_ID,
        exp4252.GAP_CODE_REPLICATION_GAP_ID,
        exp4252.GAP_REWARD_GAP_ID,
        exp4252.GAP_LIVE_LORA_RETIREMENT_GAP_ID,
    ]
    artifact = exp4252.build_artifact(
        offline_replay=replay,
        oracle_distinct_outcome=oracle,
        code_replication_outcome=code,
        verifier_reward_outcome=reward,
        live_lora_retirement=retirement,
        registry_updated=True,
        gaps_updated=gaps_updated,
        live_lora_retired_recorded=True,
        random_seed=exp4252.RANDOM_SEED,
        reproducibility_checksum=checksum,
        duration_s=0.012,
    )

    exp4252.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == gaps_updated
    assert artifact["live_lora_retired_recorded"] is True
    assert artifact["field_principles"] == exp4252.FIELD_PRINCIPLES
    assert artifact["random_seed"] == exp4252.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == checksum
    assert artifact["model_specs"]["candidate_set_sha256"] == checksum
    assert artifact["model_specs"]["trm_training_touched"] is False
    assert artifact["inference_substrate"] == exp4252.INFERENCE_SUBSTRATE
    assert artifact["adversarial_verify"]["status"] == "pending"
    assert artifact["cited_upstream_artifacts"] == [
        exp4252.ARC1_POOL_PATH,
        exp4252.ARC1_PROGRAMS_PATH,
        exp4252.EXP4244_PATH,
        exp4252.EXP4245_PATH,
        exp4252.EXP4246_PATH,
        exp4252.EXP4247_PATH,
        exp4252.EXP4248_PATH,
    ]

    for field in exp4252.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4252.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4252.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4252.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4252.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="live_lora_retired_recorded"):
        exp4252.validate_artifact({**artifact, "live_lora_retired_recorded": "yes"})
    with pytest.raises(ValueError, match="gaps_updated"):
        exp4252.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="random_seed"):
        exp4252.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4252.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="model_specs"):
        exp4252.validate_artifact({**artifact, "model_specs": {}})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4252.validate_artifact({**artifact, "inference_substrate": "live"})
    with pytest.raises(ValueError, match="field_principles"):
        exp4252.validate_artifact({**artifact, "field_principles": {}})


def test_scenario_4252_run_hygiene_writes_artifact_ledgers_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4252: run writes the deliverable JSON and ledger entries."""

    _write_minimal_repo(tmp_path)
    artifact = exp4252.run_hygiene(tmp_path, adversarial_runner=_adversarial_clean)
    exp4252.validate_artifact(artifact)

    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4252.GAP_ORACLE_DISTINCT_GAP_ID,
        exp4252.GAP_ORACLE_DISTINCT_A3_GAP_ID,
        exp4252.GAP_CODE_REPLICATION_GAP_ID,
        exp4252.GAP_REWARD_GAP_ID,
        exp4252.GAP_LIVE_LORA_RETIREMENT_GAP_ID,
    ]
    assert artifact["registry_updated"] is True
    assert artifact["live_lora_retired_recorded"] is True
    assert artifact["adversarial_verify"]["methodology_missing_clean"] is True
    written = json.loads((tmp_path / exp4252.EXP4252_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    registry = yaml.safe_load(
        (tmp_path / "ops" / "verifier_registry.yaml").read_text(encoding="utf-8")
    )
    assert exp4252._registry_contains_outcomes(registry) is True
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert exp4252.GAP_ORACLE_DISTINCT_GAP_ID in gaps
    assert exp4252.GAP_ORACLE_DISTINCT_A3_GAP_ID in gaps
    assert exp4252.GAP_CODE_REPLICATION_GAP_ID in gaps
    assert exp4252.GAP_REWARD_GAP_ID in gaps
    assert exp4252.GAP_LIVE_LORA_RETIREMENT_GAP_ID in gaps
    manifest = yaml.safe_load(
        (tmp_path / "ops" / "exclusion_manifest.yaml").read_text(encoding="utf-8")
    )
    assert exp4252._find_live_lora_manifest_entry(manifest) is not None


def test_req_4252_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    """REQ-VERIFY-4252: failed preconditions write blocked_<resource> and no ledger win."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")
    (tmp_path / "ops" / "exclusion_manifest.yaml").write_text(
        yaml.safe_dump(_minimal_manifest(), sort_keys=False),
        encoding="utf-8",
    )

    artifact = exp4252.run_hygiene(tmp_path, adversarial_runner=_adversarial_clean)
    exp4252.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_gap4_arc1_candidate_fixtures"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_updated"] is False
    assert artifact["live_lora_retired_recorded"] is False
    assert artifact["gaps_updated"] == []
    assert artifact["random_seed"] == exp4252.RANDOM_SEED
    assert artifact["reproducibility_checksum"].startswith("blocked:")
    assert exp4252.GAP_ORACLE_DISTINCT_GAP_ID not in (
        tmp_path / "ops" / "verifier_gaps.md"
    ).read_text(encoding="utf-8")


def test_scenario_4252_results_entrypoint_runs_with_monkeypatched_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4252: required results entrypoint delegates to Exp 4252."""

    called: list[bool] = []
    monkeypatch.setattr(exp4252, "main", lambda: called.append(True))

    runpy.run_path(RESULTS_WRAPPER_PATH.as_posix(), run_name="__main__")

    assert called == [True]

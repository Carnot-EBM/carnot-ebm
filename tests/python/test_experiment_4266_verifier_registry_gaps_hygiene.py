"""Tests for Exp 4266 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4266, SCENARIO-VERIFY-4266.
"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4266_verifier_registry_gaps_hygiene as exp4266_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4266 as exp4266


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4266_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4266.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
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
    for path in exp4266.REQUIRED_UPSTREAM_PATHS + [
        exp4266.ARC1_POOL_PATH,
        exp4266.ARC1_PROGRAMS_PATH,
    ]:
        source = REPO_ROOT / path
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {
        "returncode": 0,
        "reports": [{"flags": [], "flag_count": 0, "max_severity": -1}],
    }


def test_req_4266_spec_declared() -> None:
    """REQ-VERIFY-4266: OpenSpec declares the .394 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4266",
        "SCENARIO-VERIFY-4266",
        "python/carnot/experiment_4266_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4266.EXP4266_ARTIFACT_PATH,
        "blocked_v394_artifacts_missing",
        "provenance_blind_delta=0.3846153846",
        "mean_delta=0.4576923077",
        "blocked_arc_game_ids_unrecoverable",
        "synthesis_minus_oracle_delta=-0.2826086957",
        "blocked_diffusiongemma_gguf_loader_failed",
        "ready_for_out_of_band=true",
        "replication_read=corpus_specific",
        "code_predictor_minus_vote_delta=-0.00625",
        "gaps_logged",
        exp4266.GAP_CROSS_GAME_ARC_SELECTION,
        exp4266.GAP_SUPRA_ORACLE_SYNTHESIS,
        exp4266.GAP_DIFFUSIONGEMMA_PREFLIGHT,
        exp4266.GAP_CODE_ORACLE_DISTINCT_ROBUSTNESS,
    ):
        assert marker in spec
    assert exp4266.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4266.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4266.FIELD_PRINCIPLES["gaps_logged"] in spec
    assert exp4266.FIELD_PRINCIPLES["reproducibility_checksum"] in spec
    assert exp4266_wrapper.main is exp4266.main


def test_scenario_4266_preconditions_outcomes_and_gap4_guard_are_stable() -> None:
    """SCENARIO-VERIFY-4266: .394 artifacts exist and GAP-4 does not regress."""

    preflight = exp4266.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None

    guard = exp4266.run_gap4_regression_guard(REPO_ROOT)
    assert guard["regression_guard_passed"] is True
    assert guard["recorded_arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert guard["replayed_arc1_rule_exec"] == guard["recorded_arc1_rule_exec"]

    outcomes = exp4266.load_v394_outcomes(REPO_ROOT)
    assert outcomes["a1_provenance_blind"]["win_survives_provenance_blind"] is True
    assert outcomes["a1_provenance_blind"]["provenance_blind_delta"] == pytest.approx(
        0.3846153846
    )
    assert outcomes["a2_multiseed"]["oracle_distinct_win_replicates"] is True
    assert outcomes["a2_multiseed"]["mean_delta"] == pytest.approx(0.4576923077)
    assert outcomes["a3_cross_game"]["honest_verdict"] == "blocked_arc_game_ids_unrecoverable"
    assert outcomes["a3_cross_game"]["cross_game_transfer_claim"] is False
    assert outcomes["a4_synthesis"]["synthesis_beats_selection"] is False
    assert outcomes["a4_synthesis"]["synthesis_breaks_oracle_ceiling"] is False
    assert outcomes["a4_synthesis"]["synthesis_minus_oracle_delta"] == pytest.approx(
        -0.2826086957
    )
    assert outcomes["b1_diffusiongemma"]["preflight_go"] is False
    assert (
        outcomes["b1_diffusiongemma"]["honest_verdict"]
        == "blocked_diffusiongemma_gguf_loader_failed"
    )
    assert outcomes["c1_verifier_reward"]["ready_for_out_of_band"] is True
    assert outcomes["c1_verifier_reward"]["verifier_as_reward_retired"] is False
    assert outcomes["c2_code_replication"]["replication_read"] == "corpus_specific"
    assert outcomes["c2_code_replication"]["code_replication_beats_vote"] is False
    assert outcomes["c2_code_replication"]["code_predictor_minus_vote_delta"] == pytest.approx(
        -0.00625
    )


def test_req_4266_ledgers_record_hardened_state_and_missing_verifier_gaps() -> None:
    """REQ-VERIFY-4266: registry and gaps carry the .394 truth."""

    guard = exp4266.run_gap4_regression_guard(REPO_ROOT)
    outcomes = exp4266.load_v394_outcomes(REPO_ROOT)
    gaps_logged = exp4266.build_gap_entries(outcomes)
    registry, gaps_text, summary = exp4266.ensure_ledgers_record_v394(
        _minimal_registry(),
        "# Verifier Gaps\n\nHistorical note remains.\n",
        guard,
        outcomes,
        gaps_logged,
    )

    assert [entry["gap_id"] for entry in gaps_logged] == [
        exp4266.GAP_CROSS_GAME_ARC_SELECTION,
        exp4266.GAP_SUPRA_ORACLE_SYNTHESIS,
        exp4266.GAP_DIFFUSIONGEMMA_PREFLIGHT,
        exp4266.GAP_CODE_ORACLE_DISTINCT_ROBUSTNESS,
    ]
    for entry in gaps_logged:
        assert set(entry) >= {
            "gap_id",
            "failure_mode",
            "missing_discriminator",
            "candidate_design",
            "priority",
        }
    assert summary == {
        "registry_reconciled": True,
        "gaps_logged_ids": [
            exp4266.GAP_CROSS_GAME_ARC_SELECTION,
            exp4266.GAP_SUPRA_ORACLE_SYNTHESIS,
            exp4266.GAP_DIFFUSIONGEMMA_PREFLIGHT,
            exp4266.GAP_CODE_ORACLE_DISTINCT_ROBUSTNESS,
        ],
    }

    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4266"] == exp4266.EXP4266_ARTIFACT_PATH
    assert gap4["eval"]["exp4266_regression_guard_passed"] is True
    assert gap4["eval"]["exp4266_v394_hardened_state"] == exp4266.V394_HARDENED_STATE
    assert gap4["eval"]["exp4266_a1_provenance_blind_survives"] is True
    assert gap4["eval"]["exp4266_a2_multiseed_replicates"] is True
    assert gap4["eval"]["exp4266_a3_cross_game_status"] == "blocked_arc_game_ids_unrecoverable"
    assert gap4["eval"]["exp4266_a4_synthesis_beats_selection"] is False
    assert gap4["eval"]["exp4266_b1_preflight_go"] is False
    assert gap4["eval"]["exp4266_c1_ready_for_out_of_band"] is True
    assert gap4["eval"]["exp4266_c2_replication_read"] == "corpus_specific"
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4266.V394_ROLE_ID)
    assert role["v394_hardened_state"] == exp4266.V394_HARDENED_STATE
    assert role["gap_ids_logged"] == summary["gaps_logged_ids"]
    assert exp4266.registry_contains_v394(registry) is True
    assert exp4266.registry_contains_v394({}) is False
    missing_registry: dict[str, Any] = {}
    exp4266._ensure_gap4_eval(missing_registry, guard, outcomes, gaps_logged)
    assert missing_registry["verifiers"][0]["verifier_id"] == exp4266.GAP4_VERIFIER_ID
    no_gap4_registry: dict[str, Any] = {}
    exp4266._ensure_v394_role(no_gap4_registry, outcomes, gaps_logged)
    assert no_gap4_registry == {}

    assert "Historical note remains." in gaps_text
    assert exp4266.GAP_CROSS_GAME_ARC_SELECTION in gaps_text
    assert "blocked_arc_game_ids_unrecoverable" in gaps_text
    assert "game/family provenance" in gaps_text
    assert exp4266.GAP_SUPRA_ORACLE_SYNTHESIS in gaps_text
    assert "synthesis_minus_oracle_delta=-0.2826086957" in gaps_text
    assert exp4266.GAP_DIFFUSIONGEMMA_PREFLIGHT in gaps_text
    assert "blocked_diffusiongemma_gguf_loader_failed" in gaps_text
    assert exp4266.GAP_CODE_ORACLE_DISTINCT_ROBUSTNESS in gaps_text
    assert "replication_read=corpus_specific" in gaps_text


def test_req_4266_build_artifact_validates_required_fields() -> None:
    """REQ-VERIFY-4266: terminal artifact exposes the required schema fields."""

    guard = exp4266.run_gap4_regression_guard(REPO_ROOT)
    outcomes = exp4266.load_v394_outcomes(REPO_ROOT)
    gaps_logged = exp4266.build_gap_entries(outcomes)
    artifact = exp4266.build_artifact(
        regression_guard=guard,
        v394_outcomes=outcomes,
        gaps_logged=gaps_logged,
        registry_reconciled=True,
        reproducibility_checksum="sha256:" + "0" * 64,
        duration_s=0.25,
    )

    exp4266.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciled"] is True
    assert artifact["gaps_logged"] == gaps_logged
    assert artifact["field_principles"] == exp4266.FIELD_PRINCIPLES
    assert artifact["model_specs"]["method"] == "cached_v394_ledger_reconciliation"
    assert artifact["inference_substrate"] == exp4266.INFERENCE_SUBSTRATE

    for field in exp4266.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4266.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4266.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4266.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="registry_reconciled"):
        exp4266.validate_artifact({**artifact, "registry_reconciled": "yes"})
    with pytest.raises(ValueError, match="gaps_logged"):
        exp4266.validate_artifact({**artifact, "gaps_logged": "gap"})
    with pytest.raises(ValueError, match="gap entry"):
        exp4266.validate_artifact({**artifact, "gaps_logged": [{"gap_id": "GAP"}]})
    with pytest.raises(ValueError, match="random_seed"):
        exp4266.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4266.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="model_specs"):
        exp4266.validate_artifact({**artifact, "model_specs": {}})
    with pytest.raises(ValueError, match="field_principles"):
        exp4266.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4266.validate_artifact({**artifact, "inference_substrate": "live"})


def test_scenario_4266_run_hygiene_writes_artifact_and_ledgers(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4266: run writes the deliverable JSON and ledger entries."""

    _write_minimal_repo(tmp_path)
    artifact = exp4266.run_hygiene(tmp_path, adversarial_runner=_adversarial_clean)
    exp4266.validate_artifact(artifact)

    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciled"] is True
    assert [gap["gap_id"] for gap in artifact["gaps_logged"]] == [
        exp4266.GAP_CROSS_GAME_ARC_SELECTION,
        exp4266.GAP_SUPRA_ORACLE_SYNTHESIS,
        exp4266.GAP_DIFFUSIONGEMMA_PREFLIGHT,
        exp4266.GAP_CODE_ORACLE_DISTINCT_ROBUSTNESS,
    ]
    assert artifact["adversarial_verify"]["methodology_missing_clean"] is True
    written = json.loads((tmp_path / exp4266.EXP4266_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written == artifact

    registry_path = tmp_path / "ops" / "verifier_registry.yaml"
    gaps_path = tmp_path / "ops" / "verifier_gaps.md"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert exp4266.registry_contains_v394(registry) is True
    gaps = gaps_path.read_text(encoding="utf-8")
    assert exp4266.GAP_CROSS_GAME_ARC_SELECTION in gaps
    assert exp4266.GAP_SUPRA_ORACLE_SYNTHESIS in gaps
    assert exp4266.GAP_DIFFUSIONGEMMA_PREFLIGHT in gaps
    assert exp4266.GAP_CODE_ORACLE_DISTINCT_ROBUSTNESS in gaps
    assert artifact["reproducibility_checksum"] == exp4266.ledger_checksum(
        registry_path,
        gaps_path,
    )


def test_req_4266_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    """REQ-VERIFY-4266: missing .394 artifacts produce blocked_v394_artifacts_missing."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")

    artifact = exp4266.run_hygiene(tmp_path, adversarial_runner=_adversarial_clean)
    exp4266.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_v394_artifacts_missing"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_reconciled"] is False
    assert artifact["gaps_logged"] == []
    assert artifact["reproducibility_checksum"] == "blocked:v394_artifacts_missing"
    assert exp4266.GAP_CROSS_GAME_ARC_SELECTION not in (
        tmp_path / "ops" / "verifier_gaps.md"
    ).read_text(encoding="utf-8")

    bad_registry = tmp_path / "bad_registry.yaml"
    bad_registry.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registry"):
        exp4266._load_registry_for_check(bad_registry)
    empty_gaps = tmp_path / "empty_gaps.md"
    empty_gaps.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gaps"):
        exp4266._load_gaps_for_check(empty_gaps)


def test_scenario_4266_results_entrypoint_runs_with_monkeypatched_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4266: required results entrypoint delegates to Exp 4266."""

    called: list[bool] = []
    monkeypatch.setattr(exp4266, "main", lambda: called.append(True))

    runpy.run_path(RESULTS_WRAPPER_PATH.as_posix(), run_name="__main__")

    assert called == [True]

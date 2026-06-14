"""Tests for Exp 4193 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4193, SCENARIO-VERIFY-4193.
"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4193_verifier_registry_gaps_hygiene as exp4193_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4193 as exp4193


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4193_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4193.GAP4_VERIFIER_ID,
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
        "experiment_4186_efficiency_moat_verifier_vs_llm_judge.json",
        "experiment_4187_gap4_graded_execution_gate_hardening.json",
        "experiment_4188_sovereign_local_generator_gap4_self_distill.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def test_req_4193_spec_declared() -> None:
    """REQ-VERIFY-4193: OpenSpec declares the .388 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4193",
        "SCENARIO-VERIFY-4193",
        "python/carnot/experiment_4193_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        "experiment_4193_verifier_registry_gaps_hygiene.json",
        "experiment_4186_efficiency_moat_verifier_vs_llm_judge.json",
        "experiment_4187_gap4_graded_execution_gate_hardening.json",
        "experiment_4188_sovereign_local_generator_gap4_self_distill.json",
        "verifier_efficiency_win=true",
        "graded_gate_pass2_vs_vote=0.129",
        "vote_aware_guard_blocked_mispromotion=true",
        "local_induction_rate.rate=0.2258",
        "sovereign_pool_pass2.LOCAL_HARDENED_GATE=0.4839",
        "self_distillation_corpus_size=7",
        "0.4516",
        "0.5806",
    ):
        assert marker in spec
    assert exp4193.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4193.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4193.FIELD_PRINCIPLES["gaps_updated"] in spec
    assert exp4193_wrapper.main is exp4193.main


def test_scenario_4193_preconditions_and_replay_are_bitexact() -> None:
    """SCENARIO-VERIFY-4193: cached GAP-4 replay reproduces exactly."""

    preflight = exp4193.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None
    assert {check["resource"] for check in preflight["checks"]} == {
        "gap4_arc1_candidate_fixtures",
        "verifier_registry",
        "verifier_gaps",
        "exp4186_efficiency_moat",
        "exp4187_gap4_graded_gate",
        "exp4188_sovereign_generator",
    }

    replay = exp4193.replay_gap4_arc1(REPO_ROOT)
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


def test_req_4193_classifies_efficiency_gate_and_sovereign() -> None:
    """REQ-VERIFY-4193: .388 outcomes are classified honestly."""

    efficiency = exp4193.classify_efficiency_moat(REPO_ROOT)
    assert efficiency["gap_id"] == exp4193.EFFICIENCY_MOAT_GAP_ID
    assert efficiency["status"] == "filled_verifier_efficiency_win"
    assert efficiency["verifier_efficiency_win"] is True
    assert efficiency["accuracy_parity_vs_judge"]["delta"] == pytest.approx(0.18)
    assert efficiency["accuracy_parity_vs_judge"]["ci95"] == [0.08, 0.3]
    assert efficiency["cost_ratio_vs_judge"]["ten_x_cheaper_on_both_axes"] is True
    assert efficiency["cost_ratio_vs_judge"]["strictly_pareto_dominant"] is True

    graded = exp4193.classify_gap4_graded_gate(REPO_ROOT)
    assert graded["gap_id"] == exp4193.GAP4_GRADED_GATE_GAP_ID
    assert graded["status"] == "filled_guarded_graded_gate_holds_plus4_minus0"
    assert graded["graded_gate_pass2_vs_vote"] == pytest.approx(0.129)
    assert graded["gross_recovery_ledger"] == {"lost": 0, "recovered": 4}
    assert graded["vote_aware_guard_blocked_mispromotion"] is True
    assert graded["pass2_vote_wins_lost"] == 0

    sovereign = exp4193.classify_sovereign_generator(REPO_ROOT)
    assert sovereign["gap_id"] == exp4193.SOVEREIGN_GENERATOR_GAP_ID
    assert sovereign["status"] == "building_sovereign_local_generator_positive_flagged"
    assert sovereign["local_induction_rate"]["rate"] == pytest.approx(0.2258)
    assert sovereign["sovereign_pool_pass2"]["LOCAL_HARDENED_GATE"] == pytest.approx(0.4839)
    assert sovereign["sovereign_pool_pass2"]["delta_vs_vote"] == pytest.approx(0.0323)
    assert sovereign["self_distillation_corpus_size"] == 7
    assert sovereign["flagged_adversarial"] is True


def test_scenario_4193_ensure_ledgers_record_outcomes_and_registry_role() -> None:
    """SCENARIO-VERIFY-4193: registry and gaps carry the .388 truth."""

    replay = exp4193.replay_gap4_arc1(REPO_ROOT)
    efficiency = exp4193.classify_efficiency_moat(REPO_ROOT)
    graded = exp4193.classify_gap4_graded_gate(REPO_ROOT)
    sovereign = exp4193.classify_sovereign_generator(REPO_ROOT)

    registry, gaps, summary = exp4193.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n",
        replay,
        efficiency,
        graded,
        sovereign,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": [
            exp4193.EFFICIENCY_MOAT_GAP_ID,
            exp4193.GAP4_GRADED_GATE_GAP_ID,
            exp4193.SOVEREIGN_GENERATOR_GAP_ID,
        ],
        "efficiency_moat_recorded": True,
        "gap4_graded_gate_recorded": True,
        "sovereign_generator_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4193"] == exp4193.EXP4193_ARTIFACT_PATH
    assert gap4["eval"]["exp4193_regression_guard_passed"] is True
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4193.V388_ROLE_ID)
    assert role["efficiency_moat_status"] == "filled_verifier_efficiency_win"
    assert role["verifier_efficiency_win"] is True
    assert role["gap4_graded_gate_status"] == "filled_guarded_graded_gate_holds_plus4_minus0"
    assert role["sovereign_generator_status"] == "building_sovereign_local_generator_positive_flagged"
    assert exp4193._registry_contains_outcomes(registry) is True
    assert exp4193._registry_contains_outcomes({}) is False

    assert exp4193.EFFICIENCY_MOAT_GAP_ID in gaps
    assert "verifier_efficiency_win=true" in gaps
    assert "accuracy_parity_vs_judge_delta=0.18" in gaps
    assert "ten_x_cheaper_on_both_axes=true" in gaps
    assert exp4193.GAP4_GRADED_GATE_GAP_ID in gaps
    assert "graded_gate_pass2_vs_vote=0.129" in gaps
    assert "vote_aware_guard_blocked_mispromotion=true" in gaps
    assert exp4193.SOVEREIGN_GENERATOR_GAP_ID in gaps
    assert "local_induction_rate=0.2258" in gaps
    assert "self_distillation_corpus_size=7" in gaps


def test_req_4193_build_artifact_validates_required_fields() -> None:
    """REQ-VERIFY-4193: terminal artifact exposes required schema fields."""

    replay = exp4193.replay_gap4_arc1(REPO_ROOT)
    efficiency = exp4193.classify_efficiency_moat(REPO_ROOT)
    graded = exp4193.classify_gap4_graded_gate(REPO_ROOT)
    sovereign = exp4193.classify_sovereign_generator(REPO_ROOT)
    artifact = exp4193.build_artifact(
        offline_replay=replay,
        efficiency_moat=efficiency,
        gap4_graded_gate=graded,
        sovereign_generator=sovereign,
        registry_updated=True,
        gaps_updated=[
            exp4193.EFFICIENCY_MOAT_GAP_ID,
            exp4193.GAP4_GRADED_GATE_GAP_ID,
            exp4193.SOVEREIGN_GENERATOR_GAP_ID,
        ],
        duration_s=0.012,
    )

    exp4193.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4193.EFFICIENCY_MOAT_GAP_ID,
        exp4193.GAP4_GRADED_GATE_GAP_ID,
        exp4193.SOVEREIGN_GENERATOR_GAP_ID,
    ]
    assert artifact["field_principles"] == exp4193.FIELD_PRINCIPLES
    assert artifact["efficiency_moat"]["verifier_efficiency_win"] is True
    assert artifact["gap4_graded_gate"]["vote_aware_guard_blocked_mispromotion"] is True
    assert artifact["sovereign_generator"]["self_distillation_corpus_size"] == 7
    assert artifact["cited_upstream_artifacts"] == [
        exp4193.ARC1_POOL_PATH,
        exp4193.ARC1_PROGRAMS_PATH,
        exp4193.EXP4186_PATH,
        exp4193.EXP4187_PATH,
        exp4193.EXP4188_PATH,
    ]

    for field in exp4193.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4193.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4193.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4193.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4193.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="gaps_updated"):
        exp4193.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="field_principles"):
        exp4193.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4193.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_4193_helper_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4193: schema helpers fail closed without hidden inference."""

    assert exp4193._numeric_or_none(True) is None
    assert exp4193._numeric_or_none("bad") is None
    assert exp4193._round4(None) is None
    assert exp4193._round4(1.23456) == pytest.approx(1.2346)
    assert exp4193._check_json_resource(tmp_path, "missing", "missing.json") == {
        "resource": "missing",
        "available": False,
        "detail": "missing: missing.json",
    }
    assert exp4193._efficiency_moat_status({"verifier_efficiency_win": False}) == (
        "open_efficiency_moat_not_filled"
    )
    assert exp4193._gap4_graded_status({"pass2_vote_wins_lost": 1}) == (
        "open_graded_gate_regression"
    )
    assert (
        exp4193._sovereign_generator_status(
            {
                "flagged_adversarial": False,
                "self_distillation_corpus_size": 1,
                "sovereign_pool_pass2": {
                    "LOCAL_HARDENED_GATE": 0.5,
                    "TRM_VOTE": 0.4,
                },
            }
        )
        == "building_sovereign_local_generator_positive"
    )
    assert exp4193._sovereign_generator_status({"sovereign_pool_pass2": {}}) == (
        "open_sovereign_local_generator_not_established"
    )


def test_scenario_4193_run_hygiene_writes_artifact_and_ledgers(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4193: run writes the deliverable JSON and ledger entries."""

    _write_minimal_repo(tmp_path)
    artifact = exp4193.run_hygiene(tmp_path)
    exp4193.validate_artifact(artifact)

    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4193.EFFICIENCY_MOAT_GAP_ID,
        exp4193.GAP4_GRADED_GATE_GAP_ID,
        exp4193.SOVEREIGN_GENERATOR_GAP_ID,
    ]
    assert artifact["registry_updated"] is True
    written = json.loads((tmp_path / exp4193.EXP4193_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text(encoding="utf-8"))
    assert exp4193._registry_contains_outcomes(registry) is True
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert exp4193.EFFICIENCY_MOAT_GAP_ID in gaps
    assert exp4193.GAP4_GRADED_GATE_GAP_ID in gaps
    assert exp4193.SOVEREIGN_GENERATOR_GAP_ID in gaps


def test_req_4193_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    """REQ-VERIFY-4193: failed preconditions write blocked_<resource> and no ledger win."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")

    artifact = exp4193.run_hygiene(tmp_path)
    exp4193.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_gap4_arc1_candidate_fixtures"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_updated"] is False
    assert artifact["gaps_updated"] == []
    assert exp4193.EFFICIENCY_MOAT_GAP_ID not in (
        tmp_path / "ops" / "verifier_gaps.md"
    ).read_text(encoding="utf-8")


def test_scenario_4193_results_entrypoint_runs_with_monkeypatched_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4193: required results entrypoint delegates to Exp 4193."""

    called: list[bool] = []
    monkeypatch.setattr(exp4193, "main", lambda: called.append(True))

    runpy.run_path(RESULTS_WRAPPER_PATH.as_posix(), run_name="__main__")

    assert called == [True]

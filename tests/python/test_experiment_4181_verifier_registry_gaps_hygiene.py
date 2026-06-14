"""Tests for Exp 4181 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4181, SCENARIO-VERIFY-4181.
"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4181_verifier_registry_gaps_hygiene as exp4181_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4181 as exp4181


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4181_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4181.GAP4_VERIFIER_ID,
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
        "experiment_4175_headroom_gate_executable_census.json",
        "experiment_4177_decisive_headroom_controlled_moat_test.json",
        "experiment_4178_gap3_stage1_model_native_arc_energy.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def test_req_4181_spec_declared() -> None:
    """REQ-VERIFY-4181: OpenSpec declares the .387 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4181",
        "SCENARIO-VERIFY-4181",
        "python/carnot/experiment_4181_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        "experiment_4181_verifier_registry_gaps_hygiene.json",
        "experiment_4177_decisive_headroom_controlled_moat_test.json",
        "experiment_4178_gap3_stage1_model_native_arc_energy.json",
        "headroom_present_domain=code",
        "verifier_value_added=true",
        "pass2_energy_vs_vote=0.0",
        "does not advance toward `filled`",
        "0.4516",
        "0.5806",
    ):
        assert marker in spec
    assert exp4181.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4181.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4181.FIELD_PRINCIPLES["gaps_updated"] in spec
    assert exp4181_wrapper.main is exp4181.main


def test_scenario_4181_preconditions_and_replay_are_bitexact() -> None:
    """SCENARIO-VERIFY-4181: cached GAP-4 replay reproduces exactly."""

    preflight = exp4181.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None
    assert {check["resource"] for check in preflight["checks"]} == {
        "gap4_arc1_candidate_fixtures",
        "verifier_registry",
        "verifier_gaps",
        "exp4175_headroom_census",
        "exp4177_headroom_controlled_moat",
        "exp4178_gap3_stage1",
    }

    replay = exp4181.replay_gap4_arc1(REPO_ROOT)
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


def test_req_4181_classifies_moat_and_gap3_stage1() -> None:
    """REQ-VERIFY-4181: .387 moat and GAP-3 outcomes are classified honestly."""

    moat = exp4181.classify_moat_verdict(REPO_ROOT)
    assert moat["gap_id"] == exp4181.MOAT_GAP_ID
    assert moat["status"] == "filled_headroom_controlled_verifier_value_added"
    assert moat["verifier_value_added"] is True
    assert moat["headroom_present_domain"] == "code"
    assert moat["positive_control_confirmed"] is True
    assert moat["moat_delta_vs_vote"]["delta"] == pytest.approx(0.18)
    assert moat["moat_delta_vs_vote"]["ci95"] == [0.08, 0.3]
    assert moat["moat_vs_matched_control"]["delta"] == pytest.approx(0.18)

    gap3 = exp4181.classify_gap3_stage1_result(REPO_ROOT)
    assert gap3["gap_id"] == exp4181.GAP3_STAGE1_GAP_ID
    assert gap3["status"] == "open_stage1_honest_negative_does_not_advance"
    assert gap3["advances_toward_filled"] is False
    assert gap3["pass2_energy_vs_vote"] == pytest.approx(0.0)
    assert gap3["headroom_capture_fraction"] == pytest.approx(0.0)
    assert gap3["all_four_gates_pass"] is False
    assert gap3["candidate_auroc"] == pytest.approx(0.893651)
    assert gap3["coverage_fraction"] == pytest.approx(1.0)


def test_scenario_4181_ensure_ledgers_record_moat_gap3_and_registry_role() -> None:
    """SCENARIO-VERIFY-4181: registry and gaps carry the .387 truth."""

    replay = exp4181.replay_gap4_arc1(REPO_ROOT)
    moat = exp4181.classify_moat_verdict(REPO_ROOT)
    gap3 = exp4181.classify_gap3_stage1_result(REPO_ROOT)

    registry, gaps, summary = exp4181.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n",
        replay,
        moat,
        gap3,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": [exp4181.MOAT_GAP_ID, exp4181.GAP3_STAGE1_GAP_ID],
        "moat_recorded": True,
        "gap3_stage1_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4181"] == exp4181.EXP4181_ARTIFACT_PATH
    assert gap4["eval"]["exp4181_regression_guard_passed"] is True
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4181.V387_ROLE_ID)
    assert role["moat_verifier_value_added"] is True
    assert role["headroom_present_domain"] == "code"
    assert role["gap3_stage1_advances_toward_filled"] is False
    assert exp4181._registry_contains_outcomes(registry) is True
    assert exp4181._registry_contains_outcomes({}) is False

    assert exp4181.MOAT_GAP_ID in gaps
    assert "verifier_value_added=true" in gaps
    assert "headroom_present_domain=code" in gaps
    assert "moat_delta_vs_vote_delta=0.18" in gaps
    assert exp4181.GAP3_STAGE1_GAP_ID in gaps
    assert "pass2_energy_vs_vote=0.0" in gaps
    assert "advances_toward_filled=false" in gaps
    assert "all_four_gates_pass=false" in gaps


def test_req_4181_build_artifact_validates_required_fields() -> None:
    """REQ-VERIFY-4181: terminal artifact exposes required schema fields."""

    replay = exp4181.replay_gap4_arc1(REPO_ROOT)
    moat = exp4181.classify_moat_verdict(REPO_ROOT)
    gap3 = exp4181.classify_gap3_stage1_result(REPO_ROOT)
    artifact = exp4181.build_artifact(
        offline_replay=replay,
        moat_verdict=moat,
        gap3_stage1_result=gap3,
        registry_updated=True,
        gaps_updated=[exp4181.MOAT_GAP_ID, exp4181.GAP3_STAGE1_GAP_ID],
        duration_s=0.012,
    )

    exp4181.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [exp4181.MOAT_GAP_ID, exp4181.GAP3_STAGE1_GAP_ID]
    assert artifact["field_principles"] == exp4181.FIELD_PRINCIPLES
    assert artifact["moat_verdict"]["verifier_value_added"] is True
    assert artifact["gap3_stage1_result"]["advances_toward_filled"] is False
    assert artifact["cited_upstream_artifacts"] == [
        exp4181.ARC1_POOL_PATH,
        exp4181.ARC1_PROGRAMS_PATH,
        exp4181.EXP4175_PATH,
        exp4181.EXP4177_PATH,
        exp4181.EXP4178_PATH,
    ]

    for field in exp4181.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4181.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4181.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4181.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4181.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="gaps_updated"):
        exp4181.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="field_principles"):
        exp4181.validate_artifact({**artifact, "field_principles": {}})


def test_req_4181_helper_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4181: schema helpers fail closed without hidden inference."""

    assert exp4181._numeric_or_none(True) is None
    assert exp4181._numeric_or_none("bad") is None
    assert exp4181._round4(None) is None
    assert exp4181._check_json_resource(tmp_path, "missing", "missing.json") == {
        "resource": "missing",
        "available": False,
        "detail": "missing: missing.json",
    }
    assert (
        exp4181._moat_status({"verifier_value_added": False})
        == "open_headroom_controlled_no_value_added"
    )
    assert (
        exp4181._gap3_stage1_status({"advances_toward_filled": True})
        == "building_stage1_advances_toward_filled"
    )


def test_scenario_4181_run_hygiene_writes_artifact_and_ledgers(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4181: run writes the deliverable JSON and ledger entries."""

    _write_minimal_repo(tmp_path)
    artifact = exp4181.run_hygiene(tmp_path)
    exp4181.validate_artifact(artifact)

    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [exp4181.MOAT_GAP_ID, exp4181.GAP3_STAGE1_GAP_ID]
    assert artifact["registry_updated"] is True
    written = json.loads((tmp_path / exp4181.EXP4181_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text(encoding="utf-8"))
    assert exp4181._registry_contains_outcomes(registry) is True
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert exp4181.MOAT_GAP_ID in gaps
    assert exp4181.GAP3_STAGE1_GAP_ID in gaps


def test_scenario_4181_results_entrypoint_runs_with_monkeypatched_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-4181: required results entrypoint delegates to Exp 4181."""

    called: list[bool] = []
    monkeypatch.setattr(exp4181, "main", lambda: called.append(True))

    runpy.run_path(RESULTS_WRAPPER_PATH.as_posix(), run_name="__main__")

    assert called == [True]

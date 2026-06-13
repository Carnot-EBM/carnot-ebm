"""Tests for Exp 4122 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4122, SCENARIO-VERIFY-4122.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import verifier_registry_gaps_hygiene_4122 as exp4122


REPO_ROOT = Path(__file__).parents[2]


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": "gap4_program_induction_stack",
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "code_path": "python/carnot/agentic/gap4_program_induction_stack.py",
                "eval": {"metric": "pass_at_1"},
                "status": "candidate",
                "training_time_roles": [
                    {
                        "role_id": "sudoku_executable_verifier_training_time_4109",
                        "experiment": "results/experiment_4109_carnot_verifier_graft_sudoku.json",
                        "role": "candidate_trm_training_time_reward_signal_executable_domain",
                        "status": "honest_null_no_value_added",
                    }
                ],
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
        "experiment_4116_sudoku_extreme_resume_pass1.json",
        "experiment_4117_sudoku_extreme_resume_pass2.json",
        "experiment_4118_sudoku_extreme_resume_pass3.json",
        "experiment_4119_carnot_verifier_graft_sudoku.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def test_req_4122_spec_declared() -> None:
    # REQ-VERIFY-4122: OpenSpec declares the runner, inputs, fields, and principles.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4122",
        "SCENARIO-VERIFY-4122",
        "exp4122_verifier_registry_gaps_hygiene.py",
        "experiment_4122_verifier_registry_gaps_hygiene.json",
        "0.0854 -> 0.0966 -> 0.1060",
        "graft_deferred=true",
        "verifier_value_added=false",
        "regression_guard_passed",
        "gaps_updated",
    ):
        assert marker in spec
    assert exp4122.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4122.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4122.FIELD_PRINCIPLES["gaps_updated"] in spec


def test_scenario_4122_replay_gap4_arc1_regression_guard_is_bitexact() -> None:
    # SCENARIO-VERIFY-4122: cached candidates reproduce vote 0.4516 -> gated 0.5806.
    replay = exp4122.replay_gap4_arc1(REPO_ROOT)
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


def test_req_4122_classifies_sudoku_baseline_and_graft_deferral() -> None:
    # REQ-VERIFY-4122: the .381 Sudoku baseline remains unreproduced, so graft is deferred.
    baseline = exp4122.classify_sudoku_baseline(REPO_ROOT)
    assert baseline["gap_id"] == exp4122.SUDOKU_BASELINE_GAP_ID
    assert baseline["status"] == "open_baseline_not_reproduced_val_0.1060"
    assert baseline["matches_published_087"] is False
    assert baseline["published_target_val_exact_accuracy"] == pytest.approx(0.87)
    assert baseline["final_val_exact_accuracy"] == pytest.approx(0.10598958283662796)
    assert baseline["val_trajectory_rounded"] == [0.0854, 0.0966, 0.106]
    assert baseline["total_cumulative_epochs"] == 4300
    assert baseline["passes"][0]["pass_id"] == "pass1"
    assert baseline["passes"][0]["artifact_flagged_adversarial"] is True
    assert baseline["passes"][1]["artifact_path"] == exp4122.EXP4117_PATH
    assert baseline["passes"][2]["artifact_path"] == exp4122.EXP4118_PATH

    graft = exp4122.classify_sudoku_graft(REPO_ROOT)
    assert graft["gap_id"] == exp4122.SUDOKU_GRAFT_GAP_ID
    assert graft["status"] == "open_graft_deferred_verifier_value_added_false"
    assert graft["graft_deferred"] is True
    assert graft["verifier_value_added"] is False
    assert graft["flagged_adversarial"] is True
    assert graft["rerank_lift_vs_vote"] is None
    assert graft["rft_vs_ablation_delta"] is None


def test_scenario_4122_ensure_ledgers_record_baseline_graft_and_role() -> None:
    # SCENARIO-VERIFY-4122: registry and gaps carry .381 Sudoku baseline and graft status.
    replay = exp4122.replay_gap4_arc1(REPO_ROOT)
    baseline = exp4122.classify_sudoku_baseline(REPO_ROOT)
    graft = exp4122.classify_sudoku_graft(REPO_ROOT)

    registry, gaps, summary = exp4122.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n",
        replay,
        baseline,
        graft,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": [exp4122.SUDOKU_BASELINE_GAP_ID, exp4122.SUDOKU_GRAFT_GAP_ID],
        "sudoku_baseline_recorded": True,
        "sudoku_graft_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4122"] == exp4122.EXP4122_ARTIFACT_PATH
    assert gap4["eval"]["exp4122_regression_guard_passed"] is True
    role = next(
        role
        for role in gap4["training_time_roles"]
        if role["role_id"] == exp4122.SUDOKU_TRAINING_ROLE_ID
    )
    assert role["status"] == "graft_deferred_baseline_not_reproduced"
    assert role["graft_deferred"] is True
    assert role["verifier_value_added"] is False
    assert role["baseline_val_trajectory_rounded"] == [0.0854, 0.0966, 0.106]
    assert exp4122._registry_contains_outcomes(registry) is True
    assert exp4122._registry_contains_outcomes({}) is False
    assert exp4122.SUDOKU_BASELINE_GAP_ID in gaps
    assert "val_trajectory=[0.0854, 0.0966, 0.106]" in gaps
    assert "matches_published_087=false" in gaps
    assert exp4122.SUDOKU_GRAFT_GAP_ID in gaps
    assert "graft_deferred=true" in gaps
    assert "verifier_value_added=false" in gaps


def test_req_4122_build_artifact_validates_required_fields() -> None:
    # REQ-VERIFY-4122: terminal artifact exposes required schema fields and principles.
    replay = exp4122.replay_gap4_arc1(REPO_ROOT)
    baseline = exp4122.classify_sudoku_baseline(REPO_ROOT)
    graft = exp4122.classify_sudoku_graft(REPO_ROOT)
    artifact = exp4122.build_artifact(
        offline_replay=replay,
        sudoku_baseline=baseline,
        sudoku_graft=graft,
        registry_updated=True,
        gaps_updated=[exp4122.SUDOKU_BASELINE_GAP_ID, exp4122.SUDOKU_GRAFT_GAP_ID],
        duration_s=0.012,
    )

    exp4122.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [exp4122.SUDOKU_BASELINE_GAP_ID, exp4122.SUDOKU_GRAFT_GAP_ID]
    assert artifact["field_principles"] == exp4122.FIELD_PRINCIPLES
    assert artifact["sudoku_baseline"]["val_trajectory_rounded"] == [0.0854, 0.0966, 0.106]
    assert artifact["sudoku_graft"]["graft_deferred"] is True
    assert artifact["sudoku_graft"]["verifier_value_added"] is False

    for field in exp4122.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4122.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4122.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4122.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4122.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="list"):
        exp4122.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4122.validate_artifact({**artifact, "inference_substrate": "live"})


def test_scenario_4122_run_hygiene_writes_terminal_artifact_and_ledgers(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4122: run writes the deliverable JSON and stable ledger entries.
    _write_minimal_repo(tmp_path)
    artifact = exp4122.run_hygiene(tmp_path)
    exp4122.validate_artifact(artifact)

    out_path = tmp_path / exp4122.EXP4122_ARTIFACT_PATH
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["regression_guard_passed"] is True
    assert written["sudoku_baseline"]["status"] == "open_baseline_not_reproduced_val_0.1060"
    assert written["sudoku_graft"]["graft_deferred"] is True
    assert written["sudoku_graft"]["verifier_value_added"] is False

    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text())
    role_ids = [role["role_id"] for role in registry["verifiers"][0]["training_time_roles"]]
    assert exp4122.SUDOKU_TRAINING_ROLE_ID in role_ids
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert "Exp 4122 .381 Sudoku baseline reproduction status" in gaps
    assert "Exp 4122 .381 Sudoku executable-verifier graft status" in gaps


def test_req_4122_defensive_branches_cover_malformed_inputs(tmp_path: Path) -> None:
    # REQ-VERIFY-4122: malformed optional upstream fields do not fabricate a graft win.
    assert exp4122._numeric_or_none(True) is None
    assert exp4122._numeric_or_none("0.1") is None
    assert exp4122._round4(None) is None

    results = tmp_path / "results"
    results.mkdir()
    (results / Path(exp4122.EXP4116_PATH).name).write_text(
        json.dumps(
            {
                "honest_verdict": "complete: malformed-pass1-probe",
                "flagged_adversarial": True,
                "acceptance_gate_passed": False,
            }
        ),
        encoding="utf-8",
    )
    (results / Path(exp4122.EXP4117_PATH).name).write_text(
        json.dumps(
            {
                "honest_verdict": "complete: pass2-fallback",
                "pass1": "not-a-dict",
                "val_exact_accuracy": 0.2,
                "acceptance_gate_passed": True,
            }
        ),
        encoding="utf-8",
    )
    (results / Path(exp4122.EXP4118_PATH).name).write_text(
        json.dumps(
            {
                "honest_verdict": "complete: reproduced-probe",
                "pass2": ["not-a-dict"],
                "val_exact_accuracy": 0.9,
                "matches_published_087": True,
                "acceptance_gate_passed": True,
            }
        ),
        encoding="utf-8",
    )
    baseline = exp4122.classify_sudoku_baseline(tmp_path)
    assert baseline["status"] == "reproduced_val_0.9000"
    assert baseline["val_trajectory_rounded"] == [0.2, 0.9]
    assert baseline["passes"][0]["val_exact_accuracy"] is None

    graft_path = results / Path(exp4122.EXP4119_PATH).name
    graft_path.write_text(
        json.dumps(
            {
                "honest_verdict": "success: graft_value_added_probe",
                "graft_deferred": False,
                "verifier_value_added": True,
            }
        ),
        encoding="utf-8",
    )
    assert exp4122.classify_sudoku_graft(tmp_path)["status"] == "filled_verifier_value_added"
    graft_path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: graft_null_probe",
                "graft_deferred": False,
                "verifier_value_added": False,
            }
        ),
        encoding="utf-8",
    )
    assert exp4122.classify_sudoku_graft(tmp_path)["status"] == "open_honest_null_no_value_added"

    replay = {"regression_guard_passed": True, "arc1_rule_exec": {"vote_pass2": 0.4516}}
    repaired_registry: dict[str, Any] = {}
    exp4122._ensure_gap4_eval(repaired_registry, replay)
    assert repaired_registry["verifiers"][0]["verifier_id"] == exp4122.GAP4_VERIFIER_ID
    exp4122._ensure_sudoku_training_role({}, baseline, {"graft_deferred": True})

    value_registry = _minimal_registry()
    exp4122._ensure_sudoku_training_role(
        value_registry,
        baseline,
        {"graft_deferred": False, "verifier_value_added": True, "status": "filled"},
    )
    value_role = value_registry["verifiers"][0]["training_time_roles"][-1]
    assert value_role["status"] == "value_added"

    null_registry = _minimal_registry()
    exp4122._ensure_sudoku_training_role(
        null_registry,
        baseline,
        {"graft_deferred": False, "verifier_value_added": False, "status": "open"},
    )
    null_role = null_registry["verifiers"][0]["training_time_roles"][-1]
    assert null_role["status"] == "honest_null_no_value_added"

    artifact = exp4122.build_artifact(
        offline_replay=exp4122.replay_gap4_arc1(REPO_ROOT),
        sudoku_baseline=exp4122.classify_sudoku_baseline(REPO_ROOT),
        sudoku_graft=exp4122.classify_sudoku_graft(REPO_ROOT),
        registry_updated=True,
        gaps_updated=[exp4122.SUDOKU_BASELINE_GAP_ID, exp4122.SUDOKU_GRAFT_GAP_ID],
        duration_s=0.012,
    )
    with pytest.raises(ValueError, match="field_principles"):
        exp4122.validate_artifact({**artifact, "field_principles": {}})

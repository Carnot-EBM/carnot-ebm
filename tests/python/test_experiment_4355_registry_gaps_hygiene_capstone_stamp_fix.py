"""Tests for Exp 4355 registry/gaps hygiene and capstone stamp fix.

Spec refs: REQ-VERIFY-4355, SCENARIO-VERIFY-4355.
"""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4355_registry_gaps_hygiene_capstone_stamp_fix as wrapper
from carnot.reporting import capstone_v401_4346
from carnot.reporting import (
    verifier_registry_gaps_hygiene_capstone_stamp_fix_4355 as exp4355,
)


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
RESULTS_WRAPPER_PATH = Path(
    "results/experiment_4355_registry_gaps_hygiene_capstone_stamp_fix.py"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4355.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "eval": {"metric": "pass_at_2"},
                "registry_roles": [],
            }
        ]
    }


def _minimal_arc_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated": "2026-06-17",
        "games": [
            {"game": "ka59", "reproducibility": "unsolved", "levels_reproduced": 0},
            {"game": "tn36", "reproducibility": "reproduced", "levels_reproduced": 6},
        ],
        "reproducible_total_levels": 21,
        "reproducible_total_games": 13,
        "provisional_total_levels": 5,
    }


def _write_minimal_repo(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "adversarial_verify.py").write_text("# fixture\n", encoding="utf-8")
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(_minimal_arc_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text(
        "# Verifier Gaps\n\n"
        "<!-- exp4333-gap-e3-world-model-rule-ka59-4328:start -->\n"
        "### GAP-E3-WORLD-MODEL-RULE-KA59-4328: prior gap\n"
        "- status: open\n"
        "- evidence: old ka59 partial.\n"
        "- failure mode: old ka59 model partial\n"
        "- missing discriminator: old ka59 rules\n"
        "- candidate design: old design\n"
        "- priority: high\n"
        "<!-- exp4333-gap-e3-world-model-rule-ka59-4328:end -->\n",
        encoding="utf-8",
    )
    for rel_path, payload in _minimal_v402_payloads().items():
        _write_json(tmp_path / rel_path, payload)


def _minimal_v402_payloads() -> dict[str, dict[str, Any]]:
    return {
        exp4355.EXP4348_PATH: {
            "honest_verdict": "controls_not_differentiable",
            "acceptance_gate": True,
            "s3_guided_beats_control": False,
            "controls_differentiated": False,
            "s3_gain_ci95": [0.208333, 0.329167],
            "benchmark_n": 240,
            "flagged_adversarial": True,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "6f6e37ce8128fe9e77c5ad4623c1d6dfc07b4a0b9d323318eed8734d4d16c938",
        },
        exp4355.EXP4350_PATH: {
            "honest_verdict": "success_e3_ka59_L1_reproduced",
            "game": "ka59",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "residual_mismatch_class": "hidden_step_counter_hud_gap",
            "verifier_best_accuracy": 0.6375,
            "world_model_path": "results/arc_e3/ka59/world_model.py",
            "world_model_sha256": "ec4c777b8f416d8522ba36b3fbc2156f918dd03229a840cadca9438de45c8cba",
            "verifier_is_oracle": True,
            "reproducibility_checksum": "3d282a1804ca818ee28d4e6b34d4dd8d76fb4258c013a6f630926a8df675fa06",
        },
        exp4355.EXP4351_PATH: {
            "honest_verdict": "success_e3_deeper_tn36_reproduced",
            "new_levels_reproduced": 1,
            "reproducible_total_levels": 23,
            "verifier_is_oracle": True,
            "per_target_scorecard": [
                {
                    "game": "sc25",
                    "offline_reproduced": False,
                    "new_reproduced_level": 1,
                    "residual_win_mechanic_gap_class": "sc25_l2_live_recorded_not_offline_reproduced_spell_delta_gap",
                    "world_model_path": "results/arc_e3/sc25/world_model.py",
                },
                {
                    "game": "tn36",
                    "offline_reproduced": True,
                    "new_reproduced_level": 7,
                    "residual_win_mechanic_gap_class": "none",
                    "world_model_path": "scripts/arc3_tn36_offline_solver.py",
                },
                {
                    "game": "ar25",
                    "offline_reproduced": False,
                    "new_reproduced_level": 1,
                    "residual_win_mechanic_gap_class": "ar25_l2_hidden_rule_delta_not_reproduced_action7_undo_stack_gap",
                    "world_model_path": "results/arc_e3/ar25/world_model.py",
                },
            ],
            "reproducibility_checksum": "72ad0cdfb5c23845b4272b9b98c30c4b0fa4a27b47980107fc76862aa45f45c9",
        },
        exp4355.EXP4352_PATH: {
            "honest_verdict": "complete_e3_tr87_ft09_partial",
            "new_levels_reproduced": 0,
            "verifier_is_oracle": True,
            "per_game_scorecard": [
                {
                    "game": "tr87",
                    "offline_reproduced": False,
                    "reproduced_levels": 0,
                    "residual_mismatch_class": "missing_world_model_rule_gap_actions_1_2_3_4",
                    "verifier_accuracy": 0.0,
                    "world_model_path": "results/arc_e3/tr87/world_model.py",
                },
                {
                    "game": "ft09",
                    "offline_reproduced": False,
                    "reproduced_levels": 0,
                    "residual_mismatch_class": "missing_world_model_rule_gap_actions_6",
                    "verifier_accuracy": 0.05,
                    "world_model_path": "results/arc_e3/ft09/world_model.py",
                },
            ],
            "reproducibility_checksum": "88ebeb795a2569c299b14d073f12934f23c2738279d6c561623f8af4293b7f28",
        },
        exp4355.EXP4353_PATH: {
            "honest_verdict": "success: learned_action_cost_reduces_actions_25_to_16",
            "acceptance_gate_passed": True,
            "action_efficiency_improves": True,
            "held_out_actions_baseline": 25,
            "held_out_actions_learned": 16,
            "positive_control_passed": True,
            "reproduction_gated": True,
            "missing_verifier_gaps": [],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:bb2239bae156eb0fae288a54a8d20855e79543ba07db250b0d47e61173c731fa",
        },
    }


def _guard() -> dict[str, Any]:
    return {
        "regression_guard_passed": True,
        "replayed_arc1_rule_exec": {
            "n": 31,
            "vote_pass2": 0.4516,
            "gated_pass2": 0.5806,
            "headroom_recovered": 4,
            "vote_wins_lost": 0,
        },
    }


def _stamp_report() -> dict[str, Any]:
    return {
        "capstone_stamp_fix_verified": True,
        "sample_verifier_is_oracle": False,
        "circular_moat_overclaim_fired": False,
        "returncode": 0,
        "flags": [],
    }


def test_req_verify_4355_spec_declared() -> None:
    """REQ-VERIFY-4355: OpenSpec declares the .402 hygiene/stamp contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4355",
        "SCENARIO-VERIFY-4355",
        "python/carnot/experiment_4355_registry_gaps_hygiene_capstone_stamp_fix.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4355.EXP4355_ARTIFACT_PATH,
        "blocked_<file>_unreadable",
        "capstone aggregation code SHALL propagate",
        "CIRCULAR_MOAT_OVERCLAIM",
        "held_out_actions_baseline=25",
        "held_out_actions_learned=16",
    ):
        assert marker in spec
    for field in exp4355.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp4355.FIELD_PRINCIPLES.values():
        assert principle in spec
    assert wrapper.main is exp4355.main


def test_scenario_verify_4355_ledgers_record_v402_truth(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4355: .402 outcomes are recorded without pruning history."""

    _write_minimal_repo(tmp_path)
    preflight = exp4355.check_preconditions(tmp_path)
    assert preflight["ok"] is True

    outcomes = exp4355.load_v402_outcomes(tmp_path)
    gap_entries = exp4355.build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = exp4355.ensure_ledgers_record_v402(
        _minimal_registry(),
        (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8"),
        _minimal_arc_registry(),
        _guard(),
        outcomes,
        gap_entries,
    )

    assert exp4355.registry_contains_v402(registry) is True
    assert exp4355.arc_registry_contains_v402(arc_registry) is True
    assert summary["registries_reconciled"] is True
    assert summary["filled_gap_ids"] == [exp4355.GAP_E3_WORLD_MODEL_RULE_KA59_4328]
    assert exp4355.ACTION_COST_VERIFIER_ID in [
        row["verifier_id"] for row in registry["verifiers"]
    ]
    assert "status: filled (exp4350_ka59_l1_world_model)" in gaps_text
    for gap_id in [
        exp4355.GAP_E3_WORLD_MODEL_RULE_KA59_4350,
        exp4355.GAP_E3_WORLD_MODEL_RULE_SC25_L2_4351,
        exp4355.GAP_E3_WORLD_MODEL_RULE_AR25_L2_4351,
        exp4355.GAP_E3_WORLD_MODEL_RULE_TR87_4352,
        exp4355.GAP_E3_WORLD_MODEL_RULE_FT09_4352,
    ]:
        assert gap_id in gaps_text


def test_req_verify_4355_run_hygiene_writes_required_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-4355: terminal artifact exposes the required bare fields."""

    _write_minimal_repo(tmp_path)
    artifact = exp4355.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    exp4355.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gap4_regression_guard_passed"] is True
    assert artifact["capstone_stamp_fix_verified"] is True
    assert artifact["registries_reconciled"] is True
    assert artifact["v402_outcomes"]["action_cost_heuristic"]["action_efficiency_improves"] is True
    assert artifact["field_principles"] == exp4355.FIELD_PRINCIPLES

    written = json.loads(
        (tmp_path / exp4355.EXP4355_ARTIFACT_PATH).read_text(encoding="utf-8")
    )
    assert written["honest_verdict"] == artifact["honest_verdict"]
    for field in exp4355.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4355.validate_artifact(malformed)
    with pytest.raises(ValueError, match="gap4_regression_guard_passed"):
        exp4355.validate_artifact({**artifact, "gap4_regression_guard_passed": "true"})
    with pytest.raises(ValueError, match="capstone_stamp_fix_verified"):
        exp4355.validate_artifact({**artifact, "capstone_stamp_fix_verified": 1})
    with pytest.raises(ValueError, match="registries_reconciled"):
        exp4355.validate_artifact({**artifact, "registries_reconciled": None})


def test_req_verify_4355_blocks_unreadable_ledgers(tmp_path: Path) -> None:
    """REQ-VERIFY-4355: unreadable registries fail closed before mutation."""

    _write_minimal_repo(tmp_path)
    (tmp_path / "ops" / "verifier_registry.yaml").write_text("[not: registry]\n", encoding="utf-8")

    artifact = exp4355.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    assert artifact["honest_verdict"] == "blocked_verifier_registry_unreadable"
    assert artifact["gap4_regression_guard_passed"] is False
    assert artifact["capstone_stamp_fix_verified"] is False
    assert artifact["registries_reconciled"] is False


def test_req_verify_4355_defensive_readers_and_validators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4355: malformed inputs are recorded rather than fabricated."""

    assert exp4355._load_optional_json(tmp_path, "missing.json")[0] is None
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert "JSONDecodeError" in exp4355._load_optional_json(tmp_path, "bad.json")[1]
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp4355._load_optional_json(tmp_path, "list.json")[1] == (
        "top-level JSON is not an object"
    )

    assert exp4355._bool(None, "x") is None
    assert exp4355._int(None, "x") == 0
    assert exp4355._float(None, "x") is None
    assert exp4355._str(None, "x") == ""
    assert exp4355._list(None, "x") == []
    assert exp4355._int({"x": True}, "x") == 0
    assert exp4355._float({"x": "1"}, "x") is None

    assert exp4355._read_s3(None, "missing")["available"] is False
    assert exp4355._read_ka59(None, "missing")["available"] is False
    assert exp4355._read_deeper(None, "missing")["targets"] == {}
    assert exp4355._read_partial_games(None, "missing")["games"] == {}
    assert exp4355._read_action_cost(None, "missing")["available"] is False
    deeper = exp4355._read_deeper(
        {"per_target_scorecard": ["bad", {"game": ""}]},
        "",
    )
    partial = exp4355._read_partial_games(
        {"per_game_scorecard": ["bad", {"game": ""}]},
        "",
    )
    assert deeper["targets"] == {}
    assert partial["games"] == {}

    missing_repo = tmp_path / "missing_repo"
    preflight = exp4355.check_preconditions(missing_repo)
    assert preflight["ok"] is False
    assert preflight["blocked_file"] == "verifier_registry"
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "verifier_registry.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text("{}\n", encoding="utf-8")
    preflight = exp4355.check_preconditions(tmp_path)
    assert preflight["blocked_file"] == "verifier_gaps"

    monkeypatch.setattr(
        exp4355.exp4344,
        "run_gap4_regression_guard",
        lambda root: {"regression_guard_passed": True, "root": str(root)},
    )
    assert exp4355.run_gap4_regression_guard(tmp_path)["regression_guard_passed"] is True

    registry: dict[str, Any] = {"verifiers": []}
    outcomes = exp4355.load_v402_outcomes(_repo_with_payloads(tmp_path))
    gaps = exp4355.build_gap_entries(outcomes)
    exp4355._ensure_gap4_eval(registry, _guard(), outcomes, gaps)
    assert exp4355.registry_contains_v402(registry) is False
    exp4355._ensure_v402_role({"verifiers": []}, outcomes, gaps)
    arc: dict[str, Any] = {"games": []}
    exp4355._ensure_arc_registry(arc, outcomes)
    assert exp4355._find_game({}, "missing") is None
    assert exp4355.arc_registry_contains_v402(arc) is True

    assert exp4355._clean_flags_from_report({}) == []
    assert exp4355._clean_flags_from_report({"reports": [{}]}) == []

    completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=0,
        stdout=json.dumps({"reports": [{"flags": []}]}),
        stderr="",
    )
    monkeypatch.setattr(
        exp4355.capstone_v401_4346,
        "build_artifact",
        lambda *args, **kwargs: {
            "honest_verdict": "complete: gate_MET_fixture",
            "diffusiongemma_gate_status": "MET_oracle_distinct_leak_robust_replicated",
            "verifier_is_oracle": False,
        },
    )
    monkeypatch.setattr(exp4355.subprocess, "run", lambda *args, **kwargs: completed)
    stamp = exp4355.verify_capstone_stamp_fix(tmp_path)
    assert stamp["capstone_stamp_fix_verified"] is True

    bad_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout="not json",
        stderr="",
    )
    monkeypatch.setattr(exp4355.subprocess, "run", lambda *args, **kwargs: bad_completed)
    bad_stamp = exp4355.verify_capstone_stamp_fix(tmp_path)
    assert bad_stamp["capstone_stamp_fix_verified"] is False
    assert bad_stamp["flags"] == []


def _repo_with_payloads(tmp_path: Path) -> Path:
    repo = tmp_path / "payload_repo"
    _write_minimal_repo(repo)
    return repo


def test_req_verify_4355_capstone_propagates_oracle_distinct_stamp(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4355: capstone stamps verifier_is_oracle=false for a clean moat."""

    from scripts import adversarial_verify

    support = tmp_path
    (support / "scripts").mkdir(parents=True)
    (support / "scripts" / "publication_gate.py").write_text("# fixture\n", encoding="utf-8")
    (support / "ops").mkdir(parents=True)
    (support / "ops" / "arc_solve_registry.yaml").write_text(
        "schema_version: 1\nreproducible_total_levels: 17\n",
        encoding="utf-8",
    )
    payloads = {
        "4337_leak_robust_scorer": {
            "honest_verdict": "complete: leak fixture",
            "scorer_leak_audit_passed": True,
            "masked_answer_recovery_auroc": 0.55,
            "process_ranking_auroc": 0.7,
            "verifier_is_oracle": False,
        },
        "4338_in_generation_moat": {
            "honest_verdict": "complete: replication fixture",
            "in_generation_moat_replicates": True,
            "replication_ci95": [0.1, 0.3],
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": True,
            "benchmark_n": 240,
            "carnot_minus_best_control_delta": 0.2,
            "carnot_minus_self_reward_smc_delta": 0.15,
            "verifier_is_oracle": False,
        },
        "4339_e3_ar25": {"game": "ar25", "offline_reproduced": True, "reproduced_levels": 1, "verifier_is_oracle": True},
        "4340_e3_ka59": {"game": "ka59", "offline_reproduced": False, "reproduced_levels": 0, "verifier_is_oracle": True},
        "4341_e3_sc25": {"game": "sc25", "offline_reproduced": True, "reproduced_levels": 1, "verifier_is_oracle": True},
        "4342_self_learning": {"learned_encoder_transfer_helps": False, "verifier_is_oracle": False},
        "4344_hygiene": {"regression_guard_passed": True, "registry_reconciled": True, "manifest_reconciled": True, "gaps_logged": 3},
    }
    for key, payload in payloads.items():
        _write_json(support / capstone_v401_4346.DEFAULT_UPSTREAMS[key].path, payload)

    artifact = capstone_v401_4346.build_artifact(
        support,
        started_s=1.0,
        now_s=1.5,
        live_flag_runner=lambda _: [],
        summarize_runner=lambda _path, _root: 0,
        publication_gate_runner=lambda _: {"paper_ready": True, "unmet_gates": []},
    )

    assert artifact["verifier_is_oracle"] is False
    sample_path = tmp_path / "sample_capstone.json"
    _write_json(sample_path, artifact)
    report = adversarial_verify.verify_artifact(sample_path)
    assert [
        flag for flag in report["flags"] if flag["kind"] == "CIRCULAR_MOAT_OVERCLAIM"
    ] == []


def test_req_verify_4355_results_entrypoint_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4355: results entrypoint calls the package runner."""

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RESULTS_WRAPPER_PATH)])
    monkeypatch.setattr(exp4355, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp4355, "run_gap4_regression_guard", lambda _root: _guard())
    monkeypatch.setattr(exp4355, "verify_capstone_stamp_fix", lambda _root: _stamp_report())
    python_root = str(REPO_ROOT / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_root])

    runpy.run_path(str(REPO_ROOT / RESULTS_WRAPPER_PATH), run_name="__main__")

    payload = json.loads(
        (tmp_path / exp4355.EXP4355_ARTIFACT_PATH).read_text(encoding="utf-8")
    )
    assert payload["gap4_regression_guard_passed"] is True
    assert payload["capstone_stamp_fix_verified"] is True
    assert payload["registries_reconciled"] is True


def test_req_verify_4355_artifact_schema_rejects_non_object_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4355: schema validation rejects malformed object fields."""

    _write_minimal_repo(tmp_path)
    artifact = exp4355.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    for field in (
        "preconditions_checked",
        "v402_outcomes",
        "registry_reconciliation",
        "gap4_regression_guard",
        "capstone_stamp_fix",
    ):
        malformed = dict(artifact)
        malformed[field] = []
        with pytest.raises(ValueError, match=field):
            exp4355.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4355.validate_artifact({**artifact, "honest_verdict": "not_terminal"})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4355.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="random_seed"):
        exp4355.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="field_principles"):
        exp4355.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4355.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4355"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4355.validate_artifact({**artifact, "inference_substrate": "live"})

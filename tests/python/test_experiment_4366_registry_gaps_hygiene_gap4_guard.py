"""Tests for Exp 4366 registry/gaps hygiene, GAP-4 guard, and stamp durability.

Spec refs: REQ-VERIFY-4366, SCENARIO-VERIFY-4366.
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

from carnot import experiment_4366_registry_gaps_hygiene_gap4_guard as wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_gap4_guard_4366 as exp4366


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
RESULTS_WRAPPER_PATH = Path("results/experiment_4366_registry_gaps_hygiene_gap4_guard.py")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4366.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "eval": {"metric": "pass_at_2"},
                "registry_roles": [],
            },
            {
                "verifier_id": exp4366.ACTION_COST_VERIFIER_ID,
                "domain": "arc_agi3_interactive",
                "version": 1,
                "kind": "search_heuristic",
                "eval": {"eval_exp_4355": "old"},
            },
        ]
    }


def _minimal_arc_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated": "2026-06-17",
        "games": [
            {"game": "tu93", "reproducibility": "reproduced", "levels_reproduced": 3},
            {"game": "tr87", "reproducibility": "unsolved", "levels_reproduced": 0},
            {"game": "ft09", "reproducibility": "unsolved", "levels_reproduced": 0},
        ],
        "reproducible_total_levels": 23,
        "reproducible_total_games": 15,
        "provisional_total_levels": 5,
    }


def _minimal_gaps_text() -> str:
    return (
        "# Verifier Gaps\n\n"
        "<!-- exp4355-gap-e3-world-model-rule-tr87-4352:start -->\n"
        "### GAP-E3-WORLD-MODEL-RULE-TR87-4352: Exp 4355 .402 verifier gap update\n"
        "- status: open\n"
        "- evidence: old tr87 partial.\n"
        "- failure mode: tr87 partial\n"
        "- missing discriminator: tr87 rules\n"
        "- candidate design: old design\n"
        "- priority: high\n"
        "<!-- exp4355-gap-e3-world-model-rule-tr87-4352:end -->\n\n"
        "<!-- exp4355-gap-e3-world-model-rule-ft09-4352:start -->\n"
        "### GAP-E3-WORLD-MODEL-RULE-FT09-4352: Exp 4355 .402 verifier gap update\n"
        "- status: open\n"
        "- evidence: old ft09 partial.\n"
        "- failure mode: ft09 partial\n"
        "- missing discriminator: ft09 rules\n"
        "- candidate design: old design\n"
        "- priority: high\n"
        "<!-- exp4355-gap-e3-world-model-rule-ft09-4352:end -->\n"
    )


def _minimal_v403_payloads() -> dict[str, dict[str, Any]]:
    return {
        exp4366.EXP4359_PATH: {
            "honest_verdict": "scorer_leaky_in_search_corpus",
            "acceptance_gate": True,
            "benchmark_n": 0,
            "controls_differentiated": False,
            "s3_guided_beats_control": False,
            "scorer_leak_recheck_passed": False,
            "s3_gain_ci95": [0.0, 0.0],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "35e89213a3d7459ddb5610b79dbc7ab8eb638684e78d4798febd4a22fe4736d7",
        },
        exp4366.EXP4361_PATH: {
            "honest_verdict": "success_e3_deeper_tu93_reproduced",
            "new_levels_reproduced": 1,
            "reproducible_total_levels": 33,
            "verifier_is_oracle": True,
            "per_target_scorecard": [
                {
                    "game": "sc25",
                    "offline_reproduced": False,
                    "new_reproduced_level": 1,
                    "prior_best_level": 1,
                    "residual_win_mechanic_gap_class": "sc25_l2_live_recorded_not_offline_reproduced_spell_delta_gap",
                    "verifier_accuracy": 1.0,
                    "world_model_path": "results/arc_e3/sc25/world_model.py",
                },
                {
                    "game": "tn36",
                    "offline_reproduced": False,
                    "new_reproduced_level": 7,
                    "prior_best_level": 7,
                    "residual_win_mechanic_gap_class": "tn36_l8_program_editor_maze_delta_gap",
                    "verifier_accuracy": 0.875,
                    "world_model_path": "scripts/arc3_tn36_offline_solver.py",
                },
                {
                    "game": "lp85",
                    "offline_reproduced": False,
                    "new_reproduced_level": 4,
                    "prior_best_level": 4,
                    "residual_win_mechanic_gap_class": "lp85_l5_search_path_not_offline_reproduced_reset_replay_gap",
                    "verifier_accuracy": 1.0,
                    "world_model_path": "python/carnot/agentic/arc_game_adapters.py",
                },
                {
                    "game": "tu93",
                    "offline_reproduced": True,
                    "new_reproduced_level": 4,
                    "prior_best_level": 3,
                    "residual_win_mechanic_gap_class": "none",
                    "verifier_accuracy": 1.0,
                    "world_model_path": "python/carnot/agentic/arc_game_adapters.py",
                },
            ],
            "reproducibility_checksum": "bce9878e3d5396e127ea1342fd0452b841b61935eb539e1da466055962842a90",
        },
        exp4366.EXP4362_PATH: {
            "honest_verdict": "complete_e3_ar25_ka59_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 32,
            "verifier_is_oracle": True,
            "per_game_scorecard": [
                {
                    "game": "ar25",
                    "offline_reproduced": False,
                    "new_reproduced_level": 1,
                    "prior_best_level": 1,
                    "residual_gap_class": "ar25_l2_hidden_rule_delta_not_reproduced_action7_undo_stack_gap",
                    "verifier_accuracy": 0.86875,
                    "world_model_path": "results/arc_e3/ar25/world_model.py",
                },
                {
                    "game": "ka59",
                    "offline_reproduced": False,
                    "new_reproduced_level": 1,
                    "prior_best_level": 1,
                    "residual_gap_class": "ka59_l2_hidden_step_counter_hud_register_gap",
                    "verifier_accuracy": 0.64375,
                    "world_model_path": "results/arc_e3/ka59/world_model.py",
                },
            ],
            "reproducibility_checksum": "3f26809ef8c93a4f0dab633c35a36a516ffe4cc1b7d2d49fb67982681570c77d",
        },
        exp4366.EXP4363_PATH: {
            "honest_verdict": "success_e3_tr87_ft09_2_reproduced",
            "new_levels_reproduced": 2,
            "verifier_is_oracle": True,
            "per_game_scorecard": [
                {
                    "game": "tr87",
                    "offline_reproduced": True,
                    "reproduced_levels": 1,
                    "residual_mismatch_class": "none",
                    "verifier_accuracy": 1.0,
                    "mechanic_checks_passed": True,
                    "plan_action_count": 14,
                    "world_model_path": "results/arc_e3/tr87/world_model.py",
                    "world_model_sha256": "1f497c0f688a9d8b4e5439f77cfeea445fc7326599a5aa118037388ff14bd21a",
                },
                {
                    "game": "ft09",
                    "offline_reproduced": True,
                    "reproduced_levels": 1,
                    "residual_mismatch_class": "none",
                    "verifier_accuracy": 1.0,
                    "mechanic_checks_passed": True,
                    "plan_action_count": 4,
                    "world_model_path": "results/arc_e3/ft09/world_model.py",
                    "world_model_sha256": "8e4563289295741b6728075a411d0aa4ae7165140d2f3e5e5e885312dcf650ee",
                },
            ],
            "reproducibility_checksum": "fe02d5adf671bd769d10dea11d6b3a53ba1ff92e9d54d9c7bbbe8eb4d63e2168",
        },
        exp4366.EXP4364_PATH: {
            "honest_verdict": "success: action_efficiency_compounds_25_to_16",
            "acceptance_gate_passed": True,
            "action_efficiency_compounds": True,
            "deployed_into_solver_kit": True,
            "positive_control_passed": True,
            "reproduction_gated": True,
            "verifier_is_oracle": False,
            "compounding_curve": [
                {"corpus_size_k": 4, "held_out_actions_to_solve": 25},
                {"corpus_size_k": 19, "held_out_actions_to_solve": 16},
            ],
            "missing_verifier_gaps": [],
            "reproducibility_checksum": "sha256:31d46454244ed033c9c22f3020d70a57687d9d8a32e6cf38778ccff786fcab7e",
        },
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
        _minimal_gaps_text(),
        encoding="utf-8",
    )
    for rel_path, payload in _minimal_v403_payloads().items():
        _write_json(tmp_path / rel_path, payload)
    _write_json(
        tmp_path / exp4366.CAPSTONE_V402_PATH,
        {
            "honest_verdict": "complete: v402_fixture",
            "diffusiongemma_gate_status": "MET_oracle_distinct_leak_robust_replicated",
            "verifier_is_oracle": False,
        },
    )


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
        "capstone_stamp_fix_durable": True,
        "capstone_path": exp4366.CAPSTONE_V402_PATH,
        "capstone_verifier_is_oracle": False,
        "capstone_aggregation_propagates_oracle_stamp": True,
        "circular_moat_overclaim_fired": False,
        "flag_count": 0,
        "returncode": 0,
        "flags": [],
    }


def test_req_verify_4366_spec_declared() -> None:
    """REQ-VERIFY-4366: OpenSpec declares the .403 hygiene guard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4366",
        "SCENARIO-VERIFY-4366",
        "python/carnot/experiment_4366_registry_gaps_hygiene_gap4_guard.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4366.EXP4366_ARTIFACT_PATH,
        "blocked_<file>_unreadable",
        "capstone_stamp_fix_durable",
        "CIRCULAR_MOAT_OVERCLAIM",
        "reproducible_total_levels=33",
        "action_efficiency_compounds=true",
    ):
        assert marker in spec
    for field in exp4366.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp4366.FIELD_PRINCIPLES.values():
        assert principle in spec
    assert wrapper.main is exp4366.main


def test_scenario_verify_4366_ledgers_record_v403_truth(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4366: .403 outcomes reconcile ledgers without pruning."""

    _write_minimal_repo(tmp_path)
    outcomes = exp4366.load_v403_outcomes(tmp_path)
    gap_entries = exp4366.build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = exp4366.ensure_ledgers_record_v403(
        _minimal_registry(),
        _minimal_gaps_text(),
        _minimal_arc_registry(),
        _guard(),
        outcomes,
        gap_entries,
    )

    assert exp4366.registry_contains_v403(registry) is True
    assert exp4366.arc_registry_contains_v403(arc_registry) is True
    assert exp4366.gaps_contain_v403(gaps_text, gap_entries) is True
    assert summary["registries_reconciled"] is True
    assert exp4366.GAP_E3_WORLD_MODEL_RULE_TR87_4352 in summary["filled_gap_ids"]
    assert exp4366.GAP_E3_WORLD_MODEL_RULE_FT09_4352 in summary["filled_gap_ids"]
    assert "status: filled (exp4363_tr87_ft09_world_models)" in gaps_text
    assert "GAP-E3-WORLD-MODEL-RULE-TN36-L8-4361" in gaps_text
    action = next(
        row for row in registry["verifiers"] if row["verifier_id"] == exp4366.ACTION_COST_VERIFIER_ID
    )
    assert action["eval"]["action_efficiency_compounds"] is True
    assert action["eval"]["held_out_actions_learned"] == 16


def test_req_verify_4366_run_hygiene_writes_required_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-4366: terminal artifact exposes bare guard/stamp fields."""

    _write_minimal_repo(tmp_path)
    artifact = exp4366.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    exp4366.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gap4_regression_guard_passed"] is True
    assert artifact["capstone_stamp_fix_durable"] is True
    assert artifact["registries_reconciled"] is True
    assert artifact["v403_outcomes"]["action_cost_compounding"]["action_efficiency_compounds"] is True

    written = json.loads(
        (tmp_path / exp4366.EXP4366_ARTIFACT_PATH).read_text(encoding="utf-8")
    )
    assert written["honest_verdict"] == artifact["honest_verdict"]
    for field in exp4366.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4366.validate_artifact(malformed)
    with pytest.raises(ValueError, match="gap4_regression_guard_passed"):
        exp4366.validate_artifact({**artifact, "gap4_regression_guard_passed": "true"})
    with pytest.raises(ValueError, match="capstone_stamp_fix_durable"):
        exp4366.validate_artifact({**artifact, "capstone_stamp_fix_durable": 1})
    with pytest.raises(ValueError, match="registries_reconciled"):
        exp4366.validate_artifact({**artifact, "registries_reconciled": None})


def test_req_verify_4366_blocks_unreadable_ledgers(tmp_path: Path) -> None:
    """REQ-VERIFY-4366: unreadable registries fail closed before mutation."""

    _write_minimal_repo(tmp_path)
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text("[]\n", encoding="utf-8")

    artifact = exp4366.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    assert artifact["honest_verdict"] == "blocked_arc_solve_registry_unreadable"
    assert artifact["gap4_regression_guard_passed"] is False
    assert artifact["capstone_stamp_fix_durable"] is False
    assert artifact["registries_reconciled"] is False


def test_req_verify_4366_defensive_readers_and_stamp_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4366: malformed inputs are recorded and stamp durability is audited."""

    assert exp4366._load_optional_json(tmp_path, "missing.json")[0] is None
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert "JSONDecodeError" in exp4366._load_optional_json(tmp_path, "bad.json")[1]
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp4366._load_optional_json(tmp_path, "list.json")[1] == (
        "top-level JSON is not an object"
    )

    assert exp4366._bool(None, "x") is None
    assert exp4366._int(None, "x") == 0
    assert exp4366._int({"x": True}, "x") == 0
    assert exp4366._str(None, "x") == ""
    assert exp4366._list(None, "x") == []
    assert exp4366._scorecard_map(["bad", {"game": ""}], "residual") == {}
    assert exp4366._read_prism(None, "missing")["available"] is False
    assert exp4366._read_deeper(None, "missing")["targets"] == {}
    assert exp4366._read_blocked_mechanics(None, "missing")["games"] == {}
    assert exp4366._read_tail_games(None, "missing")["games"] == {}
    assert exp4366._read_action_compounding(None, "missing")["available"] is False
    assert exp4366._curve_endpoint([], 7) == (7, 7)

    missing_repo = tmp_path / "missing_repo"
    preflight = exp4366.check_preconditions(missing_repo)
    assert preflight["ok"] is False
    assert preflight["blocked_file"] == "verifier_registry"
    no_gaps_repo = tmp_path / "no_gaps_repo"
    (no_gaps_repo / "ops").mkdir(parents=True)
    (no_gaps_repo / "ops" / "verifier_registry.yaml").write_text("{}\n", encoding="utf-8")
    (no_gaps_repo / "ops" / "arc_solve_registry.yaml").write_text("{}\n", encoding="utf-8")
    no_gaps_preflight = exp4366.check_preconditions(no_gaps_repo)
    assert no_gaps_preflight["blocked_file"] == "verifier_gaps"

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(
        exp4366.exp4355,
        "run_gap4_regression_guard",
        lambda root: {"regression_guard_passed": True, "root": str(root)},
    )
    assert exp4366.run_gap4_regression_guard(tmp_path)["regression_guard_passed"] is True

    outcomes = exp4366.load_v403_outcomes(tmp_path)
    skipped = json.loads(json.dumps(outcomes))
    skipped["arc_e3"]["deeper_high_headroom"]["targets"]["sc25"]["offline_reproduced"] = True
    skipped["arc_e3"]["blocked_mechanics"]["games"].pop("ka59")
    assert exp4366.GAP_E3_WORLD_MODEL_RULE_SC25_L2_4361 not in [
        gap["gap_id"] for gap in exp4366.build_gap_entries(skipped)
    ]
    registry: dict[str, Any] = {"verifiers": []}
    gaps = exp4366.build_gap_entries(outcomes)
    exp4366._ensure_gap4_eval(registry, _guard(), outcomes, gaps)
    assert exp4366.GAP4_VERIFIER_ID in [row["verifier_id"] for row in registry["verifiers"]]
    exp4366._ensure_v403_role({"verifiers": []}, outcomes, gaps)
    exp4366._ensure_action_cost_verifier({"verifiers": []}, outcomes)
    arc: dict[str, Any] = {"games": []}
    assert exp4366._find_game(arc, "missing") is None
    assert exp4366._ensure_game(arc, "new") == {"game": "new"}
    assert exp4366._flags_from_report({}) == []
    assert exp4366._flags_from_report({"reports": [{}]}) == []

    missing_capstone = tmp_path / "missing_capstone"
    _write_minimal_repo(missing_capstone)
    (missing_capstone / exp4366.CAPSTONE_V402_PATH).unlink()
    missing_stamp = exp4366.verify_capstone_stamp_fix_durable(missing_capstone)
    assert missing_stamp["capstone_stamp_fix_durable"] is False
    assert "error" in missing_stamp

    completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=0,
        stdout=json.dumps({"reports": [{"flags": [], "flag_count": 0}], "flagged_count": 0}),
        stderr="",
    )
    monkeypatch.setattr(exp4366.subprocess, "run", lambda *args, **kwargs: completed)
    stamp = exp4366.verify_capstone_stamp_fix_durable(tmp_path)
    assert stamp["capstone_stamp_fix_durable"] is True

    bad_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout=json.dumps(
            {
                "reports": [
                    {
                        "flags": [
                            {
                                "kind": "CIRCULAR_MOAT_OVERCLAIM",
                                "severity": "critical",
                                "detail": "fixture",
                            }
                        ],
                        "flag_count": 1,
                    }
                ],
                "flagged_count": 1,
            }
        ),
        stderr="",
    )
    monkeypatch.setattr(exp4366.subprocess, "run", lambda *args, **kwargs: bad_completed)
    bad_stamp = exp4366.verify_capstone_stamp_fix_durable(tmp_path)
    assert bad_stamp["capstone_stamp_fix_durable"] is False
    assert bad_stamp["circular_moat_overclaim_fired"] is True

    invalid_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout="not json",
        stderr="broken",
    )
    monkeypatch.setattr(exp4366.subprocess, "run", lambda *args, **kwargs: invalid_completed)
    invalid_stamp = exp4366.verify_capstone_stamp_fix_durable(tmp_path)
    assert invalid_stamp["capstone_stamp_fix_durable"] is False
    assert invalid_stamp["flags"] == []


def test_req_verify_4366_artifact_schema_rejects_malformed_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4366: schema validation rejects malformed terminal artifacts."""

    _write_minimal_repo(tmp_path)
    artifact = exp4366.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    for field in (
        "preconditions_checked",
        "v403_outcomes",
        "registry_reconciliation",
        "gap4_regression_guard",
        "capstone_stamp_fix",
    ):
        malformed = dict(artifact)
        malformed[field] = []
        with pytest.raises(ValueError, match=field):
            exp4366.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4366.validate_artifact({**artifact, "honest_verdict": "not_terminal"})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4366.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="random_seed"):
        exp4366.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="field_principles"):
        exp4366.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4366.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4366"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4366.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_verify_4366_results_entrypoint_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4366: results entrypoint calls the package runner."""

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RESULTS_WRAPPER_PATH)])
    monkeypatch.setattr(exp4366, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp4366, "run_gap4_regression_guard", lambda _root: _guard())
    monkeypatch.setattr(exp4366, "verify_capstone_stamp_fix_durable", lambda _root: _stamp_report())
    python_root = str(REPO_ROOT / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_root])

    runpy.run_path(str(REPO_ROOT / RESULTS_WRAPPER_PATH), run_name="__main__")

    payload = json.loads(
        (tmp_path / exp4366.EXP4366_ARTIFACT_PATH).read_text(encoding="utf-8")
    )
    assert payload["gap4_regression_guard_passed"] is True
    assert payload["capstone_stamp_fix_durable"] is True
    assert payload["registries_reconciled"] is True

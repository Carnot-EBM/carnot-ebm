"""Tests for Exp 4388 registry/gaps hygiene, GAP-4 guard, and stamp durability.

Spec refs: REQ-VERIFY-4388, SCENARIO-VERIFY-4388.
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

from carnot import experiment_4388_registry_gaps_hygiene_gap4_guard as wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_gap4_guard_4388 as exp4388


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
RESULTS_WRAPPER_PATH = Path("results/experiment_4388_registry_gaps_hygiene_gap4_guard.py")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4388.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "eval": {"metric": "pass_at_2"},
                "registry_roles": [],
            },
            {
                "verifier_id": exp4388.FOVER_VERIFIER_ID,
                "domain": "math_reasoning",
                "version": 4,
                "kind": "ensemble",
                "eval": {"metric": "fover_dual_condition_auroc"},
            },
        ]
    }


def _minimal_arc_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated": "2026-06-18",
        "games": [{"game": "lp85", "reproducibility": "reproduced", "levels_reproduced": 5}],
        "reproducible_total_levels": 34,
        "reproducible_total_games": 17,
        "provisional_total_levels": 5,
    }


def _minimal_arc_registry_text() -> str:
    return (
        'schema_version: 1\n'
        'updated: "2026-06-18"\n'
        'games:\n'
        '  - game: lp85\n'
        '    reproducibility: reproduced\n'
        '    levels_reproduced: 5\n'
        'reproducible_total_levels: 34\n'
        'reproducible_total_games: 17\n'
    )


def _minimal_gaps_text() -> str:
    return "# Verifier Gaps\n\nHistorical note remains.\n"


def _minimal_v405_payloads() -> dict[str, dict[str, Any]]:
    return {
        exp4388.EXP4381_PATH: {
            "honest_verdict": "complete: clean_powered_null_bidirectional_not_actionable",
            "detector_localization_actionable": False,
            "localization_delta_ci95": [0.0, 0.0],
            "localization_f1_by_direction": {
                "causal_online": {"f1": 0.096491, "n_error_traces": 114},
                "bidirectional_fusion": {"f1": 0.096491, "n_error_traces": 114},
            },
            "abstention_curve": {"detector_auroc": 0.979903},
            "n_traces": 6548,
            "n_error_traces": 114,
            "missing_verifier_gaps": [
                {
                    "gap_id": exp4388.GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED,
                    "error_class": "untyped",
                    "missed_first_error_traces": 103,
                    "missing_discriminator": "first-causal-error feature",
                    "status": "open",
                }
            ],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:4381",
        },
        exp4388.EXP4382_PATH: {
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "gate_check_summary": "exp4381 detector_localization_actionable failed",
            "gates_evaluated": [{"passed": False}],
        },
        exp4388.EXP4383_PATH: {
            "honest_verdict": "complete_e3_deeper_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "verifier_is_oracle": True,
            "per_target_scorecard": [
                {
                    "game": "lp85",
                    "prior_best_level": 5,
                    "target_level": 6,
                    "new_reproduced_level": 5,
                    "offline_reproduced": False,
                    "residual_win_mechanic_gap_class": "wall_time_cap_exhausted",
                    "verifier_accuracy": 0.833333,
                    "lookahead_fidelity": 0.833333,
                },
                {
                    "game": "tu93",
                    "prior_best_level": 4,
                    "target_level": 5,
                    "new_reproduced_level": 4,
                    "offline_reproduced": False,
                    "residual_win_mechanic_gap_class": "wall_time_cap_exhausted",
                    "verifier_accuracy": 0.8,
                    "lookahead_fidelity": 0.8,
                },
                {
                    "game": "tn36",
                    "prior_best_level": 7,
                    "target_level": 8,
                    "new_reproduced_level": 7,
                    "offline_reproduced": False,
                    "residual_win_mechanic_gap_class": (
                        "tn36_l8_program_editor_object_control_gap_sxhtkytekm_palette_population"
                    ),
                    "verifier_accuracy": 0.875,
                    "lookahead_fidelity": 0.875,
                },
                {
                    "game": "tr87",
                    "prior_best_level": 6,
                    "target_level": 7,
                    "new_reproduced_level": 6,
                    "offline_reproduced": False,
                    "residual_win_mechanic_gap_class": "wall_time_cap_exhausted",
                    "verifier_accuracy": 0.857143,
                    "lookahead_fidelity": 0.857143,
                },
            ],
        },
        exp4388.EXP4384_PATH: {
            "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "verifier_is_oracle": True,
            "per_game_scorecard": [
                {
                    "game": "ar25",
                    "prior_best_level": 1,
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "residual_gap_class": "ar25_l2_action7_undo_stack_hidden_rule_gap",
                    "verifier_accuracy": 0.90625,
                    "lookahead_fidelity": 0.733333,
                    "mechanic_checks_passed": False,
                },
                {
                    "game": "ka59",
                    "prior_best_level": 1,
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "residual_gap_class": "ka59_l2_hidden_step_counter_hud_register_gap",
                    "verifier_accuracy": 0.125,
                    "lookahead_fidelity": 0.112281,
                    "mechanic_checks_passed": False,
                },
                {
                    "game": "ft09",
                    "prior_best_level": 1,
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "residual_gap_class": "ft09_l2_residual_world_model_mismatch_gap",
                    "verifier_accuracy": 0.166667,
                    "lookahead_fidelity": 0.347518,
                    "mechanic_checks_passed": False,
                },
            ],
        },
        exp4388.EXP4385_PATH: {
            "honest_verdict": "success: detector_compounds_heldout_localization_f1",
            "detector_compounds": True,
            "positive_control_passed": True,
            "compounding_delta_ci95": [0.003396, 0.032772],
            "no_learning_baseline": 0.145773,
            "learning_curve": [
                {"train_corpus_size": 491, "held_out_localization_f1": 0.371134},
                {"train_corpus_size": 4911, "held_out_localization_f1": 0.387097},
            ],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:4385",
        },
        exp4388.EXP4386_PATH: {
            "honest_verdict": "success: detector_generalizes_cross_domain_non_fover",
            "detector_generalizes_cross_domain": True,
            "detection_by_domain": [
                {
                    "domain": "gap4_arc",
                    "detection_auroc": 0.963317,
                    "auroc_ci95": [0.922285, 0.990662],
                    "selection_headroom": 0.129,
                    "n": 28443,
                    "base_rate": 0.001688,
                }
            ],
            "domains_at_chance": [],
            "missing_verifier_gaps": [],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:4386",
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
        _minimal_arc_registry_text(),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text(_minimal_gaps_text(), encoding="utf-8")
    for rel_path, payload in _minimal_v405_payloads().items():
        _write_json(tmp_path / rel_path, payload)
    _write_json(
        tmp_path / exp4388.CAPSTONE_V404_PATH,
        {
            "honest_verdict": "complete: v404_fixture",
            "verifier_is_oracle": False,
            "verifier_is_oracle_honored": True,
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
        "capstone_path": exp4388.CAPSTONE_V404_PATH,
        "capstone_verifier_is_oracle": False,
        "capstone_verifier_is_oracle_honored": True,
        "capstone_aggregation_propagates_oracle_stamp": True,
        "capstone_aggregation_uses_available_helper": True,
        "circular_moat_overclaim_fired": False,
        "flag_count": 0,
        "returncode": 0,
        "flags": [],
    }


def test_req_verify_4388_spec_declared() -> None:
    """REQ-VERIFY-4388: OpenSpec declares the .405 hygiene guard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4388",
        "SCENARIO-VERIFY-4388",
        "python/carnot/experiment_4388_registry_gaps_hygiene_gap4_guard.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4388.EXP4388_ARTIFACT_PATH,
        "blocked_<file>_unreadable",
        "capstone_stamp_fix_durable",
        "CIRCULAR_MOAT_OVERCLAIM",
        "detector_compounds=true",
        "detector_generalizes_cross_domain=true",
        "reproducible_total_levels=34",
    ):
        assert marker in spec
    for field in exp4388.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp4388.FIELD_PRINCIPLES.values():
        assert principle in spec
    assert wrapper.main is exp4388.main


def test_scenario_verify_4388_ledgers_record_v405_truth(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4388: .405 outcomes reconcile ledgers without pruning."""

    _write_minimal_repo(tmp_path)
    outcomes = exp4388.load_v405_outcomes(tmp_path)
    gap_entries = exp4388.build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = exp4388.ensure_ledgers_record_v405(
        _minimal_registry(),
        _minimal_gaps_text(),
        _minimal_arc_registry(),
        _guard(),
        outcomes,
        gap_entries,
    )

    assert exp4388.registry_contains_v405(registry) is True
    assert exp4388.arc_registry_contains_v405(arc_registry) is True
    assert exp4388.gaps_contain_v405(gaps_text, gap_entries) is True
    assert summary["registries_reconciled"] is True
    assert summary["filled_gap_ids"] == []
    assert exp4388.GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED in gaps_text
    assert exp4388.GAP_E3_WORLD_MODEL_RULE_LP85_L6_4383 in gaps_text

    gap4 = next(row for row in registry["verifiers"] if row["verifier_id"] == exp4388.GAP4_VERIFIER_ID)
    assert gap4["eval"]["exp4388_arc_reproducible_total_levels"] == 34
    assert gap4["eval"]["exp4388_new_levels_reproduced"] == 0
    assert gap4["eval"]["exp4388_detector_generalizes_cross_domain"] is True
    assert gap4["eval"]["exp4388_cross_domain_gap4_arc_auroc"] == 0.963317

    fover = next(row for row in registry["verifiers"] if row["verifier_id"] == exp4388.FOVER_VERIFIER_ID)
    assert fover["eval"]["exp4388_detector_localization_actionable"] is False
    assert fover["eval"]["exp4388_detector_compounds"] is True
    assert fover["eval"]["exp4388_compounding_delta_ci95"] == [0.003396, 0.032772]

    assert arc_registry["latest_hygiene_4388"]["new_levels_reproduced"] == 0


def test_req_verify_4388_run_hygiene_writes_required_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-4388: terminal artifact exposes bare guard/stamp fields."""

    _write_minimal_repo(tmp_path)
    artifact = exp4388.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    exp4388.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gap4_regression_guard_passed"] is True
    assert artifact["capstone_stamp_fix_durable"] is True
    assert artifact["registries_reconciled"] is True
    assert artifact["v405_outcomes"]["detector_self_learning"]["detector_compounds"] is True
    assert artifact["v405_outcomes"]["cross_domain_detector"]["detector_generalizes_cross_domain"] is True

    written = json.loads((tmp_path / exp4388.EXP4388_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == artifact["honest_verdict"]
    assert "latest_hygiene_4388:" in (
        tmp_path / "ops" / "arc_solve_registry.yaml"
    ).read_text(encoding="utf-8")
    for field in exp4388.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4388.validate_artifact(malformed)


def test_req_verify_4388_blocks_unreadable_ledgers(tmp_path: Path) -> None:
    """REQ-VERIFY-4388: unreadable registries fail closed before mutation."""

    _write_minimal_repo(tmp_path)
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text("[]\n", encoding="utf-8")

    artifact = exp4388.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    assert artifact["honest_verdict"] == "blocked_arc_solve_registry_unreadable"
    assert artifact["gap4_regression_guard_passed"] is False
    assert artifact["capstone_stamp_fix_durable"] is False
    assert artifact["registries_reconciled"] is False


def test_req_verify_4388_defensive_readers_schema_and_stamp_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4388: malformed inputs are recorded and stamp durability is audited."""

    assert exp4388._load_optional_json(tmp_path, "missing.json")[0] is None
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert "JSONDecodeError" in exp4388._load_optional_json(tmp_path, "bad.json")[1]
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp4388._load_optional_json(tmp_path, "list.json")[1] == (
        "top-level JSON is not an object"
    )

    assert exp4388._bool({"x": 1}, "x") is None
    assert exp4388._int({"x": True}, "x") == 0
    assert exp4388._float({"x": True}, "x") is None
    assert exp4388._str(None, "x") == ""
    assert exp4388._list(None, "x") == []
    assert exp4388._scorecard_map(["bad", {"game": ""}], "residual") == {}
    assert exp4388._read_actionable_detector(None, "missing")["available"] is False
    assert exp4388._read_gate_check(None, "missing")["available"] is False
    assert exp4388._read_e3_partial(None, "missing", exp4388.EXP4383_PATH, "rows", "gap")["rows"] == {}
    assert exp4388._read_detector_compounds(None, "missing")["available"] is False
    assert exp4388._read_cross_domain_detector(None, "missing")["available"] is False

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(
        exp4388.exp4377,
        "run_gap4_regression_guard",
        lambda root: {"regression_guard_passed": True, "root": str(root)},
    )
    assert exp4388.run_gap4_regression_guard(tmp_path)["regression_guard_passed"] is True

    outcomes = exp4388.load_v405_outcomes(tmp_path)
    malformed = json.loads(json.dumps(outcomes))
    malformed["actionable_detector"]["missing_verifier_gaps"].insert(0, "bad")
    malformed["actionable_detector"]["missing_verifier_gaps"].insert(1, {"gap_id": ""})
    malformed["arc_e3"]["deeper_lookahead"]["rows"]["lp85"]["offline_reproduced"] = True
    malformed["arc_e3"]["blocked_mechanics"]["rows"].pop("ka59")
    malformed_ids = [gap["gap_id"] for gap in exp4388.build_gap_entries(malformed)]
    assert exp4388.GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED in malformed_ids
    assert exp4388.GAP_E3_WORLD_MODEL_RULE_LP85_L6_4383 not in malformed_ids
    assert exp4388.GAP_E3_WORLD_MODEL_RULE_KA59_L2_4384 not in malformed_ids

    chance = json.loads(json.dumps(outcomes))
    chance["cross_domain_detector"]["domains_at_chance"] = [
        "bad",
        {"domain": "code", "detection_auroc": 0.51, "auroc_ci95": [0.49, 0.53], "n": 1000}
    ]
    assert "GAP-DETECTOR-CROSS-DOMAIN-code-4386" in [
        gap["gap_id"] for gap in exp4388.build_gap_entries(chance)
    ]
    assert exp4388._domain_row(chance["cross_domain_detector"], "missing") == {}

    arc_text = _minimal_arc_registry_text()
    patched = exp4388._patch_arc_registry_text(arc_text, outcomes)
    assert "latest_hygiene_4388:" in patched
    assert exp4388._patch_arc_registry_text(patched, outcomes) == patched
    assert exp4388._replace_or_append_gap(
        "### GAP-FOVER-BIPRM-LOCALIZATION-untyped: already present\n",
        "exp4388-gap-fover-biprm-localization-untyped",
        {
            "gap_id": exp4388.GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED,
            "status": "open",
            "evidence": "fixture",
            "failure_mode": "fixture",
            "missing_discriminator": "fixture",
            "candidate_design": "fixture",
            "priority": "medium",
        },
    ) == "### GAP-FOVER-BIPRM-LOCALIZATION-untyped: already present\n"

    fallback_repo = tmp_path / "fallback"
    _write_minimal_repo(fallback_repo)
    (fallback_repo / "ops" / "arc_solve_registry.yaml").write_text(
        _minimal_arc_registry_text()
        + "\nlatest_hygiene_4388:\n  artifact: wrong\n  new_levels_reproduced: 1\n",
        encoding="utf-8",
    )
    fallback_artifact = exp4388.run_hygiene(
        fallback_repo,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )
    assert fallback_artifact["registries_reconciled"] is True
    fallback_arc = yaml.safe_load(
        (fallback_repo / "ops" / "arc_solve_registry.yaml").read_text(encoding="utf-8")
    )
    assert fallback_arc["latest_hygiene_4388"]["artifact"] == exp4388.EXP4388_ARTIFACT_PATH

    registry: dict[str, Any] = {"verifiers": []}
    gaps = exp4388.build_gap_entries(outcomes)
    exp4388._ensure_gap4_eval(registry, _guard(), outcomes, gaps)
    assert exp4388.GAP4_VERIFIER_ID in [row["verifier_id"] for row in registry["verifiers"]]
    exp4388._ensure_v405_role({"verifiers": []}, outcomes, gaps)
    exp4388._ensure_fover_detector({"verifiers": []}, outcomes)
    assert exp4388._flags_from_report({}) == []
    assert exp4388._capstone_aggregation_propagates_oracle_stamp() is True
    assert exp4388._capstone_aggregation_uses_available_helper() is True

    missing_capstone = tmp_path / "missing_capstone"
    _write_minimal_repo(missing_capstone)
    (missing_capstone / exp4388.CAPSTONE_V404_PATH).unlink()
    missing_stamp = exp4388.verify_capstone_stamp_fix_durable(missing_capstone)
    assert missing_stamp["capstone_stamp_fix_durable"] is False
    assert "error" in missing_stamp

    completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=0,
        stdout=json.dumps({"reports": [{"flags": [], "flag_count": 0}], "flagged_count": 0}),
        stderr="",
    )
    monkeypatch.setattr(exp4388.subprocess, "run", lambda *args, **kwargs: completed)
    assert exp4388.verify_capstone_stamp_fix_durable(tmp_path)["capstone_stamp_fix_durable"] is True

    bad_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout=json.dumps(
            {
                "reports": [
                    {
                        "flags": [
                            {"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}
                        ]
                    }
                ]
            }
        ),
        stderr="",
    )
    monkeypatch.setattr(exp4388.subprocess, "run", lambda *args, **kwargs: bad_completed)
    assert exp4388.verify_capstone_stamp_fix_durable(tmp_path)[
        "circular_moat_overclaim_fired"
    ] is True

    invalid_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout="not json",
        stderr="broken",
    )
    monkeypatch.setattr(exp4388.subprocess, "run", lambda *args, **kwargs: invalid_completed)
    assert exp4388.verify_capstone_stamp_fix_durable(tmp_path)["flags"] == []


def test_req_verify_4388_artifact_schema_rejects_malformed_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4388: schema validation rejects malformed terminal artifacts."""

    _write_minimal_repo(tmp_path)
    artifact = exp4388.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    for field in (
        "preconditions_checked",
        "v405_outcomes",
        "registry_reconciliation",
        "gap4_regression_guard",
        "capstone_stamp_fix",
    ):
        malformed = dict(artifact)
        malformed[field] = []
        with pytest.raises(ValueError, match=field):
            exp4388.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4388.validate_artifact({**artifact, "honest_verdict": "not_terminal"})
    with pytest.raises(ValueError, match="gap4_regression_guard_passed"):
        exp4388.validate_artifact({**artifact, "gap4_regression_guard_passed": "true"})
    with pytest.raises(ValueError, match="capstone_stamp_fix_durable"):
        exp4388.validate_artifact({**artifact, "capstone_stamp_fix_durable": 1})
    with pytest.raises(ValueError, match="registries_reconciled"):
        exp4388.validate_artifact({**artifact, "registries_reconciled": None})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4388.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="random_seed"):
        exp4388.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="field_principles"):
        exp4388.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4388.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4388"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4388.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_verify_4388_results_entrypoint_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4388: required results entrypoint delegates to Exp 4388."""

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RESULTS_WRAPPER_PATH)])
    monkeypatch.setattr(exp4388, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp4388, "run_gap4_regression_guard", lambda _root: _guard())
    monkeypatch.setattr(exp4388, "verify_capstone_stamp_fix_durable", lambda _root: _stamp_report())
    python_root = str(REPO_ROOT / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_root])

    runpy.run_path(str(REPO_ROOT / RESULTS_WRAPPER_PATH), run_name="__main__")

    payload = json.loads((tmp_path / exp4388.EXP4388_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert payload["gap4_regression_guard_passed"] is True
    assert payload["capstone_stamp_fix_durable"] is True
    assert payload["registries_reconciled"] is True

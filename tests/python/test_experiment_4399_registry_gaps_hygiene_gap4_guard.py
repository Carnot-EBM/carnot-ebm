"""Tests for Exp 4399 registry/gaps hygiene, GAP-4 guard, and stamp durability.

Spec refs: REQ-VERIFY-4399, SCENARIO-VERIFY-4399.
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

from carnot import experiment_4399_registry_gaps_hygiene_gap4_guard as wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_gap4_guard_4399 as exp4399


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
RESULTS_WRAPPER_PATH = Path("results/experiment_4399_registry_gaps_hygiene_gap4_guard.py")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4399.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "eval": {"metric": "pass_at_2"},
                "registry_roles": [],
            },
            {
                "verifier_id": exp4399.FOVER_VERIFIER_ID,
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


def _minimal_v406_payloads() -> dict[str, dict[str, Any]]:
    return {
        exp4399.EXP4392_PATH: {
            "honest_verdict": "success: synthetic_process_localizer_beats_ensemble_baseline",
            "localizer_beats_ensemble_baseline": True,
            "localization_f1_by_domain": {
                "FoVer": {
                    "synthetic_trained_localizer": 1.0,
                    "delta": 0.904,
                    "delta_ci95": [0.904, 0.904],
                    "n_error_traces": 114,
                },
                "GAP-4 ARC": {
                    "synthetic_trained_localizer": 0.692308,
                    "delta": 0.596308,
                    "delta_ci95": [0.461692, 0.711692],
                    "n_error_traces": 52,
                },
            },
            "missing_verifier_gaps": [
                {
                    "gap_id": exp4399.GAP_4392_FIRST_ERROR_GAP4_ARC,
                    "domain": "GAP-4 ARC",
                    "error_class": "arc_candidate_process_proxy",
                    "missed_first_error_traces": 16,
                    "missing_discriminator": "first causal process break feature",
                    "candidate_design": "typed domain prefix checks",
                    "priority": "medium",
                    "status": "open",
                }
            ],
            "verifier_is_oracle": False,
            "n_traces": 6548,
            "reproducibility_checksum": "sha256:4392",
        },
        exp4399.EXP4393_PATH: {
            "honest_verdict": "complete: a1_win_quarantined_as_artifact_confounded",
            "localizer_win_is_genuine": False,
            "beats_position_only_baseline": False,
            "template_ablation_drop": 0.0,
            "held_out_real_localization_delta_ci95": [0.904, 0.904],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:4393",
        },
        exp4399.EXP4394_PATH: {
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
                    "residual_win_mechanic_gap_class": "lookahead_fidelity_below_gate",
                    "verifier_accuracy": 0.833333,
                    "lookahead_fidelity": 0.833333,
                    "fidelity_gate_passed": False,
                },
                {
                    "game": "tu93",
                    "prior_best_level": 4,
                    "target_level": 5,
                    "new_reproduced_level": 4,
                    "offline_reproduced": False,
                    "residual_win_mechanic_gap_class": "lookahead_fidelity_below_gate",
                    "verifier_accuracy": 0.8,
                    "lookahead_fidelity": 0.8,
                    "fidelity_gate_passed": False,
                },
                {
                    "game": "tn36",
                    "prior_best_level": 7,
                    "target_level": 8,
                    "new_reproduced_level": 7,
                    "offline_reproduced": False,
                    "residual_win_mechanic_gap_class": "lookahead_fidelity_below_gate",
                    "verifier_accuracy": 0.875,
                    "lookahead_fidelity": 0.875,
                    "fidelity_gate_passed": False,
                },
                {
                    "game": "tr87",
                    "prior_best_level": 6,
                    "target_level": 7,
                    "new_reproduced_level": 6,
                    "offline_reproduced": False,
                    "residual_win_mechanic_gap_class": "lookahead_fidelity_below_gate",
                    "verifier_accuracy": 0.857143,
                    "lookahead_fidelity": 0.857143,
                    "fidelity_gate_passed": False,
                },
            ],
            "reproducibility_checksum": "sha256:4394",
        },
        exp4399.EXP4395_PATH: {
            "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "verifier_is_oracle": True,
            "per_game_scorecard": [
                {
                    "game": "ar25",
                    "prior_best_level": 1,
                    "target_level": 2,
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "residual_gap_class": "ar25_l2_action7_undo_stack_hidden_rule_gap",
                    "verifier_accuracy": 0.90625,
                    "lookahead_fidelity": 0.733333,
                },
                {
                    "game": "ka59",
                    "prior_best_level": 1,
                    "target_level": 2,
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "residual_gap_class": "ka59_l2_object_relevance_step_counter_hud_register_gap",
                    "verifier_accuracy": 0.125,
                    "lookahead_fidelity": 0.112281,
                },
                {
                    "game": "ft09",
                    "prior_best_level": 1,
                    "target_level": 2,
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "residual_gap_class": "ft09_l2_residual_world_model_mismatch_gap",
                    "verifier_accuracy": 0.166667,
                    "lookahead_fidelity": 0.347518,
                },
            ],
            "reproducibility_checksum": "sha256:4395",
        },
        exp4399.EXP4396_PATH: {
            "honest_verdict": "complete: clean_saturated_null_localizer",
            "localizer_compounds": False,
            "learning_curve": [
                {"train_corpus_size": 566, "held_out_localization_f1": 1.0},
                {"train_corpus_size": 5661, "held_out_localization_f1": 1.0},
            ],
            "no_learning_baseline": 0.096,
            "positive_control_passed": True,
            "compounding_delta_ci95": [0.0, 0.0],
            "fallback_to_ensemble": False,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:4396",
        },
        exp4399.EXP4397_PATH: {
            "honest_verdict": "complete: calibrated_multi_domain_contract_false",
            "detection_calibrated_multi_domain": False,
            "detection_by_domain": [
                {
                    "domain": "fover",
                    "detection_auroc": 0.918304,
                    "auroc_ci95": [0.909218, 0.926669],
                    "ece_lodo_calibrated": 0.129427,
                    "ece_uncalibrated": 0.122522,
                    "n": 8829,
                },
                {
                    "domain": "gap4_arc",
                    "detection_auroc": 0.963317,
                    "auroc_ci95": [0.921491, 0.990625],
                    "selection_headroom": 0.129,
                    "ece_lodo_calibrated": 0.005117,
                    "ece_uncalibrated": 0.01145,
                    "n": 28443,
                },
                {
                    "domain": "gsm8k",
                    "detection_auroc": 0.990196,
                    "auroc_ci95": [0.984555, 0.994967],
                    "n": 1600,
                },
            ],
            "domains_at_chance": [],
            "missing_verifier_gaps": [],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:4397",
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
    for rel_path, payload in _minimal_v406_payloads().items():
        _write_json(tmp_path / rel_path, payload)
    _write_json(
        tmp_path / exp4399.CAPSTONE_V405_PATH,
        {
            "honest_verdict": "complete: v405_fixture",
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
        "capstone_path": exp4399.CAPSTONE_V405_PATH,
        "capstone_verifier_is_oracle": False,
        "capstone_verifier_is_oracle_honored": True,
        "capstone_aggregation_propagates_oracle_stamp": True,
        "capstone_aggregation_uses_available_helper": True,
        "circular_moat_overclaim_fired": False,
        "flag_count": 0,
        "returncode": 0,
        "flags": [],
    }


def test_req_verify_4399_spec_declared() -> None:
    """REQ-VERIFY-4399: OpenSpec declares the .406 hygiene guard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4399",
        "SCENARIO-VERIFY-4399",
        "python/carnot/experiment_4399_registry_gaps_hygiene_gap4_guard.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4399.EXP4399_ARTIFACT_PATH,
        "blocked_<file>_unreadable",
        "GAP-FOVER-BIPRM-LOCALIZATION-untyped",
        "capstone_stamp_fix_durable",
        "CIRCULAR_MOAT_OVERCLAIM",
        "localizer win",
        "calibrated multi-domain false",
        "reproduced total of 34",
    ):
        assert marker in spec
    for field in exp4399.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp4399.FIELD_PRINCIPLES.values():
        assert principle in spec
    assert wrapper.main is exp4399.main


def test_scenario_verify_4399_ledgers_record_v406_truth(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4399: .406 outcomes reconcile ledgers without pruning."""

    _write_minimal_repo(tmp_path)
    outcomes = exp4399.load_v406_outcomes(tmp_path)
    gap_entries = exp4399.build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = exp4399.ensure_ledgers_record_v406(
        _minimal_registry(),
        _minimal_gaps_text(),
        _minimal_arc_registry(),
        _guard(),
        outcomes,
        gap_entries,
    )

    assert exp4399.registry_contains_v406(registry) is True
    assert exp4399.arc_registry_contains_v406(arc_registry) is True
    assert exp4399.gaps_contain_v406(gaps_text, gap_entries) is True
    assert summary["registries_reconciled"] is True
    assert summary["filled_gap_ids"] == []
    assert exp4399.GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED in gaps_text
    assert exp4399.GAP_4393_LOCALIZER_POSITION_TEMPLATE_CONFOUND in gaps_text

    gap4 = next(row for row in registry["verifiers"] if row["verifier_id"] == exp4399.GAP4_VERIFIER_ID)
    assert gap4["eval"]["exp4399_arc_reproducible_total_levels"] == 34
    assert gap4["eval"]["exp4399_new_levels_reproduced"] == 0
    assert gap4["eval"]["exp4399_detection_calibrated_multi_domain"] is False
    assert gap4["eval"]["exp4399_cross_domain_gap4_arc_auroc"] == 0.963317

    fover = next(row for row in registry["verifiers"] if row["verifier_id"] == exp4399.FOVER_VERIFIER_ID)
    assert fover["eval"]["exp4399_localizer_beats_ensemble_baseline"] is True
    assert fover["eval"]["exp4399_localizer_win_is_genuine"] is False
    assert fover["eval"]["exp4399_localizer_compounds"] is False
    assert fover["eval"]["exp4399_compounding_delta_ci95"] == [0.0, 0.0]

    assert arc_registry["latest_hygiene_4399"]["new_levels_reproduced"] == 0


def test_req_verify_4399_run_hygiene_writes_required_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-4399: terminal artifact exposes bare guard/stamp fields."""

    _write_minimal_repo(tmp_path)
    artifact = exp4399.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    exp4399.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gap4_regression_guard_passed"] is True
    assert artifact["capstone_stamp_fix_durable"] is True
    assert artifact["registries_reconciled"] is True
    assert artifact["v406_outcomes"]["localizer"]["localizer_beats_ensemble_baseline"] is True
    assert artifact["v406_outcomes"]["localizer_skeptic_proof"]["localizer_win_is_genuine"] is False

    written = json.loads((tmp_path / exp4399.EXP4399_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == artifact["honest_verdict"]
    assert "latest_hygiene_4399:" in (
        tmp_path / "ops" / "arc_solve_registry.yaml"
    ).read_text(encoding="utf-8")
    for field in exp4399.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4399.validate_artifact(malformed)


def test_req_verify_4399_blocks_unreadable_ledgers(tmp_path: Path) -> None:
    """REQ-VERIFY-4399: unreadable registries fail closed before mutation."""

    _write_minimal_repo(tmp_path)
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text("[]\n", encoding="utf-8")

    artifact = exp4399.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    assert artifact["honest_verdict"] == "blocked_arc_solve_registry_unreadable"
    assert artifact["gap4_regression_guard_passed"] is False
    assert artifact["capstone_stamp_fix_durable"] is False
    assert artifact["registries_reconciled"] is False


def test_req_verify_4399_defensive_branches_and_stamp_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4399: malformed inputs are recorded and stamp durability is audited."""

    assert exp4399._read_localizer(None, "missing")["available"] is False
    assert exp4399._read_skeptic(None, "missing")["available"] is False
    assert exp4399._read_compounds(None, "missing")["available"] is False
    assert exp4399._read_calibration(None, "missing")["available"] is False
    assert exp4399._read_e3_partial(None, "missing", exp4399.EXP4394_PATH, "rows", "gap")["rows"] == {}

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(
        exp4399.exp4388,
        "run_gap4_regression_guard",
        lambda root: {"regression_guard_passed": True, "root": str(root)},
    )
    assert exp4399.run_gap4_regression_guard(tmp_path)["regression_guard_passed"] is True

    outcomes = exp4399.load_v406_outcomes(tmp_path)
    graduated = json.loads(json.dumps(outcomes))
    graduated["localizer_skeptic_proof"]["localizer_win_is_genuine"] = True
    graduated_ids = {gap["gap_id"]: gap for gap in exp4399.build_gap_entries(graduated)}
    assert graduated_ids[exp4399.GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED]["status"].startswith(
        "filled"
    )

    chance = json.loads(json.dumps(outcomes))
    chance["cross_domain_calibration"]["domains_at_chance"] = [
        "bad",
        {"domain": "code", "detection_auroc": 0.51, "auroc_ci95": [0.49, 0.53], "n": 1000},
    ]
    assert "GAP-DETECTOR-CROSS-DOMAIN-code-4397" in [
        gap["gap_id"] for gap in exp4399.build_gap_entries(chance)
    ]

    malformed = json.loads(json.dumps(outcomes))
    malformed["localizer"]["missing_verifier_gaps"].insert(0, "bad")
    malformed["localizer"]["missing_verifier_gaps"].insert(1, {"gap_id": ""})
    malformed["arc_e3"]["deeper_fidelity"]["rows"]["lp85"]["offline_reproduced"] = True
    malformed["arc_e3"]["blocked_mechanics"]["rows"].pop("ka59")
    malformed_ids = [gap["gap_id"] for gap in exp4399.build_gap_entries(malformed)]
    assert exp4399.GAP_4392_FIRST_ERROR_GAP4_ARC in malformed_ids
    assert exp4399.GAP_E3_WORLD_MODEL_RULE_LP85_L6_4383 not in malformed_ids
    assert exp4399.GAP_E3_WORLD_MODEL_RULE_KA59_L2_4384 not in malformed_ids
    assert exp4399._domain_row(chance["cross_domain_calibration"], "missing") == {}

    arc_text = _minimal_arc_registry_text()
    patched = exp4399._patch_arc_registry_text(arc_text, outcomes)
    assert "latest_hygiene_4399:" in patched
    assert exp4399._patch_arc_registry_text(patched, outcomes) == patched

    fallback_repo = tmp_path / "fallback"
    _write_minimal_repo(fallback_repo)
    (fallback_repo / "ops" / "arc_solve_registry.yaml").write_text(
        _minimal_arc_registry_text()
        + "\nlatest_hygiene_4399:\n  artifact: wrong\n  new_levels_reproduced: 1\n",
        encoding="utf-8",
    )
    fallback_artifact = exp4399.run_hygiene(
        fallback_repo,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )
    assert fallback_artifact["registries_reconciled"] is True
    fallback_arc = yaml.safe_load(
        (fallback_repo / "ops" / "arc_solve_registry.yaml").read_text(encoding="utf-8")
    )
    assert fallback_arc["latest_hygiene_4399"]["artifact"] == exp4399.EXP4399_ARTIFACT_PATH

    registry: dict[str, Any] = {"verifiers": []}
    gaps = exp4399.build_gap_entries(outcomes)
    exp4399._ensure_gap4_eval(registry, _guard(), outcomes, gaps)
    assert exp4399.GAP4_VERIFIER_ID in [row["verifier_id"] for row in registry["verifiers"]]
    exp4399._ensure_v406_role({"verifiers": []}, outcomes, gaps)
    exp4399._ensure_fover_detector({"verifiers": []}, outcomes)
    assert exp4399._capstone_aggregation_propagates_oracle_stamp() is True
    assert exp4399._capstone_aggregation_uses_available_helper() is True

    missing_capstone = tmp_path / "missing_capstone"
    _write_minimal_repo(missing_capstone)
    (missing_capstone / exp4399.CAPSTONE_V405_PATH).unlink()
    missing_stamp = exp4399.verify_capstone_stamp_fix_durable(missing_capstone)
    assert missing_stamp["capstone_stamp_fix_durable"] is False
    assert "error" in missing_stamp

    completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=0,
        stdout=json.dumps({"reports": [{"flags": [], "flag_count": 0}], "flagged_count": 0}),
        stderr="",
    )
    monkeypatch.setattr(exp4399.subprocess, "run", lambda *args, **kwargs: completed)
    assert exp4399.verify_capstone_stamp_fix_durable(tmp_path)["capstone_stamp_fix_durable"] is True

    bad_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout=json.dumps(
            {"reports": [{"flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM"}]}]}
        ),
        stderr="",
    )
    monkeypatch.setattr(exp4399.subprocess, "run", lambda *args, **kwargs: bad_completed)
    assert exp4399.verify_capstone_stamp_fix_durable(tmp_path)[
        "circular_moat_overclaim_fired"
    ] is True

    invalid_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout="not json",
        stderr="broken",
    )
    monkeypatch.setattr(exp4399.subprocess, "run", lambda *args, **kwargs: invalid_completed)
    assert exp4399.verify_capstone_stamp_fix_durable(tmp_path)["flags"] == []


def test_req_verify_4399_artifact_schema_rejects_malformed_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4399: schema validation rejects malformed terminal artifacts."""

    _write_minimal_repo(tmp_path)
    artifact = exp4399.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    for field in (
        "preconditions_checked",
        "v406_outcomes",
        "registry_reconciliation",
        "gap4_regression_guard",
        "capstone_stamp_fix",
    ):
        malformed = dict(artifact)
        malformed[field] = []
        with pytest.raises(ValueError, match=field):
            exp4399.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4399.validate_artifact({**artifact, "honest_verdict": "not_terminal"})
    with pytest.raises(ValueError, match="gap4_regression_guard_passed"):
        exp4399.validate_artifact({**artifact, "gap4_regression_guard_passed": "true"})
    with pytest.raises(ValueError, match="capstone_stamp_fix_durable"):
        exp4399.validate_artifact({**artifact, "capstone_stamp_fix_durable": 1})
    with pytest.raises(ValueError, match="registries_reconciled"):
        exp4399.validate_artifact({**artifact, "registries_reconciled": None})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4399.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="random_seed"):
        exp4399.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="field_principles"):
        exp4399.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4399.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4399"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4399.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_verify_4399_results_entrypoint_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4399: required results entrypoint delegates to Exp 4399."""

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RESULTS_WRAPPER_PATH)])
    monkeypatch.setattr(exp4399, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp4399, "run_gap4_regression_guard", lambda _root: _guard())
    monkeypatch.setattr(exp4399, "verify_capstone_stamp_fix_durable", lambda _root: _stamp_report())
    python_root = str(REPO_ROOT / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_root])

    runpy.run_path(str(REPO_ROOT / RESULTS_WRAPPER_PATH), run_name="__main__")

    payload = json.loads((tmp_path / exp4399.EXP4399_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert payload["gap4_regression_guard_passed"] is True
    assert payload["capstone_stamp_fix_durable"] is True
    assert payload["registries_reconciled"] is True

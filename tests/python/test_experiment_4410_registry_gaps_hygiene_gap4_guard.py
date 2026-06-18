"""Tests for Exp 4410 registry/gaps hygiene, GAP-4 guard, and stamp durability.

Spec refs: REQ-VERIFY-4410, SCENARIO-VERIFY-4410.
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

from carnot import experiment_4410_registry_gaps_hygiene_gap4_guard as wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_gap4_guard_4410 as exp4410


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
RESULTS_WRAPPER_PATH = Path("results/experiment_4410_registry_gaps_hygiene_gap4_guard.py")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4410.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "eval": {"metric": "pass_at_2"},
                "registry_roles": [],
            },
            {
                "verifier_id": exp4410.FOVER_VERIFIER_ID,
                "domain": "math_reasoning",
                "version": 4,
                "kind": "ensemble",
                "eval": {"metric": "fover_dual_condition_auroc"},
                "registry_roles": [],
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
    return "# Verifier Gaps\n\n"


def _minimal_v407_payloads() -> dict[str, dict[str, Any]]:
    return {
        exp4410.EXP4403_PATH: {
            "honest_verdict": "complete: clean_powered_null_position_only_not_beaten",
            "localizer_genuinely_beats_position_only": False,
            "beats_position_only_baseline": False,
            "position_only_baseline_f1": 0.72,
            "template_family_holdout_drop": 0.0,
            "localization_f1_by_domain": {"fover": {"held_out_f1": 0.71}},
            "missing_verifier_gaps": [
                {
                    "gap_id": "GAP-4403-REAL-INTERVENTION-LOCALIZER-POSITION-ONLY",
                    "status": "open",
                    "failure_mode": "position_only_or_template_family_control_failed",
                    "missing_discriminator": "real multi-position intervention labels",
                    "candidate_design": "collect typed multi-step interventions",
                    "priority": "high",
                }
            ],
            "verifier_is_oracle": False,
            "n_traces": 1200,
            "reproducibility_checksum": "sha256:4403",
        },
        exp4410.EXP4404_PATH: {
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "gate_check_summary": "typed taxonomy gate did not pass",
            "gates_evaluated": [{"name": "taxonomy_gate", "passed": False}],
        },
        exp4410.EXP4405_PATH: {
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
                    "residual_failing_mechanic": "lp85_l6_reproduction_not_proven",
                    "mechanic_unit_tests_passed": 1,
                    "mechanic_unit_tests_total": 1,
                    "mechanic_unit_test_pass_rate": 1.0,
                    "verifier_accuracy": 0.833333,
                    "lookahead_fidelity": 0.833333,
                }
            ],
            "reproducibility_checksum": "sha256:4405",
        },
        exp4410.EXP4406_PATH: {
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
                    "residual_gap_class": "ar25_l2_plan_not_reproduced_after_register_test",
                    "register_unit_test_passed": True,
                    "register_unit_test_pass_rate": 1.0,
                    "verifier_accuracy": 0.90625,
                    "lookahead_fidelity": 0.733333,
                }
            ],
            "reproducibility_checksum": "sha256:4406",
        },
        exp4410.EXP4407_PATH: {
            "honest_verdict": "complete: clean_null_position_bound_or_saturated",
            "localizer_compounds": False,
            "active_vs_random_learning_curve": [
                {"label_count": 128, "active_f1": 0.7, "random_f1": 0.7}
            ],
            "compounding_delta_ci95": [0.0, 0.0],
            "positive_control_passed": True,
            "missing_verifier_gaps": [
                {
                    "gap_id": "GAP-4407-ACTIVE-LOCALIZER-POSITION-BOUND",
                    "status": "open",
                    "failure_mode": "clean null",
                    "missing_discriminator": "non-degenerate content features",
                    "candidate_design": "collect non-empty suffix redirects",
                    "priority": "high",
                }
            ],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:4407",
        },
        exp4410.EXP4408_PATH: {
            "honest_verdict": "complete: calibrated_multi_domain_contract_false_deconfounded",
            "detection_calibrated_multi_domain": False,
            "detection_by_domain": [
                {
                    "domain": "code_humaneval",
                    "n": 539,
                    "detection_auroc": 0.577374,
                    "auroc_ci95": [0.461255, 0.692756],
                }
            ],
            "domains_at_chance": ["code_humaneval"],
            "missing_verifier_gaps": [
                {
                    "gap_id": "GAP-4408-CODE-HUMANEVAL-DECONFOUNDED-DETECTOR-CHANCE",
                    "domain": "code_humaneval",
                    "status": "open",
                    "failure_mode": "CI95 includes chance",
                    "missing_discriminator": "domain-native oracle-distinct verifier feature",
                    "candidate_design": "add residual wrong-mode verifier score",
                    "priority": "high",
                }
            ],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:4408",
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
    for rel_path, payload in _minimal_v407_payloads().items():
        _write_json(tmp_path / rel_path, payload)
    _write_json(
        tmp_path / exp4410.CAPSTONE_V406_PATH,
        {
            "honest_verdict": "complete: v406_fixture",
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
        "capstone_path": exp4410.CAPSTONE_V406_PATH,
        "capstone_verifier_is_oracle": False,
        "capstone_verifier_is_oracle_honored": True,
        "capstone_aggregation_propagates_oracle_stamp": True,
        "capstone_aggregation_uses_available_helper": True,
        "circular_moat_overclaim_fired": False,
        "flag_count": 0,
        "returncode": 0,
        "flags": [],
    }


def test_req_verify_4410_spec_declared() -> None:
    """REQ-VERIFY-4410: OpenSpec declares the .407 hygiene guard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4410",
        "SCENARIO-VERIFY-4410",
        "python/carnot/experiment_4410_registry_gaps_hygiene_gap4_guard.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4410.EXP4410_ARTIFACT_PATH,
        "blocked_registry_unreadable",
        "GAP-FOVER-BIPRM-LOCALIZATION-untyped",
        "regression_guard_passed",
        "gaps_reconciled",
        "capstone_stamp_fix_durable",
        "CIRCULAR_MOAT_OVERCLAIM",
    ):
        assert marker in spec
    for field in exp4410.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp4410.FIELD_PRINCIPLES.values():
        assert principle["principle"] in spec
    assert wrapper.main is exp4410.main


def test_scenario_verify_4410_ledgers_record_v407_truth(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4410: .407 outcomes reconcile ledgers without pruning."""

    _write_minimal_repo(tmp_path)
    outcomes = exp4410.load_v407_outcomes(tmp_path)
    gap_entries = exp4410.build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = exp4410.ensure_ledgers_record_v407(
        _minimal_registry(),
        _minimal_gaps_text(),
        _minimal_arc_registry(),
        _guard(),
        outcomes,
        gap_entries,
    )

    assert exp4410.registry_contains_v407(registry) is True
    assert exp4410.arc_registry_contains_v407(arc_registry) is True
    assert exp4410.gaps_contain_v407(gaps_text, gap_entries) is True
    yaml.safe_load(gaps_text)
    assert summary["registries_reconciled"] is True
    assert exp4410.GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED in summary["gaps_reconciled"]
    assert "GAP-4408-CODE-HUMANEVAL-DECONFOUNDED-DETECTOR-CHANCE" in gaps_text

    gap4 = next(row for row in registry["verifiers"] if row["verifier_id"] == exp4410.GAP4_VERIFIER_ID)
    assert gap4["eval"]["exp4410_arc_reproducible_total_levels"] == 34
    assert gap4["eval"]["exp4410_new_levels_reproduced"] == 0
    assert gap4["eval"]["exp4410_detection_calibrated_multi_domain"] is False
    assert gap4["eval"]["exp4410_code_humaneval_detection_auroc"] == 0.577374

    fover = next(row for row in registry["verifiers"] if row["verifier_id"] == exp4410.FOVER_VERIFIER_ID)
    assert fover["eval"]["exp4410_localizer_genuinely_beats_position_only"] is False
    assert fover["eval"]["exp4410_localizer_compounds"] is False
    assert fover["eval"]["exp4410_compounding_delta_ci95"] == [0.0, 0.0]

    assert arc_registry["latest_hygiene_4410"]["new_levels_reproduced"] == 0


def test_req_verify_4410_run_hygiene_writes_required_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-4410: terminal artifact exposes bare guard/stamp fields."""

    _write_minimal_repo(tmp_path)
    artifact = exp4410.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    exp4410.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["capstone_stamp_fix_durable"] is True
    assert artifact["gaps_reconciled"]
    assert artifact["v407_outcomes"]["localizer_deconfound"]["localizer_genuinely_beats_position_only"] is False
    assert artifact["availability_report"]["missing_upstream_artifacts"] == []

    written = json.loads((tmp_path / exp4410.EXP4410_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == artifact["honest_verdict"]
    assert "latest_hygiene_4410:" in (
        tmp_path / "ops" / "arc_solve_registry.yaml"
    ).read_text(encoding="utf-8")
    for field in exp4410.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4410.validate_artifact(malformed)

    stale_arc_repo = tmp_path / "stale_arc"
    _write_minimal_repo(stale_arc_repo)
    (stale_arc_repo / "ops" / "arc_solve_registry.yaml").write_text(
        _minimal_arc_registry_text()
        + "\nlatest_hygiene_4410:\n  artifact: stale\n  new_levels_reproduced: 1\n",
        encoding="utf-8",
    )
    stale_artifact = exp4410.run_hygiene(
        stale_arc_repo,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )
    assert stale_artifact["gaps_reconciled"]
    stale_arc = yaml.safe_load(
        (stale_arc_repo / "ops" / "arc_solve_registry.yaml").read_text(encoding="utf-8")
    )
    assert stale_arc["latest_hygiene_4410"]["artifact"] == exp4410.EXP4410_ARTIFACT_PATH

    list_arc_repo = tmp_path / "list_arc"
    _write_minimal_repo(list_arc_repo)
    (list_arc_repo / "ops" / "arc_solve_registry.yaml").write_text("[]\n", encoding="utf-8")
    preflight_ok = {
        "ok": True,
        "blocked_file": None,
        "files": {
            "verifier_registry": {"readable": True},
            "verifier_gaps": {"readable": True},
            "arc_solve_registry": {"readable": True},
        },
    }
    original_check = exp4410.check_preconditions
    try:
        exp4410.check_preconditions = lambda _root: preflight_ok  # type: ignore[assignment]
        list_artifact = exp4410.run_hygiene(
            list_arc_repo,
            gap4_guard_runner=lambda _: _guard(),
            capstone_stamp_runner=lambda _: _stamp_report(),
        )
    finally:
        exp4410.check_preconditions = original_check  # type: ignore[assignment]
    assert list_artifact["preconditions_checked"] == preflight_ok


def test_req_verify_4410_blocks_unreadable_registry_without_mutation(tmp_path: Path) -> None:
    """REQ-VERIFY-4410: yaml.safe_load failures fail closed before mutation."""

    _write_minimal_repo(tmp_path)
    gaps_path = tmp_path / "ops" / "verifier_gaps.md"
    original_gaps = "**not yaml\n"
    gaps_path.write_text(original_gaps, encoding="utf-8")

    artifact = exp4410.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    assert artifact["honest_verdict"] == "blocked_registry_unreadable"
    assert artifact["regression_guard_passed"] is False
    assert artifact["capstone_stamp_fix_durable"] is False
    assert artifact["gaps_reconciled"] == []
    assert artifact["preconditions_checked"]["blocked_file"] == "verifier_gaps"
    assert gaps_path.read_text(encoding="utf-8") == original_gaps


def test_req_verify_4410_missing_upstreams_and_defensive_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4410: missing upstreams become gaps and stamp durability is audited."""

    _write_minimal_repo(tmp_path)
    assert exp4410._yaml_parse_check(tmp_path, "bad", exp4410.ARC_REGISTRY_PATH, True)[
        "readable"
    ] is True
    (tmp_path / exp4410.ARC_REGISTRY_PATH).write_text("[]\n", encoding="utf-8")
    non_mapping = exp4410._yaml_parse_check(tmp_path, "bad", exp4410.ARC_REGISTRY_PATH, True)
    assert non_mapping["readable"] is False
    assert non_mapping["error"] == "top-level YAML is not a mapping"
    (tmp_path / exp4410.ARC_REGISTRY_PATH).write_text(_minimal_arc_registry_text(), encoding="utf-8")

    assert exp4410._read_localizer(None, "missing")["available"] is False
    assert exp4410._read_e3_partial(None, "missing", exp4410.EXP4405_PATH, "rows", "gap")[
        "rows"
    ] == {}
    assert exp4410._read_compounds(None, "missing")["available"] is False
    assert exp4410._read_calibration(None, "missing")["available"] is False

    (tmp_path / exp4410.EXP4404_PATH).unlink()
    outcomes = exp4410.load_v407_outcomes(tmp_path)
    gap_ids = [gap["gap_id"] for gap in exp4410.build_gap_entries(outcomes)]
    assert exp4410.GAP_UPSTREAM_MISSING_4404 in gap_ids

    generated_chance = json.loads(json.dumps(outcomes))
    generated_chance["calibration_repair"]["domains_at_chance"] = ["code_humaneval"]
    generated_chance["calibration_repair"]["missing_verifier_gaps"].append(
        {"gap_id": "GAP-4408-CODE-HUMANEVAL-DETECTOR-CHANCE"}
    )
    assert "GAP-4408-CODE-HUMANEVAL-DETECTOR-CHANCE" in [
        gap["gap_id"] for gap in exp4410.build_gap_entries(generated_chance)
    ]

    entries: dict[str, dict[str, Any]] = {}
    exp4410._add_upstream_gap(entries, {"gap_id": ""}, "ignored")
    assert entries == {}
    exp4410._add_arc_gap_entries(entries, {"available": False, "error": "missing"}, "4499", "x.json")
    assert "GAP-4410-MISSING-UPSTREAM-4499" in entries
    exp4410._add_arc_gap_entries(
        entries,
        {
            "available": True,
            "rows": {
                "bad": "row",
                "done": {"offline_reproduced": True, "residual_gap_class": "filled"},
                "empty": {"offline_reproduced": False},
            },
        },
        "4498",
        "x.json",
    )
    assert not any(key.startswith("GAP-4498") for key in entries)

    marked = exp4410._replace_yaml_safe_marked_block("# base\n", "marker", "### one\n")
    replaced = exp4410._replace_yaml_safe_marked_block(marked, "marker", "### two\n")
    assert "### two" in replaced
    assert "### one" not in replaced

    registry: dict[str, Any] = {"verifiers": []}
    exp4410._ensure_gap4_eval(registry, _guard(), outcomes, exp4410.build_gap_entries(outcomes))
    assert exp4410.GAP4_VERIFIER_ID in [row["verifier_id"] for row in registry["verifiers"]]
    exp4410._ensure_v407_role({"verifiers": []}, outcomes, [])
    fover_registry: dict[str, Any] = {"verifiers": []}
    exp4410._ensure_fover_detector(fover_registry, outcomes)
    assert exp4410.FOVER_VERIFIER_ID in [row["verifier_id"] for row in fover_registry["verifiers"]]
    assert exp4410._patch_arc_registry_text("latest_hygiene_4410:\n  artifact: old\n", outcomes).startswith(
        "latest_hygiene_4410:"
    )

    monkeypatch.setattr(
        exp4410.exp4399,
        "run_gap4_regression_guard",
        lambda root: {"regression_guard_passed": True, "root": str(root)},
    )
    assert exp4410.run_gap4_regression_guard(tmp_path)["regression_guard_passed"] is True
    assert exp4410._domain_row(outcomes["calibration_repair"], "missing") == {}
    assert exp4410._capstone_aggregation_propagates_oracle_stamp() is True
    assert exp4410._capstone_aggregation_uses_available_helper() is True

    missing_capstone = tmp_path / "missing_capstone"
    _write_minimal_repo(missing_capstone)
    (missing_capstone / exp4410.CAPSTONE_V406_PATH).unlink()
    missing_stamp = exp4410.verify_capstone_stamp_fix_durable(missing_capstone)
    assert missing_stamp["capstone_stamp_fix_durable"] is False
    assert "error" in missing_stamp

    completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=0,
        stdout=json.dumps({"reports": [{"flags": [], "flag_count": 0}], "flagged_count": 0}),
        stderr="",
    )
    monkeypatch.setattr(exp4410.subprocess, "run", lambda *args, **kwargs: completed)
    assert exp4410.verify_capstone_stamp_fix_durable(tmp_path)["capstone_stamp_fix_durable"] is True

    bad_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout=json.dumps({"reports": [{"flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM"}]}]}),
        stderr="",
    )
    monkeypatch.setattr(exp4410.subprocess, "run", lambda *args, **kwargs: bad_completed)
    assert exp4410.verify_capstone_stamp_fix_durable(tmp_path)["circular_moat_overclaim_fired"] is True

    invalid_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout="not json",
        stderr="broken",
    )
    monkeypatch.setattr(exp4410.subprocess, "run", lambda *args, **kwargs: invalid_completed)
    assert exp4410.verify_capstone_stamp_fix_durable(tmp_path)["flags"] == []


def test_req_verify_4410_artifact_schema_rejects_malformed_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4410: schema validation rejects malformed terminal artifacts."""

    _write_minimal_repo(tmp_path)
    artifact = exp4410.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    for field in (
        "preconditions_checked",
        "v407_outcomes",
        "registry_reconciliation",
        "gap4_regression_guard",
        "capstone_stamp_fix",
        "availability_report",
    ):
        malformed = dict(artifact)
        malformed[field] = []
        with pytest.raises(ValueError, match=field):
            exp4410.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4410.validate_artifact({**artifact, "honest_verdict": "not_terminal"})
    with pytest.raises(ValueError, match="regression_guard_passed"):
        exp4410.validate_artifact({**artifact, "regression_guard_passed": "true"})
    with pytest.raises(ValueError, match="capstone_stamp_fix_durable"):
        exp4410.validate_artifact({**artifact, "capstone_stamp_fix_durable": 1})
    with pytest.raises(ValueError, match="gaps_reconciled"):
        exp4410.validate_artifact({**artifact, "gaps_reconciled": {}})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4410.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="random_seed"):
        exp4410.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="field_principles"):
        exp4410.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4410.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4410"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4410.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_verify_4410_results_entrypoint_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4410: required results entrypoint delegates to Exp 4410."""

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RESULTS_WRAPPER_PATH)])
    monkeypatch.setattr(exp4410, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp4410, "run_gap4_regression_guard", lambda _root: _guard())
    monkeypatch.setattr(exp4410, "verify_capstone_stamp_fix_durable", lambda _root: _stamp_report())
    python_root = str(REPO_ROOT / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_root])

    runpy.run_path(str(REPO_ROOT / RESULTS_WRAPPER_PATH), run_name="__main__")

    payload = json.loads((tmp_path / exp4410.EXP4410_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert payload["regression_guard_passed"] is True
    assert payload["capstone_stamp_fix_durable"] is True
    assert payload["gaps_reconciled"]

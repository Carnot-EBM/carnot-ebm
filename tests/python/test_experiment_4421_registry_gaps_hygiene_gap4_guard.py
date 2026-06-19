"""Tests for Exp 4421 registry/gaps hygiene and GAP-4 guard.

Spec refs: REQ-VERIFY-4421, SCENARIO-VERIFY-4421.
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

from carnot import experiment_4421_registry_gaps_hygiene_gap4_guard as exp4421


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
RESULTS_WRAPPER_PATH = Path("results/experiment_4421_registry_gaps_hygiene_gap4_guard.py")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4421.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "kind": "process_verifier",
                "eval": {},
                "registry_roles": [],
            },
            {
                "verifier_id": exp4421.FOVER_VERIFIER_ID,
                "domain": "math_reasoning",
                "kind": "ensemble",
                "eval": {},
                "registry_roles": [],
            },
        ]
    }


def _minimal_arc_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated": "2026-06-18",
        "games": [{"game": "ka59", "reproducibility": "reproduced", "levels_reproduced": 1}],
        "reproducible_total_levels": 34,
        "reproducible_total_games": 17,
    }


def _minimal_gaps_text() -> str:
    return (
        "# Verifier Gaps\n\n"
        "### GAP-FOVER-BIPRM-LOCALIZATION-untyped: earliest causal error\n"
        "- status: open\n"
        "- evidence: prior localizer misses earliest error.\n"
        "- failure mode: downstream consequence wins.\n"
        "- missing discriminator: first causal break signal.\n"
        "- candidate design: typed first-error labels.\n"
        "- priority: medium\n"
    )


def _minimal_v408_payloads() -> dict[str, dict[str, Any]]:
    return {
        exp4421.EXP4414_PATH: {
            "honest_verdict": "complete_config_rule_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "config_win_rules_grounded": [
                {
                    "game": "ka59",
                    "predicate": "editable_count_4_equals_reference_count_4_32",
                    "tier": 2,
                    "fires_on_win": True,
                    "false_positive_rate": 0.0,
                    "literal_hardcode": False,
                }
            ],
            "per_target_scorecard": [
                {
                    "game": "ka59",
                    "grounding_tier": 2,
                    "offline_reproduced": False,
                    "prior_best_level": 1,
                    "new_reproduced_level": 1,
                    "search_blocker": "no_registered_next_level_config_adapter",
                    "win_rule_predicate": "editable_count_4_equals_reference_count_4_32",
                },
                {
                    "game": "bp35",
                    "grounding_tier": 0,
                    "offline_reproduced": False,
                    "prior_best_level": 0,
                    "new_reproduced_level": 0,
                    "search_blocker": "blocked_local_model_unavailable",
                },
            ],
            "preconditions_checked": {"trm_training_stood_down": True},
            "random_seed": 4414,
            "reproducibility_checksum": "sha256:4414",
            "verifier_is_oracle": True,
        },
        exp4421.EXP4415_PATH: {
            "honest_verdict": "complete_e3_adaptive_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "per_target_scorecard": [
                {
                    "game": "ar25",
                    "target_level": 2,
                    "prior_best_level": 1,
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "adaptive_tests_passed": 1,
                    "adaptive_tests_total": 2,
                    "residual_failing_behavior": "ar25_l2_hidden_undo_stack_state_not_visible",
                    "verifier_accuracy": 0.5,
                    "lookahead_fidelity": 0.5,
                },
                {
                    "game": "tn36",
                    "target_level": 8,
                    "prior_best_level": 7,
                    "new_reproduced_level": 7,
                    "offline_reproduced": False,
                    "adaptive_tests_passed": 1,
                    "adaptive_tests_total": 2,
                    "residual_failing_behavior": "tn36_l8_palette_population_wrong",
                    "verifier_accuracy": 0.875,
                    "lookahead_fidelity": 0.875,
                },
            ],
            "preconditions_checked": {"trm_training_stood_down": True},
            "random_seed": 4415,
            "reproducibility_checksum": "sha256:4415",
            "verifier_is_oracle": True,
        },
        exp4421.EXP4416_PATH: {
            "honest_verdict": "complete: clean_powered_null_position_only_not_beaten",
            "hidden_state_localizer_has_nonposition_signal": False,
            "position_only_baseline_f1": 1.0,
            "missing_verifier_gaps": [
                {
                    "gap_id": "GAP-FOVER-HIDDEN-STATE-LOCALIZATION-POSITION-SATURATED",
                    "status": "open",
                    "parent_gap": exp4421.GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED,
                    "failure_mode": "hidden-state probe tied the position-only baseline",
                    "missing_discriminator": "non-position first-error signal",
                    "candidate_design": "collect non-first-position FoVer traces",
                    "priority": "medium",
                }
            ],
            "preconditions_checked": [{"resource": "trm_training_stand_down", "available": True}],
            "random_seed": 4416,
            "reproducibility_checksum": "sha256:4416",
            "verifier_is_oracle": False,
        },
        exp4421.EXP4417_PATH: {
            "honest_verdict": "complete: sovereign_gap4_local_gate_holds_flat_cov_0.2333_fires_0_lost_0",
            "sovereign_gap4_gate_holds": True,
            "pass2_vs_vote": {
                "vote_pass2": 0.4516,
                "gated_pass2": 0.4516,
                "delta": 0.0,
                "graded_gate_fires": 0,
                "pass2_vote_wins_lost": 0,
            },
            "local_generator_coverage": 0.2333,
            "preconditions_checked": [{"resource": "trm_training_stood_down", "available": True}],
            "random_seed": 12345,
            "reproducibility_checksum": "sha256:4417",
            "verifier_is_oracle": True,
        },
        exp4421.EXP4418_PATH: {
            "honest_verdict": "blocked_local_model_unavailable",
            "config_rule_vocabulary": [],
            "config_rule_vocabulary_transfers": False,
            "preconditions_checked": {
                "trm_training_stood_down": True,
                "local_model_server": {"status": "blocked_local_model_unavailable"},
                "grounded_rules": {"count": 4, "games": ["ka59", "sc25", "tn36", "tr87"]},
            },
            "random_seed": 4418,
            "reproducibility_checksum": "sha256:4418",
            "verifier_is_oracle": False,
        },
        exp4421.EXP4419_PATH: {
            "honest_verdict": "complete: clean_null_steered_confidence_does_not_rescue_code_detector",
            "detection_calibrated_multi_domain": False,
            "domains_at_chance": ["code_humaneval"],
            "detection_by_domain": [
                {
                    "domain": "code_humaneval",
                    "n": 539,
                    "baseline_verifier_auroc": 0.577374,
                    "detection_auroc": 0.60191,
                    "auroc_ci95": [0.48913, 0.70483],
                    "steered_confidence_added_auroc": 0.024536,
                }
            ],
            "missing_verifier_gaps": [
                {
                    "gap_id": "GAP-4419-CODE-HUMANEVAL-STEERCONF-DETECTOR-CHANCE",
                    "domain": "code_humaneval",
                    "status": "open",
                    "failure_mode": "CI95 includes chance",
                    "missing_discriminator": "domain-native oracle-distinct code feature",
                    "candidate_design": "build domain-specific feature",
                    "priority": "high",
                }
            ],
            "preconditions_checked": [{"resource": "trm_training_stand_down", "available": True}],
            "random_seed": 4419,
            "reproducibility_checksum": "sha256:4419",
            "verifier_is_oracle": False,
        },
    }


def _write_minimal_repo(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ops/verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops/arc_solve_registry.yaml").write_text(
        yaml.safe_dump(_minimal_arc_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops/verifier_gaps.md").write_text(_minimal_gaps_text(), encoding="utf-8")
    (tmp_path / "scripts/adversarial_verify.py").write_text("# fixture\n", encoding="utf-8")
    for rel_path, payload in _minimal_v408_payloads().items():
        _write_json(tmp_path / rel_path, payload)
    _write_json(
        tmp_path / exp4421.CAPSTONE_PATH,
        {
            "honest_verdict": "complete: v407_fixture",
            "verifier_is_oracle": False,
            "verifier_is_oracle_honored": True,
        },
    )
    _write_json(
        tmp_path / exp4421.GAP4_EXECUTION_RESULT_PATH,
        {
            "gates": {
                "headroom_recovered": 4,
                "selection_beats_vote": True,
                "vote_wins_lost": 0,
            },
            "rankers": {
                "GAP4_GATED": {"pass@2": 0.5806},
                "TRM_VOTE": {"pass@2": 0.4516},
            },
            "verifier_is_oracle": False,
        },
    )


def _guard() -> dict[str, Any]:
    return {
        "gap4_execution_guard_passed": True,
        "regression_guard_passed": True,
        "arc_oracle_distinct_verifier_beats_vote": True,
        "current": {"gated_pass2": 0.5806, "vote_pass2": 0.4516},
    }


def _stamp() -> dict[str, Any]:
    return {
        "capstone_stamp_fix_durable": True,
        "capstone_verifier_is_oracle": False,
        "capstone_verifier_is_oracle_honored": True,
        "capstone_aggregation_propagates_oracle_stamp": True,
        "capstone_aggregation_uses_available_helper": True,
        "circular_moat_overclaim_fired": False,
        "flag_count": 0,
        "flags": [],
        "returncode": 0,
    }


def test_req_verify_4421_spec_declared() -> None:
    """REQ-VERIFY-4421: OpenSpec declares the .408 hygiene guard."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4421",
        "SCENARIO-VERIFY-4421",
        "python/carnot/experiment_4421_registry_gaps_hygiene_gap4_guard.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4421.EXP4421_ARTIFACT_PATH,
        "regression_guard_passed",
        "registry_reconciliation",
        "robust aggregate-available",
        "verifier_gaps.md` as a markdown text ledger",
    ):
        assert marker in spec
    for principle in exp4421.FIELD_PRINCIPLES.values():
        assert principle["principle"] in spec


def test_scenario_verify_4421_ledgers_record_v408_truth(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4421: .408 outcomes reconcile ledgers without pruning."""

    _write_minimal_repo(tmp_path)
    outcomes = exp4421.load_v408_outcomes(tmp_path)
    gap_entries = exp4421.build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = exp4421.ensure_ledgers_record_v408(
        _minimal_registry(),
        _minimal_gaps_text(),
        _minimal_arc_registry(),
        _guard(),
        outcomes,
        gap_entries,
    )

    assert exp4421.registry_contains_v408(registry) is True
    assert exp4421.arc_registry_contains_v408(arc_registry) is True
    assert exp4421.gaps_contain_v408(gaps_text, gap_entries) is True
    assert yaml.safe_load(yaml.safe_dump(registry))
    assert summary["registries_reconciled"] is True
    assert exp4421.GAP_4414_CONFIG_RULE_KA59 in summary["filled_gap_ids"]
    assert exp4421.GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED in summary["sharpened_gap_ids"]
    assert "GAP-4419-CODE-HUMANEVAL-STEERCONF-DETECTOR-CHANCE" in summary["newly_logged_gap_ids"]
    assert "exp4421-gap-4419-code-humaneval-steerconf-detector-chance:start" in gaps_text

    gap4 = next(row for row in registry["verifiers"] if row["verifier_id"] == exp4421.GAP4_VERIFIER_ID)
    assert gap4["eval"]["eval_exp_4421"] == exp4421.EXP4421_ARTIFACT_PATH
    assert gap4["eval"]["exp4421_sovereign_gap4_gate_holds"] is True
    assert gap4["eval"]["exp4421_graded_gate_fires"] == 0
    assert gap4["eval"]["exp4421_arc_reproducible_total_levels"] == 34

    fover = next(row for row in registry["verifiers"] if row["verifier_id"] == exp4421.FOVER_VERIFIER_ID)
    assert fover["eval"]["exp4421_hidden_state_localizer_has_nonposition_signal"] is False


def test_req_verify_4421_run_hygiene_writes_required_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-4421: terminal artifact exposes required bare guard fields."""

    _write_minimal_repo(tmp_path)
    artifact = exp4421.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp(),
    )

    exp4421.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: registry_gaps_reconciled_guard_passed"
    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciliation"]["registries_reconciled"] is True
    assert artifact["preconditions_checked"]["trm_training_stood_down"] is True
    assert artifact["availability_report"]["missing_upstream_artifacts"] == []
    assert artifact["v408_outcomes"]["config_rule_induction"]["config_win_rules_grounded"][0]["game"] == "ka59"

    written = json.loads((tmp_path / exp4421.EXP4421_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert "latest_hygiene_4421:" in (
        tmp_path / "ops/arc_solve_registry.yaml"
    ).read_text(encoding="utf-8")
    for field in exp4421.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4421.validate_artifact(malformed)


def test_req_verify_4421_missing_upstream_is_reported_not_global_block(tmp_path: Path) -> None:
    """REQ-VERIFY-4421: robust availability reports a missing .408 artifact per axis."""

    _write_minimal_repo(tmp_path)
    (tmp_path / exp4421.EXP4418_PATH).unlink()

    artifact = exp4421.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp(),
    )

    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciliation"]["registries_reconciled"] is True
    assert artifact["availability_report"]["missing_upstream_artifacts"]
    assert "GAP-4421-MISSING-UPSTREAM-4418" in artifact["registry_reconciliation"]["newly_logged_gap_ids"]


def test_req_verify_4421_blocks_bad_yaml_registry_but_not_markdown_gaps(tmp_path: Path) -> None:
    """REQ-VERIFY-4421: markdown gaps load as text; bad YAML registry blocks."""

    _write_minimal_repo(tmp_path)
    gaps_check = exp4421.check_preconditions(tmp_path)
    assert gaps_check["files"]["verifier_gaps"]["readable"] is True
    assert gaps_check["files"]["verifier_gaps"]["markdown_text"] is True

    original_registry = "*bad\n"
    (tmp_path / exp4421.REGISTRY_PATH).write_text(original_registry, encoding="utf-8")
    artifact = exp4421.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp(),
    )

    assert artifact["honest_verdict"] == "blocked_registry_gaps_reconciliation_unavailable"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_reconciliation"] == {}
    assert artifact["preconditions_checked"]["blocked_file"] == "verifier_registry"
    assert (tmp_path / exp4421.REGISTRY_PATH).read_text(encoding="utf-8") == original_registry


def test_req_verify_4421_schema_and_guard_defensive_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4421: malformed artifacts and guard failures are rejected."""

    _write_minimal_repo(tmp_path)
    artifact = exp4421.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp(),
    )

    for field in ("registry_reconciliation", "preconditions_checked", "availability_report", "v408_outcomes"):
        malformed = dict(artifact)
        malformed[field] = []
        with pytest.raises(ValueError, match=field):
            exp4421.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal"):
        exp4421.validate_artifact({**artifact, "honest_verdict": "complete-but-not-prefixed"})
    with pytest.raises(ValueError, match="regression_guard_passed"):
        exp4421.validate_artifact({**artifact, "regression_guard_passed": "true"})
    with pytest.raises(ValueError, match="random_seed"):
        exp4421.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4421.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="field_principles"):
        exp4421.validate_artifact({**artifact, "field_principles": {}})

    failed = exp4421.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: {"regression_guard_passed": False},
        capstone_stamp_runner=lambda _: _stamp(),
    )
    assert failed["honest_verdict"] == "complete: registry_gaps_reconciled_guard_failed"
    assert failed["regression_guard_passed"] is False

    completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=0,
        stdout=json.dumps({"reports": [{"flags": [], "flag_count": 0}], "flagged_count": 0}),
        stderr="",
    )
    monkeypatch.setattr(exp4421.subprocess, "run", lambda *args, **kwargs: completed)
    assert exp4421.verify_capstone_stamp_fix_durable(tmp_path)["capstone_stamp_fix_durable"] is True

    bad_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout=json.dumps({"reports": [{"flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM"}]}]}),
        stderr="",
    )
    monkeypatch.setattr(exp4421.subprocess, "run", lambda *args, **kwargs: bad_completed)
    assert exp4421.verify_capstone_stamp_fix_durable(tmp_path)["circular_moat_overclaim_fired"] is True

    with pytest.raises(ValueError, match="spec_refs"):
        exp4421.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4421"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4421.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_verify_4421_defensive_helper_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4421: helper edge cases fail closed without fabrication."""

    _write_minimal_repo(tmp_path)
    list_json = tmp_path / "results/list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    assert exp4421._load_optional_json(tmp_path, "results/list.json")[1] == "top-level JSON is not an object"
    non_mapping_yaml = tmp_path / "list.yaml"
    non_mapping_yaml.write_text("[]\n", encoding="utf-8")
    assert exp4421._yaml_parse_check(tmp_path, "bad", "list.yaml")["error"] == "top-level YAML is not a mapping"
    assert exp4421._bool(None, "x") is None
    assert exp4421._int(None, "x") == 0
    assert exp4421._float(None, "x") is None
    assert exp4421._float({"x": "bad"}, "x") is None
    assert exp4421._str(None, "x") == ""
    assert exp4421._list(None, "x") == []
    assert exp4421._yaml_parse_check(tmp_path, "bad", "missing.yaml")["readable"] is False
    assert exp4421._markdown_text_check(tmp_path, "bad", "missing.md")["readable"] is False

    assert exp4421._read_config_rule(None, "missing")["available"] is False
    assert exp4421._read_adaptive_e3(None, "missing")["available"] is False
    assert exp4421._read_hidden_state(None, "missing")["available"] is False
    assert exp4421._read_sovereign_gap4(None, "missing")["available"] is False
    assert exp4421._read_steerconf(None, "missing")["available"] is False
    assert exp4421._read_vocab(None, "missing")["available"] is False
    assert exp4421._trm_training_stood_down({"resource": "trm_training_stand_down"}) is True
    assert exp4421._trm_training_stood_down([{"resource": "nope"}]) is False

    entries: dict[str, dict[str, Any]] = {}
    exp4421._add_upstream_gap(entries, {"gap_id": ""}, "ignored")
    assert entries == {}

    outcomes = exp4421.load_v408_outcomes(tmp_path)
    noisy = json.loads(json.dumps(outcomes))
    noisy["config_rule_induction"]["config_win_rules_grounded"].insert(0, "bad")
    noisy["config_rule_induction"]["per_target_scorecard"].insert(0, "bad")
    noisy["adaptive_e3_repair"]["per_target_scorecard"].insert(0, "bad")
    noisy["steerconf_code_detection"]["missing_verifier_gaps"] = []
    gap_ids = [gap["gap_id"] for gap in exp4421.build_gap_entries(noisy)]
    assert "GAP-4419-CODE-HUMANEVAL-STEERCONF-DETECTOR-CHANCE" in gap_ids
    assert exp4421._domain_row(outcomes["steerconf_code_detection"], "missing") == {}

    registry: dict[str, Any] = {"verifiers": []}
    gap_entries = exp4421.build_gap_entries(outcomes)
    exp4421._ensure_gap4_eval(registry, _guard(), outcomes, gap_entries)
    exp4421._ensure_fover_eval(registry, outcomes)
    assert exp4421.GAP4_VERIFIER_ID in [row["verifier_id"] for row in registry["verifiers"]]
    assert exp4421.FOVER_VERIFIER_ID in [row["verifier_id"] for row in registry["verifiers"]]

    invalid_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout="not json",
        stderr="broken",
    )
    monkeypatch.setattr(exp4421.subprocess, "run", lambda *args, **kwargs: invalid_completed)
    bad_report = exp4421._adversarial_report(tmp_path, exp4421.CAPSTONE_PATH)
    assert bad_report["reports"] == []
    assert exp4421._ranker_pass2({"rankers": []}, "GAP4_GATED") is None
    assert exp4421._ranker_pass2({"rankers": {"GAP4_GATED": []}}, "GAP4_GATED") is None
    assert exp4421._ranker_pass2({"rankers": {"GAP4_GATED": {"pass@2": "bad"}}}, "GAP4_GATED") is None
    assert exp4421._ranker_pass2({"rankers": {"GAP4_GATED": {"pass@2": 0.58}}}, "GAP4_GATED") == 0.58

    missing_guard_repo = tmp_path / "missing_guard"
    _write_minimal_repo(missing_guard_repo)
    (missing_guard_repo / exp4421.GAP4_EXECUTION_RESULT_PATH).unlink()
    assert exp4421.run_gap4_regression_guard(missing_guard_repo)["regression_guard_passed"] is False

    good_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=0,
        stdout=json.dumps({"reports": [{"flags": []}]}),
        stderr="",
    )
    monkeypatch.setattr(exp4421.subprocess, "run", lambda *args, **kwargs: good_completed)
    assert exp4421.run_gap4_regression_guard(tmp_path)["regression_guard_passed"] is True

    bad_gates_repo = tmp_path / "bad_gates"
    _write_minimal_repo(bad_gates_repo)
    _write_json(
        bad_gates_repo / exp4421.GAP4_EXECUTION_RESULT_PATH,
        {
            "gates": [],
            "rankers": {
                "GAP4_GATED": {"pass@2": 0.5806},
                "TRM_VOTE": {"pass@2": 0.4516},
            },
        },
    )
    assert exp4421.run_gap4_regression_guard(bad_gates_repo)["regression_guard_passed"] is False

    missing_capstone_repo = tmp_path / "missing_capstone"
    _write_minimal_repo(missing_capstone_repo)
    (missing_capstone_repo / exp4421.CAPSTONE_PATH).unlink()
    assert exp4421.verify_capstone_stamp_fix_durable(missing_capstone_repo)["capstone_stamp_fix_durable"] is False

    stale_arc_repo = tmp_path / "stale_arc"
    _write_minimal_repo(stale_arc_repo)
    (stale_arc_repo / exp4421.ARC_REGISTRY_PATH).write_text(
        yaml.safe_dump(
            {
                **_minimal_arc_registry(),
                "latest_hygiene_4421": {
                    "artifact": "stale",
                    "new_levels_reproduced": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    stale_artifact = exp4421.run_hygiene(
        stale_arc_repo,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp(),
    )
    assert stale_artifact["registry_reconciliation"]["registries_reconciled"] is True
    stale_arc = yaml.safe_load((stale_arc_repo / exp4421.ARC_REGISTRY_PATH).read_text(encoding="utf-8"))
    assert stale_arc["latest_hygiene_4421"]["artifact"] == exp4421.EXP4421_ARTIFACT_PATH

    list_arc_repo = tmp_path / "list_arc"
    _write_minimal_repo(list_arc_repo)
    (list_arc_repo / exp4421.ARC_REGISTRY_PATH).write_text("[]\n", encoding="utf-8")
    preflight_ok = {
        "ok": True,
        "blocked_file": None,
        "files": {
            "verifier_registry": {"readable": True},
            "verifier_gaps": {"readable": True, "markdown_text": True},
            "arc_solve_registry": {"readable": True},
        },
    }
    monkeypatch.setattr(exp4421, "check_preconditions", lambda _root: preflight_ok)
    list_artifact = exp4421.run_hygiene(
        list_arc_repo,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp(),
    )
    assert list_artifact["registry_reconciliation"]["arc_solve_registry_reconciled"] is True


def test_req_verify_4421_results_entrypoint_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4421: required results entrypoint delegates to Exp 4421."""

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RESULTS_WRAPPER_PATH)])
    monkeypatch.setattr(exp4421, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp4421, "run_gap4_regression_guard", lambda _root: _guard())
    monkeypatch.setattr(exp4421, "verify_capstone_stamp_fix_durable", lambda _root: _stamp())
    python_root = str(REPO_ROOT / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_root])

    runpy.run_path(str(REPO_ROOT / RESULTS_WRAPPER_PATH), run_name="__main__")

    payload = json.loads((tmp_path / exp4421.EXP4421_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert payload["regression_guard_passed"] is True
    assert payload["registry_reconciliation"]["registries_reconciled"] is True

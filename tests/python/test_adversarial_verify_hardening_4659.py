"""Tests for Exp 4659 adversarial_verify .429 hardening.

Spec refs: REQ-ARC-WMTE-4659,
SCENARIO-ARC-WMTE-4659-QD-RANDOM-MUTATION-ABLATION,
SCENARIO-ARC-WMTE-4659-VALUE-ROUTING-COST-CONTROL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
A1_PATH = REPO / "results" / "experiment_4652_value_routing_cost_fix_live.json"
A2_PATH = REPO / "results" / "experiment_4653_energy_fitness_qd_generation_live.json"


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _report_for_payload(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return av.verify_artifact(_write_payload(tmp_path, payload))


def _flag_kind(report: dict[str, Any], kind: str) -> list[dict[str, str]]:
    return [flag for flag in report["flags"] if flag["kind"] == kind]


def _flag_kinds(report: dict[str, Any], kinds: set[str]) -> list[dict[str, str]]:
    return [flag for flag in report["flags"] if flag["kind"] in kinds]


def _qd_overclaim_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4659_energy_fitness_qd_overclaim_fixture",
        "game": "tn36",
        "headline": "energy-fitness QD generation win: winner_generated and solve-rate up",
        "honest_verdict": "success: energy_fitness_qd_winner_generated_1",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "winner_generated": True,
        "winner_generated_count": 1,
        "live_solve_rate_qd": 0.25,
        "live_solve_rate_search_baseline": 0.0,
        "solve_rate_delta": 0.25,
        "first_win_rate_delta": 0.0,
        "qd_lift_ci": {"ci95": [0.10, 0.40], "point": 0.25},
        "random_mutation_ablation_passed": False,
        "random_seed": 4659,
        "reproducibility_checksum": "sha256:" + "a" * 64,
    }
    payload.update(overrides)
    return payload


def _value_routing_overclaim_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4659_value_routing_cost_overclaim_fixture",
        "game": "ar25",
        "headline": "value-routing cost-fixed live first-win up",
        "honest_verdict": "success: value_routing_cost_fixed_live_firstwin_up_1",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "live_first_win_rate_value_routed": 0.08,
        "live_first_win_rate_baseline": 0.04,
        "first_win_rate_delta": 0.04,
        "solve_rate_delta": 0.0,
        "value_weight_set": 0.30,
        "random_seed": 4659,
        "reproducibility_checksum": "sha256:" + "b" * 64,
    }
    payload.update(overrides)
    return payload


def test_req_arc_wmte_4659_spec_declares_qd_and_value_routing_guards() -> None:
    """REQ-ARC-WMTE-4659: OpenSpec declares both adversarial_verify guards."""

    from carnot import experiment_4659_adversarial_verify_hardening as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4659" in spec
    assert "REQ-ARC-WMTE-5169" in spec
    assert "SCENARIO-ARC-WMTE-4659-QD-RANDOM-MUTATION-ABLATION" in spec
    assert "SCENARIO-ARC-WMTE-4659-VALUE-ROUTING-COST-CONTROL" in spec
    assert "SCENARIO-ARC-WMTE-5169-QD-CITATION-SCOPE" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4659_qd_win_false_ablation_is_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4659-QD-RANDOM-MUTATION-ABLATION: false control flags."""

    report = _report_for_payload(tmp_path, _qd_overclaim_fixture())
    flags = _flag_kind(report, av.QD_WITHOUT_RANDOM_MUTATION_ABLATION_KIND)

    assert flags
    assert flags[0]["severity"] == "critical"
    assert "random-mutation/no-energy-fitness" in flags[0]["detail"]


def test_scenario_arc_wmte_4659_qd_win_omitted_ablation_warns_and_is_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4659-QD-RANDOM-MUTATION-ABLATION: omitted field warns."""

    artifact = _qd_overclaim_fixture()
    artifact.pop("random_mutation_ablation_passed")
    report = _report_for_payload(tmp_path, artifact)
    critical = _flag_kind(report, av.QD_WITHOUT_RANDOM_MUTATION_ABLATION_KIND)
    omitted = _flag_kind(report, av.QD_RANDOM_MUTATION_ABLATION_OMITTED_KIND)

    assert critical
    assert critical[0]["severity"] == "critical"
    assert omitted
    assert omitted[0]["severity"] == "warn"


def test_scenario_arc_wmte_4659_qd_passed_ablation_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4659-QD-RANDOM-MUTATION-ABLATION: passing control is clean."""

    report = _report_for_payload(
        tmp_path,
        _qd_overclaim_fixture(random_mutation_ablation_passed=True),
    )

    assert _flag_kind(report, av.QD_WITHOUT_RANDOM_MUTATION_ABLATION_KIND) == []
    assert _flag_kind(report, av.QD_RANDOM_MUTATION_ABLATION_OMITTED_KIND) == []


def test_scenario_arc_wmte_5169_qd_archive_citation_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5169-QD-CITATION-SCOPE: archive citations are not claims."""

    report = _report_for_payload(
        tmp_path,
        {
            "experiment": "experiment_5156_archive_472_activate_473",
            "honest_verdict": "complete_archive_472_closed_473_active_runtime_clean",
            "inference_substrate": av.AGGREGATION_SUBSTRATE,
            "duration_s": 0.032874,
            "milestone_archive_summary": [
                {
                    "experiment_id": "exp5154-energy-fitness-directed-exploration-v472",
                    "honest_verdict": (
                        "complete: "
                        "energy_fitness_qd_winning_trajectory_not_surfaced_reproducible_delta_0"
                    ),
                    "classification": "honest_null_generation_axis",
                    "winning_trajectory_surfaced": False,
                    "matched_control_winning_trajectory_surfaced": False,
                    "reproducible_levels_delta": 0,
                }
            ],
            "generation_axis_retirement_signal": {
                "third_consecutive_generation_axis_null": True,
                "current_energy_fitness_result": "honest_null",
                "allocation_read": (
                    "Generation-axis exploration-signal levers should not be re-run "
                    "without a new mechanism."
                ),
            },
            "a2_energy_qd_result": {
                "energy_qd_first_win": 0.0,
                "energy_qd_vs_naive_delta": 0.0,
                "included_in_headline": True,
                "reason": "energy_qd_real_non_degenerate_zero_lift_null",
            },
            "random_seed": 5169,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
    )

    assert _flag_kind(report, av.QD_RANDOM_MUTATION_ABLATION_OMITTED_KIND) == []
    assert _flag_kind(report, av.QD_WITHOUT_RANDOM_MUTATION_ABLATION_KIND) == []


def test_scenario_arc_wmte_5169_first_party_qd_claim_still_warns(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5169-QD-CITATION-SCOPE: own QD claims still require ablation."""

    report = _report_for_payload(
        tmp_path,
        {
            "experiment": "experiment_5169_first_party_qd_fixture",
            "game": "tn36",
            "headline": "energy-fitness QD live generation measurement",
            "honest_verdict": "complete: energy_fitness_qd_live_generation_measurement",
            "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
            "solve_provenance": "live_agent_self_discovery",
            "qd_arm_result": {"winning_trajectory_surfaced": False, "reached_level": 0},
            "random_seed": 5169,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
    )
    flags = _flag_kind(report, av.QD_RANDOM_MUTATION_ABLATION_OMITTED_KIND)

    assert flags
    assert flags[0]["severity"] == "warn"
    assert _flag_kind(report, av.QD_WITHOUT_RANDOM_MUTATION_ABLATION_KIND) == []


def test_scenario_arc_wmte_4659_value_routing_omitted_controls_warn_and_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4659-VALUE-ROUTING-COST-CONTROL: omitted controls flag."""

    report = _report_for_payload(tmp_path, _value_routing_overclaim_fixture())
    critical = _flag_kind(report, av.VALUE_ROUTING_WITHOUT_COST_CONTROL_KIND)
    omitted = _flag_kind(report, av.VALUE_ROUTING_COST_CONTROL_OMITTED_KIND)

    assert critical
    assert critical[0]["severity"] == "critical"
    assert omitted
    assert omitted[0]["severity"] == "warn"
    assert "per_node_feature_cost_ms" in critical[0]["detail"]
    assert "sim_timed_out=false" in critical[0]["detail"]


def test_scenario_arc_wmte_4659_value_routing_timeout_is_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4659-VALUE-ROUTING-COST-CONTROL: timeout blocks win."""

    report = _report_for_payload(
        tmp_path,
        _value_routing_overclaim_fixture(
            per_node_feature_cost_ms=0.42,
            sim_timed_out=True,
        ),
    )
    flags = _flag_kind(report, av.VALUE_ROUTING_WITHOUT_COST_CONTROL_KIND)

    assert flags
    assert flags[0]["severity"] == "critical"
    assert "timeout" in flags[0]["detail"]


def test_scenario_arc_wmte_4659_value_routing_cost_controls_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4659-VALUE-ROUTING-COST-CONTROL: reported controls are clean."""

    report = _report_for_payload(
        tmp_path,
        _value_routing_overclaim_fixture(
            per_node_feature_cost_ms=0.42,
            sim_timed_out=False,
        ),
    )

    assert _flag_kind(report, av.VALUE_ROUTING_WITHOUT_COST_CONTROL_KIND) == []
    assert _flag_kind(report, av.VALUE_ROUTING_COST_CONTROL_OMITTED_KIND) == []


def test_req_arc_wmte_4659_honest_a1_a2_fixtures_not_false_flagged() -> None:
    """REQ-ARC-WMTE-4659: honest .429 artifacts do not fire the new guards."""

    guarded_kinds = {
        av.QD_WITHOUT_RANDOM_MUTATION_ABLATION_KIND,
        av.QD_RANDOM_MUTATION_ABLATION_OMITTED_KIND,
        av.VALUE_ROUTING_WITHOUT_COST_CONTROL_KIND,
        av.VALUE_ROUTING_COST_CONTROL_OMITTED_KIND,
    }

    a1_report = av.verify_artifact(A1_PATH)
    a2_report = av.verify_artifact(A2_PATH)

    assert _flag_kinds(a1_report, guarded_kinds) == []
    assert _flag_kinds(a2_report, guarded_kinds) == []
    assert a1_report["flag_count"] == 0
    assert a2_report["flag_count"] == 0


def test_req_arc_wmte_4659_helper_edges_keep_guards_narrow() -> None:
    """REQ-ARC-WMTE-4659: helper edges ignore metadata and non-win claims."""

    assert av._has_marker("energy-fitness QD generation", ("qd",))
    assert av._real_field_values(
        {
            "field_principles": {
                "random_mutation_ablation_passed": {"principle": "metadata only"}
            },
            "nested": [{"random_mutation_ablation_passed": True}],
        },
        "random_mutation_ablation_passed",
    ) == [True]
    assert not av._has_positive_top_level_metric(
        {
            "field_principles": {"solve_rate_delta": {"principle": "metadata only"}},
            "solve_rate_delta": 0.0,
        },
        ("solve_rate_delta",),
    )
    assert av._has_positive_metric_pair(
        {"live_solve_rate_qd": 0.25, "live_solve_rate_search_baseline": 0.0},
        av._QD_BASELINE_PAIRS,
    )
    assert not av._claims_qd_energy_fitness_generation_win(
        {
            "game": "tn36",
            "headline": "live generation first-win up",
            "winner_generated": True,
        }
    )
    assert av._claims_qd_energy_fitness_generation_win(
        {
            "game": "tn36",
            "headline": "energy-fitness QD live generation",
            "winner_generated_count": 2,
        }
    )

    qd_non_win_flags: list[av.Flag] = []
    av.check_qd_random_mutation_ablation_overclaim(
        {
            "game": "tn36",
            "headline": "energy-fitness QD live generation measurement",
            "honest_verdict": "complete: energy_fitness_qd_generation_observed",
            "solve_provenance": "live_agent_self_discovery",
        },
        qd_non_win_flags,
    )
    assert [flag.kind for flag in qd_non_win_flags] == [
        av.QD_RANDOM_MUTATION_ABLATION_OMITTED_KIND
    ]

    assert not av._claims_value_routing_live_win(
        {
            "game": "ar25",
            "headline": "value-routing offline measurement",
            "first_win_rate_delta": 0.04,
        }
    )
    value_non_win_flags: list[av.Flag] = []
    av.check_value_routing_cost_control_overclaim(
        {
            "game": "ar25",
            "headline": "value-routing live measurement",
            "honest_verdict": "complete: value_routing_cost_control_observed",
            "solve_provenance": "live_agent_self_discovery",
        },
        value_non_win_flags,
    )
    assert [flag.kind for flag in value_non_win_flags] == [
        av.VALUE_ROUTING_COST_CONTROL_OMITTED_KIND
    ]


def test_req_arc_wmte_4659_runner_builds_required_terminal_artifact() -> None:
    """REQ-ARC-WMTE-4659: Exp 4659 emits the required evidence fields."""

    from carnot import experiment_4659_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4659": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]
    assert artifact["honest_verdict"] == (
        "success: adversarial_verify_hardened_qd_ablation_and_value_routing_cost_guards_tests_green."
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["qd_ablation_guard_added"] is True
    assert artifact["value_routing_cost_guard_added"] is True
    assert artifact["honest_artifacts_not_flagged"] is True
    assert artifact["tests_added"]["passed"] is True
    assert artifact["random_seed"] == 4659
    assert artifact["preconditions_checked"]["ok"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_req_arc_wmte_4659_runner_validation_rejects_malformed_artifact() -> None:
    """REQ-ARC-WMTE-4659: artifact validation fails closed."""

    from carnot import experiment_4659_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4659": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not_terminal"
    bad["inference_substrate"] = "wrong"
    bad["qd_ablation_guard_added"] = False
    bad["value_routing_cost_guard_added"] = False
    bad["honest_artifacts_not_flagged"] = False
    bad["tests_added"] = {"passed": False}
    bad["random_seed"] = 0
    bad["preconditions_checked"] = {"ok": False}
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = "sha256:bad"
    errors = mod.validate_artifact(bad)

    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "qd_ablation_guard_added" in errors
    assert "value_routing_cost_guard_added" in errors
    assert "honest_artifacts_not_flagged" in errors
    assert "tests_added.passed" in errors
    assert "random_seed" in errors
    assert "preconditions_checked.ok" in errors
    assert "field_principles.honest_verdict" in errors
    assert "reproducibility_checksum" in errors

    bad_shape = dict(artifact)
    bad_shape["tests_added"] = None
    bad_shape["preconditions_checked"] = None
    bad_shape["field_principles"] = None
    bad_shape["reproducibility_checksum"] = "sha256:bad"
    shape_errors = mod.validate_artifact(bad_shape)

    assert "tests_added" in shape_errors
    assert "preconditions_checked" in shape_errors
    assert "field_principles" in shape_errors

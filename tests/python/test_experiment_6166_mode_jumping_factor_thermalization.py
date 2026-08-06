"""Tests for Exp6166 mode-jumping factor thermalization.

Spec refs: REQ-SAMPLE-6166, SCENARIO-SAMPLE-6166-MULTIMODAL-CNCE,
SCENARIO-SAMPLE-6166-BOUND-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6166_mode_jumping_factor_thermalization as mod


REPO = Path(__file__).resolve().parents[2]
SAMPLER_SPEC = REPO / "openspec/capabilities/samplers/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def test_req_6166_spec_declares_mode_jumping_contract() -> None:
    """REQ-SAMPLE-6166: OpenSpec anchors Exp6166 fields, paths, and scope."""

    spec = SAMPLER_SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-6166") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAMPLE-6166-MULTIMODAL-FACTOR",
        "REQ-SAMPLE-6166-EXACT-ENUMERATION",
        "REQ-SAMPLE-6166-LOCAL-CNCE",
        "REQ-SAMPLE-6166-CROSS-MODE-NOISE",
        "REQ-SAMPLE-6166-MATCHED-TRAINING",
        "REQ-SAMPLE-6166-NONZERO-ERROR",
        "REQ-SAMPLE-6166-RELATIVE-MODE-MASS",
        "REQ-SAMPLE-6166-FACTOR-JOINT-DIVERGENCE",
        "REQ-SAMPLE-6166-COMPOSITION-BOUND",
        "REQ-SAMPLE-6166-CONTROLS",
        "REQ-SAMPLE-6166-RETIRED-NONREUSE",
        "REQ-SAMPLE-6166-SOFTWARE-ONLY",
        "SCENARIO-SAMPLE-6166-MULTIMODAL-CNCE",
        "SCENARIO-SAMPLE-6166-BOUND-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.TEST_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_6166_exact_factor_support_modes_and_noise_are_frozen() -> None:
    """REQ-SAMPLE-6166-MULTIMODAL-FACTOR/EXACT-ENUMERATION: exact oracle is fixed."""

    program = mod.build_multimodal_factor_program()
    exact = mod.exact_multimodal_factor_support_distribution_and_mode_masses(program)
    noise = mod.frozen_local_and_cross_mode_noise_distributions()

    assert mod.exp6152.validate_program(program)["wire_count"] == 1
    assert exact["state_space_size"] == 7
    assert exact["support_count"] == 6
    assert exact["unsupported_states"] == ["unsupported_shadow"]
    assert exact["normalization_error"] <= mod.EXACT_TOLERANCE
    assert exact["mode_masses"]["left_mode"] == pytest.approx(0.6)
    assert exact["mode_masses"]["right_mode"] == pytest.approx(0.35)
    assert exact["relative_mode_mass_ratio"] == pytest.approx(12.0 / 7.0)

    assert noise["local_noise"]["contains_cross_mode_jump"] is False
    assert noise["cross_mode_noise"]["contains_cross_mode_jump"] is True
    assert ("left_peak", "right_peak") in {
        tuple(edge) for edge in noise["cross_mode_noise"]["undirected_edges"]
    }
    assert noise["support_mask_excludes"] == ["unsupported_shadow"]


def test_scenario_6166_mode_jump_improves_with_nonzero_error() -> None:
    """SCENARIO-SAMPLE-6166-MULTIMODAL-CNCE: mode jumping improves composition."""

    program = mod.build_multimodal_factor_program()
    arms = mod.train_matched_cnce_arms(program)
    bound = mod.preregistered_factor_to_joint_bound(program, arms)
    errors = mod.factor_and_joint_tv_kl_and_mode_mass_ratio_errors(program, arms, bound)
    nonzero = mod.deliberately_nonzero_error_receipt(errors)
    intervals = mod.seed_intervals_and_convergence(arms)

    assert bound["derived_before_joint_evaluation"] is True
    assert bound["joint_results_read_before_hash"] is False
    assert bound["precommit_sha256"].startswith("sha256:")
    assert "joint_tv" not in json.dumps(bound["precommit_payload"], sort_keys=True)

    identity = errors["arms"]["identity"]
    local = errors["arms"]["local_only"]
    jumped = errors["arms"]["mode_jump"]

    assert identity["factor_tv"] <= mod.EXACT_TOLERANCE
    assert identity["joint_tv"] <= mod.EXACT_TOLERANCE
    assert local["factor_tv"] > mod.EXACT_TOLERANCE
    assert jumped["factor_tv"] > mod.EXACT_TOLERANCE
    assert local["factor_kl_target_to_candidate"] < float("inf")
    assert jumped["factor_kl_target_to_candidate"] < float("inf")
    assert jumped["joint_tv"] < local["joint_tv"]
    assert jumped["mode_mass_ratio_error"] < local["mode_mass_ratio_error"]
    assert nonzero["approximate_error_finite_and_strictly_positive"] is True
    assert nonzero["identity_exact_table_zero_error"] is True
    assert arms["local_only"]["receipt"]["copied_exact_log_probabilities"] is False
    assert arms["mode_jump"]["receipt"]["copied_exact_log_probabilities"] is False
    assert intervals["mode_jump_improved_primary_metric_all_seeds"] is True


def test_scenario_6166_artifact_controls_scope_and_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-6166-BOUND-CONTROLS: artifact gates and controls fire."""

    artifact = mod.write_mode_jumping_factor_thermalization_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["mode_jumping_factor_thermalization_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["hardware_execution_claimed"] is False
    assert artifact["latency_power_energy_and_speedup_claimed"] is False
    assert artifact["missing_verifier_gaps"] == []

    slack = artifact["bound_slack_and_violation_counts"]
    assert slack["violation_count"] == 0
    assert slack["arms"]["local_only"]["bound_respected"] is True
    assert slack["arms"]["mode_jump"]["bound_respected"] is True

    controls = artifact[
        "identity_no_jump_wrong_jump_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls"
    ]
    assert controls["identity_zero_error_control_passed"] is True
    assert controls["no_jump_control_fired"] is True
    assert controls["wrong_jump_control_fired"] is True
    assert controls["bad_factor_control_fired"] is True
    assert controls["permuted_wire_control_fired"] is True
    assert controls["unsupported_state_control_fired"] is True
    assert controls["loose_bound_control_fired"] is True
    assert controls["all_controls_passed"] is True

    retired = artifact["retired_parity_scaling_nonreuse_receipt"]
    assert retired["retired_lineage_blocked"] is True
    assert retired["size_sweep_produced"] is False
    assert retired["carnot_vs_vendored_thrml_parity_table_produced"] is False

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))
    assert loaded == artifact


def test_req_6166_schema_guards_and_blocked_reasons(tmp_path: Path) -> None:
    """REQ-SAMPLE-6166-CONTROLS/SOFTWARE-ONLY: schema guards fail closed."""

    artifact = mod.write_mode_jumping_factor_thermalization_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )

    missing = deepcopy(artifact)
    del missing["status"]
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    hardware = deepcopy(artifact)
    hardware["hardware_execution_claimed"] = True
    hardware["mode_jumping_factor_thermalization_ready_score"] = mod.ready_score(hardware)
    hardware["status"] = mod.status(hardware)
    hardware["honest_verdict"] = mod.honest_verdict(hardware)
    hardware["reproducibility_checksum"] = mod.reproducibility_checksum(hardware)
    with pytest.raises(ValueError, match="hardware_execution_claimed"):
        mod.validate_artifact(hardware)

    substrate = deepcopy(artifact)
    substrate["inference_substrate"] = "gpu"
    substrate["mode_jumping_factor_thermalization_ready_score"] = mod.ready_score(substrate)
    substrate["status"] = mod.status(substrate)
    substrate["honest_verdict"] = mod.honest_verdict(substrate)
    substrate["reproducibility_checksum"] = mod.reproducibility_checksum(substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(substrate)

    blocked = deepcopy(artifact)
    blocked["preconditions_checked"]["preconditions_ready"] = False
    blocked["deliberately_nonzero_error_receipt"][
        "approximate_error_finite_and_strictly_positive"
    ] = False
    blocked["factor_and_joint_tv_kl_and_mode_mass_ratio_errors"][
        "mode_jump_improved_over_local_only"
    ] = False
    blocked["bound_slack_and_violation_counts"]["violation_count"] = 1
    blocked[
        "identity_no_jump_wrong_jump_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls"
    ]["all_controls_passed"] = False
    blocked["hardware_execution_claimed"] = True
    blocked["latency_power_energy_and_speedup_claimed"] = True
    blocked["test_exit_codes"] = {mod.DEFAULT_TEST_COMMANDS[0]: 7}
    reasons = mod.blocked_reasons(blocked)
    assert {
        "preconditions",
        "nonzero_error",
        "mode_jump_improvement",
        "bound_violation",
        "controls",
        "hardware_claim",
        "performance_claim",
        "missing_test_commands",
        "nonzero_test_commands",
    } <= set(reasons)
    assert mod.status(blocked) == "blocked"
    assert mod.honest_verdict(blocked).startswith("blocked:")

    assert mod._draw_from_distribution({"only": 0.0}, mod.random.Random(0)) == "only"
    edge_metrics = mod._distribution_metrics(
        {
            **mod.EXACT_PROBABILITIES,
            "left_peak": 0.0,
            "unsupported_shadow": 0.1,
        }
    )
    assert edge_metrics["joint_kl_target_to_candidate"] == float("inf")
    assert edge_metrics["support_violation_count"] == 1

    bound_blocked = deepcopy(artifact)
    bound_blocked["bound_slack_and_violation_counts"]["violation_count"] = 1
    bound_blocked["mode_jumping_factor_thermalization_ready_score"] = mod.ready_score(bound_blocked)
    assert mod.status(bound_blocked) == "blocked"

    null = deepcopy(artifact)
    null["factor_and_joint_tv_kl_and_mode_mass_ratio_errors"][
        "mode_jump_improved_over_local_only"
    ] = False
    null["mode_jumping_factor_thermalization_ready_score"] = mod.ready_score(null)
    null["status"] = mod.status(null)
    null["honest_verdict"] = mod.honest_verdict(null)
    assert null["status"] == "complete_null"
    assert null["honest_verdict"].startswith("complete_null:")

    retired = deepcopy(artifact)
    retired["retirement_triggered"] = True
    retired["mode_jumping_factor_thermalization_ready_score"] = mod.ready_score(retired)
    retired["status"] = mod.status(retired)
    retired["honest_verdict"] = mod.honest_verdict(retired)
    assert retired["status"] == "retired"
    assert retired["honest_verdict"].startswith("retired:")

    score = deepcopy(artifact)
    score["mode_jumping_factor_thermalization_ready_score"] = 0.0
    score["reproducibility_checksum"] = mod.reproducibility_checksum(score)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(score)

    latency = deepcopy(artifact)
    latency["latency_power_energy_and_speedup_claimed"] = True
    latency["mode_jumping_factor_thermalization_ready_score"] = mod.ready_score(latency)
    latency["status"] = mod.status(latency)
    latency["honest_verdict"] = mod.honest_verdict(latency)
    latency["reproducibility_checksum"] = mod.reproducibility_checksum(latency)
    with pytest.raises(ValueError, match="latency_power_energy_and_speedup_claimed"):
        mod.validate_artifact(latency)

    verifier = deepcopy(artifact)
    verifier["verifier_is_oracle"] = False
    verifier["mode_jumping_factor_thermalization_ready_score"] = mod.ready_score(verifier)
    verifier["status"] = mod.status(verifier)
    verifier["honest_verdict"] = mod.honest_verdict(verifier)
    verifier["reproducibility_checksum"] = mod.reproducibility_checksum(verifier)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(verifier)

    status = deepcopy(artifact)
    status["status"] = "blocked"
    status["reproducibility_checksum"] = mod.reproducibility_checksum(status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(status)

    verdict = deepcopy(artifact)
    verdict["honest_verdict"] = "blocked: wrong"
    verdict["reproducibility_checksum"] = mod.reproducibility_checksum(verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(verdict)

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum)

    provenance_type = deepcopy(artifact)
    provenance_type["field_provenance"] = []
    provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(provenance_type)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(provenance_type)

    provenance = deepcopy(artifact)
    provenance["field_provenance"]["status"]["principle"] = "wrong"
    provenance["reproducibility_checksum"] = mod.reproducibility_checksum(provenance)
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(provenance)

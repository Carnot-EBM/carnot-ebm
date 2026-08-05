"""Tests for Exp6153 thermalized program error audit.

Spec refs: REQ-SAMPLE-6153, SCENARIO-SAMPLE-6153-BOUND-PRECOMMIT,
SCENARIO-SAMPLE-6153-CONTEXT-MATCHING, SCENARIO-SAMPLE-6153-CONTROLS-SCOPE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random

import pytest

from carnot import experiment_6153_thermalized_program_error_audit as mod


REPO = Path(__file__).resolve().parents[2]
SAMPLER_SPEC = REPO / "openspec/capabilities/samplers/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def test_req_6153_spec_declares_error_audit_contract() -> None:
    """REQ-SAMPLE-6153: OpenSpec anchors Exp6153 fields, paths, and scope."""

    spec = SAMPLER_SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-6153") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAMPLE-6153-1",
        "REQ-SAMPLE-6153-2",
        "REQ-SAMPLE-6153-3",
        "REQ-SAMPLE-6153-4",
        "REQ-SAMPLE-6153-5",
        "REQ-SAMPLE-6153-6",
        "REQ-SAMPLE-6153-7",
        "REQ-SAMPLE-6153-8",
        "REQ-SAMPLE-6153-9",
        "SCENARIO-SAMPLE-6153-BOUND-PRECOMMIT",
        "SCENARIO-SAMPLE-6153-CONTEXT-MATCHING",
        "SCENARIO-SAMPLE-6153-CONTROLS-SCOPE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.TEST_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_6153_factor_eligibility_and_interface_receipts() -> None:
    """REQ-SAMPLE-6153-1/2: factors compile through real pinned software APIs."""

    program = mod.upstream_program()
    interfaces = mod.torx_thrml_versions_commits_import_and_api_receipts(program)
    manifest = mod.factor_eligibility_and_compilation_manifest(program)

    assert interfaces["interface_ready"] is True
    assert interfaces["torx"]["importable"] is True
    assert interfaces["torx"]["compatibility_ready"] is True
    assert interfaces["vendored_thrml"]["importable"] is True
    assert interfaces["vendored_thrml"]["api_exercised"] is True
    assert interfaces["vendored_thrml"]["version"] == "0.1.3"
    assert interfaces["jax"]["default_backend"] == "cpu"

    assert manifest["eligible_factor_count"] == 9
    assert manifest["compiled_factor_count"] == 9
    assert manifest["support_preserved"] is True
    assert manifest["normalization_error_max"] <= mod.EXACT_TOLERANCE
    assert manifest["software_ebm_kernel_count"] == 9
    assert {row["kernel_id"] for row in manifest["factors"]} == {
        kernel.identifier for kernel in program.kernels
    }
    assert all(row["eligible"] and row["compiled"] for row in manifest["factors"])
    assert all(row["support_violation_count"] == 0 for row in manifest["factors"])


def test_scenario_6153_bound_precommit_before_joint_results() -> None:
    """SCENARIO-SAMPLE-6153-BOUND-PRECOMMIT: bound hash precedes joint reads."""

    program = mod.upstream_program()
    arms = mod.train_resource_matched_arms(program)
    bound = mod.preregister_per_factor_to_joint_error_bound(program, arms)

    assert bound["derived_before_joint_evaluation"] is True
    assert bound["joint_results_read_before_hash"] is False
    assert bound["precommit_sha256"].startswith("sha256:")
    assert "joint_tv" not in json.dumps(bound["precommit_payload"], sort_keys=True)
    assert set(bound["arms"]) == {"isolated", "context_matched"}
    assert bound["arms"]["context_matched"]["tv_bound"] <= bound["arms"]["isolated"]["tv_bound"]

    exact = mod.evaluate_exact_joint_outputs(program, arms, bound)
    slack = mod.bound_slack_and_violation_counts(exact, bound)

    assert exact["bound_precommit_sha256"] == bound["precommit_sha256"]
    assert exact["arms"]["isolated"]["joint_tv"] <= mod.EXACT_TOLERANCE
    assert exact["arms"]["context_matched"]["joint_tv"] <= mod.EXACT_TOLERANCE
    assert slack["violation_count"] == 0
    assert slack["arms"]["context_matched"]["bound_respected"] is True


def test_scenario_6153_context_matching_sampled_convergence_and_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAMPLE-6153-CONTEXT-MATCHING: matched arms emit ready evidence."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.write_thermalized_program_error_audit_artifact(
        output_path=output,
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["thermalized_program_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["hardware_execution_claimed"] is False
    assert artifact["latency_power_energy_and_speedup_claimed"] is False

    counts = artifact["exact_and_sampled_case_counts"]
    assert counts["exact"]["state_space_size"] == 1536
    assert counts["sampled"]["samples_per_arm"] == mod.SAMPLES_PER_ARM
    assert counts["sampled"]["seed_count"] == len(mod.SAMPLE_SEEDS)

    intervals = artifact["context_matched_minus_isolated_intervals"]
    assert intervals["primary_metric"] == "exact_joint_tv"
    assert intervals["context_matching_noninferior"] is True
    assert intervals["exact_delta"] <= mod.EXACT_TOLERANCE

    convergence = artifact["autocorrelation_effective_sample_size_and_convergence"]
    assert convergence["nonconvergence_count"] == 0
    assert convergence["support_violation_count"] == 0
    assert convergence["arms"]["isolated"]["effective_sample_size_min"] > 0
    assert convergence["arms"]["context_matched"]["effective_sample_size_min"] > 0


def test_scenario_6153_controls_retired_scope_and_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-6153-CONTROLS-SCOPE: controls fire and gates fail closed."""

    artifact = mod.write_thermalized_program_error_audit_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )
    controls = artifact[
        "identity_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls"
    ]

    assert controls["identity_zero_error_control_passed"] is True
    assert controls["bad_factor_control_fired"] is True
    assert controls["permuted_wire_control_fired"] is True
    assert controls["unsupported_state_control_fired"] is True
    assert controls["overly_loose_bound_control_fired"] is True
    assert controls["all_controls_passed"] is True

    retired = artifact["retired_parity_scaling_nonreuse_receipt"]
    assert retired["retired_lineage_blocked"] is True
    assert retired["size_sweep_produced"] is False
    assert retired["carnot_vs_vendored_thrml_parity_table_produced"] is False
    assert retired["retirement_triggered"] is False

    missing = deepcopy(artifact)
    del missing["status"]
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    hardware = deepcopy(artifact)
    hardware["hardware_execution_claimed"] = True
    hardware["thermalized_program_ready_score"] = mod.ready_score(hardware)
    hardware["status"] = mod.status(hardware)
    hardware["honest_verdict"] = mod.honest_verdict(hardware)
    hardware["reproducibility_checksum"] = mod.reproducibility_checksum(hardware)
    with pytest.raises(ValueError, match="hardware_execution_claimed"):
        mod.validate_artifact(hardware)

    violated = deepcopy(artifact)
    violated["bound_slack_and_violation_counts"]["violation_count"] = 1
    violated["thermalized_program_ready_score"] = mod.ready_score(violated)
    violated["status"] = mod.status(violated)
    violated["honest_verdict"] = mod.honest_verdict(violated)
    violated["reproducibility_checksum"] = mod.reproducibility_checksum(violated)
    assert violated["status"] == "complete_bound_violated"
    assert violated["honest_verdict"].startswith("complete_bound_violated:")
    assert mod.validate_artifact(violated) is True

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum)

    provenance = deepcopy(artifact)
    provenance["field_provenance"]["status"]["principle"] = "wrong"
    provenance["reproducibility_checksum"] = mod.reproducibility_checksum(provenance)
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(provenance)


def test_req_6153_defensive_branches_and_divergence_edges(tmp_path: Path) -> None:
    """REQ-SAMPLE-6153-4/9: edge divergences and schema guards fail closed."""

    program = mod.upstream_program()
    arms = mod.train_resource_matched_arms(program)
    root_kernel = next(
        kernel for kernel in program.kernels if kernel.identifier == "sample_candidate_item"
    )
    zeroed = mod._replace_kernel_distribution(
        arms["context_matched"]["sample_candidate_item"],
        (),
        {"ac_e0_0": 1.0, "ac_e0_1": 0.0, "ac_e0_2": 0.0},
    )
    factor_divergence = mod._factor_divergence(program, root_kernel, zeroed)
    assert factor_divergence["weighted_kl"] == float("inf")

    zeroed_arm = dict(arms["context_matched"])
    zeroed_arm["sample_candidate_item"] = zeroed
    target = mod.exp6152.execute_exact(program)
    zeroed_joint = mod.execute_joint_from_ebm_kernels(program, zeroed_arm)
    joint_divergence = mod.distribution_divergence(target, zeroed_joint)
    assert joint_divergence["joint_kl_target_to_candidate"] == float("inf")

    unsupported_arm = dict(arms["context_matched"])
    unsupported_arm["member_group_lookup"] = mod._replace_kernel_distribution(
        arms["context_matched"]["member_group_lookup"],
        ("ac_e0_0",),
        {"ac_g0_0": 0.95, "ac_g0_1": 0.05},
    )
    unsupported_joint = mod.execute_joint_from_ebm_kernels(program, unsupported_arm)
    assert mod.distribution_divergence(target, unsupported_joint)["support_violation_count"] > 0

    assert mod._draw({"0": 0.0}, random.Random(0)) == 0
    assert mod._lag1_autocorrelation([1.0, 1.0, 1.0]) == 0.0
    assert mod._optional_import_receipt("json")["importable"] is True

    artifact = mod.write_thermalized_program_error_audit_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )

    retired = deepcopy(artifact)
    retired["retirement_triggered"] = True
    retired["thermalized_program_ready_score"] = mod.ready_score(retired)
    retired["status"] = mod.status(retired)
    retired["honest_verdict"] = mod.honest_verdict(retired)
    assert retired["status"] == "retired"
    assert retired["honest_verdict"].startswith("retired:")

    blocked = deepcopy(artifact)
    blocked["preconditions_checked"]["preconditions_ready"] = False
    blocked["torx_thrml_versions_commits_import_and_api_receipts"]["interface_ready"] = False
    blocked["factor_eligibility_and_compilation_manifest"]["support_preserved"] = False
    blocked["bound_slack_and_violation_counts"]["violation_count"] = 1
    blocked["context_matched_minus_isolated_intervals"]["context_matching_noninferior"] = False
    blocked["identity_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls"][
        "all_controls_passed"
    ] = False
    blocked["hardware_execution_claimed"] = True
    blocked["latency_power_energy_and_speedup_claimed"] = True
    blocked["test_exit_codes"] = {mod.DEFAULT_TEST_COMMANDS[0]: 7}
    reasons = mod.blocked_reasons(blocked)
    assert {
        "preconditions",
        "software_interfaces",
        "support_preservation",
        "bound_violation",
        "context_matching",
        "controls",
        "hardware_claim",
        "performance_claim",
        "missing_test_commands",
        "nonzero_test_commands",
    } <= set(reasons)
    assert mod.status(blocked) == "blocked"

    latency = deepcopy(artifact)
    latency["latency_power_energy_and_speedup_claimed"] = True
    latency["thermalized_program_ready_score"] = mod.ready_score(latency)
    latency["status"] = mod.status(latency)
    latency["honest_verdict"] = mod.honest_verdict(latency)
    latency["reproducibility_checksum"] = mod.reproducibility_checksum(latency)
    with pytest.raises(ValueError, match="latency_power_energy_and_speedup_claimed"):
        mod.validate_artifact(latency)

    substrate = deepcopy(artifact)
    substrate["inference_substrate"] = "gpu"
    substrate["thermalized_program_ready_score"] = mod.ready_score(substrate)
    substrate["status"] = mod.status(substrate)
    substrate["honest_verdict"] = mod.honest_verdict(substrate)
    substrate["reproducibility_checksum"] = mod.reproducibility_checksum(substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(substrate)

    verifier = deepcopy(artifact)
    verifier["verifier_is_oracle"] = False
    verifier["thermalized_program_ready_score"] = mod.ready_score(verifier)
    verifier["status"] = mod.status(verifier)
    verifier["honest_verdict"] = mod.honest_verdict(verifier)
    verifier["reproducibility_checksum"] = mod.reproducibility_checksum(verifier)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(verifier)

    score = deepcopy(artifact)
    score["thermalized_program_ready_score"] = 0.0
    score["reproducibility_checksum"] = mod.reproducibility_checksum(score)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(score)

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

    provenance_type = deepcopy(artifact)
    provenance_type["field_provenance"] = []
    provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(provenance_type)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(provenance_type)

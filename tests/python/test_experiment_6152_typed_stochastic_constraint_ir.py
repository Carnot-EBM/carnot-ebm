"""Tests for Exp6152 typed stochastic constraint IR.

Spec refs: REQ-SAMPLE-6152, SCENARIO-SAMPLE-6152-VALIDATION,
SCENARIO-SAMPLE-6152-EXACT, SCENARIO-SAMPLE-6152-SERIALIZATION-TORX.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random

import pytest

from carnot import experiment_6152_typed_stochastic_constraint_ir as mod


REPO = Path(__file__).resolve().parents[2]
SAMPLER_SPEC = REPO / "openspec/capabilities/samplers/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def test_req_6152_spec_declares_typed_stochastic_ir_contract() -> None:
    """REQ-SAMPLE-6152: OpenSpec anchors fields, scenarios, and paths."""

    spec = SAMPLER_SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-6152") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAMPLE-6152-1",
        "REQ-SAMPLE-6152-2",
        "REQ-SAMPLE-6152-3",
        "REQ-SAMPLE-6152-4",
        "REQ-SAMPLE-6152-5",
        "REQ-SAMPLE-6152-6",
        "REQ-SAMPLE-6152-7",
        "SCENARIO-SAMPLE-6152-VALIDATION",
        "SCENARIO-SAMPLE-6152-EXACT",
        "SCENARIO-SAMPLE-6152-SERIALIZATION-TORX",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.TEST_RELATIVE_PATH.as_posix(),
        "jax_cpu_exact_stochastic_program",
        "carnot_only_blocked_torx_compatibility",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6152_validation_rejects_graph_type_and_mass_bugs() -> None:
    """SCENARIO-SAMPLE-6152-VALIDATION: typed graph validation fails closed."""

    program = mod.compile_exp6145_bounded_workflow()
    receipt = mod.validate_program(program)

    assert receipt["ok"] is True
    assert receipt["schema_version"] == mod.IR_SCHEMA_VERSION
    assert receipt["wire_count"] == 9
    assert receipt["kernel_count"] == 9
    assert receipt["wire_type_counts"] == {"binary": 6, "categorical": 3}
    assert receipt["topological_kernel_order"][0] == "sample_candidate_item"
    assert "eligible_truth_table" in receipt["topological_kernel_order"]

    payload = mod.program_to_payload(program)

    duplicate_wire = deepcopy(payload)
    duplicate_wire["wires"].append(deepcopy(duplicate_wire["wires"][0]))
    with pytest.raises(mod.TypedStochasticIRValidationError, match="duplicate wire"):
        mod.program_from_payload(duplicate_wire)

    invalid_mass = deepcopy(payload)
    invalid_mass["kernels"][0]["params"]["probabilities"] = [0.4, 0.4, 0.1]
    with pytest.raises(mod.TypedStochasticIRValidationError, match="probability mass"):
        mod.program_from_payload(invalid_mass)

    dangling = deepcopy(payload)
    dangling["kernels"][2]["inputs"] = ["missing_wire"]
    with pytest.raises(mod.TypedStochasticIRValidationError, match="dangling wire"):
        mod.program_from_payload(dangling)

    type_mismatch = deepcopy(payload)
    type_mismatch["kernels"][2]["inputs"] = ["strategy_clean"]
    with pytest.raises(mod.TypedStochasticIRValidationError, match="type mismatch"):
        mod.program_from_payload(type_mismatch)

    bad_category = deepcopy(payload)
    bad_category["kernels"][2]["params"]["table"][0] = 99
    with pytest.raises(mod.TypedStochasticIRValidationError, match="category index"):
        mod.program_from_payload(bad_category)

    cycle = deepcopy(payload)
    cycle["kernels"][2]["inputs"] = ["member_group"]
    cycle["kernels"][2]["params"]["table"] = [0, 1]
    with pytest.raises(mod.TypedStochasticIRValidationError, match="cycle"):
        mod.program_from_payload(cycle)

    ambiguous_seed = deepcopy(payload)
    del ambiguous_seed["kernels"][0]["params"]["seed_role"]
    with pytest.raises(mod.TypedStochasticIRValidationError, match="ambiguous seed"):
        mod.program_from_payload(ambiguous_seed)


def test_req_6152_defensive_validator_branches() -> None:
    """REQ-SAMPLE-6152-2: every validation guard has an executable rejection."""

    program = mod.compile_exp6145_bounded_workflow()
    payload = mod.program_to_payload(program)

    def rejects(mutated: dict[str, object], message: str) -> None:
        with pytest.raises(mod.TypedStochasticIRValidationError, match=message):
            mod.program_from_payload(mutated)

    rejects({}, "missing top-level")
    bad_schema = deepcopy(payload)
    bad_schema["schema_version"] = "wrong"
    rejects(bad_schema, "unsupported schema")
    with pytest.raises(mod.TypedStochasticIRValidationError, match="unsupported schema"):
        mod.validate_program(
            mod.StochasticProgram(
                "wrong",
                program.program_id,
                program.wires,
                program.kernels,
                program.metadata,
            )
        )

    wires_not_list = deepcopy(payload)
    wires_not_list["wires"] = {}
    rejects(wires_not_list, "wires must be a list")

    missing_producer = deepcopy(payload)
    missing_producer["kernels"] = missing_producer["kernels"][:-1]
    rejects(missing_producer, "dangling wires without producer")

    bad_wire_id = deepcopy(payload)
    bad_wire_id["wires"][0]["id"] = ""
    rejects(bad_wire_id, "wire id")

    binary_categories = deepcopy(payload)
    binary_categories["wires"][1]["categories"] = ["bad"]
    rejects(binary_categories, "binary wires")

    empty_categories = deepcopy(payload)
    empty_categories["wires"][0]["categories"] = []
    rejects(empty_categories, "unique categories")

    duplicate_categories = deepcopy(payload)
    duplicate_categories["wires"][0]["categories"] = ["a", "a"]
    rejects(duplicate_categories, "unique categories")

    unsupported_wire = deepcopy(payload)
    unsupported_wire["wires"][0]["kind"] = "pmode"
    rejects(unsupported_wire, "unsupported wire")

    empty_kernel = deepcopy(payload)
    empty_kernel["kernels"][0]["id"] = ""
    rejects(empty_kernel, "kernel id")

    duplicate_kernel = deepcopy(payload)
    duplicate_kernel["kernels"][1]["id"] = duplicate_kernel["kernels"][0]["id"]
    rejects(duplicate_kernel, "duplicate kernel")

    dangling_output = deepcopy(payload)
    dangling_output["kernels"][2]["output"] = "missing_wire"
    rejects(dangling_output, "dangling wire output")

    duplicate_seed = deepcopy(payload)
    duplicate_seed["kernels"][1]["params"]["seed_role"] = duplicate_seed["kernels"][0]["params"][
        "seed_role"
    ]
    rejects(duplicate_seed, "ambiguous seed role reuse")

    categorical_prior_type = deepcopy(payload)
    categorical_prior_type["kernels"][0]["output"] = "strategy_clean"
    rejects(categorical_prior_type, "type mismatch for categorical_prior")

    bernoulli_prior_type = deepcopy(payload)
    bernoulli_prior_type["kernels"][1]["output"] = "candidate_item"
    rejects(bernoulli_prior_type, "type mismatch for bernoulli_prior")

    bernoulli_mass = deepcopy(payload)
    bernoulli_mass["kernels"][1]["params"]["p_true"] = 1.5
    rejects(bernoulli_mass, "probability mass for bernoulli_prior")

    truth_table_output = deepcopy(payload)
    truth_table_output["kernels"][7]["output"] = "member_group"
    rejects(truth_table_output, "type mismatch for deterministic_truth_table")

    truth_table_input = deepcopy(payload)
    truth_table_input["kernels"][7]["inputs"] = ["candidate_item"]
    rejects(truth_table_input, "type mismatch for deterministic_truth_table")

    unsupported_kernel = deepcopy(payload)
    unsupported_kernel["kernels"][0]["kind"] = "mystery"
    rejects(unsupported_kernel, "unsupported kernel")

    probability_length = deepcopy(payload)
    probability_length["kernels"][0]["params"]["probabilities"] = [1.0]
    rejects(probability_length, "probability mass length")

    table_length = deepcopy(payload)
    table_length["kernels"][2]["params"]["table"] = [0]
    rejects(table_length, "deterministic table length")

    duplicate_producer = deepcopy(payload)
    duplicate_producer["kernels"][3]["output"] = "member_group"
    duplicate_producer["kernels"][3]["params"]["table"] = [0, 1, 0]
    rejects(duplicate_producer, "duplicate producer")

    assert mod._rejects_with(payload, lambda value: value, "not raised") is False
    assert mod._draw([(3, 0.0)], random.Random(0)) == 3


def test_scenario_6152_exact_enumeration_matches_independent_reference() -> None:
    """SCENARIO-SAMPLE-6152-EXACT: exact enumeration owns probability evidence."""

    program = mod.compile_exp6145_bounded_workflow()
    exact = mod.execute_exact(program)
    reference = mod.independent_reference_distribution()
    comparison = mod.compare_exact_semantics(program)

    assert exact["state_space_size"] == 1536
    assert exact["support_count"] == 6
    assert exact["impossible_state_count"] == 1530
    assert exact["normalization"] == pytest.approx(1.0)
    assert exact["marginals"] == reference["marginals"]
    assert set(exact["joint_probabilities"]) == set(reference["joint_probabilities"])
    assert comparison["support_match"] is True
    assert comparison["max_joint_delta"] <= mod.EXACT_TOLERANCE
    assert comparison["max_conditional_delta"] <= mod.EXACT_TOLERANCE
    assert comparison["max_marginal_delta"] <= mod.EXACT_TOLERANCE
    assert comparison["normalization_delta"] <= mod.EXACT_TOLERANCE

    assert mod.probability_of(
        exact,
        {"candidate_item": "ac_e0_0", "strategy_clean": 1, "accepted": 1},
    ) == pytest.approx(0.36)
    assert mod.probability_of(
        exact,
        {"candidate_item": "ac_e0_1", "strategy_clean": 1, "accepted": 1},
    ) == pytest.approx(0.0)
    assert mod.probability_of(exact, {"eligible": 1}) == pytest.approx(0.5)
    assert mod.conditional_probability(exact, {"accepted": 1}, {"eligible": 1}) == pytest.approx(
        0.9
    )
    assert mod.conditional_probability(
        exact, {"candidate_item": "ac_e0_2"}, {"accepted": 1}
    ) == pytest.approx(0.2)


def test_scenario_6152_seed_batch_serialization_and_negative_controls() -> None:
    """SCENARIO-SAMPLE-6152-VALIDATION/EXACT: controls catch subtle bugs."""

    program = mod.compile_exp6145_bounded_workflow()
    payload = mod.program_to_payload(program)
    canonical = mod.canonical_json(payload)
    restored = mod.program_from_payload(json.loads(canonical))

    assert mod.program_checksum(restored) == mod.program_checksum(program)
    assert mod.compare_exact_semantics(restored)["max_joint_delta"] <= mod.EXACT_TOLERANCE

    first = mod.sample_batch(program, batch_size=16, seed=6152)
    replay = mod.sample_batch(program, batch_size=16, seed=6152)
    assert first == replay
    assert len(first) == 16
    assert all(set(row) == set(mod.wire_order(program)) for row in first)
    assert mod.batch_shape_contract(program, first) == {
        "batch_size": 16,
        "wire_count": 9,
        "ok": True,
    }

    with pytest.raises(mod.TypedStochasticIRValidationError, match="ambiguous seed"):
        mod.sample_batch(program, batch_size=1, seed=None)
    with pytest.raises(mod.TypedStochasticIRValidationError, match="batch_size"):
        mod.sample_batch(program, batch_size=0, seed=6152)

    controls = mod.run_negative_controls(program)
    assert controls["wire_order_bug_detected"] is True
    assert controls["category_index_bug_detected"] is True
    assert controls["invalid_category_index_rejected"] is True
    assert controls["type_mismatch_rejected"] is True
    assert controls["cycle_rejected"] is True
    assert controls["dangling_wire_rejected"] is True
    assert controls["invalid_mass_rejected"] is True
    assert controls["ambiguous_seed_rejected"] is True
    assert controls["all_negative_controls_passed"] is True


def test_scenario_6152_torx_adapter_exercises_real_import_when_available() -> None:
    """SCENARIO-SAMPLE-6152-SERIALIZATION-TORX: Torx evidence is real or blocked."""

    receipt = mod.torx_adapter_receipt(mod.compile_exp6145_bounded_workflow())

    assert receipt["package_name"] == "extro-torx"
    assert receipt["import_namespace"] == "torx"
    assert receipt["pinned_repository_commit"] == mod.TORX_PINNED_REPOSITORY_COMMIT
    if receipt["importable"]:
        assert receipt["installed_version"] == "0.0.1"
        assert receipt["api_exercised"] is True
        assert receipt["compatibility_ready"] is True
        assert {"DiscretePCircuit", "StateVectorSimulator", "PNOT", "PditShift"} <= set(
            receipt["exercised_api"]
        )
        assert receipt["psc_density_sum_delta"] <= mod.EXACT_TOLERANCE
    else:
        assert receipt["api_exercised"] is False
        assert receipt["compatibility_ready"] is False
        assert receipt["blocked_reason"]


def test_req_6152_artifact_schema_readiness_and_replay(tmp_path: Path) -> None:
    """REQ-SAMPLE-6152: terminal artifact fields are complete and deterministic."""

    artifact = mod.write_typed_stochastic_ir_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["typed_stochastic_ir_ready_score"] == 1.0
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["inference_substrate"] == "jax_cpu_exact_stochastic_program"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["structured_gate_receipt"]["exp6145_ready_score"] == 1.0
    assert artifact["exact_enumeration_case_counts"]["support_count"] == 6
    assert artifact["compiler_executor_adapter_and_test_paths"]["module"] == (
        mod.MODULE_RELATIVE_PATH.as_posix()
    )
    assert artifact["torx_compatibility_scope"]["compatibility_ready"] is True

    deltas = artifact["support_conditional_joint_normalization_and_marginal_deltas"]
    assert deltas["principle"].startswith("Exact enumeration")
    assert deltas["max_joint_delta"] <= mod.EXACT_TOLERANCE
    assert deltas["max_conditional_delta"] <= mod.EXACT_TOLERANCE
    assert deltas["max_marginal_delta"] <= mod.EXACT_TOLERANCE

    second = mod.write_typed_stochastic_ir_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )
    assert second["deterministic_rebuild_checksum"] == artifact["deterministic_rebuild_checksum"]
    assert second["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_6152_artifact_fail_closed_status_and_validation_paths(tmp_path: Path) -> None:
    """REQ-SAMPLE-6152-7: artifact status and schema checks fail closed."""

    artifact = mod.write_typed_stochastic_ir_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )

    blocked = deepcopy(artifact)
    blocked["preconditions_checked"]["preconditions_ready"] = False
    blocked["typed_stochastic_ir_ready_score"] = mod.ready_score(blocked)
    blocked["status"] = mod.status(blocked)
    blocked["honest_verdict"] = mod.honest_verdict(blocked)
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert "preconditions" in mod.blocked_reasons(blocked)

    carnot_only = deepcopy(artifact)
    carnot_only["torx_compatibility_scope"]["compatibility_ready"] = False
    carnot_only["inference_substrate"] = mod.CARNOT_ONLY_SUBSTRATE
    carnot_only["missing_verifier_gaps"] = ["torx_compatibility"]
    carnot_only["typed_stochastic_ir_ready_score"] = mod.ready_score(carnot_only)
    carnot_only["status"] = mod.status(carnot_only)
    carnot_only["honest_verdict"] = mod.honest_verdict(carnot_only)
    assert carnot_only["status"] == "complete_carnot_only"
    assert carnot_only["honest_verdict"].startswith("complete_carnot_only:")
    assert "torx_compatibility" in mod.blocked_reasons(carnot_only)

    complete_null = deepcopy(artifact)
    complete_null["test_exit_codes"].pop(mod.DEFAULT_TEST_COMMANDS[0])
    complete_null["typed_stochastic_ir_ready_score"] = mod.ready_score(complete_null)
    complete_null["status"] = mod.status(complete_null)
    complete_null["honest_verdict"] = mod.honest_verdict(complete_null)
    assert complete_null["status"] == "complete_null"
    assert complete_null["honest_verdict"].startswith("complete_null:")
    assert "missing_test_commands" in mod.blocked_reasons(complete_null)

    protected = deepcopy(artifact)
    protected["protected_files_unchanged"]["unchanged"] = False
    assert "protected_files" in mod.blocked_reasons(protected)

    structured = deepcopy(artifact)
    structured["structured_gate_receipt"]["gate_passed"] = False
    assert "structured_gate" in mod.blocked_reasons(structured)

    missing = deepcopy(artifact)
    del missing["status"]
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    verifier = deepcopy(artifact)
    verifier["verifier_is_oracle"] = False
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(verifier)

    score = deepcopy(artifact)
    score["typed_stochastic_ir_ready_score"] = 0.0
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(score)

    status = deepcopy(artifact)
    status["status"] = "wrong"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(status)

    verdict = deepcopy(artifact)
    verdict["honest_verdict"] = "wrong"
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

    provenance_principle = deepcopy(artifact)
    provenance_principle["field_provenance"]["status"]["principle"] = "wrong"
    provenance_principle["reproducibility_checksum"] = mod.reproducibility_checksum(
        provenance_principle
    )
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(provenance_principle)

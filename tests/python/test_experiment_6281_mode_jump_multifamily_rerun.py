"""Tests for Exp6281 typed mode-jump multifamily A/B rerun.

Spec refs: REQ-SAMPLER-6281,
SCENARIO-SAMPLER-6281-TYPED-MATCHED-CELLS,
SCENARIO-SAMPLER-6281-CONTROLS-SEPARATE-VALUE,
SCENARIO-SAMPLER-6281-RETIREMENT-DECISION.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6281_mode_jump_multifamily_rerun as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
FIXTURE_NAMES = {
    "control_unimodal_ising3",
    "exp6237_original_six_state",
    "ising_ferromagnetic_ring4",
    "ising_ferromagnetic_ring5",
    "potts_antiferro_triangle3_q3",
    "potts_chain3_q3",
    "typed_access_control_exp6152",
    "typed_multimodal_factor_exp6166",
}


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        run_date="20260810",
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )


def test_req_sampler_6281_spec_declares_fields_and_principles() -> None:
    """REQ-SAMPLER-6281: OpenSpec anchors the rerun artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLER-6281") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAMPLER-6281-PRECONDITIONS",
        "REQ-SAMPLER-6281-MATCHED-CELLS",
        "REQ-SAMPLER-6281-ACTIVATION",
        "REQ-SAMPLER-6281-EXACTNESS",
        "REQ-SAMPLER-6281-INTERVALS",
        "REQ-SAMPLER-6281-RETIREMENT",
        "SCENARIO-SAMPLER-6281-TYPED-MATCHED-CELLS",
        "SCENARIO-SAMPLER-6281-CONTROLS-SEPARATE-VALUE",
        "SCENARIO-SAMPLER-6281-RETIREMENT-DECISION",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_sampler_6281_artifact_schema_claim_bounds_and_retirement(
    artifact: dict[str, object],
    tmp_path: Path,
) -> None:
    """REQ-SAMPLER-6281: terminal JSON validates and keeps claims bounded."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    written = mod.write_artifact(
        output_path=output,
        root=REPO,
        run_date="20260810",
        duration_s=0.0,
        test_exit_codes=_passing_exit_codes(),
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == written
    assert written["chain_sample_hashes"] == artifact["chain_sample_hashes"]
    assert written["paired_intervals_equivalence_margins_and_sample_sizes"] == artifact[
        "paired_intervals_equivalence_margins_and_sample_sizes"
    ]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(written)
    assert mod.validate_artifact(written) is True
    assert written["status"] == "retired_value_not_ready"
    assert written["honest_verdict"].startswith("retired_value_not_ready:")
    assert written["mode_jump_safety_ready_score"] == 1.0
    assert written["mode_jump_workload_value_ready_score"] == 0.0
    assert written["retire_mechanism_recommendation"]["recommendation"] == (
        "permanent_retirement_recommended"
    )
    assert written["source_mutation_count"] == 0
    assert type(written["source_mutation_count"]) is int
    assert written["hardware_claim_count"] == 0
    assert type(written["hardware_claim_count"]) is int
    assert written["timing_speedup_claimed"] is False
    assert written["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert written["verifier_is_oracle"]["value"] is True


def test_scenario_sampler_6281_typed_matched_cells_activate_treatment(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-6281-TYPED-MATCHED-CELLS: all typed fixtures run."""

    matrix = artifact["preregistered_fixture_seed_arm_matrix"]
    config = artifact["matched_arm_configuration"]
    counts = artifact["treatment_attempt_accept_and_fire_counts_by_fixture"]
    receipts = artifact["rust_pyo3_backend_receipts"]["chains"]
    hashes = artifact["chain_sample_hashes"]["chains"]

    assert set(matrix["supported_fixtures"]) == FIXTURE_NAMES
    assert matrix["unsupported_fixtures"] == []
    assert config["matched_seeds"] == list(mod.SEEDS)
    assert config["matched_topology"] == mod.VARIABLE_CARDINALITY_TOPOLOGY
    assert config["matched_retained_sample_count"] == mod.RETAINED_SAMPLE_COUNT

    expected_chain_count = len(FIXTURE_NAMES) * len(mod.SEEDS) * len(mod.ARMS)
    assert len(receipts) == expected_chain_count
    assert len(hashes) == expected_chain_count
    assert len({row["chain_id_sha256"] for row in hashes}) == expected_chain_count

    for fixture in FIXTURE_NAMES:
        treatment = counts["fixtures"][fixture]["mode_jump_runtime"]
        assert treatment["active_backend"] == "rust_pyo3"
        assert treatment["attempted_count"] > 0
        assert treatment["accepted_count"] > 0
        assert treatment["treatment_attempt_count"] > 0
        assert treatment["treatment_accept_count"] > 0
        assert treatment["treatment_fire_count"] > 0
    assert counts["activation_proven_before_outcome_comparison"] is True

    for row in receipts:
        assert row["transition_budget"]["retained_samples"] == mod.RETAINED_SAMPLE_COUNT
        if row["arm"] == "mode_jump_runtime":
            assert row["active_backend"] == "rust_pyo3"
            assert row["topology"] == mod.VARIABLE_CARDINALITY_TOPOLOGY
        if row["arm"] == "seeded_fallback":
            assert row["active_backend"] == "python_exact_fallback"
    for row in hashes:
        assert row["sample_count"] == mod.RETAINED_SAMPLE_COUNT
        assert row["sample_labels_sha256"].startswith("sha256:")


def test_scenario_sampler_6281_controls_and_family_value_are_separate(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-6281-CONTROLS-SEPARATE-VALUE: controls do not imply value."""

    controls = artifact["positive_inactive_and_unimodal_control_results"]
    family_safety = artifact["family_level_safety_results"]
    family_value = artifact["family_level_mixing_value_results"]
    paired = artifact["paired_intervals_equivalence_margins_and_sample_sizes"]

    assert controls["positive_control"]["passed"] is True
    assert controls["inactive_treatment_control"]["decision"] == "instrument_failure"
    assert controls["unimodal_control"]["fixture"] == "control_unimodal_ising3"
    assert controls["unimodal_control"]["workload_value_claim_allowed"] is False
    assert paired["value_gate"]["workload_value_gate_passed"] is False

    assert set(family_safety["families"]) == {
        "ising_multimodal",
        "original_six_state_positive_control",
        "potts",
        "typed_factor",
        "unimodal_control",
    }
    assert family_safety["all_family_safety_passed"] is True
    assert family_value["non_toy_positive_mixing_family_count"] == 0
    assert family_value["workload_value_gate_passed"] is False
    assert artifact["harmful_regressions"] == []
    assert artifact["descriptive_wall_time_by_arm_fixture"]["timing_speedup_claimed"] is False


def test_req_sampler_6281_validation_rejects_accounting_and_gate_mutations(
    artifact: dict[str, object],
) -> None:
    """REQ-SAMPLER-6281-NEGATIVE-TESTS: matched-chain gates fail closed."""

    seed_bad = deepcopy(artifact)
    seed_bad["matched_arm_configuration"]["matched_seeds"][0] = 999999
    with pytest.raises(ValueError, match="seed mismatch"):
        mod.validate_artifact(seed_bad)

    sample_bad = deepcopy(artifact)
    sample_bad["chain_sample_hashes"]["chains"][0]["sample_count"] += 1
    with pytest.raises(ValueError, match="sample-count mismatch"):
        mod.validate_artifact(sample_bad)

    receipt_seed_bad = deepcopy(artifact)
    receipt_seed_bad["rust_pyo3_backend_receipts"]["chains"][0]["seed"] = 999999
    with pytest.raises(ValueError, match="seed mismatch"):
        mod.validate_artifact(receipt_seed_bad)

    acceptance_bad = deepcopy(artifact)
    fixture = next(iter(FIXTURE_NAMES))
    acceptance_bad["treatment_attempt_accept_and_fire_counts_by_fixture"]["fixtures"][fixture][
        "mode_jump_runtime"
    ]["accepted_count"] += 1
    with pytest.raises(ValueError, match="acceptance accounting"):
        mod.validate_artifact(acceptance_bad)

    mutations = [
        ("missing required", lambda data: data.pop("status")),
        ("field_principles", lambda data: data.__setitem__("field_principles", {})),
        ("field_provenance", lambda data: data.__setitem__("field_provenance", [])),
        (
            "field_provenance:status",
            lambda data: data["field_provenance"]["status"].__setitem__("principle", "bad"),
        ),
        ("source_mutation_count", lambda data: data.__setitem__("source_mutation_count", 1)),
        ("hardware_claim_count", lambda data: data.__setitem__("hardware_claim_count", True)),
        ("timing_speedup_claimed", lambda data: data.__setitem__("timing_speedup_claimed", True)),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "gpu")),
        ("verifier_is_oracle", lambda data: data["verifier_is_oracle"].__setitem__("value", False)),
        (
            "family_level_safety_results",
            lambda data: data["family_level_safety_results"].__setitem__(
                "all_family_safety_passed",
                False,
            ),
        ),
        (
            "paired_intervals",
            lambda data: data["paired_intervals_equivalence_margins_and_sample_sizes"][
                "value_gate"
            ].__setitem__("workload_value_gate_passed", True),
        ),
        (
            "family_level_mixing_value_results",
            lambda data: data["family_level_mixing_value_results"].__setitem__(
                "workload_value_gate_passed",
                True,
            ),
        ),
        (
            "harmful_regressions",
            lambda data: data["harmful_regressions"].append({"fixture": fixture}),
        ),
        (
            "retire_mechanism_recommendation",
            lambda data: data["retire_mechanism_recommendation"].__setitem__(
                "recommendation",
                "continue",
            ),
        ),
        (
            "mode_jump_safety_ready_score",
            lambda data: data.__setitem__("mode_jump_safety_ready_score", 0.0),
        ),
        (
            "mode_jump_workload_value_ready_score",
            lambda data: data.__setitem__("mode_jump_workload_value_ready_score", 1.0),
        ),
        ("status", lambda data: data.__setitem__("status", "bad")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "bad")),
        (
            "reproducibility_checksum",
            lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_sampler_6281_inactive_regression_blocks_safety_and_retires(
    artifact: dict[str, object],
) -> None:
    """REQ-SAMPLER-6281-NEGATIVE-TESTS: inactive treatment is instrumentation failure."""

    inactive = deepcopy(artifact)
    fixture = "potts_chain3_q3"
    treatment = inactive["treatment_attempt_accept_and_fire_counts_by_fixture"]["fixtures"][
        fixture
    ]["mode_jump_runtime"]
    treatment["treatment_attempt_count"] = 0
    treatment["treatment_accept_count"] = 0
    treatment["treatment_fire_count"] = 0
    treatment["active_backend"] = "python_exact_fallback"
    for row in inactive["rust_pyo3_backend_receipts"]["chains"]:
        if row["fixture"] == fixture and row["arm"] == "mode_jump_runtime":
            row["active_backend"] = "python_exact_fallback"
            row["treatment_attempt_count"] = 0
            row["treatment_accept_count"] = 0
            row["treatment_fire_count"] = 0
    inactive = mod.recompute_terminal_fields(inactive)

    assert inactive["positive_inactive_and_unimodal_control_results"]["positive_control"][
        "passed"
    ] is False
    assert inactive["status"] == "instrument_failure"
    assert inactive["mode_jump_safety_ready_score"] == 0.0
    assert inactive["retire_mechanism_recommendation"]["recommendation"] == (
        "blocked_no_retirement_recommendation"
    )
    assert "null" not in inactive["honest_verdict"]
    assert mod.validate_artifact(inactive) is True


def test_req_sampler_6281_exactness_regression_blocks_safety(
    artifact: dict[str, object],
) -> None:
    """REQ-SAMPLER-6281-NEGATIVE-TESTS: exactness regressions block safety."""

    regressed = deepcopy(artifact)
    fixture = "ising_ferromagnetic_ring4"
    chain = regressed["exact_distribution_error_by_arm_fixture"]["fixtures"][fixture][
        "mode_jump_runtime"
    ]["chains"][0]
    chain["total_variation_to_target"] = (
        regressed["exact_distribution_error_by_arm_fixture"]["fixtures"][fixture][
            "seeded_fallback"
        ]["chains"][0]["total_variation_to_target"]
        + mod.EQUIVALENCE_MARGINS["total_variation_to_target_delta"]
        + 0.1
    )
    regressed = mod.recompute_terminal_fields(regressed)

    assert regressed["harmful_regressions"]
    assert regressed["mode_jump_safety_ready_score"] == 0.0
    assert regressed["status"] == "blocked_safety"
    assert regressed["retire_mechanism_recommendation"]["recommendation"] == (
        "blocked_no_retirement_recommendation"
    )
    assert mod.validate_artifact(regressed) is True


def test_req_sampler_6281_helper_edges_for_coverage(
    artifact: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-SAMPLER-6281-NEGATIVE-TESTS: helper branches stay deterministic."""

    assert mod.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert mod.sha256_text("x").startswith("sha256:")
    assert mod._mean([]) == 0.0  # noqa: SLF001
    assert mod._interval([]) == [0.0, 0.0]  # noqa: SLF001
    assert mod._interval([0.5]) == [0.5, 0.5]  # noqa: SLF001
    assert mod._intervals_within_equivalence_margins({}) is False  # noqa: SLF001
    assert mod._is_non_toy_family("unimodal_control") is False  # noqa: SLF001
    assert mod._is_non_toy_family("potts") is True  # noqa: SLF001
    with pytest.raises(KeyError):
        mod._chain_by_seed([], mod.SEEDS[0])  # noqa: SLF001

    assert mod.rust_pyo3_backend_receipts([{"success": False}])["chains"] == []
    assert mod.chain_sample_hashes([{"success": False}])["chains"] == []
    failed = mod.unsupported_or_failed_cells(
        [
            {
                "success": False,
                "fixture": "x",
                "family": "f",
                "target_type": "t",
                "seed": 1,
                "arm": "mode_jump_runtime",
                "error_type": "ValueError",
                "message": "bad",
            }
        ],
        [],
    )
    assert failed[0]["classification"] == "matched_cell_failure"
    assert failed[0]["fallback_output_substituted"] is False
    unsupported = mod.unsupported_or_failed_cells(
        [],
        [{"fixture_name": "u", "family": "f", "target_type": "t"}],
    )
    assert len(unsupported) == len(mod.SEEDS) * len(mod.ARMS)
    assert unsupported[0]["classification"] == "not_supported_by_exp6280_typed_backend"

    partial = deepcopy(artifact)
    del partial["exact_distribution_error_by_arm_fixture"]["fixtures"][
        "potts_chain3_q3"
    ]["mode_jump_runtime"]
    assert "potts_chain3_q3" not in mod.paired_intervals_equivalence_margins_and_sample_sizes(
        partial
    )["fixtures"]

    value_branch = deepcopy(artifact)
    value_branch["paired_intervals_equivalence_margins_and_sample_sizes"]["fixtures"][
        "potts_chain3_q3"
    ]["workload_value_improvement_passed"] = True
    family_value = mod.family_level_mixing_value_results(value_branch)
    assert "potts" in family_value["non_toy_positive_mixing_families"]

    continued = deepcopy(artifact)
    continued["family_level_mixing_value_results"]["workload_value_gate_passed"] = True
    assert mod.mode_jump_workload_value_ready_score(continued) == 1.0
    assert mod.retire_mechanism_recommendation(continued)["recommendation"] == (
        "continue_only_with_new_value_gate"
    )
    assert mod.status(continued) == "complete_workload_value_supported"

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json(bad_json)  # noqa: SLF001

    receipts = tmp_path / "receipts.json"
    receipts.write_text(json.dumps(_passing_exit_codes()), encoding="utf-8")
    monkeypatch.setenv("CARNOT_6281_COMMAND_RECEIPTS", str(receipts))
    assert mod._external_test_exit_codes() == _passing_exit_codes()  # noqa: SLF001

    missing = tmp_path / "missing.json"
    monkeypatch.delenv("CARNOT_6281_COMMAND_RECEIPTS", raising=False)
    monkeypatch.setattr(mod, "DEFAULT_RECEIPT_PATH", missing)
    assert mod._external_test_exit_codes() == {}  # noqa: SLF001

    bad_receipts = tmp_path / "bad_receipts.json"
    bad_receipts.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("CARNOT_6281_COMMAND_RECEIPTS", str(bad_receipts))
    with pytest.raises(ValueError, match="command receipt payload"):
        mod._external_test_exit_codes()  # noqa: SLF001

    output = tmp_path / "artifact.json"
    receipts.write_text(json.dumps(_passing_exit_codes()), encoding="utf-8")
    monkeypatch.setenv("CARNOT_6281_COMMAND_RECEIPTS", str(receipts))
    assert mod.main(["--date", "20260810", "--output", str(output)]) == 0
    assert output.exists()
    assert "retired_value_not_ready" in capsys.readouterr().out

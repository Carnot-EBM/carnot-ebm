"""Tests for Exp6269 mode-jump multifamily A/B.

Spec refs: REQ-SAMPLER-6269,
SCENARIO-SAMPLER-6269-MATCHED-SUPPORTED-CELLS,
SCENARIO-SAMPLER-6269-UNSUPPORTED-CELLS-FAIL-CLOSED,
SCENARIO-SAMPLER-6269-SAFETY-VALUE-SEPARATION.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6269_mode_jump_multifamily_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
SUPPORTED_FIXTURE = "exp6237_original_six_state"


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


def test_req_sampler_6269_spec_declares_required_fields_and_scenarios() -> None:
    """REQ-SAMPLER-6269: OpenSpec anchors fields, controls, and value gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLER-6269") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAMPLER-6269-PRECONDITIONS",
        "REQ-SAMPLER-6269-MATCHED-CELLS",
        "REQ-SAMPLER-6269-ACTIVATION",
        "REQ-SAMPLER-6269-EXACTNESS",
        "REQ-SAMPLER-6269-INTERVALS",
        "REQ-SAMPLER-6269-VALUE-GATE",
        "REQ-SAMPLER-6269-NEGATIVE-TESTS",
        "SCENARIO-SAMPLER-6269-MATCHED-SUPPORTED-CELLS",
        "SCENARIO-SAMPLER-6269-UNSUPPORTED-CELLS-FAIL-CLOSED",
        "SCENARIO-SAMPLER-6269-SAFETY-VALUE-SEPARATION",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.TEST_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_sampler_6269_artifact_schema_and_claim_gates(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLER-6269: the terminal JSON validates and stays claim-bounded."""

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
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(written)
    assert mod.validate_artifact(written) is True
    assert written["status"] == "complete_safety_supported_value_not_ready"
    assert written["honest_verdict"].startswith(
        "complete_safety_supported_value_not_ready:"
    )
    assert written["mode_jump_safety_ready_score"] == 1.0
    assert written["mode_jump_workload_value_ready_score"] == 0.0
    assert written["source_mutation_count"] == 0
    assert type(written["source_mutation_count"]) is int
    assert written["hardware_claim_count"] == 0
    assert type(written["hardware_claim_count"]) is int
    assert written["timing_speedup_claimed"] is False
    assert written["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert written["verifier_is_oracle"]["value"] is True


def test_scenario_sampler_6269_matched_supported_cells_activate_treatment(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-6269-MATCHED-SUPPORTED-CELLS: activation gates outcomes."""

    config = artifact["matched_arm_configuration"]
    counts = artifact["treatment_attempt_accept_and_fire_counts_by_fixture"]
    receipts = artifact["rust_pyo3_backend_receipts"]["chains"]
    hashes = artifact["chain_sample_hashes"]["chains"]
    positive = artifact["positive_and_inactive_control_results"]

    assert config["supported_fixtures"] == [SUPPORTED_FIXTURE]
    assert config["matched_seeds"] == list(mod.SEEDS)
    assert config["matched_burn_in"] == mod.BURN_IN
    assert config["matched_retained_sample_count"] == mod.RETAINED_SAMPLE_COUNT
    assert config["matched_proposal_budget"] == mod.PROPOSAL_BUDGET

    treatment = counts["fixtures"][SUPPORTED_FIXTURE]["mode_jump_runtime"]
    assert treatment["active_backend"] == "rust_pyo3"
    assert treatment["attempted_count"] > 0
    assert treatment["accepted_count"] > 0
    assert treatment["treatment_attempt_count"] > 0
    assert treatment["treatment_accept_count"] > 0
    assert treatment["treatment_fire_count"] > 0
    assert counts["activation_proven_before_outcome_comparison"] is True
    assert positive["positive_control"]["passed"] is True
    assert positive["inactive_treatment_control"]["decision"] == "instrument_failure"

    assert len(receipts) == len(mod.SEEDS) * len(mod.ARMS)
    assert len(hashes) == len(receipts)
    for row in receipts:
        assert row["fixture"] == SUPPORTED_FIXTURE
        assert row["transition_budget"]["retained_samples"] == mod.RETAINED_SAMPLE_COUNT
        if row["arm"] == "mode_jump_runtime":
            assert row["active_backend"] == "rust_pyo3"
    for row in hashes:
        assert row["sample_count"] == mod.RETAINED_SAMPLE_COUNT
        assert row["sample_labels_sha256"].startswith("sha256:")


def test_scenario_sampler_6269_exact_metrics_and_paired_intervals(
    artifact: dict[str, object],
) -> None:
    """REQ-SAMPLER-6269-INTERVALS: safety and value are separate decisions."""

    distribution = artifact["exact_distribution_error_by_arm_fixture"]["fixtures"]
    energy = artifact["energy_error_by_arm_fixture"]["fixtures"]
    basin = artifact["basin_occupancy_and_barrier_crossings_by_arm_fixture"]["fixtures"]
    mixing = artifact["autocorrelation_ess_and_acceptance_by_arm_fixture"]["fixtures"]
    paired = artifact["paired_intervals_equivalence_margins_and_sample_sizes"]

    for arm in mod.ARMS:
        assert distribution[SUPPORTED_FIXTURE][arm]["summary"]["chain_count"] == len(mod.SEEDS)
        assert energy[SUPPORTED_FIXTURE][arm]["summary"]["chain_count"] == len(mod.SEEDS)
        assert basin[SUPPORTED_FIXTURE][arm]["summary"]["chain_count"] == len(mod.SEEDS)
        assert mixing[SUPPORTED_FIXTURE][arm]["summary"]["chain_count"] == len(mod.SEEDS)
        assert mixing[SUPPORTED_FIXTURE][arm]["summary"]["mean_acceptance_rate"] > 0.0

    fixture_intervals = paired["fixtures"][SUPPORTED_FIXTURE]
    assert fixture_intervals["paired_seed_count"] == len(mod.SEEDS)
    assert fixture_intervals["distribution_safety_equivalence_passed"] is True
    assert fixture_intervals["workload_value_improvement_passed"] is False
    assert paired["value_gate"]["workload_value_gate_passed"] is False
    assert paired["value_gate"]["non_toy_positive_mixing_family_count"] == 0
    assert artifact["harmful_regressions"] == []
    assert artifact["descriptive_wall_time_by_arm_fixture"]["timing_speedup_claimed"] is False


def test_scenario_sampler_6269_unsupported_cells_fail_closed(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-6269-UNSUPPORTED-CELLS-FAIL-CLOSED: no fallback substitution."""

    unsupported = artifact["unsupported_or_failed_cells"]
    families = {row["family"] for row in unsupported}

    assert "ising_multimodal" in families
    assert "potts" in families
    assert "typed_factor" in families
    assert all(row["fallback_output_substituted"] is False for row in unsupported)
    assert all(row["sample_hash_recorded"] is False for row in unsupported)

    suite = mod._load_upstream_fixture_artifact(REPO)  # noqa: SLF001
    receipt = next(
        row
        for row in suite["exact_enumeration_receipts"]
        if row["fixture_name"] == "potts_chain3_q3"
    )
    row = mod._unsupported_cell_from_receipt(  # noqa: SLF001
        receipt,
        arm="mode_jump_runtime",
        seed=mod.SEEDS[0],
    )
    assert row["classification"] == "unsupported_for_existing_mode_jump_backend"
    assert row["fallback_output_substituted"] is False
    assert row["fail_closed"] is True


def test_req_sampler_6269_negative_controls_recompute_terminal_fields(
    artifact: dict[str, object],
) -> None:
    """REQ-SAMPLER-6269-NEGATIVE-TESTS: inactive treatment fails as instrumentation."""

    inactive = deepcopy(artifact)
    treatment = inactive["treatment_attempt_accept_and_fire_counts_by_fixture"]["fixtures"][
        SUPPORTED_FIXTURE
    ]["mode_jump_runtime"]
    treatment["treatment_attempt_count"] = 0
    treatment["treatment_accept_count"] = 0
    treatment["treatment_fire_count"] = 0
    treatment["active_backend"] = "python_exact_fallback"
    for row in inactive["rust_pyo3_backend_receipts"]["chains"]:
        if row["fixture"] == SUPPORTED_FIXTURE and row["arm"] == "mode_jump_runtime":
            row["active_backend"] = "python_exact_fallback"
            row["treatment_attempt_count"] = 0
            row["treatment_accept_count"] = 0
            row["treatment_fire_count"] = 0
    inactive = mod.recompute_terminal_fields(inactive)

    assert inactive["positive_and_inactive_control_results"]["positive_control"]["passed"] is False
    assert inactive["status"] == "instrument_failure"
    assert inactive["mode_jump_safety_ready_score"] == 0.0
    assert "null" not in inactive["honest_verdict"]
    assert mod.validate_artifact(inactive) is True


def test_req_sampler_6269_validation_rejects_seed_sample_and_acceptance_mismatch(
    artifact: dict[str, object],
) -> None:
    """REQ-SAMPLER-6269-NEGATIVE-TESTS: matched-chain accounting is mechanical."""

    seed_bad = deepcopy(artifact)
    seed_bad["matched_arm_configuration"]["matched_seeds"][0] = 999999
    with pytest.raises(ValueError, match="seed mismatch"):
        mod.validate_artifact(seed_bad)

    sample_bad = deepcopy(artifact)
    sample_bad["chain_sample_hashes"]["chains"][0]["sample_count"] += 1
    with pytest.raises(ValueError, match="sample-count mismatch"):
        mod.validate_artifact(sample_bad)

    acceptance_bad = deepcopy(artifact)
    acceptance_bad["treatment_attempt_accept_and_fire_counts_by_fixture"]["fixtures"][
        SUPPORTED_FIXTURE
    ]["mode_jump_runtime"]["accepted_count"] += 1
    with pytest.raises(ValueError, match="acceptance accounting"):
        mod.validate_artifact(acceptance_bad)

    receipt_seed_bad = deepcopy(artifact)
    receipt_seed_bad["rust_pyo3_backend_receipts"]["chains"][0]["seed"] = 999999
    with pytest.raises(ValueError, match="seed mismatch"):
        mod.validate_artifact(receipt_seed_bad)


def test_req_sampler_6269_validation_rejects_schema_gate_mutations(
    artifact: dict[str, object],
) -> None:
    """REQ-SAMPLER-6269-NEGATIVE-TESTS: required gates fail closed."""

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
            "paired_intervals",
            lambda data: data["paired_intervals_equivalence_margins_and_sample_sizes"][
                "value_gate"
            ].__setitem__("workload_value_gate_passed", True),
        ),
        (
            "harmful_regressions",
            lambda data: data["harmful_regressions"].append({"fixture": SUPPORTED_FIXTURE}),
        ),
        (
            "mode_jump_safety_ready_score",
            lambda data: data.__setitem__("mode_jump_safety_ready_score", 0.5),
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


def test_req_sampler_6269_exactness_regression_blocks_safety(
    artifact: dict[str, object],
) -> None:
    """REQ-SAMPLER-6269-NEGATIVE-TESTS: exactness regressions block safety."""

    regressed = deepcopy(artifact)
    chain = regressed["exact_distribution_error_by_arm_fixture"]["fixtures"][
        SUPPORTED_FIXTURE
    ]["mode_jump_runtime"]["chains"][0]
    chain["total_variation_to_target"] = (
        regressed["exact_distribution_error_by_arm_fixture"]["fixtures"][SUPPORTED_FIXTURE][
            "seeded_fallback"
        ]["chains"][0]["total_variation_to_target"]
        + mod.EQUIVALENCE_MARGINS["total_variation_to_target_delta"]
        + 0.1
    )
    regressed = mod.recompute_terminal_fields(regressed)

    assert regressed["harmful_regressions"]
    assert regressed["mode_jump_safety_ready_score"] == 0.0
    assert regressed["status"] == "blocked_safety"
    assert mod.validate_artifact(regressed) is True


def test_req_sampler_6269_helper_edges_for_coverage(
    artifact: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-SAMPLER-6269-NEGATIVE-TESTS: helper edges stay deterministic."""

    assert mod.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert mod.sha256_text("x").startswith("sha256:")
    assert mod._mean([]) == 0.0  # noqa: SLF001
    assert mod._interval([]) == [0.0, 0.0]  # noqa: SLF001
    assert mod._interval([0.5]) == [0.5, 0.5]  # noqa: SLF001
    assert mod._quality_from_basin_indicator([])["degenerate"] is True  # noqa: SLF001
    assert mod._quality_from_basin_indicator([1.0, 1.0])["degenerate"] is True  # noqa: SLF001

    suite = mod._load_upstream_fixture_artifact(REPO)  # noqa: SLF001
    receipt = next(
        row
        for row in suite["exact_enumeration_receipts"]
        if row["fixture_name"] == SUPPORTED_FIXTURE
    )
    assert mod._distribution_metrics(receipt, ["left_peak"])["kl_target_to_empirical"] == float(  # noqa: SLF001
        "inf"
    )
    assert mod.rust_pyo3_backend_receipts([{"success": False}])["chains"] == []
    with pytest.raises(KeyError):
        mod._chain_by_seed([], mod.SEEDS[0])  # noqa: SLF001

    partial = deepcopy(
        mod.build_artifact(
            root=REPO,
            run_date="20260810",
            duration_s=0.0,
            test_exit_codes=_passing_exit_codes(),
        )
    )
    del partial["exact_distribution_error_by_arm_fixture"]["fixtures"][SUPPORTED_FIXTURE][
        "mode_jump_runtime"
    ]
    assert (
        mod.paired_intervals_equivalence_margins_and_sample_sizes(partial)["fixtures"] == {}
    )
    assert mod.harmful_regressions(partial) == []

    workload = deepcopy(artifact)
    workload["paired_intervals_equivalence_margins_and_sample_sizes"]["value_gate"][
        "workload_value_gate_passed"
    ] = True
    assert mod.status(workload) == "complete_workload_value_supported"

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json(bad_json)  # noqa: SLF001

    receipts = tmp_path / "receipts.json"
    receipts.write_text(json.dumps(_passing_exit_codes()), encoding="utf-8")
    monkeypatch.setenv("CARNOT_6269_COMMAND_RECEIPTS", str(receipts))
    assert mod._external_test_exit_codes() == _passing_exit_codes()  # noqa: SLF001

    missing = tmp_path / "missing.json"
    monkeypatch.delenv("CARNOT_6269_COMMAND_RECEIPTS", raising=False)
    monkeypatch.setattr(mod, "DEFAULT_RECEIPT_PATH", missing)
    assert mod._external_test_exit_codes() == {}  # noqa: SLF001

    bad_receipts = tmp_path / "bad_receipts.json"
    bad_receipts.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("CARNOT_6269_COMMAND_RECEIPTS", str(bad_receipts))
    with pytest.raises(ValueError, match="command receipt payload"):
        mod._external_test_exit_codes()  # noqa: SLF001

    output = tmp_path / "artifact.json"
    receipts.write_text(json.dumps(_passing_exit_codes()), encoding="utf-8")
    monkeypatch.setenv("CARNOT_6269_COMMAND_RECEIPTS", str(receipts))
    assert mod.main(["--date", "20260810", "--output", str(output)]) == 0
    assert output.exists()
    assert "complete_safety_supported_value_not_ready" in capsys.readouterr().out

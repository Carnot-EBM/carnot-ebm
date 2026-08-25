"""Tests for the frustrated spectral k-block scale and Rust parity artifact.

Spec refs: REQ-SAMPLER-6612,
SCENARIO-SAMPLER-6612-INDEPENDENT-SCALE-EVIDENCE,
SCENARIO-SAMPLER-6612-RUST-PARITY-AND-FAIL-CLOSED-VERDICT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_6612_spectral_k_block_scale_rust_parity as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/samplers/spec.md"


def _passing_receipts() -> list[dict[str, object]]:
    return [
        {"command": command, "exit_code": 0, "duration_s": 0.1, "outcome": "passed"}
        for command in mod.DEFAULT_TEST_COMMANDS
    ]


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
    return payload


def test_req_sampler_6612_failed_verification_receipt_names_toolchain_block() -> None:
    """REQ-SAMPLER-6612-ATTACKS keeps failed verification out of a complete claim."""

    receipts = _passing_receipts()
    assert mod._test_receipt_blockers(receipts) == []
    receipts[0] = {**receipts[0], "exit_code": 3, "outcome": "failed"}
    assert mod._test_receipt_blockers(receipts) == [
        {
            "gate": "toolchain_or_test_failure",
            "command": mod.DEFAULT_TEST_COMMANDS[0],
            "exit_code": 3,
        }
    ]
    assert mod._test_receipt_blockers([])[0]["exit_code"] is None
    receipts[0] = {**receipts[0], "exit_code": 0, "duration_s": -1.0}
    assert mod._test_receipt_blockers(receipts)[0]["command"] == mod.DEFAULT_TEST_COMMANDS[0]


def test_req_sampler_6612_spec_declares_scale_parity_and_claim_contracts() -> None:
    """REQ-SAMPLER-6612 fixes all scale requirements before implementation."""

    section = SPEC.read_text(encoding="utf-8").split("### REQ-SAMPLER-6612", 1)[1]
    for anchor in (
        "REQ-SAMPLER-6612-PARTITION",
        "REQ-SAMPLER-6612-TRANSITION",
        "REQ-SAMPLER-6612-FIXTURES",
        "REQ-SAMPLER-6612-CONTROLS",
        "REQ-SAMPLER-6612-REFERENCE",
        "REQ-SAMPLER-6612-PARITY",
        "REQ-SAMPLER-6612-COST",
        "REQ-SAMPLER-6612-ATTACKS",
        "REQ-SAMPLER-6612-BOUNDARY",
        "REQ-SAMPLER-6612-ATOMIC",
        "SCENARIO-SAMPLER-6612-INDEPENDENT-SCALE-EVIDENCE",
        mod.INFERENCE_SUBSTRATE,
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert anchor in section


def test_req_sampler_6612_fixtures_are_frozen_frustrated_and_hashed() -> None:
    """REQ-SAMPLER-6612-FIXTURES freezes six systems at each required size."""

    fixtures = mod.frozen_frustrated_fixtures()
    assert len(fixtures) == 12
    assert {fixture.n_spins for fixture in fixtures} == {16, 32}
    assert all(np.allclose(fixture.couplings, fixture.couplings.T) for fixture in fixtures)
    assert all(
        np.any(fixture.couplings > 0) and np.any(fixture.couplings < 0) for fixture in fixtures
    )
    assert all(fixture.has_non_bipartite_cycle for fixture in fixtures)
    assert all(fixture.competing_modes for fixture in fixtures)
    assert len({fixture.fixture_sha256 for fixture in fixtures}) == 12
    assert mod.frozen_frustrated_fixtures()[0].fixture_sha256 == fixtures[0].fixture_sha256
    with pytest.raises(ValueError, match="fixtures_per_size"):
        mod._selected_fixtures(mod.ExperimentConfig(fixtures_per_size=0))
    assert mod._interval([3.0]) == {"mean": 3.0, "lower": 3.0, "upper": 3.0, "sample_size": 1}
    assert mod.integrated_autocorrelation_time(np.array([1.0])) == 1.0
    assert (
        mod._failure_row(
            fixtures[0], 1, "sequential_gibbs", mod.ExperimentConfig(), RuntimeError("x")
        )["failure"]
        == "RuntimeError:x"
    )


@pytest.fixture(scope="module")
def small_artifact() -> dict[str, object]:
    """Build a small complete matrix without writing tracked state."""

    config = mod.ExperimentConfig(
        treatment_seeds=(101, 102),
        burn_in=20,
        retained_samples=128,
        reference_seeds=(201, 202),
        reference_burn_in=30,
        reference_retained_samples=256,
        fixtures_per_size=1,
        block_size=4,
    )
    return mod.build_artifact(
        root=REPO,
        run_date="20260825",
        config=config,
        test_receipts=_passing_receipts(),
    )


def test_scenario_sampler_6612_artifact_replays_rows_references_costs_and_parity(
    small_artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-6612-INDEPENDENT-SCALE-EVIDENCE checks all row classes."""

    artifact = small_artifact
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["spectral_scale_ready_score"] == 1.0
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["claim_boundaries"]["attached_hardware_execution"] is False

    config = artifact["experiment_config"]
    expected = 2 * len(config["treatment_seeds"]) * len(mod.ARMS)
    assert len(artifact["per_unit_rows"]) == expected
    assert {row["arm"] for row in artifact["per_unit_rows"]} == set(mod.ARMS)
    assert len(artifact["rust_python_parity_rows"]) == 2 * len(config["treatment_seeds"])
    assert len(artifact["fixture_and_reference_receipts"]) == 2
    assert {row["reference_method"] for row in artifact["fixture_and_reference_receipts"]} == {
        "exact_enumeration",
        "independent_long_chains",
    }
    assert all(row["failure"] is None for row in artifact["per_unit_rows"])
    assert all(row["transitions"] == 148 for row in artifact["per_unit_rows"])
    assert all(row["setup_time_s"] >= 0.0 for row in artifact["per_unit_rows"])
    assert all(row["sampling_time_s"] > 0.0 for row in artifact["per_unit_rows"])
    assert all(row["total_time_s"] >= row["sampling_time_s"] for row in artifact["per_unit_rows"])
    assert all(
        row["sample_mismatch_fraction"] == 0.0 for row in artifact["rust_python_parity_rows"]
    )
    assert all(row["passed"] for row in artifact["attack_rows"])
    assert all(
        "principle" in artifact["field_provenance"][field] for field in mod.REQUIRED_ARTIFACT_FIELDS
    )


def test_req_sampler_6612_atomic_write_and_checksum(
    small_artifact: dict[str, object], tmp_path: Path
) -> None:
    """REQ-SAMPLER-6612-ATOMIC replaces one stable JSON object."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    mod.write_json_atomic(output, small_artifact)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == small_artifact
    assert loaded["reproducibility_checksum"] == mod.reproducibility_checksum(loaded)
    assert not list(tmp_path.glob("*.tmp"))


def test_scenario_sampler_6612_mutations_fail_closed(
    small_artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-6612-RUST-PARITY-AND-FAIL-CLOSED-VERDICT."""

    cases: list[tuple[dict[str, object], str]] = []

    missing = deepcopy(small_artifact)
    missing["per_unit_rows"].pop()
    cases.append((_rehash(missing), "row matrix"))

    reference = deepcopy(small_artifact)
    reference["fixture_and_reference_receipts"][0]["independent_of_treatment"] = False
    cases.append((_rehash(reference), "reference independence"))

    burn = deepcopy(small_artifact)
    burn["per_unit_rows"][0]["burn_in"] = 0
    cases.append((_rehash(burn), "burn-in"))

    charge = deepcopy(small_artifact)
    charge["per_unit_rows"][0]["transitions"] -= 1
    cases.append((_rehash(charge), "transition charge"))

    setup = deepcopy(small_artifact)
    spectral = next(
        row for row in setup["per_unit_rows"] if row["arm"] == "spectral_k_block_python"
    )
    spectral["setup_time_s"] = 0.0
    cases.append((_rehash(setup), "setup charge"))

    parity = deepcopy(small_artifact)
    parity["rust_python_parity_rows"][0]["sample_mismatch_fraction"] = 0.1
    cases.append((_rehash(parity), "parity"))

    identity = deepcopy(small_artifact)
    identity["partition_rows"][1]["partition_sha256"] = identity["partition_rows"][0][
        "partition_sha256"
    ]
    cases.append((_rehash(identity), "partition identity"))

    hardware = deepcopy(small_artifact)
    hardware["claim_boundaries"]["attached_hardware_execution"] = True
    cases.append((_rehash(hardware), "hardware claim"))

    protected = deepcopy(small_artifact)
    protected["protected_files_unchanged"]["all_unchanged"] = False
    cases.append((_rehash(protected), "protected files"))

    substrate = deepcopy(small_artifact)
    substrate["inference_substrate"] = "fpga"
    cases.append((_rehash(substrate), "inference_substrate"))

    verifier = deepcopy(small_artifact)
    verifier["verifier_is_oracle"] = True
    cases.append((_rehash(verifier), "verifier_is_oracle"))

    verdict = deepcopy(small_artifact)
    verdict["verdict_class"] = "invented"
    cases.append((_rehash(verdict), "verdict_class"))

    binary = deepcopy(small_artifact)
    binary["spectral_scale_ready_score"] = 0.5
    cases.append((_rehash(binary), "binary"))

    failure = deepcopy(small_artifact)
    failure["per_unit_rows"][0]["failure"] = "forced"
    cases.append((_rehash(failure), "chain failure"))

    attack = deepcopy(small_artifact)
    attack["attack_rows"][0]["passed"] = False
    cases.append((_rehash(attack), "attack row"))

    provenance = deepcopy(small_artifact)
    del provenance["field_provenance"]["status"]
    cases.append((_rehash(provenance), "field_provenance"))

    checksum = deepcopy(small_artifact)
    checksum["duration_s"] = float(checksum["duration_s"]) + 1.0
    cases.append((checksum, "checksum"))

    for payload, message in cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    missing = deepcopy(small_artifact)
    del missing["status"]
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)


def test_scenario_sampler_6612_blocked_rows_remain_valid_and_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLER-6612-ATOMIC retains a genuine Rust parity block."""

    def unavailable(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced Rust toolchain block")

    monkeypatch.setattr(mod, "run_rust_chain", unavailable)
    config = mod.ExperimentConfig(
        treatment_seeds=(101,),
        burn_in=2,
        retained_samples=8,
        reference_seeds=(201,),
        reference_burn_in=2,
        reference_retained_samples=8,
        fixtures_per_size=1,
        block_size=2,
    )
    artifact = mod.build_artifact(root=REPO, config=config, test_receipts=_passing_receipts())

    assert artifact["spectral_scale_ready_score"] == 0.0
    assert str(artifact["status"]).startswith("blocked_")
    assert artifact["gate_check_summary"]
    assert mod.validate_artifact(artifact) is True

    unnamed = deepcopy(artifact)
    unnamed["gate_check_summary"] = []
    with pytest.raises(ValueError, match="named gate"):
        mod.validate_artifact(_rehash(unnamed))


def test_req_sampler_6612_receipt_loading_and_run_wrapper(
    small_artifact: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-SAMPLER-6612-ATOMIC loads receipts and keeps writes redirectable."""

    monkeypatch.delenv("CARNOT_6612_TEST_RECEIPTS", raising=False)
    assert mod._load_test_receipts() == []
    receipt_path = tmp_path / "receipts.json"
    receipt_path.write_text(json.dumps(_passing_receipts()), encoding="utf-8")
    monkeypatch.setenv("CARNOT_6612_TEST_RECEIPTS", str(receipt_path))
    assert mod._load_test_receipts() == _passing_receipts()
    receipt_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        mod._load_test_receipts()

    output = tmp_path / "wrapper.json"
    monkeypatch.setattr(mod, "build_artifact", lambda **kwargs: deepcopy(small_artifact))
    written: list[Path] = []
    monkeypatch.setattr(mod, "write_json_atomic", lambda path, payload: written.append(path))
    monkeypatch.setattr(mod, "validate_artifact", lambda payload: True)
    receipt_path.write_text(json.dumps(_passing_receipts()), encoding="utf-8")
    result = mod.run_experiment(root=REPO, output_path=output)
    assert result == small_artifact
    assert written == [output]


@pytest.mark.parametrize(
    "config",
    [
        mod.ExperimentConfig(treatment_seeds=()),
        mod.ExperimentConfig(burn_in=0),
        mod.ExperimentConfig(reference_burn_in=0),
    ],
)
def test_req_sampler_6612_invalid_experiment_budgets_fail_before_sampling(
    config: mod.ExperimentConfig,
) -> None:
    """REQ-SAMPLER-6612-CONTROLS rejects missing or zero chain budgets."""

    with pytest.raises(ValueError):
        mod.build_artifact(root=REPO, config=config)

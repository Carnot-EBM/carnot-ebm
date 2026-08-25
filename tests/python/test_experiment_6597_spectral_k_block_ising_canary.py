"""Tests for the bounded spectral k-block Ising sampler canary.

Spec refs: REQ-SAMPLER-6597,
SCENARIO-SAMPLER-6597-PAPER-FAITHFUL-PARTITION,
SCENARIO-SAMPLER-6597-MATCHED-EXACT-EVIDENCE,
SCENARIO-SAMPLER-6597-FAIL-CLOSED-VERDICT.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_6597_spectral_k_block_ising_canary as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/samplers/spec.md"


def _passing_receipts() -> list[dict[str, object]]:
    return [
        {"command": command, "exit_code": 0, "duration_s": 1.0}
        for command in mod.DEFAULT_TEST_COMMANDS
    ]


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
    return payload


def test_req_sampler_6597_spec_declares_the_complete_canary_contract() -> None:
    """REQ-SAMPLER-6597: OpenSpec fixes the canary before implementation."""

    section = SPEC.read_text(encoding="utf-8").split("### REQ-SAMPLER-6597", 1)[1]
    for anchor in (
        "REQ-SAMPLER-6597-PARTITION",
        "REQ-SAMPLER-6597-MATCHED",
        "REQ-SAMPLER-6597-EXACT",
        "REQ-SAMPLER-6597-FLOOR",
        "REQ-SAMPLER-6597-PAIRED",
        "REQ-SAMPLER-6597-ATTACKS",
        "REQ-SAMPLER-6597-BOUNDARY",
        "REQ-SAMPLER-6597-ATOMIC",
        "SCENARIO-SAMPLER-6597-PAPER-FAITHFUL-PARTITION",
        "SCENARIO-SAMPLER-6597-MATCHED-EXACT-EVIDENCE",
        "SCENARIO-SAMPLER-6597-FAIL-CLOSED-VERDICT",
        mod.INFERENCE_SUBSTRATE,
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert anchor in section


def test_scenario_sampler_6597_exact_operator_and_spectral_partitions() -> None:
    """SCENARIO-SAMPLER-6597-PAPER-FAITHFUL-PARTITION checks P-squared rounding."""

    fixture = mod.frozen_fixtures()[1]
    states = mod.enumerate_states(fixture.n_spins)
    target = mod.exact_distribution(fixture, states)
    operator = mod.heat_bath_transition_matrix(fixture, states)

    assert states.shape == (64, 6)
    assert np.allclose(operator.sum(axis=1), 1.0)
    assert np.min(operator) >= 0.0
    assert np.max(np.abs(target[:, None] * operator - target[None, :] * operator.T)) < 1.0e-12
    assert "exact_target" not in inspect.signature(mod.run_chain).parameters

    partitions = [mod.build_spectral_partition(fixture, states, k) for k in mod.BLOCK_SIZES]
    for k, partition in zip(mod.BLOCK_SIZES, partitions, strict=True):
        members = sorted(index for block in partition.blocks for index in block)
        assert members == list(range(len(states)))
        assert len(partition.blocks) == k
        assert partition.operator == "squared_random_site_heat_bath_P2"
        assert partition.eigensolver == "numpy.linalg.eigh_symmetric_similarity"
        assert tuple(sorted(partition.selected_eigenvalues)) == partition.selected_eigenvalues
        assert partition.objective_f >= 0.0
        assert partition.setup_time_s >= 0.0
        assert partition.failure is None

    replay = mod.build_spectral_partition(fixture, states, mod.BLOCK_SIZES[0])
    assert replay.blocks == partitions[0].blocks
    with pytest.raises(ValueError, match="block count"):
        mod.build_spectral_partition(fixture, states, 1)


def test_scenario_sampler_6597_matched_chain_and_independent_evaluator() -> None:
    """SCENARIO-SAMPLER-6597-MATCHED-EXACT-EVIDENCE enforces the sample floor."""

    fixture = mod.frozen_fixtures()[2]
    states = mod.enumerate_states(fixture.n_spins)
    target = mod.exact_distribution(fixture, states)
    partition = mod.build_spectral_partition(fixture, states, 2)
    stream = mod.matched_random_stream(mod.SEEDS[0], mod.TRANSITION_BUDGET)
    start = mod.initial_state_index(mod.SEEDS[0], len(states))

    gibbs = mod.run_chain(
        fixture,
        states,
        None,
        initial_index=start,
        random_stream=stream,
        burn_in=mod.BURN_IN,
        retained_samples=mod.RETAINED_SAMPLES,
    )
    spectral = mod.run_chain(
        fixture,
        states,
        partition,
        initial_index=start,
        random_stream=stream,
        burn_in=mod.BURN_IN,
        retained_samples=mod.RETAINED_SAMPLES,
    )
    replay = mod.run_chain(
        fixture,
        states,
        partition,
        initial_index=start,
        random_stream=stream,
        burn_in=mod.BURN_IN,
        retained_samples=mod.RETAINED_SAMPLES,
    )

    for run in (gibbs, spectral):
        assert len(run.sample_indices) == mod.RETAINED_SAMPLES
        assert run.transitions == mod.TRANSITION_BUDGET
        assert run.failure is None
    assert spectral.sample_sha256 == replay.sample_sha256
    assert gibbs.random_stream_sha256 == spectral.random_stream_sha256
    assert gibbs.initial_state_index == spectral.initial_state_index

    metrics = mod.evaluate_samples(fixture, states, target, spectral.sample_indices)
    assert metrics["retained_sample_count"] == mod.RETAINED_SAMPLES
    assert metrics["total_variation_error"] < 0.2
    assert metrics["mean_l2_error"] >= 0.0
    assert metrics["covariance_frobenius_error"] >= 0.0
    assert 0.0 < metrics["effective_sample_size"] <= mod.RETAINED_SAMPLES
    assert metrics["integrated_autocorrelation_time"] >= 1.0


@pytest.fixture(scope="module")
def complete_artifact() -> dict[str, object]:
    """Build the full preregistered matrix once for all schema tests."""

    return mod.build_artifact(
        root=REPO,
        run_date="20260825",
        test_receipts=_passing_receipts(),
    )


def test_scenario_sampler_6597_complete_artifact_has_per_seed_evidence(
    complete_artifact: dict[str, object],
) -> None:
    """REQ-SAMPLER-6597-ATOMIC: every required row and receipt is present."""

    artifact = complete_artifact
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] in mod.ALLOWED_VERDICT_CLASSES
    assert str(artifact["honest_verdict"]).startswith("complete:")
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True

    expected_rows = len(mod.frozen_fixtures()) * len(mod.SEEDS) * (1 + len(mod.BLOCK_SIZES))
    assert len(artifact["per_unit_rows"]) == expected_rows
    assert len(artifact["sampler_run_rows"]) == expected_rows
    assert len(artifact["exact_distribution_comparison"]) == expected_rows
    assert len(artifact["spectral_partition_rows"]) == len(mod.frozen_fixtures()) * len(
        mod.BLOCK_SIZES
    )
    assert len(artifact["paired_statistical_receipts"]) == len(mod.frozen_fixtures()) * len(
        mod.BLOCK_SIZES
    )
    assert len(artifact["fixture_receipts"]) == 3
    assert all(row["passed"] for row in artifact["attack_rows"])
    assert all(row["retained_sample_count"] >= 10_000 for row in artifact["per_unit_rows"])
    assert {row["seed"] for row in artifact["per_unit_rows"]} == set(mod.SEEDS)
    assert {row["family"] for row in artifact["per_unit_rows"]} == {
        "independent",
        "ferromagnetic",
        "frustrated",
    }
    assert all(
        "principle" in artifact["field_provenance"][field] for field in mod.REQUIRED_ARTIFACT_FIELDS
    )


def test_req_sampler_6597_atomic_write_and_checksum(
    complete_artifact: dict[str, object], tmp_path: Path
) -> None:
    """REQ-SAMPLER-6597-ATOMIC writes one stable JSON object by replacement."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    mod.write_json_atomic(output, complete_artifact)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == complete_artifact
    assert loaded["reproducibility_checksum"] == mod.reproducibility_checksum(loaded)
    assert not list(tmp_path.glob("*.tmp"))


def test_scenario_sampler_6597_attacks_and_schema_mutations_fail_closed(
    complete_artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-6597-FAIL-CLOSED-VERDICT rejects favorable evidence edits."""

    cases: list[tuple[dict[str, object], str]] = []

    sample_floor = deepcopy(complete_artifact)
    sample_floor["per_unit_rows"][0]["retained_sample_count"] = 9_999
    cases.append((_rehash(sample_floor), "sample floor"))

    seed_drop = deepcopy(complete_artifact)
    seed_drop["per_unit_rows"] = [
        row for row in seed_drop["per_unit_rows"] if row["seed"] != mod.SEEDS[-1]
    ]
    cases.append((_rehash(seed_drop), "row matrix"))

    omitted_setup = deepcopy(complete_artifact)
    spectral_row = next(
        row for row in omitted_setup["per_unit_rows"] if row["arm"] == "spectral_k_block"
    )
    spectral_row["charged_setup_time_s"] = 0.0
    cases.append((_rehash(omitted_setup), "setup cost"))

    oracle = deepcopy(complete_artifact)
    oracle["verifier_is_oracle"] = True
    cases.append((_rehash(oracle), "verifier_is_oracle"))

    rebrand = deepcopy(complete_artifact)
    rebrand["method_source_receipt"]["hardware_claimed"] = True
    cases.append((_rehash(rebrand), "hardware claim"))

    pimi = deepcopy(complete_artifact)
    pimi["method_source_receipt"]["pimi_claimed"] = True
    cases.append((_rehash(pimi), "PIMI claim"))

    protected = deepcopy(complete_artifact)
    protected["protected_files_unchanged"]["all_unchanged"] = False
    cases.append((_rehash(protected), "protected files"))

    transition = deepcopy(complete_artifact)
    transition["per_unit_rows"][0]["transitions"] -= 1
    cases.append((_rehash(transition), "transition budget"))

    sampler_rows = deepcopy(complete_artifact)
    sampler_rows["sampler_run_rows"].pop()
    cases.append((_rehash(sampler_rows), "sampler or exact"))

    partitions = deepcopy(complete_artifact)
    partitions["spectral_partition_rows"].pop()
    cases.append((_rehash(partitions), "spectral partition row"))

    attack = deepcopy(complete_artifact)
    attack["attack_rows"][0]["passed"] = False
    cases.append((_rehash(attack), "attack row"))

    provenance = deepcopy(complete_artifact)
    del provenance["field_provenance"]["status"]
    cases.append((_rehash(provenance), "field_provenance"))

    verdict = deepcopy(complete_artifact)
    verdict["verdict_class"] = "invented"
    cases.append((_rehash(verdict), "verdict_class"))

    substrate = deepcopy(complete_artifact)
    substrate["inference_substrate"] = "fpga"
    cases.append((_rehash(substrate), "inference_substrate"))

    for payload, message in cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    checksum = deepcopy(complete_artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum)

    missing = deepcopy(complete_artifact)
    del missing["status"]
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)


def test_req_sampler_6597_helpers_cover_degenerate_and_paired_cases() -> None:
    """REQ-SAMPLER-6597-PAIRED: numerical helpers stay finite and conservative."""

    assert mod.integrated_autocorrelation_time(np.ones(8)) == 1.0
    assert mod.integrated_autocorrelation_time(np.array([1.0])) == 1.0
    assert mod.lag_one_autocorrelation(np.ones(8)) == 0.0
    assert mod.lag_one_autocorrelation(np.ones(1)) == 0.0
    assert mod.lag_one_autocorrelation(np.arange(8, dtype=float)) > 0.0

    interval = mod.paired_interval([0.1, 0.2, 0.3, 0.4, 0.5])
    assert interval["sample_size"] == 5
    assert interval["lower"] < interval["mean"] < interval["upper"]
    singleton = mod.paired_interval([0.25])
    assert singleton == {"mean": 0.25, "lower": 0.25, "upper": 0.25, "sample_size": 1}

    fixture = mod.frozen_fixtures()[0]
    with pytest.raises(ValueError, match="random stream"):
        mod.run_chain(
            fixture,
            mod.enumerate_states(fixture.n_spins),
            None,
            initial_index=0,
            random_stream=np.zeros((3, 2)),
            burn_in=2,
            retained_samples=2,
        )

    empty_blocks = mod._canonical_blocks(np.zeros(4, dtype=np.int64), 2)  # noqa: SLF001
    assert empty_blocks is None
    zero_embedding = np.zeros((4, 2), dtype=np.float64)
    empty_kmeans = mod._weighted_kmeans_candidate(  # noqa: SLF001
        zero_embedding,
        np.full(4, 0.25),
        2,
        np.random.default_rng(1),
    )
    assert empty_kmeans is None


def test_req_sampler_6597_empty_rounding_receipts_and_runner_paths(
    complete_artifact: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLER-6597-ATTACKS covers blocked rounding and receipt boundaries."""

    monkeypatch.setattr(mod, "_weighted_kmeans_candidate", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="no nonempty partition"):
        mod._round_spectral_embedding(  # noqa: SLF001
            np.zeros((4, 2)),
            np.full(4, 0.25),
            np.eye(4),
            3,
        )

    monkeypatch.delenv("CARNOT_6597_TEST_RECEIPTS", raising=False)
    assert mod._load_test_receipts() == []  # noqa: SLF001

    invalid_receipts = tmp_path / "invalid-receipts.json"
    invalid_receipts.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CARNOT_6597_TEST_RECEIPTS", str(invalid_receipts))
    with pytest.raises(ValueError, match="must contain a list"):
        mod._load_test_receipts()  # noqa: SLF001

    valid_receipts = tmp_path / "valid-receipts.json"
    valid_receipts.write_text(json.dumps(_passing_receipts()), encoding="utf-8")
    monkeypatch.setenv("CARNOT_6597_TEST_RECEIPTS", str(valid_receipts))
    assert mod._load_test_receipts() == _passing_receipts()  # noqa: SLF001

    output = tmp_path / "runner.json"
    written: list[Path] = []
    monkeypatch.setattr(mod, "build_artifact", lambda **kwargs: complete_artifact)
    monkeypatch.setattr(mod, "write_json_atomic", lambda path, payload: written.append(path))
    result = mod.run_experiment(root=REPO, run_date="20260825", output_path=output)
    assert result is complete_artifact
    assert written == [output]

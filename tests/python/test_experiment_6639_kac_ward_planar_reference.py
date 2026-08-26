"""Tests for the bounded exact Kac--Ward planar-Ising reference.

Spec refs: REQ-SAMPLER-6639,
SCENARIO-SAMPLER-6639-EXACT-AUTOREGRESSIVE-PARITY,
SCENARIO-SAMPLER-6639-FAIL-CLOSED-NUMERICS,
SCENARIO-SAMPLER-6639-SEALED-INDEPENDENT-EVIDENCE.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_6639_kac_ward_planar_reference as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/samplers/spec.md"


def _passing_receipts() -> list[dict[str, object]]:
    return [
        {
            "scope": scope,
            "command": f"verified {scope}",
            "exit_code": 0,
            "duration_s": 0.01,
        }
        for scope in mod.REQUIRED_TEST_SCOPES
    ]


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
    return payload


def _path_instance(n_spins: int, *, instance_id: str = "path") -> mod.PlanarIsingInstance:
    return mod.PlanarIsingInstance(
        instance_id=instance_id,
        n_spins=n_spins,
        edges=tuple((index, index + 1, 0.2) for index in range(n_spins - 1)),
        fields=tuple(0.0 for _ in range(n_spins)),
        positions=tuple((float(index), 0.0) for index in range(n_spins)),
        order=tuple(range(n_spins)),
        temperature=1.0,
        seed=6639,
    )


def test_req_sampler_6639_spec_precedes_implementation() -> None:
    """REQ-SAMPLER-6639 fixes every exact-reference gate in OpenSpec."""

    section = SPEC.read_text(encoding="utf-8").split("### REQ-SAMPLER-6639", 1)[1]
    for anchor in (
        "REQ-SAMPLER-6639-PLANARITY",
        "REQ-SAMPLER-6639-ZERO-FIELD",
        "REQ-SAMPLER-6639-AUXILIARY",
        "REQ-SAMPLER-6639-CONDITIONING",
        "REQ-SAMPLER-6639-LIKELIHOOD",
        "REQ-SAMPLER-6639-RNG",
        "REQ-SAMPLER-6639-ENUMERATION",
        "REQ-SAMPLER-6639-PRECISION",
        "REQ-SAMPLER-6639-ATTACKS",
        "REQ-SAMPLER-6639-ATOMIC",
        "SCENARIO-SAMPLER-6639-FAIL-CLOSED-NUMERICS",
        mod.INFERENCE_SUBSTRATE,
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert anchor in section


def test_req_sampler_6639_fixtures_are_frozen_planar_zero_field_and_hashed() -> None:
    """REQ-SAMPLER-6639-PLANARITY freezes the bounded case matrix."""

    config = mod.ExperimentConfig()
    fixtures = mod.frozen_instances(config)
    assert len(fixtures) == 12
    assert {fixture.graph_id for fixture in fixtures} == {
        "mixed_square_n4",
        "frustrated_triangle_tail_n4",
        "mixed_ladder_n6",
    }
    assert {fixture.temperature for fixture in fixtures} == set(config.temperatures)
    assert {fixture.seed for fixture in fixtures} == set(config.seeds)
    assert len({fixture.fixture_sha256 for fixture in fixtures}) == len(fixtures)
    assert mod.frozen_instances(config)[0].fixture_sha256 == fixtures[0].fixture_sha256
    for fixture in fixtures:
        receipt = mod.validate_instance(fixture, enumeration_limit=config.enumeration_limit)
        assert receipt["planar"] is True
        assert receipt["connected_search_order"] is True
        assert not any(fixture.fields)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda item: replace(item, fields=(0.0, 0.0, 1.0e-9, 0.0)), "zero field"),
        (lambda item: replace(item, order=(0, 2, 1, 3)), "connected search"),
        (lambda item: replace(item, edges=item.edges + ((0, 0, 0.2),)), "self-loop"),
        (lambda item: replace(item, edges=item.edges + (item.edges[0],)), "duplicate edge"),
        (
            lambda item: replace(
                item,
                positions=((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)),
            ),
            "straight-line embedding",
        ),
        (
            lambda item: replace(
                item,
                edges=item.edges[:-1] + ((0, 3, float("nan")),),
            ),
            "finite coupling",
        ),
        (lambda item: replace(item, temperature=0.0), "temperature"),
    ],
)
def test_scenario_sampler_6639_invalid_instances_fail_closed(mutator: object, message: str) -> None:
    """SCENARIO-SAMPLER-6639-FAIL-CLOSED-NUMERICS rejects bad inputs."""

    square = mod.frozen_instances(mod.ExperimentConfig())[0]
    with pytest.raises(mod.UnsupportedInstanceError, match=message):
        mod.validate_instance(mutator(square), enumeration_limit=mod.ENUMERATION_LIMIT)  # type: ignore[operator]


def test_req_sampler_6639_rejects_nonplanar_oversized_and_nonfinite_coordinates() -> None:
    """REQ-SAMPLER-6639-PLANARITY rejects unsupported graph classes."""

    edges = tuple((left, right, 0.2) for left in range(3) for right in range(3, 6))
    k33 = mod.PlanarIsingInstance(
        instance_id="k33",
        n_spins=6,
        edges=edges,
        fields=(0.0,) * 6,
        positions=tuple((float(index % 3), float(index // 3)) for index in range(6)),
        order=tuple(range(6)),
        temperature=1.0,
        seed=1,
    )
    with pytest.raises(mod.UnsupportedInstanceError, match="nonplanar"):
        mod.validate_instance(k33)
    with pytest.raises(mod.UnsupportedInstanceError, match="enumeration limit"):
        mod.validate_instance(_path_instance(mod.ENUMERATION_LIMIT + 1))
    bad_position = replace(
        _path_instance(3), positions=((0.0, 0.0), (float("inf"), 0.0), (2.0, 0.0))
    )
    with pytest.raises(mod.UnsupportedInstanceError, match="finite coordinate"):
        mod.validate_instance(bad_position)


def test_req_sampler_6639_kac_ward_partition_matches_independent_enumeration() -> None:
    """REQ-SAMPLER-6639-CONDITIONING checks the determinant against enumeration."""

    for instance in mod.frozen_instances(
        mod.ExperimentConfig(
            graph_ids=("mixed_square_n4", "mixed_ladder_n6"),
            temperatures=(0.8,),
            seeds=(6639001,),
            sample_count=4,
        )
    ):
        enumeration = mod.enumerate_reference(instance)
        log_partition, diagnostics = mod.kac_ward_log_partition(instance)
        assert np.exp(log_partition) == pytest.approx(
            enumeration["partition_function"], rel=mod.PARITY_TOLERANCES["partition"]
        )
        assert diagnostics["condition_number"] <= mod.CONDITION_NUMBER_LIMIT
        assert diagnostics["determinant_phase_abs"] <= mod.DETERMINANT_PHASE_TOLERANCE


def test_scenario_sampler_6639_all_probabilities_conditionals_and_moments_match() -> None:
    """SCENARIO-SAMPLER-6639-EXACT-AUTOREGRESSIVE-PARITY checks all math."""

    instance = mod.frozen_instances(
        mod.ExperimentConfig(
            graph_ids=("frustrated_triangle_tail_n4",),
            temperatures=(1.4,),
            seeds=(6639002,),
            sample_count=4,
        )
    )[0]
    parity = mod.cross_check_instance(instance)
    assert parity["passed"] is True
    assert parity["partition_error"] <= mod.PARITY_TOLERANCES["partition"]
    assert parity["state_probability_error_max"] <= mod.PARITY_TOLERANCES["probability"]
    assert parity["conditional_error_max"] <= mod.PARITY_TOLERANCES["conditional"]
    assert parity["first_moment_error_max"] <= mod.PARITY_TOLERANCES["moment"]
    assert parity["second_moment_error_max"] <= mod.PARITY_TOLERANCES["moment"]
    assert parity["energy_moment_error"] <= mod.PARITY_TOLERANCES["moment"]
    assert parity["normalization_error"] <= mod.PARITY_TOLERANCES["normalization"]
    assert parity["state_count"] == 16
    assert parity["unique_prefix_count"] == 15


def test_req_sampler_6639_normalized_likelihood_is_chain_product() -> None:
    """REQ-SAMPLER-6639-LIKELIHOOD keeps likelihood normalized and exact."""

    instance = mod.frozen_instances(
        mod.ExperimentConfig(graph_ids=("mixed_square_n4",), temperatures=(0.8,), seeds=(6639001,))
    )[0]
    enumeration = mod.enumerate_reference(instance)
    total = 0.0
    for state, expected in zip(enumeration["states"], enumeration["probabilities"], strict=True):
        likelihood = mod.autoregressive_likelihood(instance, tuple(state))
        assert likelihood["probability"] == pytest.approx(
            expected, abs=mod.PARITY_TOLERANCES["probability"]
        )
        assert likelihood["probability"] == pytest.approx(
            np.prod(likelihood["selected_conditionals"]), abs=1.0e-15
        )
        total += likelihood["probability"]
    assert total == pytest.approx(1.0, abs=mod.PARITY_TOLERANCES["normalization"])


def test_req_sampler_6639_precision_condition_and_branch_failures_are_named() -> None:
    """REQ-SAMPLER-6639-PRECISION rejects unsafe determinant arithmetic."""

    instance = mod.frozen_instances(
        mod.ExperimentConfig(graph_ids=("mixed_square_n4",), temperatures=(0.8,), seeds=(6639001,))
    )[0]
    with pytest.raises(mod.KacWardPrecisionError, match="unsupported precision"):
        mod.kac_ward_log_partition(instance, precision="float32")
    with pytest.raises(mod.KacWardPrecisionError, match="ill-conditioned"):
        mod.kac_ward_log_partition(instance, condition_limit=1.0)
    with pytest.raises(mod.KacWardPrecisionError, match="determinant branch"):
        mod._validated_logdet(np.array([[-1.0 + 0.0j]]), condition_limit=10.0)

    singular = replace(
        instance,
        instance_id="singular_cycle",
        edges=tuple((left, right, 1.0e308) for left, right, _ in instance.edges),
    )
    with pytest.raises(mod.KacWardPrecisionError, match="singular|ill-conditioned"):
        mod.kac_ward_log_partition(singular)


def test_req_sampler_6639_rng_is_reproducible_and_domain_separated() -> None:
    """REQ-SAMPLER-6639-RNG gives each case a fresh deterministic stream."""

    instances = mod.frozen_instances(
        mod.ExperimentConfig(
            graph_ids=("mixed_square_n4",), temperatures=(0.8, 1.4), seeds=(6639001,)
        )
    )
    first = mod.sample_reference_rows(instances[0], sample_count=16)
    replay = mod.sample_reference_rows(instances[0], sample_count=16)
    other = mod.sample_reference_rows(instances[1], sample_count=16)
    assert first["rows"] == replay["rows"]
    assert first["sample_rows_sha256"] == replay["sample_rows_sha256"]
    assert first["domain_seed"] == replay["domain_seed"]
    assert first["domain_seed"] != other["domain_seed"]
    assert first["sample_rows_sha256"] != other["sample_rows_sha256"]
    assert all(row["likelihood_parity_passed"] is True for row in first["rows"])


def test_req_sampler_6639_orientation_and_graph_permutation_are_invariant() -> None:
    """REQ-SAMPLER-6639-ATTACKS attacks orientation and node permutation."""

    instance = mod.frozen_instances(
        mod.ExperimentConfig(graph_ids=("mixed_square_n4",), temperatures=(0.8,), seeds=(6639001,))
    )[0]
    reversed_edges = replace(
        instance,
        instance_id="reversed_edges",
        edges=tuple((right, left, coupling) for left, right, coupling in instance.edges),
    )
    assert mod.kac_ward_log_partition(reversed_edges)[0] == pytest.approx(
        mod.kac_ward_log_partition(instance)[0], abs=1.0e-12
    )

    permuted, inverse = mod.permute_instance(instance, (2, 0, 3, 1))
    original = mod.enumerate_reference(instance)
    changed = mod.enumerate_reference(permuted)
    original_probabilities = {
        tuple(state): probability
        for state, probability in zip(original["states"], original["probabilities"], strict=True)
    }
    for state, probability in zip(changed["states"], changed["probabilities"], strict=True):
        unpermuted = tuple(state[inverse[index]] for index in range(instance.n_spins))
        assert probability == pytest.approx(original_probabilities[unpermuted], abs=1.0e-14)


@pytest.fixture(scope="module")
def small_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build passing evidence under a temporary destination."""

    temporary = tmp_path_factory.mktemp("exp6639")
    config = mod.ExperimentConfig(
        graph_ids=("mixed_square_n4",),
        temperatures=(0.8,),
        seeds=(6639001, 6639002),
        sample_count=16,
    )
    return mod.build_artifact(
        root=REPO,
        run_date="20260826",
        config=config,
        sample_dir=temporary / "samples",
        test_receipts=_passing_receipts(),
    )


def test_scenario_sampler_6639_artifact_has_rows_banks_attacks_and_provenance(
    small_artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-6639-SEALED-INDEPENDENT-EVIDENCE checks evidence."""

    assert mod.validate_artifact(small_artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(small_artifact)
    assert small_artifact["status"] == "complete_bounded_exact_reference_ready"
    assert str(small_artifact["honest_verdict"]).startswith("complete:")
    assert small_artifact["verdict_class"] == "positive"
    assert small_artifact["kac_ward_reference_ready_score"] == 1.0
    assert small_artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert small_artifact["verifier_is_oracle"] is False
    assert len(small_artifact["per_instance_rows"]) == 2
    assert len(small_artifact["reference_sample_manifest"]) == 2
    assert {row["attack"] for row in small_artifact["attack_rows"]} == set(mod.REQUIRED_ATTACKS)
    assert all(row["passed"] for row in small_artifact["attack_rows"])
    assert small_artifact["parity_metrics"]["all_instances_passed"] is True
    assert small_artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert set(small_artifact["field_provenance"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    for manifest in small_artifact["reference_sample_manifest"]:
        path = Path(manifest["resolved_path"])
        assert path.is_file()
        assert mod.sha256_file(path) == manifest["sha256"]
        assert path.stat().st_mode & 0o222 == 0
        assert manifest["normalized_likelihoods"]


def test_req_sampler_6639_atomic_writes_replace_and_leave_no_temporary_file(
    small_artifact: dict[str, object], tmp_path: Path
) -> None:
    """REQ-SAMPLER-6639-ATOMIC syncs and replaces the terminal JSON."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    receipt = mod.write_json_atomic(output, small_artifact)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == small_artifact
    assert receipt["atomic_replace"] is True
    assert receipt["file_fsync"] is True
    assert receipt["directory_fsync"] is True
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="nonfinite JSON"):
        mod.write_json_atomic(tmp_path / "bad.json", {"bad": float("nan")})


def test_req_sampler_6639_test_receipts_and_math_failures_block_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-SAMPLER-6639-BOUNDARY makes readiness conjunctive."""

    receipts = _passing_receipts()
    receipts[0]["exit_code"] = 1
    config = mod.ExperimentConfig(
        graph_ids=("mixed_square_n4",),
        temperatures=(0.8,),
        seeds=(6639001,),
        sample_count=4,
    )
    blocked = mod.build_artifact(
        root=REPO,
        config=config,
        sample_dir=tmp_path / "receipt-block",
        test_receipts=receipts,
    )
    assert blocked["kac_ward_reference_ready_score"] == 0.0
    assert str(blocked["status"]).startswith("blocked_")
    assert blocked["gate_check_summary"]["failed_checks"][0]["category"] == "test"
    assert mod.validate_artifact(blocked) is True

    monkeypatch.setitem(mod.PARITY_TOLERANCES, "probability", -1.0)
    parity_block = mod.build_artifact(
        root=REPO,
        config=config,
        sample_dir=tmp_path / "parity-block",
        test_receipts=_passing_receipts(),
    )
    assert parity_block["reference_sample_manifest"] == []
    assert parity_block["kac_ward_reference_ready_score"] == 0.0
    assert any(
        row["category"] == "parity" for row in parity_block["gate_check_summary"]["failed_checks"]
    )


def test_scenario_sampler_6639_artifact_mutations_fail_closed(
    small_artifact: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-6639-FAIL-CLOSED-NUMERICS rejects evidence drift."""

    cases: list[tuple[dict[str, object], str]] = []
    missing = deepcopy(small_artifact)
    del missing["method_contract"]
    cases.append((missing, "missing required fields"))

    score = deepcopy(small_artifact)
    score["kac_ward_reference_ready_score"] = 0.5
    cases.append((_rehash(score), "binary"))

    attack = deepcopy(small_artifact)
    attack["attack_rows"][0]["passed"] = False
    cases.append((_rehash(attack), "attack"))

    parity = deepcopy(small_artifact)
    parity["per_instance_rows"][0]["parity"]["passed"] = False
    cases.append((_rehash(parity), "parity"))

    protected = deepcopy(small_artifact)
    protected["protected_files_unchanged"]["all_unchanged"] = False
    cases.append((_rehash(protected), "protected"))

    substrate = deepcopy(small_artifact)
    substrate["inference_substrate"] = "hardware"
    cases.append((_rehash(substrate), "inference_substrate"))

    oracle = deepcopy(small_artifact)
    oracle["verifier_is_oracle"] = True
    cases.append((_rehash(oracle), "verifier_is_oracle"))

    verdict = deepcopy(small_artifact)
    verdict["verdict_class"] = "invented"
    cases.append((_rehash(verdict), "verdict_class"))

    bank = deepcopy(small_artifact)
    bank["reference_sample_manifest"].pop()
    cases.append((_rehash(bank), "sample manifest"))

    provenance = deepcopy(small_artifact)
    del provenance["field_provenance"]["status"]
    cases.append((_rehash(provenance), "field_provenance"))

    checksum = deepcopy(small_artifact)
    checksum["duration_s"] = float(checksum["duration_s"]) + 1.0
    cases.append((checksum, "checksum"))

    ready_status = deepcopy(small_artifact)
    ready_status["status"] = "complete_wrong"
    cases.append((_rehash(ready_status), "ready status"))

    ready_class = deepcopy(small_artifact)
    ready_class["verdict_class"] = "blocked"
    cases.append((_rehash(ready_class), "ready verdict_class"))

    ready_tests = deepcopy(small_artifact)
    ready_tests["tests_run"] = []
    cases.append((_rehash(ready_tests), "test receipts"))

    ready_gate = deepcopy(small_artifact)
    ready_gate["gate_check_summary"]["passed"] = False
    cases.append((_rehash(ready_gate), "gate summary"))

    attack_names = deepcopy(small_artifact)
    attack_names["attack_rows"].pop()
    cases.append((_rehash(attack_names), "attack rows"))

    missing_bank = deepcopy(small_artifact)
    missing_bank["reference_sample_manifest"][0]["resolved_path"] = "/missing/bank"
    cases.append((_rehash(missing_bank), "sample manifest hash"))

    unsealed_bank = deepcopy(small_artifact)
    unsealed_path = Path(unsealed_bank["reference_sample_manifest"][0]["resolved_path"])
    unsealed_path.chmod(0o644)
    cases.append((_rehash(unsealed_bank), "not sealed"))

    for payload, message in cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)
    unsealed_path.chmod(0o444)


def test_req_sampler_6639_blocked_artifact_status_and_class_are_consistent(
    small_artifact: dict[str, object],
) -> None:
    """REQ-SAMPLER-6639-BOUNDARY validates both members of the blocked enum."""

    blocked = deepcopy(small_artifact)
    blocked["kac_ward_reference_ready_score"] = 0.0
    blocked["verdict_class"] = "blocked"
    blocked["status"] = "not_blocked"
    with pytest.raises(ValueError, match="blocked status"):
        mod.validate_artifact(_rehash(blocked))
    blocked["status"] = "blocked_test_check_failed"
    blocked["verdict_class"] = "positive"
    with pytest.raises(ValueError, match="blocked verdict_class"):
        mod.validate_artifact(_rehash(blocked))


def test_req_sampler_6639_cli_validation_and_run_exit_codes(
    small_artifact: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-SAMPLER-6639-ATOMIC covers the redirectable command entry point."""

    artifact_path = tmp_path / "artifact.json"
    mod.write_json_atomic(artifact_path, small_artifact)
    assert mod.main(["--validate", str(artifact_path)]) == 0

    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "status": "complete_bounded_exact_reference_ready",
            "kac_ward_reference_ready_score": 1.0,
        }

    output = tmp_path / "run.json"
    samples = tmp_path / "samples"
    monkeypatch.setattr(mod, "run_experiment", fake_run)
    assert (
        mod.main(
            [
                "--date",
                "20260826",
                "--output",
                str(output),
                "--sample-dir",
                str(samples),
            ]
        )
        == 0
    )
    assert calls[0]["output_path"] == output
    monkeypatch.setattr(
        mod,
        "run_experiment",
        lambda **kwargs: {
            "status": "blocked_test_check_failed",
            "kac_ward_reference_ready_score": 0.0,
        },
    )
    assert mod.main(["--output", str(output), "--sample-dir", str(samples)]) == 2


def test_req_sampler_6639_receipt_loader_and_redirectable_run_wrapper(
    small_artifact: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-SAMPLER-6639-ATOMIC keeps test and output paths redirectable."""

    monkeypatch.delenv(mod.TEST_RECEIPT_ENV, raising=False)
    assert mod.load_test_receipts() == []
    receipt_path = tmp_path / "receipts.json"
    receipt_path.write_text(json.dumps(_passing_receipts()), encoding="utf-8")
    monkeypatch.setenv(mod.TEST_RECEIPT_ENV, str(receipt_path))
    assert mod.load_test_receipts() == _passing_receipts()
    receipt_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        mod.load_test_receipts()

    output = tmp_path / "artifact.json"
    samples = tmp_path / "banks"
    monkeypatch.setattr(mod, "build_artifact", lambda **kwargs: deepcopy(small_artifact))
    writes: list[Path] = []
    monkeypatch.setattr(
        mod,
        "write_json_atomic",
        lambda path, payload: writes.append(path) or {"path": str(path)},
    )
    monkeypatch.setattr(mod, "validate_artifact", lambda payload: True)
    receipt_path.write_text(json.dumps(_passing_receipts()), encoding="utf-8")
    result = mod.run_experiment(
        root=REPO,
        output_path=output,
        sample_dir=samples,
        run_date="20260826",
    )
    assert result == small_artifact
    assert writes == [output]


def test_req_sampler_6639_configuration_and_state_validation() -> None:
    """REQ-SAMPLER-6639-ENUMERATION rejects invalid bounded budgets and states."""

    with pytest.raises(ValueError, match="graph_ids"):
        mod.frozen_instances(mod.ExperimentConfig(graph_ids=("missing",)))
    with pytest.raises(ValueError, match="sample_count"):
        mod.frozen_instances(mod.ExperimentConfig(sample_count=0))
    with pytest.raises(ValueError, match="seeds"):
        mod.frozen_instances(mod.ExperimentConfig(seeds=()))
    with pytest.raises(ValueError, match="temperatures"):
        mod.frozen_instances(mod.ExperimentConfig(temperatures=(0.0,)))
    with pytest.raises(ValueError, match="enumeration_limit"):
        mod.frozen_instances(mod.ExperimentConfig(enumeration_limit=0))
    instance = _path_instance(3)
    with pytest.raises(mod.UnsupportedInstanceError, match="spin state"):
        mod.autoregressive_likelihood(instance, (1, 0, -1))
    with pytest.raises(mod.UnsupportedInstanceError, match="spin state"):
        mod.autoregressive_likelihood(instance, (1, -1))
    with pytest.raises(ValueError, match="sample_count"):
        mod.sample_reference_rows(instance, sample_count=0)
    with pytest.raises(ValueError, match="permutation"):
        mod.permute_instance(instance, (0, 0, 2))


def test_req_sampler_6639_low_level_validation_and_trivial_graph_paths() -> None:
    """REQ-SAMPLER-6639-PRECISION covers small and malformed linear systems."""

    path = _path_instance(3)
    with pytest.raises(mod.UnsupportedInstanceError, match="spin order"):
        mod.validate_instance(replace(path, order=(0, 1, 1)))
    with pytest.raises(mod.UnsupportedInstanceError, match="endpoint"):
        mod.validate_instance(replace(path, edges=((0, 3, 0.2),)))
    with pytest.raises(mod.KacWardPrecisionError, match="nonempty and square"):
        mod._validated_logdet(np.asarray([], dtype=np.complex128))
    with pytest.raises(mod.KacWardPrecisionError, match="nonfinite"):
        mod._validated_logdet(np.asarray([[complex(float("nan"), 0.0)]]))
    with pytest.raises(mod.KacWardPrecisionError, match="singular"):
        mod._validated_logdet(np.asarray([[0.0 + 0.0j]]))
    with pytest.raises(mod.KacWardPrecisionError, match="threshold"):
        mod.kac_ward_log_partition(path, condition_limit=float("nan"))

    singleton = _path_instance(1, instance_id="singleton")
    log_partition, diagnostics = mod.kac_ward_log_partition(singleton)
    assert log_partition == pytest.approx(np.log(2.0))
    assert diagnostics["directed_edge_count"] == 0.0
    graph = mod.nx.Graph()
    graph.add_node(9)
    assert mod._planar_positions(graph) == {9: (0.0, 0.0)}
    graph = mod.nx.complete_bipartite_graph(3, 3)
    with pytest.raises(mod.UnsupportedInstanceError, match="nonplanar"):
        mod._planar_positions(graph)


def test_req_sampler_6639_atomic_cleanup_and_missing_package_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-SAMPLER-6639-ATOMIC removes temporary files after replace failure."""

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"blocked replace {source} {destination}")

    monkeypatch.setattr(mod.os, "replace", fail_replace)
    with pytest.raises(OSError, match="blocked replace"):
        mod.write_jsonl_atomic(tmp_path / "failed.jsonl", [{"ok": True}], seal=False)
    with pytest.raises(OSError, match="blocked replace"):
        mod.write_json_atomic(tmp_path / "failed.json", {"ok": True})
    assert not list(tmp_path.glob("*.tmp"))

    real_version = mod.metadata.version

    def missing_version(package: str) -> str:
        if package == "coverage":
            raise mod.metadata.PackageNotFoundError(package)
        return real_version(package)

    monkeypatch.setattr(mod.metadata, "version", missing_version)
    assert mod._package_versions()["coverage"] == "missing"


def test_req_sampler_6639_build_failure_reducers_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-SAMPLER-6639-BOUNDARY names math, sample, attack, and protection blocks."""

    config = mod.ExperimentConfig(
        graph_ids=("mixed_square_n4",),
        temperatures=(0.8,),
        seeds=(6639001,),
        sample_count=2,
    )
    monkeypatch.setattr(
        mod,
        "cross_check_instance",
        lambda instance: (_ for _ in ()).throw(mod.KacWardPrecisionError("unsafe")),
    )
    math_block = mod.build_artifact(
        root=REPO,
        config=config,
        sample_dir=tmp_path / "math",
        test_receipts=_passing_receipts(),
    )
    assert math_block["gate_check_summary"]["failed_checks"][0]["category"] == "math"

    monkeypatch.undo()
    original_sample = mod.sample_reference_rows

    def mismatched_sample(
        instance: mod.PlanarIsingInstance, *, sample_count: int
    ) -> dict[str, object]:
        result = original_sample(instance, sample_count=sample_count)
        result["rows"][0]["likelihood_parity_passed"] = False
        return result

    monkeypatch.setattr(mod, "sample_reference_rows", mismatched_sample)
    sample_block = mod.build_artifact(
        root=REPO,
        config=config,
        sample_dir=tmp_path / "sample",
        test_receipts=_passing_receipts(),
    )
    assert sample_block["reference_sample_manifest"] == []

    monkeypatch.undo()
    real_attacks = mod.build_attack_rows()
    real_attacks[0]["passed"] = False
    monkeypatch.setattr(mod, "build_attack_rows", lambda: real_attacks)
    attack_block = mod.build_artifact(
        root=REPO,
        config=config,
        sample_dir=tmp_path / "attack",
        test_receipts=_passing_receipts(),
    )
    assert any(
        row["category"] == "attack" for row in attack_block["gate_check_summary"]["failed_checks"]
    )

    monkeypatch.undo()
    real_compare = mod.protected_files_unchanged

    def failed_protection(before: dict[str, str], after: dict[str, str]) -> dict[str, object]:
        receipt = real_compare(before, after)
        receipt["all_unchanged"] = False
        receipt["rows"][0]["unchanged"] = False
        return receipt

    monkeypatch.setattr(mod, "protected_files_unchanged", failed_protection)
    protection_block = mod.build_artifact(
        root=REPO,
        config=config,
        sample_dir=tmp_path / "protection",
        test_receipts=_passing_receipts(),
    )
    assert any(
        row["category"] == "protection"
        for row in protection_block["gate_check_summary"]["failed_checks"]
    )


def test_req_sampler_6639_protected_files_stay_unchanged() -> None:
    """REQ-SAMPLER-6639-ATOMIC protects conductor and reconciler inputs."""

    before = mod.protected_hashes(REPO)
    after = mod.protected_hashes(REPO)
    receipt = mod.protected_files_unchanged(before, after)
    assert receipt["all_unchanged"] is True
    changed = dict(after)
    changed[next(iter(changed))] = "sha256:changed"
    assert mod.protected_files_unchanged(before, changed)["all_unchanged"] is False


def test_req_sampler_6639_auxiliary_and_enumeration_attacks_have_real_witnesses() -> None:
    """REQ-SAMPLER-6639-ATTACKS proves omission and mismatch are detectable."""

    rows = {row["attack"]: row for row in mod.build_attack_rows()}
    assert rows["auxiliary_spin_omission"]["observed_value"] > mod.PARITY_TOLERANCES["conditional"]
    assert rows["enumeration_mismatch"]["passed"] is True
    assert rows["coupling_sign"]["passed"] is True
    assert rows["rng_reuse"]["passed"] is True


def test_req_sampler_6639_atomic_bank_replay_overwrites_read_only_target(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAMPLER-6639-SEALED-INDEPENDENT-EVIDENCE seals replay banks."""

    target = tmp_path / "bank.jsonl"
    rows = [{"sample_index": 0, "spins": [1, -1], "normalized_likelihood": 0.25}]
    first = mod.write_jsonl_atomic(target, rows, seal=True)
    second = mod.write_jsonl_atomic(target, rows, seal=True)
    assert first["sha256"] == second["sha256"]
    assert target.stat().st_mode & 0o222 == 0
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert not [path for path in os.listdir(tmp_path) if path.endswith(".tmp")]

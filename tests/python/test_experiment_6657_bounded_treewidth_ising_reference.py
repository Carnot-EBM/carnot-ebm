"""Tests for the bounded-treewidth exact Ising reference.

Spec refs: REQ-SAMPLER-6657, REQ-REPORT-6657,
SCENARIO-SAMPLER-6657-EXACT-PARITY,
SCENARIO-SAMPLER-6657-ANCESTRAL-SAMPLING,
SCENARIO-SAMPLER-6657-FAIL-CLOSED,
SCENARIO-REPORT-6657-ATOMIC-CHECKSUM.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_6657_bounded_treewidth_ising_reference as mod


REPO = Path(__file__).resolve().parents[2]
SAMPLER_SPEC = REPO / "openspec/capabilities/samplers/spec.md"
REPORT_SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"


def _passing_receipts() -> list[dict[str, object]]:
    return [
        {"scope": scope, "command": f"verified {scope}", "exit_code": 0, "summary": "passed"}
        for scope in mod.REQUIRED_TEST_SCOPES
    ]


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
    return payload


def _path_instance(n_spins: int = 4) -> mod.IsingInstance:
    return mod.IsingInstance(
        instance_id=f"path_{n_spins}",
        n_spins=n_spins,
        edges=tuple((index, index + 1, 0.3) for index in range(n_spins - 1)),
        fields=tuple(0.1 * ((-1) ** index) for index in range(n_spins)),
        temperature=1.2,
        seed=6657001,
    )


def test_req_sampler_6657_specs_precede_implementation() -> None:
    """REQ-SAMPLER-6657 and REQ-REPORT-6657 freeze the full contract."""

    sampler = SAMPLER_SPEC.read_text(encoding="utf-8").split("### REQ-SAMPLER-6657", 1)[1]
    report = REPORT_SPEC.read_text(encoding="utf-8").split("### REQ-REPORT-6657", 1)[1]
    for anchor in (
        "REQ-SAMPLER-6657-DECOMPOSITION",
        "REQ-SAMPLER-6657-ELIMINATION",
        "REQ-SAMPLER-6657-NORMALIZATION",
        "REQ-SAMPLER-6657-MARGINALS",
        "REQ-SAMPLER-6657-SAMPLING",
        "REQ-SAMPLER-6657-REJECTION",
        "REQ-SAMPLER-6657-PARITY",
        "SCENARIO-SAMPLER-6657-FAIL-CLOSED",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert anchor in sampler
    assert mod.RESULT_RELATIVE_PATH.as_posix() in report
    assert "verdict_class=null" in report


def test_req_sampler_6657_fixture_manifest_is_frozen_and_broad() -> None:
    """REQ-SAMPLER-6657-PARITY freezes twelve supported and explicit rejected cases."""

    fixtures = mod.frozen_fixtures()
    supported = [item for item in fixtures if item.expected_supported]
    rejected = [item for item in fixtures if not item.expected_supported]
    assert len(supported) >= 12
    assert len(rejected) >= 3
    assert {item.family for item in supported} >= {
        "tree",
        "cycle",
        "ferromagnetic",
        "antiferromagnetic",
        "frustrated",
        "field",
    }
    assert {item.expected_rejection for item in rejected} >= {
        "treewidth",
        "self-loop",
        "duplicate edge",
    }
    assert len({item.fixture_sha256 for item in fixtures}) == len(fixtures)
    assert mod.frozen_fixtures()[0].fixture_sha256 == fixtures[0].fixture_sha256


def test_req_sampler_6657_deterministic_decomposition_validates_width_and_running_intersection() -> (
    None
):
    """REQ-SAMPLER-6657-DECOMPOSITION checks the certified tree decomposition."""

    path = _path_instance()
    first = mod.deterministic_tree_decomposition(path)
    second = mod.deterministic_tree_decomposition(path)
    assert first == second
    receipt = mod.validate_tree_decomposition(path, first)
    assert receipt["valid"] is True
    assert receipt["width"] == 1
    assert receipt["running_intersection"] is True

    k5 = next(item for item in mod.frozen_fixtures() if item.instance_id == "complete_k5_tw4")
    decomposition = mod.deterministic_tree_decomposition(k5)
    assert decomposition.width == mod.MAX_TREEWIDTH
    assert mod.validate_tree_decomposition(k5, decomposition)["edge_coverage"] is True


@pytest.mark.parametrize(
    ("decomposition", "message"),
    [
        (mod.TreeDecomposition(((0, 1),), (), (0, 1, 2, 3), 1), "vertex coverage"),
        (
            mod.TreeDecomposition(((0, 1), (1, 2), (2, 3)), ((0, 1),), (0, 1, 2, 3), 1),
            "tree",
        ),
        (
            mod.TreeDecomposition(((0, 1), (1, 2), (0, 3)), ((0, 1), (1, 2)), (0, 1, 2, 3), 1),
            "edge coverage",
        ),
        (
            mod.TreeDecomposition(
                ((0, 1), (1, 2, 3), (0, 1, 3)),
                ((0, 1), (1, 2)),
                (0, 1, 2, 3),
                2,
            ),
            "running intersection",
        ),
        (
            mod.TreeDecomposition(((0, 1), (1, 2), (2, 3)), ((0, 1), (1, 2)), (0, 1, 2, 3), 9),
            "declared width",
        ),
    ],
)
def test_scenario_sampler_6657_malformed_decompositions_fail_closed(
    decomposition: mod.TreeDecomposition, message: str
) -> None:
    """SCENARIO-SAMPLER-6657-FAIL-CLOSED rejects invalid bag certificates."""

    with pytest.raises(mod.UnsupportedGraphError, match=message):
        mod.validate_tree_decomposition(_path_instance(), decomposition)


@pytest.mark.parametrize(
    ("instance", "message"),
    [
        (replace(_path_instance(), n_spins=0, fields=()), "spin count"),
        (replace(_path_instance(), fields=(0.0,)), "field count"),
        (replace(_path_instance(), temperature=0.0), "temperature"),
        (replace(_path_instance(), edges=((0, 0, 1.0),)), "self-loop"),
        (
            replace(_path_instance(), edges=((0, 1, 1.0), (1, 0, -1.0))),
            "duplicate edge",
        ),
        (replace(_path_instance(), edges=((0, 8, 1.0),)), "endpoint"),
        (replace(_path_instance(), edges=(("0", 1, 1.0),)), "integer"),
        (replace(_path_instance(), fields=(0.0, 0.0, float("nan"), 0.0)), "finite"),
        (
            replace(_path_instance(), edges=((0, 1, float("inf")),)),
            "finite",
        ),
        (_path_instance(mod.MAX_SPINS + 1), "supported limit"),
    ],
)
def test_scenario_sampler_6657_malformed_models_fail_closed(
    instance: mod.IsingInstance, message: str
) -> None:
    """REQ-SAMPLER-6657-REJECTION rejects malformed model inputs."""

    with pytest.raises(mod.UnsupportedGraphError, match=message):
        mod.validate_instance(instance)


def test_scenario_sampler_6657_width_above_four_is_rejected() -> None:
    """REQ-SAMPLER-6657-REJECTION rejects a valid simple K6 above width four."""

    k6 = next(item for item in mod.frozen_fixtures() if item.instance_id == "unsupported_k6_tw5")
    mod.validate_instance(k6)
    decomposition = mod.deterministic_tree_decomposition(k6)
    assert decomposition.width == 5
    with pytest.raises(mod.UnsupportedGraphError, match="treewidth"):
        mod.solve_exact(k6)


def test_req_sampler_6657_low_level_certificate_and_evidence_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLER-6657-DECOMPOSITION names low-level certificate failures."""

    instance = _path_instance()
    valid = mod.deterministic_tree_decomposition(instance)
    cases = (
        (mod.TreeDecomposition((), (), valid.elimination_order, 0), "vertex coverage"),
        (
            replace(valid, bags=((0, 0, 1), *valid.bags[1:])),
            "bag vertices",
        ),
        (
            replace(valid, tree_edges=((0, 0), *valid.tree_edges[1:])),
            "invalid edge",
        ),
        (replace(valid, elimination_order=(0, 0, 2, 3)), "elimination order"),
    )
    for decomposition, message in cases:
        with pytest.raises(mod.UnsupportedGraphError, match=message):
            mod.validate_tree_decomposition(instance, decomposition)
    with pytest.raises(mod.UnsupportedGraphError, match="evidence"):
        mod._initial_factors(instance, {8: 1})

    singleton = _path_instance(1)
    decomposition = mod.deterministic_tree_decomposition(singleton)
    monkeypatch.setattr(
        mod,
        "_initial_factors",
        lambda *_: [mod._Factor((0,), np.asarray([float("nan"), float("nan")]))],
    )
    with pytest.raises(mod.UnsupportedGraphError, match="nonfinite probability"):
        mod._eliminate(singleton, decomposition, {}, False)


def test_scenario_sampler_6657_factor_elimination_matches_brute_force() -> None:
    """SCENARIO-SAMPLER-6657-EXACT-PARITY checks all supported fixtures."""

    for instance in (item for item in mod.frozen_fixtures() if item.expected_supported):
        row = mod.cross_check_fixture(instance)
        assert row["passed"] is True
        assert row["partition_error"] <= mod.EXACT_TOLERANCES["partition"]
        assert row["log_probability_error_max"] <= mod.EXACT_TOLERANCES["log_probability"]
        assert row["node_marginal_error_max"] <= mod.EXACT_TOLERANCES["marginal"]
        assert row["edge_marginal_error_max"] <= mod.EXACT_TOLERANCES["marginal"]


def test_req_sampler_6657_probabilities_normalize_and_validate_states() -> None:
    """REQ-SAMPLER-6657-NORMALIZATION derives probabilities from the DP partition."""

    instance = _path_instance(3)
    solution = mod.solve_exact(instance)
    brute = mod.brute_force_reference(instance)
    mass = 0.0
    for state, expected in zip(brute["states"], brute["probabilities"], strict=True):
        log_probability = mod.configuration_log_probability(instance, state, solution)
        probability = mod.configuration_probability(instance, state, solution)
        assert np.exp(log_probability) == pytest.approx(probability, abs=1.0e-15)
        assert probability == pytest.approx(expected, abs=mod.EXACT_TOLERANCES["probability"])
        mass += probability
    assert mass == pytest.approx(1.0, abs=mod.EXACT_TOLERANCES["normalization"])
    with pytest.raises(mod.UnsupportedGraphError, match="spin state"):
        mod.configuration_probability(instance, (1, 0, -1), solution)
    with pytest.raises(mod.UnsupportedGraphError, match="spin state"):
        mod.configuration_probability(instance, (1, -1), solution)


def test_req_sampler_6657_exact_node_and_edge_marginals_match_enumeration() -> None:
    """REQ-SAMPLER-6657-MARGINALS returns complete normalized marginal tables."""

    instance = next(
        item for item in mod.frozen_fixtures() if item.instance_id == "frustrated_triangle"
    )
    solution = mod.solve_exact(instance)
    marginals = mod.exact_marginals(instance, solution)
    brute = mod.brute_force_reference(instance)
    assert marginals["node_plus"] == pytest.approx(brute["node_plus"], abs=1.0e-12)
    for edge_id, table in marginals["edge_joint"].items():
        assert sum(table.values()) == pytest.approx(1.0, abs=1.0e-12)
        assert table == pytest.approx(brute["edge_joint"][edge_id], abs=1.0e-12)


def test_scenario_sampler_6657_ancestral_samples_replay_and_match_frequencies() -> None:
    """SCENARIO-SAMPLER-6657-ANCESTRAL-SAMPLING checks exact independent draws."""

    instance = next(item for item in mod.frozen_fixtures() if item.instance_id == "cycle4_field")
    solution = mod.solve_exact(instance)
    first = mod.independent_samples(instance, 4096, 6657123, solution)
    replay = mod.independent_samples(instance, 4096, 6657123, solution)
    other = mod.independent_samples(instance, 4096, 6657124, solution)
    assert first["sample_sha256"] == replay["sample_sha256"]
    assert first["samples"] == replay["samples"]
    assert first["sample_sha256"] != other["sample_sha256"]
    row = mod.sample_check_fixture(instance, sample_count=4096, seed=6657123)
    assert row["passed"] is True
    assert row["likelihood_error_max"] <= mod.EXACT_TOLERANCES["log_probability"]
    assert row["state_frequency_error_max"] <= mod.SAMPLE_TOLERANCES["state"]
    assert row["node_frequency_error_max"] <= mod.SAMPLE_TOLERANCES["node"]
    assert row["edge_frequency_error_max"] <= mod.SAMPLE_TOLERANCES["edge"]
    with pytest.raises(ValueError, match="sample_count"):
        mod.independent_samples(instance, 0, 1, solution)


@pytest.fixture(scope="module")
def complete_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build one complete artifact without writing tracked state."""

    temporary = tmp_path_factory.mktemp("exp6657")
    return mod.build_artifact(
        root=REPO,
        run_date="20260827",
        config=mod.ExperimentConfig(sample_count=mod.DEFAULT_SAMPLE_COUNT),
        test_receipts=_passing_receipts(),
        timing_clock=mod.DeterministicEvidenceClock(),
        output_parent=temporary,
    )


def test_req_report_6657_artifact_contains_all_rows_and_recomputes(
    complete_artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6657-READY checks complete row-backed evidence."""

    assert mod.validate_artifact(complete_artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(complete_artifact)
    assert complete_artifact["status"] == "complete_bounded_treewidth_exact_reference_ready"
    assert str(complete_artifact["honest_verdict"]).startswith("complete:")
    assert complete_artifact["verdict_class"] is None
    assert complete_artifact["ising_reference_ready"] is True
    assert complete_artifact["verifier_is_oracle"] is True
    assert complete_artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert len(complete_artifact["fixture_manifest"]) >= 15
    assert len(complete_artifact["exact_parity_rows"]) >= 12
    assert len(complete_artifact["exact_sample_rows"]) >= 12
    assert len(complete_artifact["decomposition_rows"]) >= 15
    assert len(complete_artifact["per_unit_rows"]) >= 51
    assert complete_artifact["aggregate_row_recomputation"]["ready"] is True
    assert complete_artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert set(complete_artifact["field_provenance"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert complete_artifact["prior_failure_receipt"]["terminal_record_count"] == 3
    assert all(
        "no performance claim" in row["claim_boundary"] for row in complete_artifact["timing_rows"]
    )


def test_req_report_6657_atomic_write_and_checksum(
    complete_artifact: dict[str, object], tmp_path: Path
) -> None:
    """SCENARIO-REPORT-6657-ATOMIC-CHECKSUM protects the final JSON."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    receipt = mod.write_json_atomic(output, complete_artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == complete_artifact
    assert receipt["atomic_replace"] is True
    assert receipt["file_fsync"] is True
    assert receipt["directory_fsync"] is True
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="nonfinite JSON"):
        mod.write_json_atomic(tmp_path / "bad.json", {"bad": float("nan")})


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda item: item.pop("supported_domain_contract"), "missing required fields"),
        (lambda item: item.__setitem__("verdict_class", "positive"), "verdict_class"),
        (lambda item: item.__setitem__("verifier_is_oracle", False), "verifier_is_oracle"),
        (lambda item: item.__setitem__("inference_substrate", "hardware"), "inference_substrate"),
        (
            lambda item: item["aggregate_row_recomputation"].__setitem__("ready", False),
            "aggregate",
        ),
        (
            lambda item: item["protected_files_unchanged"].__setitem__("all_unchanged", False),
            "protected",
        ),
        (lambda item: item["exact_sample_rows"][0].__setitem__("passed", False), "sample"),
        (lambda item: item["field_provenance"].pop("status"), "field_provenance"),
        (lambda item: item["tests_run"].clear(), "test receipts"),
        (
            lambda item: item["prior_failure_receipt"].__setitem__("terminal_record_count", 2),
            "prior failure",
        ),
    ],
)
def test_scenario_report_6657_artifact_mutations_fail_closed(
    complete_artifact: dict[str, object], mutator: object, message: str
) -> None:
    """SCENARIO-REPORT-6657-BLOCKED rejects row and schema drift."""

    changed = deepcopy(complete_artifact)
    mutator(changed)  # type: ignore[operator]
    if "reproducibility_checksum" in changed:
        _rehash(changed)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(changed)


def test_req_report_6657_raw_checksum_mutation_fails(
    complete_artifact: dict[str, object],
) -> None:
    """REQ-REPORT-6657 detects content edits that do not update the checksum."""

    changed = deepcopy(complete_artifact)
    changed["duration_s"] = float(changed["duration_s"]) + 1.0
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(changed)


def test_req_sampler_6657_failed_receipt_blocks_readiness(tmp_path: Path) -> None:
    """REQ-SAMPLER-6657-READINESS makes tests part of the ready conjunction."""

    receipts = _passing_receipts()
    receipts[0]["exit_code"] = 1
    artifact = mod.build_artifact(
        root=REPO,
        config=mod.ExperimentConfig(sample_count=mod.DEFAULT_SAMPLE_COUNT),
        test_receipts=receipts,
        timing_clock=mod.DeterministicEvidenceClock(),
        output_parent=tmp_path,
    )
    assert artifact["ising_reference_ready"] is False
    assert str(artifact["status"]).startswith("blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_checks"][0]["check"] == "tests"
    assert mod.validate_artifact(artifact) is True


def test_req_report_6657_precondition_and_unexpected_acceptance_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-6657 preserves frozen sample and expected-rejection gates."""

    with pytest.raises(ValueError, match="frozen count"):
        mod.build_artifact(
            root=REPO,
            config=mod.ExperimentConfig(sample_count=1),
            test_receipts=_passing_receipts(),
        )
    monkeypatch.setattr(
        mod.metadata,
        "version",
        lambda package: (_ for _ in ()).throw(mod.metadata.PackageNotFoundError(package)),
    )
    assert set(mod._package_versions().values()) == {mod.platform.python_version(), "missing"}

    unexpectedly_valid = replace(
        _path_instance(2),
        instance_id="unexpectedly_valid",
        expected_supported=False,
        expected_rejection="unexpectedly accepted",
    )
    monkeypatch.setattr(mod, "frozen_fixtures", lambda: (unexpectedly_valid,))
    artifact = mod.build_artifact(
        root=REPO,
        config=mod.ExperimentConfig(),
        test_receipts=_passing_receipts(),
        timing_clock=mod.DeterministicEvidenceClock(),
        output_parent=tmp_path,
    )
    assert artifact["decomposition_rows"][0]["observed_error"] == (
        "unsupported fixture was unexpectedly accepted"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.__setitem__("status", "complete_wrong"), "ready status"),
        (lambda item: item.__setitem__("verdict_class", "blocked"), "ready verdict_class"),
        (lambda item: item["exact_parity_rows"][0].__setitem__("passed", False), "parity"),
        (
            lambda item: item["normalized_mass_receipts"][0].__setitem__("passed", False),
            "normalization",
        ),
        (
            lambda item: item["decomposition_rows"][0].__setitem__("passed", False),
            "decomposition",
        ),
        (lambda item: item["gate_check_summary"].__setitem__("passed", False), "gate summary"),
        (
            lambda item: (
                item.__setitem__("ising_reference_ready", False),
                item.__setitem__("verdict_class", "blocked"),
                item["gate_check_summary"].__setitem__("passed", False),
                item["gate_check_summary"].__setitem__(
                    "failed_checks", [{"check": "test", "observed_value": False}]
                ),
            ),
            "blocked status",
        ),
        (
            lambda item: (
                item.__setitem__("ising_reference_ready", False),
                item.__setitem__("status", "blocked_test_check_failed"),
                item["gate_check_summary"].__setitem__("passed", False),
                item["gate_check_summary"].__setitem__(
                    "failed_checks", [{"check": "test", "observed_value": False}]
                ),
            ),
            "blocked verdict_class",
        ),
        (
            lambda item: (
                item.__setitem__("ising_reference_ready", False),
                item.__setitem__("status", "blocked_test_check_failed"),
                item.__setitem__("verdict_class", "blocked"),
                item["gate_check_summary"].__setitem__("failed_checks", []),
            ),
            "blocked gate summary",
        ),
        (lambda item: item["fixture_manifest"].clear(), "fixture manifest"),
        (lambda item: item["per_unit_rows"].pop(), "per-unit"),
        (
            lambda item: item["timing_rows"][0].__setitem__("claim_boundary", "speed claim"),
            "timing rows",
        ),
    ],
)
def test_req_report_6657_validator_rejects_each_terminal_inconsistency(
    complete_artifact: dict[str, object], mutation: object, message: str
) -> None:
    """REQ-REPORT-6657 validates each terminal state and row family."""

    changed = deepcopy(complete_artifact)
    mutation(changed)  # type: ignore[operator]
    _rehash(changed)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(changed)


def test_req_report_6657_atomic_cleanup_on_replace_failure(
    complete_artifact: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-6657-ATOMIC-CHECKSUM removes an uncommitted temporary file."""

    monkeypatch.setattr(mod.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("replace")))
    with pytest.raises(OSError, match="replace"):
        mod.write_json_atomic(tmp_path / "failed.json", complete_artifact)
    assert not list(tmp_path.iterdir())


def test_req_report_6657_receipt_loader_and_redirectable_cli(
    complete_artifact: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-6657 keeps test receipts and writes redirectable for unit tests."""

    monkeypatch.delenv(mod.TEST_RECEIPT_ENV, raising=False)
    assert mod.load_test_receipts() == []
    receipt_path = tmp_path / "receipts.json"
    receipt_path.write_text(json.dumps(_passing_receipts()), encoding="utf-8")
    monkeypatch.setenv(mod.TEST_RECEIPT_ENV, str(receipt_path))
    assert mod.load_test_receipts() == _passing_receipts()
    receipt_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        mod.load_test_receipts()

    artifact_path = tmp_path / "artifact.json"
    mod.write_json_atomic(artifact_path, complete_artifact)
    assert mod.main(["--validate", str(artifact_path)]) == 0

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        mod, "run_experiment", lambda **kwargs: calls.append(kwargs) or complete_artifact
    )
    output = tmp_path / "run.json"
    assert mod.main(["--date", "20260827", "--output", str(output)]) == 0
    assert calls[0]["output_path"] == output
    blocked = deepcopy(complete_artifact)
    blocked["ising_reference_ready"] = False
    monkeypatch.setattr(mod, "run_experiment", lambda **kwargs: blocked)
    assert mod.main(["--output", str(output)]) == 2


def test_req_report_6657_run_wrapper_validates_and_writes(
    complete_artifact: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-6657 runs the builder and atomic writer once."""

    monkeypatch.setattr(mod, "build_artifact", lambda **kwargs: deepcopy(complete_artifact))
    writes: list[Path] = []
    monkeypatch.setattr(
        mod, "write_json_atomic", lambda path, payload: writes.append(path) or {"path": str(path)}
    )
    output = tmp_path / "result.json"
    result = mod.run_experiment(root=REPO, output_path=output, run_date="20260827")
    assert result == complete_artifact
    assert writes == [output]

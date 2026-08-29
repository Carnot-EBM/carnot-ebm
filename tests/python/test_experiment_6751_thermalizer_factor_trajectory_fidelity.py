"""Tests for the bounded typed-factor compiler fidelity reference.

Spec: REQ-HW-6751, REQ-HW-6751-TYPES, REQ-HW-6751-EXACT,
REQ-HW-6751-MATCHED, REQ-HW-6751-METRICS,
REQ-HW-6751-SERIALIZATION, REQ-HW-6751-PROVENANCE,
REQ-HW-6751-COMPLETION, REQ-HW-6751-BOUNDARY,
SCENARIO-HW-6751-EXACT-COMPILATION,
SCENARIO-HW-6751-REFINEMENT, SCENARIO-HW-6751-FAIL-CLOSED.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_6751_thermalizer_factor_trajectory_fidelity as mod


def test_req_hw_6751_spec_precedes_implementation() -> None:
    """REQ-HW-6751: The hardware capability owns the compiler contract."""

    text = (mod.REPO_ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    anchors = set(mod.spec_anchors(text))
    assert {
        "REQ-HW-6751-TYPES",
        "REQ-HW-6751-EXACT",
        "REQ-HW-6751-MATCHED",
        "REQ-HW-6751-METRICS",
        "REQ-HW-6751-SERIALIZATION",
        "REQ-HW-6751-PROVENANCE",
        "REQ-HW-6751-COMPLETION",
        "REQ-HW-6751-BOUNDARY",
        "SCENARIO-HW-6751-EXACT-COMPILATION",
        "SCENARIO-HW-6751-REFINEMENT",
        "SCENARIO-HW-6751-FAIL-CLOSED",
    } <= anchors


def test_req_hw_6751_typed_kernels_validate_and_reject_invalid_inputs() -> None:
    """REQ-HW-6751-TYPES: Typed kernels expose finite exact semantics."""

    kernels = mod.frozen_kernels()
    assert {kernel.kind for kernel in kernels} == {"binary", "categorical"}
    assert {kernel.n_categories for kernel in kernels} == {2, 3}
    assert all(kernel.validate() is None for kernel in kernels)
    assert all(np.allclose(np.asarray(kernel.target).sum(axis=1), 1.0) for kernel in kernels)

    binary = kernels[0]
    with pytest.raises(mod.CompilerInputError, match="target shape"):
        mod.TypedKernel(
            factor_id="bad_shape",
            kind="binary",
            categories=("zero", "one"),
            target=((1.0, 0.0),),
            feature_names=binary.feature_names,
            features=binary.features,
            parameter_bound=binary.parameter_bound,
            couplers=binary.couplers,
        ).validate()
    with pytest.raises(mod.CompilerInputError, match="row normalization"):
        mod.TypedKernel(
            factor_id="bad_mass",
            kind="binary",
            categories=("zero", "one"),
            target=((0.5, 0.6), (0.4, 0.6)),
            feature_names=binary.feature_names,
            features=binary.features,
            parameter_bound=binary.parameter_bound,
            couplers=binary.couplers,
        ).validate()
    with pytest.raises(mod.CompilerInputError, match="feature shape"):
        mod.TypedKernel(
            factor_id="bad_features",
            kind="binary",
            categories=("zero", "one"),
            target=binary.target,
            feature_names=binary.feature_names,
            features=(((0.0,),),),
            parameter_bound=binary.parameter_bound,
            couplers=binary.couplers,
        ).validate()
    with pytest.raises(mod.CompilerInputError, match="coupler parameter"):
        mod.TypedKernel(
            factor_id="bad_coupler",
            kind="binary",
            categories=("zero", "one"),
            target=binary.target,
            feature_names=binary.feature_names,
            features=binary.features,
            parameter_bound=binary.parameter_bound,
            couplers=(("input:zero", "output:one", "missing"),),
        ).validate()

    for changed, message in (
        (replace(binary, kind="continuous"), "kernel kind"),
        (replace(binary, categories=("same", "same")), "categories"),
        (replace(binary, target=((float("nan"), 0.0), (0.4, 0.6))), "finite"),
        (replace(binary, parameter_bound=0.0), "parameter bound"),
    ):
        with pytest.raises(mod.CompilerInputError, match=message):
            changed.validate()
    with pytest.raises(mod.CompilerInputError, match="parameter vector"):
        binary.compiled_conditional(
            np.asarray([float("nan"), 0.0]), mod.PRECISION_SPECS["binary32"]
        )
    with pytest.raises(mod.CompilerInputError, match="canonical JSON"):
        mod.canonical_json({"invalid": float("nan")})
    with pytest.raises(mod.CompilerInputError, match="no frozen contexts"):
        mod.frozen_contexts(replace(binary, factor_id="unknown"))


def test_req_hw_6751_exact_conditionals_and_trajectories_normalize() -> None:
    """REQ-HW-6751-EXACT: Every selected state space is enumerated exactly."""

    for kernel in mod.frozen_kernels():
        context = mod.frozen_contexts(kernel)[0]
        theta = np.zeros(kernel.n_parameters, dtype=np.float64)
        conditional = kernel.compiled_conditional(theta, mod.PRECISION_SPECS["binary32"])
        assert conditional.shape == (kernel.n_categories, kernel.n_categories)
        assert np.max(np.abs(conditional.sum(axis=1) - 1.0)) <= mod.NORMALIZATION_TOLERANCE
        per_input, weighted = mod.conditional_kl(
            np.asarray(kernel.target), conditional, np.asarray(context.initial)
        )
        assert len(per_input) == kernel.n_categories
        assert weighted >= 0.0

        for depth in mod.DEPTHS:
            target = mod.enumerate_trajectory_distribution(
                np.asarray(context.initial), np.asarray(kernel.target), depth
            )
            compiled = mod.enumerate_trajectory_distribution(
                np.asarray(context.initial), conditional, depth
            )
            assert target.paths.shape == (kernel.n_categories ** (depth + 1), depth + 1)
            assert abs(float(target.probabilities.sum()) - 1.0) <= mod.NORMALIZATION_TOLERANCE
            assert abs(float(compiled.probabilities.sum()) - 1.0) <= mod.NORMALIZATION_TOLERANCE
            assert 0.0 <= mod.total_variation(target.probabilities, compiled.probabilities) <= 1.0

    with pytest.raises(mod.CompilerInputError, match="positive depth"):
        mod.enumerate_trajectory_distribution(np.asarray([0.5, 0.5]), np.eye(2), 0)
    with pytest.raises(mod.CompilerInputError, match="initial distribution"):
        mod.enumerate_trajectory_distribution(np.asarray([1.0]), np.eye(2), 1)


def test_req_hw_6751_exact_helpers_reject_ambiguous_probability_inputs() -> None:
    """REQ-HW-6751-EXACT: Exact metrics fail before using malformed laws."""

    target = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    compiled = np.asarray([[0.4, 0.6], [0.4, 0.6]])
    with pytest.raises(mod.CompilerInputError, match="same matrix shape"):
        mod.conditional_kl(target, compiled[:1], np.asarray([0.5, 0.5]))
    with pytest.raises(mod.CompilerInputError, match="input categories"):
        mod.conditional_kl(target, compiled, np.asarray([-0.1, 1.1]))
    with pytest.raises(mod.CompilerInputError, match="weights must normalize"):
        mod.conditional_kl(target, compiled, np.asarray([0.2, 0.2]))
    with pytest.raises(mod.CompilerInputError, match="full support"):
        mod.conditional_kl(target, np.eye(2), np.asarray([0.5, 0.5]))

    with pytest.raises(mod.CompilerInputError, match="finite and nonnegative"):
        mod.enumerate_trajectory_distribution(np.asarray([float("nan"), 0.0]), np.eye(2), 1)
    with pytest.raises(mod.CompilerInputError, match="must normalize"):
        mod.enumerate_trajectory_distribution(np.asarray([0.2, 0.2]), np.eye(2), 1)
    with pytest.raises(mod.CompilerInputError, match="square matrix"):
        mod.enumerate_trajectory_distribution(np.asarray([0.5, 0.5]), np.ones((2, 3)), 1)
    with pytest.raises(mod.CompilerInputError, match="finite and nonnegative"):
        mod.enumerate_trajectory_distribution(
            np.asarray([0.5, 0.5]), np.asarray([[1.0, 0.0], [-0.1, 1.1]]), 1
        )
    with pytest.raises(mod.CompilerInputError, match="rows must normalize"):
        mod.enumerate_trajectory_distribution(
            np.asarray([0.5, 0.5]), np.asarray([[0.2, 0.2], [0.5, 0.5]]), 1
        )
    with pytest.raises(mod.CompilerReferenceUnavailable, match="exceeds"):
        mod.enumerate_trajectory_distribution(np.full(4, 0.25), np.eye(4), 8)
    with pytest.raises(mod.CompilerInputError, match="equal shape"):
        mod.total_variation(np.asarray([1.0]), np.asarray([0.5, 0.5]))


def test_req_hw_6751_topology_precision_and_candidate_serialization() -> None:
    """REQ-HW-6751-SERIALIZATION: Receipts round-trip without semantic drift."""

    topology = mod.topology_receipts()
    precision = mod.precision_receipts()
    assert {row["factor_id"] for row in topology} == {
        kernel.factor_id for kernel in mod.frozen_kernels()
    }
    assert {row["precision_id"] for row in precision} == set(mod.PRECISIONS)
    assert all(row["couplers"] for row in topology)
    assert all(row["categories"] for row in topology)
    assert all(row["receipt_sha256"] == mod.receipt_hash(row) for row in topology + precision)
    assert json.loads(mod.canonical_json(topology)) == topology

    kernel = mod.frozen_kernels()[0]
    seed = mod.SEED_BUNDLES[0]
    bank_a = mod.candidate_bank(
        kernel, mod.PRECISION_SPECS["fixed_q3_4"], seed, mod.TRAINING_BUDGET
    )
    bank_b = mod.candidate_bank(
        kernel, mod.PRECISION_SPECS["fixed_q3_4"], seed, mod.TRAINING_BUDGET
    )
    assert np.array_equal(bank_a, bank_b)
    assert bank_a.shape == (mod.TRAINING_BUDGET, kernel.n_parameters)
    assert np.allclose(bank_a * 16.0, np.rint(bank_a * 16.0))
    with pytest.raises(mod.CompilerInputError, match="candidate budget"):
        mod.candidate_bank(kernel, mod.PRECISION_SPECS["binary32"], seed, 2)
    with pytest.raises(mod.CompilerInputError, match="unknown compiler arm"):
        mod._fit_arm(
            kernel, mod.frozen_contexts(kernel)[0], "unknown", mod.PRECISION_SPECS["binary32"], seed
        )


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    """Build the full bounded row product once for the module."""

    return mod.build_rows()


@pytest.fixture(scope="module")
def artifact(rows: list[dict]) -> dict:
    """Build one completed artifact from retained exact rows."""

    return mod.build_artifact(rows=rows, duration_s=0.25, torx_sidecar=mod.inspect_torx_sidecar())


def test_req_hw_6751_matched_arms_have_complete_cartesian_rows(rows: list[dict]) -> None:
    """REQ-HW-6751-MATCHED: Each arm uses the same finite candidate bank."""

    expected = (
        len(mod.frozen_kernels())
        * len(mod.CONTEXT_LABELS)
        * len(mod.ARMS)
        * len(mod.DEPTHS)
        * len(mod.PRECISIONS)
        * len(mod.SEED_BUNDLES)
    )
    assert len(rows) == expected
    assert len({row["row_id"] for row in rows}) == expected
    assert all(row["row_sha256"] == mod.row_hash(row) for row in rows)
    assert all(row["candidate_evaluations"] == mod.TRAINING_BUDGET for row in rows)
    assert all(
        row["factor_capacity"] == row["candidate_bank_receipt"]["factor_capacity"] for row in rows
    )

    matched = {}
    for row in rows:
        key = (
            row["factor_id"],
            row["context_id"],
            row["precision"],
            row["seed_bundle_id"],
        )
        matched.setdefault(key, set()).add(row["candidate_bank_receipt"]["bank_sha256"])
    assert all(len(hashes) == 1 for hashes in matched.values())


def test_req_hw_6751_metrics_are_row_derived_and_refinement_reduces_tv(
    rows: list[dict], artifact: dict
) -> None:
    """REQ-HW-6751-METRICS: Exact rows support the aggregate conclusion."""

    derived = mod.derive_aggregates(rows)
    assert artifact["conditional_kl_by_factor"] == derived["conditional_kl_by_factor"]
    assert artifact["trajectory_tv_by_depth"] == derived["trajectory_tv_by_depth"]
    assert artifact["normalization_error_by_row"] == derived["normalization_error_by_row"]
    gate = artifact["positive_result_gate"]
    assert gate["passed"] is True
    assert gate["context_reduced"] or gate["trajectory_reduced"]
    assert gate["best_refined_mean_trajectory_tv"] < gate["independent_mean_trajectory_tv"]
    assert all(
        row["trajectory_fidelity_score"] == pytest.approx(1.0 - row["trajectory_tv"])
        for row in rows
    )


def test_req_hw_6751_scientific_gate_reports_each_owned_row_failure(rows: list[dict]) -> None:
    """REQ-HW-6751-COMPLETION: Each internal gate reports observed evidence."""

    changed_rows = deepcopy(rows[:-1])
    changed_rows[0]["row_sha256"] = "sha256:changed"
    changed_rows[1]["maximum_normalization_error"] = 1.0
    topology = mod.topology_receipts()
    topology[0]["receipt_sha256"] = "sha256:changed"
    precision = mod.precision_receipts()
    precision[0]["receipt_sha256"] = "sha256:changed"
    checks = {
        row["check"] for row in mod._scientific_gate_errors(changed_rows, topology, precision)
    }
    assert checks == {
        "complete_row_product",
        "row_hashes",
        "exact_normalization",
        "topology_receipts",
        "precision_receipts",
    }


def test_req_hw_6751_terminal_classes_follow_row_derived_gate(
    rows: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-HW-6751-COMPLETION: Positive, circular, null, and partial remain distinct."""

    sidecar = mod.unavailable_sidecar("test")
    partial = mod.build_artifact(rows=rows[:-1], duration_s=0.1, torx_sidecar=sidecar)
    assert partial["status"] == "complete_partial"
    assert partial["verdict_class"] == "partial"

    original = mod.derive_aggregates
    for gate_changes, status, verdict_class in (
        (
            {"context_reduced": False, "trajectory_reduced": True, "passed": True},
            "complete_circular_positive",
            "circular_positive",
        ),
        (
            {"context_reduced": False, "trajectory_reduced": False, "passed": False},
            "complete_null",
            "null",
        ),
    ):

        def selected_aggregates(selected_rows: object, changes: dict = gate_changes) -> dict:
            aggregate = original(selected_rows)  # type: ignore[arg-type]
            aggregate["positive_result_gate"].update(changes)
            return aggregate

        with monkeypatch.context() as context:
            context.setattr(mod, "derive_aggregates", selected_aggregates)
            built = mod.build_artifact(rows=rows, duration_s=0.1, torx_sidecar=sidecar)
        assert built["status"] == status
        assert built["verdict_class"] == verdict_class


def test_req_hw_6751_completed_artifact_is_simulator_only_and_self_consistent(
    artifact: dict,
) -> None:
    """REQ-HW-6751-COMPLETION: Completion follows exhaustive row evidence."""

    assert mod.validate_artifact(artifact) == []
    assert artifact["compiler_fidelity_completed"] is True
    assert artifact["status"] == "complete_circular_positive"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["hardware_used"] is False
    assert artifact["simulator_used"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["claim_scope"] == mod.CLAIM_SCOPE
    assert artifact["gate_check_summary"] == []
    assert artifact["field_principles"].keys() == artifact.keys()
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert artifact["compiler_provenance"]["internal"]["module_sha256"].startswith("sha256:")


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda value: value["rows"].pop(), "row_count_mismatch"),
        (
            lambda value: value["rows"][0].update(trajectory_tv=2.0),
            "row_hash_mismatch",
        ),
        (
            lambda value: value.update(conditional_kl_by_factor=[]),
            "conditional_kl_by_factor_mismatch",
        ),
        (
            lambda value: value.update(trajectory_tv_by_depth=[]),
            "trajectory_tv_by_depth_mismatch",
        ),
        (
            lambda value: value.update(normalization_error_by_row={}),
            "normalization_error_by_row_mismatch",
        ),
        (
            lambda value: value["topology_receipts"][0].update(categories=[]),
            "topology_receipt_hash_mismatch",
        ),
        (
            lambda value: value["precision_receipts"][0].update(format="changed"),
            "precision_receipt_hash_mismatch",
        ),
        (
            lambda value: value.update(field_principles={}),
            "field_principles_mismatch",
        ),
        (
            lambda value: value.update(inference_substrate="physical_tsu"),
            "inference_substrate_mismatch",
        ),
        (lambda value: value.update(hardware_used=True), "hardware_boundary_mismatch"),
        (lambda value: value.update(simulator_used=False), "simulator_boundary_mismatch"),
        (
            lambda value: value.update(reproducibility_checksum="sha256:bad"),
            "reproducibility_checksum_mismatch",
        ),
        (
            lambda value: value.update(compiler_fidelity_completed=False),
            "completion_mismatch",
        ),
        (lambda value: value.update(duration_s=-1.0), "duration_invalid"),
    ],
)
def test_scenario_hw_6751_mutated_artifacts_fail_closed(
    artifact: dict, mutation: object, expected_error: str
) -> None:
    """SCENARIO-HW-6751-FAIL-CLOSED: Evidence mutations cannot stay complete."""

    changed = deepcopy(artifact)
    mutation(changed)  # type: ignore[operator]
    assert expected_error in mod.validate_artifact(changed)


def test_req_hw_6751_official_torx_sidecar_is_measured_but_optional(artifact: dict) -> None:
    """REQ-HW-6751-PROVENANCE: Torx evidence is separate from internal authority."""

    sidecar = artifact["compiler_provenance"]["official_sidecar"]
    assert sidecar["distribution"] == "extro-torx"
    assert sidecar["version"] == "0.0.1"
    assert sidecar["available"] is True
    assert sidecar["passed"] is True
    assert {row["api"] for row in sidecar["conformance_rows"]} == {
        "torx.psc.PNOT.get_matrix",
        "torx.psc.PditShift.get_matrix",
    }

    failed = mod.inspect_torx_sidecar(
        importer=lambda _name: (_ for _ in ()).throw(ImportError("gone"))
    )
    assert failed["available"] is False
    assert failed["passed"] is False
    optional = mod.build_artifact(rows=artifact["rows"], duration_s=0.5, torx_sidecar=failed)
    assert optional["compiler_fidelity_completed"] is True
    assert optional["compiler_provenance"]["official_sidecar"]["failure"] == "ImportError: gone"


def test_scenario_hw_6751_internal_reference_block_writes_complete_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-6751-FAIL-CLOSED: An owned internal block stays explicit."""

    def unavailable() -> list[dict]:
        raise mod.CompilerReferenceUnavailable("enumerator disabled")

    output = tmp_path / "blocked.json"
    blocked = mod.run(output_path=output, row_builder=unavailable)
    assert output.is_file()
    assert blocked["status"] == "complete_blocked_compiler_reference"
    assert blocked["compiler_fidelity_completed"] is False
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("complete_blocked_compiler_reference")
    assert blocked["gate_check_summary"][0]["check"] == "internal_exact_reference"
    assert mod.validate_artifact(blocked) == []

    changed = deepcopy(blocked)
    changed["compiler_fidelity_completed"] = True
    assert "blocked_terminal_state_mismatch" in mod.validate_artifact(changed)


def test_req_hw_6751_writer_is_deterministic_except_monotonic_duration(tmp_path: Path) -> None:
    """REQ-HW-6751-COMPLETION: The command writes stable reproducible evidence."""

    first = mod.run(
        output_path=tmp_path / "first.json", torx_sidecar=mod.unavailable_sidecar("test")
    )
    second = mod.run(
        output_path=tmp_path / "second.json", torx_sidecar=mod.unavailable_sidecar("test")
    )
    assert first["duration_s"] >= 0.0
    assert second["duration_s"] >= 0.0
    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert first["rows"] == second["rows"]
    assert json.loads((tmp_path / "first.json").read_text(encoding="utf-8")) == first


def test_scenario_hw_6751_validation_and_writer_fail_closed(
    artifact: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-HW-6751-FAIL-CLOSED: Missing fields and writer errors stay visible."""

    missing = deepcopy(artifact)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["required_fields_missing"]

    stale_gate = deepcopy(artifact)
    stale_gate["gate_check_summary"] = [{"check": "stale"}]
    assert "completed_gate_summary_not_empty" in mod.validate_artifact(stale_gate)

    with monkeypatch.context() as context:
        context.setattr(mod, "validate_artifact", lambda _artifact: ["forced_failure"])
        with pytest.raises(mod.CompilerInputError, match="artifact validation failed"):
            mod.run(
                output_path=tmp_path / "invalid.json",
                row_builder=lambda: [],
                torx_sidecar=mod.unavailable_sidecar("test"),
            )


def test_req_hw_6751_cli_parser_writes_requested_path(tmp_path: Path) -> None:
    """REQ-HW-6751-BOUNDARY: The entry point accepts only bounded local output."""

    output = tmp_path / "cli.json"
    exit_code = mod.main(["--output", str(output), "--skip-torx-sidecar"])
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["compiler_fidelity_completed"] is True
    assert payload["compiler_provenance"]["official_sidecar"]["available"] is False


def test_req_hw_6751_module_main_guard_runs_bounded_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-HW-6751-BOUNDARY: Direct module execution uses the same bounded path."""

    output = tmp_path / "runpy.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(mod.REPO_ROOT / mod.MODULE_PATH), "--output", str(output), "--skip-torx-sidecar"],
    )
    with pytest.raises(SystemExit, match="0"):
        runpy.run_path(str(mod.REPO_ROOT / mod.MODULE_PATH), run_name="__main__")
    assert json.loads(output.read_text(encoding="utf-8"))["compiler_fidelity_completed"] is True

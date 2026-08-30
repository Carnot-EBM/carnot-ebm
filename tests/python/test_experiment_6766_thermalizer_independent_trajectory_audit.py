"""Tests for the independent Thermalizer trajectory audit.

Spec: REQ-HW-6766, REQ-HW-6766-PRECONDITIONS,
REQ-HW-6766-INDEPENDENCE, REQ-HW-6766-EXACT,
REQ-HW-6766-SAMPLER, REQ-HW-6766-ROWS,
REQ-HW-6766-CIRCULARITY, REQ-HW-6766-COMPLETION,
REQ-HW-6766-BOUNDARY, SCENARIO-HW-6766-COLD-REPRODUCTION,
SCENARIO-HW-6766-CIRCULAR-REFINEMENT,
SCENARIO-HW-6766-FAIL-CLOSED, REQ-REPORT-6766,
SCENARIO-REPORT-6766-ATOMIC, SCENARIO-REPORT-6766-BLOCKED.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_6766_thermalizer_independent_trajectory_audit as mod


@pytest.fixture(scope="module")
def source() -> dict:
    """REQ-HW-6766-PRECONDITIONS: Load the frozen source without running it."""

    return json.loads((mod.REPO_ROOT / mod.SOURCE_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifact(source: dict) -> dict:
    """REQ-HW-6766-ROWS: Build the complete cold grid once for this test module."""

    return mod.build_artifact(source, run_date="20260830", duration_s=0.25, samples_per_row=1024)


def test_req_hw_6766_specs_precede_implementation() -> None:
    """REQ-HW-6766: Both capability specs own the cold-audit contract."""

    hardware = (mod.REPO_ROOT / mod.HARDWARE_SPEC_PATH).read_text(encoding="utf-8")
    reporting = (mod.REPO_ROOT / mod.REPORTING_SPEC_PATH).read_text(encoding="utf-8")
    anchors = set(mod.find_spec_anchors(hardware + reporting))
    assert {
        "REQ-HW-6766-PRECONDITIONS",
        "REQ-HW-6766-INDEPENDENCE",
        "REQ-HW-6766-EXACT",
        "REQ-HW-6766-SAMPLER",
        "REQ-HW-6766-ROWS",
        "REQ-HW-6766-CIRCULARITY",
        "REQ-HW-6766-COMPLETION",
        "REQ-HW-6766-BOUNDARY",
        "SCENARIO-HW-6766-COLD-REPRODUCTION",
        "SCENARIO-HW-6766-CIRCULAR-REFINEMENT",
        "SCENARIO-HW-6766-FAIL-CLOSED",
        "REQ-REPORT-6766",
        "SCENARIO-REPORT-6766-ATOMIC",
        "SCENARIO-REPORT-6766-BLOCKED",
    } <= anchors


def test_req_hw_6766_evaluator_has_no_exp6751_import_or_function_edge() -> None:
    """REQ-HW-6766-INDEPENDENCE: The evaluator does not call source helpers."""

    tree = ast.parse((mod.REPO_ROOT / mod.MODULE_PATH).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("experiment_6751" in name for name in imports)
    assert not set(mod.FORBIDDEN_SOURCE_FUNCTIONS) & set(mod.local_call_names(tree))


def test_req_hw_6766_serialized_factors_and_conditionals_normalize(source: dict) -> None:
    """REQ-HW-6766-EXACT: Independent factor schemas reproduce finite laws."""

    factors = mod.serialized_factors()
    assert {row["factor_id"] for row in factors} == {
        "binary_sticky_transition",
        "categorical_ring_transition",
    }
    topology = {row["factor_id"]: row for row in source["topology_receipts"]}
    for factor in factors:
        assert all(math.fsum(row) == pytest.approx(1.0) for row in factor["target_conditional"])
        assert (
            mod.json_digest(factor["target_conditional"])
            == topology[factor["factor_id"]]["target_conditional_sha256"]
        )
        zero = [0.0] * len(factor["parameter_names"])
        conditional = mod.build_compiled_conditional(factor, zero, "binary32")
        assert all(math.fsum(row) == pytest.approx(1.0) for row in conditional)
        assert all(value > 0.0 for row in conditional for value in row)

    binary = factors[0]
    with pytest.raises(mod.AuditInputError, match="parameter count"):
        mod.build_compiled_conditional(binary, [0.0], "binary32")
    with pytest.raises(mod.AuditInputError, match="unknown precision"):
        mod.build_compiled_conditional(binary, [0.0, 0.0], "float16")
    with pytest.raises(mod.AuditInputError, match="finite"):
        mod.build_compiled_conditional(binary, [float("nan"), 0.0], "binary32")
    changed = deepcopy(binary)
    changed["feature_tensor"][0][0] = [1.0]
    with pytest.raises(mod.AuditInputError, match="feature tensor"):
        mod.build_compiled_conditional(changed, [0.0, 0.0], "binary32")
    with pytest.raises(mod.AuditInputError, match="canonical JSON"):
        mod.canonical_json_text({"bad": float("nan")})


def test_req_hw_6766_exact_enumerator_and_metrics_are_independent() -> None:
    """REQ-HW-6766-EXACT: Exact path laws and metrics have direct known values."""

    initial = [0.75, 0.25]
    target = [[0.8, 0.2], [0.1, 0.9]]
    compiled = [[0.7, 0.3], [0.2, 0.8]]
    target_law = mod.enumerate_path_law(initial, target, 2, maximum_paths=8)
    compiled_law = mod.enumerate_path_law(initial, compiled, 2, maximum_paths=8)
    assert len(target_law) == 8
    assert math.fsum(target_law.values()) == pytest.approx(1.0)
    assert target_law[(0, 0, 1)] == pytest.approx(0.75 * 0.8 * 0.2)
    tv = mod.law_total_variation(target_law, compiled_law)
    metrics = mod.conditional_kl_metrics(target, compiled, initial, 2)
    assert 0.0 < tv < 1.0
    assert len(metrics["conditional_kl_by_input"]) == 2
    assert metrics["conditional_kl"] > 0.0
    assert math.fsum(metrics["input_weights"]) == pytest.approx(1.0)

    with pytest.raises(mod.AuditInputError, match="positive"):
        mod.enumerate_path_law(initial, target, 0, maximum_paths=8)
    with pytest.raises(mod.AuditInputError, match="square"):
        mod.enumerate_path_law(initial, [[1.0, 0.0]], 1, maximum_paths=8)
    with pytest.raises(mod.AuditInputError, match="initial"):
        mod.enumerate_path_law([1.0], target, 1, maximum_paths=8)
    with pytest.raises(mod.AuditBoundError, match="exceeds"):
        mod.enumerate_path_law(initial, target, 3, maximum_paths=8)
    with pytest.raises(mod.AuditInputError, match="same support"):
        mod.law_total_variation({(0,): 1.0}, {(1,): 1.0})
    with pytest.raises(mod.AuditInputError, match="normalize"):
        mod.enumerate_path_law([0.2, 0.2], target, 1, maximum_paths=8)
    with pytest.raises(mod.AuditInputError, match="finite and nonnegative"):
        mod.enumerate_path_law([float("nan"), 0.0], target, 1, maximum_paths=8)
    with pytest.raises(mod.AuditInputError, match="same size"):
        mod.conditional_kl_metrics(target, [[1 / 3] * 3] * 3, initial, 2)
    with pytest.raises(mod.AuditInputError, match="full support"):
        mod.conditional_kl_metrics(target, [[1.0, 0.0], [0.2, 0.8]], initial, 2)


def test_req_hw_6766_direct_sampler_is_seeded_and_does_not_enumerate() -> None:
    """REQ-HW-6766-SAMPLER: Likelihood-ratio sampling is reproducible."""

    initial = [0.75, 0.25]
    target = [[0.8, 0.2], [0.1, 0.9]]
    compiled = [[0.7, 0.3], [0.2, 0.8]]
    exact = mod.law_total_variation(
        mod.enumerate_path_law(initial, target, 2, maximum_paths=8),
        mod.enumerate_path_law(initial, compiled, 2, maximum_paths=8),
    )
    first = mod.sample_trajectory_tv(initial, target, compiled, 2, seed=6766, samples=20000)
    second = mod.sample_trajectory_tv(initial, target, compiled, 2, seed=6766, samples=20000)
    changed = mod.sample_trajectory_tv(initial, target, compiled, 2, seed=6767, samples=20000)
    assert first == second
    assert first != changed
    assert abs(first["sampled_trajectory_tv"] - exact) < 0.02
    assert first["ci99_low"] <= exact <= first["ci99_high"]
    assert first["api_path"].endswith("sample_trajectory_tv")
    with pytest.raises(mod.AuditInputError, match="sample count"):
        mod.sample_trajectory_tv(initial, target, compiled, 2, seed=1, samples=1)
    with pytest.raises(mod.AuditInputError, match="positive"):
        mod.sample_trajectory_tv(initial, target, compiled, 0, seed=1, samples=2)

    class LastCategory:
        def random(self) -> float:
            return 1.0

    assert mod._draw_category([0.5, 0.5], LastCategory()) == 1  # type: ignore[arg-type]


def test_req_hw_6766_sampler_rejects_a_forced_zero_support_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-HW-6766-SAMPLER: A zero target likelihood cannot enter a ratio."""

    monkeypatch.setattr(mod, "_draw_category", lambda _probabilities, _generator: 1)
    with pytest.raises(mod.AuditInputError, match="full support"):
        mod.sample_trajectory_tv(
            [1.0, 0.0],
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.5, 0.5], [0.5, 0.5]],
            1,
            seed=1,
            samples=2,
        )


def test_req_hw_6766_source_preconditions_detect_receipt_and_bound_failures(source: dict) -> None:
    """REQ-HW-6766-PRECONDITIONS: Frozen inputs fail closed before evaluation."""

    clean = mod.check_source_preconditions(source)
    assert clean["gate_check_summary"] == []
    assert clean["topology_mismatches"] == []
    assert clean["precision_mismatches"] == []
    assert clean["maximum_planned_path_count"] == 19683

    missing = deepcopy(source)
    missing["rows"].pop()
    assert "complete_source_row_grid" in {
        row["check"] for row in mod.check_source_preconditions(missing)["gate_check_summary"]
    }
    topology = deepcopy(source)
    topology["topology_receipts"][0]["categories"] = []
    checked = mod.check_source_preconditions(topology)
    assert checked["topology_mismatches"]
    assert "topology_receipts" in {row["check"] for row in checked["gate_check_summary"]}
    precision = deepcopy(source)
    precision["precision_receipts"][0]["format"] = "changed"
    checked = mod.check_source_preconditions(precision)
    assert checked["precision_mismatches"]
    missing_precision = deepcopy(source)
    missing_precision["precision_receipts"].pop()
    assert mod.check_source_preconditions(missing_precision)["precision_mismatches"]
    bounded = deepcopy(source)
    bounded["frozen_config"]["maximum_enumerated_trajectory_states"] = 100
    assert "bounded_exact_enumeration" in {
        row["check"] for row in mod.check_source_preconditions(bounded)["gate_check_summary"]
    }
    malformed = deepcopy(source)
    malformed.pop("random_seed")
    assert "required_source_fields" in {
        row["check"] for row in mod.check_source_preconditions(malformed)["gate_check_summary"]
    }


def test_req_hw_6766_isolation_gate_rejects_a_dependency_edge(
    source: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-HW-6766-INDEPENDENCE: A compiler import blocks the cold evaluator."""

    receipt = mod.dependency_graph_receipt(source)
    receipt["evaluator_imports_compiler_module"] = True
    receipt["evaluator_to_compiler_dependency_edge"] = True
    receipt["receipt_sha256"] = mod.receipt_digest(receipt)
    monkeypatch.setattr(mod, "dependency_graph_receipt", lambda _source: receipt)
    checked = mod.check_source_preconditions(source)
    assert "independent_evaluator_isolation" in {
        row["check"] for row in checked["gate_check_summary"]
    }


def test_req_hw_6766_full_rows_recompute_source_metrics_and_receipts(artifact: dict) -> None:
    """REQ-HW-6766-ROWS: Every source unit has exact and sampled audit rows."""

    assert artifact["independent_trajectory_audit_completed"] is True
    assert len(artifact["rows"]) == 384
    assert len({row["row_id"] for row in artifact["rows"]}) == 384
    assert all(row["row_sha256"] == mod.audit_row_digest(row) for row in artifact["rows"])
    exact = [row for row in artifact["rows"] if row["evaluator_path"] == "exact_enumerator"]
    sampled = [row for row in artifact["rows"] if row["evaluator_path"] == "direct_sampler"]
    assert len(exact) == len(sampled) == 192
    assert max(abs(row["source_conditional_kl_delta"]) for row in exact) < 1.0e-12
    assert max(abs(row["source_trajectory_tv_delta"]) for row in exact) < 1.0e-12
    assert all(row["target_trajectory_receipt_matches"] for row in exact)
    assert artifact["source_metric_crosscheck"]["compiled_trajectory_receipt_mismatch_count"] > 0
    assert (
        artifact["source_metric_crosscheck"]["compiled_trajectory_bit_identity_required"] is False
    )
    assert artifact["source_metric_crosscheck"]["passed"] is True
    assert all(row["conditional_kl"] is None for row in sampled)
    assert all(row["sample_count"] == 1024 for row in sampled)


def test_req_hw_6766_reducer_derives_intervals_deltas_and_circularity(artifact: dict) -> None:
    """REQ-HW-6766-CIRCULARITY: Claims and oracle state derive from rows."""

    reduced = mod.reduce_audit_rows(artifact["rows"])
    assert artifact["conditional_kl_by_factor"] == reduced["conditional_kl_by_factor"]
    assert artifact["trajectory_tv_by_depth"] == reduced["trajectory_tv_by_depth"]
    assert artifact["paired_trajectory_deltas"] == reduced["paired_trajectory_deltas"]
    assert artifact["direct_sampler_crosscheck"] == reduced["direct_sampler_crosscheck"]
    context = next(
        row
        for row in artifact["paired_trajectory_deltas"]
        if row["method"] == "context_matched" and row["depth"] == "all"
    )
    assert context["mean_independent_minus_method_tv"] > 0.0
    assert context["ci95_low"] > 0.0
    assert context["interval_excludes_zero"] is True
    circular = [row for row in artifact["rows"] if row["mechanism_consumes_evaluator_outcome"]]
    assert circular
    assert {row["method"] for row in circular} == {"trajectory_refinement"}
    assert {row["evaluator_path"] for row in circular} == {"exact_enumerator"}
    assert artifact["evaluator_distinct"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_hw_6766_provenance_records_source_drift_without_sharing_code(artifact: dict) -> None:
    """REQ-HW-6766-INDEPENDENCE: Code identity and source drift remain visible."""

    dependency = artifact["dependency_graph_receipt"]
    assert dependency["evaluator_imports_compiler_module"] is False
    assert dependency["shared_callable_objects"] == []
    assert dependency["compiler_to_evaluator_dependency_edge"] is False
    assert dependency["evaluator_to_compiler_dependency_edge"] is False
    assert dependency["same_module_sha256"] is False
    assert dependency["methods_consuming_exact_evaluator_outcome"] == ["trajectory_refinement"]
    assert dependency["receipt_sha256"] == mod.receipt_digest(dependency)
    assert artifact["compiler_provenance"]["declared_module_sha256"].startswith("sha256:")
    assert artifact["evaluator_provenance"]["module_sha256"].startswith("sha256:")
    assert artifact["source_artifact_sha256"].startswith("sha256:")
    assert artifact["claim_boundary"] == mod.CLAIM_BOUNDARY
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["field_principles"].keys() == artifact.keys()


def test_scenario_hw_6766_blocked_artifact_keeps_complete_schema(source: dict) -> None:
    """SCENARIO-HW-6766-FAIL-CLOSED: A failed gate emits no invented rows."""

    changed = deepcopy(source)
    changed["rows"].pop()
    blocked = mod.build_artifact(changed, run_date="20260830", duration_s=0.1, samples_per_row=64)
    assert blocked["status"] == "complete_blocked_thermalizer_audit"
    assert blocked["honest_verdict"].startswith("complete_blocked_thermalizer_audit")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["independent_trajectory_audit_completed"] is False
    assert blocked["rows"] == []
    assert blocked["gate_check_summary"]
    assert blocked["field_principles"].keys() == blocked.keys()
    assert mod.validate_artifact(blocked) == []

    broken = deepcopy(blocked)
    broken["rows"] = [{}]
    assert "blocked_terminal_state_mismatch" in mod.validate_artifact(broken)


def test_scenario_hw_6766_evaluation_error_becomes_an_owned_block(
    source: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-HW-6766-FAIL-CLOSED: Evaluator errors keep observed reasons."""

    monkeypatch.setattr(
        mod,
        "evaluate_source_rows",
        lambda _source, *, samples_per_row: (_ for _ in ()).throw(
            mod.AuditInputError(f"forced at {samples_per_row}")
        ),
    )
    blocked = mod.build_artifact(source, run_date="20260830", duration_s=0.1, samples_per_row=2)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["check"] == "independent_evaluation"


def test_req_hw_6766_terminal_reducer_covers_partial_positive_and_null(
    source: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-HW-6766-COMPLETION: Each closed terminal class follows row evidence."""

    original_evaluate = mod.evaluate_source_rows
    original_reduce = mod.reduce_audit_rows

    def partial_reduce(rows: object) -> dict:
        reduced = original_reduce(rows)  # type: ignore[arg-type]
        reduced["direct_sampler_crosscheck"]["passed"] = False
        return reduced

    with monkeypatch.context() as context:
        context.setattr(mod, "reduce_audit_rows", partial_reduce)
        partial = mod.build_artifact(source, run_date="20260830", duration_s=0.1, samples_per_row=2)
    assert partial["verdict_class"] == "partial"

    def noncircular_rows(source_value: object, *, samples_per_row: int) -> list[dict]:
        rows = original_evaluate(source_value, samples_per_row=samples_per_row)  # type: ignore[arg-type]
        for row in rows:
            row["mechanism_consumes_evaluator_outcome"] = False
            row["row_sha256"] = mod.audit_row_digest(row)
        return rows

    def completed_reduce(rows: object) -> dict:
        reduced = original_reduce(rows)  # type: ignore[arg-type]
        reduced["direct_sampler_crosscheck"]["passed"] = True
        return reduced

    with monkeypatch.context() as context:
        context.setattr(mod, "evaluate_source_rows", noncircular_rows)
        context.setattr(mod, "reduce_audit_rows", completed_reduce)
        positive = mod.build_artifact(
            source, run_date="20260830", duration_s=0.1, samples_per_row=2
        )
    assert positive["verdict_class"] == "positive"

    def null_reduce(rows: object) -> dict:
        reduced = original_reduce(rows)  # type: ignore[arg-type]
        reduced["direct_sampler_crosscheck"]["passed"] = True
        for row in reduced["paired_trajectory_deltas"]:
            if row["method"] == "context_matched" and row["depth"] == "all":
                row["ci95_low"] = -0.01
                row["interval_excludes_zero"] = False
        return reduced

    with monkeypatch.context() as context:
        context.setattr(mod, "reduce_audit_rows", null_reduce)
        null = mod.build_artifact(source, run_date="20260830", duration_s=0.1, samples_per_row=2)
    assert null["verdict_class"] == "null"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value["rows"].pop(), "row_grid_mismatch"),
        (lambda value: value["rows"][0].update(trajectory_tv=2.0), "row_hash_mismatch"),
        (lambda value: value.update(conditional_kl_by_factor=[]), "aggregate_mismatch"),
        (lambda value: value.update(evaluator_distinct=False), "distinctness_mismatch"),
        (lambda value: value.update(verifier_is_oracle=False), "oracle_mismatch"),
        (lambda value: value.update(verdict_class="positive"), "verdict_class_mismatch"),
        (lambda value: value.update(reproducibility_checksum="sha256:bad"), "checksum_mismatch"),
        (lambda value: value.update(field_principles={}), "field_principles_mismatch"),
        (lambda value: value.update(duration_s=-1.0), "duration_invalid"),
        (lambda value: value.update(claim_boundary="physical Z1"), "claim_boundary_mismatch"),
    ],
)
def test_scenario_report_6766_mutated_artifacts_fail_validation(
    artifact: dict, mutation: object, expected: str
) -> None:
    """SCENARIO-REPORT-6766-ATOMIC: Mutated evidence cannot remain complete."""

    changed = deepcopy(artifact)
    mutation(changed)  # type: ignore[operator]
    assert expected in mod.validate_artifact(changed)


def test_req_report_6766_validation_covers_schema_and_receipt_failures(artifact: dict) -> None:
    """REQ-REPORT-6766: Schema, substrate, and dependency receipts fail closed."""

    missing = deepcopy(artifact)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["required_fields_missing"]
    substrate = deepcopy(artifact)
    substrate["inference_substrate"] = "physical"
    assert "inference_substrate_mismatch" in mod.validate_artifact(substrate)
    dependency = deepcopy(artifact)
    dependency["dependency_graph_receipt"]["receipt_sha256"] = "sha256:bad"
    assert "dependency_receipt_mismatch" in mod.validate_artifact(dependency)


def test_req_report_6766_run_and_cli_write_only_requested_paths(
    source: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6766: The writer is atomic and supports a dated cold run."""

    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "audit.json"
    written = mod.run(
        source_path=source_path,
        output_path=output,
        run_date="20260830",
        samples_per_row=128,
    )
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == written
    assert mod.validate_artifact(written) == []

    cli_output = tmp_path / "cli.json"
    assert (
        mod.main(
            [
                "--source",
                str(source_path),
                "--output",
                str(cli_output),
                "--date",
                "20260830",
                "--samples-per-row",
                "64",
            ]
        )
        == 0
    )
    assert json.loads(cli_output.read_text(encoding="utf-8"))["run_date"] == "20260830"

    blocked_source = tmp_path / "blocked.json"
    blocked_source.write_text("{}", encoding="utf-8")
    guard_output = tmp_path / "guard.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(mod.REPO_ROOT / mod.MODULE_PATH),
            "--source",
            str(blocked_source),
            "--output",
            str(guard_output),
            "--date",
            "20260830",
            "--samples-per-row",
            "64",
        ],
    )
    with pytest.raises(SystemExit, match="0"):
        runpy.run_path(str(mod.REPO_ROOT / mod.MODULE_PATH), run_name="__main__")
    assert json.loads(guard_output.read_text(encoding="utf-8"))["verdict_class"] == "blocked"


def test_req_hw_6766_load_failure_is_a_complete_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6766-BLOCKED: Invalid JSON records the parse failure."""

    source_path = tmp_path / "invalid.json"
    source_path.write_text("{", encoding="utf-8")
    output = tmp_path / "blocked.json"
    artifact = mod.run(
        source_path=source_path,
        output_path=output,
        run_date="20260830",
        samples_per_row=64,
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"][0]["check"] == "source_artifact_parse"
    assert mod.validate_artifact(artifact) == []

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    list_artifact = mod.run(
        source_path=non_object,
        output_path=tmp_path / "list-blocked.json",
        run_date="20260830",
        samples_per_row=2,
    )
    assert list_artifact["verdict_class"] == "blocked"


def test_req_report_6766_run_refuses_an_invalid_built_artifact(
    source: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6766: A validation error prevents the atomic write."""

    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced"])
    with pytest.raises(mod.AuditInputError, match="artifact validation failed"):
        mod.run(
            source_path=source_path,
            output_path=tmp_path / "must-not-exist.json",
            run_date="20260830",
            samples_per_row=2,
        )

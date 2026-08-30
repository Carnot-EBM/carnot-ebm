"""Tests for the exact-enumerable temporal exchange comparison.

Spec: REQ-SAMPLE-097, SCENARIO-SAMPLE-097, SCENARIO-SAMPLE-098,
SCENARIO-SAMPLE-099.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from carnot import experiment_6793_temporal_exchange_ising_ab as mod


@pytest.fixture(scope="module")
def completed_parts() -> tuple[list[dict], list[dict], dict]:
    """Build the fixed panels once because all artifact tests share them."""

    targets = mod.build_exact_targets()
    rows = mod.build_headline_rows(targets)
    stress_rows = mod.build_stress_rows()
    artifact = mod.build_artifact(
        rows=rows,
        stress_rows=stress_rows,
        targets=targets,
        run_date="20260830",
        duration_s=1.0,
        preconditions=mod.check_preconditions(),
    )
    return rows, stress_rows, artifact


def test_req_sample_097_frozen_manifest_is_bounded_and_preregistered() -> None:
    """REQ-SAMPLE-097: Graphs, temperatures, seeds, work, and signs are frozen."""

    manifest = mod.frozen_manifest()
    graphs = mod.frozen_graphs()
    assert [graph.n_spins for graph in graphs] == [6, 7, 8]
    assert manifest["temperatures"] == [0.75, 2.0]
    assert manifest["seeds"] == list(range(679300, 679320))
    assert manifest["collected_samples"] == 1024
    assert manifest["burn_in_sweeps"] == 128
    assert manifest["target_law_noninferiority_margin_tv"] == 0.03
    assert manifest["coupling_grid"] == [-0.08, 0.0, 0.08]
    assert mod.coupling_for_temperature(0.75) == -0.08
    assert mod.coupling_for_temperature(2.0) == 0.08
    assert all(np.max(np.abs(graph.biases)) <= 0.15 for graph in graphs)
    assert all(np.max(np.abs(graph.couplings)) <= 0.60 for graph in graphs)


def test_req_sample_097_preconditions_cover_sampler_bounds_clock_ram_and_wall() -> None:
    """REQ-SAMPLE-097: Owned CPU resources pass before the panel starts."""

    checks = mod.check_preconditions()
    assert {row["check"] for row in checks} == {
        "cpu_sampler_importable",
        "headline_exact_enumeration_bound",
        "temperature_and_coefficient_ranges",
        "monotonic_timing",
        "ram_budget",
        "wall_budget",
    }
    assert all(row["passed"] for row in checks)


def test_scenario_sample_099_exact_targets_normalize_and_have_stable_hashes() -> None:
    """SCENARIO-SAMPLE-099: Every headline graph-temperature law is exact."""

    first = mod.build_exact_targets()
    second = mod.build_exact_targets()
    assert set(first) == set(second)
    for key, target in first.items():
        assert target["exact"] is True
        assert len(target["probabilities"]) == 2 ** target["n_spins"]
        assert sum(target["probabilities"]) == pytest.approx(1.0)
        assert target["target_sha256"] == second[key]["target_sha256"]


def test_scenario_sample_098_headline_rows_are_matched_and_zero_arm_is_identical(
    completed_parts: tuple[list[dict], list[dict], dict],
) -> None:
    """SCENARIO-SAMPLE-098: Every matched cell has equal work and collection."""

    rows, _stress, _artifact = completed_parts
    expected = 3 * 2 * 20 * 3
    assert len(rows) == expected
    assert len({row["row_id"] for row in rows}) == expected
    assert all(row["row_sha256"] == mod.row_digest(row) for row in rows)

    index = {(row["graph_id"], row["temperature"], row["seed"], row["arm"]): row for row in rows}
    for graph in mod.frozen_graphs():
        for temperature in mod.TEMPERATURES:
            for seed in mod.SEEDS:
                common = (graph.graph_id, temperature, seed)
                ordinary = index[(*common, "ordinary_gibbs")]
                temporal = index[(*common, "temporal_exchange")]
                disabled = index[(*common, "temporal_exchange_zero_coupling")]
                assert ordinary["update_count"] == temporal["update_count"]
                assert ordinary["collection_update_counts"] == temporal["collection_update_counts"]
                assert ordinary["initial_state_pair"] == temporal["initial_state_pair"]
                assert ordinary["sampler_seed"] == temporal["sampler_seed"]
                assert ordinary["trajectory_sha256"] == disabled["trajectory_sha256"]
                assert ordinary["empirical_marginal"] == disabled["empirical_marginal"]
                assert ordinary["energy_trace"] == disabled["energy_trace"]
                assert ordinary["optimum_hitting_updates"] == disabled["optimum_hitting_updates"]


def test_scenario_sample_099_artifact_reductions_and_terminal_gate_are_row_derived(
    completed_parts: tuple[list[dict], list[dict], dict],
) -> None:
    """SCENARIO-SAMPLE-099: Complete measured rows support positive or null."""

    rows, stress_rows, artifact = completed_parts
    assert artifact["temporal_exchange_comparison_completed"] is True
    assert artifact["verdict_class"] in {"positive", "null"}
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["physical_hardware_invoked"] is False
    assert artifact["inference_substrate"] == (
        "CPU exact-enumerable Ising simulation, no physical hardware"
    )
    assert artifact["stress_rows_separate"] is True
    assert artifact["stress_rows"] == stress_rows
    assert all(row["headline_fidelity_eligible"] is False for row in stress_rows)
    assert artifact["gate_check_summary"] == []
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    derived = mod.derive_aggregates(rows)
    for field in mod.AGGREGATE_FIELDS:
        assert artifact[field] == derived[field]
    assert len(artifact["paired_efficiency_effects"]) == 6
    assert len(artifact["paired_efficiency_lcb"]["by_stratum"]) == 6


def test_req_sample_097_rows_preserve_required_statistical_evidence(
    completed_parts: tuple[list[dict], list[dict], dict],
) -> None:
    """REQ-SAMPLE-097: Each row keeps target, trace, dependence, and work data."""

    rows, _stress, _artifact = completed_parts
    for row in rows:
        assert sum(row["target_marginal"]) == pytest.approx(1.0)
        assert sum(row["empirical_marginal"]) == pytest.approx(1.0)
        assert len(row["energy_trace"]) == mod.COLLECTED_SAMPLES
        assert len(row["collection_update_counts"]) == mod.COLLECTED_SAMPLES
        assert set(row["autocorrelation"]) == {"energy", "magnetization"}
        assert row["effective_samples"]["energy"] > 0.0
        assert row["effective_samples_per_update"] > 0.0
        assert row["target_total_variation"] >= 0.0
        assert row["magnetization_error"] >= 0.0
        assert row["energy_error"] >= 0.0
        assert row["diversity"]["unique_state_count"] >= 1
        assert row["wall_time_s"] >= 0.0


def test_req_sample_097_initial_state_pairs_and_rows_are_deterministic() -> None:
    """REQ-SAMPLE-097: Stable seeds reproduce initial pairs and one complete cell."""

    graph = mod.frozen_graphs()[0]
    first_pair = mod.initial_state_pair(graph, mod.SEEDS[0])
    second_pair = mod.initial_state_pair(graph, mod.SEEDS[0])
    assert first_pair == second_pair

    target = mod.build_exact_targets()[(graph.graph_id, mod.TEMPERATURES[0])]
    first = mod.build_headline_row(
        graph,
        mod.TEMPERATURES[0],
        mod.SEEDS[0],
        "ordinary_gibbs",
        target,
    )
    second = mod.build_headline_row(
        graph,
        mod.TEMPERATURES[0],
        mod.SEEDS[0],
        "ordinary_gibbs",
        target,
    )
    assert mod.reproducible_row(first) == mod.reproducible_row(second)
    assert first["row_sha256"] == second["row_sha256"]


def test_req_sample_097_blocked_precondition_artifact_is_complete_schema(tmp_path: Path) -> None:
    """REQ-SAMPLE-097: A failed owned precondition writes the required block."""

    failed = [
        {
            "check": "ram_budget",
            "passed": False,
            "expected": {"available_bytes_at_least": 1_000_000},
            "observed": {"available_bytes": 10},
        }
    ]
    output = tmp_path / "blocked.json"
    artifact = mod.run(
        output_path=output,
        run_date="20260830",
        precondition_getter=lambda: failed,
    )
    assert artifact["status"] == "complete_blocked_temporal_exchange_ab"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["temporal_exchange_comparison_completed"] is False
    assert artifact["gate_check_summary"] == failed
    assert artifact["honest_verdict"].startswith("complete_blocked_temporal_exchange_ab")
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_sample_099_validation_detects_row_and_aggregate_tampering(
    completed_parts: tuple[list[dict], list[dict], dict],
) -> None:
    """SCENARIO-SAMPLE-099: Validation rejects changed evidence and verdicts."""

    _rows, _stress, artifact = completed_parts
    changed = deepcopy(artifact)
    changed["rows"][0]["update_count"] += 1
    changed["target_law_error_by_arm_family"] = []
    changed["physical_hardware_invoked"] = True
    changed["verdict_class"] = "invalid"
    errors = mod.validate_artifact(changed)
    assert "row_hash_mismatch" in errors
    assert "aggregate_mismatch" in errors
    assert "physical_hardware_boundary_mismatch" in errors
    assert "verdict_class_mismatch" in errors


def test_req_sample_097_cli_wrapper_dispatches_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SAMPLE-097: The required script is a thin package entry point."""

    called: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> int:
        called.append(argv)
        return 0

    monkeypatch.setattr(mod, "main", fake_main)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(mod.REPO_ROOT / mod.SCRIPT_PATH), run_name="__main__")
    assert exc.value.code == 0
    assert called == [None]


def test_req_sample_097_helper_and_precondition_failure_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLE-097: Helper failures remain explicit and attributable."""

    with pytest.raises(mod.TemporalExchangeExperimentError, match="canonical JSON"):
        mod.canonical_json({"bad": float("nan")})
    with pytest.raises(mod.TemporalExchangeExperimentError, match="frozen schedule"):
        mod.coupling_for_temperature(1.25)
    monkeypatch.setattr(mod.os, "sysconf", lambda _name: (_ for _ in ()).throw(OSError("no")))
    assert mod._available_ram_bytes() == 0
    assert mod._autocorrelation([2.0, 2.0, 2.0]) == {
        "integrated_time": 3.0,
        "effective_samples": 1.0,
        "positive_lag_count": 0,
    }


def test_scenario_sample_099_completion_errors_and_partial_artifact_are_fail_closed(
    completed_parts: tuple[list[dict], list[dict], dict],
) -> None:
    """SCENARIO-SAMPLE-099: Missing, changed, and mixed rows prevent completion."""

    rows, stress_rows, _artifact = completed_parts
    targets = mod.build_exact_targets()
    changed_rows = deepcopy(rows)
    changed_rows[0]["row_sha256"] = "sha256:changed"
    disabled = next(
        row
        for row in changed_rows
        if row["arm"] == "temporal_exchange_zero_coupling"
        and row["graph_id"] == mod.frozen_graphs()[0].graph_id
        and row["temperature"] == mod.TEMPERATURES[0]
        and row["seed"] == mod.SEEDS[0]
    )
    disabled["trajectory_sha256"] = "sha256:different"
    disabled["row_sha256"] = mod.row_digest(disabled)
    changed_stress = deepcopy(stress_rows)
    changed_stress[0]["headline_fidelity_eligible"] = True
    errors = mod._completion_errors(changed_rows, changed_stress, {})
    assert {row["check"] for row in errors} == {
        "headline_row_hashes",
        "exact_target_grid",
        "disabled_coupling_equivalence",
        "separate_stress_rows",
    }

    partial = mod.build_artifact(
        rows=rows[:-1],
        stress_rows=stress_rows,
        targets=targets,
        run_date="20260830",
        duration_s=1.0,
        preconditions=mod.check_preconditions(),
    )
    assert partial["verdict_class"] == "partial"
    assert partial["temporal_exchange_comparison_completed"] is False


def test_scenario_sample_099_positive_branch_uses_only_frozen_gates(
    completed_parts: tuple[list[dict], list[dict], dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-SAMPLE-099: Positive classification requires both declared gates."""

    rows, stress_rows, artifact = completed_parts
    positive_aggregates = {field: deepcopy(artifact[field]) for field in mod.AGGREGATE_FIELDS}
    positive_aggregates["paired_efficiency_lcb"]["passed"] = True
    positive_aggregates["target_law_noninferiority"]["passed"] = True
    monkeypatch.setattr(mod, "derive_aggregates", lambda _rows: positive_aggregates)
    positive = mod.build_artifact(
        rows=rows,
        stress_rows=stress_rows,
        targets=mod.build_exact_targets(),
        run_date="20260830",
        duration_s=1.0,
        preconditions=mod.check_preconditions(),
    )
    assert positive["verdict_class"] == "positive"
    assert positive["honest_verdict"].startswith("complete:")


def test_scenario_sample_099_validation_covers_schema_and_terminal_failures(
    completed_parts: tuple[list[dict], list[dict], dict],
) -> None:
    """SCENARIO-SAMPLE-099: Every terminal boundary has a validation failure."""

    _rows, _stress, artifact = completed_parts
    missing = deepcopy(artifact)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["required_fields_missing"]

    changed = deepcopy(artifact)
    changed["field_principles"].pop("rows")
    changed["inference_substrate"] = "hardware"
    changed["verifier_is_oracle"] = True
    changed["honest_verdict"] = "null_without_prefix"
    changed["duration_s"] = -1.0
    errors = mod.validate_artifact(changed)
    assert "field_principles_mismatch" in errors
    assert "inference_substrate_mismatch" in errors
    assert "oracle_boundary_mismatch" in errors
    assert "honest_verdict_prefix_mismatch" in errors
    assert "duration_invalid" in errors

    blocked = mod._blocked_artifact(
        run_date="20260830",
        duration_s=0.1,
        preconditions=[{"check": "ram", "passed": False}],
    )
    blocked["rows"] = [{"unexpected": True}]
    assert "blocked_terminal_state_mismatch" in mod.validate_artifact(blocked)

    row_grid = deepcopy(artifact)
    row_grid["rows"] = row_grid["rows"][:-1]
    assert "row_grid_mismatch" in mod.validate_artifact(row_grid)

    target_hash = deepcopy(artifact)
    target_hash["exact_target_hashes"] = {}
    assert "exact_target_hash_mismatch" in mod.validate_artifact(target_hash)

    disabled = deepcopy(artifact)
    disabled_row = next(
        row for row in disabled["rows"] if row["arm"] == "temporal_exchange_zero_coupling"
    )
    disabled_row["trajectory_sha256"] = "sha256:different"
    disabled_row["row_sha256"] = mod.row_digest(disabled_row)
    assert "disabled_coupling_equivalence_mismatch" in mod.validate_artifact(disabled)

    stress = deepcopy(artifact)
    stress["stress_rows_separate"] = False
    assert "stress_row_mismatch" in mod.validate_artifact(stress)

    verdict = deepcopy(artifact)
    verdict["verdict_class"] = "positive" if artifact["verdict_class"] == "null" else "null"
    assert "verdict_class_mismatch" in mod.validate_artifact(verdict)


def test_req_sample_097_run_success_validation_failure_and_real_main(
    tmp_path: Path,
    completed_parts: tuple[list[dict], list[dict], dict],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-SAMPLE-097: Run and CLI write only validated terminal artifacts."""

    rows, stress_rows, _artifact = completed_parts
    targets = mod.build_exact_targets()
    monkeypatch.setattr(mod, "build_exact_targets", lambda: targets)
    monkeypatch.setattr(mod, "build_headline_rows", lambda _targets: rows)
    monkeypatch.setattr(mod, "build_stress_rows", lambda: stress_rows)
    output = tmp_path / "complete.json"
    completed = mod.run(output_path=output, run_date="20260830")
    assert completed["temporal_exchange_comparison_completed"] is True
    assert output.is_file()

    original_validate = mod.validate_artifact
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced"])
    with pytest.raises(mod.TemporalExchangeExperimentError, match="validation failed"):
        mod.run(output_path=tmp_path / "invalid.json", run_date="20260830")
    monkeypatch.setattr(mod, "validate_artifact", original_validate)

    monkeypatch.setattr(mod, "run", lambda **_kwargs: completed)
    assert mod.main(["--date", "20260830", "--output", str(tmp_path / "cli.json")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["temporal_exchange_comparison_completed"] is True

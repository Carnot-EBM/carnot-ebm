"""Boundary cost and sample-quality evidence tests.

Spec: REQ-RUSTPY-7202 and SCENARIO-RUSTPY-7202-*.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import ModuleType

import pytest

from carnot import experiment_7189_v633_rust_slice_parity as exp7189
from carnot import experiment_7201_v634_slice_pyo3 as exp7201
from carnot import experiment_7202_v634_slice_cost_quality as exp


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def compiled_tools() -> tuple[ModuleType, dict[str, object], Path]:
    """REQ-RUSTPY-7202-ROSTER uses the real compiled paths in focused tests."""

    extension = exp7201.build_pyo3_extension(REPO)
    binding, receipt = exp7201.load_compiled_binding(REPO, extension)
    bridge = exp7189.build_rust_bridge(REPO)
    return binding, receipt, bridge


def test_req_rustpy_7202_frozen_contract() -> None:
    """REQ-RUSTPY-7202-ROSTER and QUALITY freeze every budget before timing."""

    assert exp.SIZES == (32, 64, 128)
    assert exp.CARDINALITIES == (2, 4)
    assert exp.BATCH_SIZES == (1, 16, 64)
    assert exp.SEEDS == tuple(range(720200, 720210))
    assert exp.ARMS == ("python_control", "subprocess_rust", "persistent_pyo3")
    assert exp.PROPOSALS_PER_CHAIN == 160
    assert exp.WALL_BUDGET_S == 0.05
    assert exp.QUALITY_BURN_IN == 1024
    assert exp.QUALITY_RETAINED == 8192
    assert exp.QUALITY_MIN_RETAINED == 4096
    assert exp.QUALITY_MIN_ESS == 100.0
    assert exp.ENERGY_STANDARDIZED_TOLERANCE == 0.02
    assert exp.ESS_RATE_RATIO_CI_MIN == 0.90
    assert exp.NFR_01_SPEEDUP_TARGET == 10.0
    assert exp.PRIMARY_CELL == {"n": 64, "k": 4, "batch_size": 1, "protocol": "equal_work"}


def test_req_rustpy_7202_ess_refuses_constant_and_short_evidence() -> None:
    """REQ-RUSTPY-7202-LAW refuses bogus ESS and labels short timing traces."""

    constant = exp.ess_diagnostics([3.0] * 8192, latency_s=1.0)
    short = exp.ess_diagnostics([float(index % 2) for index in range(160)], latency_s=0.5)
    varied = exp.ess_diagnostics([float(index % 7) for index in range(8192)], latency_s=2.0)

    assert constant["ess"] is None
    assert constant["lag_correlations"] is None
    assert constant["evidence_sufficient"] is False
    assert short["ess"] is not None
    assert short["evidence_sufficient"] is False
    assert varied["ess"] is not None
    assert varied["ess_per_second"] == pytest.approx(varied["ess"] / 2.0)
    assert varied["evidence_sufficient"] is True


def test_req_rustpy_7202_statistics_are_prespecified_and_deterministic() -> None:
    """REQ-RUSTPY-7202-QUALITY fixes standardization and paired CI behavior."""

    left = [float(index) for index in range(100)]
    right = [value + 0.01 for value in left]
    assert exp.standardized_mean_difference(left, left) == 0.0
    assert 0.0 < exp.standardized_mean_difference(left, right) < 0.02
    with pytest.raises(ValueError, match="nonconstant"):
        exp.standardized_mean_difference([1.0, 1.0], [2.0, 2.0])
    interval = exp.paired_bootstrap_ci([1.1, 1.2, 1.3, 1.4], seed=7202, resamples=200)
    assert interval == exp.paired_bootstrap_ci([1.1, 1.2, 1.3, 1.4], seed=7202, resamples=200)
    assert interval[0] <= interval[1] <= interval[2]
    with pytest.raises(ValueError, match="positive"):
        exp.paired_bootstrap_ci([1.0, 0.0], seed=1, resamples=10)


def test_req_rustpy_7202_large_tape_creation_does_not_enumerate_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-RUSTPY-7202-ROSTER keeps the frozen n=128 case bounded."""

    def reject_enumeration(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("large tape creation must not enumerate the slice")

    monkeypatch.setattr(exp.exp7189, "make_replay_tape", reject_enumeration)
    states, tapes = exp._batch_inputs(128, 4, 2, 720200, 160)

    assert len(states) == len(tapes) == 2
    assert all(len(tape) == 160 for tape in tapes)
    assert all(sum(spin == 1 for spin in state) == 4 for state in states)


def test_req_rustpy_7202_exact_law_and_negative_controls_fail_closed() -> None:
    """REQ-RUSTPY-7202-LAW checks exact laws and both required controls."""

    laws = exp.run_exact_law_checks(sizes=(8, 12))
    controls = exp.run_negative_controls()

    assert {row["n"] for row in laws} == {8, 12}
    assert all(row["k"] == 2 and row["passed"] is True for row in laws)
    assert all(row["stationary_residual"] <= 1.0e-10 for row in laws)
    assert {row["control"] for row in controls} == {
        "constant_chain_ess",
        "biased_transition_target",
    }
    assert all(row["control_detected"] is True for row in controls)
    assert all(row["speed_eligible"] is False for row in controls)


def test_req_rustpy_7202_preconditions_bind_gate_and_quarantine() -> None:
    """REQ-RUSTPY-7202-PREFLIGHT checks quarantine before the upstream gate."""

    checks, hashes = exp.collect_preconditions(REPO)
    by_name = {row["check"]: row for row in checks}
    assert all(row["passed"] is True for row in checks)
    assert by_name["upstream_quarantine_flags"]["observed_value"]["quarantined"] is False
    assert by_name["upstream_pyo3_gate"]["observed_value"] == 1
    assert by_name["upstream_known_failed_value"]["observed_value"] is False
    assert by_name["same_milestone_gate_fields"]["observed_value"] == exp.EXPECTED_TASK_CONTRACT
    assert len(hashes) == len(exp.REQUIRED_SOURCE_PATHS)

    upstream = json.loads((REPO / exp.UPSTREAM_RESULT_PATH).read_text(encoding="utf-8"))
    upstream["quarantined"] = True
    observation = exp.upstream_quarantine_observation(upstream, manifest_match=False)
    assert observation["quarantined"] is True
    assert exp.gated_upstream_value(upstream, observation, "pyo3_slice_ready_score") == (
        "not_consumed_due_to_quarantine"
    )


def test_scenario_rustpy_7202_small_real_boundary_measurement(
    compiled_tools: tuple[ModuleType, dict[str, object], Path],
) -> None:
    """SCENARIO-RUSTPY-7202-MATCHED-BOUNDARIES executes every real arm."""

    binding, _, bridge = compiled_tools
    cold, rows = exp.run_throughput_benchmarks(
        binding,
        bridge,
        sizes=(32,),
        cardinalities=(2,),
        batch_sizes=(1,),
        seeds=(720200,),
        proposals=16,
        wall_budget_s=0.003,
    )

    assert len(cold) == 3
    assert len(rows) == 6
    assert {row["arm"] for row in rows} == set(exp.ARMS)
    assert {row["protocol"] for row in rows} == {"equal_work", "equal_wall"}
    assert all(row["sector_violations"] == 0 for row in rows)
    assert all(row["data_transfer_and_synchronization_charged"] is True for row in rows)
    assert all(
        row["deadline_overshoot_s"] is not None for row in rows if row["protocol"] == "equal_wall"
    )
    assert all(row["parity_passed"] is True for row in rows if row["protocol"] == "equal_work")


def test_req_rustpy_7202_small_quality_panel_retains_insufficient_evidence(
    compiled_tools: tuple[ModuleType, dict[str, object], Path],
) -> None:
    """REQ-RUSTPY-7202-QUALITY keeps short-panel metrics but fails its gate."""

    binding, _, bridge = compiled_tools
    rows = exp.run_quality_panels(
        binding,
        bridge,
        sizes=(32,),
        cardinalities=(2,),
        seeds=(720200,),
        burn_in=8,
        retained=64,
    )

    assert len(rows) == 3
    assert all(row["retained"] == 64 for row in rows)
    assert all(row["sector_violations"] == 0 for row in rows)
    assert all(row["quality_row_sufficient"] is False for row in rows)
    rust_rows = {row["arm"]: row for row in rows}
    assert rust_rows["persistent_pyo3"]["energy_mean"] == pytest.approx(
        rust_rows["subprocess_rust"]["energy_mean"]
    )


def test_req_rustpy_7202_summary_keeps_completion_separate_from_value() -> None:
    """REQ-RUSTPY-7202-PRIMARY and NFR keep null completion honest."""

    rows = []
    for seed, subprocess_s, persistent_s, python_s in (
        (720200, 0.020, 0.010, 0.030),
        (720201, 0.024, 0.011, 0.033),
        (720202, 0.022, 0.010, 0.032),
    ):
        for arm, latency in (
            ("subprocess_rust", subprocess_s),
            ("persistent_pyo3", persistent_s),
            ("python_control", python_s),
        ):
            rows.append(
                {
                    "n": 64,
                    "k": 4,
                    "batch_size": 1,
                    "seed": seed,
                    "arm": arm,
                    "protocol": "equal_work",
                    "latency_s": latency,
                    "parity_passed": True,
                    "sector_violations": 0,
                }
            )
    summary = exp.summarize_primary_gate(rows, sample_quality_sufficient=False)

    assert summary["latency_speedup_over_subprocess_ci95"]["lower"] > 1.0
    assert summary["python_speedup_ci95"]["lower"] > 1.0
    assert summary["nfr_01_10x_met"] is False
    assert summary["boundary_value_score"] == 0


def test_req_rustpy_7202_blocked_artifact_names_exact_external_gate(tmp_path: Path) -> None:
    """REQ-RUSTPY-7202-PREFLIGHT publishes a terminal diagnosed block."""

    failed = [
        {
            "check": "upstream_pyo3_gate",
            "upstream": str(exp.UPSTREAM_RESULT_PATH),
            "field": "pyo3_slice_ready_score",
            "expected_value": 1,
            "observed_value": 0,
            "passed": False,
        }
    ]
    artifact = exp.build_artifact(root=REPO, preconditions=failed, source_hashes={})
    output = tmp_path / "blocked.json"
    receipt = exp.atomic_write(output, artifact)

    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["observed_value"] == 0
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert exp.validate_artifact(artifact) == []
    assert receipt["atomic_replace"] is True
    assert exp.main(["--validate", str(output)]) == 0
    assert exp.main(["--date", "19000101"]) == 2


def test_scenario_rustpy_7202_terminal_artifact_recomputes() -> None:
    """SCENARIO-RUSTPY-7202-QUALITY-GATES-SPEED validates durable evidence."""

    payload = json.loads((REPO / exp.RESULT_PATH).read_text(encoding="utf-8"))
    assert exp.validate_artifact(payload, root=REPO) == []
    assert payload["status"] == "complete"
    assert payload["slice_comparison_complete_score"] == 1
    assert payload["MODEL_SPECS"] == []
    assert payload["model_invoked"] is False
    assert len(payload["throughput_rows"]) == 1080
    assert len(payload["distribution_rows"]) == 2
    assert len(payload["quality_rows"]) == 180
    assert payload["acceptance_gate_boundary"] == exp.PRIMARY_CELL
    assert payload["nfr_01_10x_met"] == (
        payload["primary_gate"]["latency_speedup_over_subprocess_ci95"]["lower"] >= 10.0
        and payload["sample_quality_sufficient"]
    )


def test_req_rustpy_7202_validator_rejects_deleted_or_promoted_evidence() -> None:
    """REQ-RUSTPY-7202-ARTIFACT rejects row deletion and inflated claims."""

    payload = json.loads((REPO / exp.RESULT_PATH).read_text(encoding="utf-8"))
    attacks = [
        (lambda item: item["throughput_rows"].pop(), "throughput_rows_invalid"),
        (lambda item: item.update(nfr_01_10x_met=True), "nfr_01_gate_invalid"),
        (
            lambda item: item["upstream_performance_null"].update(promoted=True),
            "upstream_failed_value_promoted",
        ),
        (lambda item: item.update(boundary_value_score=1), "boundary_value_score_invalid"),
    ]
    for mutate, expected in attacks:
        changed = copy.deepcopy(payload)
        mutate(changed)
        changed["rows"] = exp.combined_rows(changed)
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed)

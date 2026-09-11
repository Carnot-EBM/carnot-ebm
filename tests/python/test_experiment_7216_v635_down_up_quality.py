"""Tests for the V635 matched-cost down-up quality study.

Spec: REQ-ISING-7216 and SCENARIO-ISING-7216-*.
"""

from __future__ import annotations

import copy
import gzip
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7187_v633_slice_sampler as slices
from carnot import experiment_7216_v635_down_up_quality as exp


REPO = Path(__file__).resolve().parents[2]


def test_req_ising_7216_freezes_roster_budgets_and_claims() -> None:
    """REQ-ISING-7216 fixes cells, samples, costs, seeds, and the host scope."""

    assert exp.CELLS == ((32, 1.0), (32, 2.0), (16, 1.0))
    assert exp.PRIMARY_CELL == {"n": 32, "k": 2, "beta": 1.0}
    assert exp.GRAPH_SEEDS == tuple(range(7216001, 7216011))
    assert exp.CHAINS == (0, 1, 2, 3)
    assert exp.ARMS == ("down_up", "pair_swap_metropolis")
    assert exp.ENERGY_EVALUATION_BUDGET == 100_000
    assert exp.WALL_BUDGET_S == 2.0
    assert exp.QUALITY_BURN_IN == 4096
    assert exp.QUALITY_RETAINED == 16384
    assert exp.MEASUREMENT_CAP_S == 1800.0
    assert exp.BOOTSTRAP_SEED == 7216002
    assert exp.BOOTSTRAP_RESAMPLES == 10_000
    assert exp.MODEL_SPECS == []


def test_scenario_ising_7216_law_trace_uses_independent_authority() -> None:
    """SCENARIO-ISING-7216-LAW-TRACE enumerates exact moments before sampling."""

    for n in (16, 32):
        instance = slices.make_frustrated_instance(n, 7216001)
        authority = exp.enumerate_exact_authority(instance, beta=1.0)
        expected_count = math.comb(n, 2)

        assert len(authority.states) == expected_count
        assert sum(authority.probabilities) == pytest.approx(1.0)
        assert authority.normalization_error <= exp.EXACT_TOLERANCE
        assert authority.energy_parity_error <= exp.EXACT_TOLERANCE
        assert authority.energy_mean == pytest.approx(
            sum(
                probability * energy
                for probability, energy in zip(
                    authority.probabilities, authority.energies, strict=True
                )
            )
        )
        assert len(authority.occupancy_means) == n
        assert len(authority.occupancy_variances) == n
        assert sum(authority.occupancy_means) == pytest.approx(2.0)
        assert authority.probes == (0, n // 3, 2 * n // 3)
        assert all(authority.occupancy_variances[site] > 0.0 for site in authority.probes)

        evaluator = exp.TargetEnergyEvaluator(instance, authority)
        observed = evaluator.evaluate_indices((0, expected_count - 1))
        assert observed == pytest.approx(
            (
                slices.ising_energy(instance, exp.down_up.subset_to_spins(n, authority.states[0])),
                slices.ising_energy(instance, exp.down_up.subset_to_spins(n, authority.states[-1])),
            )
        )
        assert evaluator.evaluations == 2


def test_scenario_ising_7216_matched_work_charges_conditional_cost() -> None:
    """SCENARIO-ISING-7216-MATCHED-BUDGETS charges exact energy work."""

    instance = slices.make_frustrated_instance(16, 7216002)
    authority = exp.enumerate_exact_authority(instance, beta=1.0)
    initial = exp.overdispersed_initial_indices(authority)[0]
    budget = 100

    down = exp.sample_chain(
        instance,
        authority,
        arm="down_up",
        rng_seed=11,
        initial_index=initial,
        energy_budget=budget,
    )
    pair = exp.sample_chain(
        instance,
        authority,
        arm="pair_swap_metropolis",
        rng_seed=12,
        initial_index=initial,
        energy_budget=budget,
    )

    assert down.energy_evaluations == pair.energy_evaluations == budget
    assert down.transitions < pair.transitions
    assert down.normalizations == down.transitions
    assert pair.normalizations == 0
    assert down.unfinished_conditional_energy_evaluations > 0
    assert len(down.trace_indices) == down.transitions
    assert len(pair.trace_indices) == pair.transitions
    assert all(len(authority.states[index]) == 2 for index in down.trace_indices)
    assert all(len(authority.states[index]) == 2 for index in pair.trace_indices)
    assert exp.derive_stream_seed(7216002, 16, 1.0, 0, "down_up", "quality") != (
        exp.derive_stream_seed(7216002, 16, 1.0, 0, "pair_swap_metropolis", "quality")
    )

    with pytest.raises(ValueError, match="exactly one stopping budget"):
        exp.sample_chain(
            instance,
            authority,
            arm="down_up",
            rng_seed=1,
            initial_index=initial,
        )
    with pytest.raises(ValueError, match="unknown arm"):
        exp.sample_chain(
            instance,
            authority,
            arm="bad",
            rng_seed=1,
            initial_index=initial,
            transition_limit=1,
        )


def test_scenario_ising_7216_quality_rejects_constant_nondegenerate_trace() -> None:
    """SCENARIO-ISING-7216-QUALITY never turns a stuck trace into infinite ESS."""

    constant = exp.ess_diagnostics([1.0] * 32, target_variance=0.25, latency_s=1.0)
    degenerate = exp.ess_diagnostics([1.0] * 32, target_variance=0.0, latency_s=1.0)
    moving = exp.ess_diagnostics([0.0, 1.0] * 32, target_variance=0.25, latency_s=2.0)

    assert constant["constant_observed"] is True
    assert constant["ess"] is None
    assert constant["qualified"] is False
    assert degenerate["structurally_degenerate"] is True
    assert degenerate["qualified"] is True
    assert moving["ess"] == pytest.approx(64.0)
    assert moving["ess_per_second"] == pytest.approx(32.0)
    assert moving["monte_carlo_standard_error"] is not None

    chains = [[0.0, 1.0] * 16 for _ in range(4)]
    assert exp.split_rhat(chains) <= 1.0
    assert exp.split_rhat([[1.0] * 32 for _ in range(4)]) is None
    with pytest.raises(ValueError, match="four chains"):
        exp.split_rhat(chains[:3])


def test_req_ising_7216_quality_row_scores_energy_occupancy_and_tv() -> None:
    """REQ-ISING-7216 scores every retained transition against finite authority."""

    instance = slices.make_frustrated_instance(16, 7216003)
    authority = exp.enumerate_exact_authority(instance, beta=1.0)
    initial = exp.overdispersed_initial_indices(authority)[0]
    run = exp.sample_chain(
        instance,
        authority,
        arm="down_up",
        rng_seed=17,
        initial_index=initial,
        transition_limit=96,
    )
    row = exp.quality_row_from_run(
        instance,
        authority,
        run,
        graph_seed=7216003,
        chain_id=0,
        arm="down_up",
        burn_in=32,
        retained=64,
        arm_order=("down_up", "pair_swap_metropolis"),
        order_index=0,
    )

    assert row["retained"] == 64
    assert row["sector_violations"] == 0
    assert row["self_transitions"] >= 0
    assert set(row["probe_diagnostics"]) == {"energy", "occupancy_0", "occupancy_5", "occupancy_10"}
    assert row["empirical_total_variation"] >= 0.0
    assert row["total_variation_mcse"] >= 0.0
    assert row["exact_authority_sha256"] == authority.authority_sha256
    assert row["row_sha256"] == exp.sha256_json(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )


def test_scenario_ising_7216_gate_qualifies_before_bootstrap() -> None:
    """SCENARIO-ISING-7216-GATE withholds throughput for a biased comparator."""

    summaries = []
    for seed in exp.GRAPH_SEEDS:
        for arm, rate in (("down_up", 4.0), ("pair_swap_metropolis", 2.0)):
            summaries.append(
                {
                    "n": 32,
                    "k": 2,
                    "beta": 1.0,
                    "graph_seed": seed,
                    "arm": arm,
                    "complete": True,
                    "zero_sector_violations": True,
                    "all_chain_ess_passed": True,
                    "all_split_rhat_passed": True,
                    "exact_mean_tolerances_passed": True,
                    "minimum_probe_ess_per_second": rate,
                }
            )
    exact = [
        {"n": 32, "k": 2, "beta": 1.0, "graph_seed": seed, "finite_law_passed": True}
        for seed in exp.GRAPH_SEEDS
    ]

    passed = exp.evaluate_primary_gate(summaries, exact, panel_complete=True)
    assert passed["comparator_quality_sufficient"] is True
    assert passed["ess_per_second_ratio_ci95"]["lower"] > 1.0
    assert passed["passed"] is True

    biased = copy.deepcopy(summaries)
    biased[0]["exact_mean_tolerances_passed"] = False
    failed = exp.evaluate_primary_gate(biased, exact, panel_complete=True)
    assert failed["comparator_quality_sufficient"] is False
    assert failed["ess_per_second_ratio_ci95"] is None
    assert failed["passed"] is False
    assert failed["abstention_reason"] == "insufficient_comparator_quality"


def test_scenario_ising_7216_preflight_is_fail_closed() -> None:
    """SCENARIO-ISING-7216-PREFLIGHT unwraps only real wrappers after quarantine."""

    wrapped = {"value": 1, "principle": "Read the exact gate."}
    arbitrary = {"value": 1, "metadata": "not a wrapper"}
    clean = exp.upstream_quarantine_observation({}, manifest_match=False)
    dirty = exp.upstream_quarantine_observation({"flagged_adversarial": True}, manifest_match=False)

    assert exp.unwrap_principled_value(wrapped) == 1
    assert exp.unwrap_principled_value(arbitrary) is arbitrary
    assert exp.gated_upstream_value({"gate": wrapped}, clean, "gate") == 1
    assert exp.gated_upstream_value({"gate": arbitrary}, clean, "gate") is arbitrary
    assert (
        exp.gated_upstream_value({"gate": wrapped}, dirty, "gate")
        == "not_consumed_due_to_quarantine"
    )
    assert exp.upstream_quarantine_observation({}, manifest_match=True)["quarantined"] is True

    checks, hashes = exp.collect_preconditions(REPO)
    by_name = {row["check"]: row for row in checks}
    assert all(row["passed"] is True for row in checks)
    assert by_name["driving_capability_spec"]["observed_value"]["req_present"] is True
    assert by_name["roadmap_task_contract"]["observed_value"] == exp.EXPECTED_TASK_CONTRACT
    assert by_name["upstream_quarantine"]["observed_value"]["quarantined"] is False
    assert by_name["upstream_authentication"]["observed_value"]["producer_valid"] is True
    assert by_name["upstream_gate"]["observed_value"] == 1
    assert by_name["v634_nfr_null"]["observed_value"]["promoted"] is False
    assert set(hashes) == {str(path) for path in exp.REQUIRED_SOURCE_PATHS}


def test_scenario_ising_7216_blocked_artifact_and_cli(tmp_path: Path) -> None:
    """SCENARIO-ISING-7216-ARTIFACT retains a terminal external diagnosis."""

    failed = {
        "check": "upstream_gate",
        "upstream": str(exp.UPSTREAM_PATH),
        "field": "down_up_kernel_ready_score",
        "expected_value": 1,
        "observed_value": 0,
        "passed": False,
    }
    artifact = exp.build_artifact(
        root=REPO,
        output=tmp_path / "unused.json",
        trace_output=tmp_path / "unused.jsonl.gz",
        preconditions=[failed],
        source_hashes={},
    )
    output = tmp_path / "blocked.json"
    exp.atomic_write(output, artifact)

    assert artifact["status"] == "blocked_external_precondition"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["failed_check"] == "upstream_gate"
    assert exp.validate_artifact(artifact) == []
    assert exp.main(["--validate", str(output)]) == 0
    assert exp.main(["--validate", str(tmp_path / "missing.json")]) == 2
    assert exp.main(["--date", "19000101"]) == 2


def test_req_ising_7216_trace_archive_is_compressed_and_bound(tmp_path: Path) -> None:
    """REQ-ISING-7216 persists complete repeated-state traces under checkpoints."""

    path = tmp_path / "results" / "checkpoints" / "traces.jsonl.gz"
    writer = exp.TraceArchiveWriter(path)
    writer.open()
    writer.write(
        {
            "unit_id": "trace:test",
            "rng_seed": 7,
            "trace_indices": [1, 1, 2, 2],
        }
    )
    receipt = writer.close()

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        decoded = json.loads(handle.readline())
    assert decoded["trace_indices"] == [1, 1, 2, 2]
    assert receipt["compressed"] is True
    assert receipt["record_count"] == 1
    assert receipt["sha256"] == exp.sha256_file(path)


def test_req_ising_7216_small_end_to_end_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ISING-7216 runs law, kernels, scoring, gate evaluation, and serialization."""

    monkeypatch.setattr(exp, "CELLS", ((16, 1.0),))
    monkeypatch.setattr(exp, "GRAPH_SEEDS", (7216001,))
    monkeypatch.setattr(exp, "ENERGY_EVALUATION_BUDGET", 40)
    monkeypatch.setattr(exp, "WALL_BUDGET_S", 0.001)
    monkeypatch.setattr(exp, "QUALITY_BURN_IN", 4)
    monkeypatch.setattr(exp, "QUALITY_RETAINED", 16)
    monkeypatch.setattr(exp, "MEASUREMENT_CAP_S", 30.0)
    output = tmp_path / "results" / "small.json"
    trace = tmp_path / "results" / "checkpoints" / "small-traces.jsonl.gz"
    passed = {
        "check": "fixture",
        "upstream": "test",
        "field": "fixture",
        "expected_value": True,
        "observed_value": True,
        "passed": True,
    }

    artifact = exp.build_artifact(
        root=tmp_path,
        output=output,
        trace_output=trace,
        preconditions=[passed],
        source_hashes={},
    )
    exp.atomic_write(output, artifact)

    assert artifact["status"] == "complete"
    assert artifact["down_up_comparison_complete_score"] == 1
    assert artifact["down_up_value_score"] == 0
    assert len(artifact["exact_authority_rows"]) == 1
    assert len(artifact["matched_budget_rows"]) == 16
    assert len(artifact["quality_rows"]) == 8
    assert len(artifact["quality_summary_rows"]) == 2
    assert artifact["trace_archive"]["record_count"] == 24
    assert exp.validate_artifact(artifact) == []
    assert exp.main(["--validate", str(output)]) == 0


def test_scenario_ising_7216_terminal_artifact_recomputes() -> None:
    """SCENARIO-ISING-7216-ARTIFACT validates the durable full-roster evidence."""

    payload = json.loads((REPO / exp.RESULT_PATH).read_text(encoding="utf-8"))
    assert exp.validate_artifact(payload, root=REPO) == []
    assert payload["down_up_comparison_complete_score"] == 1
    assert payload["verdict_class"] in {"null", "circular_positive"}
    assert payload["MODEL_SPECS"] == []
    assert payload["model_invoked"] is False
    assert payload["nfr_01_10x_met"] is False
    assert payload["paper_replication_claimed"] is False
    assert payload["rust_10x_speed_claimed"] is False
    assert payload["tsu_execution_claimed"] is False
    assert payload["sparse_sk_theorem_claimed"] is False
    assert payload["hardware_power_savings_claimed"] is False


def test_req_ising_7216_validator_rejects_tampering() -> None:
    """REQ-ISING-7216 binds rows, gates, completion, and limited claims."""

    payload = json.loads((REPO / exp.RESULT_PATH).read_text(encoding="utf-8"))
    broken = copy.deepcopy(payload)
    broken["matched_budget_rows"][0]["energy_evaluations"] = -1
    broken["rows"][len(broken["exact_authority_rows"])]["energy_evaluations"] = -1
    broken["paper_replication_claimed"] = True
    broken["down_up_value_score"] = 1 - broken["down_up_value_score"]
    broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
    errors = exp.validate_artifact(broken)

    assert "matched_budget_rows_invalid" in errors
    assert "rows_invalid" in errors
    assert "claim_limits_invalid" in errors
    assert "value_score_invalid" in errors

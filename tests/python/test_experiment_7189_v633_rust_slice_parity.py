"""Conformance tests for REQ-SAMPLER-7189 and its named scenarios."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_7189_v633_rust_slice_parity as exp
from carnot.experiment_7187_v633_slice_sampler import (
    BENCHMARK_SEEDS,
    SliceInstance,
    enumerate_slice,
    independent_exact_law,
    make_frustrated_instance,
)


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def rust_bridge() -> Path:
    """REQ-SAMPLER-7189 requires the tests to execute a compiled Rust path."""

    return exp.build_rust_bridge(REPO)


@pytest.fixture(scope="session")
def ready_artifact(rust_bridge: Path) -> dict:
    """Build the full frozen roster once for artifact and coverage checks."""

    return exp.build_artifact(root=REPO, run_date=exp.RUN_DATE, bridge_path=rust_bridge)


def test_req_sampler_7189_preflight_binds_source_tools_and_gate(tmp_path: Path) -> None:
    """REQ-SAMPLER-7189-PREFLIGHT checks bytes, tools, paths, and Exp7187."""

    checks, hashes = exp.collect_preconditions(
        REPO,
        result_path=tmp_path / "result.json",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert all(row["passed"] is True for row in checks)
    by_name = {row["check"]: row for row in checks}
    assert by_name["driving_capability_spec"]["observed_value"] == {
        "exists": True,
        "req_present": True,
    }
    assert by_name["upstream_slice_sampler_gate"]["expected_value"] == 1
    assert by_name["upstream_slice_sampler_gate"]["observed_value"] == 1
    assert by_name["same_milestone_gate_fields"]["expected_value"] == (
        by_name["same_milestone_gate_fields"]["observed_value"]
    )
    assert len(hashes) == len(exp.REQUIRED_SOURCE_PATHS)
    assert all(exp.HASH_PATTERN.fullmatch(value) for value in hashes.values())


def test_scenario_sampler_7189_shared_tape_matches_compiled_rust(rust_bridge: Path) -> None:
    """SCENARIO-SAMPLER-7189-REPLAY compares every deterministic transition."""

    instance = make_frustrated_instance(8, 718701)
    initial = enumerate_slice(8, 2)[3]
    tape = exp.make_replay_tape(8, 2, seed=718900, steps=32)
    python_result = exp.python_replay(instance, 2, 2.0, initial, tape)
    rust_result, receipt = exp.rust_request(
        rust_bridge,
        exp.replay_request(instance, 2, 2.0, initial, tape),
    )
    rows = exp.compare_replays("test", tape, python_result, rust_result)
    assert receipt["returncode"] == 0
    assert len(rows) == len(tape)
    assert all(row["passed"] is True for row in rows)
    assert all(row["python_cardinality"] == row["rust_cardinality"] == 2 for row in rows)
    assert max(row["delta_energy_error"] for row in rows) <= exp.TOLERANCE


def test_req_sampler_7189_replay_validation_and_singletons(rust_bridge: Path) -> None:
    """REQ-SAMPLER-7189-KERNEL rejects bad tapes and preserves singleton slices."""

    instance = make_frustrated_instance(8, 718702)
    with pytest.raises(ValueError, match="state length"):
        exp.python_replay(instance, 2, 1.0, (1, -1), [])
    with pytest.raises(ValueError, match="cardinality"):
        exp.python_replay(instance, 2, 1.0, (-1,) * 8, [])
    with pytest.raises(ValueError, match="uniform"):
        exp.python_replay(
            instance,
            2,
            1.0,
            enumerate_slice(8, 2)[0],
            [{"positive_index": 0, "negative_index": 0, "uniform": 1.0}],
        )
    with pytest.raises(ValueError, match="proposal index"):
        exp.python_replay(
            instance,
            2,
            1.0,
            enumerate_slice(8, 2)[0],
            [{"positive_index": 9, "negative_index": 0, "uniform": 0.5}],
        )
    for k in (0, 8):
        initial = enumerate_slice(8, k)[0]
        tape = exp.make_replay_tape(8, k, seed=2, steps=2)
        py = exp.python_replay(instance, k, 1.0, initial, tape)
        rust, _ = exp.rust_request(rust_bridge, exp.replay_request(instance, k, 1.0, initial, tape))
        assert py["final_state"] == rust["final_state"] == list(initial)
        assert all(step["accepted"] is True for step in py["steps"])


def test_scenario_sampler_7189_independent_distributions_match_exact_law(
    rust_bridge: Path,
) -> None:
    """SCENARIO-SAMPLER-7189-DISTRIBUTION checks separate RNG implementations."""

    instance = make_frustrated_instance(8, 718701)
    law = independent_exact_law(instance, 2, 2.0)
    rows = exp.run_distribution_checks(
        rust_bridge,
        instance=instance,
        law=law,
        seeds=(718900, 718901),
        burn_in=500,
        retained=4_000,
        tv_limit=0.25,
        energy_mean_limit=0.5,
    )
    assert len(rows) == 4
    assert {row["language"] for row in rows} == {"python", "rust"}
    assert all(row["cardinality_valid"] is True for row in rows)
    assert all(row["passed"] is True for row in rows)


def test_scenario_sampler_7189_e2e_serialized_parameters(rust_bridge: Path, tmp_path: Path) -> None:
    """SCENARIO-SAMPLER-7189-E2E loads one serialized energy in both paths."""

    receipt = exp.run_e2e_receipt(rust_bridge, tmp_path)
    assert receipt["passed"] is True
    assert receipt["rust_returncode"] == 0
    assert receipt["energy_error"] <= exp.TOLERANCE
    assert receipt["python_cardinality"] == receipt["rust_cardinality"] == 2
    assert receipt["parameter_sha256"].startswith("sha256:")
    assert receipt["rust_output_sha256"].startswith("sha256:")


def test_req_sampler_7189_throughput_rows_include_bridge_cost(rust_bridge: Path) -> None:
    """SCENARIO-SAMPLER-7189-THROUGHPUT retains raw and p50/p95 costs."""

    raw, aggregates = exp.run_throughput_benchmarks(
        rust_bridge,
        sizes=(32,),
        cardinalities=(2,),
        seeds=BENCHMARK_SEEDS[:2],
        energy_budget=40,
        wall_time_budget_s=0.002,
    )
    assert len(raw) == 8
    assert len(aggregates) == 4
    assert all(row["latency_s"] > 0.0 and row["work"] > 0 for row in raw)
    assert all(row["includes_setup_and_bridge_overhead"] is True for row in raw)
    assert all(row["latency_p95_s"] >= row["latency_p50_s"] > 0.0 for row in aggregates)


def test_scenario_sampler_7189_artifact_recomputes_readiness(ready_artifact: dict) -> None:
    """SCENARIO-SAMPLER-7189-ARTIFACT rejects evidence deletion and score edits."""

    assert exp.validate_artifact(ready_artifact, root=REPO) == []
    assert ready_artifact["rust_slice_parity_score"] == 1
    assert ready_artifact["compiled_rust_execution"] is True
    assert len(ready_artifact["distribution_rows"]) == 20
    assert len(ready_artifact["throughput_raw_rows"]) == 160
    assert len(ready_artifact["throughput_rows"]) == 16
    assert len(ready_artifact["rows"]) == (
        len(ready_artifact["cross_language_rows"])
        + len(ready_artifact["distribution_rows"])
        + len(ready_artifact["throughput_raw_rows"])
        + len(ready_artifact["throughput_rows"])
    )
    assert ready_artifact["nfr_01_10x_met"] in {True, False}
    assert ready_artifact["performance_verdict_class"] in {"positive", "null"}

    attacks = [
        (lambda item: item["cross_language_rows"].pop(), "cross_language_rows_incomplete"),
        (lambda item: item.update(rust_slice_parity_score=0), "readiness_invalid"),
        (lambda item: item.update(compiled_rust_execution=False), "compiled_execution_missing"),
    ]
    for mutate, expected in attacks:
        changed = copy.deepcopy(ready_artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed, root=REPO)


def test_req_sampler_7189_blocked_artifact_and_cli_validation(tmp_path: Path) -> None:
    """REQ-SAMPLER-7189-ARTIFACT retains exact external failures and validates bytes."""

    checks = [{
        "check": "upstream_slice_sampler_gate",
        "upstream": str(exp.UPSTREAM_RESULT_PATH),
        "field": "slice_sampler_ready_score",
        "expected_value": 1,
        "observed_value": 0,
        "passed": False,
    }]
    artifact = exp.build_artifact(root=REPO, preconditions=checks, source_hashes={})
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["observed_value"] == 0
    assert exp.validate_artifact(artifact, root=REPO) == []

    path = tmp_path / "artifact.json"
    exp.atomic_write(path, artifact)
    assert exp.main(["--validate", str(path)]) == 0
    path.write_text("not-json", encoding="utf-8")
    assert exp.main(["--validate", str(path)]) == 2
    assert exp.main(["--date", "19000101"]) == 2


def test_req_sampler_7189_canonical_json_rejects_nonfinite() -> None:
    """REQ-SAMPLER-7189-ARTIFACT never emits non-standard JSON numbers."""

    with pytest.raises(ValueError, match="nonfinite"):
        exp.canonical_json({"bad": float("nan")})
    with pytest.raises(ValueError, match="positive"):
        exp.run_distribution_checks(
            Path("missing"),
            instance=make_frustrated_instance(8, 718701),
            law=independent_exact_law(make_frustrated_instance(8, 718701), 2, 2.0),
            seeds=(1,),
            burn_in=0,
            retained=0,
            tv_limit=0.2,
            energy_mean_limit=0.2,
        )


def test_req_sampler_7189_e2e_fixture_has_known_energy() -> None:
    """REQ-SAMPLER-7189-E2E freezes the serialized model's analytic energy."""

    instance = SliceInstance(
        n=4,
        seed=7189,
        edges=((0, 1, 1.0), (0, 2, 1.0), (1, 2, -1.0)),
        fields=(0.1, -0.2, 0.3, -0.2),
    )
    assert exp.python_energy(instance, (1, 1, -1, -1)) == pytest.approx(-0.8)
    payload = exp.instance_payload(instance, cardinality=2, beta=2.0)
    assert json.loads(exp.canonical_json(payload))["cardinality"] == 2

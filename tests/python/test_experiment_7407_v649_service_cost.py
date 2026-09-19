"""Tests for the V649 complete decision-service cost capture.

Spec refs: REQ-REPORT-7407 and SCENARIO-REPORT-7407-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_7407_v649_service_cost as experiment
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-FIXTURE
def test_authenticates_fixed_fixture_and_keeps_optional_trial_optional(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    checks, hashes, context = experiment.collect_preconditions(
        ROOT,
        paths,
        candidate_paths=[tmp_path / "no-gatemate-receipt.md"],
    )

    mandatory = [row for row in checks if row["category"] == "mandatory_precondition"]
    assert mandatory and all(row["passed"] for row in mandatory)
    assert context["fixed_service"]["eligible"] is True
    assert context["fixed_service"]["source_experiment"] == "Exp7385"
    assert context["fixed_service"]["source_verdict_class"] == "null"
    assert context["fixed_service"]["arm"] == "natural_prevalence_bernoulli_gibbs"
    assert context["fixed_service"]["seed"] == 7382001
    assert context["fixed_service"]["source_text_count"] == 128
    assert context["optional_adaptive_service"] == {
        "source_experiment": "Exp7399",
        "available": False,
        "eligible": False,
        "disposition": "optional_absent_does_not_block_fixed_service",
    }
    assert hashes[experiment.EXP7385_PATH.as_posix()] == experiment.EXPECTED_EXP7385_SHA256
    assert hashes[experiment.CHECKPOINT_PATH.as_posix()] == experiment.EXPECTED_CHECKPOINT_SHA256
    assert hashes[experiment.CORPUS_PATH.as_posix()] == experiment.EXPECTED_CORPUS_SHA256
    assert context["hardware_history"]["original_transcripts_authenticated"] is True
    assert context["gatemate_changed_state"]["exists"] is False


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-PARITY
def test_scalar_vector_service_and_threshold_boundaries_agree() -> None:
    fixture = experiment.load_fixed_fixture(ROOT)
    texts = fixture["texts"][:32]
    state = fixture["state"]

    scalar = experiment.run_service(texts, state, arm="scalar_numpy")
    vector = experiment.run_service(texts, state, arm="vectorized_numpy")
    parity = experiment.compare_service_outputs(scalar, vector)
    boundaries = experiment.threshold_boundary_rows(state["policy"])

    assert parity["passed"] is True
    assert parity["energy_max_abs_delta"] <= experiment.FLOAT64_TOLERANCE
    assert parity["probability_max_abs_delta"] <= experiment.FLOAT64_TOLERANCE
    assert parity["actions_identical"] is True
    assert scalar["return_payload"] == vector["return_payload"]
    assert scalar["measured_bytes"]["raw_text_utf8"] > 0
    assert scalar["measured_bytes"]["journal_json_utf8"] > 0
    assert scalar["measured_bytes"]["complete_return_json_utf8"] > 0
    assert len(boundaries) == 6
    assert all(row["scalar_action"] == row["vectorized_action"] for row in boundaries)
    assert {row["threshold"] for row in boundaries} == {"accept", "reject"}

    with pytest.raises(ValueError, match="arm"):
        experiment.run_service(texts, state, arm="gpu")


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-TIMING
def test_paired_measurement_retains_each_stage_order_and_parity() -> None:
    fixture = experiment.load_fixed_fixture(ROOT)
    rows = experiment.measure_service_cost(
        fixture["texts"],
        fixture["state"],
        batch_sizes=(1, 32, 128),
        blocks=2,
        seed=experiment.RANDOM_SEED,
    )

    assert len(rows) == 12
    for batch_size in (1, 32, 128):
        selected = [row for row in rows if row["batch_size"] == batch_size]
        orders = [
            {row["pair_order"] for row in selected if row["block"] == block}.pop()
            for block in (0, 1)
        ]
        assert orders in (
            ["scalar_numpy_then_vectorized_numpy", "vectorized_numpy_then_scalar_numpy"],
            ["vectorized_numpy_then_scalar_numpy", "scalar_numpy_then_vectorized_numpy"],
        )
        assert {row["block"] for row in selected} == {0, 1}
        assert {row["arm"] for row in selected} == {"scalar_numpy", "vectorized_numpy"}
        assert all(row["parity_passed"] for row in selected)
        assert all(row["complete_return_s"] > 0 for row in selected)
        assert all(set(row["stage_durations_s"]) == set(experiment.STAGE_NAMES) for row in selected)
        assert all(row["measured_bytes"]["complete_return_json_utf8"] > 0 for row in selected)


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-TIMING
def test_paired_ratio_intervals_use_full_service_not_kernel_alone() -> None:
    rows: list[dict[str, Any]] = []
    for block in range(30):
        for arm, scoring, complete in (
            ("scalar_numpy", 4.0, 10.0),
            ("vectorized_numpy", 1.0, 10.5),
        ):
            rows.append(
                experiment.synthetic_service_row(
                    batch_size=32,
                    block=block,
                    arm=arm,
                    scoring_s=scoring,
                    complete_s=complete,
                )
            )
    summary = experiment.summarize_cost_rows(rows, draws=500, seed=experiment.RANDOM_SEED)
    batch = summary["batch_rows"][0]

    assert batch["scoring_ratio_scalar_over_vector"]["estimate"] == pytest.approx(4.0)
    assert batch["complete_ratio_scalar_over_vector"]["estimate"] == pytest.approx(10 / 10.5)
    assert batch["kernel_faster"] is True
    assert batch["full_service_benefit"] is False
    assert (
        summary["honest_cost_verdict"] == "complete_null_faster_kernel_without_full_service_benefit"
    )


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-PLACEMENT
def test_stage_fractions_and_amdahl_bounds_require_small_unaccelerated_fraction() -> None:
    rows = [
        experiment.synthetic_service_row(
            batch_size=128,
            block=index,
            arm="vectorized_numpy",
            scoring_s=1.0,
            complete_s=100.0,
        )
        for index in range(30)
    ]
    analysis = experiment.compute_amdahl(rows, batch_size=128, arm="vectorized_numpy")

    assert sum(analysis["measured_stage_fractions"].values()) <= 1.0 + 1e-12
    assert analysis["scoring_fraction"] == pytest.approx(0.01)
    assert analysis["unaccelerated_fraction"] == pytest.approx(0.99)
    assert analysis["finite_speedup_at_assumed_100x"] == pytest.approx(1 / 0.9901)
    assert analysis["infinite_device_upper_bound"] == pytest.approx(1 / 0.99)
    assert analysis["hundred_x_necessary_condition"] == {
        "operator": "<=",
        "expected": 0.01,
        "observed": pytest.approx(0.99),
        "passed": False,
    }
    assert analysis["recommendation"] == "retain_cpu_or_batch_before_port"


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-BOARDS
def test_three_board_rows_preserve_claim_boundaries_and_expected_block(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    _, _, context = experiment.collect_preconditions(
        ROOT,
        paths,
        candidate_paths=[tmp_path / "missing.md"],
    )
    rows = experiment.build_board_rows(context)
    boards = {row["board"]: row for row in rows}

    assert set(boards) == {"KV260", "GateMate", "PolarFire"}
    assert boards["KV260"]["terminal_state"] == "graduated_preserved"
    assert boards["KV260"]["last_authenticated_venue"] == "kv260_fpga_fabric"
    assert boards["KV260"]["future_access"] == "ssh_only"
    assert boards["KV260"]["architecture_limit"] == "k_max<=5"
    assert boards["PolarFire"]["terminal_state"] == "graduated_cpu_dispatch_preserved"
    assert boards["PolarFire"]["last_authenticated_venue"] == "polarfire_linux_cpu"
    assert boards["PolarFire"]["fpga_sampling_claimed"] is False
    assert boards["GateMate"]["terminal_state"] == "blocked_changed_physical_state"
    assert boards["GateMate"]["hardware_ready_score"] == 0
    assert boards["GateMate"]["error"] == experiment.MISSING_RECEIPT
    assert all(row["new_hardware_execution_claimed"] is False for row in rows)


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-ARTIFACT
def test_terminal_fixture_schema_checksum_and_mutations_fail_closed(tmp_path: Path) -> None:
    artifact = experiment.build_fixture_artifact()

    assert experiment.validate_artifact(artifact) == []
    assert artifact["schema"] == experiment.SCHEMA
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == experiment.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["service_cost_capture_complete_score"] == 1
    assert artifact["hardware_ready_score"] == 0
    assert artifact["hardware_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert len(artifact["service_cost_rows"]) == 180
    assert artifact["board_rows"][1]["terminal_state"] == "blocked_changed_physical_state"
    assert artifact["honest_verdict"].startswith("complete_")

    changed = deepcopy(artifact)
    changed["service_cost_rows"][0]["complete_return_s"] *= 2
    assert "checksum" in experiment.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["service_cost_rows"][0]["parity_passed"] = False
    changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
    assert "service_rows" in experiment.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["hardware_ready_score"] = 1
    changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
    assert "scores" in experiment.validate_artifact(changed)

    path = tmp_path / "artifact.json"
    receipt = experiment.write_artifact(path, artifact)
    assert receipt["sha256"] == experiment.sha256_file(path)
    assert json.loads(path.read_text(encoding="utf-8")) == artifact


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-ARTIFACT
def test_independent_reduction_and_scoped_validation_plan(tmp_path: Path) -> None:
    artifact = experiment.build_fixture_artifact()
    raw = {
        "service_cost_rows": deepcopy(artifact["service_cost_rows"]),
        "board_rows": deepcopy(artifact["board_rows"]),
        "source_artifact_hashes": deepcopy(artifact["source_artifact_hashes"]),
    }
    replay = experiment.independent_reduce(artifact, raw)
    assert all(replay.values())

    commands = experiment.build_validation_plan(ROOT, tmp_path / "private")
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert experiment.validate_validation_plan(ROOT, commands) == []
    assert all(command.name != "full_python_suite" for command in commands)
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert "-n" in focused.argv
    assert "0" in focused.argv
    assert "-o" in focused.argv
    assert "addopts=" in focused.argv
    assert "--no-cov" in focused.argv
    coverage = next(command for command in commands if command.name == "changed_module_coverage")
    assert any(argument.startswith("--basetemp=") for argument in coverage.argv)
    report = next(
        command for command in commands if command.name == "changed_module_coverage_report"
    )
    assert report.command_environment

    expanded = [*commands, commands[0]]
    assert "duplicate_command:worktree_imports" in experiment.validate_validation_plan(
        ROOT, expanded
    )


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-FIXTURE
def test_numeric_helpers_reject_invalid_inputs() -> None:
    fixture = experiment.load_fixed_fixture(ROOT)
    state = deepcopy(fixture["state"])

    with pytest.raises(ValueError, match="texts"):
        experiment.run_service([], state, arm="scalar_numpy")
    bad = deepcopy(state)
    bad["weights"]["w1"] = [[1.0, 2.0]]
    with pytest.raises(ValueError, match="shape"):
        experiment.run_service(["1 + 1. Therefore 2."], bad, arm="scalar_numpy")
    with pytest.raises(ValueError, match="paired rows"):
        experiment.summarize_cost_rows([], draws=10, seed=1)
    with pytest.raises(ValueError, match="rows"):
        experiment.compute_amdahl([], batch_size=128, arm="vectorized_numpy")

    vector = np.asarray([[0.0, 1.0]], dtype=np.float64)
    energies = experiment.vectorized_energy(state["weights"], vector)
    assert energies.shape == (1,)


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-FIXTURE
def test_input_and_numeric_defensive_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable_json"):
        experiment.load_json(invalid)
    assert experiment._referenced_hash({}, experiment.KV260_TRANSCRIPT_PATH) is None
    with pytest.raises(ValueError, match="board_row_not_unique"):
        experiment._board_by_name({}, "KV260")
    with pytest.raises(ValueError, match="not unique"):
        experiment._unit_state({}, {})

    real_load = experiment.load_json
    monkeypatch.setattr(
        experiment,
        "load_json",
        lambda path: [] if path.name == experiment.CORPUS_PATH.name else real_load(path),
    )
    with pytest.raises(ValueError, match="incomplete"):
        experiment.load_fixed_fixture(ROOT)
    monkeypatch.setattr(
        experiment,
        "load_json",
        lambda path: [{}] * 128 if path.name == experiment.CORPUS_PATH.name else real_load(path),
    )
    with pytest.raises(ValueError, match="128 valid rows"):
        experiment.load_fixed_fixture(ROOT)
    monkeypatch.setattr(experiment, "load_json", real_load)

    fixture = experiment.load_fixed_fixture(ROOT)
    state = fixture["state"]
    with pytest.raises(ValueError, match="fields"):
        experiment.scalar_energy({}, [0.0, 0.0])
    nonfinite = deepcopy(state["weights"])
    nonfinite["b_out"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        experiment.scalar_energy(nonfinite, [0.0, 0.0])
    with pytest.raises(ValueError, match="features"):
        experiment.scalar_energy(state["weights"], [0.0])
    with pytest.raises(ValueError, match="feature matrix"):
        experiment.vectorized_energy(state["weights"], np.asarray([0.0, 1.0]))
    no_policy = deepcopy(state)
    no_policy.pop("policy")
    with pytest.raises(ValueError, match="policy"):
        experiment.run_service(["1 + 1. Therefore 2."], no_policy, arm="scalar_numpy")

    disabled = {
        "accept_threshold": 0.1,
        "reject_threshold": 0.9,
        "accept_enabled": False,
        "reject_enabled": True,
    }
    action = experiment._policy_action(0.05, disabled, "test")
    assert action["decision"] == "escalate"
    assert action["reason"] == "accept_action_uncertified_escalation"


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-FIXTURE
def test_optional_adaptive_state_is_classified_without_gating_fixed_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adaptive = tmp_path / "adaptive.json"
    experiment.atomic_json(
        adaptive,
        {
            "verdict_class": "null",
            "flagged_adversarial": False,
            "required_checks_passed": True,
        },
    )
    monkeypatch.setattr(experiment, "EXP7399_PATH", adaptive)
    _, hashes, context = experiment.collect_preconditions(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path / "outputs"),
        candidate_paths=[tmp_path / "missing.md"],
    )
    assert context["optional_adaptive_service"]["eligible"] is True
    assert context["optional_adaptive_service"]["disposition"] == "eligible_report_separately"
    assert hashes[adaptive.as_posix()] == experiment.sha256_file(adaptive)


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-TIMING
def test_reducers_cover_positive_no_kernel_and_invalid_pair_boundaries() -> None:
    with pytest.raises(ValueError, match="benchmark batch"):
        experiment.measure_service_cost(["text"], {}, batch_sizes=(2,), blocks=1)
    with pytest.raises(ValueError, match="synthetic timing"):
        experiment.synthetic_service_row(
            batch_size=1, block=0, arm="scalar_numpy", scoring_s=2.0, complete_s=1.0
        )
    with pytest.raises(ValueError, match="common non-empty shape"):
        experiment._bootstrap_ratio(np.asarray([]), np.asarray([]), draws=1, seed=1)
    one = experiment.synthetic_service_row(
        batch_size=1, block=0, arm="scalar_numpy", scoring_s=1.0, complete_s=2.0
    )
    with pytest.raises(ValueError, match="exactly one row per arm"):
        experiment.summarize_cost_rows([one], draws=10, seed=1)

    positive = []
    no_kernel = []
    for block in range(30):
        positive.extend(
            [
                experiment.synthetic_service_row(
                    batch_size=128,
                    block=block,
                    arm="scalar_numpy",
                    scoring_s=4.0,
                    complete_s=12.0,
                ),
                experiment.synthetic_service_row(
                    batch_size=128,
                    block=block,
                    arm="vectorized_numpy",
                    scoring_s=1.0,
                    complete_s=10.0,
                ),
            ]
        )
        no_kernel.extend(
            [
                experiment.synthetic_service_row(
                    batch_size=128,
                    block=block,
                    arm="scalar_numpy",
                    scoring_s=1.0,
                    complete_s=10.0,
                ),
                experiment.synthetic_service_row(
                    batch_size=128,
                    block=block,
                    arm="vectorized_numpy",
                    scoring_s=2.0,
                    complete_s=10.5,
                ),
            ]
        )
    assert (
        experiment.summarize_cost_rows(positive, draws=50, seed=1)["honest_cost_verdict"]
        == "complete_positive_vectorized_full_service_benefit"
    )
    assert (
        experiment.summarize_cost_rows(no_kernel, draws=50, seed=1)["honest_cost_verdict"]
        == "complete_null_no_vectorized_scoring_or_full_service_benefit"
    )


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-ARTIFACT
def test_blocked_and_failed_validation_artifact_classification() -> None:
    blocked = experiment.assemble_artifact(
        service_rows=[],
        board_rows=experiment._fixture_board_rows(),
        preconditions=[
            experiment.gate_row("mandatory_fixture", "mandatory_precondition", True, False)
        ],
        source_hashes={"fixture": None},
        validation_receipts=experiment._fixture_receipts(),
        phase_spans=[],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:01+00:00",
        duration_s=1.0,
        historical_sidecars=[],
        optional_adaptive_service={},
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_fixed_service_fixture_unavailable"
    assert experiment.validate_artifact(blocked) == []

    fixture = experiment.build_fixture_artifact()
    receipts = experiment._fixture_receipts()
    receipts[0]["passed"] = False
    receipts[0]["exit_code"] = 1
    failed = experiment.assemble_artifact(
        service_rows=fixture["service_cost_rows"],
        board_rows=fixture["board_rows"],
        preconditions=[experiment.gate_row("fixture", "mandatory_precondition", True, True)],
        source_hashes=fixture["source_artifact_hashes"],
        validation_receipts=receipts,
        phase_spans=[],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:01+00:00",
        duration_s=1.0,
        historical_sidecars=[],
        optional_adaptive_service={},
    )
    assert failed["verdict_class"] == "disqualified"
    assert failed["honest_verdict"].startswith("complete_disqualified")

    changed = deepcopy(fixture)
    changed["promotion_score"] = 1
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        experiment.write_artifact(Path("/tmp/exp7407-invalid.json"), changed)


# REQ-REPORT-7407 / SCENARIO-REPORT-7407-ARTIFACT
def test_cli_cold_replay_and_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = experiment.build_fixture_artifact()
    raw = {
        "service_cost_rows": artifact["service_cost_rows"],
        "board_rows": artifact["board_rows"],
        "source_artifact_hashes": artifact["source_artifact_hashes"],
    }
    candidate = tmp_path / "candidate.json"
    raw_path = tmp_path / "raw.json"
    experiment.atomic_json(candidate, artifact)
    experiment.atomic_json(raw_path, raw)

    assert experiment.main(["--cold-replay", str(candidate), "--raw", str(raw_path)]) == 0
    with pytest.raises(SystemExit, match="--raw"):
        experiment.main(["--cold-replay", str(candidate)])
    with pytest.raises(SystemExit, match="--date"):
        experiment.main([])

    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        experiment,
        "run_experiment",
        lambda root, date, paths: calls.append((root, date)) or artifact,
    )
    assert experiment.main(["--date", experiment.RUN_DATE]) == 0
    assert calls == [(experiment.REPO_ROOT, experiment.RUN_DATE)]

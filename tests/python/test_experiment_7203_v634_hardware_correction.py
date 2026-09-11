"""Tests for the V634 host correction-cost and board-continuity envelope.

Spec refs: REQ-ISING-7203 and SCENARIO-ISING-7203-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7187_v633_slice_sampler as slices
from carnot import experiment_7188_v633_quantized_transition_audit as correction
from carnot import experiment_7203_v634_hardware_correction as exp


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = exp.artifact_checksum(payload)
    return payload


@pytest.fixture(scope="module")
def ready_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the production-size CPU panel once for all artifact assertions."""

    private = tmp_path_factory.mktemp("exp7203")
    return exp.build_artifact(
        exp.PROJECT_ROOT,
        exp.RUN_DATE,
        output_path=private / "result.json",
        checkpoint_path=private / "checkpoint.json",
    )


def test_req_ising_7203_spec_precedes_implementation() -> None:
    """REQ-ISING-7203 names the host envelope and executable before code use."""

    text = (exp.PROJECT_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "### REQ-ISING-7203" in text
    assert "### SCENARIO-ISING-7203-PREFLIGHT" in text
    assert "### SCENARIO-ISING-7203-BOARDS" in text
    assert "### SCENARIO-ISING-7203-SMALL-LAW" in text
    assert "### SCENARIO-ISING-7203-COST" in text
    assert "### SCENARIO-ISING-7203-BREAK-EVEN" in text
    assert "### SCENARIO-ISING-7203-ARTIFACT" in text
    assert "experiment_7203_v634_hardware_correction.py" in text


def test_scenario_ising_7203_small_law_and_exact_grid_control() -> None:
    """SCENARIO-ISING-7203-SMALL-LAW checks correction and no-change control."""

    rows = exp.run_small_law_checks(seeds=(exp.SEEDS[0],), bits_values=(4,))
    measured = next(row for row in rows if not row["exact_grid_negative_control"])
    control = next(row for row in rows if row["exact_grid_negative_control"])

    assert measured["changed_coefficient_count"] > 0
    assert measured["corrected_stationary_residual_max"] <= exp.LAW_TOLERANCE
    assert measured["corrected_detailed_balance_error_max"] <= exp.LAW_TOLERANCE
    assert measured["quantized_target_tv_from_full"] > 0.0
    assert control["changed_coefficient_count"] == 0
    assert control["quantized_target_tv_from_full"] == pytest.approx(0.0, abs=1e-15)
    assert control["passed"] is True


def test_req_ising_7203_reuses_full_graph_and_quantizer() -> None:
    """REQ-ISING-7203 retains every Exp7187 edge, field, and graph statistic."""

    instance = slices.make_frustrated_instance(64, exp.SEEDS[0])
    metrics = exp.graph_metrics(instance)
    quantized = correction.quantize_instance(instance, 16)

    assert metrics["edge_count"] == len(instance.edges)
    assert metrics["maximum_degree"] <= 16
    assert metrics["edges_dropped"] == 0
    assert metrics["nonzero_field_count"] == 64
    assert metrics["frustrated_triangle"] is True
    assert exp.changed_coefficient_count(instance, quantized) > 0
    assert exp.exact_grid_control(8).fields == (1.0, -1.0) * 4


def test_scenario_ising_7203_equal_work_preserves_rejected_states() -> None:
    """SCENARIO-ISING-7203-COST retains one state after every rejected move."""

    instance = slices.make_frustrated_instance(16, exp.SEEDS[0])
    full = exp.run_chain(
        instance,
        bits=4,
        arm=exp.FULL_ARM,
        budget_kind=exp.EQUAL_WORK,
        seed=exp.SEEDS[0],
        proposal_limit=128,
        wall_budget_s=0.001,
    )
    corrected = exp.run_chain(
        instance,
        bits=4,
        arm=exp.CORRECTED_ARM,
        budget_kind=exp.EQUAL_WORK,
        seed=exp.SEEDS[0],
        proposal_limit=128,
        wall_budget_s=0.001,
    )

    assert full["proposal_count"] == corrected["proposal_count"] == 128
    assert full["full_energy_calls"] == 129
    assert corrected["full_energy_calls"] <= 129
    assert corrected["cheap_stage_reject_count"] + corrected["stage_one_accept_count"] == 128
    for row in (full, corrected):
        assert row["chain_state_count"] == row["proposal_count"] + 1
        assert row["rejected_state_retention_count"] == (
            row["proposal_count"] - row["accepted_move_count"]
        )
        assert row["trace_sha256"].startswith("sha256:")


def test_scenario_ising_7203_equal_wall_is_real_elapsed_work() -> None:
    """SCENARIO-ISING-7203-COST uses elapsed time instead of a padded duration."""

    row = exp.run_chain(
        slices.make_frustrated_instance(16, exp.SEEDS[1]),
        bits=8,
        arm=exp.CORRECTED_ARM,
        budget_kind=exp.EQUAL_WALL,
        seed=exp.SEEDS[1],
        proposal_limit=8,
        wall_budget_s=0.003,
    )

    assert row["elapsed_wall_time_s"] >= 0.003
    assert row["proposal_count"] > 0
    assert row["wall_budget_s"] == 0.003
    assert row["latency_s_per_proposal"] == pytest.approx(
        row["elapsed_wall_time_s"] / row["proposal_count"]
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"arm": "bad"}, "unknown arm"),
        ({"budget_kind": "bad"}, "unknown budget"),
        ({"proposal_limit": 0}, "proposal_limit"),
        ({"wall_budget_s": 0.0}, "wall_budget_s"),
    ],
)
def test_req_ising_7203_chain_contract_rejects_invalid_budgets(
    kwargs: dict[str, Any], message: str
) -> None:
    """REQ-ISING-7203 refuses measurements outside the frozen budget contract."""

    call = {
        "bits": 4,
        "arm": exp.FULL_ARM,
        "budget_kind": exp.EQUAL_WORK,
        "seed": exp.SEEDS[0],
        "proposal_limit": 8,
        "wall_budget_s": 0.001,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        exp.run_chain(slices.make_frustrated_instance(16, exp.SEEDS[0]), **call)


def test_scenario_ising_7203_board_rows_keep_evidence_distinct() -> None:
    """SCENARIO-ISING-7203-BOARDS retains three authentic, distinct dispositions."""

    board_rows, operator = exp.load_board_evidence(exp.PROJECT_ROOT)
    by_board = {row["board"]: row for row in board_rows}

    assert set(by_board) == {"KV260", "GateMate", "PolarFire"}
    assert by_board["KV260"]["disposition"] == "graduated_preserved"
    assert by_board["KV260"]["programmable_logic_sampling_observed"] is True
    assert by_board["GateMate"]["disposition"] == "blocked_inherited_no_new_physical_state"
    assert by_board["GateMate"]["hardware_command_count"] == 0
    assert by_board["PolarFire"]["board_cpu_work_observed"] is True
    assert by_board["PolarFire"]["programmable_logic_sampling_observed"] is False
    assert by_board["PolarFire"]["raw_transcript_hash"] is None
    assert operator["newer_than_exp6559"] is False
    assert operator["authorized_action_for_later_task"] is None
    assert operator["hardware_command_count"] == 0
    assert all(row["source_hash"].startswith("sha256:") for row in board_rows)


def test_scenario_ising_7203_preflight_rejects_quarantine_before_field() -> None:
    """SCENARIO-ISING-7203-PREFLIGHT rejects a clean-looking quarantined score."""

    clean = {"corrected_kernel_ready_score": 1, "flagged_adversarial": False}
    quarantined = {"corrected_kernel_ready_score": 1, "flagged_adversarial": True}

    assert exp.upstream_gate(clean, "source.json", "corrected_kernel_ready_score", 1)["passed"]
    rejected = exp.upstream_gate(quarantined, "source.json", "corrected_kernel_ready_score", 1)
    assert rejected["passed"] is False
    assert rejected["observed_value"] == {"value": 1, "quarantined": True}


def test_req_ising_7203_preconditions_bind_real_contract() -> None:
    """REQ-ISING-7203 checks bytes, tools, output paths, gates, and quarantine."""

    checks, hashes = exp.collect_preconditions(
        exp.PROJECT_ROOT,
        output_path=exp.PROJECT_ROOT / exp.RESULT_PATH,
        checkpoint_path=exp.PROJECT_ROOT / exp.CHECKPOINT_PATH,
    )
    assert checks
    assert all(row["passed"] for row in checks)
    assert set(exp.REQUIRED_SOURCE_PATHS) <= {Path(path) for path in hashes}
    task = next(row for row in checks if row["check"] == "same_milestone_gate_fields")
    assert task["observed_value"] == exp.EXPECTED_TASK_CONTRACT
    quarantine = [row for row in checks if row["check"].endswith("_not_quarantined")]
    assert len(quarantine) == 2
    assert all(row["observed_value"]["quarantined"] is False for row in quarantine)


def test_scenario_ising_7203_complete_artifact_contract(
    ready_artifact: dict[str, Any],
) -> None:
    """SCENARIO-ISING-7203-ARTIFACT keeps every requested unit and claim limit."""

    artifact = ready_artifact
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["run_date"] == "20260911"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["hardware_execution_claimed"] is False
    assert artifact["topology_fit"] == "topology_unknown"
    assert artifact["hardware_envelope_complete_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["board_rows"]) == 3
    assert len(artifact["exact_law_rows"]) == 33
    assert len(artifact["correction_cost_rows"]) == 360
    assert len(artifact["break_even_rows"]) == 144
    assert artifact["rows"] == (
        artifact["board_rows"]
        + artifact["exact_law_rows"]
        + artifact["correction_cost_rows"]
        + artifact["break_even_rows"]
    )
    assert all(
        {"unit_id", "arm", "seed", "metric", "error", "abstention"} <= set(row)
        for row in artifact["rows"]
    )
    assert artifact["sample_size_budget"]["completed"]["cost_rows"] == 360
    assert artifact["sample_size_budget"]["exclusions"] == []


def test_scenario_ising_7203_all_cost_cells_and_graphs_are_retained(
    ready_artifact: dict[str, Any],
) -> None:
    """SCENARIO-ISING-7203-COST reports every precision, seed, arm, and budget."""

    rows = ready_artifact["correction_cost_rows"]
    cells = {
        (row["n"], row["seed"], row["precision_bits"], row["budget_kind"], row["arm"])
        for row in rows
    }
    expected = {
        (n, seed, bits, budget, arm)
        for n in exp.SIZES
        for seed in exp.SEEDS
        for bits in exp.BITS
        for budget in exp.BUDGET_KINDS
        for arm in exp.ARMS
    }
    assert cells == expected
    assert all(row["edge_count"] > 0 for row in rows)
    assert all(row["maximum_degree"] <= 16 for row in rows)
    assert all(row["edges_dropped"] == 0 for row in rows)
    assert all(row["changed_coefficient_count"] > 0 for row in rows)
    assert all(row["topology_fit"] == "topology_unknown" for row in rows)
    equal_work = [row for row in rows if row["budget_kind"] == exp.EQUAL_WORK]
    equal_wall = [row for row in rows if row["budget_kind"] == exp.EQUAL_WALL]
    assert all(row["proposal_count"] == 1024 for row in equal_work)
    assert all(row["elapsed_wall_time_s"] >= 0.1 for row in equal_wall)


def test_scenario_ising_7203_break_even_is_hypothetical_per_transition(
    ready_artifact: dict[str, Any],
) -> None:
    """SCENARIO-ISING-7203-BREAK-EVEN does not turn an envelope into speed evidence."""

    rows = ready_artifact["break_even_rows"]
    assert all(row["device_timing_source"] == "explicit_hypothesis" for row in rows)
    assert all(row["measured_device_timing"] == "unknown" for row in rows)
    assert all(row["claim_unit"] == "per_proposed_transition" for row in rows)
    assert all(row["equal_effective_sample_throughput_established"] is False for row in rows)
    assert all(row["mixing_speed_established"] is False for row in rows)
    assert ready_artifact["device_timing_available"] is False
    assert ready_artifact["power_savings_claimed"] is False


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("hardware_envelope_complete_score", 0, "completion_score_invalid"),
        ("hardware_execution_claimed", True, "hardware_claim_invalid"),
        ("topology_fit", "fits", "topology_fit_invalid"),
        ("MODEL_SPECS", ["model"], "model_contract_invalid"),
    ],
)
def test_req_ising_7203_validator_rejects_claim_inflation(
    ready_artifact: dict[str, Any], field: str, value: Any, error: str
) -> None:
    """REQ-ISING-7203 validation recomputes scores and execution boundaries."""

    changed = deepcopy(ready_artifact)
    changed[field] = value
    _rehash(changed)
    assert error in exp.validate_artifact(changed)


def test_scenario_ising_7203_external_failure_is_terminal_blocked(tmp_path: Path) -> None:
    """SCENARIO-ISING-7203-PREFLIGHT emits a diagnosed block without CPU rows."""

    artifact = exp.build_artifact(
        exp.PROJECT_ROOT,
        exp.RUN_DATE,
        output_path=tmp_path / "missing" / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    assert artifact["status"] == "blocked_external_precondition"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["passed"] is False
    assert artifact["gate_check_summary"]["failed_check"] == "output_directories"
    assert exp.validate_artifact(artifact) == []


def test_req_ising_7203_atomic_writer_and_validation_cli(
    ready_artifact: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ISING-7203 publishes atomically and validates the caller-selected file."""

    output = tmp_path / "artifact.json"
    receipt = exp.atomic_write(output, ready_artifact)
    assert receipt["atomic_replace"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == ready_artifact
    assert exp.main(["--validate", str(output)]) == 0
    assert "validation_passed" in capsys.readouterr().out

    invalid = deepcopy(ready_artifact)
    invalid["hardware_execution_claimed"] = True
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    assert exp.main(["--validate", str(invalid_path)]) == 2
    with pytest.raises(ValueError, match="invalid Exp7203 artifact"):
        exp.atomic_write(tmp_path / "rejected.json", invalid)


def test_req_ising_7203_cli_rejects_wrong_date(tmp_path: Path) -> None:
    """REQ-ISING-7203 never silently substitutes a historical execution date."""

    assert (
        exp.main(
            [
                "--date",
                "20260910",
                "--output",
                str(tmp_path / "result.json"),
                "--checkpoint",
                str(tmp_path / "checkpoint.json"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "result.json").exists()


def test_req_ising_7203_hash_and_json_defensive_paths(tmp_path: Path) -> None:
    """REQ-ISING-7203 rejects nonfinite hashes and non-object artifacts."""

    with pytest.raises(ValueError, match="nonfinite value"):
        exp.canonical_json({"bad": float("nan")})
    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        exp._read_json(list_path)


def test_scenario_ising_7203_task_parser_defensive_paths(tmp_path: Path) -> None:
    """SCENARIO-ISING-7203-PREFLIGHT rejects missing and malformed roadmaps."""

    assert exp._task_contract(tmp_path) is None
    roadmap = tmp_path / exp.ROADMAP_PATH
    roadmap.write_text("[", encoding="utf-8")
    assert exp._task_contract(tmp_path) is None
    roadmap.write_text("tasks: wrong\n", encoding="utf-8")
    assert exp._task_contract(tmp_path) is None
    roadmap.write_text("tasks: []\n", encoding="utf-8")
    assert exp._task_contract(tmp_path) is None


def test_scenario_ising_7203_missing_upstreams_are_checked_not_consumed(tmp_path: Path) -> None:
    """SCENARIO-ISING-7203-PREFLIGHT records absent upstream JSON as failed checks."""

    checks, hashes = exp.collect_preconditions(
        tmp_path,
        output_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    by_name = {row["check"]: row for row in checks}
    assert hashes == {}
    assert by_name["exp7188_not_quarantined"]["passed"] is False
    assert by_name["exp7190_not_quarantined"]["passed"] is False
    assert by_name["exp7188_exact_gate_fields"]["observed_value"] == ("not_consumed_quarantined")
    assert by_name["exp7190_exact_gate_fields"]["observed_value"] == ("not_consumed_quarantined")


def _write_board_fixture(root: Path, board_rows: Any, operator: Any = None) -> None:
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    payload = {"board_rows": board_rows, "operator_state_receipt": operator}
    (root / exp.UPSTREAM_BOARD_PATH).write_text(json.dumps(payload), encoding="utf-8")
    for relative in (
        "evidence.json",
        "transcript.json",
        "results/experiment_6559_gatemate_changed_state_continuity.json",
        "results/experiment_7146_v627_gatemate_changed_state.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def _minimal_board(board: str) -> dict[str, Any]:
    terminal = board == "KV260"
    return {
        "board": board,
        "evidence_path": "evidence.json",
        "recorded_date": "20260901",
        "terminal_criterion_met": terminal,
        "exact_next_prerequisite": None if terminal else "later evidence",
        "supporting_evidence_paths": ["transcript.json"] if terminal else [],
        "raw_transcript_hash": "sha256:does-not-match",
        "disposition": "graduated_preserved" if terminal else "blocked",
        "hardware_command_count": 0,
    }


def test_scenario_ising_7203_board_parser_defensive_paths(tmp_path: Path) -> None:
    """SCENARIO-ISING-7203-BOARDS fails closed on malformed or unsupported receipts."""

    _write_board_fixture(tmp_path, {})
    with pytest.raises(ValueError, match="must be a list"):
        exp.load_board_evidence(tmp_path)

    _write_board_fixture(tmp_path, [1])
    with pytest.raises(ValueError, match="row is malformed"):
        exp.load_board_evidence(tmp_path)

    missing = _minimal_board("KV260")
    missing["evidence_path"] = "absent.json"
    _write_board_fixture(tmp_path, [missing])
    with pytest.raises(ValueError, match="evidence is missing"):
        exp.load_board_evidence(tmp_path)

    _write_board_fixture(tmp_path, [_minimal_board("KV260")])
    with pytest.raises(ValueError, match="exactly the three"):
        exp.load_board_evidence(tmp_path)

    rows = [_minimal_board(board) for board in ("KV260", "GateMate", "PolarFire")]
    _write_board_fixture(tmp_path, rows, operator=[])
    parsed, operator = exp.load_board_evidence(tmp_path)
    kv260 = next(row for row in parsed if row["board"] == "KV260")
    assert kv260["disposition"] == "blocked_invalid_kv260_graduation_receipt"
    assert kv260["programmable_logic_sampling_observed"] is False
    assert kv260["abstention"] is True
    assert operator["newer_than_exp6559"] is False


def test_scenario_ising_7203_cost_panel_fails_closed_on_controls_and_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7203-COST enforces coefficient change and the host cap."""

    monkeypatch.setattr(exp, "changed_coefficient_count", lambda _instance, _quantized: 0)
    with pytest.raises(ValueError, match="changed no coefficient"):
        exp.run_cost_panel(tmp_path / "checkpoint.json")
    monkeypatch.undo()

    monkeypatch.setattr(exp, "HOST_MEASUREMENT_CAP_SECONDS", 0.0)
    with pytest.raises(TimeoutError, match="exceeded 900 seconds"):
        exp.run_cost_panel(tmp_path / "checkpoint.json")


def test_scenario_ising_7203_break_even_requires_all_seed_rows() -> None:
    """SCENARIO-ISING-7203-BREAK-EVEN refuses an incomplete measured basis."""

    with pytest.raises(ValueError, match="ten matched seed rows"):
        exp.build_break_even_rows([])


def test_scenario_ising_7203_small_law_failure_blocks_cost_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7203-SMALL-LAW remains authority for later cost claims."""

    monkeypatch.setattr(
        exp,
        "collect_preconditions",
        lambda *_args, **_kwargs: ([{"passed": True}], {}),
    )
    monkeypatch.setattr(exp, "load_board_evidence", lambda _root: ([], {}))
    monkeypatch.setattr(exp, "run_small_law_checks", lambda: [{"passed": False}])
    with pytest.raises(ValueError, match="small-law correction authority failed"):
        exp.build_artifact(
            tmp_path,
            exp.RUN_DATE,
            output_path=tmp_path / "result.json",
            checkpoint_path=tmp_path / "checkpoint.json",
        )
    with pytest.raises(ValueError, match="run date"):
        exp.build_artifact(
            tmp_path,
            "20260910",
            output_path=tmp_path / "result.json",
            checkpoint_path=tmp_path / "checkpoint.json",
        )


def test_req_ising_7203_validator_defensive_mutations(
    ready_artifact: dict[str, Any],
) -> None:
    """REQ-ISING-7203 independently rejects every evidence and claim row class."""

    mutations: list[tuple[str, Any]] = [
        ("missing_required_fields", lambda value: value.pop("task_id")),
        ("field_principles_invalid", lambda value: value.update(field_principles={})),
        ("run_date_invalid", lambda value: value.update(run_date="20260910")),
        ("hardware_command_invalid", lambda value: value.update(hardware_command_count=1)),
        ("unsupported_claim_invalid", lambda value: value.update(power_savings_claimed=True)),
        ("blocked_state_invalid", lambda value: value.update(verdict_class="blocked")),
        ("terminal_state_invalid", lambda value: value.update(status="running")),
        ("row_census_invalid", lambda value: value["exact_law_rows"].pop()),
        ("rows_invalid", lambda value: value.update(rows=[])),
        ("row_identity_invalid", lambda value: value["rows"][0].pop("unit_id")),
        (
            "board_rows_invalid",
            lambda value: value["board_rows"][0].update(disposition="unsupported"),
        ),
        (
            "exact_law_invalid",
            lambda value: next(
                row for row in value["exact_law_rows"] if not row["exact_grid_negative_control"]
            ).update(changed_coefficient_count=0),
        ),
        ("cost_cells_invalid", lambda value: value["correction_cost_rows"].pop()),
        (
            "cost_row_invalid",
            lambda value: value["correction_cost_rows"][0].update(edges_dropped=1),
        ),
        (
            "break_even_invalid",
            lambda value: value["break_even_rows"][0].update(device_timing_source="measured"),
        ),
    ]
    for expected, mutate in mutations:
        changed = deepcopy(ready_artifact)
        mutate(changed)
        _rehash(changed)
        assert expected in exp.validate_artifact(changed)


def test_req_ising_7203_run_boundaries_and_relative_cli_paths(
    ready_artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ISING-7203 exposes every phase and resolves caller-relative paths."""

    stable = deepcopy(ready_artifact)
    monkeypatch.setattr(exp, "build_artifact", lambda *_args, **_kwargs: stable)
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: [])
    monkeypatch.setattr(
        exp,
        "atomic_write",
        lambda path, _artifact: {
            "path": str(path),
            "sha256": "sha256:" + "0" * 64,
            "atomic_replace": True,
        },
    )
    result = exp.run_experiment(
        tmp_path,
        exp.RUN_DATE,
        output_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    assert result is stable
    assert "[phase 8 end]" in capsys.readouterr().out

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced_invalid"])
    with pytest.raises(ValueError, match="invalid Exp7203 artifact"):
        exp.run_experiment(
            tmp_path,
            exp.RUN_DATE,
            output_path=tmp_path / "result.json",
            checkpoint_path=tmp_path / "checkpoint.json",
        )

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda *_args, **_kwargs: {
            "hardware_envelope_complete_score": 1,
            "verdict_class": "circular_positive",
        },
    )
    assert (
        exp.main(
            [
                "--root",
                str(tmp_path),
                "--output",
                "relative.json",
                "--checkpoint",
                "relative-checkpoint.json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert str(tmp_path / "relative.json") in output

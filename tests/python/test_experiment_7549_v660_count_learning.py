"""Tests for REQ-CL-7549 and SCENARIO-CL-7549-*.

Small private streams exercise causal boundaries. The checked-in artifact test
uses the real 159-source measurement without turning old labels into fresh data.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7549_v660_count_learning as exp
from carnot.experiment_7547_v660_count_stream import compute_count_config
from carnot.reporting import experiment_7303_validation_scope as validation_scope


ROOT = Path(__file__).resolve().parents[2]


def _online_rows(count: int = 20) -> list[dict[str, Any]]:
    """Create varied forecasts so delayed local counts have several bins."""

    return [
        {
            "group_id": f"online-{index:03d}",
            "role": "online",
            "source_hash": f"sha256:{index + 1:064x}",
            "source_family": f"family-{index % 3}",
            "base_probability": 0.05 + 0.9 * ((index % 10) / 9),
        }
        for index in range(count)
    ]


def _retention_rows(count: int = 8) -> list[dict[str, Any]]:
    """Build evaluator rows whose labels never enter learner updates."""

    return [
        {
            "group_id": f"retention-{index:03d}",
            "role": "test",
            "source_hash": f"sha256:{1000 + index:064x}",
            "source_family": f"retained-{index % 2}",
            "base_probability": 0.1 + 0.8 * ((index % 5) / 4),
            "label": index % 2,
        }
        for index in range(count)
    ]


def _config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the qualified count configuration rather than test-only arithmetic."""

    training = [{**row, "raw_whole_expectation": row["base_probability"]} for row in rows]
    config = compute_count_config(training)
    return {
        "bin_means": list(config.bin_means),
        "global_mean": config.global_mean,
        "kappa": config.kappa,
    }


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Represent one completed bounded command for pure artifact tests."""

    return {
        "name": name,
        "command": f"private {name}",
        "command_argv": ["private", name],
        "scope": "private_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "a" * 64,
        "output_tail": "private fixture",
        "passed": passed,
        "timed_out": False,
    }


def _receipts(*, terminal: bool = True) -> list[dict[str, Any]]:
    """Supply each required validation class without running child commands."""

    names = list(validation_scope.REQUIRED_CHECK_NAMES)
    if terminal:
        names.extend(exp.TERMINAL_CHECK_NAMES)
    return [_receipt(name) for name in names]


def test_decision_costs_and_escalation_ties() -> None:
    """REQ-CL-7549; SCENARIO-CL-7549-CONTROLS."""

    assert exp.typed_decision(0.01, 0)["action"] == "accept"
    assert exp.typed_decision(0.5, 1)["action"] == "escalate"
    assert exp.typed_decision(0.9, 1)["action"] == "reject"
    low_tie = exp.typed_decision(0.04, 0)
    high_tie = exp.typed_decision(0.8, 1)
    assert low_tie["action"] == high_tie["action"] == "escalate"
    assert low_tie["expected_costs"] == {
        "accept": pytest.approx(0.2),
        "reject": pytest.approx(0.96),
        "escalate": pytest.approx(0.2),
    }
    assert exp.brier(0.25, 1) == pytest.approx(0.5625)
    assert exp.log_loss(0.25, 1) == pytest.approx(1.38629436112)


def test_predict_release_update_persist_reload_and_retention(tmp_path: Path) -> None:
    """REQ-CL-7549; SCENARIO-CL-7549-CHRONOLOGY, -RETENTION, and -RESTART."""

    rows = _online_rows()
    labels = {row["group_id"]: index % 2 for index, row in enumerate(rows)}
    measured = exp.measure_order(
        rows,
        labels,
        _retention_rows(),
        [row["group_id"] for row in rows],
        _config(rows),
        seed=7549001,
        checkpoint_dir=tmp_path,
        retention_checkpoints=(0, 20),
    )
    assert len(measured["prediction_update_rows"]) == 20 * 4
    assert len(measured["retention_rows"]) == 2 * 8 * 4
    assert measured["released_source_count"] == 8
    assert measured["censored_source_count"] == 12
    assert measured["chronology_violation_count"] == 0
    assert measured["restart_mismatch_count"] == 0
    assert measured["duplicate_feedback_rejection_count"] == 1
    assert measured["retention_state_mutation_count"] == 0

    by_arm = {
        row["arm"]: row
        for row in measured["prediction_update_rows"]
        if row["event_id"] == "online-000"
    }
    assert set(by_arm) == set(exp.ARMS)
    assert all(row["label_available_at_prediction"] is False for row in by_arm.values())
    assert all(len(row["energies"]) == 2 for row in by_arm.values())
    assert all(row["feedback_update_id"] == "seed-7549001-release-000" for row in by_arm.values())
    assert by_arm["frozen"]["arm_update_applied"] is False
    assert by_arm["local_count"]["arm_update_applied"] is True
    assert measured["persistence_receipts"][0]["payload_equal_after_reload"] is True
    assert measured["persistence_receipts"][0]["uninterrupted_state_equal"] is True
    assert measured["release_rows"][0]["changed_label_binding_count"] > 0
    assert measured["update_timing"]["completed_updates"] == 8 * 3
    assert measured["persistence_timing"]["completed_persists"] == 1


def test_duplicate_and_wrong_rosters_fail_closed(tmp_path: Path) -> None:
    """REQ-CL-7549; SCENARIO-CL-7549-CHRONOLOGY."""

    rows = _online_rows()
    labels = {row["group_id"]: index % 2 for index, row in enumerate(rows)}
    with pytest.raises(ValueError, match="measurement_roster_mismatch"):
        exp.measure_order(
            rows,
            labels,
            _retention_rows(),
            [row["group_id"] for row in rows[:-1]],
            _config(rows),
            seed=7549001,
            checkpoint_dir=tmp_path,
            retention_checkpoints=(0, 20),
        )
    duplicate = deepcopy(rows)
    duplicate[-1]["group_id"] = duplicate[0]["group_id"]
    with pytest.raises(ValueError, match="measurement_roster_mismatch"):
        exp.measure_order(
            duplicate,
            labels,
            _retention_rows(),
            [row["group_id"] for row in rows],
            _config(rows),
            seed=7549001,
            checkpoint_dir=tmp_path,
            retention_checkpoints=(0, 20),
        )


def test_prediction_and_reload_parity_mutations_are_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7549; SCENARIO-CL-7549-RESTART detects both parity surfaces."""

    rows = _online_rows()
    labels = {row["group_id"]: index % 2 for index, row in enumerate(rows)}
    original_predict = exp.CountEventMachine.predict
    calls = 0

    def changed_prediction(
        machine: exp.CountEventMachine, event_id: str, p0: float, *, release_index: int
    ) -> Any:
        nonlocal calls
        calls += 1
        result = original_predict(machine, event_id, p0, release_index=release_index)
        copied = {name: dict(value) for name, value in result.items()}
        if calls % 2 == 0:
            copied["frozen"]["probability"] += 1e-9
        return copied

    monkeypatch.setattr(exp.CountEventMachine, "predict", changed_prediction)
    changed = exp.measure_order(
        rows,
        labels,
        _retention_rows(),
        [row["group_id"] for row in rows],
        _config(rows),
        seed=7549001,
        checkpoint_dir=tmp_path / "prediction",
        retention_checkpoints=(0, 20),
    )
    assert changed["restart_mismatch_count"] > 0
    monkeypatch.setattr(exp.CountEventMachine, "predict", original_predict)

    original_load = exp.CountEventMachine.load

    def changed_load(path: Path) -> exp.CountEventMachine:
        machine = original_load(path)
        machine.arms["local"]._counts[0][0] += 1e-9
        return machine

    monkeypatch.setattr(exp.CountEventMachine, "load", changed_load)
    changed = exp.measure_order(
        rows,
        labels,
        _retention_rows(),
        [row["group_id"] for row in rows],
        _config(rows),
        seed=7549001,
        checkpoint_dir=tmp_path / "reload",
        retention_checkpoints=(0, 20),
    )
    assert changed["restart_mismatch_count"] > 0


def test_prerelease_identity_fails_before_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7549; SCENARIO-CL-7549-CHRONOLOGY rejects unseen feedback."""

    rows = _online_rows()
    order = [row["group_id"] for row in rows]
    labels = {event_id: index % 2 for index, event_id in enumerate(order)}
    original_blocks = exp._blocks

    def changed_blocks(values: list[str]) -> list[dict[str, Any]]:
        blocks = original_blocks(values)
        blocks[0]["event_ids"][0] = values[-1]
        return blocks

    monkeypatch.setattr(exp, "_blocks", changed_blocks)
    with pytest.raises(ValueError, match="feedback_event_unknown"):
        exp.measure_order(
            rows,
            labels,
            _retention_rows(),
            order,
            _config(rows),
            seed=7549001,
            checkpoint_dir=tmp_path,
            retention_checkpoints=(0, 20),
        )


def test_source_bootstrap_replays_orders_and_reduces_registered_gates() -> None:
    """REQ-CL-7549; SCENARIO-CL-7549-UNCERTAINTY."""

    rows = _online_rows()
    labels = {row["group_id"]: index % 2 for index, row in enumerate(rows)}
    orders = {
        7549001: [row["group_id"] for row in rows],
        7549002: [row["group_id"] for row in reversed(rows)],
    }
    progress_units: list[int] = []
    draws = exp.source_cluster_bootstrap(
        rows,
        labels,
        _retention_rows(),
        orders,
        _config(rows),
        replicates=12,
        bootstrap_seed=7549010,
        retention_checkpoints=(0, 20),
        progress_hook=progress_units.append,
    )
    assert len(draws) == 12
    assert all(row["sampled_source_count"] == 20 for row in draws)
    assert all(row["order_count"] == 2 for row in draws)
    assert all(row["chronology_replayed"] is True for row in draws)
    assert all(set(row["online_delta_brier"]) == set(exp.COMPARATORS) for row in draws)
    assert progress_units == [12]
    assert exp._upper([], 0.95) is None
    with pytest.raises(ValueError, match="bootstrap_roster_mismatch"):
        exp.source_cluster_bootstrap([], {}, _retention_rows(), {}, _config(rows), replicates=1)
    with pytest.raises(ValueError, match="bootstrap_order_roster_mismatch"):
        exp.source_cluster_bootstrap(
            rows,
            labels,
            _retention_rows(),
            {7549001: orders[7549001][:-1]},
            _config(rows),
            replicates=1,
        )

    reduction = exp.reduce_measurement(
        prediction_rows=[
            {
                "event_id": f"g-{index}",
                "arm": arm,
                "label": index % 2,
                "brier": value,
                "disposition": "complete",
            }
            for index in range(20)
            for arm, value in (
                ("local_count", 0.1),
                ("frozen", 0.2),
                ("global_count", 0.2),
                ("shuffled_local", 0.2),
            )
        ],
        retention_rows=[],
        replay_summaries=[
            {
                "seed": seed,
                "changed_label_binding_count": 50,
                "chronology_violation_count": 0,
                "restart_mismatch_count": 0,
                "duplicate_feedback_rejection_count": 1,
            }
            for seed in exp.ORDER_SEEDS
        ],
        bootstrap_rows=[
            {
                "online_delta_brier": {name: -0.1 for name in exp.COMPARATORS},
                "retention_brier_deterioration": 0.0,
                "retention_cost_deterioration": 0.0,
            }
            for _ in range(1000)
        ],
        label_counts={"0": 100, "1": 59},
        expected_sources=20,
    )
    assert reduction["support_passed"] is False
    assert reduction["effect_passed"] is False
    assert reduction["mean_delta_brier"]["frozen"] == pytest.approx(-0.1)
    assert reduction["chronology_passed"] is True


def test_build_validate_and_mutation_boundaries(tmp_path: Path) -> None:
    """REQ-CL-7549; SCENARIO-CL-7549-ARTIFACT."""

    fixture = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    reduction = exp.validate_artifact(fixture, root=tmp_path, verify_sidecars=True)
    assert reduction["measurement_complete"] is True
    assert fixture["count_measurement_complete_score"] == 1
    assert fixture["confirmatory_benefit_score"] == 0
    assert fixture["model_invoked"] is False
    assert fixture["MODEL_SPECS"] == fixture["model_specs"] == []
    assert fixture["inference_substrate_class"] == "no_model_load"
    assert fixture["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert fixture["execution_venue"] == "host"
    assert fixture["continuous_self_learning_task"] is True
    assert set(fixture["prediction_update_rows"]) == {"path", "sha256", "bytes", "rows"}
    assert set(fixture["retention_rows"]) == {"path", "sha256", "bytes", "rows"}

    changed = deepcopy(fixture)
    changed["confirmatory_benefit_score"] = 1
    with pytest.raises(ValueError, match="confirmatory_benefit_score_mismatch"):
        exp.validate_artifact(changed, root=tmp_path, verify_sidecars=False)
    changed = deepcopy(fixture)
    changed["MODEL_SPECS"] = [{"model": "forbidden"}]
    with pytest.raises(ValueError, match="model_specs_must_be_empty"):
        exp.validate_artifact(changed, root=tmp_path, verify_sidecars=False)
    changed = deepcopy(fixture)
    changed["count_measurement_complete_score"] = 0
    with pytest.raises(ValueError, match="measurement_complete_score_mismatch"):
        exp.validate_artifact(changed, root=tmp_path, verify_sidecars=False)
    changed = deepcopy(fixture)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        exp.validate_artifact(changed, root=tmp_path, verify_sidecars=False)

    mutations = (
        ("schema", "wrong", "artifact_identity_mismatch"),
        ("milestone", "wrong", "artifact_date_or_milestone_mismatch"),
        ("model_invoked", True, "current_model_invocation_mismatch"),
        ("inference_substrate_class", "wrong", "inference_substrate_class_mismatch"),
        ("inference_substrate", "wrong", "inference_substrate_mismatch"),
        ("execution_venue", "host_cpu", "execution_venue_mismatch"),
        ("positive_claim", True, "positive_claim_mismatch"),
        ("exploratory_effect_score", 1, "exploratory_effect_score_mismatch"),
        ("restart_parity_score", 0, "restart_parity_score_mismatch"),
        ("verdict_class", "positive", "verdict_class_mismatch"),
    )
    for field, replacement, error in mutations:
        changed = deepcopy(fixture)
        changed[field] = replacement
        with pytest.raises(ValueError, match=error):
            exp.validate_artifact(changed, root=tmp_path, verify_sidecars=False)
    changed = deepcopy(fixture)
    changed["independent_reduction"]["support_passed"] = False
    with pytest.raises(ValueError, match="independent_reduction_mismatch"):
        exp.validate_artifact(changed, root=tmp_path, verify_sidecars=False)
    changed = deepcopy(fixture)
    changed["field_principles"].pop("rows")
    with pytest.raises(ValueError, match="field_principles_incomplete"):
        exp.validate_artifact(changed, root=tmp_path, verify_sidecars=False)

    missing_terminal = exp.build_test_artifact(
        tmp_path / "missing-terminal", validation_receipts=_receipts(terminal=False)
    )
    with pytest.raises(ValueError, match="terminal_validation_missing_or_failed"):
        exp.validate_artifact(missing_terminal, root=tmp_path / "missing-terminal")

    positive = exp.build_test_artifact(
        tmp_path / "positive", validation_receipts=_receipts(), effect=True
    )
    assert positive["verdict_class"] == "circular_positive"
    assert positive["exploratory_effect_score"] == 1
    assert exp.validate_artifact(positive, root=tmp_path / "positive")["effect_passed"] is True


def test_sidecar_hash_size_and_row_count_fail_closed(tmp_path: Path) -> None:
    """REQ-CL-7549 raw custody rejects every independent byte boundary."""

    fixture = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    receipt = deepcopy(fixture["prediction_update_rows"])
    receipt["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="raw_sidecar_hash_mismatch"):
        exp._read_sidecar(tmp_path, receipt, "prediction_update_rows")
    receipt = deepcopy(fixture["prediction_update_rows"])
    receipt["bytes"] += 1
    with pytest.raises(ValueError, match="raw_sidecar_size_mismatch"):
        exp._read_sidecar(tmp_path, receipt, "prediction_update_rows")
    receipt = deepcopy(fixture["prediction_update_rows"])
    receipt["rows"] += 1
    with pytest.raises(ValueError, match="raw_sidecar_row_count_mismatch"):
        exp._read_sidecar(tmp_path, receipt, "prediction_update_rows")


def test_valid_null_is_complete_but_failed_validation_is_disqualified(tmp_path: Path) -> None:
    """REQ-CL-7549; SCENARIO-CL-7549-ARTIFACT."""

    null = exp.build_test_artifact(tmp_path / "null", validation_receipts=_receipts())
    assert null["verdict_class"] == "null"
    assert null["honest_verdict"].startswith("complete_null_")
    assert null["count_measurement_complete_score"] == 1
    assert null["exploratory_effect_score"] == 0

    failed = exp.build_test_artifact(
        tmp_path / "failed",
        validation_receipts=[
            _receipt(name, passed=name != "focused_pytest")
            for name in (*validation_scope.REQUIRED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
        ],
    )
    assert failed["verdict_class"] == "disqualified"
    assert failed["honest_verdict"].startswith("complete_disqualified_")
    assert failed["count_measurement_complete_score"] == 0
    with pytest.raises(ValueError, match="required_validation_failed"):
        exp.validate_artifact(failed, root=tmp_path / "failed", verify_sidecars=True)


def test_blocked_artifact_names_exact_upstream_failure() -> None:
    """REQ-CL-7549 blocks external absence without fabricated rows."""

    failed = {
        "check": "exp7547_cached_stream_ready",
        "upstream": "exp7547-count-stream",
        "path": "results/experiment_7547_v660_count_stream.json",
        "field": "cached_stream_ready_score",
        "expected": 1,
        "observed": None,
        "passed": False,
        "required": True,
    }
    value = exp.build_blocked_artifact(failed, [failed], duration_s=0.1)
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["gate_check_summary"]["first_failure"] == failed
    assert value["rows"] == []
    assert value["count_measurement_complete_score"] == 0
    assert value["exploratory_effect_score"] == 0


def test_real_preconditions_authenticate_upstream_and_sidecars() -> None:
    """REQ-CL-7549 checks exact same-milestone inputs before compute."""

    checks = exp.collect_preconditions(ROOT)
    assert checks
    assert all(row["passed"] for row in checks if row["required"])
    upstream = next(row for row in checks if row["check"] == "exp7547_cached_stream_ready")
    assert upstream["path"] == "results/experiment_7547_v660_count_stream.json"
    assert upstream["observed"] == 1
    assert any(row["check"] == "host_cpu_available" for row in checks)
    inputs = exp.load_measurement_inputs(ROOT)
    assert len(inputs["public"]["online"]) == 159
    assert len(inputs["retention"]) == 116
    assert inputs["frozen"]["protocol_hash"]


def test_input_protocol_and_retention_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7549 rejects changed upstream protocol and retention rosters."""

    original_freeze = exp.stream.freeze_public_protocol
    monkeypatch.setattr(exp.stream, "freeze_public_protocol", lambda _public: {})
    with pytest.raises(ValueError, match="frozen_protocol_rebuild_mismatch"):
        exp.load_measurement_inputs(ROOT)
    monkeypatch.setattr(exp.stream, "freeze_public_protocol", original_freeze)
    monkeypatch.setattr(exp, "load_jsonl", lambda _path: [])
    with pytest.raises(ValueError, match="retention_roster_mismatch"):
        exp.load_measurement_inputs(ROOT)


def test_cold_replay_and_cli_readers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-CL-7549; SCENARIO-CL-7549-ARTIFACT fresh-process API."""

    value = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert exp.cold_replay(path, root=tmp_path)["measurement_complete"] is True
    assert exp.main(["--cold-replay", str(path), "--root", str(tmp_path)]) == 0
    assert "cold_replay_passed" in capsys.readouterr().out
    assert exp.main(["--independent-reduce", str(path), "--root", str(tmp_path)]) == 0
    assert "independent_reduction_passed" in capsys.readouterr().out
    args = exp.parse_args(["--date", "20260923"])
    assert args.date == "20260923"


def test_wrong_date_and_unreadable_reader_fail_closed(tmp_path: Path) -> None:
    """REQ-CL-7549 rejects wrong identity before scientific reduction."""

    with pytest.raises(ValueError, match="run_date_must_equal_20260923"):
        exp.main(["--date", "20260922"])
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="artifact_unreadable_or_not_object"):
        exp.cold_replay(missing, root=tmp_path)


def test_declared_wrapper_is_thin_and_real_artifact_is_strict_when_present() -> None:
    """REQ-CL-7549 command wiring and terminal evidence stay independently readable."""

    wrapper = (ROOT / exp.WRAPPER_PATH).read_text(encoding="utf-8")
    assert "experiment_7549_v660_count_learning import main" in wrapper
    assert "raise SystemExit(main())" in wrapper

    result = ROOT / exp.RESULT_PATH
    if result.is_file():
        value = json.loads(result.read_text(encoding="utf-8"))
        reduction = exp.validate_artifact(value, root=ROOT, verify_sidecars=True)
        assert reduction["measurement_complete"] is True
        assert value["run_date"] == "20260923"
        assert value["sample_size_budget"]["independent_online_groups"]["completed"] == 159
        assert value["sample_size_budget"]["retention_groups"]["completed"] == 116
        assert value["bootstrap_replicates"] == 1000
        assert value["confirmatory_benefit_score"] == 0


def test_gate_rows_have_operands_and_principles(tmp_path: Path) -> None:
    """REQ-CL-7549 keeps readiness, validity, and benefit independently reducible."""

    value = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    assert all(
        {"check", "category", "expected", "observed", "op", "passed", "principle"} <= set(row)
        for row in value["acceptance_gate_results"]
    )
    assert set(value) <= set(value["field_principles"])
    assert value["verifier_is_oracle"] is False
    assert value["flagged_adversarial"] is False
    assert value["positive_claim"] is False
    assert value["no_headroom"] is False


def test_validation_command_manifest_and_replay_helpers(tmp_path: Path) -> None:
    """REQ-CL-7549 freezes command scope and current source identities."""

    commands = exp.build_validation_commands(ROOT, tmp_path / "private")
    assert {command.name for command in commands} == set(validation_scope.REQUIRED_CHECK_NAMES)
    terminal = exp.terminal_commands(tmp_path / "candidate.json", ROOT)
    assert [command.name for command in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert all(command.timeout_s == 300.0 for command in terminal)

    manifest = tmp_path / "manifest.json"
    exp._write_affected_manifest(manifest)
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_value["test_paths"] == [exp.TEST_PATH.as_posix()]
    hashes = exp._source_hashes(ROOT)
    assert exp.MODULE_PATH.as_posix() in hashes

    summary = exp._replay_summary(
        {
            "seed": 1,
            "released_source_count": 8,
            "censored_source_count": 12,
            "changed_label_binding_count": 4,
            "chronology_violation_count": 0,
            "restart_mismatch_count": 0,
            "duplicate_feedback_rejection_count": 1,
            "retention_state_mutation_count": 0,
            "update_timing": {},
            "persistence_timing": {},
            "final_state_hash": "sha256:" + "0" * 64,
            "disposition": "complete",
        }
    )
    assert summary["released_source_count"] == 8
    assert set(summary) == {
        "seed",
        "released_source_count",
        "censored_source_count",
        "changed_label_binding_count",
        "chronology_violation_count",
        "restart_mismatch_count",
        "duplicate_feedback_rejection_count",
        "retention_state_mutation_count",
        "update_timing",
        "persistence_timing",
        "final_state_hash",
        "disposition",
    }

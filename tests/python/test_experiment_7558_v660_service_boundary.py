"""Tests for REQ-REPORT-7558 and SCENARIO-REPORT-7558-* contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7558_v660_service_boundary as exp


def test_preconditions_authenticate_prototype_and_four_board_records() -> None:
    """REQ-REPORT-7558 reads exact sources before dependent measurement."""

    context = exp.collect_preconditions(exp.REPO_ROOT)

    assert context["ready"] is True
    assert context["blocker"] is None
    assert context["prototype"]["count_memory_ready_score"] == 1
    assert set(exp.BOARD_PATHS) <= set(context["source_artifact_hashes"])
    assert all(row["passed"] for row in context["rows"])


def test_durable_trial_exposes_equal_semantics_and_reloads(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7558-DURABILITY measures the full service boundary."""

    events = exp.frozen_events(8)
    one = exp.measure_policy_trial("batch_one", 0, events, tmp_path / "one")
    batch = exp.measure_policy_trial("batch_256", 0, events, tmp_path / "batch")

    required_segments = {
        "predict_ns",
        "release_update_ns",
        "serialize_ns",
        "write_ns",
        "file_fsync_ns",
        "rename_ns",
        "directory_fsync_ns",
        "reload_ns",
        "full_service_ns",
    }
    assert required_segments <= set(one["latency_segments_ns"])
    assert required_segments <= set(batch["latency_segments_ns"])
    assert one["event_sequence_sha256"] == batch["event_sequence_sha256"]
    assert one["acknowledged_event_count"] == batch["acknowledged_event_count"] == 8
    assert one["reconstruction_parity"] is True
    assert batch["reconstruction_parity"] is True
    assert one["durability_semantics"] == batch["durability_semantics"]
    assert batch["ack_latency_includes_batch_delay"] is True
    assert one["bytes_written"] > batch["bytes_written"]


def test_crash_panel_preserves_exactly_once_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7558-CRASH kills only owned children at three boundaries."""

    panel = exp.run_crash_panel(tmp_path)

    assert [row["boundary"] for row in panel] == [
        "pre_commit",
        "post_file_fsync",
        "post_rename",
    ]
    assert all(row["owned_child_only"] for row in panel)
    assert all(row["lost_acknowledged_event_count"] == 0 for row in panel)
    assert all(row["duplicate_update_count"] == 0 for row in panel)
    assert all(row["passed"] for row in panel)


def test_board_rows_preserve_fabric_cpu_and_physical_scopes() -> None:
    """SCENARIO-REPORT-7558-BOARDS forbids a fresh board operation."""

    context = exp.collect_preconditions(exp.REPO_ROOT)
    rows, summary = exp.build_board_rows(context["board_artifacts"], None)

    assert [row["board"] for row in rows] == ["KV260", "PolarFire", "GateMate"]
    assert rows[0]["exact_claim_scope"] == "historical_kv260_fpga_fabric_sampling_only"
    assert rows[1]["exact_claim_scope"] == (
        "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
    )
    assert rows[2]["current_disposition"] == "blocked_unchanged_physical_prerequisite"
    assert rows[2]["gate_check_summary"]["observed"] == 0
    assert summary["board_continuity_complete_score"] == 1
    assert all(row["hardware_operations_issued"] == [] for row in rows)


def test_paired_summary_and_amdahl_use_measured_service_denominator() -> None:
    """REQ-REPORT-7558 forbids imported device or vendor speedup ratios."""

    rows = []
    for trial in range(30):
        for policy, service, update in (
            ("batch_one", 1_000 + trial, 100 + trial),
            ("batch_256", 500 + trial, 80 + trial),
        ):
            rows.append(
                {
                    "policy": policy,
                    "trial": trial,
                    "latency_segments_ns": {
                        "full_service_ns": service,
                        "release_update_ns": update,
                    },
                }
            )

    summary = exp.summarize_service_rows(rows, seed=7558001)

    assert summary["paired_trial_count"] == 30
    assert summary["batch_one_median_full_service_ns"] == pytest.approx(1014.5)
    assert summary["batch_256_median_full_service_ns"] == pytest.approx(514.5)
    assert summary["paired_difference_interval_ns"]["lower"] <= 500
    assert summary["paired_difference_interval_ns"]["upper"] >= 500
    assert summary["hardware_acceleration_bound"]["assumed_component_speedup"] is None
    assert summary["hardware_acceleration_bound"]["ideal_update_only_speedup"] < 2
    assert summary["hardware_acceleration_bound"]["denominator"] == "measured_full_service_ns"


def test_complete_fixture_reduces_and_mutations_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7558-VALIDATION keeps readiness separate from benefit."""

    artifact = exp.build_fixture_artifact(tmp_path)
    reduction = exp.independent_reduce(artifact)

    assert reduction["service_cost_complete_score"] == 1
    assert reduction["board_continuity_complete_score"] == 1
    assert artifact["positive_claim"] is False
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert exp.validate_artifact(artifact, verify_sources=False) == []

    mutated = deepcopy(artifact)
    mutated["service_rows"][0]["reconstruction_parity"] = False
    assert "independent_reduction_mismatch" in exp.validate_artifact(mutated, verify_sources=False)

    mutated = deepcopy(artifact)
    mutated["hardware_acceleration_bound"]["assumed_component_speedup"] = 100
    assert "imported_speedup_forbidden" in exp.validate_artifact(mutated, verify_sources=False)

    mutated = deepcopy(artifact)
    mutated["invocation_counts"]["model_loads"]["attempted"] = 1
    assert "current_inference_declaration_invalid" in exp.validate_artifact(
        mutated, verify_sources=False
    )


def test_blocked_artifact_names_exact_missing_upstream(tmp_path: Path) -> None:
    """REQ-REPORT-7558 makes missing external prerequisites terminal blocked."""

    blocker = {
        "check": "prototype_ready",
        "upstream": "Exp7534",
        "path": exp.COUNT_ARTIFACT_PATH.as_posix(),
        "field": "count_memory_ready_score",
        "expected": 1,
        "observed": None,
    }
    artifact = exp.build_blocked_artifact(blocker, tmp_path)

    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == blocker
    assert artifact["sample_size_budget"]["service_trials"]["unstarted"] == 60
    assert exp.validate_artifact(artifact, verify_sources=False) == []


def test_serialized_candidate_cold_replay_and_cli_modes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7558-VALIDATION exercises both fresh-reader modes."""

    artifact = exp.build_fixture_artifact(tmp_path)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(path), "--no-source-check"]) == 0
    assert (
        exp.main(["--date", exp.RUN_DATE, "--independent-reduce", str(path), "--no-source-check"])
        == 0
    )
    assert '"errors": []' in capsys.readouterr().out

    path.write_text("[]", encoding="utf-8")
    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(path)]) == 1


def test_validation_manifest_and_terminal_commands_use_exact_scope(tmp_path: Path) -> None:
    """REQ-REPORT-7558 freezes private scoped validation and strict readers."""

    commands = exp.build_validation_commands(tmp_path / "private")
    terminal = exp.terminal_commands(tmp_path / "candidate.json")

    by_name = {row.name: row for row in commands}
    assert set(by_name) == set(exp.validation_scope.REQUIRED_CHECK_NAMES)
    assert exp.TEST_PATH.as_posix() in by_name["focused_pytest"].argv
    assert "--fail-under=100" in by_name["changed_module_coverage_report"].argv
    assert [row.spec.name for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert "--strict" in terminal[-1].spec.argv


def test_invalid_policy_and_source_labels_fail_or_remain_explicit(tmp_path: Path) -> None:
    """REQ-REPORT-7558 rejects an undeclared policy and preserves source identity."""

    with pytest.raises(ValueError, match="unknown_policy"):
        exp.measure_policy_trial("unknown", 0, exp.frozen_events(1), tmp_path / "bad")

    source = tmp_path / "source.json"
    source.write_text('{"honest_verdict":"complete_null_fixture"}\n', encoding="utf-8")
    row = exp.source_row(source, tmp_path / "different-root")
    assert row["path"] == str(source.resolve())
    assert row["original_honest_verdict"] == "complete_null_fixture"


def test_validator_reports_each_fail_closed_contract(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7558-VALIDATION rejects misleading terminal mutations."""

    artifact = exp.build_fixture_artifact(tmp_path)

    mutations = [
        (
            "vendor_ratio_imported",
            lambda value: value["hardware_acceleration_bound"].update(vendor_ratio_imported=True),
        ),
        (
            "board_continuity_complete_score_mismatch",
            lambda value: value.update(board_continuity_complete_score=0),
        ),
        ("terminal_verdict_prefix_invalid", lambda value: value.update(honest_verdict="null")),
        ("verdict_class_invalid", lambda value: value.update(verdict_class="unknown")),
        ("field_principles_incomplete", lambda value: value["field_principles"].pop("rows")),
        (
            "gate_principle_missing",
            lambda value: value["acceptance_gate_results"][0].update(principle=""),
        ),
    ]
    for expected, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert expected in exp.validate_artifact(changed, verify_sources=False)

    changed = deepcopy(artifact)
    changed["external_blocker"] = {"check": "missing"}
    assert "external_blocker_incomplete" in exp.validate_artifact(changed, verify_sources=False)
    assert "blocked_class_invalid" in exp.validate_artifact(changed, verify_sources=False)

    changed = deepcopy(artifact)
    changed["board_rows"] = [1]
    assert "independent_reduction_mismatch" in exp.validate_artifact(changed, verify_sources=False)

    source = tmp_path / "bound.json"
    source.write_text("{}\n", encoding="utf-8")
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = {"bound": exp.source_row(source, tmp_path)}
    changed["source_artifact_hashes"]["bound"]["sha256"] = "sha256:" + "0" * 64
    assert "source_hash_invalid:bound" in exp.validate_artifact(changed, root=tmp_path)

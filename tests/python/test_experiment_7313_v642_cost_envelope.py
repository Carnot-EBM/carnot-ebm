"""Behavior tests for REQ-CL-7313 and SCENARIO-CL-7313-*.

The tests use the shipped immutable measurement because this task is a reducer.
They do not create a second implementation-shaped timing fixture.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_7313_v642_cost_envelope as envelope


@pytest.fixture(scope="module")
def authenticated() -> tuple[list[dict[str, object]], dict[str, object]]:
    """SCENARIO-CL-7313-PRECONDITIONS authenticates the real historical inputs."""

    return envelope.collect_preconditions(envelope.REPO_ROOT, envelope.ExperimentPaths.defaults())


@pytest.fixture(scope="module")
def reduced(
    authenticated: tuple[list[dict[str, object]], dict[str, object]],
) -> dict[str, object]:
    """SCENARIO-CL-7313-E2E reduces the immutable measured rows once."""

    checks, evidence = authenticated
    assert envelope.gate_summary(checks)["passed"] is True
    return envelope.reduce_cost_envelope(
        evidence["exp7299"], evidence["raw_rows"], evidence["exp7270"]
    )


def test_req_cl_7313_freezes_aggregation_only_contract() -> None:
    """REQ-CL-7313 fixes the no-model substrate and original seed population."""

    spec = (envelope.REPO_ROOT / envelope.SPEC_RELATIVE).read_text(encoding="utf-8")
    assert "REQ-CL-7313" in spec
    assert all(f"SCENARIO-CL-7313-{name}" in spec for name in envelope.SCENARIOS)
    assert envelope.MODEL_SPECS == []
    assert envelope.MODEL_INVOKED is False
    assert set(envelope.INVOCATION_COUNTS.values()) == {0}
    assert envelope.INFERENCE_SUBSTRATE == "aggregation_from_upstream_artifacts"
    assert envelope.INFERENCE_SUBSTRATE_CLASS == "aggregation"
    assert envelope.EXECUTION_VENUE == "host"
    assert envelope.EVALUATION_SEEDS == tuple(range(7299001, 7299009))


def test_scenario_cl_7313_preconditions_preserve_exact_external_failure(
    authenticated: tuple[list[dict[str, object]], dict[str, object]], tmp_path: Path
) -> None:
    """SCENARIO-CL-7313-PRECONDITIONS keeps exact failure fields and no rows."""

    checks, evidence = authenticated
    assert envelope.gate_summary(checks)["passed"] is True
    assert evidence["artifact_hashes"]["exp7299"] == envelope.EXPECTED_EXP7299_SHA256
    assert evidence["artifact_hashes"]["exp7270"] == envelope.EXPECTED_EXP7270_SHA256
    assert evidence["artifact_hashes"]["raw_rows"] == envelope.EXPECTED_RAW_ROWS_SHA256

    failed, _ = envelope.collect_preconditions(
        envelope.REPO_ROOT,
        envelope.ExperimentPaths.under(tmp_path),
        exp7299_path=tmp_path / "missing.json",
    )
    first = next(row for row in failed if row["passed"] is False)
    blocked = envelope.blocked_artifact_for_test(first)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == blocked["cost_bound_rows"] == []
    assert blocked["gate_check_summary"]["checks"][0] == first
    assert blocked["gate_check_summary"]["artifact_field"] == first["field"]
    assert envelope.validate_artifact(blocked) == []


def test_scenario_cl_7313_reconstruction_counts_shared_timers_once(
    reduced: dict[str, object],
) -> None:
    """SCENARIO-CL-7313-RECONSTRUCTION prevents group-stage double counting."""

    rows = reduced["cost_bound_rows"]
    burst = next(
        row for row in rows if row["seed"] == 7299001 and row["arrival_process"] == "burst"
    )
    assert burst["durable_group_count"] == 16
    assert burst["shared_group_timer_policy"] == "count_once_per_group"
    assert burst["sync_deduplicated_ns"] < burst["sync_latency_area_ns"]
    assert burst["native_update_deduplicated_ns"] < burst["native_update_latency_area_ns"]
    assert burst["serialization_identifiability"] == "bounded_not_point_identified"
    assert burst["serialization_lower_ns"] <= burst["serialization_upper_ns"]
    assert burst["queue_wait_class"] == "overlapping_latency_area_not_amdahl_work"
    assert burst["acknowledgment_class"] == "overlapping_latency_area_not_amdahl_work"
    assert burst["irreducible_floor_ns"] == max(
        burst["sync_deduplicated_ns"] + burst["native_update_deduplicated_ns"],
        burst["arrival_span_ns"],
    )
    assert burst["counterfactual_s_max"] == pytest.approx(
        burst["measured_total_ns"] / burst["irreducible_floor_ns"]
    )


def test_scenario_cl_7313_uncertainty_keeps_arrivals_and_original_bounds_separate(
    reduced: dict[str, object],
) -> None:
    """SCENARIO-CL-7313-UNCERTAINTY resamples paired seeds without pooling."""

    summaries = reduced["arrival_process_summaries"]
    assert set(summaries) == {"burst", "steady_paced", "interactive_dependent"}
    assert all(row["paired_seed_count"] == 8 for row in summaries.values())
    assert all(row["bootstrap_draws"] == 10_000 for row in summaries.values())
    assert all(row["bootstrap_unit"] == "paired_seed" for row in summaries.values())
    assert len(reduced["cost_bound_rows"]) == 24
    assert reduced["original_gate_observations"]["warm_group16_lower_ci95"] == pytest.approx(
        1.4558337555069685
    )
    assert reduced["original_gate_observations"]["cold_group16_lower_ci95"] == pytest.approx(
        1.5549724849050095
    )
    assert reduced["nfr01_assessment"]["passed"] is False
    assert reduced["nfr01_assessment"]["target_speedup"] == 10.0


def test_scenario_cl_7313_envelope_is_a_bound_not_an_implemented_speedup(
    reduced: dict[str, object],
) -> None:
    """SCENARIO-CL-7313-ENVELOPE preserves durability and labels assumptions."""

    for summary in reduced["arrival_process_summaries"].values():
        assert summary["counterfactual_only"] is True
        assert summary["implemented_speedup"] is False
        assert summary["durability_acknowledgment_contract_changed"] is False
        assert summary["counterfactual_s_max_ci95"][0] >= 1.0
    assert reduced["next_mechanism_warrant"]["warranted"] is False
    assert reduced["next_mechanism_warrant"]["technique"] is None
    assert reduced["decision"]["action"] == "defer"
    assert "exclusive" in reduced["decision"]["changed_prerequisite"]
    assert reduced["historical_context"]["exp7270_declared_bottleneck"] == "durable_sync"
    assert reduced["historical_context"]["exp7270_journal_optimization_warranted_score"] == 0


def test_scenario_cl_7313_e2e_rejects_changed_or_incomplete_rows(
    authenticated: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    """SCENARIO-CL-7313-E2E rejects row changes and missing paired evidence."""

    _, evidence = authenticated
    changed_raw = dict(evidence["raw_rows"])
    changed_raw["rows"] = list(changed_raw["rows"])
    changed_raw["rows"][0] = {**changed_raw["rows"][0], "sync_ns": 0}
    with pytest.raises(ValueError, match="row_hash"):
        envelope.reduce_cost_envelope(evidence["exp7299"], changed_raw, evidence["exp7270"])

    incomplete_raw = dict(evidence["raw_rows"])
    incomplete_raw["per_run_results"] = list(incomplete_raw["per_run_results"][:-1])
    with pytest.raises(ValueError, match="population"):
        envelope.reduce_cost_envelope(evidence["exp7299"], incomplete_raw, evidence["exp7270"])


def test_scenario_cl_7313_reducer_fail_closed_controls(
    authenticated: tuple[list[dict[str, object]], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7313-E2E rejects each parity and timer boundary independently."""

    _, evidence = authenticated

    def changed(group: str, predicate, **values: object) -> dict[str, object]:
        raw = dict(evidence["raw_rows"])
        selected = list(raw[group])
        index = next(index for index, row in enumerate(selected) if predicate(row))
        selected[index] = envelope._finish_row({**selected[index], **values})
        raw[group] = selected
        return raw

    sqlite_group16 = lambda row: (
        row["max_group_size"] == 16
        and row["storage_arm"] == "sqlite_persist"
        and row["seed"] == 7299001
        and row["arrival_process"] == "burst"
    )
    atomic_group16 = lambda row: (
        row["max_group_size"] == 16
        and row["storage_arm"] == "atomic_replace"
        and row["seed"] == 7299001
        and row["arrival_process"] == "burst"
    )

    no_group16 = dict(evidence["raw_rows"])
    no_group16["rows"] = [
        row for row in no_group16["rows"] if not (sqlite_group16(row) and row["event_index"] == 0)
    ]
    with pytest.raises(ValueError, match="group16_population"):
        envelope.reduce_cost_envelope(evidence["exp7299"], no_group16, evidence["exp7270"])

    missing_pair = changed("per_run_results", atomic_group16, seed=9999999)
    with pytest.raises(ValueError, match="paired_population"):
        envelope.reduce_cost_envelope(evidence["exp7299"], missing_pair, evidence["exp7270"])

    wrong_pair = changed("per_run_results", atomic_group16, event_stream_sha256="sha256:changed")
    with pytest.raises(ValueError, match="paired_input_parity"):
        envelope.reduce_cost_envelope(evidence["exp7299"], wrong_pair, evidence["exp7270"])

    moved_event = changed("rows", sqlite_group16, seed=7299002)
    with pytest.raises(ValueError, match="event_population"):
        envelope.reduce_cost_envelope(evidence["exp7299"], moved_event, evidence["exp7270"])

    bad_ack = changed("per_run_results", sqlite_group16, exact_native_state_parity=False)
    with pytest.raises(ValueError, match="acknowledgment_or_recovery"):
        envelope.reduce_cost_envelope(evidence["exp7299"], bad_ack, evidence["exp7270"])

    bad_groups = changed("per_run_results", sqlite_group16, durable_group_count=999)
    with pytest.raises(ValueError, match="durable_group_count"):
        envelope.reduce_cost_envelope(evidence["exp7299"], bad_groups, evidence["exp7270"])

    bad_phase = changed("phase_cost_rows", sqlite_group16, sync_ns=0)
    with pytest.raises(ValueError, match="phase_cost_parity"):
        envelope.reduce_cost_envelope(evidence["exp7299"], bad_phase, evidence["exp7270"])

    bad_floor = changed("per_run_results", sqlite_group16, elapsed_ns=1)
    with pytest.raises(ValueError, match="irreducible_floor"):
        envelope.reduce_cost_envelope(evidence["exp7299"], bad_floor, evidence["exp7270"])

    monkeypatch.setattr(envelope, "EXPECTED_WARM_LOWER", 0.0)
    with pytest.raises(ValueError, match="original_gate_reconstruction"):
        envelope.reduce_cost_envelope(
            evidence["exp7299"], evidence["raw_rows"], evidence["exp7270"]
        )

    first_group = [
        row
        for row in evidence["raw_rows"]["rows"]
        if sqlite_group16(row) and row["group_id"] == "group-0001"
    ][:2]
    first_group[1] = {**first_group[1], "sync_ns": first_group[1]["sync_ns"] + 1}
    with pytest.raises(ValueError, match="shared_group_timer_mismatch"):
        envelope._stage_totals(first_group)


def test_req_cl_7313_helper_and_orchestrator_paths(
    authenticated: tuple[list[dict[str, object]], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-CL-7313 covers malformed input, receipts, blocking, retry, and publication."""

    nonobject = tmp_path / "nonobject.json"
    nonobject.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        envelope._read_json(nonobject)
    monkeypatch.setattr(
        envelope, "artifact_checksum", lambda _value: (_ for _ in ()).throw(TypeError("bad"))
    )
    assert envelope._checksum_valid({}) is False
    monkeypatch.setattr(envelope, "artifact_checksum", envelope.exp7299.artifact_checksum)

    commands = envelope._validation_commands()
    assert {name for name, _, _ in commands} == {
        "focused_pytest",
        "affected_suites",
        "full_python_suite",
        "scoped_100_percent_coverage",
        "ruff_check",
        "ruff_format",
        "changed_module_mypy",
        "scoped_spec_coverage",
    }
    monkeypatch.setattr(
        envelope.exp7299,
        "_stream_subprocess",
        lambda command, **_kwargs: {"exit_code": 0, "output": "wrapped", "command": command},
    )
    assert (
        envelope._stream_subprocess(
            ["true"], root=envelope.REPO_ROOT, operation="test", timeout_s=1
        )["exit_code"]
        == 0
    )

    failed = envelope.check("missing", "upstream", "field", 1, None, False)
    paths = envelope.ExperimentPaths.under(tmp_path / "blocked")
    monkeypatch.setattr(envelope, "collect_preconditions", lambda *_args, **_kwargs: ([failed], {}))
    blocked = envelope.run_experiment(
        envelope.REPO_ROOT, paths.artifact, envelope.RUN_DATE, paths=paths
    )
    assert blocked["status"] == "blocked"
    assert paths.artifact.is_file()

    with pytest.raises(ValueError, match="run date"):
        envelope.run_experiment(envelope.REPO_ROOT, paths.artifact, "bad", paths=paths)

    checks, evidence = authenticated
    monkeypatch.setattr(
        envelope, "collect_preconditions", lambda *_args, **_kwargs: (checks, evidence)
    )
    monkeypatch.setattr(envelope, "_validation_commands", lambda: [("owned", ["owned"], 1)])
    monkeypatch.setattr(
        envelope,
        "_stream_subprocess",
        lambda *_args, **_kwargs: {"exit_code": 1, "output": "failed"},
    )
    retry_paths = envelope.ExperimentPaths.under(tmp_path / "retry")
    with pytest.raises(RuntimeError, match="validation failed"):
        envelope.run_experiment(
            envelope.REPO_ROOT, retry_paths.artifact, envelope.RUN_DATE, paths=retry_paths
        )
    assert retry_paths.checkpoint.is_file()

    monkeypatch.setattr(envelope, "_validation_commands", lambda: [])
    candidate_paths = envelope.ExperimentPaths.under(tmp_path / "candidate-fail")
    with pytest.raises(RuntimeError, match="candidate validation failed"):
        envelope.run_experiment(
            envelope.REPO_ROOT,
            candidate_paths.artifact,
            envelope.RUN_DATE,
            paths=candidate_paths,
        )
    assert candidate_paths.checkpoint.is_file()

    monkeypatch.setattr(
        envelope,
        "_stream_subprocess",
        lambda *_args, **_kwargs: {"exit_code": 0, "output": "passed"},
    )
    original_validator = envelope.validate_artifact
    invalid_paths = envelope.ExperimentPaths.under(tmp_path / "invalid-candidate")
    monkeypatch.setattr(envelope, "validate_artifact", lambda _artifact: ["bad"])
    with pytest.raises(ValueError, match="invalid Exp7313 candidate"):
        envelope.run_experiment(
            envelope.REPO_ROOT,
            invalid_paths.artifact,
            envelope.RUN_DATE,
            paths=invalid_paths,
        )

    final_calls = iter(([], ["bad"]))
    monkeypatch.setattr(envelope, "validate_artifact", lambda _artifact: next(final_calls))
    invalid_final_paths = envelope.ExperimentPaths.under(tmp_path / "invalid-final")
    with pytest.raises(ValueError, match="invalid validated Exp7313"):
        envelope.run_experiment(
            envelope.REPO_ROOT,
            invalid_final_paths.artifact,
            envelope.RUN_DATE,
            paths=invalid_final_paths,
        )

    monkeypatch.setattr(envelope, "validate_artifact", original_validator)
    monkeypatch.setattr(
        envelope,
        "_validation_commands",
        lambda: [("full_python_suite", ["full"], 1)],
    )
    success_paths = envelope.ExperimentPaths.under(tmp_path / "success")
    artifact = envelope.run_experiment(
        envelope.REPO_ROOT,
        success_paths.artifact,
        envelope.RUN_DATE,
        paths=success_paths,
    )
    assert artifact["status"] == "complete"
    assert success_paths.artifact.is_file()


def test_req_cl_7313_main_run_and_error_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CL-7313 returns explicit codes for malformed validation and run errors."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[", encoding="utf-8")
    assert envelope.main(["--validate", str(malformed)]) == 2

    calls: list[tuple[Path, Path, str]] = []
    monkeypatch.setattr(
        envelope,
        "run_experiment",
        lambda root, output, date: calls.append((root, output, date)) or {},
    )
    assert envelope.main(["--output", "results/test-exp7313.json"]) == 0
    assert calls[0][1] == envelope.REPO_ROOT / "results/test-exp7313.json"
    monkeypatch.setattr(
        envelope,
        "run_experiment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    assert envelope.main(["--output", str(tmp_path / "failed.json")]) == 2


def test_scenario_cl_7313_terminal_artifact_and_validator(
    authenticated: tuple[list[dict[str, object]], dict[str, object]],
    reduced: dict[str, object],
) -> None:
    """SCENARIO-CL-7313-TERMINAL separates complete reduction from speed value."""

    checks, evidence = authenticated
    artifact = envelope.assemble_complete_artifact(checks, evidence, reduced, [])
    assert artifact["cost_envelope_complete_score"] == 1
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null")
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert not any(artifact["invocation_counts"].values())
    assert envelope.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["cost_bound_rows"][0]["counterfactual_s_max"] += 1.0
    changed["reproducibility_checksum"] = envelope.artifact_checksum(changed)
    assert "independent_reduction" in envelope.validate_artifact(changed)

    bad_counts = deepcopy(artifact)
    bad_counts["invocation_counts"]["attempted_generations"] = 1
    bad_counts["reproducibility_checksum"] = envelope.artifact_checksum(bad_counts)
    assert "invocation_counts" in envelope.validate_artifact(bad_counts)


def test_req_cl_7313_thin_entrypoint_and_read_only_cli(
    monkeypatch: pytest.MonkeyPatch,
    authenticated: tuple[list[dict[str, object]], dict[str, object]],
    reduced: dict[str, object],
    tmp_path: Path,
) -> None:
    """REQ-CL-7313 keeps the wrapper thin and validation read-only."""

    script = envelope.REPO_ROOT / "scripts/experiments/experiment_7313_v642_cost_envelope.py"
    calls: list[object] = []
    original_main = envelope.main
    monkeypatch.setattr(envelope, "main", lambda argv=None: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", [str(script), "--date", envelope.RUN_DATE])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    assert calls == [None]
    assert len(script.read_text(encoding="utf-8").splitlines()) <= 12
    monkeypatch.setattr(envelope, "main", original_main)

    checks, evidence = authenticated
    artifact = envelope.assemble_complete_artifact(checks, evidence, reduced, [])
    candidate = tmp_path / "candidate.json"
    envelope.atomic_write(candidate, artifact)
    before = candidate.read_bytes()
    assert envelope.main(["--validate", str(candidate)]) == 0
    assert candidate.read_bytes() == before
    candidate.write_text("{}\n", encoding="utf-8")
    assert envelope.main(["--validate", str(candidate)]) == 2
    assert envelope.main(["--date", "bad", "--output", str(candidate)]) == 2

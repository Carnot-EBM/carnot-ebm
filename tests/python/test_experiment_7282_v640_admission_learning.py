"""Verify prospective learning with independent admission.

Spec refs: REQ-CL-7282 and SCENARIO-CL-7282-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
import time as real_time
from types import SimpleNamespace

import pytest

from carnot import experiment_7282_v640_admission_learning as exp


@pytest.fixture(scope="module")
def one_stream_panel(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[exp.ExperimentPaths, exp.LearningPanel]:
    """Run one sealed stream once for row-level contract tests."""

    paths = exp.ExperimentPaths.under(tmp_path_factory.mktemp("exp7282-panel"))
    upstream = exp._load_object(exp.REPO_ROOT / exp.DEFAULT_UPSTREAM_ARTIFACT)
    views = exp.load_sealed_views(exp.REPO_ROOT, upstream)
    panel = exp.run_learning_panel(
        views,
        paths,
        stream_ids=("prospective-01",),
        progress=False,
    )
    return paths, panel


@pytest.fixture(scope="module")
def one_stream_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[exp.ExperimentPaths, dict[str, object]]:
    """Build one terminal candidate for schema and file checks."""

    paths = exp.ExperimentPaths.under(tmp_path_factory.mktemp("exp7282-artifact"))
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        stream_ids=("prospective-01",),
        bootstrap_draws=128,
        progress=False,
    )
    return paths, artifact


def test_req_cl_7282_fixes_contract_and_no_model_work() -> None:
    """REQ-CL-7282 fixes arms, samples, quotas, bounds, and CPU provenance."""

    spec = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-CL-7282" in spec
    assert len(set(exp.SCENARIO_PATTERN.findall(spec))) == 9
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert exp.INVOCATION_COUNTS == {
        "attempted_model_loads": 0,
        "completed_model_loads": 0,
        "attempted_generation_calls": 0,
        "completed_generation_calls": 0,
        "usable_answers": 0,
    }
    assert exp.ARMS == exp.fixture.ARMS
    assert (exp.STREAM_COUNT, exp.EVENTS_PER_STREAM, exp.WARMUP_COUNT) == (24, 1_024, 128)
    assert exp.BOOTSTRAP_RESAMPLES == 10_000
    assert exp.MEASUREMENT_LIMIT_S == 900
    assert exp.EXPECTED_EVENT_ROWS == 172_032
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(exp.FIELD_PRINCIPLES)


def test_scenario_cl_7282_preconditions_authenticate_fixture(tmp_path: Path) -> None:
    """SCENARIO-CL-7282-PRECONDITIONS authenticates every sealed input."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, hashes, upstream = exp.collect_preconditions(exp.REPO_ROOT, paths)

    assert exp.gate_summary(checks)["passed"] is True
    assert upstream["admission_fixture_ready_score"] == 1
    assert (
        hashes[str(exp.REPO_ROOT / exp.DEFAULT_UPSTREAM_ARTIFACT)] == exp.EXPECTED_UPSTREAM_SHA256
    )
    assert any(row["check"] == "sealed_stream_receipts" for row in checks)


def test_scenario_cl_7282_preconditions_external_failure_is_blocked(tmp_path: Path) -> None:
    """SCENARIO-CL-7282-PRECONDITIONS maps an absent upstream to terminal blocked."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, hashes, _ = exp.collect_preconditions(
        exp.REPO_ROOT,
        paths,
        upstream_path=tmp_path / "missing.json",
    )
    artifact = exp.build_blocked_artifact(
        checks,
        hashes,
        ("prospective-01",),
        started_at="2026-09-13T00:00:00+00:00",
        duration_s=0.1,
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert str(artifact["honest_verdict"]).startswith("blocked_")
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["first_failure"] is not None
    assert exp.validate_artifact(artifact, expected_stream_ids=("prospective-01",)) == []


def test_scenario_cl_7282_prequential_rows_are_complete_and_ordered(
    one_stream_panel: tuple[exp.ExperimentPaths, exp.LearningPanel],
) -> None:
    """SCENARIO-CL-7282-PREQUENTIAL retains every prediction before release."""

    paths, panel = one_stream_panel
    assert len(panel.event_rows) == len(exp.ARMS) * exp.EVENTS_PER_STREAM
    assert exp.prequential_row_errors(panel.event_rows, ("prospective-01",)) == []
    assert {row["segment"] for row in panel.event_rows} == {
        "warmup",
        "initial_future",
        "shift",
        "recurrence",
    }
    assert all(row["prediction_frozen_before_release"] is True for row in panel.event_rows)
    assert all(row["learner_read_private_authority"] is False for row in panel.event_rows)
    assert paths.prequential_rows.exists()


def test_scenario_cl_7282_common_candidate_and_opportunity_denominators(
    one_stream_panel: tuple[exp.ExperimentPaths, exp.LearningPanel],
) -> None:
    """SCENARIO-CL-7282-OPPORTUNITIES retains harm, misses, zero gain, and costs."""

    _, panel = one_stream_panel
    diagnostic = exp.run_common_candidate_diagnostic(panel.opportunity_rows)
    summaries = exp.build_opportunity_summary_rows(panel.opportunity_rows)

    assert diagnostic["passed"] is True
    assert diagnostic["checked_opportunity_count"] == exp.MAX_OPPORTUNITIES
    assert exp.opportunity_row_errors(panel.opportunity_rows, ("prospective-01",)) == []
    assert len(panel.opportunity_rows) == len(exp.ARMS) * exp.MAX_OPPORTUNITIES
    assert len([row for row in summaries if row["opportunity_index"] == "overall"]) == len(exp.ARMS)
    assert all(row["opportunity_denominator"] > 0 for row in summaries)
    assert all(row["acquisition_label_cost"] == 8 for row in panel.opportunity_rows)
    assert all(row["admission_label_cost"] == 8 for row in panel.opportunity_rows)
    assert all(row["pending_queue_bytes"] > 0 for row in panel.opportunity_rows)
    assert all(row["update_decision_cost"] == 1 for row in panel.opportunity_rows)
    assert (
        sum(
            row["zero_available_gain_count"]
            for row in summaries
            if row["opportunity_index"] == "overall"
        )
        >= 0
    )


def test_scenario_cl_7282_cold_reducer_and_fixed_bootstrap(
    one_stream_panel: tuple[exp.ExperimentPaths, exp.LearningPanel],
) -> None:
    """SCENARIO-CL-7282-COMPARISONS cold-reduces rows and fixed paired intervals."""

    paths, panel = one_stream_panel
    reduced, opportunities = exp.independent_reduce(
        paths.prequential_rows,
        paths.opportunity_rows,
    )
    comparisons = exp.build_comparison_rows(reduced, draws=128)

    assert reduced == panel.rows
    assert opportunities == exp.build_opportunity_summary_rows(panel.opportunity_rows)
    assert comparisons == exp.build_comparison_rows(reduced, draws=128)
    assert {row["comparison_id"] for row in comparisons} == {
        item[0] for item in exp.COMPARISON_SPECS
    }
    assert all(row["independent_unit_count"] == 1 for row in comparisons)
    assert all(row["bootstrap_resamples"] == 128 for row in comparisons)
    with pytest.raises(ValueError, match="raw_rows_unavailable"):
        exp.independent_reduce(paths.prequential_rows.with_name("absent"), paths.opportunity_rows)


def test_scenario_cl_7282_science_gates_require_admission_and_later_change() -> None:
    """SCENARIO-CL-7282-CAUSAL rejects zero-admission success and classifies nulls."""

    comparisons = []
    for comparison_id, metric, treatment, control in exp.COMPARISON_SPECS:
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "metric": metric,
                "treatment_arm": treatment,
                "control_arm": control,
                "stratum": "overall",
                "independent_unit_count": 24,
                "estimate": -0.03,
                "ci95_upper": -0.01,
            }
        )
    causal = {
        "paired_admission_count": 0,
        "admitted_state_change_count": 0,
        "later_changed_prediction_count": 0,
        "chronology_violation_count": 0,
        "quota_or_memory_violation_count": 0,
    }
    gates = exp.score_acceptance_gates(comparisons, causal)
    value, verdict_class, verdict = exp.classify_result(gates)

    assert gates["admitted_causal_change"]["passed"] is False
    assert value == 0
    assert verdict_class == "null"
    assert verdict.startswith("complete_null")
    causal.update(
        paired_admission_count=1,
        admitted_state_change_count=1,
        later_changed_prediction_count=1,
    )
    passing = exp.score_acceptance_gates(comparisons, causal)
    assert exp.classify_result(passing)[:2] == (1, "circular_positive")


def test_scenario_cl_7282_e2e_covers_journal_admission_and_reduction(tmp_path: Path) -> None:
    """SCENARIO-CL-7282-E2E covers the full ordered learning lifecycle."""

    rows = exp.run_e2e_controls(tmp_path)

    assert [row["stage"] for row in rows] == [
        "complete_stream",
        "ordered_prediction_release_journal",
        "common_candidate_admission",
        "future_error_rows",
        "opportunity_denominator",
        "cold_evaluator_recomputation",
    ]
    assert all(row["passed"] is True for row in rows)


def test_scenario_cl_7282_terminal_candidate_is_complete_not_partial(
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7282-TERMINAL keeps completion separate from value."""

    paths, artifact = one_stream_artifact
    assert artifact["status"] == "complete"
    assert artifact["admission_run_complete_score"] == 1
    assert artifact["admission_value_score"] in {0, 1}
    assert artifact["verdict_class"] in {"null", "circular_positive"}
    assert str(artifact["honest_verdict"]).startswith("complete_")
    assert artifact["sample_size_budget"]["completed_event_arm_rows"] == 7_168
    assert (
        exp.validate_artifact(
            artifact,
            expected_stream_ids=("prospective-01",),
            check_files=True,
        )
        == []
    )
    exp.write_artifact(
        paths.artifact,
        artifact,
        expected_stream_ids=("prospective-01",),
    )
    assert json.loads(paths.artifact.read_text())["status"] == "complete"


def test_validation_rejects_tampering_and_bad_receipts(
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
) -> None:
    """REQ-CL-7282 rejects altered chronology, classes, and receipt schemas."""

    _, artifact = one_stream_artifact
    changed = deepcopy(artifact)
    changed["verdict_class"] = "positive"
    assert "complete_contract" in exp.validate_artifact(
        changed,
        expected_stream_ids=("prospective-01",),
    )
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "checksum" in exp.validate_artifact(
        changed,
        expected_stream_ids=("prospective-01",),
    )
    with pytest.raises(ValueError, match="validation_receipt_schema"):
        exp.attach_validation_receipts(artifact, [{"command": "bad"}])
    receipt = {
        "command": "focused check",
        "exit_code": 0,
        "classification": "passed",
        "duration_s": 0.1,
        "log_sha256": "sha256:" + "a" * 64,
    }
    attached = exp.attach_validation_receipts(artifact, [receipt])
    assert attached["validation_receipts"] == [receipt]
    assert attached["reproducibility_checksum"] == exp.reproducibility_checksum(attached)


def test_entrypoint_is_thin_and_private_modes_use_private_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7282 keeps script execution thin and test outputs isolated."""

    wrapper = exp.REPO_ROOT / exp.WRAPPER_PATH
    text = wrapper.read_text(encoding="utf-8")
    assert "experiment_7282_v640_admission_learning" in text
    assert "main" in text

    candidate_paths = exp.ExperimentPaths.under(tmp_path / "candidate")
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        candidate_paths,
        stream_ids=("prospective-01",),
        bootstrap_draws=32,
        progress=False,
    )
    exp._atomic_write(candidate_paths.terminal_candidate, exp._canonical_bytes(artifact))
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--validate",
                "--artifact-path",
                str(candidate_paths.terminal_candidate),
                "--stream-id",
                "prospective-01",
            ]
        )
        == 0
    )
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--e2e-worker",
                "--output-root",
                str(tmp_path / "worker"),
            ]
        )
        == 0
    )
    with pytest.raises(SystemExit, match="run_date_must_be"):
        exp.main(["--date", "20260912", "--output-root", str(tmp_path / "bad")])
    with pytest.raises(SystemExit, match="artifact_path_required"):
        exp.main(["--date", exp.RUN_DATE, "--validate"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(wrapper),
            "--date",
            exp.RUN_DATE,
            "--e2e-worker",
            "--output-root",
            str(tmp_path / "runpy"),
        ],
    )
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert raised.value.code == 0


def test_defensive_row_and_stream_checks_fail_closed(
    one_stream_panel: tuple[exp.ExperimentPaths, exp.LearningPanel],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-CL-7282 rejects changed seals, rows, candidates, and paired inputs."""

    _, panel = one_stream_panel
    upstream = exp._load_object(exp.REPO_ROOT / exp.DEFAULT_UPSTREAM_ARTIFACT)
    with monkeypatch.context() as patch:
        patch.setattr(exp.fixture, "stream_conformance_errors", lambda *_: ["changed"])
        with pytest.raises(ValueError, match="sealed_stream_conformance"):
            exp.load_sealed_views(exp.REPO_ROOT, upstream)

    diagnostic_rows = [deepcopy(panel.opportunity_rows[0]), deepcopy(panel.opportunity_rows[1])]
    diagnostic_rows[1]["candidate_state_hash"] = "sha256:" + "0" * 64
    assert exp.run_common_candidate_diagnostic(diagnostic_rows)["passed"] is False

    invalid_events = [deepcopy(panel.event_rows[0])]
    invalid_events[0]["prediction_frozen_before_release"] = False
    assert exp.prequential_row_errors(invalid_events, ("prospective-01",))
    invalid_opportunities = [deepcopy(panel.opportunity_rows[0])]
    invalid_opportunities[0]["pending_queue_bytes"] = 0
    assert exp.opportunity_row_errors(invalid_opportunities, ("prospective-01",))

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_rows_unavailable"):
        exp._read_jsonl(empty)
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_rows_unavailable"):
        exp._read_jsonl(malformed)
    with pytest.raises(ValueError, match="paired_streams_unavailable"):
        exp._bootstrap_interval([], 10, "absent")

    bad_events = tmp_path / "bad-events.jsonl"
    bad_opportunities = tmp_path / "bad-opportunities.jsonl"
    bad_events.write_text(json.dumps(panel.event_rows[0]) + "\n", encoding="utf-8")
    bad_opportunities.write_text(json.dumps(panel.opportunity_rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_row_conformance"):
        exp.independent_reduce(bad_events, bad_opportunities)


def test_checkpoint_resume_censoring_heartbeat_and_corruption(
    one_stream_panel: tuple[exp.ExperimentPaths, exp.LearningPanel],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CL-7282 resumes complete units and accounts for bounded failures."""

    paths, panel = one_stream_panel
    upstream = exp._load_object(exp.REPO_ROOT / exp.DEFAULT_UPSTREAM_ARTIFACT)
    views = exp.load_sealed_views(exp.REPO_ROOT, upstream)
    clock = iter((0.0, 0.0, 61.0))
    fake_time = SimpleNamespace(
        monotonic=lambda: next(clock),
        perf_counter_ns=real_time.perf_counter_ns,
    )
    with monkeypatch.context() as patch:
        patch.setattr(exp, "time", fake_time)
        resumed = exp.run_learning_panel(
            views,
            paths,
            stream_ids=("prospective-01",),
            progress=True,
        )
    assert resumed.rows == panel.rows
    assert "benchmark heartbeat" in capsys.readouterr().out

    censored = exp.run_learning_panel(
        views,
        exp.ExperimentPaths.under(tmp_path / "censored"),
        stream_ids=("prospective-01",),
        measurement_limit_s=-1,
    )
    assert censored.completed_stream_ids == []
    assert censored.censored_stream_ids == ["prospective-01"]

    broken_paths = exp.ExperimentPaths.under(tmp_path / "broken-events")
    exp._atomic_write(
        broken_paths.stream_shards / "prospective-01.json",
        exp._canonical_bytes(
            {
                "schema": exp.SCHEMA,
                "status": "complete",
                "event_rows": [],
                "opportunity_rows": panel.opportunity_rows,
            }
        ),
    )
    with pytest.raises(ValueError, match="prequential_conformance"):
        exp.run_learning_panel(views, broken_paths, stream_ids=("prospective-01",))

    broken_paths = exp.ExperimentPaths.under(tmp_path / "broken-opportunities")
    exp._atomic_write(
        broken_paths.stream_shards / "prospective-01.json",
        exp._canonical_bytes(
            {
                "schema": exp.SCHEMA,
                "status": "complete",
                "event_rows": panel.event_rows,
                "opportunity_rows": [],
            }
        ),
    )
    with pytest.raises(ValueError, match="opportunity_conformance"):
        exp.run_learning_panel(views, broken_paths, stream_ids=("prospective-01",))

    with monkeypatch.context() as patch:
        patch.setattr(
            exp,
            "run_common_candidate_diagnostic",
            lambda _: {
                "passed": False,
                "ran_before_benchmark": False,
                "checked_opportunity_count": 0,
            },
        )
        with pytest.raises(ValueError, match="common_candidate_diagnostic_failed"):
            exp.run_learning_panel(
                views,
                exp.ExperimentPaths.under(tmp_path / "bad-diagnostic"),
                stream_ids=("prospective-01",),
            )


def test_causal_summary_counts_a_changed_admission(
    one_stream_panel: tuple[exp.ExperimentPaths, exp.LearningPanel],
) -> None:
    """SCENARIO-CL-7282-CAUSAL follows an admitted change into later predictions."""

    _, panel = one_stream_panel
    opportunities = deepcopy(panel.opportunity_rows)
    row = next(value for value in opportunities if value["arm"] == "paired_gated")
    row["decision"] = "accept"
    row["admitted_state_change"] = True
    causal = exp.causal_summary(panel.event_rows, opportunities)

    assert causal["paired_admission_count"] > 0
    assert causal["admitted_state_change_count"] > 0
    assert causal["later_changed_prediction_count"] >= 0


def test_build_progress_block_and_internal_consistency_failures(
    one_stream_panel: tuple[exp.ExperimentPaths, exp.LearningPanel],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CL-7282 reports progress and refuses internal evidence mismatches."""

    _, known_panel = one_stream_panel
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        exp.ExperimentPaths.under(tmp_path / "progress"),
        stream_ids=("prospective-01",),
        bootstrap_draws=16,
        progress=True,
    )
    assert artifact["status"] == "complete"
    assert "BEFORE model load" in capsys.readouterr().out

    failed_check = exp.gate_check("missing", "upstream", "field", True, False)
    with monkeypatch.context() as patch:
        patch.setattr(
            exp, "collect_preconditions", lambda *_args, **_kwargs: ([failed_check], {}, {})
        )
        blocked = exp.build_and_seal(
            exp.REPO_ROOT,
            exp.ExperimentPaths.under(tmp_path / "blocked"),
            stream_ids=("prospective-01",),
            progress=True,
        )
    assert blocked["status"] == "blocked"

    with monkeypatch.context() as patch:
        patch.setattr(
            exp,
            "run_common_candidate_diagnostic",
            lambda _: {"passed": False, "ran_before_benchmark": True},
        )
        with pytest.raises(ValueError, match="common_candidate_diagnostic_failed"):
            exp.build_and_seal(
                exp.REPO_ROOT,
                exp.ExperimentPaths.under(tmp_path / "diagnostic-failure"),
                stream_ids=("prospective-01",),
                bootstrap_draws=16,
            )

    original_reduce = exp.independent_reduce
    with monkeypatch.context() as patch:
        patch.setattr(
            exp,
            "independent_reduce",
            lambda *args: ([], original_reduce(*args)[1]),
        )
        with pytest.raises(ValueError, match="cold_evaluator_reducer_mismatch"):
            exp.build_and_seal(
                exp.REPO_ROOT,
                exp.ExperimentPaths.under(tmp_path / "reduce-failure"),
                stream_ids=("prospective-01",),
                bootstrap_draws=16,
            )

    def censored_panel(
        views: object,
        paths: exp.ExperimentPaths,
        **kwargs: object,
    ) -> exp.LearningPanel:
        del views, kwargs
        exp._atomic_write(
            paths.prequential_rows, exp.fixture.prototype.jsonl_bytes(known_panel.event_rows)
        )
        exp._atomic_write(
            paths.opportunity_rows,
            exp.fixture.prototype.jsonl_bytes(known_panel.opportunity_rows),
        )
        exp._atomic_write(paths.diagnostic_rows, b"{}\n")
        return exp.LearningPanel(
            known_panel.event_rows,
            known_panel.opportunity_rows,
            known_panel.rows,
            known_panel.diagnostic,
            ["prospective-01"],
            ["prospective-01"],
            known_panel.maximum_memory_bytes,
        )

    with monkeypatch.context() as patch:
        patch.setattr(exp, "run_learning_panel", censored_panel)
        incomplete = exp.build_and_seal(
            exp.REPO_ROOT,
            exp.ExperimentPaths.under(tmp_path / "censored-build"),
            stream_ids=("prospective-01",),
            bootstrap_draws=16,
        )
    assert incomplete["admission_run_complete_score"] == 0
    assert str(incomplete["honest_verdict"]).startswith("complete_null")

    with monkeypatch.context() as patch:
        patch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["forced"])
        with pytest.raises(ValueError, match="artifact_validation_failed"):
            exp.build_and_seal(
                exp.REPO_ROOT,
                exp.ExperimentPaths.under(tmp_path / "validation-failure"),
                stream_ids=("prospective-01",),
                bootstrap_draws=16,
            )


def test_validation_and_main_defensive_orchestration(
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-CL-7282 validates private modes and retains failed command receipts."""

    _, artifact = one_stream_artifact
    unknown = deepcopy(artifact)
    unknown["status"] = "unfinished"
    unknown["reproducibility_checksum"] = exp.reproducibility_checksum(unknown)
    assert "status" in exp.validate_artifact(unknown, expected_stream_ids=("prospective-01",))

    missing_raw = deepcopy(artifact)
    missing_raw["prequential_rows_path"] = str(tmp_path / "absent.jsonl")
    missing_raw["reproducibility_checksum"] = exp.reproducibility_checksum(missing_raw)
    assert "cold_reducer" in exp.validate_artifact(
        missing_raw,
        expected_stream_ids=("prospective-01",),
        check_files=True,
    )
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.write_artifact(
            tmp_path / "invalid.json",
            unknown,
            expected_stream_ids=("prospective-01",),
        )

    commands = exp._validation_commands(tmp_path / "candidate.json")
    assert any("adversarial_verify.py" in command for row in commands for command in row)
    affected = next(row for row in commands if any("::test_" in item for item in row))
    for selector in (item for item in affected if "::test_" in item):
        path, node = selector.split("::", 1)
        assert f"def {node}(" in (exp.REPO_ROOT / path).read_text(encoding="utf-8")

    failed_check = exp.gate_check("missing", "upstream", "field", True, False)
    blocked = exp.build_blocked_artifact(
        [failed_check],
        {},
        ("prospective-01",),
        started_at="2026-09-13T00:00:00+00:00",
        duration_s=0.1,
    )
    with monkeypatch.context() as patch:
        patch.setattr(exp, "build_and_seal", lambda *_args, **_kwargs: blocked)
        assert (
            exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "main-blocked")]) == 0
        )

    writes: list[Path] = []
    with monkeypatch.context() as patch:
        patch.setattr(exp, "build_and_seal", lambda *_args, **_kwargs: deepcopy(artifact))
        patch.setattr(exp, "_validation_commands", lambda _candidate: [])
        patch.setattr(
            exp,
            "write_artifact",
            lambda path, _artifact: writes.append(path),
        )
        assert (
            exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "main-complete")])
            == 0
        )
    assert writes

    failure_receipt = {
        "command": "forced failure",
        "exit_code": 1,
        "classification": "failed",
        "duration_s": 0.1,
        "log_sha256": "sha256:" + "f" * 64,
    }
    with monkeypatch.context() as patch:
        patch.setattr(exp, "build_and_seal", lambda *_args, **_kwargs: deepcopy(artifact))
        patch.setattr(exp, "_validation_commands", lambda _candidate: [["forced"]])
        patch.setattr(exp.fixture, "_command_receipt", lambda _command: failure_receipt)
        with pytest.raises(RuntimeError, match="focused_validation_failed"):
            exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "main-failed")])

    bad_candidate = tmp_path / "bad-candidate.json"
    bad_candidate.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="artifact_validation_failed"):
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--validate",
                "--artifact-path",
                str(bad_candidate),
                "--stream-id",
                "prospective-01",
            ]
        )

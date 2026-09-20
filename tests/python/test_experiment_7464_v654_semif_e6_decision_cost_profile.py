"""Tests for the SEMIF E6 decision-cost reducer.

Spec refs: REQ-ARC-WMTE-7464 and SCENARIO-ARC-WMTE-7464-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from carnot import experiment_7464_v654_semif_e6_decision_cost_profile as exp


def _raw_unit(game: str, seed: int, condition: str = "shadow") -> dict:
    return {
        "episode_id": f"{game}:seed-{seed}:{condition}",
        "game": game,
        "seed": seed,
        "condition": condition,
        "disposition": "complete",
        "censored": False,
        "episode_elapsed_s": 10.0,
        "action_count": 4,
        "selected_action_ids": ["RESET", "ACTION1", "ACTION6"],
        "start_level": 0,
        "terminal_level": 0,
        "banked_progress": 0,
        "request_spans": [
            {
                "request_id": "r0",
                "start_monotonic_s": 1.0,
                "end_monotonic_s": 3.0,
                "input_tokens": 40,
                "output_tokens": 5,
                "total_tokens": 45,
                "usable": False,
                "gpu_prompt_ms": 400.0,
                "gpu_generation_ms": 1500.0,
            },
            {
                "request_id": "r1",
                "start_monotonic_s": 2.0,
                "end_monotonic_s": 4.0,
                "input_tokens": 50,
                "output_tokens": 5,
                "total_tokens": 55,
                "usable": False,
                "gpu_prompt_ms": 500.0,
                "gpu_generation_ms": 1400.0,
            },
        ],
        "supervisor": {
            "mode": condition,
            "actions_observed": 3,
            "window": 120,
            "arms_enabled": [
                "drop_goal_bias",
                "allow_reinduction",
                "force_exploration_diversity",
            ],
            "selected_arms": ["drop_goal_bias"],
            "full_eligible_candidate_set_observed": False,
        },
        "model_identity": {
            "repository": "unsloth/Qwen3.8-27B-GGUF",
            "revision": "fixture-revision",
            "quantization": "Q4_K_M",
        },
    }


def _fixture_units() -> list[dict]:
    return [
        _raw_unit("bp35", 7464001, "shadow"),
        _raw_unit("bp35", 7464001, "applied"),
        _raw_unit("bp35", 7464002, "shadow"),
        _raw_unit("bp35", 7464002, "applied"),
        _raw_unit("cn04", 7464001, "shadow"),
        _raw_unit("cn04", 7464001, "applied"),
        _raw_unit("cn04", 7464002, "shadow"),
        _raw_unit("cn04", 7464002, "applied"),
    ]


def test_overlap_safe_cost_reduction_keeps_unknown_stage_time_missing() -> None:
    """REQ-ARC-WMTE-7464: nested spans count once and unknown time stays null."""

    reduced = exp.reduce_raw_units(_fixture_units(), bootstrap_seed=7464, bootstrap_draws=64)

    assert reduced["episode_summaries"][0]["measured_generation_s"] == pytest.approx(3.0)
    assert reduced["episode_summaries"][0]["unclassified_residual_s"] == pytest.approx(7.0)
    assert reduced["stage_attribution"]["trace_time_fraction"] == pytest.approx(0.3)
    assert reduced["stage_attribution"]["model_token_fraction"] == pytest.approx(1.0)
    assert len(reduced["rows"]) == 8 * len(exp.SEAMS)

    candidate = next(row for row in reduced["rows"] if row["seam"] == "candidate_action_selection")
    generation = next(row for row in reduced["rows"] if row["seam"] == "downstream_generation")
    hypothesis = next(row for row in reduced["rows"] if row["seam"] == "hypothesis_gate")
    assert candidate["elapsed_s"] is None
    assert candidate["candidate_option_presence"] == "selected_only"
    assert generation["elapsed_s"] == pytest.approx(3.0)
    assert generation["input_tokens"] == 90
    assert generation["output_tokens"] == 10
    assert hypothesis["invocation_status"] == "uninvoked"
    assert hypothesis["elapsed_s"] == 0.0


def test_replaceable_bounds_and_game_cluster_interval_are_sample_limited() -> None:
    """SCENARIO-ARC-WMTE-7464-BOUNDS: residual widens bounds by game cluster."""

    reduced = exp.reduce_raw_units(_fixture_units(), bootstrap_seed=7464, bootstrap_draws=128)
    bounds = reduced["replaceable_share_bounds"]
    interval = reduced["game_cluster_interval"]

    assert bounds["lower"] == 0.0
    assert bounds["upper"] == pytest.approx(0.7)
    assert bounds["perfect_removal_speedup_lower"] == 1.0
    assert bounds["perfect_removal_speedup_upper"] == pytest.approx(10.0 / 3.0)
    assert interval["cluster_count"] == 2
    assert interval["sample_limited"] is True
    assert interval["broad_transfer_claim_allowed"] is False
    assert reduced["sample_size_budget"] == {
        "planned_independent_units": 8,
        "attempted_independent_units": 8,
        "complete_independent_units": 8,
        "failed_independent_units": 0,
        "censored_independent_units": 0,
        "unstarted_independent_units": 0,
        "independent_game_clusters": 2,
    }


def test_invalid_episode_time_and_incomplete_token_usage_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-7464-SEAM-ACCOUNTING: invalid spans cannot create shares."""

    invalid = _raw_unit("bp35", 1)
    invalid["request_spans"][1]["end_monotonic_s"] = 12.0
    with pytest.raises(ValueError, match="measured_time_exceeds_episode"):
        exp.reduce_raw_units([invalid], bootstrap_seed=1, bootstrap_draws=8)

    incomplete = _raw_unit("bp35", 1)
    incomplete["request_spans"][0]["input_tokens"] = None
    reduced = exp.reduce_raw_units([incomplete], bootstrap_seed=1, bootstrap_draws=8)
    generation = next(row for row in reduced["rows"] if row["seam"] == "downstream_generation")
    assert generation["input_tokens"] is None
    assert reduced["stage_attribution"]["model_token_fraction"] is None


def test_positive_control_recovers_injected_delay_and_tokens() -> None:
    """REQ-ARC-WMTE-7464: the registered timing and token control must pass."""

    control = exp.run_positive_control()

    assert control["passed"] is True
    assert control["observed_generation_s"] == 3.0
    assert control["observed_input_tokens"] == 90
    assert control["observed_output_tokens"] == 10


def test_observation_spec_names_missing_events_and_candidate_sets() -> None:
    """SCENARIO-ARC-WMTE-7464-OBSERVATION-SPEC: Exp7471 gets exact telemetry."""

    spec = exp.build_seam_observation_spec()

    assert spec["target_experiment"] == "exp7471"
    assert spec["environment_or_model_calls_authorized"] is False
    by_seam = {row["seam"]: row for row in spec["seams"]}
    assert "candidate_set" in by_seam["candidate_action_selection"]["missing_events"]
    assert "stable_candidate_id" in by_seam["hypothesis_gate"]["required_candidate_fields"]
    assert "no_redirect" in by_seam["supervisor_arm_selection"]["required_candidate_ids"]
    assert "induce_now" in by_seam["induction_timing"]["required_candidate_ids"]
    assert "parent_decision_id" in spec["cross_cutting_fields"]


def test_ladder_stays_closed_when_e0_or_e6_gates_fail() -> None:
    """REQ-ARC-WMTE-7464: both reports exist, but failed gates prevent queuing."""

    e0 = {
        "status": "complete_scored_runtime_unavailable",
        "verdict_class": "blocked",
        "local_runtime_parity_score": 0,
        "scored_runtime_parity_score": 0,
    }
    reduction = exp.reduce_raw_units(_fixture_units(), bootstrap_seed=7464, bootstrap_draws=32)
    disposition = exp.build_ladder_disposition(e0, reduction)

    assert disposition["e0_report_present"] is True
    assert disposition["e6_report_complete"] is True
    assert disposition["queue_authorized"] is False
    by_rung = {row["rung"]: row for row in disposition["rungs"]}
    assert by_rung["E7"]["decision"] == "insufficient_evidence"
    assert by_rung["E8"]["decision"] == "stop"
    assert by_rung["E12"]["decision"] == "stop"


def test_test_artifact_is_schema_complete_and_mutations_are_rejected() -> None:
    """SCENARIO-ARC-WMTE-7464-TERMINAL: cold validation recomputes raw claims."""

    artifact = exp.build_artifact_for_test()

    assert exp.validate_artifact(artifact, require_validation=False) == []
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["decision_profile_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["ladder_disposition"]["queue_authorized"] is False
    assert artifact["small_ebm_training"]["performed"] is False
    assert set(artifact["producer_code_hashes"]) == {
        exp.MODULE_PATH.as_posix(),
        exp.WRAPPER_PATH.as_posix(),
        exp.TEST_PATH.as_posix(),
    }
    assert set(artifact["field_principles"]) == set(artifact)

    changed = deepcopy(artifact)
    changed["rows"][0]["elapsed_s"] = 999.0
    assert "stored_reduction_mismatch" in exp.validate_artifact(changed, require_validation=False)

    missing_disposition = deepcopy(artifact)
    missing_disposition["rows"][0].pop("censoring")
    missing_disposition["reproducibility_checksum"] = exp.artifact_checksum(missing_disposition)
    assert "row_disposition_incomplete" in exp.validate_artifact(
        missing_disposition, require_validation=False
    )


def test_validation_plan_is_affected_only_and_uses_private_paths(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7464-TERMINAL: no full-suite command enters the manifest."""

    commands = exp.build_validation_plan(exp.REPO_ROOT, tmp_path)

    assert exp.validate_validation_plan(exp.REPO_ROOT, commands) == []
    assert {command.name for command in commands} == set(exp.AFFECTED_CHECK_NAMES)
    assert all("tests/python -q" not in " ".join(command.argv) for command in commands)
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert "-n" in focused.argv and "0" in focused.argv
    assert "--no-cov" in focused.argv
    assert any(argument.startswith("--basetemp=") for argument in focused.argv)
    coverage_report = next(
        command for command in commands if command.name == "changed_module_coverage_report"
    )
    assert "COVERAGE_FILE" in dict(coverage_report.command_environment)


def test_source_authentication_and_cold_extraction_preserve_history() -> None:
    """SCENARIO-ARC-WMTE-7464-SOURCE-CLASSIFICATION: cold inputs stay typed."""

    checks, hashes, artifacts = exp.collect_preconditions(exp.REPO_ROOT)
    units = exp.extract_raw_units(exp.REPO_ROOT, artifacts["v653"])

    assert checks and all(row["passed"] for row in checks)
    assert hashes[exp.V653_RESULT.as_posix()]["original_flags"]["model_class"] == (
        "current_qwen3_8_archived"
    )
    assert hashes[exp.FLAGGED_RESULT.as_posix()]["original_flags"]["quantitative_use"] == (
        "excluded"
    )
    assert len(units) == 8
    assert all(len(row["request_spans"]) == 2 for row in units)
    assert sum(row["banked_progress"] for row in units) == 0
    assert units[0]["request_spans"][0]["input_tokens"] == 10337
    assert units[0]["model_identity"]["repository"] == "unsloth/Qwen3.8-27B-GGUF"


def test_fail_closed_boundaries_and_empty_panel_are_explicit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7464: malformed spans, sources, and identities fail closed."""

    assert exp._load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert exp._load_object(array) == {}
    object_path = tmp_path / "object.json"
    object_path.write_text('{"ok": true}', encoding="utf-8")
    assert exp._load_object(object_path) == {"ok": True}

    with pytest.raises(ValueError, match="negative_request_span"):
        exp._union_duration([{"start_monotonic_s": 2.0, "end_monotonic_s": 1.0}])
    assert exp._union_duration([{"start_monotonic_s": None, "end_monotonic_s": 1.0}]) == 0.0
    with pytest.raises(ValueError, match="empty_percentile"):
        exp._percentile([], 0.5)
    assert exp._percentile([2.0], 0.5) == 2.0
    with pytest.raises(ValueError, match="duplicate_episode_id"):
        exp.reduce_raw_units(
            [_raw_unit("bp35", 1), _raw_unit("bp35", 1)],
            bootstrap_seed=1,
            bootstrap_draws=0,
        )
    invalid_elapsed = _raw_unit("bp35", 2)
    invalid_elapsed["episode_elapsed_s"] = -1.0
    with pytest.raises(ValueError, match="invalid_episode_elapsed"):
        exp.reduce_raw_units([invalid_elapsed], bootstrap_seed=1, bootstrap_draws=1)
    empty = exp.reduce_raw_units([], bootstrap_seed=1, bootstrap_draws=0)
    assert empty["stage_attribution"]["trace_time_fraction"] is None
    assert empty["game_cluster_interval"]["upper_share_ci95"] is None

    assert "+00:00" in exp.utc_now()
    exp.progress(time.monotonic(), "test", "boundary", units=1)
    assert "phase=test" in capsys.readouterr().out


def test_cold_extractor_accepts_relative_sidecars_and_missing_gpu_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7464: response ledgers may be relative and timing may be absent."""

    raw = tmp_path / "raw"
    raw.mkdir()
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                "timings": {},
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "rows": [
            {
                "episode_id": "g:seed-1:shadow",
                "game": "g",
                "seed": 1,
                "condition": "shadow",
                "disposition": "failed",
                "elapsed_s": 1.0,
                "action_count": 1,
                "action_rows": [{"action": "RESET"}],
                "start_level": 0,
                "terminal_level": 0,
                "banked_progress": 0,
                "request_budget_receipt": {
                    "callback_rows": [
                        {
                            "request_id": "r",
                            "reserved_monotonic": 0.0,
                            "terminal_monotonic": 1.0,
                        }
                    ]
                },
                "server_request_rows": [
                    {
                        "call_index": 0,
                        "response_path": "response.json",
                        "request_path": "request.json",
                    }
                ],
                "supervisor_receipt": {"would_have_redirects": []},
            }
        ]
    }
    (raw / "episode_rows.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(exp, "V653_RAW", Path("raw"))

    units = exp.extract_raw_units(
        tmp_path,
        {"MODEL_SPECS": [{"hf_id": "model", "revision": "rev", "quantization": "q"}]},
    )

    assert units[0]["censored"] is True
    assert units[0]["request_spans"][0]["gpu_total_ms"] is None


def test_validation_rejects_schema_and_receipt_mutations() -> None:
    """SCENARIO-ARC-WMTE-7464-TERMINAL: each required receipt remains mandatory."""

    artifact = exp.build_artifact_for_test()
    artifact["schema"] = "wrong"
    artifact["invocation_counts"] = {}
    artifact["field_principles"] = {}
    artifact["decision_profile_complete_score"] = 0
    artifact["raw_unit_rows"][0]["episode_elapsed_s"] = "invalid"
    artifact["validation_receipts"] = []

    errors = exp.validate_artifact(artifact)

    assert "identity_mismatch:schema" in errors
    assert "current_invocation_counts_nonzero" in errors
    assert "field_principles_incomplete" in errors
    assert "raw_reduction_failed" in errors
    assert "decision_profile_incomplete" in errors
    assert "reproducibility_checksum_mismatch" in errors
    assert any(error.startswith("required_validation_failed:") for error in errors)


def test_terminal_plan_phase_receipt_and_argument_parser(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7464-TERMINAL: cold readers and CLI are frozen."""

    candidate = tmp_path / "candidate.json"
    commands = exp.terminal_command_specs(exp.REPO_ROOT, candidate)
    assert tuple(command.name for command in commands) == exp.TERMINAL_CHECK_NAMES
    assert str(candidate) in commands[0].argv
    span = exp._phase_span("unit", time.monotonic(), time.monotonic() - 1.0, 2)
    assert span["phase"] == "unit" and span["completed_units"] == 2
    args = exp.parse_args(["--date", exp.RUN_DATE, "--replay", str(candidate)])
    assert args.date == exp.RUN_DATE and args.replay == candidate

    broad = exp.CommandSpec(
        "focused_pytest",
        (".venv/bin/pytest", "tests/python", "-q"),
        "bad",
    )
    errors = exp.validate_validation_plan(exp.REPO_ROOT, [broad])
    assert any(error.startswith("full_suite_forbidden:") for error in errors)

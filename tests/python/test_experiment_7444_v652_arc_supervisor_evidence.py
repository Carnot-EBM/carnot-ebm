"""Tests for REQ-ARC-WMTE-7444 and SCENARIO-ARC-WMTE-7444-*.

Private fixtures exercise the reducer. They do not run an ARC environment or
write the repository's durable research record.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7444_v652_arc_supervisor_evidence as exp


def _episode(game: str = "bp35") -> dict:
    return {
        "episode_id": f"{game}:seed-7431651",
        "game": game,
        "seed": 7431651,
        "disposition": "complete",
        "actions": 62,
        "start_level": 0,
        "max_level": 0,
        "level_progress": 0,
        "tool_feedback_consumed": 0,
        "adapter_disabled": True,
        "banked_solution_disabled": True,
        "saved_engine_disabled": True,
        "request_budget_receipt": {
            "attempted": 2,
            "completed": 2,
            "failed": 0,
            "cancelled": 0,
            "in_flight": 0,
            "callback_rows": [
                {"request_id": "r1", "disposition": "completed"},
                {"request_id": "r2", "disposition": "completed"},
            ],
        },
        "supervisor": {
            "mode": "shadow",
            "arms_enabled": list(exp.DEFAULT_ENABLED_ARMS),
            "fired": 0,
            "consumed": 0,
        },
    }


def _observations(levels: list[int]) -> list[dict]:
    return [
        {
            "event": "action_observation",
            "episode_id": "bp35:seed-7431651",
            "action_index": index,
            "level": level,
            "elapsed_s": index / 10,
        }
        for index, level in enumerate(levels)
    ]


def _detailed_supervisor(*, enabled: list[str], fired: list[str], exhausted: bool = False) -> dict:
    windows = []
    if exhausted:
        windows.append(
            {
                "action_index": 600,
                "stretch_level": 0,
                "arms_enabled": list(exp.CURATED_ARMS),
                "arms_used": list(exp.CURATED_ARMS),
            }
        )
    return {
        "mode": "shadow",
        "window": exp.SHIPPED_FIRING_THRESHOLD,
        "actions_observed": 600,
        "arms_enabled": enabled,
        "would_have_redirects": [
            {"arm": arm, "action_index": 120 * (index + 1), "stretch_level": 0}
            for index, arm in enumerate(fired)
        ],
        "stagnations_unredirected": len(windows),
        "unredirected_windows": windows,
        "unredirected_windows_dropped": 0,
    }


def test_zero_firings_are_no_effect_evidence_not_a_failed_intervention() -> None:
    """SCENARIO-ARC-WMTE-7444-ZERO-FIRINGS: non-exposure cannot fail an arm."""

    row = exp.reduce_episode(_episode(), None, _observations([0] * 63))
    assert row["callbacks_attempted"] == 2
    assert row["completions_received"] == 2
    assert row["supervisor_mode"] == "shadow"
    assert row["supervisor_firings"] == 0
    assert row["consumed_redirects"] == 0
    assert row["firing_threshold_reached"] is False
    assert row["eligible_stagnation_windows"] is None
    assert row["window_evidence_status"] == "unknown_missing_timestamped_window_receipt"
    assert row["arm_effect_evidence"] == "none_no_firing"
    assert all(arm["effect_evidence"] == "none" for arm in row["arm_rows"])


def test_transient_progress_does_not_become_banked_progress_or_help() -> None:
    """SCENARIO-ARC-WMTE-7444-TRANSIENT-PROGRESS: 0->1->0 is not banked."""

    episode = _episode()
    row = exp.reduce_episode(episode, None, _observations([0, 0, 1, 1, 0]))
    assert row["level_increase_events"] == 1
    assert row["level_decrease_events"] == 1
    assert row["transient_level_progress"] == 1
    assert row["banked_progress"] == 0
    assert row["supervisor_helped_banked_progress"] is False
    assert row["solve_credit"] == 0


def test_disabled_curated_arm_prevents_all_arm_exhaustion() -> None:
    """SCENARIO-ARC-WMTE-7444-DISABLED-ARM: three arms cannot exhaust four."""

    enabled = list(exp.DEFAULT_ENABLED_ARMS)
    supervisor = _detailed_supervisor(enabled=enabled, fired=enabled, exhausted=False)
    episode = _episode()
    episode["actions"] = 600
    row = exp.reduce_episode(episode, supervisor, _observations([0] * 5))
    assert row["disabled_arms"] == ["tool_loop_reinduction"]
    assert row["all_curated_arms_fired"] is False
    assert row["new_arm_evidence"] is False
    assert row["next_live_prerequisite"]["requires_arm_enablement"] == ["tool_loop_reinduction"]


def test_all_arms_and_later_exhausted_window_are_required_for_new_arm_evidence() -> None:
    """SCENARIO-ARC-WMTE-7444-ALL-ARMS-EXHAUSTED: require both facts."""

    episode = _episode()
    episode["actions"] = 600
    all_arms = list(exp.CURATED_ARMS)
    fired_only = exp.reduce_episode(
        episode,
        _detailed_supervisor(enabled=all_arms, fired=all_arms, exhausted=False),
        _observations([0] * 5),
    )
    assert fired_only["all_curated_arms_fired"] is True
    assert fired_only["new_arm_evidence"] is False

    exhausted = exp.reduce_episode(
        episode,
        _detailed_supervisor(enabled=all_arms, fired=all_arms, exhausted=True),
        _observations([0] * 5),
    )
    assert exhausted["all_curated_arms_fired"] is True
    assert exhausted["all_arms_exhausted_windows"] == 1
    assert exhausted["new_arm_evidence"] is True
    assert exhausted["automatic_policy_change"] is False


def test_private_ledger_keeps_zero_firing_recommendations_empty() -> None:
    """REQ-ARC-WMTE-7444: the empty outcome ledger makes no policy recommendation."""

    episodes = [_episode("bp35"), _episode("cn04")]
    observations = {episode["episode_id"]: _observations([0] * 3) for episode in episodes}
    ledger = exp.reduce_private_ledger(episodes, {}, observations)
    assert ledger["episode_count"] == 2
    assert ledger["supervisor_firings"] == 0
    assert ledger["measured_failed_interventions"] == 0
    assert ledger["supervisor_recommendations"] == []
    assert ledger["new_arm_evidence"] is False
    assert ledger["banked_progress"] == 0


def test_upstream_authentication_preserves_original_flags_and_exact_games() -> None:
    """REQ-ARC-WMTE-7444: source identity and original flags fail closed."""

    source = {
        "experiment_id": "exp7431-v651-arc-live-sentinel",
        "milestone": "2026.09.651",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "status": "complete_null_arc_live_sentinel",
        "per_game_results": [_episode("bp35"), _episode("cn04")],
        "supervisor_outcomes": [
            {"episode_id": "bp35:seed-7431651", "mode": "shadow", "fired": 0},
            {"episode_id": "cn04:seed-7431651", "mode": "shadow", "fired": 0},
        ],
    }
    checks = exp.authenticate_upstream(source)
    assert all(row["passed"] for row in checks)
    assert {row["artifact_field"] for row in checks} >= {
        "experiment_id",
        "verdict_class",
        "flagged_adversarial",
        "per_game_results.games",
    }
    broken = deepcopy(source)
    broken["per_game_results"][1]["game"] = "dc22"
    assert not all(row["passed"] for row in exp.authenticate_upstream(broken))


def test_registry_precheck_requires_existing_full_clears() -> None:
    """REQ-ARC-WMTE-7444: known public targets are observations, never solve targets."""

    registry = {
        "games": [
            {"game": "bp35", "levels_reproduced": 9, "full_game_clear": True},
            {"game": "cn04", "levels_reproduced": 6, "full_game_clear": True},
        ]
    }
    checks = exp.registry_precheck(registry)
    assert all(row["passed"] for row in checks)
    registry["games"][1]["full_game_clear"] = False
    assert not all(row["passed"] for row in exp.registry_precheck(registry))


def test_sidecar_payload_keeps_archived_model_counts_out_of_current_receipt(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-7444: archived callbacks are hash-bound, not current calls."""

    source_path = tmp_path / "source.json"
    source_path.write_text("{}\n", encoding="utf-8")
    payload = exp.archived_episode_payload(
        _episode(),
        {"episode_id": "bp35:seed-7431651", "fired": 0, "mode": "shadow"},
        _observations([0, 0]),
        source_path=source_path,
    )
    assert payload["schema"] == "carnot.exp7444.archived_episode.v1"
    assert payload["scope"] == "historical_model_receipts"
    assert payload["episode_receipt"]["request_budget_receipt"]["attempted"] == 2
    receipt = exp.current_aggregation_receipt(
        started_monotonic_ns=10,
        ended_monotonic_ns=20,
        sidecar_references=[],
        phase_spans=[],
    )
    assert receipt["MODEL_SPECS"] == []
    assert receipt["model_invoked"] is False
    assert receipt["invocation_counts"] == exp.ZERO_CURRENT_INVOCATIONS
    assert receipt["inference_substrate_class"] == "aggregation"


def test_terminal_artifact_validates_and_independent_reduction_matches(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7444-TERMINAL: raw rows support the complete null."""

    rows = [
        exp.reduce_episode(_episode("bp35"), None, _observations([0, 0])),
        exp.reduce_episode(_episode("cn04"), None, _observations([0, 0])),
    ]
    artifact = exp.build_artifact_fixture(rows)
    assert exp.validate_artifact(artifact, root=tmp_path, require_terminal=False) == []
    reduced = exp.independent_reduce(artifact)
    assert reduced["matches_declared"] is True
    assert reduced["supervisor_firings"] == 0
    assert artifact["honest_verdict"].startswith("complete_null")
    assert artifact["promotion_score"] == 0
    assert artifact["solve_credit"] == 0
    assert artifact["new_level_credit"] == 0


def test_validation_rejects_a_policy_recommendation_without_firings(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7444: zero exposure cannot support promotion or retirement."""

    rows = [
        exp.reduce_episode(_episode("bp35"), None, _observations([0, 0])),
        exp.reduce_episode(_episode("cn04"), None, _observations([0, 0])),
    ]
    artifact = exp.build_artifact_fixture(rows)
    artifact["supervisor_recommendations"] = [{"action": "retire", "arm": "drop_goal_bias"}]
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert "zero_firing_recommendation_forbidden" in exp.validate_artifact(
        artifact, root=tmp_path, require_terminal=False
    )


def test_replay_cli_checks_a_candidate_without_writing_the_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-ARC-WMTE-7444-TERMINAL: cold replay is read-only and deterministic."""

    rows = [
        exp.reduce_episode(_episode("bp35"), None, _observations([0, 0])),
        exp.reduce_episode(_episode("cn04"), None, _observations([0, 0])),
    ]
    artifact = exp.build_artifact_fixture(rows)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--replay", str(candidate)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["reduced"]["matches_declared"] is True
    assert output["validation_errors"] == []


def test_io_progress_and_gate_helpers_preserve_fail_closed_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7444: malformed inputs remain missing, not fabricated."""

    assert "+00:00" in exp.utc_now()
    exp.progress(0.0, "test", "boundary", completed_units=1)
    assert "phase=test event=boundary" in capsys.readouterr().out

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    valid = tmp_path / "valid.json"
    valid.write_text('{"value": 3}', encoding="utf-8")
    assert exp._load_json(malformed) == {}
    assert exp._load_json(sequence) == {}
    assert exp._load_json(tmp_path / "missing.json") == {}
    assert exp._load_json(valid) == {"value": 3}

    events = tmp_path / "events.jsonl"
    events.write_text('{"event":"ok"}\nnot-json\n[]\n', encoding="utf-8")
    assert exp._read_jsonl(events) == [{"event": "ok"}]
    assert exp._read_jsonl(tmp_path / "missing.jsonl") == []
    assert exp.gate_check_summary([]) == {
        "passed": True,
        "failed_count": 0,
        "first_failure": None,
    }
    failed = exp.authenticate_upstream({})
    summary = exp.gate_check_summary(failed)
    assert summary["passed"] is False
    assert summary["first_failure"]["observed"] is None


def test_unknown_threshold_and_applied_effect_branches_stay_distinct() -> None:
    """REQ-ARC-WMTE-7444: unknown exposure, progress, and failure are distinct."""

    episode = _episode()
    episode["actions"] = 130
    unknown = exp.reduce_episode(
        episode,
        {
            "mode": "shadow",
            "window": 120,
            "actions_observed": 130,
            "arms_enabled": list(exp.DEFAULT_ENABLED_ARMS),
            "fired": 0,
        },
        _observations([0]),
    )
    assert unknown["firing_threshold_reached"] is None

    active = _detailed_supervisor(enabled=list(exp.CURATED_ARMS), fired=["drop_goal_bias"])
    active["mode"] = "applied"
    active["consumed"] = 1
    progressed_episode = _episode()
    progressed_episode["actions"] = 130
    progressed_episode["level_progress"] = 1
    progressed = exp.reduce_episode(progressed_episode, active, _observations([0, 1]))
    assert progressed["arm_effect_evidence"] == "measured_progress_after_intervention"
    assert progressed["arm_rows"][0]["effect_evidence"] == "measured_progress"

    failed_episode = deepcopy(progressed_episode)
    failed_episode["level_progress"] = 0
    failed = exp.reduce_episode(failed_episode, active, _observations([0]))
    assert failed["arm_effect_evidence"] == "measured_failed_intervention"
    assert failed["arm_rows"][0]["effect_evidence"] == "measured_failed_intervention"

    aggregate_only = deepcopy(active)
    aggregate_only.pop("would_have_redirects")
    aggregate_only.pop("window")
    aggregate_only.pop("actions_observed")
    aggregate_only["fired"] = 1
    aggregate = exp.reduce_episode(failed_episode, aggregate_only, _observations([0]))
    assert aggregate["arm_rows"][0]["fired"] is None


def test_artifact_validation_names_each_closed_boundary(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7444: each identity, provenance, and policy gate fails closed."""

    rows = [
        exp.reduce_episode(_episode("bp35"), None, _observations([0])),
        exp.reduce_episode(_episode("cn04"), None, _observations([0])),
    ]
    base = exp.build_artifact_fixture(rows)
    mutations = {
        "schema_mismatch": ("schema", "bad"),
        "experiment_id_mismatch": ("experiment_id", "bad"),
        "run_identity_mismatch": ("run_date", "19000101"),
        "current_model_boundary_invalid": ("model_invoked", True),
        "current_invocation_counts_nonzero": (
            "invocation_counts",
            {**exp.ZERO_CURRENT_INVOCATIONS, "generation_calls_attempted": 1},
        ),
        "substrate_class_invalid": ("inference_substrate_class", "model_bounded_generation"),
        "execution_venue_invalid": ("execution_venue", "external"),
        "credit_or_promotion_nonzero": ("solve_credit", 1),
        "production_default_change_forbidden": ("production_defaults_changed", True),
        "reproducibility_checksum_mismatch": ("reproducibility_checksum", "sha256:bad"),
    }
    for expected, (field, value) in mutations.items():
        candidate = deepcopy(base)
        candidate[field] = value
        assert expected in exp.validate_artifact(candidate, root=tmp_path, require_terminal=False)
    assert "terminal_receipts_invalid" in exp.validate_artifact(
        base, root=tmp_path, require_terminal=True
    )


def test_phase_hash_note_and_terminal_command_plan_are_bounded(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7444-TERMINAL: outputs and readers have explicit scope."""

    spans: list[dict] = []
    exp._phase(spans, "unit", 5.0, 4.0, 2)
    assert spans[0]["phase"] == "unit"
    assert spans[0]["completed_units"] == 2

    source = tmp_path / "source.txt"
    source.write_text("bytes\n", encoding="utf-8")
    hashed = exp._source_hash_row(source, "fixture", original_flags={"clean": True})
    assert hashed["bytes"] == 6
    assert hashed["sha256"].startswith("sha256:")
    assert hashed["original_flags"] == {"clean": True}

    note = tmp_path / "nested" / "note.md"
    exp._write_note(note)
    assert "zero supervisor firings" in note.read_text(encoding="utf-8").lower()

    specs = exp._terminal_specs(tmp_path, tmp_path / "candidate.json")
    assert [row.name for row in specs] == list(exp.TERMINAL_CHECK_NAMES)
    assert all(row.timeout_s == 300 for row in specs)

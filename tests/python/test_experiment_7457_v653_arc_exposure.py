"""Tests for the bounded V653 ARC supervisor exposure comparison.

Spec refs: REQ-ARC-WMTE-7457 and SCENARIO-ARC-WMTE-7457-THRESHOLD/
SCHEDULE/EXPOSURE/ARM-EXHAUSTION/TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7457_v653_arc_exposure as exposure
from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER, Redirect


REPO = Path(__file__).resolve().parents[2]


def _episode(
    episode_id: str,
    game: str,
    seed: int,
    condition: str,
    *,
    actions: int = 180,
    disposition: str = "complete",
    redirects: list[dict] | None = None,
) -> dict:
    applied = condition == "applied"
    redirect_rows = redirects or []
    return {
        "episode_id": episode_id,
        "game": game,
        "seed": seed,
        "condition": condition,
        "disposition": disposition,
        "action_count": actions,
        "threshold_reached": actions >= exposure.SUPERVISOR_THRESHOLD,
        "peak_level": 0,
        "terminal_level": 0,
        "banked_progress": 0,
        "supervisor_receipt": {
            "mode": condition,
            "window": exposure.SUPERVISOR_THRESHOLD,
            "arms_enabled": list(exposure.ENABLED_ARMS),
            "redirects" if applied else "would_have_redirects": redirect_rows,
            "unredirected_windows": [],
        },
        "action_rows": [
            {
                "episode_id": episode_id,
                "action_index": index + 1,
                "monotonic_s": float(index + 1),
                "level": 0,
                "plateau_counter": (index + 1) % exposure.SUPERVISOR_THRESHOLD,
                "eligible_window": index + 1 == exposure.SUPERVISOR_THRESHOLD,
                "proposed_redirect": (
                    redirect_rows[0]["arm"]
                    if redirect_rows and index + 1 == exposure.SUPERVISOR_THRESHOLD
                    else None
                ),
                "applied_redirect": bool(
                    applied and redirect_rows and index + 1 == exposure.SUPERVISOR_THRESHOLD
                ),
                "callback_outcome": "not_called",
                "later_progress": False,
            }
            for index in range(actions)
        ],
        "request_budget_receipt": {
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "in_flight": 0,
            "callback_rows": [],
        },
        "elapsed_s": 1.0,
        "solve_provenance": "no_level_reached",
        "new_level_credit": 0,
    }


def test_req_arc_wmte_7457_spec_and_fixed_protocol() -> None:
    """REQ-ARC-WMTE-7457: the fixed protocol exists before implementation."""

    text = (REPO / exposure.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("REQ-ARC-WMTE-7457") :]
    for anchor in (
        "SCENARIO-ARC-WMTE-7457-THRESHOLD",
        "SCENARIO-ARC-WMTE-7457-SCHEDULE",
        "SCENARIO-ARC-WMTE-7457-EXPOSURE",
        "SCENARIO-ARC-WMTE-7457-ARM-EXHAUSTION",
        "SCENARIO-ARC-WMTE-7457-TERMINAL",
    ):
        assert anchor in section
    assert exposure.MODEL_SPECS == ["unsloth/Qwen3.8-27B-GGUF"]
    assert exposure.INFERENCE_SUBSTRATE_CLASS == "model_bounded_generation"
    assert exposure.EXECUTION_VENUE == "host"
    assert exposure.SUPERVISOR_THRESHOLD == 120
    assert exposure.ACTION_LIMIT == 180
    assert exposure.EPISODE_LIMIT_S == 240.0
    assert exposure.AGGREGATE_LIVE_LIMIT_S == 2400.0
    assert exposure.REQUEST_LIMIT == 2
    assert exposure.MAX_NEW_TOKENS == 256
    assert exposure.CURATED_ARMS == tuple(ARM_ORDER)


def test_scenario_arc_wmte_7457_schedule_is_interleaved_and_paired() -> None:
    """SCENARIO-ARC-WMTE-7457-SCHEDULE: all eight dispositions are sealed."""

    rows = exposure.build_schedule(("bp35", "cn04"))
    assert len(rows) == 8
    assert [(row["game"], row["seed"], row["condition"]) for row in rows] == [
        ("bp35", exposure.EPISODE_SEEDS[0], "shadow"),
        ("bp35", exposure.EPISODE_SEEDS[0], "applied"),
        ("cn04", exposure.EPISODE_SEEDS[0], "shadow"),
        ("cn04", exposure.EPISODE_SEEDS[0], "applied"),
        ("bp35", exposure.EPISODE_SEEDS[1], "shadow"),
        ("bp35", exposure.EPISODE_SEEDS[1], "applied"),
        ("cn04", exposure.EPISODE_SEEDS[1], "shadow"),
        ("cn04", exposure.EPISODE_SEEDS[1], "applied"),
    ]
    assert all(row["action_limit"] == 180 for row in rows)
    assert all(row["episode_limit_s"] == 240.0 for row in rows)
    assert all(row["request_limit"] == 2 for row in rows)
    assert all(row["adapter_disabled"] is True for row in rows)
    assert all(row["stored_engines_disabled"] is True for row in rows)


def test_scenario_arc_wmte_7457_actual_policy_hook_qualification() -> None:
    """SCENARIO-ARC-WMTE-7457-THRESHOLD: actual hooks fire at 120 only."""

    rows = exposure.qualify_scripted_policy_hooks()
    assert [row["condition"] for row in rows] == ["shadow", "applied"]
    for row in rows:
        assert row["evidence_class"] == "synthetic_policy_hook_qualification"
        assert row["actions_before_boundary"] == 119
        assert row["redirects_before_boundary"] == 0
        assert row["actions_at_boundary"] == 120
        assert row["proposed_redirect"] in exposure.ENABLED_ARMS
        assert row["threshold"] == 120
    assert rows[0]["redirect_applied"] is False
    assert rows[1]["redirect_applied"] is True


def test_scenario_arc_wmte_7457_exposure_gap_is_not_an_intervention_null() -> None:
    """SCENARIO-ARC-WMTE-7457-EXPOSURE: censoring retains the exact gap."""

    schedule = exposure.build_schedule(("bp35", "cn04"))
    first = schedule[0]
    row = _episode(
        first["episode_id"],
        first["game"],
        first["seed"],
        first["condition"],
        actions=62,
        disposition="censored_timeout",
    )
    reduced = exposure.reduce_episode(first, row)
    assert reduced["threshold_reached"] is False
    assert reduced["remaining_exposure_gap"] == 58
    assert reduced["effect_evidence"] == "exposure_limited"
    assert reduced["intervention_null"] is False


def test_scenario_arc_wmte_7457_arm_exhaustion_requires_later_window() -> None:
    """SCENARIO-ARC-WMTE-7457-ARM-EXHAUSTION: no new arm is inferred early."""

    partial = {
        "arms_enabled": list(exposure.ENABLED_ARMS),
        "redirects": [{"arm": exposure.ENABLED_ARMS[0], "action_index": 120, "level": 0}],
        "unredirected_windows": [],
    }
    assert exposure.new_arm_evidence(partial) is False

    all_fired = deepcopy(partial)
    all_fired["redirects"] = [
        {"arm": arm, "action_index": 120 * (index + 1), "stretch_level": 0}
        for index, arm in enumerate(exposure.ENABLED_ARMS)
    ]
    assert exposure.new_arm_evidence(all_fired) is False
    all_fired["unredirected_windows"] = [
        {
            "action_index": 120 * (len(exposure.ENABLED_ARMS) + 1),
            "stretch_level": 0,
            "arms_used": list(exposure.ENABLED_ARMS),
            "arms_enabled": list(exposure.ENABLED_ARMS),
        }
    ]
    assert exposure.new_arm_evidence(all_fired) is True


def test_scenario_arc_wmte_7457_panel_reducer_keeps_all_dispositions() -> None:
    """SCENARIO-ARC-WMTE-7457-TERMINAL: raw rows recompute paired exposure."""

    schedule = exposure.build_schedule(("bp35", "cn04"))
    observed = []
    for sealed in schedule[:-1]:
        redirects = (
            [{"arm": exposure.ENABLED_ARMS[0], "action_index": 120, "stretch_level": 0}]
            if sealed["condition"] == "applied"
            else []
        )
        observed.append(
            _episode(
                sealed["episode_id"],
                sealed["game"],
                sealed["seed"],
                sealed["condition"],
                redirects=redirects,
            )
        )
    reduced = exposure.reduce_panel(schedule, observed)
    assert len(reduced["rows"]) == 8
    assert reduced["sample_size_budget"] == {
        "planned_units": 8,
        "attempted_units": 7,
        "completed_units": 7,
        "failed_units": 0,
        "censored_units": 0,
        "unstarted_units": 1,
        "fixed_stop_rule": True,
    }
    assert reduced["all_dispositions_present"] is True
    assert len(reduced["per_game_results"]) == 4
    assert reduced["scientific_null_eligible_pairs"] == 3


def test_req_arc_wmte_7457_preconditions_and_terminal_fixture() -> None:
    """REQ-ARC-WMTE-7457: exact upstream flags and terminal claims reduce."""

    gates, hashes, registry = exposure.collect_preconditions(REPO, force_live="1")
    assert all(row["passed"] is True for row in gates)
    assert exposure.UPSTREAM_PATH.as_posix() in hashes
    assert hashes[exposure.UPSTREAM_PATH.as_posix()]["original_flags"] == {
        "status": "complete_null",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "capture_lifecycle_ready_score": 1,
    }
    registry_check = exposure.registry_precheck(registry)
    assert registry_check["passed"] is True
    assert registry_check["already_solved"] == {"bp35": True, "cn04": True}

    artifact = exposure.build_artifact_for_test()
    assert exposure.validate_artifact(artifact, require_terminal=False) == []
    assert exposure.independent_reduce(artifact)["matches_declared"] is True
    assert artifact["promotion_score"] == 0
    assert artifact["arm_value_score"] == 0
    assert artifact["new_level_credit"] == 0
    assert artifact["supervisor_threshold"] == 120
    assert artifact["supervisor_arm_order"] == list(ARM_ORDER)

    changed = deepcopy(artifact)
    changed["supervisor_threshold"] = 119
    assert "supervisor_threshold_mismatch" in exposure.validate_artifact(
        changed, require_terminal=False
    )


def test_req_arc_wmte_7457_helpers_and_defensive_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7457: helper boundaries fail closed and remain reproducible."""

    assert exposure.utc_now().endswith("Z")
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert exposure.load_object(missing) == {}
    assert exposure.load_object(malformed) == {}
    assert exposure.load_object(scalar) == {}
    assert exposure._compare(">=", 2, 2) is True
    assert exposure._compare(">=", 2, None) is False
    with pytest.raises(ValueError, match="unsupported operator"):
        exposure._compare("!=", 1, 2)

    root = tmp_path / "root"
    (root / exposure.REGISTRY_PATH.parent).mkdir(parents=True)
    (root / exposure.REGISTRY_PATH).write_text("[", encoding="utf-8")
    (root / exposure.EXCLUSION_PATH).write_text("", encoding="utf-8")
    gates, _hashes, registry = exposure.collect_preconditions(root, force_live="1")
    assert registry == {}
    assert any(row["check"] == "registry_parse" and row["passed"] is False for row in gates)

    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", "prior")
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW", "37")
    monkeypatch.setenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", "prior")
    assert all(row["passed"] for row in exposure.qualify_scripted_policy_hooks())
    assert os.environ["CARNOT_ARC_TRAJECTORY_SUPERVISOR"] == "prior"
    assert os.environ["CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW"] == "37"
    assert os.environ["CARNOT_ARC_SUPERVISOR_TOOL_ARM"] == "prior"


def test_req_arc_wmte_7457_probe_plans_and_cold_file_reduction(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7457: observation and validation use the shipped boundaries."""

    class FakeSupervisor:
        window = 120
        _actions_since_progress = 119

        def observe(self, snapshot: object) -> Redirect:
            self._actions_since_progress = 0
            return Redirect("drop_goal_bias", 120, 0, "fixture")

        def receipt(self) -> dict:
            return {"window": self.window}

    fake = FakeSupervisor()
    probe = exposure.SupervisorProbe(fake, True)
    redirect = probe.observe(SimpleNamespace(level=0))
    assert redirect.arm == "drop_goal_bias"
    assert probe.last == {
        "level": 0,
        "plateau_before": 119,
        "plateau_counter": 0,
        "eligible_window": True,
        "proposed_redirect": "drop_goal_bias",
        "applied_redirect": True,
    }
    assert probe.receipt() == {"window": 120}
    assert probe.window == 120

    artifact = exposure.build_artifact_for_test()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exposure.independent_reduce_file(path)["matches_declared"] is True

    private = tmp_path / "private"
    plan = exposure.build_validation_plan(REPO, private)
    assert exposure.validate_validation_plan(REPO, plan) == []
    assert [row.name for row in plan] == list(exposure.validation_scope.REQUIRED_CHECK_NAMES)
    e2e = exposure.e2e_command_specs(REPO, private / "e2e")
    assert [row.name for row in e2e] == list(exposure.REQUIRED_E2E)
    terminal = exposure.terminal_command_specs(REPO, path)
    assert [row.name for row in terminal] == list(exposure.REQUIRED_TERMINAL)

    spans: list[dict] = []
    exposure._phase(spans, "fixture", 1.0, 0.0, 3, "checkpoint.json")
    assert spans[0]["phase"] == "fixture"
    assert spans[0]["completed_units"] == 3


def test_req_arc_wmte_7457_request_rows_and_validator_mutations() -> None:
    """REQ-ARC-WMTE-7457: terminal validation rejects each protected claim."""

    callback = {
        "request_id": "request-1",
        "disposition": "completed",
        "request_dispatched": True,
    }
    rows = exposure._request_rows(
        [
            {"episode_id": "no-receipt"},
            {"episode_id": "mixed", "request_budget_receipt": {"callback_rows": [1, callback]}},
        ]
    )
    assert rows == [{"episode_id": "mixed", **callback}]

    artifact = exposure.build_artifact_for_test()
    mutations = {
        "identity_mismatch": ("schema", "changed"),
        "supervisor_arm_order_mismatch": ("supervisor_arm_order", []),
        "execution_venue_mismatch": ("execution_venue", "external"),
        "invocation_events_invalid": (
            "current_invocation_events",
            [{"_malformed": True, "error": "fixture"}],
        ),
        "MODEL_SPECS_mismatch": ("MODEL_SPECS", []),
        "duration_invalid": ("duration_s", 0),
        "duration_floor_invalid": ("duration_s", 5),
        "credit_or_promotion_nonzero": ("promotion_score", 1),
        "new_arm_proposal_forbidden": ("new_arm_proposed", True),
        "protected_default_changed": ("production_defaults_changed", True),
        "synthetic_live_separation_invalid": ("scripted_rows_in_live_metrics", True),
        "verdict_class_invalid": ("verdict_class", "unknown"),
        "honest_verdict_invalid": ("honest_verdict", "null_without_complete_prefix"),
        "field_principles_incomplete": ("field_principles", {}),
        "reproducibility_checksum_mismatch": ("reproducibility_checksum", "sha256:changed"),
    }
    for expected, (field, value) in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exposure.validate_artifact(changed, require_terminal=False)

    assert "validation_receipts_invalid" in exposure.validate_artifact(
        artifact, require_terminal=True
    )
    parsed = exposure.parse_args(["--date", exposure.RUN_DATE])
    assert parsed.role == "experiment"
    replay = exposure.parse_args(["--replay", "/tmp/candidate.json", "--reduce-only"])
    assert replay.reduce_only is True
    with pytest.raises(SystemExit):
        exposure.parse_args([])

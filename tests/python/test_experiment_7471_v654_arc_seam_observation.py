"""Tests for the V654 ARC decision-seam observation pilot.

Spec refs: REQ-ARC-WMTE-7471 and SCENARIO-ARC-WMTE-7471-OBSERVER-PARITY/
FROZEN-PANEL/SEAM-RECEIPTS/REDUCTION/TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7471_v654_arc_seam_observation as observation


REPO = Path(__file__).resolve().parents[2]


class _FakeSupervisor:
    """Expose the same observe/receipt seam without selecting test actions."""

    def __init__(self) -> None:
        self.actions = 0
        self.redirects: list[dict] = []

    def observe(self, snapshot: object) -> object | None:
        self.actions += 1
        if self.actions == 1:
            redirect = SimpleNamespace(
                arm="force_exploration_diversity",
                action_index=1,
                level=0,
                diagnosis="fixture",
            )
            self.redirects.append(
                {
                    "arm": redirect.arm,
                    "action_index": 1,
                    "level": 0,
                    "resolved_by_levelup": False,
                }
            )
            return redirect
        return None

    def receipt(self) -> dict:
        return {
            "actions_observed": self.actions,
            "arms_enabled": list(observation.SUPERVISOR_ARM_ORDER),
            "redirects": deepcopy(self.redirects),
        }


class _FakePolicy:
    """Small E3-shaped policy used to prove the observer is a pass-through."""

    def __init__(self) -> None:
        self.phase = "explore"
        self._trajectory_supervisor = _FakeSupervisor()
        self._trajectory_supervisor_applies = True
        self.world_model_trust_selection = None
        self.induction_attempts: list[dict] = []
        self._current_goal_level = 1
        self._observer_action_count = 0

    def next_move(self, frames: list, latest: object) -> tuple[str, None]:
        self._observer_action_count += 1
        self._should_enter_induction(stalled=False, won=False)
        self._maybe_supervise_trajectory(latest)
        return "ACTION1", None

    def _should_enter_induction(self, *, stalled: bool, won: bool) -> tuple[bool, str | None]:
        return (stalled and not won, "stall" if stalled and not won else None)

    def _maybe_supervise_trajectory(self, latest: object) -> None:
        redirect = self._trajectory_supervisor.observe(latest)
        if redirect is not None and self._trajectory_supervisor_applies:
            self.phase = "explore"

    def _world_model_candidates(self, engine: object, is_done: object) -> list:
        del engine, is_done
        return [SimpleNamespace(name="dsl"), SimpleNamespace(name="generated")]

    def _induce_and_plan_timed(self) -> None:
        candidates = self._world_model_candidates(object(), object())
        self.world_model_trust_selection = SimpleNamespace(
            selected_name=candidates[1].name,
            selected_score=SimpleNamespace(binary_gate_pass=True),
        )
        self.induction_attempts.append(
            {"selected_candidate_name": candidates[1].name, "planned": False}
        )


def _episode(sealed: dict, *, disposition: str = "complete", progressed: bool = False) -> dict:
    actions = 3 if progressed else 2
    action_rows = [
        {
            "action_index": index + 1,
            "level": 1 if progressed and index == actions - 1 else 0,
            "later_progress": progressed and index < actions - 1,
        }
        for index in range(actions)
    ]
    return {
        **deepcopy(sealed),
        "disposition": disposition,
        "action_count": actions,
        "start_level": 0,
        "peak_level": 1 if progressed else 0,
        "terminal_level": 1 if progressed else 0,
        "action_rows": action_rows,
        "request_budget_receipt": {
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "in_flight": 0,
            "callback_rows": [],
        },
        "solve_provenance": ("live_agent_self_discovery" if progressed else "no_level_reached"),
        "trace_reproduction": {"attempted": progressed, "passed": progressed},
        "elapsed_s": 1.0,
        "new_level_credit": 0,
        "error": None,
    }


def test_req_arc_wmte_7471_spec_and_fixed_protocol() -> None:
    """REQ-ARC-WMTE-7471 fixes the model and bounded eight-episode protocol."""

    text = (REPO / observation.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("REQ-ARC-WMTE-7471") :]
    for anchor in (
        "SCENARIO-ARC-WMTE-7471-OBSERVER-PARITY",
        "SCENARIO-ARC-WMTE-7471-FROZEN-PANEL",
        "SCENARIO-ARC-WMTE-7471-SEAM-RECEIPTS",
        "SCENARIO-ARC-WMTE-7471-REDUCTION",
        "SCENARIO-ARC-WMTE-7471-TERMINAL",
    ):
        assert anchor in section
    assert observation.MODEL_SPECS == ["unsloth/Qwen3.8-27B-GGUF"]
    assert observation.INFERENCE_SUBSTRATE_CLASS == "model_bounded_generation"
    assert observation.ACTION_LIMIT == 180
    assert observation.EPISODE_LIMIT_S == 240.0
    assert observation.AGGREGATE_LIVE_LIMIT_S == 2400.0
    assert observation.REQUEST_LIMIT == 2
    assert observation.MAX_NEW_TOKENS == 256


def test_scenario_arc_wmte_7471_frozen_panel_uses_stable_hashes() -> None:
    """SCENARIO-ARC-WMTE-7471-FROZEN-PANEL seals four games before outcomes."""

    available = ["z9", "bp35", "b2", "a1", "cn04", "d4", "c3", "e5"]
    first = observation.freeze_game_panel(available)
    second = observation.freeze_game_panel(reversed(available))
    assert first == second
    assert len(first["games"]) == 4
    assert not {"bp35", "cn04"}.intersection(first["games"])
    assert all(row["stable_hash"].startswith("sha256:") for row in first["game_rows"])
    assert first["selection_used_current_outcomes"] is False

    schedule = observation.build_schedule(first["games"])
    assert len(schedule) == 8
    assert len({row["episode_id"] for row in schedule}) == 8
    assert {row["game"] for row in schedule} == set(first["games"])
    assert all(row["action_limit"] == 180 for row in schedule)
    assert all(row["request_limit"] == 2 for row in schedule)
    assert all(row["adapter_disabled"] is True for row in schedule)
    assert all(row["cross_game_state_disabled"] is True for row in schedule)


def test_scenario_arc_wmte_7471_observer_is_action_and_request_neutral(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7471-OBSERVER-PARITY preserves policy sequences."""

    baseline = _FakePolicy()
    baseline_actions = [baseline.next_move([], SimpleNamespace(levels_completed=0))]

    event_path = tmp_path / "seams.jsonl"
    observed = _FakePolicy()
    observer = observation.E3SeamObserver(
        episode_id="g1:seed-1",
        event_path=event_path,
        clock_ns=iter(range(1_000, 100_000, 100)).__next__,
    )
    observer.install(observed)
    observed_actions = [observed.next_move([], SimpleNamespace(levels_completed=0))]
    observed._induce_and_plan_timed()

    assert observed_actions == baseline_actions
    assert observed._trajectory_supervisor.receipt() == baseline._trajectory_supervisor.receipt()
    rows = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert {row["seam"] for row in rows} == {
        "candidate_action_selection",
        "hypothesis_gate",
        "supervisor_arm_selection",
        "induction_timing",
    }
    assert {row["event"] for row in rows}.issuperset(
        {"stage_start", "candidate_set", "selection", "stage_end"}
    )
    assert all(row["clock_identity"] == "time.monotonic_ns" for row in rows)
    assert all(row["episode_id"] == "g1:seed-1" for row in rows)
    assert observer.errors == []


def test_scenario_arc_wmte_7471_seam_rows_keep_missing_options_and_intervals(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7471-SEAM-RECEIPTS retains unavailable evidence."""

    path = tmp_path / "seams.jsonl"
    policy = _FakePolicy()
    observer = observation.E3SeamObserver("episode", path)
    observer.install(policy)
    policy.next_move([], SimpleNamespace(levels_completed=0))
    rows = observation.read_jsonl(path)
    action_rows = [row for row in rows if row["seam"] == "candidate_action_selection"]
    assert [row["event"] for row in action_rows] == [
        "stage_start",
        "candidate_set",
        "selection",
        "stage_end",
    ]
    candidate = action_rows[1]
    assert candidate["candidate_ids"] == []
    assert candidate["missing_options_reason"] == "legal_action_payloads_unavailable_at_call_site"
    assert (
        action_rows[-1]["interval_end_monotonic_ns"]
        >= action_rows[0]["interval_start_monotonic_ns"]
    )
    assert action_rows[-1]["observer_cpu_ns"] >= 0


def test_scenario_arc_wmte_7471_reducer_keeps_censoring_and_game_clusters() -> None:
    """SCENARIO-ARC-WMTE-7471-REDUCTION recomputes all fixed units."""

    schedule = observation.build_schedule(("g1", "g2", "g3", "g4"))
    episodes = [
        _episode(row, progressed=index in {0, 1}) for index, row in enumerate(schedule[:-1])
    ]
    episodes[-1]["disposition"] = "censored_timeout"
    seams = [
        {
            "episode_id": schedule[0]["episode_id"],
            "decision_id": "d1",
            "seam": "candidate_action_selection",
            "event": "stage_start",
            "interval_start_monotonic_ns": 10,
        },
        {
            "episode_id": schedule[0]["episode_id"],
            "decision_id": "d1",
            "seam": "candidate_action_selection",
            "event": "stage_end",
            "interval_start_monotonic_ns": 10,
            "interval_end_monotonic_ns": 30,
            "observer_cpu_ns": 3,
        },
        {
            "episode_id": schedule[0]["episode_id"],
            "decision_id": "callback",
            "seam": "downstream_generation",
            "event": "stage_end",
            "event_monotonic_ns": 1_000,
            "interval_start_monotonic_ns": 100,
            "interval_end_monotonic_ns": 1_000,
        },
    ]
    reduced = observation.reduce_panel(schedule, episodes, seams)
    assert len(reduced["rows"]) == 8
    assert reduced["sample_size_budget"] == {
        "planned_independent_units": 8,
        "attempted_independent_units": 7,
        "complete_independent_units": 6,
        "failed_independent_units": 0,
        "censored_independent_units": 1,
        "unstarted_independent_units": 1,
        "independent_game_clusters": 4,
    }
    assert len(reduced["per_game_results"]) == 4
    assert reduced["actions_to_progress"]["observed"] == [3, 3]
    assert reduced["replaceable_cost_bounds_ns"] == {
        "observed_lower": 20,
        "observed_upper": 20,
        "observation_overhead": 3,
        "unfinished_decisions": 0,
    }
    assert reduced["rows"][-1]["disposition"] == "unstarted"


def test_req_arc_wmte_7471_preconditions_fixture_and_mutation_detection() -> None:
    """REQ-ARC-WMTE-7471 authenticates E6 and independently reduces claims."""

    checks, hashes, registry = observation.collect_preconditions(REPO, force_live="1")
    assert all(row["passed"] is True for row in checks)
    upstream = hashes[observation.UPSTREAM_PATH.as_posix()]
    assert upstream["original_flags"] == {
        "status": "complete_sample_limited_cost_profile",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "decision_profile_complete_score": 1,
    }
    assert registry

    artifact = observation.build_artifact_for_test()
    assert observation.validate_artifact(artifact, require_terminal=False) == []
    assert observation.independent_reduce(artifact)["matches_declared"] is True
    assert artifact["arc_observation_complete_score"] == 1
    assert artifact["public_development_generalization_proxy"] is True
    assert artifact["promotion_score"] == 0
    assert artifact["new_level_credit"] == 0

    changed = deepcopy(artifact)
    changed["sample_size_budget"]["complete_independent_units"] = 99
    assert "independent_reduction_mismatch" in observation.validate_artifact(
        changed, require_terminal=False
    )


def test_scenario_arc_wmte_7471_terminal_schema_and_cli_replay(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7471-TERMINAL keeps the required public schema."""

    artifact = observation.build_artifact_for_test()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert observation.main(["--replay", str(path), "--reduce-only"]) == 0
    for field in observation.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["schema"].startswith("carnot.exp7471.v654")
    assert artifact["experiment_id"] == "exp7471-v654-arc-seam-observation"
    assert artifact["milestone"] == "2026.09.654"
    assert artifact["run_date"] == "20260921"


def test_req_arc_wmte_7471_support_readers_and_fail_closed_branches(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7471 keeps malformed and unavailable evidence explicit."""

    assert observation.utc_now().endswith("Z")
    assert observation.load_object(tmp_path / "missing.json") == {}
    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    assert observation.load_object(non_object) == {}
    assert observation.read_jsonl(tmp_path / "missing.jsonl") == []
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text('{"ok": 1}\nnot-json\n[]\n', encoding="utf-8")
    rows = observation.read_jsonl(mixed)
    assert rows[0] == {"ok": 1}
    assert rows[1]["event"] == "malformed_jsonl"
    assert rows[2] == {"event": "non_object"}

    assert observation._compare("in", ["null", "blocked"], "null") is True
    with pytest.raises(ValueError, match="unsupported operator"):
        observation._compare("bad", 1, 1)

    registry = {
        "games": [
            {"game": "g1", "levels_reproduced": 2, "full_game_clear": True},
            None,
        ]
    }
    receipt = observation.registry_precheck(registry, ("g1", "g2", "g3", "g4"))
    assert receipt["passed"] is True
    assert receipt["rows"][0]["registered"] is True
    assert receipt["rows"][1]["registered"] is False
    assert receipt["policy_received_registry_data"] is False


def test_scenario_arc_wmte_7471_observer_defensive_paths(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7471-OBSERVER-PARITY keeps guard paths observational."""

    assert observation.E3SeamObserver._move(SimpleNamespace(name="ACTION2")) == {
        "action": "ACTION2",
        "payload": None,
    }
    assert observation.E3SeamObserver._legal_candidates(None) == (
        [],
        "legal_action_payloads_unavailable_at_call_site",
    )

    class BadSupervisor:
        def receipt(self) -> dict:
            raise RuntimeError("fixture")

    assert observation.E3SeamObserver._supervisor_rows(BadSupervisor()) == []
    fallback = SimpleNamespace(
        world_model_trust_selection=None,
        induction_attempts=[{"selected_candidate_name": "fallback"}],
    )
    assert observation.E3SeamObserver._selected_hypothesis(fallback) == ("fallback", None)

    policy = _FakePolicy()
    observer = observation.E3SeamObserver("episode", tmp_path / "seams.jsonl")
    observer.install(policy)
    with pytest.raises(ValueError, match="already has"):
        observer.install(policy)


def test_scenario_arc_wmte_7471_reducer_empty_and_file_shard_paths(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7471-REDUCTION handles empty clusters and file shards."""

    assert observation._wilson(0, 0) == {
        "estimate": None,
        "lower_95": None,
        "upper_95": None,
        "clusters": 0,
    }
    shard = tmp_path / "shard.jsonl"
    shard.write_text(
        json.dumps(
            {
                "episode_id": "e",
                "decision_id": "d",
                "seam": "induction_timing",
                "event": "stage_start",
                "event_monotonic_ns": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events = observation._events_from_shards([{"path": shard.name}], root=tmp_path)
    assert events[0]["decision_id"] == "d"

    spans: list[dict] = []
    observation._phase(spans, "fixture", 1.0, 0.0, 2, "checkpoint")
    assert spans[0]["phase"] == "fixture"
    assert spans[0]["completed_units"] == 2

    shards = observation._seam_shards(
        tmp_path,
        [
            {"episode_id": "missing"},
            {"episode_id": "also-missing", "seam_event_path": str(tmp_path / "nope")},
            {"episode_id": "e", "seam_event_path": str(shard)},
        ],
    )
    assert len(shards) == 1
    assert shards[0]["path"] == "shard.jsonl"
    assert shards[0]["row_count"] == 1
    external = observation._seam_shards(
        tmp_path / "different-root",
        [{"episode_id": "e", "seam_event_path": str(shard)}],
    )
    assert external[0]["path"] == str(shard)


def test_req_arc_wmte_7471_validation_scope_and_rejection_matrix(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7471 freezes exact checks and rejects false terminal claims."""

    plan = observation.build_validation_plan(REPO, tmp_path / "scope")
    assert observation.validate_validation_plan(REPO, plan) == []
    assert [row.name for row in plan] == list(observation.validation_scope.REQUIRED_CHECK_NAMES)
    terminal = observation.terminal_command_specs(REPO, tmp_path / "candidate.json")
    assert [row.name for row in terminal] == list(observation.REQUIRED_TERMINAL)

    artifact = observation.build_artifact_for_test()
    missing = deepcopy(artifact)
    del missing["schema"]
    assert "missing_field:schema" in observation.validate_artifact(missing, require_terminal=False)
    wrong_identity = deepcopy(artifact)
    wrong_identity["run_date"] = "bad"
    assert "identity_mismatch:run_date" in observation.validate_artifact(
        wrong_identity, require_terminal=False
    )
    unbalanced = deepcopy(artifact)
    unbalanced["invocation_counts"]["generation_calls_attempted"] = 1
    assert "unbalanced_invocations:generation_calls" in observation.validate_artifact(
        unbalanced, require_terminal=False
    )
    invalid = deepcopy(artifact)
    invalid["verdict_class"] = "invented"
    assert "invalid_verdict_class" in observation.validate_artifact(invalid, require_terminal=False)
    oracle_positive = deepcopy(artifact)
    oracle_positive["verdict_class"] = "positive"
    assert "oracle_positive_forbidden" in observation.validate_artifact(
        oracle_positive, require_terminal=False
    )
    assert "required_validation_missing_or_failed" in observation.validate_artifact(
        artifact, require_terminal=True
    )
    with pytest.raises(SystemExit):
        observation.parse_args([])

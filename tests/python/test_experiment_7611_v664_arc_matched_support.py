"""REQ-ARC-WMTE-7611 matched-prefix measurement tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7611_v664_arc_matched_support as exp


def _frame(value: int, *, level: int = 0, legal: tuple[str, ...] = ("1", "2", "3", "6")):
    return {
        "frame": [[value]],
        "level": level,
        "legal_actions": list(legal),
        "termination": "NOT_FINISHED",
    }


def _action(kind: int | str, x: int | None = None, y: int | None = None):
    data = None if x is None else {"x": x, "y": y}
    return {"kind": kind, "coordinates": deepcopy(data), "data": deepcopy(data)}


def _step(index, before, action, after, *, reset=False, boundary=False):
    return {
        "action_index": index,
        "observation": deepcopy(before),
        "action": deepcopy(action),
        "outcome": deepcopy(after),
        "reset_boundary": reset,
        "level_boundary": boundary,
        "terminated": after is None,
    }


def _episode(game: str, seed: int, branch: int, *, target_value: int = 8):
    start = _frame(1)
    shared = _frame(2, legal=("6",))
    return {
        "schema": exp.RAW_EPISODE_SCHEMA,
        "episode_id": f"{game}:{seed}",
        "game": game,
        "seed": seed,
        "policy": "E3AgentPolicy",
        "action_limit": exp.ACTION_LIMIT,
        "induction_disabled": True,
        "adapter_withheld": True,
        "stored_solutions_withheld": True,
        "game_source_read": False,
        "hidden_state_read": False,
        "offline_ground_truth_bfs": False,
        "per_game_masks_used": False,
        "live_llm_invoked": False,
        "steps": [
            _step(0, None, _action("RESET"), start, reset=True),
            _step(1, start, _action(branch), shared),
            _step(2, shared, _action(6, 4, 5), _frame(target_value)),
        ],
        "trajectory_supervisor": {
            "would_have_arm_outcomes": {},
            "actions_observed": 2,
        },
        "termination": {"reason": "action_limit", "action_opportunities": 3},
        "model_call_counts": exp.zero_invocations(),
    }


class HiddenHistoryEnv:
    """Expose the same frame after two branches while retaining hidden history."""

    def __init__(self, unstable_token: int = 0):
        self.hidden = 0
        self.unstable_token = unstable_token

    def reset(self):
        self.hidden = 0
        return _frame(1)

    def step(self, action):
        kind = action["kind"]
        if kind in (1, 2):
            self.hidden = int(kind)
            return _frame(2, legal=("6",))
        if kind == 6:
            return _frame(7 + self.hidden + self.unstable_token)
        raise AssertionError(kind)


def _execute(env, action):
    return env.reset() if action["kind"] == "RESET" else env.step(action)


def _pair():
    selection = exp.select_matched_prefixes(
        [_episode("su15", exp.SEEDS[0], 1), _episode("su15", exp.SEEDS[1], 2)],
        max_per_game=20,
    )
    return selection["selected_pairs"][0]


# REQ-ARC-WMTE-7611; SCENARIO-ARC-WMTE-7611-MATCHED-PREFIX.
def test_coordinate_key_and_hash_only_selection_ignore_future_outcomes():
    episodes = [_episode("su15", exp.SEEDS[0], 1), _episode("su15", exp.SEEDS[1], 2)]
    selection = exp.select_matched_prefixes(episodes, max_per_game=20)
    assert selection["natural_trajectory_denominator"]["selected_matched_keys"] == 1
    pair = selection["selected_pairs"][0]
    assert pair["target_action"]["coordinates"] == {"x": 4, "y": 5}
    assert pair["histories_distinct"] is True
    assert pair["history_keys"]["0"][0] == pair["history_keys"]["0"][1]
    assert pair["history_keys"]["1"][0] != pair["history_keys"]["1"][1]

    changed = deepcopy(episodes)
    changed[0]["steps"][2]["outcome"] = _frame(99)
    changed[1]["steps"][2]["outcome"] = _frame(100)
    changed_selection = exp.select_matched_prefixes(changed, max_per_game=20)
    assert changed_selection["selection_checksum"] == selection["selection_checksum"]


# REQ-ARC-WMTE-7611; SCENARIO-ARC-WMTE-7611-FRESH-REPLAY.
def test_two_hidden_histories_replay_twice_and_form_a_witness():
    measurement = exp.replay_matched_pair(_pair(), HiddenHistoryEnv, _execute)
    assert measurement["fresh_environment_count"] == 4
    assert measurement["left_within_prefix_stable"] is True
    assert measurement["right_within_prefix_stable"] is True
    assert measurement["cross_history_disagreement"] is True
    assert measurement["history_disambiguation_witness"] is True
    assert measurement["replay_exclusion"] is None


# REQ-ARC-WMTE-7611; SCENARIO-ARC-WMTE-7611-STABILITY.
def test_unstable_prefix_is_excluded_from_cross_history_witness():
    counter = iter((0, 0, 0, 1))

    def factory():
        return HiddenHistoryEnv(next(counter))

    measurement = exp.replay_matched_pair(_pair(), factory, _execute)
    assert measurement["right_within_prefix_stable"] is False
    assert measurement["history_disambiguation_witness"] is False
    assert measurement["replay_exclusion"] == "unstable_prefix"


# REQ-ARC-WMTE-7611; SCENARIO-ARC-WMTE-7611-MATCHED-PREFIX.
def test_singletons_and_long_prefixes_keep_separate_denominators():
    singleton = _episode("sp80", exp.SEEDS[0], 1)
    single = exp.select_matched_prefixes([singleton], max_per_game=20)
    assert single["selected_pairs"] == []
    assert single["natural_trajectory_denominator"]["singleton_keys"] == 2

    left = _episode("ft09", exp.SEEDS[0], 1)
    right = _episode("ft09", exp.SEEDS[1], 2)
    for raw in (left, right):
        branch = raw["steps"][1]
        target = raw["steps"][2]
        raw["steps"] = [raw["steps"][0]]
        for index in range(1, exp.PREFIX_ACTION_CAP + 1):
            row = _step(index, _frame(index), _action(1), _frame(index + 1))
            raw["steps"].append(row)
        branch["observation"] = _frame(exp.PREFIX_ACTION_CAP + 1)
        branch["action_index"] = exp.PREFIX_ACTION_CAP + 1
        target["action_index"] = exp.PREFIX_ACTION_CAP + 2
        raw["steps"].extend((branch, target))
    censored = exp.select_matched_prefixes([left, right], max_per_game=20)
    denominator = censored["natural_trajectory_denominator"]
    assert denominator["long_prefix_censored_occurrences"] == 4
    assert denominator["selected_matched_keys"] == 0


class BoundaryEnv:
    def __init__(self):
        self.level = 0
        self.reset_count = 0

    def reset(self):
        self.level = 0
        self.reset_count += 1
        return _frame(1, level=0, legal=("3",))

    def step(self, action):
        assert action["kind"] == 3
        self.level += 1
        return _frame(1 + self.level, level=self.level, legal=("3",))


# REQ-ARC-WMTE-7611; SCENARIO-ARC-WMTE-7611-FRESH-REPLAY.
def test_stateful_reset_and_same_action_level_boundary_are_verified():
    start = _frame(1, legal=("3",))
    level_one = _frame(2, level=1, legal=("3",))
    occurrence = {
        "prefix_steps": [
            _step(0, None, _action("RESET"), start, reset=True),
            _step(1, start, _action(3), level_one, boundary=True),
        ],
        "target_observation": level_one,
        "target_action": _action(3),
        "target_key": exp.target_key("sb26", level_one, _action(3)),
    }
    outcome = exp.replay_prefix_once(occurrence, BoundaryEnv, _execute)
    assert outcome["replayable"] is True
    assert outcome["verified_reset_boundaries"] == 1
    assert outcome["verified_level_boundaries"] == 1
    assert outcome["target_outcome_sha256"] == exp.frame_sha256(_frame(3, level=2))

    broken = deepcopy(occurrence)
    broken["prefix_steps"][1]["level_boundary"] = False
    excluded = exp.replay_prefix_once(broken, BoundaryEnv, _execute)
    assert excluded["replayable"] is False
    assert excluded["exclusion_reason"] == "level_boundary_mismatch:1"


# REQ-ARC-WMTE-7611; SCENARIO-ARC-WMTE-7611-STABILITY.
def test_protocol_fixtures_cover_all_required_controls():
    fixtures = exp.measure_protocol_fixtures()
    assert fixtures["matched_support_ready_score"] == 1
    assert fixtures["fixture_results"] == {
        "coordinate_action": True,
        "same_action_level_boundary": True,
        "singleton": True,
        "stateful_reset": True,
        "two_hidden_histories": True,
        "unstable_prefix": True,
    }
    assert fixtures["verdict_class"] == "circular_positive"


# REQ-ARC-WMTE-7611; SCENARIO-ARC-WMTE-7611-DENOMINATORS.
def test_reducer_keeps_natural_and_intervention_denominators_separate():
    pair = _pair()
    selection = exp.select_matched_prefixes(
        [_episode("su15", exp.SEEDS[0], 1), _episode("su15", exp.SEEDS[1], 2)],
        max_per_game=20,
    )
    witness = exp.replay_matched_pair(pair, HiddenHistoryEnv, _execute)
    unstable = deepcopy(witness)
    unstable["right_within_prefix_stable"] = False
    unstable["history_disambiguation_witness"] = False
    unstable["replay_exclusion"] = "unstable_prefix"
    reduced = exp.reduce_measurements(selection, [witness, unstable])
    assert reduced["natural_trajectory_denominator"]["selected_matched_keys"] == 1
    assert reduced["replay_intervention_denominator"] == {
        "intended_matched_keys": 2,
        "replayed_matched_keys": 2,
        "stable_matched_keys": 1,
        "excluded_unreplayable_keys": 0,
        "excluded_unstable_keys": 1,
        "fresh_environment_count": 8,
    }
    assert reduced["primary_h0_vs_h1"]["witness_numerator"] == 1
    assert reduced["primary_h0_vs_h1"]["stable_denominator"] == 1
    assert reduced["descriptive_history_lengths"] == [2, 4]


# REQ-ARC-WMTE-7611; SCENARIO-ARC-WMTE-7611-TERMINAL.
def test_test_artifact_has_required_principles_and_validates(tmp_path):
    artifact = exp.build_test_artifact(tmp_path)
    assert exp.validate_artifact(artifact) == []
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "null"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["no_model_load"] is True
    assert artifact["matched_support_ready_score"] == 1
    assert artifact["sample_size_budget"]["censored_independent_units"] == 0
    assert artifact["sample_size_budget"]["censored_prefix_occurrences"] == 0
    assert artifact["acceptance_gate_results"]["benefit"]["result"] is False
    assert set(exp.REQUIRED_FIELD_PRINCIPLES) <= set(artifact["field_principles"])
    assert {row["arm"] for row in artifact["rows"]} == {"h0", "h1", "h2", "h4"}

    damaged = deepcopy(artifact)
    damaged["MODEL_SPECS"] = [{"model": "invented"}]
    damaged["rows"][0]["denominator"] += 1
    damaged["reproducibility_checksum"] = "bad"
    errors = exp.validate_artifact(damaged)
    assert "MODEL_SPECS" in errors
    assert "row_operand_mismatch:0" in errors
    assert "reproducibility_checksum" in errors


# REQ-ARC-WMTE-7611; SCENARIO-ARC-WMTE-7611-TERMINAL.
def test_blocked_artifact_names_every_failed_operand(tmp_path):
    failed = {
        "check": "registry_game_present",
        "upstream": "ops/arc_solve_registry.yaml",
        "path": "ops/arc_solve_registry.yaml",
        "field": "games.su15",
        "operator": "present",
        "expected": True,
        "observed": False,
    }
    artifact = exp.build_blocked_artifact(tmp_path, [failed], duration_s=0.25)
    assert artifact["honest_verdict"] == "complete_blocked_registry_game_present"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"] == [failed]
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["actual_inference_substrate_class"] == "blocked_no_run"
    assert exp.validate_artifact(artifact) == []


# REQ-ARC-WMTE-7611; SCENARIO-ARC-WMTE-7611-TERMINAL.
def test_cold_replay_and_reproducibility_bind_rows(tmp_path):
    artifact = exp.build_test_artifact(tmp_path)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_replay(path) == []
    assert exp.independent_replay(path) == []
    changed = deepcopy(artifact)
    changed["rows"][0]["numerator"] += 1
    assert exp.reproducibility_checksum(changed) != artifact["reproducibility_checksum"]


# REQ-ARC-WMTE-7611; SCENARIO-ARC-WMTE-7611-TERMINAL.
def test_command_manifests_are_scoped_and_private(tmp_path):
    commands = exp.build_validation_commands(exp.repo_root(), tmp_path)
    assert tuple(command.name for command in commands) == exp.REQUIRED_SCOPED_CHECKS
    flat = " ".join(part for command in commands for part in command.argv)
    assert "tests/python/test_experiment_7611_v664_arc_matched_support.py" in flat
    assert "pytest tests/python" not in flat
    assert str(tmp_path) in flat

    e2e = exp.build_e2e_commands(exp.repo_root(), tmp_path)
    assert {command.name for command in e2e} == {
        "e2e_009",
        "e2e_010",
        "e2e_011",
        "e2e_012",
        "e2e_013",
        "foreign_cwd_llm_off_smoke",
    }
    assert all(str(tmp_path) in " ".join(command.argv) for command in e2e)


def test_episode_validation_rejects_forbidden_inputs():
    raw = _episode("su15", exp.SEEDS[0], 1)
    assert exp.validate_natural_episode(raw) == []
    raw["game_source_read"] = True
    raw["steps"][2]["action"]["kind"] = 9
    errors = exp.validate_natural_episode(raw)
    assert "game_source_read" in errors
    assert "illegal_target_action:2" in errors


def test_parse_args_supports_reader_modes(tmp_path):
    args = exp.parse_args(["--root", str(tmp_path), "--date", "20260924"])
    assert args.root == tmp_path
    assert args.date == "20260924"
    reader = exp.parse_args(["--cold-replay", str(tmp_path / "a.json")])
    assert reader.cold_replay == tmp_path / "a.json"
    with pytest.raises(SystemExit):
        exp.parse_args(["--date", "bad"])


def test_action_normalization_progress_and_raw_validation_branches(capsys):
    assert exp.canonical_action({"kind": 6, "data": {"x": 2, "y": 3}})["coordinates"] == {
        "x": 2,
        "y": 3,
    }
    assert exp.action_is_legal({}, {"kind": "RESET"}) is True
    assert len(exp._history_items(_episode("su15", exp.SEEDS[0], 1)["steps"][:1])) == 1
    exp.progress(0.0, "fixture", "complete", units=1)
    assert "phase=fixture event=complete" in capsys.readouterr().out

    raw = _episode("su15", exp.SEEDS[0], 1)
    raw["game"] = "unknown"
    raw["seed"] = 1
    raw["steps"] = "bad"
    errors = exp.validate_natural_episode(raw)
    assert {"game_not_frozen", "seed_not_frozen", "steps_invalid"} <= set(errors)

    raw = _episode("su15", exp.SEEDS[0], 1)
    raw["steps"][1] = {"action_index": 9}
    raw["steps"][2]["action"] = None
    errors = exp.validate_natural_episode(raw)
    assert "action_index_invalid:1" in errors
    assert "action_invalid:2" in errors
    with pytest.raises(ValueError, match="raw_episode_invalid"):
        exp.select_matched_prefixes([raw])

    raw = _episode("su15", exp.SEEDS[0], 1)
    raw["steps"][1]["action"] = _action("RESET")
    reset_selection = exp.select_matched_prefixes([raw])
    assert reset_selection["natural_trajectory_denominator"]["invalid_target_occurrences"] == 1


def test_replay_prefix_explicit_exclusion_paths():
    pair = _pair()
    original = deepcopy(pair["prefixes"][0])
    original["game"] = pair["game"]

    occurrence = deepcopy(original)
    occurrence["prefix_steps"][0]["observation"] = _frame(0)
    assert (
        exp.replay_prefix_once(occurrence, HiddenHistoryEnv, _execute)["exclusion_reason"]
        == "observation_mismatch:0"
    )

    occurrence = deepcopy(original)
    occurrence["prefix_steps"][0]["reset_boundary"] = False
    assert (
        exp.replay_prefix_once(occurrence, HiddenHistoryEnv, _execute)["exclusion_reason"]
        == "reset_boundary_mismatch:0"
    )

    occurrence = deepcopy(original)
    occurrence["prefix_steps"][0]["outcome"] = _frame(99)
    assert (
        exp.replay_prefix_once(occurrence, HiddenHistoryEnv, _execute)["exclusion_reason"]
        == "outcome_mismatch:0"
    )

    occurrence = deepcopy(original)
    occurrence["target_observation"] = _frame(99, legal=("6",))
    assert (
        exp.replay_prefix_once(occurrence, HiddenHistoryEnv, _execute)["exclusion_reason"]
        == "target_observation_mismatch"
    )

    occurrence = deepcopy(original)
    occurrence["target_action"] = _action(9)
    assert (
        exp.replay_prefix_once(occurrence, HiddenHistoryEnv, _execute)["exclusion_reason"]
        == "target_action_not_legal"
    )

    occurrence = deepcopy(original)
    occurrence["target_key"] = "bad"
    assert (
        exp.replay_prefix_once(occurrence, HiddenHistoryEnv, _execute)["exclusion_reason"]
        == "target_key_mismatch"
    )

    def terminal_execute(env, action):
        if action["kind"] == 6:
            return None
        return _execute(env, action)

    terminal = exp.replay_prefix_once(original, HiddenHistoryEnv, terminal_execute)
    assert terminal["replayable"] is False
    assert terminal["exclusion_reason"] == "target_terminated_without_frame"
    unreplayable = exp.replay_matched_pair(pair, HiddenHistoryEnv, terminal_execute)
    assert unreplayable["replay_exclusion"] == "unreplayable_prefix"


def test_artifact_validator_defensive_errors(tmp_path):
    base = exp.build_test_artifact(tmp_path)
    damaged = deepcopy(base)
    damaged.update(
        {
            "experiment_id": 0,
            "honest_verdict": "bad",
            "verdict_class": "bad",
            "model_invoked": True,
            "invocation_counts": {"calls": 1},
            "field_principles": {},
            "acceptance_gate_results": {},
        }
    )
    errors = exp.validate_artifact(damaged)
    assert {
        "identity",
        "honest_verdict",
        "verdict_class",
        "model_invoked",
        "invocation_counts",
        "field_principles",
        "acceptance_gate_results",
    } <= set(errors)

    damaged = deepcopy(base)
    damaged["acceptance_gate_results"]["validity"]["principle"] = ""
    assert "gate_principles" in exp.validate_artifact(damaged)

    damaged = deepcopy(base)
    damaged["verdict_class"] = "blocked"
    damaged["gate_check_summary"] = []
    damaged["actual_inference_substrate_class"] = "no_model_load"
    errors = exp.validate_artifact(damaged)
    assert "blocked_gate_check_summary" in errors
    assert "blocked_substrate" in errors

    damaged = deepcopy(base)
    damaged["rows"] = ["bad"]
    assert "row_invalid:0" in exp.validate_artifact(damaged)


def test_independent_reader_detects_rate_and_replay_count(tmp_path):
    artifact = exp.build_test_artifact(tmp_path)
    artifact["rows"][0]["rate"] = 0.5
    artifact["replay_intervention_denominator"]["fresh_environment_count"] = 3
    path = tmp_path / "damaged.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    errors = exp.independent_replay(path)
    assert "row_rate_mismatch:0" in errors
    assert "fresh_environment_count" in errors


def test_preconditions_plans_and_command_helpers(tmp_path):
    private = exp.repo_root() / "tests/python"
    checks, hashes, registry = exp.collect_preconditions(exp.repo_root(), private)
    assert all(row["passed"] for row in checks if row["check"] != "resource_ownership")
    assert next(row for row in checks if row["check"] == "resource_ownership")["passed"] is False
    assert exp.MODULE_REL.as_posix() in hashes
    assert {row["game"] for row in registry} == set(exp.GAMES)

    plan = exp.frozen_episode_plan()
    assert len(plan) == 12
    episode_commands = exp.build_episode_commands(exp.repo_root(), tmp_path)
    assert len(episode_commands) == 12
    assert all("--run-episode" in command.argv for command in episode_commands)

    command = exp.CommandSpec(
        "prepare",
        (
            "/bin/true",
            f"--basetemp={tmp_path / 'base'}",
            f"COVERAGE_FILE={tmp_path / 'coverage' / '.coverage'}",
            "--output",
            str(tmp_path / "output" / "x.json"),
            "-C",
            str(tmp_path / "cwd"),
        ),
        "fixture",
    )
    exp.prepare_command_parent(command)
    assert (tmp_path / "base").is_dir()
    assert (tmp_path / "coverage").is_dir()
    assert (tmp_path / "output").is_dir()
    assert (tmp_path / "cwd").is_dir()

    receipt = exp.run_prepared_commands(
        exp.repo_root(),
        [exp.CommandSpec("true", ("/bin/true",), "fixture")],
        log_dir=tmp_path / "logs",
    )
    assert receipt[0]["passed"] is True


def test_terminal_manifest_helpers_and_supervisor_reduction(tmp_path):
    terminal = exp.build_terminal_commands(exp.repo_root(), tmp_path / "candidate.json")
    assert len(terminal) == 5
    receipts = [{"name": row.name, "passed": True} for row in terminal]
    assert exp._all_passed(receipts, [row.name for row in terminal]) is True
    assert exp._all_passed(receipts, ["missing"]) is False

    episodes = [_episode("su15", exp.SEEDS[0], 1)]
    episodes[0]["trajectory_supervisor"]["would_have_arm_outcomes"] = {
        "arm": {"fired": 2, "helped": 1},
        "ignored": "bad",
    }
    assert exp._supervisor_scope(episodes)["refinement_supported"] is True

    selection = exp.select_matched_prefixes(
        [_episode("su15", exp.SEEDS[0], 1), _episode("su15", exp.SEEDS[1], 2)]
    )
    summary = exp._selection_summary(selection)
    assert summary["selected_pairs"][0]["prefixes"][0]["prefix_action_count"] == 2
    span = exp._phase_span("fixture", 1.0, 0.0)
    assert span["phase"] == "fixture"
    hashes = exp._source_hashes(
        exp.repo_root(),
        {"a": "sha256:a"},
        [{"path": "raw.json", "sha256": "sha256:b"}],
    )
    assert hashes["raw.json"] == "sha256:b"


def test_main_reader_and_dispatch_modes(tmp_path, monkeypatch, capsys):
    artifact = exp.build_test_artifact(tmp_path)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--validate", str(path)]) == 0
    assert exp.main(["--cold-replay", str(path)]) == 0
    assert exp.main(["--independent-reduce", str(path)]) == 0
    assert '"errors": []' in capsys.readouterr().out
    with pytest.raises(SystemExit, match="requires"):
        exp.main(["--run-episode"])

    calls = []
    monkeypatch.setattr(exp, "run_live_episode", lambda *args: calls.append(args))
    assert (
        exp.main(
            [
                "--run-episode",
                "--game",
                "su15",
                "--seed",
                str(exp.SEEDS[0]),
                "--raw-output",
                str(tmp_path / "raw.json"),
            ]
        )
        == 0
    )
    monkeypatch.setattr(exp, "run_experiment", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert exp.main([]) == 0
    assert len(calls) == 2

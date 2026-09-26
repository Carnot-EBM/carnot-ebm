"""REQ-REPORT-7681 and REQ-ARC-PROBE-7681 live episode contracts."""

from __future__ import annotations

from types import SimpleNamespace
import json
from pathlib import Path
import time

import numpy as np
import pytest

from carnot.agentic.arc_probe_protocol import ArcProbeProtocol
import carnot.experiment_7681_v669_arc_live_probes as experiment
from carnot.experiment_7681_v669_arc_live_probes import (
    block_check,
    reduce_episode,
    select_target,
)


def frame(seed: int) -> SimpleNamespace:
    """Expose public pixels and legal actions, with no hidden game state."""
    grid = np.zeros((8, 8), dtype=int)
    grid[2:6, 2:6] = seed % 4
    return SimpleNamespace(frame=[grid.tolist()], levels_completed=0, available_actions=[1, 2, 3])


def test_select_target_freezes_registry_novelty_and_adapter_exclusion() -> None:
    """SCENARIO-REPORT-7681-LIVE: selection uses only SDK IDs and registry levels."""
    roster = ["aa11", "bb22", "cc33"]
    registry = [
        {"game": "aa11", "levels_reproduced": 2},
        {"game": "bb22", "levels_reproduced": 4, "adapter": "supplied"},
        {"game": "cc33", "levels_reproduced": 1},
    ]
    selection = select_target(roster, registry, salt="fixed")
    assert selection["game"] in {"aa11", "cc33"}
    assert selection["target_level"] == selection["levels_reproduced"] + 1
    assert "bb22" in selection["excluded_adapter_games"]
    assert selection == select_target(reversed(roster), reversed(registry), salt="fixed")
    with pytest.raises(ValueError, match="no_eligible_game"):
        select_target(["bb22"], registry, salt="fixed")


def test_guided_choice_records_two_unexecuted_shadows() -> None:
    """SCENARIO-ARC-PROBE-7681-SHADOW: one live history yields three legal choices."""
    probe = ArcProbeProtocol("guided")
    for seed, action in ((0, 1), (1, 2)):
        probe.record_action(frame(seed), (action, None))
        probe.observe(frame(seed + 1))
    chosen = probe.select(frame(2), [1, 2, 3], (1, None))
    receipt = probe.diagnostics()["decisions"][-1]
    assert receipt["current_policy_shadow"]["action"] == 1
    assert receipt["novelty_only_shadow"]["action"] in {1, 2, 3}
    assert receipt["factual_probe_action"] == chosen[0]
    assert receipt["current_policy_shadow"]["executed"] is False
    assert receipt["novelty_only_shadow"]["executed"] is False
    assert receipt["compute_cost_s"] >= 0
    assert receipt["route_reachable"] is True


def test_zero_engine_episode_reduces_to_null_without_solve_credit() -> None:
    """SCENARIO-ARC-PROBE-7681-ZERO-ENGINE: valid reachability is not a solve."""
    events = [
        {"event": "decision", "action_index": 1, "factual_action": 1, "admitted": False},
        {"event": "observation", "action_index": 1, "level": 0, "state": "NOT_FINISHED"},
        {"event": "decision", "action_index": 2, "factual_action": 2, "admitted": True},
        {"event": "observation", "action_index": 2, "level": 0, "state": "NOT_FINISHED"},
    ]
    reduced = reduce_episode(events, baseline_level=0, registry_level=0)
    assert reduced["actions"] == 2
    assert reduced["probes_admitted"] == 1
    assert reduced["sdk_confirmed_level_ups"] == 0
    assert reduced["credited_solve"] is False
    with pytest.raises(ValueError, match="unpaired_decision"):
        reduce_episode(events[:-1], baseline_level=0, registry_level=0)


def test_block_receipt_has_exact_operands() -> None:
    """SCENARIO-REPORT-7681-BLOCKED: absent authentication is external custody."""
    check = block_check("sdk_auth", "arc_sdk", "/tmp/key", "usable", True, False)
    assert check == {
        "check": "sdk_auth",
        "upstream": "arc_sdk",
        "path": "/tmp/key",
        "field": "usable",
        "operator": "==",
        "expected": True,
        "observed": False,
        "passed": False,
    }


def test_selection_excludes_cleared_and_unknown_games() -> None:
    """SCENARIO-REPORT-7681-BLOCKED: completed catalogue games are ineligible."""
    registry = [{"game": "aa11", "levels_reproduced": 6, "full_game_clear": True}]
    with pytest.raises(ValueError, match="no_eligible_game"):
        select_target(["unknown", "aa11"], registry)


def test_reducer_rejects_duplicate_and_unmatched_observations() -> None:
    """SCENARIO-REPORT-7681-TERMINAL: malformed event order fails cold reduction."""
    decision = {"event": "decision", "action_index": 1, "factual_action": 1}
    observation = {"event": "observation", "action_index": 1, "level": 0}
    with pytest.raises(ValueError, match="unpaired_decision"):
        reduce_episode([decision, decision, observation], baseline_level=0, registry_level=0)
    with pytest.raises(ValueError, match="observation_without_decision"):
        reduce_episode([observation], baseline_level=0, registry_level=0)


def test_preconditions_preserve_missing_bytes_and_sdk_failure(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7681-BLOCKED: missing files and SDK errors stay literal."""
    import carnot.agentic.arc_solver_kit as kit
    import carnot.experiment_7630_v666_cuda_ownership as gpu
    import carnot.inference.sota_models as models

    monkeypatch.setattr(experiment, "INPUTS", ("absent",))
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/arc_solve_registry.yaml").write_text("games: []\n")
    predecessor = tmp_path / "results/experiment_7667_v668_arc_live_goal_observation.json"
    predecessor.parent.mkdir()
    predecessor.write_text("{}\n")
    monkeypatch.setattr(kit, "offline_arcade", lambda: (_ for _ in ()).throw(RuntimeError("auth")))
    monkeypatch.setattr(models, "cached_current_model", lambda: None)
    monkeypatch.setattr(gpu, "_current_inventory", lambda: [])
    monkeypatch.setattr(gpu, "select_owned_capacity", lambda _inv, _owner: (None, []))
    checks, hashes, context = experiment.collect_preconditions(tmp_path, time.monotonic())
    assert any(row["check"] == "sdk_catalogue" and not row["passed"] for row in checks)
    assert context["sdk_error"] == "RuntimeError: auth"
    assert "absent" in hashes["missing_evidence"]
    assert hashes["pre_gate_receipts"]


def test_blocked_artifact_reduces_exact_custody(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7681-BLOCKED: missing target has zero claimed science."""
    check = block_check("eligible_novel_target", "registry", str(tmp_path), "eligible", True, False)
    context = {
        "registry_games": [{"game": "aa11", "levels_reproduced": 6, "full_game_clear": True}],
        "sdk_roster": ["aa11"], "sdk_error": None, "eligible_game_ids": [],
        "model_path": None, "model_sha256": None, "gpu_uuid": None,
    }
    value = experiment.blocked_artifact(
        [check], {"producers": {}, "pre_gate_receipts": {}, "missing_evidence": []},
        context, run_date="20260926", started=time.monotonic(), phase_spans=[]
    )
    assert value["honest_verdict"] == "complete_blocked_eligible_novel_target"
    assert value["registry_precheck"]["excluded_full_clear_games"] == ["aa11"]
    assert value["arc_live_measurement_complete_score"] == 0
    assert all(not gate["passed"] for gate in value["acceptance_gate_results"].values())


def test_cold_reduce_reads_factual_log_only(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7681-TERMINAL: a fresh process can reject changed rows."""
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    events = [
        {"event": "decision", "action_index": 1, "factual_action": 1, "admitted": True},
        {"event": "observation", "action_index": 1, "level": 1},
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events))
    candidate = tmp_path / "candidate.json"
    value = {
        "rows": [{"event_log": "events.jsonl", "start_level": 0, "registry_level": 0}],
        "independent_reduction": reduce_episode(events, baseline_level=0, registry_level=0),
    }
    candidate.write_text(json.dumps(value))
    assert experiment.main(["--date", "20260926", "--cold-reduce", str(candidate)]) == 0
    value["independent_reduction"]["actions"] = 99
    candidate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="cold_reduction_mismatch"):
        experiment.main(["--date", "20260926", "--cold-reduce", str(candidate)])

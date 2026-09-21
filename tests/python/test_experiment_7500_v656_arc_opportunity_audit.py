"""Tests for REQ-ARC-WMTE-7500 and its opportunity-audit scenarios."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_7500_v656_arc_opportunity_audit as audit


def _event(
    episode_id: str,
    decision_id: str,
    event: str,
    tick: int,
    *,
    clock: str = "clock-a",
    process_id: int = 7,
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": "run-a",
        "process_id": process_id,
        "episode_id": episode_id,
        "decision_id": decision_id,
        "seam": "supervisor_arm_selection",
        "work_class": "replaceable_decision",
        "clock_identity": clock,
        "event": event,
        "event_monotonic_ns": tick,
    }
    if event == "stage_start":
        row["interval_start_monotonic_ns"] = tick
    if event in {"stage_end", "stage_terminal"}:
        row["interval_end_monotonic_ns"] = tick
        row["disposition"] = "completed"
    row.update(extra)
    return row


def _decision(
    episode_id: str,
    decision_id: str,
    left: int,
    right: int | None,
    *,
    selected: str,
    eligibility: bool | None,
) -> list[dict[str, object]]:
    rows = [
        _event(episode_id, decision_id, "stage_start", left),
        _event(
            episode_id,
            decision_id,
            "eligible_candidate_set",
            left + 1,
            candidates=[
                {"stable_candidate_id": selected, "rank": 0, "eligibility": eligibility}
            ],
            candidate_ids=[selected],
        ),
        _event(
            episode_id,
            decision_id,
            "selection",
            left + 2,
            selected_candidate_ids=[selected],
            supervisor_fired=selected != "no_redirect",
            applied_redirection=selected != "no_redirect",
        ),
    ]
    if right is not None:
        rows.append(_event(episode_id, decision_id, "stage_end", right))
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _panel_a_fixture(root: Path) -> None:
    games = ("sk48", "tr87", "s5i5", "lp85", "lf52", "cn04")
    seeds = (65501, 65502, 65503)
    rows: list[dict[str, object]] = []
    shards: list[dict[str, object]] = []
    for game in games:
        for seed in seeds:
            episode_id = f"panel-a:{game}:seed-{seed}"
            path = Path("results/raw/panel-a") / f"{game}-{seed}.jsonl"
            absolute = root / path
            absolute.parent.mkdir(parents=True, exist_ok=True)
            events = _decision(
                episode_id,
                f"{episode_id}:supervisor:0",
                100,
                120,
                selected="no_redirect",
                eligibility=None,
            )
            absolute.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
                encoding="utf-8",
            )
            shards.append(
                {
                    "episode_id": episode_id,
                    "path": path.as_posix(),
                    "sha256": _sha256(absolute),
                    "row_count": len(events),
                }
            )
            rows.append(
                {
                    "panel": "A",
                    "episode_id": episode_id,
                    "game": game,
                    "seed": seed,
                    "disposition": "complete",
                    "elapsed_s": 0.0000001,
                    "exclusive_cost": {"episode_start_ns": 90, "episode_end_ns": 190},
                    "solve_provenance": "live_agent_self_discovery",
                    "offline_reproduced": False,
                }
            )
    artifact = {
        "schema": "carnot.exp7485.v655.arc_cost_panel_a.v1",
        "experiment_id": "exp7485-v655-arc-cost-panel-a",
        "arc_panel_a_complete_score": 1,
        "verdict_class": "null",
        "honest_verdict": "complete_null_live_arc_cost_panel_a",
        "flagged_adversarial": False,
        "rows": rows,
        "seam_event_shards": shards,
        "model_specs": [{"sha256": "sha256:model", "runtime_settings": {"n_ctx": 4096}}],
        "source_artifact_hashes": {
            "python/carnot/agentic/arc_decision_telemetry.py": {
                "sha256": "sha256:observer"
            },
            "python/carnot/agentic/arc_competition_agent.py": {"sha256": "sha256:policy"},
            "python/carnot/experiment_7478_v655_arc_interval_protocol.py": {
                "sha256": "sha256:protocol"
            },
        },
        "solve_provenance": "live_agent_self_discovery",
    }
    _write_json(root / audit.PANEL_A_PATH, artifact)


def test_exclusive_union_is_identity_scoped_and_incomplete_is_not_lower_bound() -> None:
    """SCENARIO-ARC-WMTE-7500-EXCLUSIVE-UNION."""

    episode_id = "panel-a:g1:seed-1"
    events = _decision(
        episode_id, "complete", 100, 150, selected="redirect", eligibility=True
    )
    events.extend(deepcopy(events))
    events.extend(
        _decision(episode_id, "nested", 110, 130, selected="redirect", eligibility=True)
    )
    events.extend(
        _decision(episode_id, "incomplete", 160, None, selected="redirect", eligibility=True)
    )
    events.extend(
        [
            _event(episode_id, "wrong-clock", "stage_start", 170, clock="clock-a"),
            _event(episode_id, "wrong-clock", "stage_end", 190, clock="clock-b"),
        ]
    )

    row = audit.reduce_episode(
        {"episode_id": episode_id, "game": "g1", "seed": 1, "disposition": "complete"},
        events,
        episode_start_ns=90,
        episode_end_ns=200,
    )

    assert row["replaceable_lower_ns"] == 50
    assert row["incomplete_interval_count"] == 1
    assert row["mismatched_clock_count"] == 1
    assert row["duplicate_event_count"] == len(events[:4])
    assert row["bounds_valid"] is True


def test_supervisor_eligibility_requires_true_selection_and_runtime() -> None:
    """SCENARIO-ARC-WMTE-7500-SUPERVISOR-ELIGIBILITY."""

    episode_id = "panel-a:g1:seed-1"
    events = [
        *_decision(episode_id, "true", 10, 20, selected="redirect", eligibility=True),
        *_decision(episode_id, "false", 30, 50, selected="redirect", eligibility=False),
        *_decision(episode_id, "unknown", 60, 80, selected="redirect", eligibility=None),
        *_decision(episode_id, "abstain", 90, 100, selected="no_redirect", eligibility=None),
        *_decision(episode_id, "unfinished", 110, None, selected="redirect", eligibility=True),
    ]

    reduced = audit.reduce_supervisor_episode(episode_id, events)

    assert reduced["selection_exposure_count"] == 5
    assert reduced["explicit_true_selected_count"] == 2
    assert reduced["explicit_false_selected_count"] == 1
    assert reduced["unknown_eligibility_selected_count"] == 2
    assert reduced["abstention_count"] == 1
    assert reduced["triggered_opportunity_count"] == 1
    assert reduced["eligible_service_lower_ns"] == 10
    assert reduced["eligible_service_upper_ns"] == 10
    assert [row["decision_id"] for row in reduced["intervention_ledger"]] == ["true"]


def test_pooling_requires_versions_episode_floor_and_game_multiplicity() -> None:
    """SCENARIO-ARC-WMTE-7500-POOLING."""

    identity = {
        "observer": "sha256:observer",
        "policy": "sha256:policy",
        "model": "sha256:model",
        "protocol": "sha256:protocol",
    }
    rows = [
        {"panel": "A" if index < 15 else "B", "game": f"g{index // 3}", "valid": True}
        for index in range(30)
    ]
    pooled = audit.evaluate_pooling(
        rows,
        {"A": identity, "B": deepcopy(identity)},
        panel_states={"A": "valid", "B": "valid"},
    )
    assert pooled["pooling_allowed"] is True
    assert pooled["valid_episode_count"] == 30
    assert pooled["valid_game_count"] == 10

    drifted = deepcopy(identity)
    drifted["policy"] = "sha256:changed"
    assert (
        audit.evaluate_pooling(
            rows,
            {"A": identity, "B": drifted},
            panel_states={"A": "valid", "B": "valid"},
        )["pooling_allowed"]
        is False
    )
    assert (
        audit.evaluate_pooling(
            rows[:27],
            {"A": identity, "B": identity},
            panel_states={"A": "valid", "B": "valid"},
        )["pooling_allowed"]
        is False
    )


def test_missing_panel_b_stays_explicit_without_fabricated_rows(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7500-MISSING-PANEL."""

    _panel_a_fixture(tmp_path)
    reduction = audit.reduce_upstreams(tmp_path)

    assert reduction["panel_results"]["A"]["state"] == "valid"
    assert reduction["panel_results"]["B"]["state"] == "blocked_missing"
    assert reduction["panel_results"]["B"]["rows"] == []
    assert len(reduction["rows"]) == 18
    assert reduction["pooling"]["pooling_allowed"] is False
    assert reduction["sample_size_budget"]["planned_independent_units"] == 36
    assert reduction["sample_size_budget"]["unstarted_independent_units"] == 18


def test_terminal_shaped_audit_is_complete_but_cross_panel_claim_is_blocked(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7500-ZERO-OPPORTUNITY and TERMINAL."""

    _panel_a_fixture(tmp_path)
    artifact = audit.build_artifact_for_test(tmp_path)

    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["arc_opportunity_audit_complete_score"] == 1
    assert artifact["opportunity_present_score"] == 0
    assert artifact["intervention_ledger"] == []
    assert artifact["efficacy_estimate"] is None
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["amdahl_upper_bound"]["speedup_upper"] == 1.0
    assert artifact["supervisor_disposition"]["selected_arms"] == {"no_redirect": 18}
    assert not audit.validate_artifact(artifact, root=tmp_path, require_terminal=False)
    assert all(artifact["field_principles"].get(key) for key in artifact)
    assert all(row.get("principle") for row in artifact["acceptance_gate_results"])


def test_tampered_shard_fails_independent_replay(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7500 binds exact upstream and shard bytes."""

    _panel_a_fixture(tmp_path)
    artifact = audit.build_artifact_for_test(tmp_path)
    shard = next((tmp_path / "results/raw/panel-a").glob("*.jsonl"))
    shard.write_text(shard.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    replay = audit.independent_reduce(artifact, tmp_path)

    assert replay["matches_declared"] is False
    assert any("shard_hash_mismatch" in error for error in replay["errors"])
    assert "independent_reduction_mismatch" in audit.validate_artifact(
        artifact, root=tmp_path, require_terminal=False
    )


def test_gate_rejects_unknown_operator() -> None:
    """REQ-ARC-WMTE-7500 gates use explicit, reproducible comparisons."""

    with pytest.raises(ValueError, match="unsupported_gate_operator"):
        audit.gate(
            "bad",
            "validity",
            1,
            1,
            op="!=",
            upstream="fixture",
            field="value",
            principle="An unknown comparison could silently invert a gate.",
        )

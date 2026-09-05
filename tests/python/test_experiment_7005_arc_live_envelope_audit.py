"""Real ARC live-envelope audit tests.

Spec refs: REQ-ARC-WMTE-7005 and SCENARIO-ARC-WMTE-7005-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7005_arc_live_envelope_audit as exp
from carnot.agentic.arc_active_probe import _exact_action_delta_candidate
from carnot.agentic.arc_executable_world_model import Transition
from carnot.agentic.arc_producer_evidence import (
    EngineEvidenceTransaction,
    EvidencePublishInterrupted,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_SOURCE = b"""import numpy as np

def engine(grid, action, data):
    out = np.asarray(grid).copy()
    if int(action) == 6 and data == {"x": 1, "y": 0}:
        out[0, 1] = 7
    return out

def is_level_complete(grid):
    return False
"""


def _transition(
    value: int,
    *,
    data: dict[str, int] | None = None,
    level: int = 0,
    target: bool = True,
) -> dict:
    return {
        "grid": [[0, 0], [0, 0]],
        "action": 6,
        "data": data if data is not None else {"x": value, "y": 0},
        "next_grid": [[0, value], [0, 0]] if target else None,
        "level_before": level,
        "level_after": level,
    }


def _publish(
    store: Path,
    *,
    game: str,
    stamp: str,
    transitions: list[dict],
    source_kind: str | None = "live_agent_attempts",
    run_suffix: str = "abcdef123456",
    failpoint: str | None = None,
) -> dict | None:
    transaction = EngineEvidenceTransaction.begin(
        store_root=store,
        game=game,
        raw_prompt=f"prompt {stamp}".encode(),
        transitions=transitions,
        transition_source_kind=source_kind,
        environment_receipt={
            "schema": "carnot.arc.environment_receipt.v1",
            "game": game,
            "policy": "E3AgentPolicy",
            "transition_count": len(transitions),
        },
        run_id=f"arc-{stamp}-{run_suffix}",
        timestamp=stamp,
    )
    try:
        return transaction.publish(
            ENGINE_SOURCE,
            writer="write_world_model",
            model="Qwen3.8-27B",
            timestamp=stamp,
            failpoint=failpoint,
        ).row
    except EvidencePublishInterrupted:
        return None


def test_req_arc_wmte_7005_spec_precedes_implementation() -> None:
    """REQ-ARC-WMTE-7005 defines every live-envelope evidence surface."""
    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-ARC-WMTE-7005", 1)[1].split("### REQ-", 1)[0]
    for marker in (
        "SCENARIO-ARC-WMTE-7005-ELIGIBILITY-AND-SELECTION",
        "SCENARIO-ARC-WMTE-7005-HASH-REPLAY",
        "SCENARIO-ARC-WMTE-7005-SPLIT-FREEZE",
        "SCENARIO-ARC-WMTE-7005-NO-LEAKAGE",
        "SCENARIO-ARC-WMTE-7005-CONTROLS-AND-METRICS",
        "SCENARIO-ARC-WMTE-7005-FRESH-PROCESS",
        "SCENARIO-ARC-WMTE-7005-TERMINAL-CLAIMS",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_7005_eligibility_orders_without_quality_and_rejects_bad_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7005-ELIGIBILITY-AND-SELECTION preserves every rejection."""
    store = tmp_path / "results" / "arc_e3"
    before = _publish(
        store,
        game="aa01",
        stamp="20260904T190000_000000",
        transitions=[_transition(1)],
        run_suffix="000000000001",
    )
    earliest = _publish(
        store,
        game="aa01",
        stamp="20260904T210000_000000",
        transitions=[_transition(1)],
        run_suffix="000000000002",
    )
    _publish(
        store,
        game="aa01",
        stamp="20260904T220000_000000",
        transitions=[_transition(1), _transition(2)],
        run_suffix="000000000003",
    )
    _publish(
        store,
        game="fixture7005",
        stamp="20260904T204000_000000",
        transitions=[_transition(1)],
        run_suffix="000000000004",
    )
    _publish(
        store,
        game="aa01",
        stamp="20260904T221000_000000",
        transitions=[_transition(3)],
        source_kind="offline_bfs",
        run_suffix="000000000005",
    )
    duplicate = _publish(
        store,
        game="aa01",
        stamp="20260904T222000_000000",
        transitions=[_transition(3)],
        run_suffix="000000000006",
    )
    assert before and earliest and duplicate
    manifest = store / "aa01" / "attempts" / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(duplicate, sort_keys=True, separators=(",", ":")) + "\n")
        handle.write(json.dumps({"ts": "20260903T000000_000000", "file": "old.py"}) + "\n")
    _publish(
        store,
        game="bb02",
        stamp="20260904T223000_000000",
        transitions=[_transition(4)],
        run_suffix="000000000007",
        failpoint="after_bundle_publish",
    )
    tampered = _publish(
        store,
        game="cc03",
        stamp="20260904T224000_000000",
        transitions=[_transition(5)],
        run_suffix="000000000008",
    )
    assert tampered
    (store / tampered["engine"]["path"]).write_bytes(b"changed")

    replay = exp.enumerate_envelopes(store)
    selected = exp.select_earliest_eligible(replay["envelope_eligibility_rows"])
    classes = {row["classification"] for row in replay["envelope_eligibility_rows"]}

    assert selected is not None
    assert selected["run_id"] == earliest["run_id"]
    assert selected["quality_fields_consulted"] == []
    assert {
        "pre_exp6993",
        "development_fixture",
        "invalid_transition_source",
        "duplicate_run_id",
        "legacy",
        "interrupted_unpublished",
        "hash_mismatch",
    } <= classes
    assert len(replay["envelope_inventory_rows"]) == len(replay["envelope_eligibility_rows"])
    assert all(row["terminal"] for row in replay["envelope_eligibility_rows"])


def test_scenario_7005_recomputes_all_producer_hashes(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7005-HASH-REPLAY covers every immutable producer hash."""
    store = tmp_path / "arc_e3"
    row = _publish(
        store,
        game="aa01",
        stamp="20260904T210000_000000",
        transitions=[_transition(1)],
    )
    assert row
    replay = exp.enumerate_envelopes(store)

    for field in (
        "prompt_hash_replay_rows",
        "transition_hash_replay_rows",
        "engine_hash_replay_rows",
        "environment_hash_replay_rows",
        "scorer_hash_replay_rows",
        "policy_hash_replay_rows",
        "factory_hash_replay_rows",
        "manifest_hash_replay_rows",
        "envelope_hash_replay_rows",
    ):
        assert len(replay[field]) == 1
        assert replay[field][0]["passed"] is True
        assert replay[field][0]["expected_hash"] == replay[field][0]["observed_hash"]


def test_scenario_7005_split_freezes_attempts_episodes_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7005-SPLIT-FREEZE keeps only later unique rows held out."""
    store = tmp_path / "arc_e3"
    construction = [_transition(1), _transition(2)]
    first = _publish(
        store,
        game="aa01",
        stamp="20260904T210000_000000",
        transitions=construction,
        run_suffix="000000000001",
    )
    _publish(
        store,
        game="aa01",
        stamp="20260904T220000_000000",
        transitions=[*construction, _transition(3)],
        run_suffix="000000000002",
    )
    _publish(
        store,
        game="aa01",
        stamp="20260904T230000_000000",
        transitions=[_transition(4, level=1)],
        run_suffix="000000000003",
    )
    replay = exp.enumerate_envelopes(store)
    selected = exp.select_earliest_eligible(replay["envelope_eligibility_rows"])
    assert first and selected

    frozen = exp.freeze_transition_splits(store, replay, selected)
    construction_ids = {row["transition_id"] for row in frozen["construction_transition_rows"]}
    heldout_ids = {row["transition_id"] for row in frozen["heldout_transition_rows"]}

    assert len(construction_ids) == 2
    assert len(heldout_ids) == 2
    assert construction_ids.isdisjoint(heldout_ids)
    assert {row["attempt_id"] for row in frozen["heldout_transition_rows"]} == {
        "arc-20260904T220000_000000-000000000002",
        "arc-20260904T230000_000000-000000000003",
    }
    assert len({row["episode_id"] for row in frozen["heldout_transition_rows"]}) == 2
    assert any(
        row["decision"] == "duplicate_of_construction" for row in frozen["split_manifest_rows"]
    )
    assert frozen["split_manifest_hash"].startswith("sha256:")
    assert all(row["passed"] for row in frozen["leakage_check_rows"])


def test_scenario_7005_heldout_targets_cannot_change_pre_engine_baseline() -> None:
    """SCENARIO-ARC-WMTE-7005-NO-LEAKAGE excludes held-out targets from construction."""
    construction = [_transition(7, data={"x": 1, "y": 0})]
    heldout = [_transition(3, data={"x": 2, "y": 0})]
    changed = deepcopy(heldout)
    changed[0]["next_grid"] = [[9, 9], [9, 9]]
    changed[0]["level_after"] = 9

    original_predictions = exp.pre_engine_predictions(construction, heldout)
    changed_predictions = exp.pre_engine_predictions(construction, changed)

    assert original_predictions == changed_predictions
    assert exp.baseline_construction_hash(construction) == exp.baseline_construction_hash(
        construction
    )
    assert exp.inert_predictions(heldout) == exp.inert_predictions(changed)


def test_scenario_7005_pre_engine_control_matches_shipped_active_probe() -> None:
    """SCENARIO-ARC-WMTE-7005-CONTROLS-AND-METRICS replays the exact shipped baseline."""
    construction = [_transition(7, data={"x": 1, "y": 0})]
    evaluation = [
        _transition(7, data={"x": 1, "y": 0}),
        _transition(3, data={"x": 2, "y": 0}),
    ]
    transitions = [
        Transition(
            np.asarray(row["grid"]),
            row["action"],
            row["data"],
            np.asarray(row["next_grid"]),
            row["level_before"],
            row["level_after"],
        )
        for row in construction
    ]
    candidate = _exact_action_delta_candidate(transitions)
    assert candidate is not None

    expected = [
        candidate.engine(np.asarray(row["grid"]), row["action"], row["data"]).tolist()
        for row in evaluation
    ]
    assert exp.pre_engine_predictions(construction, evaluation) == expected
    assert expected[0] == [[0, 7], [0, 0]]
    assert expected[1] == evaluation[1]["grid"]


def test_scenario_7005_metrics_recompute_and_missing_targets_stay_missing() -> None:
    """SCENARIO-ARC-WMTE-7005-CONTROLS-AND-METRICS keeps sufficient statistics."""
    rows = [
        {**_transition(7), "transition_id": "a", "attempt_id": "r1", "episode_id": "e1"},
        {**_transition(2), "transition_id": "b", "attempt_id": "r1", "episode_id": "e1"},
        {
            **_transition(0, target=False),
            "transition_id": "c",
            "attempt_id": "r2",
            "episode_id": "e2",
        },
    ]
    predictions = [
        [[0, 7], [0, 0]],
        [[0, 9], [0, 0]],
        [[0, 0], [0, 0]],
    ]
    scored = exp.score_predictions("engine", "heldout", rows, predictions)
    mixed = [dict(scored[0], stratum="construction"), *scored]
    aggregate = exp.aggregate_scores("engine", "heldout", mixed)

    assert aggregate["transition_count"] == 3
    assert aggregate["target_transition_count"] == 2
    assert aggregate["exact_next_frame_accuracy"] == 0.5
    assert aggregate["changed_cell_recall"] == 0.5
    assert aggregate["changed_cell_precision"] == 0.5
    assert aggregate["calibrated_frame_error"] == 0.125
    assert aggregate["transition_coverage"] == 1.0
    assert aggregate["abstention_rate"] == 0.0
    assert scored[2]["target_available"] is False
    assert scored[2]["exact_next_frame_correct"] is None
    assert scored[2]["calibrated_frame_error"] is None


def test_scenario_7005_fresh_process_disables_external_capabilities(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7005-FRESH-PROCESS scores all arms under denied capabilities."""
    engine_path = tmp_path / "engine.py"
    engine_path.write_bytes(ENGINE_SOURCE)
    construction = [
        {
            **_transition(7, data={"x": 1, "y": 0}),
            "transition_id": "construction",
            "attempt_id": "attempt-a",
            "episode_id": "attempt-a:0",
        }
    ]
    heldout = [
        {
            **_transition(3, data={"x": 2, "y": 0}),
            "transition_id": "heldout",
            "attempt_id": "attempt-b",
            "episode_id": "attempt-b:0",
        }
    ]

    scored = exp.run_restricted_scoring(engine_path, construction, heldout)
    receipt = scored["read_only_enforcement_receipt"]

    assert receipt["fresh_process"] is True
    assert receipt["network_disabled"] is True
    assert receipt["gpu_devices_visible"] == []
    assert receipt["llm_disabled"] is True
    assert receipt["arc_service_disabled"] is True
    assert receipt["game_source_disabled"] is True
    assert receipt["subprocess_disabled"] is True
    assert receipt["writes_disabled"] is True
    assert receipt["passed"] is True
    assert len(scored["engine_score_rows"]) == 2
    assert len(scored["inert_control_rows"]) == 2
    assert len(scored["pre_engine_control_rows"]) == 2


def test_req_7005_real_artifact_is_terminal_and_has_mandatory_false_claims() -> None:
    """REQ-ARC-WMTE-7005 audits the real post-contract envelope without solve credit."""
    artifact = exp.build_artifact(run_date="20260905", repo_root=REPO_ROOT)

    assert exp.validate_artifact(artifact) == []
    assert artifact["selected_envelope_hash"] == (
        "sha256:1bc599d2bae89d8840c9fc4e4e5768abefe5aea9e6216fa8062d408ab87b918f"
    )
    assert artifact["arc_live_envelope_audit_complete_score"] == 1
    assert artifact["arc_engine_quality_evaluable_score"] == 1
    assert artifact["verdict_class"] in {"positive", "null"}
    assert artifact["source_disagreement_rows"] == []
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["solve_provenance_applicable"] is False
    for field in (
        "solve_claimed",
        "level_claimed",
        "registry_updated",
        "submitted_to_leaderboard",
        "game_source_inspected",
        "development_fixture_used_for_quality",
        "verifier_is_oracle",
    ):
        assert artifact[field] is False


def test_scenario_7005_absence_is_terminal_blocked_not_partial(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7005-TERMINAL-CLAIMS makes an exhaustive absence terminal."""
    artifact = exp.build_artifact(run_date="20260905", repo_root=tmp_path)

    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_arc_live_envelope_audit"
    assert artifact["arc_live_envelope_audit_complete_score"] == 1
    assert artifact["arc_engine_quality_evaluable_score"] == 0
    assert artifact["arc_engine_quality_positive_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "exp6993_contract_hash"
    assert artifact["gate_check_summary"]["expected_value"] == exp.EXPECTED_EXP6993_HASH
    assert artifact["gate_check_summary"]["observed_value"] is None
    assert exp.validate_artifact(artifact) == []


def test_scenario_7005_validator_rejects_overclaim_and_checksum_drift() -> None:
    """SCENARIO-ARC-WMTE-7005-TERMINAL-CLAIMS rejects invalid scientific claims."""
    artifact = exp.build_artifact(run_date="20260905", repo_root=Path("/missing-exp7005"))
    changed = deepcopy(artifact)
    changed["solve_claimed"] = True
    changed["field_principles"].pop("rows")
    changed["arc_engine_quality_positive_score"] = True
    changed["verdict_class"] = "unknown"
    changed["honest_verdict"] = "partial_wrong"

    errors = exp.validate_artifact(changed)
    assert "field_principles_mismatch" in errors
    assert "solve_claimed_must_be_false" in errors
    assert "arc_engine_quality_positive_score_not_bare_int" in errors
    assert "verdict_class_invalid" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_req_7005_main_writes_blocked_artifact_for_missing_repo(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7005 keeps the command surface terminal on missing evidence."""
    output = tmp_path / "result.json"
    status = exp.main(
        [
            "--date",
            "20260905",
            "--repo-root",
            str(tmp_path / "missing"),
            "--output",
            str(output),
        ]
    )

    assert status == 1
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "blocked_arc_live_envelope_audit"

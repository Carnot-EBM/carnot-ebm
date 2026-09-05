"""Tests for the transactional ARC belief ledger.

Spec refs: REQ-CSL-7020, SCENARIO-CSL-7020-*, REQ-ARC-WMTE-7020,
and SCENARIO-ARC-WMTE-7020-*.
"""

from __future__ import annotations

from copy import deepcopy
from base64 import b64encode
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.agentic import arc_belief_ledger as ledger_mod


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _event(
    stream_index: int,
    outcome: str,
    *,
    mechanic: str = "mechanic-a",
    event_id: str | None = None,
    include_observation: bool = True,
) -> dict:
    key_hash = _sha(mechanic)
    outcome_hash = _sha(outcome)
    row = {
        "schema": ledger_mod.EVENT_SCHEMA,
        "event_id": event_id or f"event-{stream_index}-{mechanic}-{outcome}",
        "stream_index": stream_index,
        "source_attempt_time": "20260905T010000_000000",
        "source_transition_index": stream_index,
        "pre_action": {
            "grid_hash": _sha(f"before-{stream_index}"),
            "observed_level": 0,
        },
        "action": {"type": 6, "data": {"x": 1, "y": 1}},
        "next_observation": {
            "grid_hash": _sha(f"after-{stream_index}"),
            "observed_level": 0,
            "state_delta": {
                "change_scale": "single",
                "level_boundary": False,
                "touches_action_coordinate": True,
            },
        },
        "contradiction": {
            "hypothesis_key": key_hash,
            "observed_outcome_key": outcome_hash,
            "is_contradiction": False,
            "prior_event_ids": [],
            "support_at_observation": 1,
        },
        "mechanic_signature": {
            "action_type": 6,
            "hypothesis_key": key_hash,
            "observed_outcome_key": outcome_hash,
            "change_scale": "single",
            "level_boundary": False,
            "mechanic_group": "action_6:single:same_level:at_action",
            "state_delta": {
                "change_scale": "single",
                "level_boundary": False,
                "touches_action_coordinate": True,
            },
            "spatial_summary": {"touches_action_coordinate": True},
        },
    }
    if not include_observation:
        row.pop("next_observation")
    row["row_hash"] = ledger_mod.event_row_hash(row)
    return row


def test_req_7020_specs_precede_implementation() -> None:
    """REQ-CSL-7020 and REQ-ARC-WMTE-7020 define every safety scenario."""

    csl = (REPO_ROOT / ledger_mod.CSL_SPEC_PATH).read_text(encoding="utf-8")
    arc = (REPO_ROOT / ledger_mod.ARC_SPEC_PATH).read_text(encoding="utf-8")
    for marker in (
        "SCENARIO-CSL-7020-STATES-AND-SUPPORT",
        "SCENARIO-CSL-7020-CONTRADICTION-TOMBSTONE-SUPERSESSION",
        "SCENARIO-CSL-7020-TRANSACTION-RESTART-ROLLBACK",
        "SCENARIO-CSL-7020-CAPACITY-AND-POISON",
        "SCENARIO-CSL-7020-AUTHORITY-AND-FAILED-WRITE",
        "SCENARIO-CSL-7020-ARTIFACT",
    ):
        assert marker in csl
    for marker in (
        "SCENARIO-ARC-WMTE-7020-OBSERVATION-TIME",
        "SCENARIO-ARC-WMTE-7020-GAME-BLIND-QUERY",
        "SCENARIO-ARC-WMTE-7020-CONSTRUCTION-REPLAY",
        "SCENARIO-ARC-WMTE-7020-SAFETY-GATE",
    ):
        assert marker in arc
    for field in ledger_mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in csl


def test_scenario_7020_observation_time_and_game_blind_key(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7020-OBSERVATION-TIME rejects early or hidden data."""

    ledger = ledger_mod.BeliefLedger(tmp_path / "ledger", capacity=8, min_support=2)
    complete = _event(0, "outcome")
    same_observable = deepcopy(complete)
    same_observable["game_id"] = "different-private-source"
    key = ledger_mod.mechanic_key_from_event(complete)
    before = ledger.snapshot()

    early = ledger.observe(_event(0, "outcome", include_observation=False))
    hidden = ledger.observe(same_observable)

    assert early == {"accepted": False, "reason": "next_observation_required"}
    assert hidden["accepted"] is False
    assert hidden["reason"] == "forbidden_updater_fields"
    assert hidden["forbidden_paths"] == ["game_id"]
    assert ledger.snapshot() == before
    assert key == ledger_mod.MechanicKey(
        action_type=6,
        hypothesis_key=_sha("mechanic-a"),
        change_scale="single",
        level_boundary=False,
        touches_action_coordinate=True,
        mechanic_group="action_6:single:same_level:at_action",
    )
    assert ledger.query(key)["state"] == "uncertain"


def test_scenario_7020_four_states_support_tombstone_and_supersession(tmp_path: Path) -> None:
    """SCENARIO-CSL-7020-STATES-AND-SUPPORT applies grouped contradictions first."""

    ledger = ledger_mod.BeliefLedger(tmp_path / "ledger", capacity=12, min_support=2)
    key = ledger_mod.mechanic_key_from_event(_event(0, "left"))
    left = _sha("left")
    right = _sha("right")

    first = ledger.observe(_event(0, "left"))
    second = ledger.observe(_event(1, "left"))
    conflict = ledger.observe(_event(2, "right"))

    assert first["belief_state"] == "possible"
    assert first["support_after"] == 1
    assert second["belief_state"] == "known"
    assert second["support_after"] == 2
    assert conflict["belief_state"] == "uncertain"
    assert conflict["tombstones"][0]["outcome_key"] == left
    assert ledger.query(key, outcome_key=left)["state"] == "contradicted"
    assert ledger.query(key, outcome_key=right)["state"] == "uncertain"
    assert ledger.query(key)["state"] == "uncertain"
    assert ledger.clusters()[0]["outcome_keys"] == sorted([left, right])

    renewed = ledger.observe(_event(3, "left"))
    promoted = ledger.observe(_event(4, "left"))

    assert renewed["supersessions"][0]["prior_generation"] == 1
    assert renewed["belief_state"] == "uncertain"
    assert promoted["belief_state"] == "known"
    assert promoted["support_after"] == 2
    left_facts = [row for row in ledger.facts() if row["outcome_key"] == left]
    assert [(row["generation"], row["state"]) for row in left_facts] == [
        (1, "contradicted"),
        (2, "known"),
    ]
    assert {row["state"] for row in ledger.facts()} >= {"known", "contradicted"}


def test_scenario_7020_authority_conflict_and_duplicate_are_nonmutating(tmp_path: Path) -> None:
    """SCENARIO-CSL-7020-AUTHORITY-AND-FAILED-WRITE preserves state bytes."""

    ledger = ledger_mod.BeliefLedger(tmp_path / "ledger", capacity=8, min_support=2)
    event = _event(0, "left", event_id="same-event")
    assert ledger.observe(event)["accepted"] is True
    parent = ledger.snapshot()

    duplicate = ledger.observe(deepcopy(event))
    conflicting = _event(1, "right", event_id="same-event")
    authority = ledger.observe(conflicting)

    assert duplicate["reason"] == "duplicate_event"
    assert authority["reason"] == "authority_conflict"
    assert authority["existing_row_hash"] == event["row_hash"]
    assert ledger.snapshot() == parent
    assert [row["phase"] for row in ledger.journal_rows()][-2:] == ["reject", "reject"]


def test_scenario_7020_capacity_eviction_and_poison_protection(tmp_path: Path) -> None:
    """SCENARIO-CSL-7020-CAPACITY-AND-POISON evicts weak facts only."""

    evicting = ledger_mod.BeliefLedger(tmp_path / "evict", capacity=2, min_support=2)
    evicting.observe(_event(0, "a", mechanic="a"))
    evicting.observe(_event(1, "b", mechanic="b"))
    receipt = evicting.observe(_event(2, "c", mechanic="c"))
    assert receipt["accepted"] is True
    assert receipt["evictions"][0]["mechanic_key"]["hypothesis_key"] == _sha("a")
    assert [row["outcome_key"] for row in evicting.facts()] == [_sha("b"), _sha("c")]

    protected = ledger_mod.BeliefLedger(tmp_path / "protected", capacity=2, min_support=2)
    protected.observe(_event(0, "a", mechanic="a"))
    protected.observe(_event(1, "a", mechanic="a"))
    protected.observe(_event(2, "b", mechanic="b"))
    protected.observe(_event(3, "b", mechanic="b"))
    parent = protected.snapshot()
    poison = protected.observe(_event(4, "poison", mechanic="poison"))

    assert poison["accepted"] is False
    assert poison["reason"] == "capacity_protected"
    assert poison["poison_protected"] is True
    assert protected.snapshot() == parent
    assert all(row["state"] == "known" and row["protected"] for row in protected.facts())


def test_scenario_7020_restart_interruption_rollback_and_audit(tmp_path: Path) -> None:
    """SCENARIO-CSL-7020-TRANSACTION-RESTART-ROLLBACK preserves exact bytes."""

    root = tmp_path / "ledger"
    ledger = ledger_mod.BeliefLedger(root, capacity=8, min_support=2)
    initial = ledger.snapshot()
    first_receipt = ledger.observe(_event(0, "left"))
    committed = ledger.snapshot()
    restarted = ledger_mod.BeliefLedger(root, capacity=8, min_support=2)
    assert restarted.snapshot() == committed

    with pytest.raises(ledger_mod.ForcedInterruption):
        restarted.observe(_event(1, "left"), interrupt_after_prepare=True)
    recovered = ledger_mod.BeliefLedger(root, capacity=8, min_support=2)
    assert recovered.snapshot() == committed
    assert recovered.recovery_rows[-1]["phase"] == "abort_recovered"

    with pytest.raises(ledger_mod.ForcedInterruption):
        recovered.observe(_event(1, "left"), interrupt_after_publish=True)
    published = ledger_mod.BeliefLedger(root, capacity=8, min_support=2)
    assert published.query(ledger_mod.mechanic_key_from_event(_event(0, "left")))["state"] == (
        "known"
    )
    assert published.recovery_rows[-1]["phase"] == "commit_recovered"
    rollback = published.rollback(
        initial,
        transaction_id=first_receipt["transaction_id"],
        reason="fixture",
    )
    assert rollback["rolled_back"] is True
    assert published.snapshot() == initial
    assert published.audit_state()["passed"] is True
    assert all(
        row["previous_row_hash"]
        == (None if index == 0 else published.journal_rows()[index - 1]["row_hash"])
        for index, row in enumerate(published.journal_rows())
    )

    before_invalid = published.snapshot()
    with pytest.raises(ValueError, match="rollback state"):
        published.rollback(b"{}\n", transaction_id="invalid", reason="fixture")
    assert published.snapshot() == before_invalid


def test_scenario_7020_journal_tamper_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-CSL-7020-TRANSACTION-RESTART-ROLLBACK rejects changed history."""

    root = tmp_path / "ledger"
    ledger = ledger_mod.BeliefLedger(root, capacity=8, min_support=2)
    ledger.observe(_event(0, "left"))
    lines = ledger.journal_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["candidate_state_hash"] = _sha("tampered")
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    ledger.journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ledger_mod.JournalIntegrityError, match="row hash"):
        ledger_mod.BeliefLedger(root, capacity=8, min_support=2)


def test_req_7020_real_stream_replay_and_safety_fixtures(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7020-CONSTRUCTION-REPLAY covers all 27 frozen events."""

    fixture = REPO_ROOT / ledger_mod.EXP7019_FIXTURE_PATH
    events = ledger_mod.load_updater_events(fixture)
    replay = ledger_mod.replay_events(
        events,
        tmp_path / "replay",
        capacity=ledger_mod.DEFAULT_CAPACITY,
        min_support=ledger_mod.DEFAULT_MIN_SUPPORT,
    )
    safety = ledger_mod.run_safety_fixtures(tmp_path / "safety")

    assert len(events) == 27
    assert len(replay["per_event_results"]) == 27
    assert all(row["terminal"] is True for row in replay["per_event_results"])
    assert replay["audit_state"]["passed"] is True
    assert replay["future_paths"] == []
    assert replay["snapshot_hashes"][0] != replay["snapshot_hashes"][-1]
    assert safety["passed"] is True
    assert set(safety["observed_states"]) == {
        "known",
        "possible",
        "contradicted",
        "uncertain",
    }
    assert safety["authority_conflict_rows"][0]["state_unchanged"] is True
    assert safety["restart_rows"][0]["passed"] is True
    assert safety["rollback_rows"][0]["passed"] is True
    assert safety["poison_rows"][0]["protected_state_unchanged"] is True


def test_req_7020_blocked_preconditions_write_complete_artifact(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7020 writes an honest blocked artifact on missing Exp7019."""

    output = tmp_path / "blocked.json"
    artifact = ledger_mod.build_artifact(
        repo_root=tmp_path / "missing-repo",
        run_date="20260905",
        output_path=output,
        checkpoint_root=tmp_path / "checkpoint",
    )

    assert artifact["belief_ledger_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_counterexample_belief_ledger"
    assert artifact["gate_check_summary"]["failed_check"] == "exp7019_artifact_readable"
    assert set(artifact["field_principles"]) == set(ledger_mod.REQUIRED_ARTIFACT_FIELDS)
    assert ledger_mod.validate_artifact(artifact, repo_root=tmp_path / "missing-repo") == []


def test_req_7020_complete_artifact_is_valid_and_fresh_process_stable(tmp_path: Path) -> None:
    """SCENARIO-CSL-7020-ARTIFACT requires deterministic replay and zero leakage."""

    artifact = ledger_mod.build_artifact(
        repo_root=REPO_ROOT,
        run_date="20260905",
        output_path=tmp_path / "result.json",
        checkpoint_root=tmp_path / "checkpoint",
    )

    assert ledger_mod.validate_artifact(artifact, repo_root=REPO_ROOT) == []
    assert artifact["belief_ledger_ready_score"] == 1
    assert artifact["inference_substrate"] == ("deterministic_arc_belief_ledger_replay_no_llm")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_positive_")
    assert artifact["gate_check_summary"]["passed"] is True
    assert len(artifact["per_event_results"]) == 27
    assert artifact["restart_rows"]
    assert artifact["rollback_rows"]
    assert artifact["authority_conflict_rows"]
    assert artifact["supersession_rows"]
    assert artifact["poison_rows"]
    assert {row["state"] for row in artifact["belief_state_rows"]} == {
        "known",
        "possible",
        "contradicted",
        "uncertain",
    }
    assert all(row["passed"] for row in artifact["leakage_check_rows"])
    assert len(artifact["cited_upstream_artifacts"]) == 3
    assert artifact["snapshot_hashes"][-1]["fresh_process_match"] is True
    assert artifact["ledger_state_bytes"] <= ledger_mod.DEFAULT_MAX_STATE_BYTES


def test_req_7020_artifact_validation_rejects_drift(tmp_path: Path) -> None:
    """REQ-CSL-7020 validates fields, row terminals, checksums, and bare scores."""

    blocked = ledger_mod.build_artifact(
        repo_root=tmp_path / "missing-repo",
        run_date="20260905",
        output_path=tmp_path / "blocked.json",
        checkpoint_root=tmp_path / "checkpoint",
    )
    cases = []
    missing = deepcopy(blocked)
    missing.pop("rows")
    cases.append((missing, "required_fields"))
    principles = deepcopy(blocked)
    principles["field_principles"].pop("rows")
    cases.append((principles, "field_principles"))
    boolean_score = deepcopy(blocked)
    boolean_score["belief_ledger_ready_score"] = False
    cases.append((boolean_score, "ready_score"))
    bad_verdict = deepcopy(blocked)
    bad_verdict["honest_verdict"] = "complete_positive_wrong"
    cases.append((bad_verdict, "verdict_prefix"))
    bad_checksum = deepcopy(blocked)
    bad_checksum["random_seed"] += 1
    cases.append((bad_checksum, "reproducibility_checksum"))
    nonterminal = deepcopy(blocked)
    nonterminal["rows"] = [{"terminal": False}]
    nonterminal["reproducibility_checksum"] = ledger_mod.artifact_checksum(nonterminal)
    cases.append((nonterminal, "nonterminal_row"))

    for artifact, marker in cases:
        assert any(
            marker in error
            for error in ledger_mod.validate_artifact(
                artifact,
                repo_root=tmp_path / "missing-repo",
            )
        )


def test_req_7020_command_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-CSL-7020-ARTIFACT exercises the required command surface."""

    output = tmp_path / "result.json"
    command = [
        sys.executable,
        str(REPO_ROOT / ledger_mod.WRAPPER_PATH),
        "--date",
        "20260905",
        "--repo-root",
        str(REPO_ROOT),
        "--output",
        str(output),
        "--checkpoint-root",
        str(tmp_path / "checkpoint"),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert completed.returncode == 0, completed.stderr
    assert artifact["belief_ledger_ready_score"] == 1
    assert ledger_mod.validate_artifact(artifact, repo_root=REPO_ROOT) == []
    assert "belief_ledger_ready_score=1" in completed.stdout


def test_req_7020_typed_key_and_store_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CSL-7020 rejects malformed typed keys and durable state."""

    valid = ledger_mod.mechanic_key_from_event(_event(0, "outcome"))
    invalid_keys = [
        {**valid.to_dict(), "action_type": True},
        {**valid.to_dict(), "hypothesis_key": "bad"},
        {**valid.to_dict(), "change_scale": ""},
        {**valid.to_dict(), "level_boundary": 0},
    ]
    for values in invalid_keys:
        with pytest.raises(ValueError):
            ledger_mod.MechanicKey(**values)

    fallback = _event(0, "outcome")
    fallback["mechanic_signature"].pop("state_delta")
    fallback["mechanic_signature"]["spatial_summary"] = "invalid"
    assert ledger_mod.mechanic_key_from_event(fallback) == valid
    with pytest.raises(ValueError, match="mechanic_signature"):
        ledger_mod.mechanic_key_from_event({})
    with pytest.raises(ValueError, match="positive"):
        ledger_mod.BeliefLedger(tmp_path / "bad-limits", capacity=0)
    with pytest.raises(ValueError, match="initial ledger state"):
        ledger_mod.BeliefLedger(tmp_path / "tiny-initial", max_state_bytes=1)

    valid_root = tmp_path / "valid"
    stored = ledger_mod.BeliefLedger(valid_root, capacity=3)
    original = stored.state_bytes
    with pytest.raises(ValueError, match="configuration"):
        ledger_mod.BeliefLedger(valid_root, capacity=4)
    with pytest.raises(ValueError, match="byte limit"):
        ledger_mod.BeliefLedger(valid_root, capacity=3, max_state_bytes=len(original) - 1)

    invalid_json_root = tmp_path / "invalid-json"
    invalid_json = ledger_mod.BeliefLedger(invalid_json_root)
    invalid_json.state_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid ledger state"):
        ledger_mod.BeliefLedger(invalid_json_root)

    schema_root = tmp_path / "schema"
    schema = ledger_mod.BeliefLedger(schema_root)
    state = json.loads(schema.state_bytes)
    state["schema"] = "wrong"
    schema.state_path.write_bytes(ledger_mod.canonical_bytes(state))
    with pytest.raises(ValueError, match="state schema"):
        ledger_mod.BeliefLedger(schema_root)

    collections_root = tmp_path / "collections"
    collections = ledger_mod.BeliefLedger(collections_root)
    state = json.loads(collections.state_bytes)
    state["facts"] = {}
    collections.state_path.write_bytes(ledger_mod.canonical_bytes(state))
    with pytest.raises(ValueError, match="state collections"):
        ledger_mod.BeliefLedger(collections_root)

    target = tmp_path / "atomic" / "state.json"

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("fixture replace failure")

    monkeypatch.setattr(ledger_mod.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        ledger_mod._atomic_write(target, b"value")
    assert not list(target.parent.glob(".state.json.*"))


def test_req_7020_event_query_and_byte_limit_rejections(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7020 rejects every malformed write without state change."""

    ledger = ledger_mod.BeliefLedger(tmp_path / "ledger", capacity=8)
    mutations: list[tuple[object, str]] = [(None, "event_object_required")]
    schema = _event(0, "x")
    schema["schema"] = "wrong"
    schema["row_hash"] = ledger_mod.event_row_hash(schema)
    mutations.append((schema, "event_schema_invalid"))
    missing_id = _event(0, "x")
    missing_id["event_id"] = ""
    missing_id["row_hash"] = ledger_mod.event_row_hash(missing_id)
    mutations.append((missing_id, "event_id_invalid"))
    bad_index = _event(0, "x")
    bad_index["stream_index"] = True
    bad_index["row_hash"] = ledger_mod.event_row_hash(bad_index)
    mutations.append((bad_index, "stream_index_invalid"))
    bad_hash = _event(0, "x")
    bad_hash["row_hash"] = _sha("wrong")
    mutations.append((bad_hash, "event_row_hash_invalid"))
    bad_mechanic = _event(0, "x")
    bad_mechanic["mechanic_signature"]["action_type"] = "six"
    bad_mechanic["row_hash"] = ledger_mod.event_row_hash(bad_mechanic)
    mutations.append((bad_mechanic, "mechanic_key_invalid"))
    bad_outcome = _event(0, "x")
    bad_outcome["mechanic_signature"]["observed_outcome_key"] = "bad"
    bad_outcome["row_hash"] = ledger_mod.event_row_hash(bad_outcome)
    mutations.append((bad_outcome, "observed_outcome_key_invalid"))
    parent = ledger.snapshot()
    for event, reason in mutations:
        assert ledger.observe(event)["reason"] == reason  # type: ignore[arg-type]
        assert ledger.snapshot() == parent

    first = _event(1, "x")
    assert ledger.observe(first)["accepted"] is True
    out_of_order = _event(0, "late")
    assert ledger.observe(out_of_order)["reason"] == "non_chronological_event"
    key = ledger_mod.mechanic_key_from_event(first)
    with pytest.raises(TypeError, match="typed MechanicKey"):
        ledger.query("not-typed")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outcome_key"):
        ledger.query(key, outcome_key="bad")

    limited = ledger_mod.BeliefLedger(tmp_path / "limited", capacity=8)
    limited.max_state_bytes = len(limited.state_bytes) + 1
    assert limited.observe(_event(0, "large"))["reason"] == "capacity_protected"

    with pytest.raises(ValueError, match="valid JSON"):
        ledger.rollback(b"not-json", transaction_id="bad", reason="fixture")
    large_state = json.loads(ledger.snapshot())
    large_state["padding"] = "x" * ledger.max_state_bytes
    with pytest.raises(ValueError, match="byte limit"):
        ledger.rollback(
            ledger_mod.canonical_bytes(large_state),
            transaction_id="large",
            reason="fixture",
        )


def test_req_7020_journal_parse_chain_and_recovery_errors(tmp_path: Path) -> None:
    """SCENARIO-CSL-7020-TRANSACTION-RESTART-ROLLBACK fails on corrupt durability."""

    parse_root = tmp_path / "parse"
    parse = ledger_mod.BeliefLedger(parse_root)
    parse.journal_path.write_text("{", encoding="utf-8")
    with pytest.raises(ledger_mod.JournalIntegrityError, match="parse failed"):
        ledger_mod.BeliefLedger(parse_root)

    chain_root = tmp_path / "chain"
    chain = ledger_mod.BeliefLedger(chain_root)
    chain.observe(_event(0, "x"))
    rows = chain.journal_rows()
    rows[1]["previous_row_hash"] = _sha("wrong")
    payload = dict(rows[1])
    payload.pop("row_hash")
    rows[1]["row_hash"] = ledger_mod.sha256_bytes(ledger_mod.canonical_bytes(payload))
    chain.journal_path.write_bytes(b"".join(ledger_mod.canonical_bytes(row) for row in rows))
    with pytest.raises(ledger_mod.JournalIntegrityError, match="chain mismatch"):
        ledger_mod.BeliefLedger(chain_root)

    recovery_root = tmp_path / "recovery"
    recovery = ledger_mod.BeliefLedger(recovery_root)
    with pytest.raises(ledger_mod.ForcedInterruption):
        recovery.observe(_event(0, "x"), interrupt_after_prepare=True)
    third_state = json.loads(recovery.snapshot())
    third_state["version"] = 99
    recovery.state_path.write_bytes(ledger_mod.canonical_bytes(third_state))
    with pytest.raises(ledger_mod.JournalIntegrityError, match="state mismatch"):
        ledger_mod.BeliefLedger(recovery_root)


def test_req_7020_audit_detects_semantic_journal_and_state_faults(tmp_path: Path) -> None:
    """REQ-CSL-7020 audit_state detects faults beyond the row hash chain."""

    ledger = ledger_mod.BeliefLedger(tmp_path / "ledger", capacity=2)
    ledger.observe(_event(0, "x"))
    wrong = _sha("wrong")
    encoded_empty = b64encode(b"{}\n").decode("ascii")
    ledger._append_journal("reject", "reject-bad", state_hash=wrong)
    ledger._append_journal(
        "prepare",
        "prepare-bad",
        parent_state_hash=wrong,
        candidate_state_hash=wrong,
        parent_state_b64=encoded_empty,
        candidate_state_b64=encoded_empty,
    )
    ledger._append_journal("commit", "commit-bad", candidate_state_hash=wrong)
    ledger._append_journal("abort_recovered", "abort-bad")
    ledger._append_journal(
        "rollback",
        "rollback-bad",
        restored_state_hash=wrong,
        restored_state_b64=encoded_empty,
    )
    ledger._append_journal("unknown", "unknown-bad")
    ledger._state["game_id"] = "forbidden"
    ledger._state["facts"].extend(deepcopy(ledger._state["facts"]) * 2)
    ledger.max_state_bytes = 1

    audit = ledger.audit_state()

    assert audit["passed"] is False
    assert {error.split(":", 1)[0] for error in audit["errors"]} >= {
        "reject_state_mismatch",
        "prepare_parent_bytes_mismatch",
        "prepare_candidate_bytes_mismatch",
        "prepare_parent_mismatch",
        "commit_without_prepare",
        "abort_without_prepare",
        "rollback_bytes_mismatch",
        "unknown_phase",
        "replay_final_state_mismatch",
        "forbidden_state_fields",
        "fact_capacity_exceeded",
        "state_byte_capacity_exceeded",
    }


def test_req_7020_updater_loader_rejects_all_stream_drift(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7020-GAME-BLIND-QUERY rejects fixture drift."""

    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="unreadable"):
        ledger_mod.load_updater_events(missing)
    invalid_json = tmp_path / "invalid.jsonl"
    invalid_json.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        ledger_mod.load_updater_events(invalid_json)

    cases = []
    forbidden = _event(0, "x")
    forbidden["game_id"] = "private"
    forbidden["row_hash"] = ledger_mod.event_row_hash(forbidden)
    cases.append((forbidden, "forbidden fields"))
    chronology = _event(1, "x")
    cases.append((chronology, "complete chronological"))
    bad_hash = _event(0, "x")
    bad_hash["row_hash"] = _sha("wrong")
    cases.append((bad_hash, "row hash mismatch"))
    for index, (row, message) in enumerate(cases):
        path = tmp_path / f"case-{index}.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            ledger_mod.load_updater_events(path)

    replay = ledger_mod.replay_events(
        [_event(0, "x"), _event(0, "x")],
        tmp_path / "duplicate-replay",
    )
    assert replay["rejection_rows"][0]["reason"] == "duplicate_event"


def test_req_7020_fresh_process_failures_and_partial_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7020-SAFETY-GATE cannot promote unstable replay."""

    def completed(returncode: int, stdout: str, stderr: str = "") -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess([], returncode, stdout, stderr)

    monkeypatch.setattr(
        ledger_mod.subprocess,
        "run",
        lambda *args, **kwargs: completed(1, "", "failed"),
    )
    assert (
        ledger_mod._fresh_process_digest(REPO_ROOT, REPO_ROOT / ledger_mod.EXP7019_FIXTURE_PATH)[
            "error"
        ]
        == "failed"
    )
    monkeypatch.setattr(
        ledger_mod.subprocess,
        "run",
        lambda *args, **kwargs: completed(0, "not-json"),
    )
    assert (
        "invalid JSON"
        in ledger_mod._fresh_process_digest(
            REPO_ROOT,
            REPO_ROOT / ledger_mod.EXP7019_FIXTURE_PATH,
        )["error"]
    )
    monkeypatch.setattr(
        ledger_mod.subprocess,
        "run",
        lambda *args, **kwargs: completed(0, "[]"),
    )
    assert ledger_mod._fresh_process_digest(
        REPO_ROOT,
        REPO_ROOT / ledger_mod.EXP7019_FIXTURE_PATH,
    ) == {"error": "fresh process returned non-object"}

    monkeypatch.setattr(
        ledger_mod,
        "_fresh_process_digest",
        lambda *_args: {"digest": "different", "state_hash": "different", "event_count": 27},
    )
    artifact = ledger_mod.build_artifact(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "partial.json",
        checkpoint_root=tmp_path / "checkpoint",
    )
    assert artifact["belief_ledger_ready_score"] == 0
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("partial_")
    assert artifact["gate_check_summary"]["failed_check"] == "fresh_process_replay"


def test_req_7020_validator_and_direct_main_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CSL-7020 validates every terminal gate and direct command branch."""

    assert ledger_mod.validate_artifact([]) == ["artifact_object_required"]
    blocked = ledger_mod.build_artifact(
        repo_root=tmp_path / "missing",
        output_path=tmp_path / "blocked.json",
        checkpoint_root=tmp_path / "checkpoint",
    )
    cases = []
    invalid_class = deepcopy(blocked)
    invalid_class["verdict_class"] = "other"
    cases.append((invalid_class, "verdict_class_invalid"))
    not_list = deepcopy(blocked)
    not_list["rows"] = {}
    cases.append((not_list, "rows_not_list"))
    substrate = deepcopy(blocked)
    substrate["inference_substrate"] = "other"
    cases.append((substrate, "inference_substrate_mismatch"))
    oracle = deepcopy(blocked)
    oracle["verifier_is_oracle"] = True
    cases.append((oracle, "verifier_is_oracle_mismatch"))
    gate = deepcopy(blocked)
    gate["gate_check_summary"] = {}
    cases.append((gate, "gate_check_summary_invalid"))
    for artifact, marker in cases:
        artifact["reproducibility_checksum"] = ledger_mod.artifact_checksum(artifact)
        assert marker in ledger_mod.validate_artifact(artifact)

    ready = deepcopy(blocked)
    ready["belief_ledger_ready_score"] = 1
    ready["verdict_class"] = "positive"
    ready["honest_verdict"] = "complete_positive_fixture"
    ready["gate_check_summary"] = {"failed_check": "x", "passed": False}
    ready["leakage_check_rows"] = [{"passed": False, "terminal": True}]
    ready["reproducibility_checksum"] = ledger_mod.artifact_checksum(ready)
    errors = ledger_mod.validate_artifact(ready)
    assert "ready_gate_inconsistent" in errors
    assert "ready_event_count_mismatch" in errors
    assert "ready_leakage_check_failed" in errors

    assert (
        ledger_mod.main(["--replay-digest", str(REPO_ROOT / ledger_mod.EXP7019_FIXTURE_PATH)]) == 0
    )
    assert json.loads(capsys.readouterr().out)["event_count"] == 27

    output = tmp_path / "direct.json"
    assert (
        ledger_mod.main(
            [
                "--repo-root",
                str(tmp_path / "missing"),
                "--output",
                str(output),
                "--checkpoint-root",
                str(tmp_path / "direct-checkpoint"),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["verdict_class"] == "blocked"

    monkeypatch.setattr(ledger_mod, "build_artifact", lambda **_kwargs: {})
    with pytest.raises(ValueError, match="validation failed"):
        ledger_mod.main(["--repo-root", str(tmp_path)])

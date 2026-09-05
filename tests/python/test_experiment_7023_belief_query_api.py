"""Tests for the bounded ARC belief-query interface.

Spec refs: REQ-ARC-WMTE-7023 and SCENARIO-ARC-WMTE-7023-*.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.agentic import arc_belief_ledger as ledger_mod
from carnot.agentic import arc_belief_query as query_mod


REPO_ROOT = Path(__file__).resolve().parents[2]


def _event(stream_index: int, outcome: str, *, mechanic: str = "query") -> dict:
    return ledger_mod._fixture_event(
        stream_index,
        mechanic=mechanic,
        outcome=outcome,
    )


def _request(event: dict, observation_index: int) -> query_mod.BeliefQueryRequest:
    return query_mod.BeliefQueryRequest(
        mechanic_signature=ledger_mod.mechanic_key_from_event(event),
        observation_index=observation_index,
    )


def _known_ledger(root: Path) -> tuple[ledger_mod.BeliefLedger, dict]:
    first = _event(0, "left")
    ledger = ledger_mod.BeliefLedger(root, capacity=8, min_support=2)
    assert ledger.observe(first)["accepted"] is True
    assert ledger.observe(_event(1, "left"))["accepted"] is True
    return ledger, first


def _tie_ledger(root: Path) -> tuple[ledger_mod.BeliefLedger, dict]:
    ledger = ledger_mod.BeliefLedger(root, capacity=8, min_support=2)
    first = _event(0, "left", mechanic="tie")
    assert ledger.observe(first)["accepted"] is True
    state = json.loads(ledger.snapshot())
    second_event = _event(0, "right", mechanic="tie")
    second = deepcopy(state["facts"][0])
    second["outcome_key"] = second_event["mechanic_signature"]["observed_outcome_key"]
    second["fact_id"] = ledger_mod.sha256_bytes(
        ledger_mod.canonical_bytes(
            {"tie": second["outcome_key"], "mechanic_key": second["mechanic_key"]}
        )
    )
    second["event_ids"] = [second_event["event_id"]]
    state["facts"].append(second)
    state["events"][second_event["event_id"]] = {
        "row_hash": second_event["row_hash"],
        "stream_index": 0,
        "mechanic_key": second["mechanic_key"],
        "outcome_key": second["outcome_key"],
    }
    state["facts"] = sorted(state["facts"], key=lambda row: row["fact_id"])
    ledger.state_path.write_bytes(ledger_mod.canonical_bytes(state))
    return ledger.restart(), first


def test_req_7023_spec_anchors_exact_contract() -> None:
    """REQ-ARC-WMTE-7023 exists before implementation and names every scenario."""

    spec = (REPO_ROOT / "openspec/capabilities/arc-world-model-trust-energy/spec.md").read_text(
        encoding="utf-8"
    )
    assert "REQ-ARC-WMTE-7023" in spec
    assert {
        "SCENARIO-ARC-WMTE-7023-EXACT-BOUNDED-RESULT",
        "SCENARIO-ARC-WMTE-7023-CONFLICT-AND-TRUNCATION",
        "SCENARIO-ARC-WMTE-7023-GAME-BLIND-REJECTION",
        "SCENARIO-ARC-WMTE-7023-RESTART-STABILITY",
        "SCENARIO-ARC-WMTE-7023-READINESS-GATE",
    } <= {line.removeprefix("#### ") for line in spec.splitlines()}


def test_scenario_7023_exact_typed_known_result_and_prompt_bytes(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7023-EXACT-BOUNDED-RESULT fixes keys and encoding."""

    ledger, first = _known_ledger(tmp_path / "known")
    result = query_mod.BoundedBeliefQuery().query(ledger, _request(first, 1))
    payload = query_mod.canonical_prompt_bytes(result)
    value = result.to_dict()

    assert set(value) == query_mod.EXACT_RESULT_KEYS
    assert all(
        set(row) == query_mod.EXACT_BELIEF_KEYS
        for field in ("known", "possible", "contradicted", "uncertain")
        for row in value[field]
    )
    assert len(result.known) == 1
    assert result.support == 2
    assert result.contradiction_count == 0
    assert result.age == 0
    assert result.abstained is False
    assert result.ranked_outcome_keys == (result.known[0].outcome_key,)
    assert json.loads(payload) == value
    assert b"\n" not in payload
    assert len(payload) <= result.byte_budget
    assert len(result.source_event_hashes) <= result.event_budget
    with pytest.raises(FrozenInstanceError):
        result.support = 9  # type: ignore[misc]


def test_scenario_7023_low_support_empty_and_stale_abstain(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7023 abstains when evidence cannot support a safe rank."""

    weak = ledger_mod.BeliefLedger(tmp_path / "weak", capacity=8, min_support=2)
    weak_event = _event(0, "weak", mechanic="weak")
    weak.observe(weak_event)
    weak_result = query_mod.BoundedBeliefQuery().query(weak, _request(weak_event, 0))
    assert weak_result.possible
    assert weak_result.abstained is True
    assert weak_result.abstention_reason == "low_support"

    empty = ledger_mod.BeliefLedger(tmp_path / "empty", capacity=8, min_support=2)
    empty_result = query_mod.BoundedBeliefQuery().query(empty, _request(weak_event, 0))
    assert empty_result.support == 0
    assert empty_result.abstained is True
    assert empty_result.abstention_reason == "no_evidence"

    stale, first = _known_ledger(tmp_path / "stale")
    stale_result = query_mod.BoundedBeliefQuery(query_mod.BeliefQueryLimits(max_age=2)).query(
        stale, _request(first, 9)
    )
    assert stale_result.known
    assert stale_result.age == 8
    assert stale_result.abstained is True
    assert stale_result.abstention_reason == "stale_evidence"


def test_scenario_7023_conflict_tombstone_and_provenance_are_preserved(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7023-CONFLICT-AND-TRUNCATION keeps both sides."""

    ledger, first = _known_ledger(tmp_path / "conflict")
    conflicting = _event(2, "right")
    assert ledger.observe(conflicting)["accepted"] is True
    result = query_mod.BoundedBeliefQuery().query(ledger, _request(first, 2))

    assert len(result.contradicted) == 1
    assert len(result.uncertain) == 1
    assert result.contradiction_count == 1
    assert result.abstained is True
    assert result.abstention_reason == "conflicting_evidence"
    assert set(result.source_event_hashes) == {
        _event(0, "left")["row_hash"],
        _event(1, "left")["row_hash"],
        conflicting["row_hash"],
    }
    assert "accepted" not in result.to_dict()
    assert "rejected" not in result.to_dict()


def test_scenario_7023_semantic_tie_is_stable_and_abstains(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7023 uses hashes only after identifying semantic ties."""

    ledger, first = _tie_ledger(tmp_path / "tie")
    request = _request(first, 0)
    result = query_mod.BoundedBeliefQuery().query(ledger, request)
    restarted = query_mod.BoundedBeliefQuery().query(ledger.restart(), request)

    assert result.abstained is True
    assert result.abstention_reason == "tied_active_evidence"
    assert result.ranked_outcome_keys == tuple(sorted(result.ranked_outcome_keys))
    assert query_mod.query_serialization_hash(result) == query_mod.query_serialization_hash(
        restarted
    )


def test_scenario_7023_event_and_byte_budgets_report_truncation(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7023-CONFLICT-AND-TRUNCATION never hides pressure."""

    ledger, first = _known_ledger(tmp_path / "bounded")
    ledger.observe(_event(2, "right"))
    limits = query_mod.BeliefQueryLimits(
        max_events=2,
        max_bytes=query_mod.MIN_QUERY_BYTES,
        max_age=32,
    )
    result = query_mod.BoundedBeliefQuery(limits).query(ledger, _request(first, 2))
    payload = query_mod.canonical_prompt_bytes(result)

    assert len(result.source_event_hashes) <= 2
    assert len(payload) <= query_mod.MIN_QUERY_BYTES
    assert result.truncated is True
    assert result.omitted_event_count > 0 or result.omitted_belief_count > 0
    assert result.contradiction_count == 1
    assert result.omitted_contradiction_event_count >= 0
    assert result.abstained is True


def test_scenario_7023_rejects_prohibited_fields_and_arbitrary_mechanics(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7023-GAME-BLIND-REJECTION covers every denied field."""

    ledger, first = _known_ledger(tmp_path / "rejections")
    state_before = ledger.snapshot()
    journal_before = ledger.journal_path.read_bytes()
    for field in sorted(query_mod.PROHIBITED_QUERY_KEYS):
        assert query_mod.find_prohibited_query_paths({"safe": {field: "x"}}) == [f"safe.{field}"]

    bad_key = ledger_mod.MechanicKey(
        **{**ledger_mod.mechanic_key_from_event(first).to_dict(), "mechanic_group": "game_id:x"}
    )
    with pytest.raises(ValueError, match="normalized game-blind"):
        query_mod.BoundedBeliefQuery().query(
            ledger,
            query_mod.BeliefQueryRequest(bad_key, 1),
        )
    with pytest.raises(TypeError, match="BeliefQueryRequest"):
        query_mod.BoundedBeliefQuery().query(ledger, {"game_id": "secret"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        query_mod.BeliefQueryRequest(  # type: ignore[call-arg]
            mechanic_signature=ledger_mod.mechanic_key_from_event(first),
            observation_index=1,
            future_outcome="secret",
        )
    assert ledger.snapshot() == state_before
    assert ledger.journal_path.read_bytes() == journal_before


def test_scenario_7023_rejects_invalid_limits_time_and_result_values(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7023 rejects unbounded, time-travel, and unsafe values."""

    for values in (
        {"max_events": 0},
        {"max_events": query_mod.MAX_QUERY_EVENTS + 1},
        {"max_bytes": query_mod.MIN_QUERY_BYTES - 1},
        {"max_bytes": query_mod.MAX_QUERY_BYTES + 1},
        {"max_age": -1},
    ):
        with pytest.raises(ValueError):
            query_mod.BeliefQueryLimits(**values)

    ledger, first = _known_ledger(tmp_path / "time")
    with pytest.raises(ValueError, match="observation_index"):
        query_mod.BoundedBeliefQuery().query(ledger, _request(first, 0))
    with pytest.raises(TypeError, match="BeliefLedger"):
        query_mod.BoundedBeliefQuery().query(object(), _request(first, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BeliefQueryResult"):
        query_mod.canonical_prompt_bytes({"future_outcome": "x"})  # type: ignore[arg-type]


def test_req_7023_typed_value_and_serializer_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-7023 validates typed rows again at the serialization boundary."""

    with pytest.raises(TypeError, match="typed MechanicKey"):
        query_mod.BeliefQueryRequest("bad", 0)  # type: ignore[arg-type]
    base = ledger_mod.mechanic_key_from_event(_event(0, "left"))
    unsupported = ledger_mod.MechanicKey(
        **{
            **base.to_dict(),
            "change_scale": "private",
            "mechanic_group": "action_6:private:same_level:at_action",
        }
    )
    with pytest.raises(ValueError, match="change scale"):
        query_mod.BeliefQueryRequest(unsupported, 0)
    with pytest.raises(ValueError, match="outcome_key"):
        query_mod.BeliefSummary("bad", 1, 0, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        query_mod.BeliefSummary(ledger_mod.sha256_bytes(b"x"), -1, 0, 0)

    ledger, first = _known_ledger(tmp_path / "serializer")
    result = query_mod.BoundedBeliefQuery().query(ledger, _request(first, 1))
    prohibited = replace(
        result,
        mechanic_signature={**result.mechanic_signature, "game_id": "secret"},
    )
    with pytest.raises(ValueError, match="prohibited fields"):
        query_mod.canonical_prompt_bytes(prohibited)
    with pytest.raises(ValueError, match="byte budget"):
        query_mod.canonical_prompt_bytes(replace(result, byte_budget=1))
    with pytest.raises(ValueError, match="event budget"):
        query_mod.canonical_prompt_bytes(replace(result, event_budget=0))
    with pytest.raises(ValueError, match="source provenance"):
        query_mod.canonical_prompt_bytes(replace(result, source_event_hashes=("bad",)))

    monkeypatch.setattr(query_mod, "EXACT_RESULT_KEYS", frozenset())
    with pytest.raises(ValueError, match="result keys"):
        result.to_dict()


def test_req_7023_rejects_malformed_snapshot_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-7023 rejects malformed snapshot state before returning evidence."""

    def run_case(name: str, mutate: object, message: str) -> None:
        ledger, first = _known_ledger(tmp_path / name)
        state = json.loads(ledger.snapshot())
        mutate(state)  # type: ignore[operator]
        ledger.state_path.write_bytes(ledger_mod.canonical_bytes(state))
        with pytest.raises(ValueError, match=message):
            query_mod.BoundedBeliefQuery().query(ledger, _request(first, 1))

    run_case("schema", lambda state: state.update(schema="bad"), "snapshot.*schema")
    run_case(
        "state-index",
        lambda state: state.update(last_stream_index=True),
        "snapshot.*stream index",
    )
    run_case(
        "belief-state",
        lambda state: state["facts"][0].update(state="bad"),
        "belief state",
    )
    run_case(
        "fact-time",
        lambda state: state["facts"][0].update(last_stream_index=2),
        "newer than",
    )
    run_case(
        "missing-event",
        lambda state: state["events"].clear(),
        "missing source event",
    )
    run_case(
        "event-index",
        lambda state: next(iter(state["events"].values())).update(stream_index=True),
        "event.*stream index",
    )

    ledger, first = _known_ledger(tmp_path / "too-small")
    monkeypatch.setattr(query_mod, "_raw_prompt_bytes", lambda _value: b"x" * 2048)
    with pytest.raises(ValueError, match="fixed query contract"):
        query_mod.BoundedBeliefQuery(
            query_mod.BeliefQueryLimits(max_bytes=query_mod.MIN_QUERY_BYTES)
        ).query(ledger, _request(first, 1))


def test_req_7023_detects_snapshot_change_during_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7023-RESTART-STABILITY detects a changing reader."""

    ledger, first = _known_ledger(tmp_path / "changing")
    original = ledger.snapshot
    calls = 0

    def changing_snapshot() -> bytes:
        nonlocal calls
        calls += 1
        return original() if calls == 1 else original() + b" "

    monkeypatch.setattr(ledger, "snapshot", changing_snapshot)
    with pytest.raises(RuntimeError, match="mutated"):
        query_mod.BoundedBeliefQuery().query(ledger, _request(first, 1))


def test_scenario_7023_query_does_not_mutate_and_restart_hash_is_exact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7023-RESTART-STABILITY protects ledger bytes."""

    ledger, first = _known_ledger(tmp_path / "immutable")
    state_before = ledger.snapshot()
    state_hash_before = ledger.state_hash
    journal_before = ledger.journal_path.read_bytes()
    result = query_mod.BoundedBeliefQuery().query(ledger, _request(first, 1))
    restarted = ledger.restart()
    restarted_result = query_mod.BoundedBeliefQuery().query(restarted, _request(first, 1))

    assert ledger.snapshot() == state_before == restarted.snapshot()
    assert ledger.state_hash == state_hash_before == restarted.state_hash
    assert ledger.journal_path.read_bytes() == journal_before
    assert query_mod.canonical_prompt_bytes(result) == query_mod.canonical_prompt_bytes(
        restarted_result
    )


def test_req_7023_blocked_preconditions_are_complete(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7023 emits a blocked artifact for an absent Exp7022 gate."""

    artifact = query_mod.build_artifact(
        repo_root=tmp_path / "missing",
        output_path=tmp_path / "blocked.json",
        checkpoint_root=tmp_path / "checkpoint",
    )

    assert artifact["belief_query_api_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_belief_query_api"
    assert artifact["gate_check_summary"]["failed_check"] == "source_hash:exp7022"
    assert set(artifact["field_principles"]) == set(query_mod.REQUIRED_ARTIFACT_FIELDS)
    assert query_mod.validate_artifact(artifact) == []


def test_req_7023_complete_artifact_has_all_acceptance_and_rejection_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7023-READINESS-GATE derives readiness from every row."""

    artifact = query_mod.build_artifact(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "result.json",
        checkpoint_root=tmp_path / "checkpoint",
    )

    assert query_mod.validate_artifact(artifact) == []
    assert artifact["belief_query_api_ready_score"] == 1
    assert artifact["inference_substrate"] == "deterministic_bounded_belief_query_no_llm"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_positive_")
    assert len(artifact["query_fixture_rows"]) >= 8
    assert all(
        row["passed"] for field in query_mod.ACCEPTANCE_ROW_TABLES for row in artifact[field]
    )
    assert all(row["serialized_bytes"] <= row["byte_budget"] for row in artifact["budget_rows"])
    assert all(row["source_event_count"] <= row["event_budget"] for row in artifact["budget_rows"])
    assert artifact["truncation_rows"]
    assert artifact["contradiction_preservation_rows"]
    assert len(artifact["prohibited_field_rows"]) == len(query_mod.PROHIBITED_QUERY_KEYS) + 1
    assert artifact["restart_stability_rows"][0]["fresh_process_match"] is True


def test_req_7023_validator_rejects_contract_and_gate_drift(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7023-READINESS-GATE rejects false positive artifacts."""

    blocked = query_mod.build_artifact(
        repo_root=tmp_path / "missing",
        output_path=tmp_path / "blocked.json",
        checkpoint_root=tmp_path / "checkpoint",
    )
    cases: list[tuple[dict, str]] = []
    missing = deepcopy(blocked)
    missing.pop("rows")
    cases.append((missing, "required_fields_missing"))
    principles = deepcopy(blocked)
    principles["field_principles"].pop("rows")
    cases.append((principles, "field_principles_mismatch"))
    bad_score = deepcopy(blocked)
    bad_score["belief_query_api_ready_score"] = False
    cases.append((bad_score, "ready_score_not_bare_integer"))
    bad_class = deepcopy(blocked)
    bad_class["verdict_class"] = "other"
    cases.append((bad_class, "verdict_class_invalid"))
    bad_prefix = deepcopy(blocked)
    bad_prefix["honest_verdict"] = "complete_positive_wrong"
    cases.append((bad_prefix, "verdict_prefix_mismatch"))
    bad_substrate = deepcopy(blocked)
    bad_substrate["inference_substrate"] = "other"
    cases.append((bad_substrate, "inference_substrate_mismatch"))
    bad_oracle = deepcopy(blocked)
    bad_oracle["verifier_is_oracle"] = True
    cases.append((bad_oracle, "verifier_is_oracle_mismatch"))
    nonterminal = deepcopy(blocked)
    nonterminal["rows"] = [{"terminal": False}]
    cases.append((nonterminal, "nonterminal_row"))
    checksum = deepcopy(blocked)
    checksum["random_seed"] += 1
    cases.append((checksum, "reproducibility_checksum_mismatch"))

    for artifact, marker in cases:
        assert any(marker in error for error in query_mod.validate_artifact(artifact))

    assert query_mod.validate_artifact([]) == ["artifact_object_required"]


def test_req_7023_fresh_process_failure_and_partial_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7023-READINESS-GATE cannot hide a child failure."""

    monkeypatch.setattr(
        query_mod,
        "_fresh_process_fixture_digest",
        lambda *_args: {"error": "fixture child failed"},
    )
    artifact = query_mod.build_artifact(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "partial.json",
        checkpoint_root=tmp_path / "checkpoint",
    )
    assert artifact["belief_query_api_ready_score"] == 0
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("partial_")
    assert artifact["gate_check_summary"]["failed_check"] == "restart_stability_rows"


def test_req_7023_fresh_process_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7023-RESTART-STABILITY preserves child failures."""

    def completed(returncode: int, stdout: str, stderr: str = "") -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess([], returncode, stdout, stderr)

    monkeypatch.setattr(
        query_mod.subprocess,
        "run",
        lambda *args, **kwargs: completed(1, "", "child failed"),
    )
    assert query_mod._fresh_process_fixture_digest(
        REPO_ROOT, REPO_ROOT / query_mod.FIXTURE_PATH
    ) == {"error": "child failed"}
    monkeypatch.setattr(
        query_mod.subprocess,
        "run",
        lambda *args, **kwargs: completed(0, "not-json"),
    )
    assert query_mod._fresh_process_fixture_digest(
        REPO_ROOT, REPO_ROOT / query_mod.FIXTURE_PATH
    ) == {"error": "fresh process returned invalid JSON"}
    monkeypatch.setattr(
        query_mod.subprocess,
        "run",
        lambda *args, **kwargs: completed(0, "[]"),
    )
    assert query_mod._fresh_process_fixture_digest(
        REPO_ROOT, REPO_ROOT / query_mod.FIXTURE_PATH
    ) == {"error": "fresh process returned non-object"}


def test_req_7023_validator_ready_and_query_result_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-7023 validator derives readiness from rows and exact results."""

    blocked = query_mod.build_artifact(
        repo_root=tmp_path / "missing-ready",
        output_path=tmp_path / "blocked-ready.json",
        checkpoint_root=tmp_path / "checkpoint-ready",
    )
    not_list = deepcopy(blocked)
    not_list["rows"] = {}
    assert "rows_not_list" in query_mod.validate_artifact(not_list)

    bad_gate = deepcopy(blocked)
    bad_gate["gate_check_summary"] = []
    assert "gate_check_summary_invalid" in query_mod.validate_artifact(bad_gate)

    bad_result = deepcopy(blocked)
    bad_result["query_result_rows"] = [{"terminal": True, "result": {}}]
    assert "query_result_keys_mismatch" in query_mod.validate_artifact(bad_result)

    prohibited_result = deepcopy(blocked)
    complete_result = {key: [] for key in query_mod.EXACT_RESULT_KEYS}
    prohibited_result["query_result_rows"] = [{"terminal": True, "result": complete_result}]
    monkeypatch.setattr(
        query_mod,
        "PROHIBITED_QUERY_KEYS",
        query_mod.PROHIBITED_QUERY_KEYS | {"schema"},
    )
    assert "query_result_prohibited_fields" in query_mod.validate_artifact(prohibited_result)

    ready = deepcopy(blocked)
    ready["belief_query_api_ready_score"] = 1
    ready["verdict_class"] = "positive"
    ready["honest_verdict"] = "complete_positive_invalid"
    assert "ready_gate_inconsistent" in query_mod.validate_artifact(ready)
    assert any(
        error.startswith("ready_rows_failed:") for error in query_mod.validate_artifact(ready)
    )

    budget = deepcopy(ready)
    budget["gate_check_summary"] = {
        "failed_check": None,
        "passed": True,
    }
    for field in query_mod.ACCEPTANCE_ROW_TABLES:
        budget[field] = [{"terminal": True, "passed": True}]
    budget["budget_rows"] = [
        {
            "terminal": True,
            "passed": True,
            "serialized_bytes": 2,
            "byte_budget": 1,
            "source_event_count": 0,
            "event_budget": 1,
        }
    ]
    assert "ready_budget_exceeded" in query_mod.validate_artifact(budget)


def test_req_7023_direct_writer_and_main_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-7023 covers the direct command branches without tracked writes."""

    blocked = query_mod.build_artifact(
        repo_root=tmp_path / "missing-main",
        output_path=tmp_path / "blocked-main.json",
        checkpoint_root=tmp_path / "checkpoint-main",
    )
    direct = tmp_path / "direct.json"
    query_mod.write_artifact(direct, blocked)
    assert json.loads(direct.read_text(encoding="utf-8")) == blocked

    fixture = REPO_ROOT / query_mod.FIXTURE_PATH
    assert query_mod.main(["--fresh-fixture-digest", str(fixture)]) == 0
    assert json.loads(capsys.readouterr().out)["ledger_unchanged"] is True

    output = tmp_path / "main.json"
    monkeypatch.setattr(query_mod, "build_artifact", lambda **_kwargs: blocked)
    assert query_mod.main(["--output", str(output), "--repo-root", str(REPO_ROOT)]) == 0
    assert output.is_file()
    assert "belief_query_api_ready_score=0" in capsys.readouterr().out

    monkeypatch.setattr(query_mod, "validate_artifact", lambda _artifact: ["invalid"])
    with pytest.raises(ValueError, match="artifact validation failed"):
        query_mod.main(["--output", str(tmp_path / "invalid.json")])


def test_req_7023_required_command_writes_only_requested_artifact(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7023 exercises the experiment command with isolated outputs."""

    output = tmp_path / "result.json"
    command = [
        sys.executable,
        str(REPO_ROOT / query_mod.WRAPPER_PATH),
        "--date",
        "20260905",
        "--repo-root",
        str(REPO_ROOT),
        "--output",
        str(output),
        "--checkpoint-root",
        str(tmp_path / "checkpoint"),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert completed.returncode == 0, completed.stderr
    assert artifact["belief_query_api_ready_score"] == 1
    assert query_mod.validate_artifact(artifact) == []
    assert "belief_query_api_ready_score=1" in completed.stdout

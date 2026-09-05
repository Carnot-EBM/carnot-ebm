"""Tests for the sealed prospective ARC belief comparison.

Spec refs: REQ-CSL-7021 and SCENARIO-CSL-7021-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.agentic import arc_belief_ledger as ledger_mod
from carnot.agentic import arc_prospective_belief_utility as mod


ROOT = Path(__file__).resolve().parents[2]


def _event(
    index: int,
    *,
    attempt: str,
    hypothesis: str,
    action_x: int,
    action_y: int,
    target: str,
    mechanic_group: str = "action_6:single:same_level:away_from_action",
) -> dict:
    """Build a small row whose outcome class is explicit in observable fields."""

    row = ledger_mod._fixture_event(
        index,
        mechanic=f"mechanic-{hypothesis}",
        outcome=f"outcome-{index}",
    )
    row["source_attempt_time"] = attempt
    row["source_transition_index"] = index
    row["action"] = {"type": 6, "data": {"x": action_x, "y": action_y}}
    row["mechanic_signature"]["hypothesis_key"] = ledger_mod._sha256_json(
        {"hypothesis": hypothesis}
    )
    row["mechanic_signature"]["mechanic_group"] = mechanic_group
    delta = row["next_observation"]["state_delta"]
    delta["changed_cell_count"] = 3 if target == "progress" else (1 if target != "invalid" else 0)
    delta["touches_action_coordinate"] = target == "progress"
    row["mechanic_signature"]["state_delta"] = deepcopy(delta)
    row["mechanic_signature"]["spatial_summary"] = {
        "touches_action_coordinate": delta["touches_action_coordinate"]
    }
    row["contradiction"]["hypothesis_key"] = row["mechanic_signature"]["hypothesis_key"]
    row["contradiction"]["is_contradiction"] = target == "contradicted"
    row["row_hash"] = ledger_mod.event_row_hash(row)
    return row


def _stream() -> list[dict]:
    """Return construction and held rows with support, headroom, and missing candidates."""

    first = "20260905T010000_000000"
    later = "20260905T020000_000000"
    return [
        _event(0, attempt=first, hypothesis="shared", action_x=9, action_y=9, target="progress"),
        _event(1, attempt=first, hypothesis="noise-a", action_x=7, action_y=7, target="valid"),
        _event(2, attempt=first, hypothesis="noise-b", action_x=8, action_y=8, target="valid"),
        _event(
            3,
            attempt=later,
            hypothesis="shared",
            action_x=1,
            action_y=1,
            target="contradicted",
        ),
        _event(4, attempt=later, hypothesis="shared", action_x=9, action_y=9, target="progress"),
        _event(5, attempt=later, hypothesis="solo", action_x=2, action_y=2, target="valid"),
        _event(6, attempt=later, hypothesis="shared", action_x=1, action_y=1, target="valid"),
    ]


def _sidecar(events: list[dict]) -> list[dict]:
    """Build sealed receipts without putting future fields into a policy input."""

    return [
        {
            "schema": mod.SIDECAR_SCHEMA,
            "after_stream_index": event["stream_index"],
            "event_id": event["event_id"],
            "later_event_count": len(events) - event["stream_index"] - 1,
            "later_level_boundary_count": 0,
            "later_max_observed_level": 0,
            "later_same_hypothesis_outcome_keys": [],
            "row_hash": ledger_mod._sha256_json({"sidecar": event["stream_index"]}),
        }
        for event in events
    ]


def test_req_csl_7021_spec_precedes_implementation() -> None:
    """REQ-CSL-7021 defines all planned scenarios and artifact fields."""

    text = (ROOT / mod.CSL_SPEC_PATH).read_text(encoding="utf-8")
    for marker in (
        "SCENARIO-CSL-7021-FROZEN-CHRONOLOGY",
        "SCENARIO-CSL-7021-CAPACITY-AND-RETENTION",
        "SCENARIO-CSL-7021-SUPPORT-HEADROOM-AND-TIES",
        "SCENARIO-CSL-7021-CLUSTER-BOOTSTRAP",
        "SCENARIO-CSL-7021-LEAKAGE-AND-RECOMPUTATION",
        "SCENARIO-CSL-7021-TERMINAL-GATES",
    ):
        assert marker in text
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in text


def test_scenario_7021_freeze_does_not_read_held_outcomes() -> None:
    """SCENARIO-CSL-7021-FROZEN-CHRONOLOGY freezes outcome-blind unit IDs."""

    events = _stream()
    frozen = mod.freeze_protocol(events, capacity=2, random_seed=17)
    changed = deepcopy(events)
    changed[3]["next_observation"]["state_delta"]["changed_cell_count"] = 99
    changed[3]["contradiction"]["is_contradiction"] = False
    changed[3]["row_hash"] = ledger_mod.event_row_hash(changed[3])

    assert mod.freeze_protocol(changed, capacity=2, random_seed=17) == frozen
    assert frozen["construction_stream_indices"] == [0, 1, 2]
    assert frozen["held_stream_indices"] == [3, 4, 5, 6]
    assert frozen["protocol_frozen_before_sidecar_open"] is True
    assert frozen["capacity"] == 2
    assert set(frozen["arm_names"]) == set(mod.ARMS)


def test_scenario_7021_strict_chronology_equal_capacity_and_retention(tmp_path: Path) -> None:
    """SCENARIO-CSL-7021-CAPACITY-AND-RETENTION keeps every query prospective."""

    events = _stream()
    protocol = mod.freeze_protocol(events, capacity=2, random_seed=17)
    comparison = mod.run_comparison(events, _sidecar(events), protocol, tmp_path)

    assert len(comparison["per_decision_results"]) == 4 * len(mod.ARMS)
    assert all(row["max_evidence_stream_index"] < row["stream_index"] for row in comparison["per_decision_results"])
    assert {row["declared_capacity"] for row in comparison["per_decision_results"]} == {2}
    assert all(row["memory_item_count"] <= 2 for row in comparison["per_decision_results"])
    assert len(comparison["query_cost_rows"]) == 4 * len(mod.ARMS)
    assert len(comparison["retention_rows"]) == 4 * len(mod.ARMS)
    assert all(row["memory_bytes"] >= 0 and row["query_time_us"] >= 0 for row in comparison["query_cost_rows"])


def test_scenario_7021_support_headroom_ties_and_missing_candidates() -> None:
    """SCENARIO-CSL-7021-SUPPORT-HEADROOM-AND-TIES keeps null classes separate."""

    useful = {"action_id": "a", "target_class": "progress_producing"}
    contradicted = {"action_id": "b", "target_class": "contradicted"}
    valid = {"action_id": "c", "target_class": "valid"}

    assert mod.classify_candidate_support([useful]) == "unsupported"
    assert mod.classify_candidate_support([useful, valid]) == "no_headroom"
    assert mod.classify_candidate_support([contradicted, contradicted]) == "no_headroom"
    score = mod.pairwise_ranking_accuracy([useful, contradicted], {"a": 0, "b": 0})
    assert score == {"accuracy": 0.5, "pair_count": 1, "credit_sum": 0.5}
    assert mod.target_class(_event(0, attempt="a", hypothesis="h", action_x=0, action_y=0, target="invalid")) == "invalid"


def test_scenario_7021_cluster_bootstrap_uses_decision_groups() -> None:
    """SCENARIO-CSL-7021-CLUSTER-BOOTSTRAP samples one value per decision."""

    values = {"unit-a": 1.0, "unit-b": 0.0, "unit-c": 0.5}
    first = mod.cluster_bootstrap(values, seed=7021, resamples=500)
    second = mod.cluster_bootstrap(dict(reversed(list(values.items()))), seed=7021, resamples=500)

    assert first == second
    assert first["cluster_count"] == 3
    assert first["point_estimate"] == pytest.approx(0.5)
    assert first["lower"] <= first["point_estimate"] <= first["upper"]
    assert mod.cluster_bootstrap({}, seed=1, resamples=10)["point_estimate"] is None
    with pytest.raises(TypeError, match="mapping"):
        mod.cluster_bootstrap([1.0, 0.0], seed=1, resamples=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        mod.cluster_bootstrap(values, seed=1, resamples=0)


def test_scenario_7021_current_outcome_cannot_change_its_ranking(tmp_path: Path) -> None:
    """SCENARIO-CSL-7021-LEAKAGE-AND-RECOMPUTATION excludes the current target."""

    events = _stream()
    protocol = mod.freeze_protocol(events, capacity=3, random_seed=23)
    original = mod.run_comparison(events, _sidecar(events), protocol, tmp_path / "first")
    changed = deepcopy(events)
    changed[4]["next_observation"]["state_delta"]["changed_cell_count"] = 0
    changed[4]["next_observation"]["state_delta"]["touches_action_coordinate"] = False
    changed[4]["mechanic_signature"]["state_delta"] = deepcopy(
        changed[4]["next_observation"]["state_delta"]
    )
    changed[4]["row_hash"] = ledger_mod.event_row_hash(changed[4])
    replay = mod.run_comparison(changed, _sidecar(changed), protocol, tmp_path / "second")

    before = [row for row in original["per_decision_results"] if row["stream_index"] == 4]
    after = [row for row in replay["per_decision_results"] if row["stream_index"] == 4]
    assert [(row["arm"], row["candidate_scores"], row["selected_action_id"]) for row in before] == [
        (row["arm"], row["candidate_scores"], row["selected_action_id"]) for row in after
    ]
    assert all(row["current_outcome_used_by_policy"] is False for row in before + after)
    assert all(row["strict_chronology"] is True for row in before + after)


def test_scenario_7021_aggregate_recomputation_detects_drift(tmp_path: Path) -> None:
    """SCENARIO-CSL-7021-LEAKAGE-AND-RECOMPUTATION reduces only row evidence."""

    events = _stream()
    protocol = mod.freeze_protocol(events, capacity=3, random_seed=29)
    comparison = mod.run_comparison(events, _sidecar(events), protocol, tmp_path)
    recomputed = mod.recompute_aggregate_rows(comparison)

    assert recomputed == comparison["rows"]
    changed = deepcopy(comparison)
    changed["rows"][0]["action_ranking_accuracy"] = 99.0
    audit = mod.audit_aggregate_recomputation(changed)
    assert audit["matches"] is False
    assert audit["expected_rows"] == recomputed
    assert audit["observed_rows"] == changed["rows"]


def test_req_csl_7021_blocked_preconditions_are_complete(tmp_path: Path) -> None:
    """REQ-CSL-7021 writes the first exact missing precondition in a blocked artifact."""

    artifact = mod.build_artifact(
        repo_root=tmp_path / "missing",
        output_path=tmp_path / "blocked.json",
        run_date="20260905",
    )

    assert artifact["belief_utility_comparison_complete_score"] == 0
    assert artifact["belief_future_utility_positive_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_prospective_belief_utility"
    assert artifact["gate_check_summary"]["failed_check"] == "exp7020_artifact_readable"
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_artifact(artifact) == []


def test_req_csl_7021_real_artifact_recomputes_and_has_all_rows(tmp_path: Path) -> None:
    """SCENARIO-CSL-7021-TERMINAL-GATES validates the real sealed comparison."""

    artifact = mod.build_artifact(
        repo_root=ROOT,
        output_path=tmp_path / "result.json",
        run_date="20260905",
    )

    assert mod.validate_artifact(artifact) == []
    assert artifact["belief_utility_comparison_complete_score"] == 1
    assert type(artifact["belief_future_utility_positive_score"]) is int
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["frozen_split"]["protocol_frozen_before_sidecar_open"] is True
    assert artifact["frozen_split"]["sidecar_opened_after_freeze"] is True
    assert len(artifact["per_decision_results"]) == 10 * len(mod.ARMS)
    assert len(artifact["unsupported_rows"]) + len(artifact["no_headroom_rows"]) + len(
        {row["unit_id"] for row in artifact["action_ranking_rows"]}
    ) == 10
    assert all(row["same_capacity"] for row in artifact["capacity_match_rows"])
    assert artifact["aggregate_row_recomputation"]["matches"] is True
    assert all(row["passed"] for row in artifact["leakage_check_rows"])
    assert artifact["verdict_class"] in {"positive", "null"}
    if artifact["belief_future_utility_positive_score"] == 0:
        assert artifact["verdict_class"] == "null"
        assert artifact["honest_verdict"].startswith("complete_null_")


def test_req_csl_7021_validator_rejects_schema_and_gate_drift(tmp_path: Path) -> None:
    """REQ-CSL-7021 rejects missing fields, bad scores, drift, and oracle claims."""

    blocked = mod.build_artifact(
        repo_root=tmp_path / "missing",
        output_path=tmp_path / "blocked.json",
    )
    assert mod.validate_artifact([]) == ["artifact_object_required"]
    cases: list[tuple[dict, str]] = []
    missing = deepcopy(blocked)
    missing.pop("rows")
    cases.append((missing, "required_fields"))
    principle = deepcopy(blocked)
    principle["field_principles"].pop("rows")
    cases.append((principle, "field_principles"))
    boolean = deepcopy(blocked)
    boolean["belief_future_utility_positive_score"] = False
    cases.append((boolean, "positive_score"))
    substrate = deepcopy(blocked)
    substrate["inference_substrate"] = "llm"
    cases.append((substrate, "inference_substrate"))
    oracle = deepcopy(blocked)
    oracle["verifier_is_oracle"] = True
    cases.append((oracle, "verifier_is_oracle"))
    verdict = deepcopy(blocked)
    verdict["honest_verdict"] = "complete_positive_wrong"
    cases.append((verdict, "verdict_prefix"))
    checksum = deepcopy(blocked)
    checksum["random_seed"] += 1
    cases.append((checksum, "reproducibility_checksum"))
    terminal = deepcopy(blocked)
    terminal["rows"] = [{"terminal": False}]
    terminal["reproducibility_checksum"] = mod.artifact_checksum(terminal)
    cases.append((terminal, "nonterminal"))
    for artifact, marker in cases:
        assert any(marker in error for error in mod.validate_artifact(artifact))


def test_req_csl_7021_sidecar_and_freeze_validation_errors(tmp_path: Path) -> None:
    """REQ-CSL-7021 fails closed on malformed splits and sealed receipts."""

    with pytest.raises(ValueError, match="chronological"):
        mod.freeze_protocol(list(reversed(_stream())))
    with pytest.raises(ValueError, match="later attempt"):
        mod.freeze_protocol(_stream()[:3])
    with pytest.raises(ValueError, match="capacity"):
        mod.freeze_protocol(_stream(), capacity=0)
    path = tmp_path / "sidecar.jsonl"
    with pytest.raises(ValueError, match="unreadable"):
        mod.load_sidecar(path)
    path.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        mod.load_sidecar(path)
    path.write_text(json.dumps({"schema": "wrong"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        mod.load_sidecar(path)


def test_req_csl_7021_command_writes_requested_artifact(tmp_path: Path) -> None:
    """REQ-CSL-7021 exercises the required command without writing tracked test state."""

    output = tmp_path / "result.json"
    command = [
        sys.executable,
        str(ROOT / mod.WRAPPER_PATH),
        "--date",
        "20260905",
        "--repo-root",
        str(ROOT),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert completed.returncode == 0, completed.stderr
    assert mod.validate_artifact(artifact) == []
    assert "belief_utility_comparison_complete_score=1" in completed.stdout


def test_req_csl_7021_direct_main_and_writer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-CSL-7021 covers direct blocked execution and atomic artifact writing."""

    output = tmp_path / "direct.json"
    assert mod.main(["--repo-root", str(tmp_path / "missing"), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["verdict_class"] == "blocked"
    assert "belief_utility_comparison_complete_score=0" in capsys.readouterr().out

    artifact = mod.build_artifact(repo_root=tmp_path / "missing", output_path=output)
    artifact["random_seed"] += 1
    with pytest.raises(ValueError, match="validation failed"):
        mod.write_validated_artifact(tmp_path / "bad.json", artifact)


def test_req_csl_7021_defensive_input_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-CSL-7021 rejects malformed visible inputs and mismatched receipts."""

    with pytest.raises(ValueError, match="action mapping"):
        mod.action_id({"action": None})
    assert mod.target_class({}) == "invalid"

    missing_attempt = _stream()
    missing_attempt[0]["source_attempt_time"] = ""
    with pytest.raises(ValueError, match="source attempt"):
        mod.freeze_protocol(missing_attempt)

    interleaved = _stream()
    interleaved[4]["source_attempt_time"] = interleaved[0]["source_attempt_time"]
    with pytest.raises(ValueError, match="precede"):
        mod.freeze_protocol(interleaved)

    missing_key = _stream()
    missing_key[3]["mechanic_signature"].pop("hypothesis_key")
    with pytest.raises(ValueError, match="hypothesis key"):
        mod.freeze_protocol(missing_key)

    events = _stream()
    protocol = mod.freeze_protocol(events)
    mismatched = _sidecar(events)
    mismatched[-1]["event_id"] = "wrong"
    with pytest.raises(ValueError, match="does not match"):
        mod.run_comparison(events, mismatched, protocol, tmp_path / "mismatch")

    def reject_split(_events: list[dict], **_kwargs: object) -> dict:
        raise ValueError("forced split rejection")

    monkeypatch.setattr(mod, "freeze_protocol", reject_split)
    checks, _hashes, loaded, frozen = mod.collect_preconditions(ROOT, tmp_path / "result.json")
    assert loaded is not None and frozen is None
    fixture_check = next(row for row in checks if row["check"] == "fixture_loadable")
    assert fixture_check["observed_value"] == "forced split rejection"


def test_req_csl_7021_positive_partial_and_validator_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CSL-7021 covers terminal verdict and complete-artifact guard paths."""

    events = ledger_mod.load_updater_events(ROOT / mod.FIXTURE_PATH)
    protocol = mod.freeze_protocol(events)
    sidecar = mod.load_sidecar(ROOT / mod.SIDECAR_PATH)
    comparison = mod.run_comparison(events, sidecar, protocol, tmp_path / "base")
    original_run_comparison = mod.run_comparison
    original_positive_gate = mod._positive_gate

    monkeypatch.setattr(mod, "run_comparison", lambda *_args, **_kwargs: deepcopy(comparison))
    monkeypatch.setattr(mod, "_positive_gate", lambda _artifact: (True, {"forced": True}))
    positive = mod.build_artifact(repo_root=ROOT, output_path=tmp_path / "positive.json")
    assert positive["verdict_class"] == "positive"
    assert positive["honest_verdict"].startswith("complete_positive_")

    partial_rows = deepcopy(comparison)
    partial_rows["per_decision_results"].pop()
    monkeypatch.setattr(mod, "run_comparison", lambda *_args, **_kwargs: deepcopy(partial_rows))
    monkeypatch.setattr(mod, "_positive_gate", original_positive_gate)
    partial = mod.build_artifact(repo_root=ROOT, output_path=tmp_path / "partial.json")
    assert partial["verdict_class"] == "partial"
    assert partial["honest_verdict"].startswith("partial_")

    monkeypatch.setattr(mod, "run_comparison", original_run_comparison)
    real = mod.build_artifact(repo_root=ROOT, output_path=tmp_path / "real.json")
    assert mod.validate_artifact(real) == []

    invalid_class = deepcopy(real)
    invalid_class["verdict_class"] = "unknown"
    invalid_class["reproducibility_checksum"] = mod.artifact_checksum(invalid_class)
    assert "verdict_class_invalid" in mod.validate_artifact(invalid_class)

    invalid_gate = deepcopy(real)
    invalid_gate["gate_check_summary"] = {}
    invalid_gate["reproducibility_checksum"] = mod.artifact_checksum(invalid_gate)
    assert "gate_check_summary_invalid" in mod.validate_artifact(invalid_gate)

    drift = deepcopy(real)
    drift["aggregate_row_recomputation"]["matches"] = False
    drift["reproducibility_checksum"] = mod.artifact_checksum(drift)
    assert "aggregate_row_recomputation_mismatch" in mod.validate_artifact(drift)

    short = deepcopy(real)
    short["per_decision_results"].pop()
    short["reproducibility_checksum"] = mod.artifact_checksum(short)
    assert "complete_decision_count_mismatch" in mod.validate_artifact(short)

    leaky = deepcopy(real)
    leaky["leakage_check_rows"][0]["passed"] = False
    leaky["reproducibility_checksum"] = mod.artifact_checksum(leaky)
    assert "complete_leakage_check_failed" in mod.validate_artifact(leaky)

    false_positive = deepcopy(real)
    false_positive["belief_future_utility_positive_score"] = 1
    false_positive["verdict_class"] = "positive"
    false_positive["honest_verdict"] = "complete_positive_invalid"
    false_positive["reproducibility_checksum"] = mod.artifact_checksum(false_positive)
    assert "positive_gate_inconsistent" in mod.validate_artifact(false_positive)

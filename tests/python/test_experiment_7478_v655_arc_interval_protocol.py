"""Tests for the V655 ARC interval accounting protocol.

Spec refs: REQ-ARC-WMTE-7478 and SCENARIO-ARC-WMTE-7478-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random

import pytest

from carnot import experiment_7478_v655_arc_interval_protocol as protocol
from carnot.reporting.experiment_7303_validation_scope import CommandSpec


REPO = Path(__file__).resolve().parents[2]


def _event(
    decision: str,
    seam: str,
    event: str,
    tick: int,
    *,
    episode: str = "episode-1",
    parent: str | None = None,
    clock: str = "clock-a",
    work_class: str = "unattributed_decision_seam",
    disposition: str | None = None,
) -> dict:
    row = {
        "run_id": "run-1",
        "process_id": 11,
        "episode_id": episode,
        "decision_id": decision,
        "parent_decision_id": parent,
        "seam": seam,
        "event": event,
        "event_monotonic_ns": tick,
        "clock_identity": clock,
        "work_class": work_class,
        "observer_cpu_ns": 0,
    }
    if event == "stage_start":
        row["interval_start_monotonic_ns"] = tick
    else:
        row["interval_end_monotonic_ns"] = tick
        row["disposition"] = disposition or "completed"
    return row


def _span(
    decision: str,
    seam: str,
    start: int,
    end: int,
    **kwargs: object,
) -> list[dict]:
    return [
        _event(decision, seam, "stage_start", start, **kwargs),
        _event(decision, seam, "stage_end", end, **kwargs),
    ]


def test_req_arc_wmte_7478_spec_and_frozen_schedule() -> None:
    """REQ-ARC-WMTE-7478 fixes aggregation identity and all 36 units."""

    text = (REPO / protocol.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("REQ-ARC-WMTE-7478") : text.index("REQ-ARC-WMTE-7471")]
    for anchor in (
        "SCENARIO-ARC-WMTE-7478-INTERVAL-UNION",
        "SCENARIO-ARC-WMTE-7478-INCOMPLETE-AND-CLOCKS",
        "SCENARIO-ARC-WMTE-7478-EPISODE-RESET",
        "SCENARIO-ARC-WMTE-7478-OBSERVER-PARITY",
        "SCENARIO-ARC-WMTE-7478-SCHEDULE",
        "SCENARIO-ARC-WMTE-7478-TERMINAL",
    ):
        assert anchor in section
    assert protocol.MODEL_SPECS == []
    assert protocol.INFERENCE_SUBSTRATE_CLASS == "aggregation"
    assert protocol.EXECUTION_VENUE == "host"

    manifest = protocol.build_arc_schedule_manifest()
    assert manifest["panel_a_games"] == ["sk48", "tr87", "s5i5", "lp85", "lf52", "cn04"]
    assert manifest["panel_b_games"] == ["tu93", "g50t", "tn36", "vc33", "re86", "dc22"]
    assert len(manifest["rows"]) == 36
    assert len({row["episode_id"] for row in manifest["rows"]}) == 36
    assert {row["seed"] for row in manifest["rows"]} == {65501, 65502, 65503}
    assert all(row["disposition"] == "unstarted" for row in manifest["rows"])
    assert all(row["action_limit"] == 180 for row in manifest["rows"])
    assert all(row["episode_limit_s"] == 240 for row in manifest["rows"])
    assert all(row["panel_live_limit_s"] == 3600 for row in manifest["rows"])
    assert manifest["manifest_sha256"] == protocol.schedule_checksum(manifest["rows"])


def test_scenario_arc_wmte_7478_nested_crossing_duplicate_and_zero() -> None:
    """SCENARIO-ARC-WMTE-7478-INTERVAL-UNION counts each tick once."""

    events = [
        *_span("outer", "candidate_action_selection", 0, 80),
        *_span(
            "choice",
            "induction_timing",
            10,
            30,
            parent="outer",
            work_class="replaceable_decision",
        ),
        *_span(
            "generation",
            "downstream_generation",
            15,
            20,
            parent="choice",
            work_class="text_generation",
        ),
        *_span("crossing", "hypothesis_gate", 70, 90),
        *_span(
            "zero",
            "supervisor_arm_selection",
            50,
            50,
            work_class="replaceable_decision",
        ),
    ]
    events.extend(deepcopy(events[:2]))
    reduced = protocol.reduce_episode_intervals(
        events,
        episode_id="episode-1",
        episode_start_ns=0,
        episode_end_ns=100,
    )

    assert reduced["stage_union_ns"] == 90
    assert reduced["unattributed_ns"] == 10
    assert reduced["stage_union_ns"] + reduced["unattributed_ns"] == 100
    assert reduced["replaceable_lower_ns"] == 15
    assert reduced["replaceable_upper_ns"] == 85
    assert reduced["duplicate_event_count"] == 2
    assert reduced["zero_duration_interval_count"] == 1
    assert reduced["nested_interval_count"] >= 2
    assert reduced["crossing_interval_pair_count"] >= 1
    assert reduced["bounds_valid"] is True
    outer = next(row for row in reduced["stage_rows"] if row["decision_id"] == "outer")
    assert outer["exclusive_ns"] == 60


def test_scenario_arc_wmte_7478_incomplete_clocks_and_killed_child() -> None:
    """SCENARIO-ARC-WMTE-7478-INCOMPLETE-AND-CLOCKS fails closed."""

    events = [
        *_span(
            "killed",
            "supervisor_arm_selection",
            10,
            20,
            work_class="replaceable_decision",
            disposition="killed_child",
        ),
        _event("unfinished", "candidate_action_selection", "stage_start", 40),
        _event("wrong-clock", "hypothesis_gate", "stage_start", 30, clock="clock-a"),
        _event("wrong-clock", "hypothesis_gate", "stage_end", 60, clock="clock-b"),
    ]
    reduced = protocol.reduce_episode_intervals(
        events,
        episode_id="episode-1",
        episode_start_ns=0,
        episode_end_ns=100,
    )

    assert reduced["stage_union_ns"] == 10
    assert reduced["stage_upper_union_ns"] == 70
    assert reduced["unattributed_ns"] == 90
    assert reduced["incomplete_interval_count"] == 1
    assert reduced["mismatched_clock_count"] == 1
    assert reduced["killed_child_interval_count"] == 1
    assert reduced["replaceable_lower_ns"] == 10
    assert 10 <= reduced["replaceable_upper_ns"] <= 100
    assert reduced["bounds_valid"] is True


def test_scenario_arc_wmte_7478_episode_reset_and_forged_duplicate() -> None:
    """SCENARIO-ARC-WMTE-7478-EPISODE-RESET isolates repeated IDs."""

    first = _span(
        "same-id",
        "induction_timing",
        10,
        20,
        episode="episode-1",
        work_class="replaceable_decision",
    )
    forged = [
        *first,
        _event(
            "same-id",
            "induction_timing",
            "stage_end",
            90,
            episode="episode-1",
            work_class="replaceable_decision",
        ),
    ]
    baseline = protocol.reduce_episode_intervals(
        first, episode_id="episode-1", episode_start_ns=0, episode_end_ns=100
    )
    attacked = protocol.reduce_episode_intervals(
        forged, episode_id="episode-1", episode_start_ns=0, episode_end_ns=100
    )
    second = protocol.reduce_episode_intervals(
        _span(
            "same-id",
            "induction_timing",
            40,
            55,
            episode="episode-2",
            work_class="replaceable_decision",
        ),
        episode_id="episode-2",
        episode_start_ns=0,
        episode_end_ns=100,
    )

    assert attacked["stage_union_ns"] == baseline["stage_union_ns"] == 10
    assert attacked["conflicting_duplicate_count"] == 1
    assert second["stage_union_ns"] == 15
    assert second["replaceable_lower_ns"] == 15


def test_scenario_arc_wmte_7478_observer_preserves_behavior() -> None:
    """SCENARIO-ARC-WMTE-7478-OBSERVER-PARITY preserves all policy effects."""

    def run(observer: protocol.IntervalProtocolObserver | None) -> dict:
        rng = random.Random(65501)
        calls: list[str] = []
        provenance: list[dict] = []

        def choose() -> str:
            calls.append("choose")
            action = rng.choice(["ACTION1", "ACTION2"])
            provenance.append({"action": action, "source": "deterministic_fixture"})
            return action

        action = (
            choose()
            if observer is None
            else observer.call("candidate_action_selection", "replaceable_decision", choose)
        )
        return {
            "action": action,
            "calls": calls,
            "provenance": provenance,
            "random_state": rng.getstate(),
        }

    rows: list[dict] = []
    observer = protocol.IntervalProtocolObserver(
        run_id="fixture-run",
        process_id=99,
        episode_id="fixture-episode",
        sink=rows.append,
        clock_ns=iter(range(100, 1000, 10)).__next__,
    )
    assert run(observer) == run(None)
    assert [row["event"] for row in rows] == ["stage_start", "stage_end"]
    assert rows[0]["parent_decision_id"] is None
    assert rows[0]["work_class"] == "replaceable_decision"
    assert rows[0]["run_id"] == "fixture-run"
    assert rows[0]["process_id"] == 99


def test_req_arc_wmte_7478_historical_reduction_and_flags() -> None:
    """REQ-ARC-WMTE-7478 independently corrects all eight historical rows."""

    checks, hashes, upstream = protocol.collect_preconditions(REPO)
    assert all(row["passed"] is True for row in checks)
    assert upstream["honest_verdict"] == "complete_null_live_arc_seam_observation"
    assert upstream["verdict_class"] == "null"
    assert upstream["flagged_adversarial"] is False
    assert hashes[protocol.UPSTREAM_PATH.as_posix()]["original_flags"]["verdict_class"] == "null"

    reduction = protocol.reduce_exp7471(REPO, upstream)
    assert len(reduction["interval_rows"]) == 8
    assert reduction["timing_correction"]["historical_inclusive_sum_ns"] == 2_996_406_284_078
    assert reduction["timing_correction"]["historical_live_work_s"] == 1533.539562
    assert reduction["timing_correction"]["corrected_episode_union_ns"] < 2_996_406_284_078
    assert reduction["timing_correction"]["historical_parentage_complete"] is False
    assert reduction["timing_correction"]["historical_process_identity_complete"] is False
    assert all(row["bounds_valid"] for row in reduction["interval_rows"])
    assert all(
        row["stage_union_ns"] + row["unattributed_ns"] == row["observed_episode_ns"]
        for row in reduction["interval_rows"]
    )
    assert all(row["repeated_decision_id_count"] == 0 for row in reduction["interval_rows"])
    assert all(row["incomplete_interval_count"] == 0 for row in reduction["interval_rows"])


def test_scenario_arc_wmte_7478_terminal_fixture_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7478-TERMINAL is complete and independently reducible."""

    artifact = protocol.build_artifact_for_test(REPO)
    assert protocol.validate_artifact(artifact, require_terminal=False) == []
    assert protocol.independent_reduce(artifact)["matches_declared"] is True
    assert artifact["arc_interval_protocol_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert all(value == 0 for value in artifact["invocation_counts"].values())
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["historical_receipt_sidecars"]
    serialized_sidecars = json.dumps(artifact["historical_receipt_sidecars"], sort_keys=True)
    assert "MODEL_SPECS" not in serialized_sidecars
    assert "model_invoked" not in serialized_sidecars
    assert "invocation_counts" not in serialized_sidecars
    assert all(
        row["scope"] == "historical"
        and row["sha256"].startswith("sha256:")
        and len(row["sha256"]) == 71
        for row in artifact["historical_receipt_sidecars"]
    )
    assert artifact["verdict_class"] == "null"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["observer_parity"]["passed"] is True
    assert len(artifact["arc_schedule_manifest"]["rows"]) == 36
    assert artifact["sample_size_budget"]["unstarted_independent_units"] == 36
    for field in protocol.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]

    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert protocol.main(["--replay", str(path)]) == 0
    changed = deepcopy(artifact)
    changed["interval_rows"][0]["stage_union_ns"] += 1
    assert "independent_reduction_mismatch" in protocol.validate_artifact(
        changed, require_terminal=False
    )


def test_req_arc_wmte_7478_manifest_and_command_scope(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7478 freezes focused validation and capability E2E checks."""

    commands = protocol.build_validation_plan(REPO, tmp_path)
    assert protocol.validate_validation_plan(REPO, commands) == []
    by_name = {row.name: row for row in commands}
    assert set(protocol.AFFECTED_CHECK_NAMES).issubset(by_name)
    assert set(protocol.CAPABILITY_E2E_NAMES).issubset(by_name)
    assert all("tests/python" not in row.argv for row in commands)
    assert "tests/python/test_arc_decision_telemetry.py" in by_name["e2e_011"].argv
    assert by_name["private_arc_smoke"].scope == "private_real_environment_smoke"

    terminal = protocol.terminal_command_specs(REPO, tmp_path / "candidate.json")
    assert [row.name for row in terminal] == list(protocol.REQUIRED_TERMINAL_NAMES)


def test_req_arc_wmte_7478_validation_fails_closed() -> None:
    """REQ-ARC-WMTE-7478 keeps readiness separate from benefit and validity."""

    artifact = protocol.build_artifact_for_test(REPO)
    changed = deepcopy(artifact)
    changed["arc_interval_protocol_ready_score"] = 0
    assert "readiness_score_mismatch" in protocol.validate_artifact(changed, require_terminal=False)

    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["archived-model"]
    assert "current_model_declaration_invalid" in protocol.validate_artifact(
        changed, require_terminal=False
    )

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:forged"
    assert "reproducibility_checksum_mismatch" in protocol.validate_artifact(
        changed, require_terminal=False
    )


def test_req_arc_wmte_7478_defensive_inputs_and_observer_failures(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7478 rejects malformed rows and retains failed call ends."""

    assert protocol.utc_now().endswith("Z")
    assert protocol.load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert protocol.load_object(malformed) == {}
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert protocol.load_object(array) == {}

    malformed_jsonl = tmp_path / "malformed.jsonl"
    malformed_jsonl.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed_jsonl"):
        protocol.read_jsonl(malformed_jsonl)
    non_object = tmp_path / "non-object.jsonl"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non_object_jsonl"):
        protocol.read_jsonl(non_object)
    with pytest.raises(ValueError, match="unsupported_gate_operator"):
        protocol._gate(
            "bad",
            "validity",
            1,
            1,
            op="!=",
            upstream="fixture",
            field="fixture",
            principle="An unknown comparison must fail closed.",
        )

    assert protocol.union_duration_ns([(10, 5), (20, 25)]) == 5
    with pytest.raises(ValueError, match="episode_boundary_order_invalid"):
        protocol.reduce_episode_intervals(
            [], episode_id="episode-1", episode_start_ns=2, episode_end_ns=1
        )
    fallback_rows = [
        {
            **_event("fallback", "induction_timing", "stage_start", 10),
            "interval_start_monotonic_ns": None,
        },
        {
            **_event("fallback", "induction_timing", "stage_end", 20),
            "interval_end_monotonic_ns": None,
        },
        *_span("outside", "candidate_action_selection", -20, -10),
    ]
    fallback = protocol.reduce_episode_intervals(
        fallback_rows, episode_id="episode-1", episode_start_ns=0, episode_end_ns=100
    )
    assert fallback["stage_union_ns"] == 10

    rows: list[dict] = []
    observer = protocol.IntervalProtocolObserver(
        run_id="failure-run",
        process_id=5,
        episode_id="failure-episode",
        sink=rows.append,
        clock_ns=iter((1, 2, 3, 4)).__next__,
    )
    with pytest.raises(RuntimeError, match="fixture"):
        observer.call(
            "candidate_action_selection",
            "replaceable_decision",
            lambda: (_ for _ in ()).throw(RuntimeError("fixture")),
        )
    with pytest.raises(SystemExit):
        observer.call(
            "candidate_action_selection",
            "replaceable_decision",
            lambda: (_ for _ in ()).throw(SystemExit(2)),
        )
    assert [row["disposition"] for row in rows if row["event"] == "stage_end"] == [
        "failed",
        "killed_child",
    ]


def test_req_arc_wmte_7478_malformed_historical_shapes_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7478 does not invent units from malformed old shapes."""

    shard = tmp_path / "events.jsonl"
    shard.write_text(json.dumps({"episode_id": "all"}) + "\n", encoding="utf-8")
    harmless = protocol.reduce_exp7471(
        tmp_path,
        {"seam_event_shards": [42, {"path": "events.jsonl"}], "rows": [42]},
    )
    assert harmless["interval_rows"] == []

    upstream_path = tmp_path / protocol.UPSTREAM_PATH
    upstream_path.parent.mkdir(parents=True)
    upstream_path.write_text(
        json.dumps(
            {
                "status": "complete_null_live_arc_seam_observation",
                "honest_verdict": "complete_null_live_arc_seam_observation",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "arc_observation_complete_score": 1,
                "seam_event_shards": [42],
            }
        ),
        encoding="utf-8",
    )
    checks, _hashes, _upstream = protocol.collect_preconditions(tmp_path)
    assert any(row["passed"] is False for row in checks)

    with pytest.raises(ValueError, match="episode_action_boundaries_missing"):
        protocol.reduce_exp7471(
            tmp_path,
            {"seam_event_shards": [], "rows": [{"episode_id": "missing", "action_rows": []}]},
        )


def test_req_arc_wmte_7478_validation_mutations_and_plan_rejections(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7478 names each invalid terminal and command-plan shape."""

    artifact = protocol.build_artifact_for_test(REPO)
    mutations = []
    changed = deepcopy(artifact)
    del changed["schema"]
    mutations.append((changed, "missing_field:schema"))
    changed = deepcopy(artifact)
    changed["milestone"] = "wrong"
    mutations.append((changed, "identity_mismatch:milestone"))
    changed = deepcopy(artifact)
    changed["invocation_counts"]["forward_calls_attempted"] = 1
    mutations.append((changed, "current_invocation_accounting_invalid"))
    changed = deepcopy(artifact)
    changed["interval_rows"] = []
    changed["reproducibility_checksum"] = protocol.artifact_checksum(changed)
    mutations.append((changed, "independent_reduction_mismatch"))
    changed = deepcopy(artifact)
    changed["arc_schedule_manifest"]["manifest_sha256"] = "forged"
    mutations.append((changed, "independent_reduction_mismatch"))
    changed = deepcopy(artifact)
    changed["validation_receipts"] = []
    changed["reproducibility_checksum"] = protocol.artifact_checksum(changed)
    assert "required_validation_receipts_invalid" in protocol.validate_artifact(
        changed, require_terminal=True
    )
    for candidate, expected in mutations:
        assert expected in protocol.validate_artifact(candidate, require_terminal=False)

    commands = protocol.build_validation_plan(REPO, tmp_path / "valid")
    missing = commands[1:]
    assert any(
        error.startswith("command_count:") or error.startswith("missing_command:")
        for error in protocol.validate_validation_plan(REPO, missing)
    )
    broad = [
        *commands,
        CommandSpec("broad", (".venv/bin/pytest", "tests/python"), "bad"),
    ]
    assert "broad_test_target:broad" in protocol.validate_validation_plan(REPO, broad)

    with pytest.raises(SystemExit):
        protocol.parse_args([])

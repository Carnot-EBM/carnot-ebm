"""Tests for the Exp6761 capacity-controlled procedural memory stream.

Spec refs: REQ-CL-6761, SCENARIO-CL-6761-CHRONOLOGY,
SCENARIO-CL-6761-CAPACITY, SCENARIO-CL-6761-TRANSACTIONS,
SCENARIO-CL-6761-POISON, SCENARIO-CL-6761-ROWS, REQ-REPORT-6761,
SCENARIO-REPORT-6761-ATOMIC, SCENARIO-REPORT-6761-BLOCKED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_6761_procedural_memory_stream as mod
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / mod.EXP6748_RELATIVE_PATH
SPEC = REPO / mod.SPEC_RELATIVE_PATH
REPORT_SPEC = REPO / mod.REPORT_SPEC_RELATIVE_PATH


def _artifact(tmp_path: Path) -> mod.JsonDict:
    return mod.build_artifact(
        root=REPO,
        fixture_path=FIXTURE,
        state_root=tmp_path / "state",
        duration_s=0.25,
        tests_run=mod.DEFAULT_TESTS_RUN,
    )


def test_req_cl_6761_specs_own_stream_and_reporting_contracts() -> None:
    """REQ-CL-6761/REQ-REPORT-6761: both capability specs anchor the work."""

    learning = SPEC.read_text(encoding="utf-8").split("## REQ-CL-6761", 1)[1]
    reporting = REPORT_SPEC.read_text(encoding="utf-8").split("### REQ-REPORT-6761", 1)[1]
    for marker in (
        "SCENARIO-CL-6761-CHRONOLOGY",
        "SCENARIO-CL-6761-CAPACITY",
        "SCENARIO-CL-6761-TRANSACTIONS",
        "SCENARIO-CL-6761-POISON",
        "SCENARIO-CL-6761-ROWS",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.SCRIPT_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "complete_blocked_procedural_stream",
    ):
        assert marker in learning
    assert "SCENARIO-REPORT-6761-ATOMIC" in reporting
    assert "SCENARIO-REPORT-6761-BLOCKED" in reporting
    assert "transaction_receipts" in reporting


def test_scenario_cl_6761_stream_has_six_nonzero_orders() -> None:
    """SCENARIO-CL-6761-ROWS: every frozen order has accept and reject headroom."""

    events = mod.base_events()
    manifest = mod.freeze_stream(events)

    assert isinstance(events, tuple)
    assert manifest["frozen_before_dry_replay"] is True
    assert len(manifest["orders"]) >= 6
    assert len({row["order_hash"] for row in manifest["orders"]}) == len(manifest["orders"])
    assert all(len(row["event_ids"]) == len(events) for row in manifest["orders"])
    assert sum(row["expected_transaction_class"] == "accept" for row in events) >= 12
    assert sum(row["expected_transaction_class"] == "reject" for row in events) >= 12
    assert {
        "reusable_procedure",
        "naive_distractor",
        "held_family",
        "retention_anchor",
        "contradiction",
        "stale_lesson",
        "duplicate",
        "poison_candidate",
        "provenance_loss",
    } <= {row["event_kind"] for row in events}
    assert all(
        set(event["depends_on"]) < set(order["event_ids"][:position])
        for order in manifest["orders"]
        for position, event_id in enumerate(order["event_ids"])
        for event in events
        if event["event_id"] == event_id and event["depends_on"]
    )


def test_scenario_cl_6761_representation_pairs_are_equal_and_answer_free() -> None:
    """SCENARIO-CL-6761-CAPACITY: paired values share evidence and fixed slots."""

    pairs = [
        mod.representation_pair(event)
        for event in mod.base_events()
        if event["eligibility"] == "transaction_candidate"
    ]

    assert pairs
    for pair in pairs:
        detailed = pair["representations"]["detailed_trajectory"]
        procedural = pair["representations"]["procedural_lesson"]
        assert detailed["evidence_hash"] == procedural["evidence_hash"] == pair["evidence_hash"]
        assert detailed["allocated_bytes"] == procedural["allocated_bytes"]
        assert detailed["content_bytes"] <= detailed["allocated_bytes"]
        assert procedural["content_bytes"] <= procedural["allocated_bytes"]
        assert pair["answer_content_present"] is False
        assert set(procedural["payload"]) == {
            "abstract_constraint",
            "applicability_scope",
            "repair_procedure",
        }
        serialized = json.dumps(pair["representations"], sort_keys=True).lower()
        assert "current_answer" not in serialized
        assert "future_answer" not in serialized
        assert "target_answer" not in serialized


def test_scenarios_cl_6761_rows_chronology_capacity_and_counts(tmp_path: Path) -> None:
    """SCENARIO-CL-6761-CHRONOLOGY/CAPACITY/ROWS: rows close every stream gate."""

    artifact = _artifact(tmp_path)
    rows = artifact["rows"]
    manifest = artifact["stream_manifest"]

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["procedural_memory_stream_ready"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["order_count"] == len(manifest["orders"]) >= 6
    assert len(rows) == artifact["order_count"] * len(manifest["events"])
    assert len({row["row_key"] for row in rows}) == len(rows)
    assert (
        artifact["eligible_accepts_by_order"]
        == mod.derive_row_counts(rows)["eligible_accepts_by_order"]
    )
    assert (
        artifact["eligible_rejects_by_order"]
        == mod.derive_row_counts(rows)["eligible_rejects_by_order"]
    )
    assert artifact["hard_cases_by_order"] == mod.derive_row_counts(rows)["hard_cases_by_order"]
    assert min(artifact["eligible_accepts_by_order"].values()) >= 12
    assert min(artifact["eligible_rejects_by_order"].values()) >= 12
    assert artifact["future_evidence_violations"] == 0
    assert artifact["read_only_episode_enforced"] is True
    for order in manifest["orders"]:
        order_rows = sorted(
            (row for row in rows if row["order_id"] == order["order_id"]),
            key=lambda row: row["chronology"]["position"],
        )
        for position, row in enumerate(order_rows):
            assert row["chronology"]["visible_event_ids"] == order["event_ids"][:position]
            assert row["chronology"]["current_evidence_visible"] is False
            assert row["chronology"]["future_evidence_visible"] is False
    capacity = artifact["capacity_contract"]
    assert capacity["arms"]["detailed_trajectory"] == capacity["arms"]["procedural_lesson"]
    assert all(
        used < capacity["storage_ceiling_bytes"]
        for used in capacity["max_committed_bytes_by_arm"].values()
    )
    assert mod.validate_artifact(artifact) == []


def test_scenario_cl_6761_read_only_transaction_restart_and_rollback(tmp_path: Path) -> None:
    """SCENARIO-CL-6761-TRANSACTIONS: writes wait, restart, and reverse exactly."""

    event = next(row for row in mod.base_events() if row["expected_transaction_class"] == "accept")
    pair = mod.representation_pair(event)
    journal = mod.AtomicRepresentationJournal(
        tmp_path / "journal",
        "detailed_trajectory",
        mod.CAPACITY_CONTRACT,
    )
    parent = journal.state_bytes()
    snapshot = journal.begin_episode(event["event_id"], [])

    with pytest.raises(mod.ReadOnlyEpisodeError, match="active episode is read-only"):
        journal.transact(event, pair["representations"]["detailed_trajectory"], 1, "test")

    assert journal.state_bytes() == parent == snapshot["state_bytes"]
    journal.end_episode()
    receipt = journal.transact(
        event,
        pair["representations"]["detailed_trajectory"],
        1,
        "test",
    )
    assert receipt["transaction_class"] == "accept"
    assert receipt["committed"] is True
    assert set(mod.TRANSACTION_REQUIRED_FIELDS) <= set(receipt)
    assert all(receipt["atomic_write"].values())
    assert all(
        receipt["atomic_restart_receipt"][key] is True for key in ("bytes_match", "hash_match")
    )
    rollback = journal.rollback(receipt)
    assert rollback["byte_identical"] is True
    assert rollback["restored_hash"] == receipt["parent_hash"]


def test_scenario_cl_6761_poison_rejects_for_preregistered_reasons(tmp_path: Path) -> None:
    """SCENARIO-CL-6761-POISON: unsafe candidates reject for their intended reason."""

    artifact = _artifact(tmp_path)
    receipts = artifact["poison_fixture_receipts"]

    assert receipts
    assert {row["admission_reason"] for row in receipts} >= {
        "reject_poison_exact_authority",
        "reject_contradiction",
        "reject_duplicate",
        "reject_stale",
        "reject_evidence_mismatch",
        "reject_provenance_loss",
    }
    assert all(row["transaction_class"] == "reject" for row in receipts)
    assert all(row["committed"] is False for row in receipts)
    assert all(row["admission_reason"] == row["intended_admission_reason"] for row in receipts)
    assert all(row["state_hash"] == row["parent_hash"] for row in receipts)
    assert all(row["atomic_restart_receipt"]["bytes_match"] is True for row in receipts)


def test_scenario_report_6761_blocked_preconditions_are_complete(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6761-BLOCKED: missing fixture mechanics cannot invent rows."""

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture["transaction_memory_ready"] = False
    blocked_fixture = tmp_path / "blocked-fixture.json"
    blocked_fixture.write_text(json.dumps(fixture), encoding="utf-8")
    artifact = mod.build_artifact(
        root=REPO,
        fixture_path=blocked_fixture,
        state_root=tmp_path / "blocked-state",
        duration_s=0.1,
        precondition_overrides={"rollback_helper_available": False},
        tests_run=mod.DEFAULT_TESTS_RUN,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_blocked_procedural_stream"
    assert artifact["honest_verdict"].startswith("complete_blocked_procedural_stream:")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["procedural_memory_stream_ready"] is False
    assert artifact["rows"] == []
    assert artifact["transaction_receipts"] == []
    assert artifact["gate_check_summary"]["failed_checks"] == [
        "exp6748_transaction_fixture_ready",
        "rollback_helper_available",
    ]
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_6761_validation_atomic_cli_and_adversarial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6761-ATOMIC: validation and the wrapper publish one object."""

    result = tmp_path / "artifact.json"
    state = tmp_path / "cli-state"
    assert (
        mod.main(
            [
                "--date",
                mod.RUN_DATE,
                "--fixture-path",
                str(FIXTURE),
                "--result-path",
                str(result),
                "--state-root",
                str(state),
            ]
        )
        == 0
    )
    assert mod.main(["--validate", "--result-path", str(result)]) == 0
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["reproducibility_checksum"] == mod.reproducibility_checksum(payload)
    report = adversarial_verify.verify_artifact(result, declared=False)
    assert not any(row["severity"] == "critical" for row in report["flags"])
    floor = adversarial_verify.duration_floor_for_artifact(payload)
    assert floor is not None and floor["reason"] == "deterministic_verifier"

    for mutator, expected in (
        (lambda value: value.pop("rows"), "required field set mismatch"),
        (
            lambda value: value.update(inference_substrate="live_llm_inference"),
            "inference_substrate mismatch",
        ),
        (
            lambda value: value.update(verdict_class="unexpected"),
            "verdict_class outside closed enum",
        ),
        (lambda value: value.update(field_principles={}), "field_principles coverage mismatch"),
        (
            lambda value: value.update(reproducibility_checksum="sha256:bad"),
            "reproducibility_checksum mismatch",
        ),
        (
            lambda value: value.update(eligible_accepts_by_order={}),
            "row-derived counts mismatch",
        ),
        (
            lambda value: value.update(future_evidence_violations=1),
            "readiness gates mismatch",
        ),
        (lambda value: value.update(verifier_is_oracle=True), "verifier_is_oracle must be false"),
    ):
        broken = deepcopy(payload)
        mutator(broken)
        if "reproducibility_checksum" in broken and expected != "reproducibility_checksum mismatch":
            broken["reproducibility_checksum"] = mod.reproducibility_checksum(broken)
        assert expected in mod.validate_artifact(broken)

    broken = deepcopy(payload)
    broken["future_evidence_violations"] = 1
    broken["reproducibility_checksum"] = mod.reproducibility_checksum(broken)
    with pytest.raises(ValueError, match="readiness gates mismatch"):
        mod.write_artifact(tmp_path / "invalid.json", broken)

    invalid_path = tmp_path / "invalid-existing.json"
    invalid_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="readiness gates mismatch"):
        mod.main(["--validate", "--result-path", str(invalid_path)])

    wrapper_result = tmp_path / "wrapper.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(REPO / mod.SCRIPT_RELATIVE_PATH),
            "--date",
            mod.RUN_DATE,
            "--fixture-path",
            str(FIXTURE),
            "--result-path",
            str(wrapper_result),
            "--state-root",
            str(tmp_path / "wrapper-state"),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(REPO / mod.SCRIPT_RELATIVE_PATH), run_name="__main__")
    assert exit_info.value.code == 0
    assert wrapper_result.is_file()


def test_req_cl_6761_default_state_and_atomic_failure_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6761: owned temporary state and atomic failures remain closed."""

    artifact = mod.build_artifact(
        root=REPO,
        fixture_path=FIXTURE,
        duration_s=0.2,
        tests_run=mod.DEFAULT_TESTS_RUN,
    )
    assert artifact["procedural_memory_stream_ready"] is True

    original_replace = mod.os.replace

    def fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("injected replace failure")

    monkeypatch.setattr(mod.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        mod.atomic_write(tmp_path / "replace-failure.json", b"data\n")
    assert list(tmp_path.glob(".replace-failure.json.*.tmp")) == []
    monkeypatch.setattr(mod.os, "replace", original_replace)

    monkeypatch.setattr(mod, "validate_artifact", lambda artifact: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        mod.build_artifact(
            root=REPO,
            fixture_path=FIXTURE,
            state_root=tmp_path / "forced",
            duration_s=0.2,
            tests_run=mod.DEFAULT_TESTS_RUN,
        )


def test_req_cl_6761_defensive_transaction_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6761: malformed, oversized, saturated, and invalid operations fail closed."""

    event = next(row for row in mod.base_events() if row["expected_transaction_class"] == "accept")
    with monkeypatch.context() as context:
        context.setattr(mod, "RECORD_SLOT_BYTES", 1)
        with pytest.raises(ValueError, match="exceeds its fixed record slot"):
            mod.representation_pair(event)

    with pytest.raises(ValueError, match="unsupported representation type"):
        mod.AtomicRepresentationJournal(tmp_path / "unsupported", "other", mod.CAPACITY_CONTRACT)

    malformed = mod.AtomicRepresentationJournal(
        tmp_path / "malformed",
        "detailed_trajectory",
        mod.CAPACITY_CONTRACT,
    )
    malformed.state_path.write_text('{"schema":"wrong"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid representation journal state"):
        mod.AtomicRepresentationJournal(
            tmp_path / "malformed",
            "detailed_trajectory",
            mod.CAPACITY_CONTRACT,
        )

    tiny_capacity = deepcopy(mod.CAPACITY_CONTRACT)
    tiny_capacity["storage_ceiling_bytes"] = mod.RECORD_SLOT_BYTES
    saturated = mod.AtomicRepresentationJournal(
        tmp_path / "saturated",
        "detailed_trajectory",
        tiny_capacity,
    )
    pair = mod.representation_pair(event)
    with pytest.raises(ValueError, match="reached the frozen storage ceiling"):
        saturated.transact(
            event,
            pair["representations"]["detailed_trajectory"],
            1,
            "saturated",
        )

    poison = next(
        row
        for row in mod.base_events()
        if row["intended_admission_reason"] == "reject_poison_exact_authority"
    )
    poison_pair = mod.representation_pair(poison)
    rejected = saturated.transact(
        poison,
        poison_pair["representations"]["detailed_trajectory"],
        2,
        "rejected",
    )
    with pytest.raises(ValueError, match="only committed transactions can roll back"):
        saturated.rollback(rejected)

    invalid_fixture = tmp_path / "invalid-fixture.json"
    invalid_fixture.write_text("{", encoding="utf-8")
    checks = mod.check_preconditions(
        root=REPO,
        fixture_path=invalid_fixture,
        state_root=tmp_path / "invalid-preconditions",
    )
    assert checks["all_passed"] is False
    assert checks["checks"]["exp6748_fixture_parses"]["observed"] is False


def test_scenario_cl_6761_chronology_guard_rejects_each_shape(tmp_path: Path) -> None:
    """SCENARIO-CL-6761-CHRONOLOGY: every prefix field is load-bearing."""

    artifact = _artifact(tmp_path)
    mutations = (
        lambda value: value["rows"].pop(),
        lambda value: value["rows"][0]["chronology"].update(position=99),
        lambda value: value["rows"][1]["chronology"].update(visible_event_ids=[]),
        lambda value: value["rows"][0]["chronology"].update(current_evidence_visible=True),
        lambda value: value["rows"][0]["chronology"].update(future_evidence_visible=True),
    )
    for mutate in mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        broken["reproducibility_checksum"] = mod.reproducibility_checksum(broken)
        assert "row chronology mismatch" in mod.validate_artifact(broken)

    blocked_with_rows = deepcopy(artifact)
    blocked_with_rows["verdict_class"] = "blocked"
    blocked_with_rows["reproducibility_checksum"] = mod.reproducibility_checksum(blocked_with_rows)
    assert "blocked artifact must not contain stream rows" in mod.validate_artifact(
        blocked_with_rows
    )


def test_scenario_cl_6761_read_only_guard_failure_is_observable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-6761-TRANSACTIONS: a silent active write closes the gate."""

    manifest = mod.freeze_stream(mod.base_events())
    manifest["orders"] = manifest["orders"][:1]
    original = mod.AtomicRepresentationJournal.transact
    allowed_once = False

    def allow_one_active_write(
        self: mod.AtomicRepresentationJournal,
        event: mod.Mapping[str, object],
        representation: mod.Mapping[str, object],
        position: int,
        order_id: str,
    ) -> mod.JsonDict:
        nonlocal allowed_once
        if self._episode is not None and not allowed_once:
            allowed_once = True
            return {}
        return original(self, event, representation, position, order_id)

    monkeypatch.setattr(mod.AtomicRepresentationJournal, "transact", allow_one_active_write)
    replay = mod.dry_replay(manifest, tmp_path / "silent-write")

    assert allowed_once is True
    assert replay["read_only_checks"][0] is False

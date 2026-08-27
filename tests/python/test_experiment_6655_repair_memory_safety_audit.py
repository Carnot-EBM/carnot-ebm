"""Tests for the independent repair-memory safety audit.

Spec refs: REQ-LEARN-6655, REQ-LEARN-6655-PRECONDITIONS,
REQ-LEARN-6655-RECOMPUTE, REQ-LEARN-6655-ATTACKS,
REQ-LEARN-6655-RESTART, REQ-LEARN-6655-ROLLBACK,
REQ-LEARN-6655-CLAIM, REQ-LEARN-6655-ROWS,
REQ-LEARN-6655-ATOMIC, SCENARIO-LEARN-6655-RECOMPUTATION,
SCENARIO-LEARN-6655-POISON-CONFLICT,
SCENARIO-LEARN-6655-ATOMIC-RESTART,
SCENARIO-LEARN-6655-BYTE-ROLLBACK,
SCENARIO-LEARN-6655-CLAIM-DOWNGRADE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6655_repair_memory_safety_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
PASSING_TESTS = [
    {"command": command, "exit_code": 0, "summary": "passed", "gating": True}
    for command in mod.DEFAULT_TEST_COMMANDS
]


@pytest.fixture(scope="module")
def inputs() -> tuple[dict[str, object], dict[str, object]]:
    return mod.read_inputs(REPO)


@pytest.fixture(scope="module")
def replay(inputs: tuple[dict[str, object], dict[str, object]]) -> dict[str, object]:
    fixture, prospective = inputs
    return mod.replay_patch_ledger(fixture, prospective)


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, object]:
    return mod.build_artifact(
        repo_root=REPO,
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        date="20260827",
        duration_s=1.0,
        tests_run=PASSING_TESTS,
        write=write,
    )


def test_req_learn_6655_spec_declares_every_audit_boundary() -> None:
    """REQ-LEARN-6655: OpenSpec owns all audit and claim rules."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-6655") :]
    for marker in (
        "REQ-LEARN-6655-PRECONDITIONS",
        "REQ-LEARN-6655-RECOMPUTE",
        "REQ-LEARN-6655-ATTACKS",
        "REQ-LEARN-6655-RESTART",
        "REQ-LEARN-6655-ROLLBACK",
        "REQ-LEARN-6655-CLAIM",
        "REQ-LEARN-6655-ROWS",
        "REQ-LEARN-6655-ATOMIC",
        "SCENARIO-LEARN-6655-RECOMPUTATION",
        "SCENARIO-LEARN-6655-POISON-CONFLICT",
        "SCENARIO-LEARN-6655-ATOMIC-RESTART",
        "SCENARIO-LEARN-6655-BYTE-ROLLBACK",
        "SCENARIO-LEARN-6655-CLAIM-DOWNGRADE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section


def test_req_6655_preconditions_bind_gate_schema_code_and_no_llm(
    inputs: tuple[dict[str, object], dict[str, object]],
) -> None:
    """REQ-LEARN-6655-PRECONDITIONS: hashes and resources freeze first."""

    fixture, prospective = inputs
    receipt = mod.upstream_gate_receipt(REPO, prospective)
    checks = mod.preconditions_checked(REPO, fixture, prospective)

    assert receipt["field"] == "prospective_memory_comparison_complete"
    assert receipt["value"] is True
    assert receipt["passed"] is True
    assert str(receipt["sha256"]).startswith("sha256:")
    assert checks["preconditions_ready"] is True
    assert checks["resources"] == {
        "llm_calls": 0,
        "model_weights_loaded": False,
        "network_calls": 0,
    }
    assert str(checks["hashes"]["memory_schema_sha256"]).startswith("sha256:")
    assert str(checks["hashes"]["memory_code_sha256"]).startswith("sha256:")


def test_scenario_6655_recomputes_every_event_order_and_arm(
    inputs: tuple[dict[str, object], dict[str, object]],
) -> None:
    """SCENARIO-LEARN-6655-RECOMPUTATION: raw rows own all metrics."""

    _fixture, prospective = inputs
    events = mod.event_recomputation_rows(prospective)
    metrics = mod.independent_recomputation_rows(prospective, events)

    assert len(events) == 324
    assert all(row["row_hash_valid"] for row in events)
    assert all(row["exact_outcome_match"] for row in events)
    assert all(row["regret_match"] for row in events)
    assert len(metrics) == 9
    assert all(row["stored_matches_rebuilt"] for row in metrics)
    assert {(row["order_id"], row["arm"]) for row in metrics} == {
        (order, arm) for order in mod.ORDER_IDS for arm in mod.ARMS
    }
    deltas = mod.order_deltas(metrics)
    assert [row["delta"] for row in deltas] == pytest.approx([1 / 36, 3 / 36, 2 / 36])


def test_scenario_6655_replays_patch_ledger_from_empty_state(
    replay: dict[str, object],
) -> None:
    """REQ-LEARN-6655-RECOMPUTE: each stored patch receipt replays exactly."""

    rows = replay["patch_rows"]
    assert len(rows) == 108
    assert all(row["started_from_registered_chain"] for row in rows)
    assert all(row["version_match"] for row in rows)
    assert all(row["checkpoint_match"] for row in rows)
    assert all(row["patch_checksum_match"] for row in rows)
    assert all(row["state_after_match"] for row in rows)
    assert all(row["source_evidence_match"] for row in rows)
    assert replay["all_patch_rows_match"] is True
    for order_id in mod.ORDER_IDS:
        chain = [row for row in rows if row["order_id"] == order_id]
        assert chain[0]["memory_version_before"] == 0
        assert chain[-1]["memory_version_after"] == 36


def test_scenario_6655_poison_conflict_and_integrity_attacks_fail_closed(
    inputs: tuple[dict[str, object], dict[str, object]], replay: dict[str, object]
) -> None:
    """SCENARIO-LEARN-6655-POISON-CONFLICT: unsafe candidates never activate."""

    fixture, prospective = inputs
    rows = mod.build_poison_attack_rows(fixture, prospective, replay)

    assert {row["attack_type"] for row in rows} == set(mod.ATTACK_TYPES)
    assert all(row["accepted"] is False for row in rows)
    assert all(row["failed_closed"] is True for row in rows)
    assert all(row["rejection_reasons"] for row in rows)
    assert all(row["state_unchanged"] is True for row in rows)
    reasons = {reason for row in rows for reason in row["rejection_reasons"]}
    assert {
        "duplicate_event_id",
        "conflicting_exact_witness",
        "unsupported_applicability",
        "support_below_floor",
        "poisoned_reward_validity_conflict",
        "future_label_leakage",
        "stale_memory_version",
        "patch_checksum_mismatch",
    } <= reasons


def test_scenario_6655_atomic_restart_recovers_only_old_or_new(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6655-ATOMIC-RESTART: interruptions expose no mix."""

    old = {"version": 7, "items": {"old": {"support": ["a"]}}, "decisions": ["p7"]}
    new = {"version": 8, "items": {"new": {"support": ["a", "b"]}}, "decisions": ["p7", "p8"]}
    rows = mod.exercise_restart_attacks(tmp_path, old, new)

    assert {row["interruption_point"] for row in rows} == set(mod.INTERRUPTION_POINTS)
    assert all(row["atomicity_result"] == "old_or_new_complete" for row in rows)
    assert all(row["recovered_state"] in {"old", "new"} for row in rows)
    assert all(row["checksum_valid"] for row in rows)
    for row in rows:
        expected = "new" if row["replace_completed"] else "old"
        assert row["recovered_state"] == expected


def test_scenario_6655_rolls_back_each_patch_and_each_reverse_chain(
    replay: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6655-BYTE-ROLLBACK: all inverse paths are byte exact."""

    rows = mod.build_rollback_rows(replay)
    patch_count = len(replay["patch_rows"])

    assert len(rows) == patch_count * 2
    assert {row["rollback_mode"] for row in rows} == {"individual", "reverse_sequence"}
    assert all(row["state_bytes_restored"] for row in rows)
    assert all(row["version_restored"] for row in rows)
    assert all(row["support_restored"] for row in rows)
    assert all(row["decision_restored"] for row in rows)
    assert all(row["byte_exact_restoration"] for row in rows)
    assert all(row["inverse_patch_id"].startswith("inverse:") for row in rows)


def test_req_6655_support_and_anchor_recheck_is_independent(
    replay: dict[str, object],
) -> None:
    """REQ-LEARN-6655-ROLLBACK: support and anchors do not trust stored booleans."""

    recheck = replay["support_and_anchor_recheck"]
    assert recheck["patch_count"] == 108
    assert recheck["support_floor"] == 1.0
    assert recheck["minimum_support_before"] >= 1.0
    assert recheck["minimum_support_after"] >= 1.0
    assert recheck["support_failure_count"] == 0
    assert recheck["anchor_regression_count"] == 0
    assert recheck["all_support_and_anchor_checks_pass"] is True


def test_scenario_6655_claim_downgrades_without_manufacturing_a_win() -> None:
    """SCENARIO-LEARN-6655-CLAIM-DOWNGRADE: uncertainty and safety own class."""

    narrowed = mod.decide_claim(
        future_delta=2 / 3,
        order_delta_interval=(-0.01, 0.12),
        safety_ok=True,
    )
    preserved = mod.decide_claim(
        future_delta=0.2,
        order_delta_interval=(0.01, 0.3),
        safety_ok=True,
    )
    nullified = mod.decide_claim(
        future_delta=0.0,
        order_delta_interval=(-0.1, 0.1),
        safety_ok=True,
    )
    blocked = mod.decide_claim(
        future_delta=0.2,
        order_delta_interval=(0.01, 0.3),
        safety_ok=False,
    )

    assert narrowed["claim_disposition"] == "narrow"
    assert narrowed["verdict_class"] == "null"
    assert preserved["claim_disposition"] == "preserve"
    assert preserved["verdict_class"] == "positive"
    assert nullified["claim_disposition"] == "nullify"
    assert nullified["verdict_class"] == "null"
    assert blocked["claim_disposition"] == "block"
    assert blocked["verdict_class"] == "blocked"


def test_req_6655_artifact_is_complete_atomic_and_honestly_null(tmp_path: Path) -> None:
    """REQ-LEARN-6655-ATOMIC: one durable artifact carries all audit evidence."""

    artifact = _artifact(tmp_path, write=True)
    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    written = mod.read_json(output)

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["verdict_class"] == "null"
    assert artifact["claim_disposition"] == "narrow"
    assert artifact["retirement_recommendation"]["retire"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["aggregate_row_recomputation"]["all_audit_units_pass"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert set(artifact) <= set(artifact["field_provenance"])
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_req_6655_validator_rejects_tampering(tmp_path: Path) -> None:
    """REQ-LEARN-6655-ROWS: schema, rows, safety, and hashes fail closed."""

    artifact = _artifact(tmp_path)
    mutations = (
        ("missing_required_fields", lambda value: value.pop("status")),
        ("verdict_class_invalid", lambda value: value.update(verdict_class="maybe")),
        ("claim_disposition_invalid", lambda value: value.update(claim_disposition="promote")),
        (
            "upstream_gate_mismatch",
            lambda value: value["upstream_gate_receipt"].update(passed=False),
        ),
        (
            "recomputation_failure",
            lambda value: value["independent_recomputation_rows"][0].update(
                stored_matches_rebuilt=False
            ),
        ),
        (
            "attack_failure",
            lambda value: value["poison_attack_rows"][0].update(failed_closed=False),
        ),
        (
            "restart_failure",
            lambda value: value["restart_rows"][0].update(atomicity_result="partial"),
        ),
        (
            "rollback_failure",
            lambda value: value["rollback_rows"][0].update(byte_exact_restoration=False),
        ),
        (
            "aggregate_failure",
            lambda value: value["aggregate_row_recomputation"].update(all_audit_units_pass=False),
        ),
        (
            "protected_files_changed",
            lambda value: value["protected_files_unchanged"].update(unchanged=False),
        ),
        ("substrate_mismatch", lambda value: value.update(inference_substrate="llm")),
        ("oracle_mismatch", lambda value: value.update(verifier_is_oracle=True)),
        ("test_failure", lambda value: value["tests_run"][0].update(exit_code=1)),
        ("field_provenance_missing", lambda value: value["field_provenance"].pop("status")),
        (
            "per_unit_hash_mismatch",
            lambda value: value["per_unit_rows"][0].update(unit_sha256="sha256:bad"),
        ),
        ("checksum_mismatch", lambda value: value.update(reproducibility_checksum="sha256:bad")),
    )
    for expected, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        if expected != "checksum_mismatch":
            changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert expected in mod.validate_artifact(changed)


def test_req_6655_helpers_and_cli_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6655: malformed input, disk corruption, and CLI errors close."""

    assert mod.sha256_file(tmp_path / "missing") is None
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod.read_json(invalid)

    state_path = tmp_path / "state.json"
    mod.atomic_commit_state(state_path, {"version": 1})
    envelope = json.loads(state_path.read_text(encoding="utf-8"))
    envelope["state_checksum"] = "sha256:bad"
    state_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="state_checksum_mismatch"):
        mod.load_committed_state(state_path)

    missing_state = tmp_path / "missing-state.json"
    missing_state.write_text(json.dumps({"state_checksum": "sha256:bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="committed_state_missing"):
        mod.load_committed_state(missing_state)

    with pytest.raises(ValueError, match="unknown_interruption_point"):
        mod.atomic_commit_state(tmp_path / "unknown.json", {}, interrupt_at="unknown")

    assert mod._fsync_directory(tmp_path / "does-not-exist") is False

    failed_replace = tmp_path / "failed-replace.json"
    failed_replace.mkdir()
    (failed_replace / "keep").write_text("occupied", encoding="utf-8")
    with pytest.raises(OSError):
        mod.atomic_write_json(failed_replace, {"complete": True})
    assert not list(tmp_path.glob(f".{failed_replace.name}.*.tmp"))

    assert mod._number_equal("same", "same") is True
    assert mod._order_delta_interval([0.25]) == (0.25, 0.25)
    assert mod.build_rollback_rows({"snapshots": []}) == []

    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "validate_artifact", lambda _artifact: ["forced_error"])
        with pytest.raises(ValueError, match="forced_error"):
            mod.build_artifact(
                repo_root=REPO,
                output_path=tmp_path / "forced.json",
                date="20260827",
                duration_s=1.0,
                tests_run=PASSING_TESTS,
            )

    output = tmp_path / "result.json"
    assert mod.main(["--date", "20260827", "--output", str(output), "--duration-s", "1.0"]) == 0
    assert str(output) in capsys.readouterr().out
    measured_output = tmp_path / "measured-result.json"
    assert mod.main(["--date", "20260827", "--output", str(measured_output)]) == 0
    assert mod.read_json(measured_output)["duration_s"] >= 0.001
    assert mod.main(["--validate", "--output", str(output)]) == 0
    assert mod.main(["--check-rows", "--output", str(output)]) == 0

    broken = mod.read_json(output)
    broken["aggregate_row_recomputation"]["all_audit_units_pass"] = False
    broken["reproducibility_checksum"] = mod.reproducibility_checksum(broken)
    output.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate_failure"):
        mod.main(["--check-rows", "--output", str(output)])

    checksum_broken = _artifact(tmp_path)
    checksum_broken["reproducibility_checksum"] = "sha256:bad"
    output.write_text(json.dumps(checksum_broken), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum_mismatch"):
        mod.main(["--validate", "--output", str(output)])


def test_e2e_6655_artifact_replay_attack_restart_and_rollback(tmp_path: Path) -> None:
    """REQ-LEARN-6655: E2E rebuilds, attacks, restarts, writes, and reloads."""

    artifact = _artifact(tmp_path, write=True)
    reloaded = mod.read_json(tmp_path / mod.RESULT_RELATIVE_PATH.name)

    assert reloaded == artifact
    assert all(row["stored_matches_rebuilt"] for row in reloaded["independent_recomputation_rows"])
    assert all(row["failed_closed"] for row in reloaded["poison_attack_rows"])
    assert all(row["atomicity_result"] == "old_or_new_complete" for row in reloaded["restart_rows"])
    assert all(row["byte_exact_restoration"] for row in reloaded["rollback_rows"])
    assert mod.validate_artifact(reloaded) == []

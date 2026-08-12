"""Tests for Exp6342 anytime e-value release ledger.

Spec refs: REQ-LEARN-6342, REQ-LEARN-6342-LEDGER,
REQ-LEARN-6342-VALIDITY, REQ-LEARN-6342-POWER,
REQ-LEARN-6342-ATTACKS, REQ-LEARN-6342-GUARD,
REQ-LEARN-6342-PROVENANCE, SCENARIO-LEARN-6342-OPTIONAL-STOPPING,
SCENARIO-LEARN-6342-REPLAY, SCENARIO-LEARN-6342-ATTACKS,
SCENARIO-LEARN-6342-EXACT-GUARD.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6342_anytime_evalue_release_ledger as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, object]:
    return mod.run(
        date="20260812",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=write,
    )


def _refresh(artifact: dict[str, object]) -> dict[str, object]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def _read_json(receipt: dict[str, object]) -> dict[str, object]:
    return json.loads(Path(str(receipt["path"])).read_text(encoding="utf-8"))


def test_req_learn_6342_spec_declares_contract_and_principles() -> None:
    """REQ-LEARN-6342-PROVENANCE: OpenSpec owns fields and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6342") :]

    for token in (
        "REQ-LEARN-6342-LEDGER",
        "REQ-LEARN-6342-VALIDITY",
        "REQ-LEARN-6342-POWER",
        "REQ-LEARN-6342-ATTACKS",
        "REQ-LEARN-6342-GUARD",
        "SCENARIO-LEARN-6342-OPTIONAL-STOPPING",
        "SCENARIO-LEARN-6342-REPLAY",
        "SCENARIO-LEARN-6342-ATTACKS",
        "SCENARIO-LEARN-6342-EXACT-GUARD",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6342_ledger_replay_is_byte_identical(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6342-REPLAY: ledger rows replay to the same state."""

    artifact = _artifact(tmp_path)
    ledger_receipt = artifact["evalue_ledger_path_and_hash"]
    schema = _read_json(artifact["ledger_schema_path_and_hash"])
    manifest = _read_json(artifact["synthetic_stream_manifest_path_and_hash"])
    rows = mod.read_jsonl(Path(str(ledger_receipt["path"])))
    replay = mod.replay_ledger_rows(rows, expected_predecision_hash=mod.PREDECISION_HASH)

    assert schema["schema"] == mod.LEDGER_ROW_SCHEMA
    assert manifest["null_stream_count"] == mod.NULL_STREAM_COUNT
    assert ledger_receipt["sha256"] == mod.sha256_file(Path(str(ledger_receipt["path"])))
    assert ledger_receipt["row_count"] == len(rows)
    assert replay["state_hash"] == artifact["restart_reconstruction_results"]["state_hash"]
    assert replay["ledger_hash"] == artifact["restart_reconstruction_results"]["ledger_hash"]
    assert artifact["restart_reconstruction_results"]["byte_identical"] is True
    assert artifact["restart_reconstruction_results"]["release_count"] >= 1
    assert all(row["evalue_increment"] >= 0.0 for row in rows)
    assert rows[0]["previous_row_hash"] == mod.GENESIS_ROW_HASH
    assert all(
        rows[index]["previous_row_hash"] == rows[index - 1]["row_hash"]
        for index in range(1, len(rows))
    )
    assert any(row["release_decision"] == "released" for row in rows)


def test_scenario_learn_6342_null_optional_stopping_and_power(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6342-OPTIONAL-STOPPING: null peeking stays bounded."""

    artifact = _artifact(tmp_path)
    nulls = artifact["null_stream_results"]
    alternatives = artifact["alternative_stream_results"]
    optional = artifact["optional_stopping_results"]
    repeated = artifact["repeated_look_results"]
    type_i = artifact["type_i_error_interval_and_sample_size"]
    power = artifact["power_interval_and_sample_size"]
    delays = artifact["release_delay_distribution"]

    assert nulls["empirical_type_i_error"] <= mod.PREREGISTERED_TYPE_I_BOUND
    assert type_i["upper"] <= mod.PREREGISTERED_TYPE_I_BOUND
    assert type_i["n"] == mod.NULL_STREAM_COUNT
    assert optional["stopped_on_first_crossing"] is True
    assert optional["empirical_type_i_error"] == nulls["empirical_type_i_error"]
    assert repeated["look_count_per_stream"] == mod.LOOKS_PER_STREAM
    assert repeated["optional_crossing_count"] >= repeated["fixed_terminal_crossing_count"]
    assert alternatives["empirical_power"] >= mod.PREREGISTERED_POWER_LOWER_BOUND
    assert power["lower"] >= mod.PREREGISTERED_POWER_LOWER_BOUND
    assert power["n"] == mod.ALTERNATIVE_STREAM_COUNT
    assert delays["released_stream_count"] == alternatives["release_count"]
    assert delays["median_look"] <= mod.LOOKS_PER_STREAM


def test_scenario_learn_6342_attacks_and_exact_guard_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-6342-ATTACKS: evidence abuse and unsafe release fail closed."""

    artifact = _artifact(tmp_path)
    grouped = artifact["duplicate_cross_factor_reorder_and_selection_attack_results"]
    tamper = artifact["append_only_tamper_results"]
    guard = artifact["exact_safety_guard_contract"]

    assert grouped["all_attacks_fail_closed"] is True
    assert grouped["released_attack_count"] == 0
    for name in (
        "duplicate_evidence",
        "cross_factor_reuse",
        "reordered_event",
        "selected_hypothesis_after_outcome",
    ):
        assert grouped[name]["fail_closed"] is True
        assert grouped[name]["released"] is False
    assert tamper["all_tamper_attacks_detected"] is True
    assert tamper["truncation"]["detected"] is True
    assert tamper["row_mutation"]["detected"] is True
    assert tamper["previous_hash_break"]["detected"] is True
    assert tamper["evalue_reset_attack"]["detected"] is True
    assert guard["statistical_release_requires_exact_guard"] is True
    assert guard["unsafe_threshold_crossing_release_decision"] == "blocked_by_exact_guard"

    ledger = mod.EValueLedger()
    for look in range(12):
        row = ledger.append(mod.build_event("alternative", 900, look, outcome=1, safe=False))
    assert row["crossed_threshold"] is True
    assert row["release_decision"] == "blocked_by_exact_guard"
    assert ledger.release_count == 0


def test_req_learn_6342_cli_schema_checksum_and_validation(tmp_path: Path) -> None:
    """REQ-LEARN-6342-PROVENANCE: CLI writes a valid terminal artifact."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260812", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["generated_label_count"] == 0
    assert type(artifact["generated_label_count"]) is int
    assert artifact["llm_call_count"] == 0
    assert type(artifact["llm_call_count"]) is int
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["anytime_release_certificate_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is None

    missing = dict(artifact)
    missing.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing)

    bad_zero = json.loads(json.dumps(artifact))
    bad_zero["generated_label_count"] = True
    _refresh(bad_zero)
    with pytest.raises(ValueError, match="generated_label_count"):
        mod.validate_artifact(bad_zero)

    failed_guard = json.loads(json.dumps(artifact))
    failed_guard["exact_safety_guard_contract"]["unsafe_threshold_crossing_release_decision"] = (
        "released"
    )
    _refresh(failed_guard)
    assert failed_guard["anytime_release_certificate_ready_score"] == 0.0

    bad_status = json.loads(json.dumps(failed_guard))
    bad_status["status"] = "complete_positive"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_learn_6342_eprocess_properties_and_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6342-VALIDITY: helpers enforce e-process invariants."""

    assert mod.betting_increment(1) > 1.0
    assert 0.0 <= mod.betting_increment(0) < 1.0
    assert mod.supermartingale_fixture()["expected_increment_under_null"] <= 1.0
    assert mod.wilson_interval(0, 0) == {"lower": 0.0, "upper": 0.0, "estimate": 0.0}
    assert mod.wilson_interval(0, 10)["lower"] == 0.0
    assert mod.wilson_interval(10, 10)["upper"] == 1.0
    assert mod.release_delay_distribution([])["released_stream_count"] == 0
    assert mod._path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod.sha256_json({"ok": True}).startswith("sha256:")

    with pytest.raises(ValueError, match="forced"):
        mod._require(False, "forced")
    with pytest.raises(ValueError, match="outcome"):
        mod.betting_increment(2)
    with pytest.raises(ValueError, match="unknown_stream_kind"):
        mod.deterministic_outcome("bad", 0, 0)

    ledger = mod.EValueLedger()
    event = mod.build_event("alternative", 901, 0, outcome=1)
    ledger.append(event)
    duplicate = ledger.try_append(event)
    assert duplicate["fail_closed"] is True
    assert duplicate["reason"] == "duplicate_evidence"
    accepted_receipt = mod.EValueLedger().try_append(
        mod.build_event("alternative", 905, 0, outcome=0)
    )
    assert accepted_receipt == {
        "fail_closed": True,
        "reason": "not_released",
        "released": False,
    }
    bad_predecision = ledger.try_append(
        mod.build_event("alternative", 902, 1, outcome=1, predecision_hash="sha256:bad")
    )
    assert bad_predecision["reason"] == "predecision_hash_mismatch"
    factor_mismatch = ledger.try_append(
        mod.build_event(
            "alternative",
            903,
            2,
            outcome=1,
            hypothesis_id="accept_factor_release",
            factor_id="repair_factor",
        )
    )
    assert factor_mismatch["reason"] == "factor_hypothesis_mismatch"
    invalid_outcome = mod.build_event("alternative", 904, 3, outcome=1)
    invalid_outcome["outcome"] = 2
    assert ledger.try_append(invalid_outcome)["reason"] == "outcome"

    clean_rows = mod.build_certificate_ledger().rows
    assert mod._tamper_replay_receipt(clean_rows, "accepted")["detected"] is False
    with monkeypatch.context() as patch:
        patch.setattr(mod, "ALTERNATIVE_STREAM_COUNT", 0)
        assert mod.build_certificate_ledger().release_count == 0

    artifact = _artifact(tmp_path, write=False)
    no_tests = json.loads(json.dumps(artifact))
    no_tests["test_exit_codes"] = {mod.DEFAULT_TEST_COMMANDS[0]: 1}
    _refresh(no_tests)
    assert no_tests["anytime_release_certificate_ready_score"] == 0.0

    for field in (
        "null_stream_results",
        "alternative_stream_results",
        "duplicate_cross_factor_reorder_and_selection_attack_results",
        "restart_reconstruction_results",
        "append_only_tamper_results",
        "test_exit_codes",
        "protected_files_unchanged",
    ):
        malformed = json.loads(json.dumps(artifact))
        malformed[field] = []
        assert mod.ready_score(malformed) == 0.0

    output = tmp_path / "cli-no-validate.json"
    assert mod.main(["--date", "20260812", "--output", str(output)]) == 0
    assert output.exists()

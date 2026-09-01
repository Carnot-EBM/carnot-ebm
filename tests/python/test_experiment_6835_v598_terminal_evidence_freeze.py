"""REQ-CONSTRAINT-6835 terminal evidence freeze tests."""

from __future__ import annotations

import ast
import base64
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6835_v598_terminal_evidence_freeze as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def source_bytes() -> dict[str, bytes]:
    """REQ-CONSTRAINT-6835 freezes only terminal source artifact bytes."""

    return {
        name: (REPO_ROOT / path).read_bytes() for name, path in exp.SOURCE_ARTIFACT_PATHS.items()
    }


@pytest.fixture(scope="module")
def source_payloads(source_bytes: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    """SCENARIO-CONSTRAINT-6835-ACCOUNTING uses real frozen payload shapes."""

    return {name: json.loads(raw) for name, raw in source_bytes.items()}


def _mutated(
    source_bytes: dict[str, bytes],
    name: str,
    mutator: Any,
) -> dict[str, bytes]:
    payload = json.loads(source_bytes[name])
    mutator(payload)
    changed = dict(source_bytes)
    changed[name] = json.dumps(payload, sort_keys=True).encode()
    return changed


def _source_row(
    scenario: dict[str, Any], raw: bytes, *, row_id: str = "model|scenario|typed"
) -> dict:
    return {
        "row_id": row_id,
        "model_id": "model",
        "scenario_id": scenario["scenario_id"],
        "arm": "typed",
        "raw_output_bytes_b64": base64.b64encode(raw).decode("ascii"),
        "raw_output_byte_length": len(raw),
        "raw_output_sha256": exp.sha256_bytes(raw),
    }


def test_req_6835_spec_precedes_implementation() -> None:
    """REQ-CONSTRAINT-6835 declares scenarios and required artifact fields."""

    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("## REQ-CONSTRAINT-6835:") :]
    for marker in (
        "SCENARIO-CONSTRAINT-6835-PRECONDITIONS",
        "SCENARIO-CONSTRAINT-6835-FRESH-PARSER",
        "SCENARIO-CONSTRAINT-6835-TAXONOMY",
        "SCENARIO-CONSTRAINT-6835-ACCOUNTING",
        "SCENARIO-CONSTRAINT-6835-NULL-PRESERVATION",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in exp.FIELD_PRINCIPLES


def test_scenario_6835_preconditions_catch_missing_duplicate_and_hash_drift(
    source_bytes: dict[str, bytes],
) -> None:
    """SCENARIO-CONSTRAINT-6835-PRECONDITIONS fails closed on bad sources."""

    clean = exp.evaluate_preconditions(source_bytes)
    assert all(row["passed"] for row in clean), [row for row in clean if not row["passed"]]

    missing = _mutated(source_bytes, "exp6833", lambda data: data["per_unit_rows"].pop())
    missing_check = exp.check_by_name(
        exp.evaluate_preconditions(missing), "exp6833_unique_row_identities"
    )
    assert missing_check["passed"] is False
    assert missing_check["observed"]["row_count"] == 899
    assert missing_check["observed"]["missing_count"] == 1

    duplicate = _mutated(
        source_bytes,
        "exp6833",
        lambda data: data["per_unit_rows"].append(deepcopy(data["per_unit_rows"][0])),
    )
    duplicate_check = exp.check_by_name(
        exp.evaluate_preconditions(duplicate), "exp6833_unique_row_identities"
    )
    assert duplicate_check["passed"] is False
    assert duplicate_check["observed"]["row_count"] == 901
    assert duplicate_check["observed"]["duplicate_count"] == 1

    drift = dict(source_bytes)
    drift["exp6831"] = source_bytes["exp6831"] + b"\n"
    drift_check = exp.check_by_name(exp.evaluate_preconditions(drift), "exp6831_file_sha256")
    assert drift_check["passed"] is False
    assert drift_check["observed"].startswith("sha256:")


def test_scenario_6835_parser_is_fresh_and_strict(
    source_payloads: dict[str, dict[str, Any]],
) -> None:
    """SCENARIO-CONSTRAINT-6835-FRESH-PARSER rejects transport repair paths."""

    good = exp.canonical_bytes({"selected_action_ids": ["b", "a"]})
    assert exp.parse_raw_output(good) == {
        "error": None,
        "parsed": True,
        "selected_action_ids": ["a", "b"],
    }
    for raw, error in (
        (b"\xff", "invalid_utf8"),
        (b"not-json", "invalid_json"),
        (b'{"selected_action_ids": ["a"]}', "non_canonical_json"),
        (b"[]", "invalid_response_fields"),
        (b'{"selected_action_ids":1}', "invalid_action_list"),
        (b'{"selected_action_ids":[""]}', "invalid_action_id"),
        (b'{"selected_action_ids":["a","a"]}', "duplicate_action_id"),
    ):
        assert exp.parse_raw_output(raw) == {
            "error": error,
            "parsed": False,
            "selected_action_ids": [],
        }

    tree = ast.parse((REPO_ROOT / exp.MODULE_PATH).read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("experiment_6833" in name or "experiment_6834" in name for name in imports)
    assert source_payloads["exp6834"]["verdict_class"] == "null"


def test_scenario_6835_taxonomy_separates_malformed_omission_contradiction_and_pass(
    source_payloads: dict[str, dict[str, Any]],
) -> None:
    """SCENARIO-CONSTRAINT-6835-TAXONOMY labels each failure axis separately."""

    scenario = source_payloads["exp6832"]["scenarios"][0]
    decoy = next(
        row["action_id"]
        for row in scenario["candidates"]
        if row["authority"] == "untrusted_candidate"
    )

    malformed = exp.classify_source_row(scenario, _source_row(scenario, b"not-json"))[0]
    assert malformed["protocol_failure"] is True
    assert malformed["parse_error"] == "invalid_json"
    assert malformed["failure_class"] == "protocol_failure"

    omitted = exp.classify_source_row(
        scenario,
        _source_row(scenario, exp.canonical_bytes({"selected_action_ids": []})),
    )[0]
    assert omitted["protocol_failure"] is False
    assert omitted["omission"] is True
    assert omitted["contradiction"] is False
    assert omitted["atom_pass"] is False

    contradiction = exp.classify_source_row(
        scenario,
        _source_row(scenario, exp.canonical_bytes({"selected_action_ids": [decoy]})),
    )[0]
    assert contradiction["contradiction"] is True
    assert contradiction["atom_pass"] is False

    passed = exp.classify_source_row(
        scenario,
        _source_row(
            scenario, exp.canonical_bytes({"selected_action_ids": scenario["legal_action_ids"]})
        ),
    )[0]
    assert passed["failure_class"] == "atom_pass"
    assert passed["atom_pass"] is True
    assert passed["joint_pass"] is True


def test_scenario_6835_accounting_and_null_preservation(
    source_bytes: dict[str, bytes],
    source_payloads: dict[str, dict[str, Any]],
) -> None:
    """SCENARIO-CONSTRAINT-6835-NULL-PRESERVATION keeps readiness non-scientific."""

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_bytes=source_bytes,
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact) == set(artifact["field_principles"])
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["v598_evidence_root_ready_score"] == 1
    assert artifact["obligation_failure_taxonomy_complete_score"] == 1
    assert len(artifact["rows"]) == 3780
    assert artifact["failure_taxonomy"]["source_units"]["protocol_failure"] == 835
    assert artifact["observed_target_cells"]["count"] == 395
    assert artifact["missing_target_cells"]["count"] == 18505
    assert artifact["compatible_policy_lower_bound"]["lower_bound"] == "2^18505"
    assert artifact["source_null_preserved"]["preserved"] is True
    assert (
        artifact["source_null_preserved"]["source_honest_verdict"]
        == (source_payloads["exp6834"]["honest_verdict"])
    )
    assert artifact["gate_check_summary"]["passed"] is True
    assert exp.validate_artifact(artifact) == []

    rebuilt = deepcopy(artifact)
    rebuilt["duration_s"] = 999.0
    assert exp.reproducibility_checksum(rebuilt) == artifact["reproducibility_checksum"]


def test_req_6835_blocked_artifact_validator_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_bytes: dict[str, bytes],
) -> None:
    """REQ-CONSTRAINT-6835 writes terminal artifacts and validates invariants."""

    blocked = exp.build_artifact(
        run_date="20260901",
        duration_s=0.1,
        source_bytes={**source_bytes, "exp6834": b"not-json"},
    )
    assert blocked["status"] == "complete_blocked_v598_terminal_evidence_freeze"
    assert blocked["rows"] == []
    assert blocked["verdict_class"] == "blocked"
    assert blocked["v598_evidence_root_ready_score"] == 0
    assert blocked["gate_check_summary"]["failed_check"] is not None
    assert exp.validate_artifact(blocked) == []

    broken = deepcopy(blocked)
    broken["field_principles"].pop("schema")
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "field principles do not cover every top-level field" in exp.validate_artifact(broken)

    assert exp._json_object(exp._read_source(tmp_path / "missing.json"))["read_error"] == (
        "FileNotFoundError"
    )

    output = tmp_path / "experiment_6835.json"
    for name, path in exp.SOURCE_ARTIFACT_PATHS.items():
        monkeypatch.setitem(exp.SOURCE_ARTIFACT_PATHS, name, REPO_ROOT / path)
    assert exp.main(["--date", "20260901", "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["run_date"] == "20260901"
    assert written["source_null_preserved"]["preserved"] is True


def test_req_6835_defensive_branches_and_validator_errors(
    source_bytes: dict[str, bytes],
    source_payloads: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-CONSTRAINT-6835 covers malformed receipts and terminal validators."""

    scenario = source_payloads["exp6832"]["scenarios"][0]
    with pytest.raises(exp.FreezeError, match="missing_check"):
        exp.check_by_name([], "missing")
    assert exp.check_obligation(scenario, [], "missing") == {
        "fields": {field: False for field in exp.OBLIGATION_FIELDS},
        "passed": False,
    }
    assert exp.check_joint(scenario, b"not-json")["parse_error"] == "invalid_json"

    row = _source_row(scenario, exp.canonical_bytes({"selected_action_ids": []}))
    missing = deepcopy(row)
    missing.pop("raw_output_bytes_b64")
    assert exp._decode_receipt(missing)[1] == "missing_raw_output_bytes"
    invalid = deepcopy(row)
    invalid["raw_output_bytes_b64"] = "not-base64"
    assert exp._decode_receipt(invalid)[1] == "invalid_raw_output_base64"
    wrong_length = deepcopy(row)
    wrong_length["raw_output_byte_length"] += 1
    assert exp._decode_receipt(wrong_length)[1] == "raw_output_length_mismatch"
    wrong_hash = deepcopy(row)
    wrong_hash["raw_output_sha256"] = "sha256:" + "0" * 64
    assert exp._decode_receipt(wrong_hash)[1] == "raw_output_hash_mismatch"
    assert exp._raw_receipt_errors([wrong_hash]) == [
        "model|scenario|typed:raw_output_hash_mismatch"
    ]

    receipts = deepcopy(source_payloads)
    receipts["exp6833"].pop("code_receipts")
    assert exp._recorded_source_receipts(receipts)["missing_receipts"] == ["exp6833"]

    unknown = exp.classify_source_row(
        scenario,
        _source_row(scenario, exp.canonical_bytes({"selected_action_ids": ["unknown-action"]})),
    )[0]
    assert unknown["contradiction"] is True
    assert unknown["contradicting_action_ids"] == ["unknown-action"]

    with pytest.raises(exp.FreezeError, match="unknown_scenario"):
        exp.build_rows(source_payloads["exp6832"], {"per_unit_rows": [{"scenario_id": "missing"}]})
    assert (
        exp._failure_class(
            {
                "protocol_failure": False,
                "omission": False,
                "contradiction": False,
                "atom_pass": False,
            }
        )
        == "joint_semantic_failure"
    )

    complete = exp.build_artifact(
        run_date="20260901",
        duration_s=0.1,
        source_bytes=source_bytes,
    )

    def complete_errors(**changes: object) -> set[str]:
        changed = deepcopy(complete)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        return set(exp.validate_artifact(changed))

    missing_field = deepcopy(complete)
    missing_field.pop("rows")
    missing_field["field_principles"].pop("rows")
    missing_field["reproducibility_checksum"] = exp.reproducibility_checksum(missing_field)
    assert "required artifact fields are missing" in exp.validate_artifact(missing_field)
    assert "inference substrate mismatch" in complete_errors(inference_substrate="wrong")
    assert "verifier_is_oracle must be false" in complete_errors(verifier_is_oracle=True)
    assert "verdict class is outside the closed set" in complete_errors(verdict_class="wrong")
    assert "honest verdict lacks a terminal prefix" in complete_errors(honest_verdict="partial")
    checksum = deepcopy(complete)
    checksum["reproducibility_checksum"] = "bad"
    assert "reproducibility checksum mismatch" in exp.validate_artifact(checksum)
    assert "complete artifact status mismatch" in complete_errors(status="wrong")
    assert "complete artifact obligation-row count mismatch" in complete_errors(rows=[])
    assert "observed target-cell count mismatch" in complete_errors(
        observed_target_cells={"count": 0}
    )
    assert "missing target-cell count mismatch" in complete_errors(
        missing_target_cells={"count": 0}
    )
    assert "compatible-policy lower bound mismatch" in complete_errors(
        compatible_policy_lower_bound={"lower_bound": "1"}
    )
    assert "source null was not preserved" in complete_errors(
        source_null_preserved={"preserved": False}
    )
    assert "taxonomy complete score mismatch" in complete_errors(
        obligation_failure_taxonomy_complete_score=0
    )
    assert "evidence root ready score mismatch" in complete_errors(v598_evidence_root_ready_score=0)
    assert "readiness was converted into a positive result" in complete_errors(
        verdict_class="positive"
    )
    assert "complete artifact gate did not pass" in complete_errors(
        gate_check_summary={"passed": False}
    )

    blocked = exp.build_artifact(
        run_date="20260901",
        duration_s=0.1,
        source_bytes={**source_bytes, "exp6834": b"not-json"},
    )

    def blocked_errors(**changes: object) -> set[str]:
        changed = deepcopy(blocked)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        return set(exp.validate_artifact(changed))

    assert "blocked status mismatch" in blocked_errors(status="wrong")
    assert "blocked artifact emitted rows" in blocked_errors(rows=[{}])
    assert "blocked artifact is marked ready" in blocked_errors(v598_evidence_root_ready_score=1)
    assert "blocked artifact lacks failed check" in blocked_errors(gate_check_summary={})

    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["forced failure"])
    assert exp.main(["--date", "20260901", "--output", str(tmp_path / "never.json")]) == 1

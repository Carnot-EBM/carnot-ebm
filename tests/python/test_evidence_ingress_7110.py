"""Tests for REQ-REPORT-7110 and its fail-closed ingress scenarios."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from carnot.reporting.evidence_ingress import ingest_evidence


def _write_artifact(tmp_path: Path, name: str, **updates: object) -> Path:
    """Write one small input in temporary storage, never in the research record."""

    payload: dict[str, object] = {
        "run_date": "20260907",
        "flagged_adversarial": False,
        "honest_verdict": "complete_null_fixture",
        "verdict_class": "null",
        "metric": 7,
    }
    payload.update(updates)
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_scenario_7110_accepts_clean_dated_input_without_mutation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7110-ACCEPT: clean bytes enter with hash and date receipts."""

    path = _write_artifact(tmp_path, "clean.json")
    before = path.read_bytes()

    result = ingest_evidence(
        [{"path": path, "expected_sha256": _sha256(path), "consumer": "exp7115"}]
    )

    assert result["classification_complete"] is True
    assert result["rejected_upstream_rows"] == []
    assert result["excluded_flagged_upstreams"] == []
    accepted = result["accepted_upstreams"][0]
    assert accepted["sha256"] == _sha256(path)
    assert accepted["run_date"] == "20260907"
    assert accepted["normalized_run_date"] == "20260907"
    assert accepted["consumer"] == "exp7115"
    assert accepted["payload"]["metric"] == 7
    assert path.read_bytes() == before


def test_scenario_7110_rejects_artifact_level_flag_and_preserves_verdict(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7110-FLAGS: a stamped input is excluded, not rewritten."""

    path = _write_artifact(tmp_path, "stamped.json", flagged_adversarial=True)
    before = path.read_bytes()

    result = ingest_evidence([{"path": path, "consumer": "exp7120"}])

    rejected = result["rejected_upstream_rows"][0]
    assert rejected["reason_codes"] == ["artifact_flagged_adversarial"]
    assert rejected["honest_verdict"] == "complete_null_fixture"
    assert rejected["verdict_class"] == "null"
    assert result["excluded_flagged_upstreams"] == [rejected]
    assert path.read_bytes() == before


def test_scenario_7110_rejects_sidecar_or_verifier_critical_flag(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7110-FLAGS: external critical evidence also quarantines."""

    path = _write_artifact(tmp_path, "sidecar.json", flagged_adversarial=None)
    report = {
        "loaded": True,
        "flags": [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "fixture",
            }
        ],
    }

    result = ingest_evidence([{"path": path, "consumer": "exp7119"}], verifier=lambda _path: report)

    rejected = result["rejected_upstream_rows"][0]
    assert rejected["reason_codes"] == ["verifier_flagged_adversarial"]
    assert rejected["flag_status"] == "flagged"
    assert rejected["flag_sources"] == ["verifier"]
    assert rejected["consumer"] == "exp7119"
    assert result["accepted_upstreams"] == []


def test_scenario_7110_accepts_clean_verifier_and_ignores_info_only_flag(tmp_path: Path) -> None:
    """REQ-REPORT-7110: informational findings do not impersonate critical flags."""

    path = _write_artifact(tmp_path, "info.json", flagged_adversarial=None)
    report = {
        "loaded": True,
        "flags": [{"kind": "NOTE", "severity": "info", "detail": "fixture"}],
    }

    result = ingest_evidence([{"path": path}], verifier=lambda _path: report)

    assert len(result["accepted_upstreams"]) == 1
    assert result["accepted_upstreams"][0]["flag_status"] == "clean"


def test_scenario_7110_rejects_unknown_flag_status(tmp_path: Path) -> None:
    """REQ-REPORT-7110: missing or failed verification rejects by default."""

    path = _write_artifact(tmp_path, "unknown.json", flagged_adversarial=None)
    missing = ingest_evidence([{"path": path}])
    failed = ingest_evidence(
        [{"path": path}], verifier=lambda _path: {"loaded": False, "flags": []}
    )

    def broken_verifier(_path: Path) -> dict[str, object]:
        raise RuntimeError("fixture verifier unavailable")

    raised = ingest_evidence([{"path": path}], verifier=broken_verifier)

    assert missing["rejected_upstream_rows"][0]["reason_codes"] == ["flag_status_unknown"]
    assert failed["rejected_upstream_rows"][0]["reason_codes"] == ["flag_status_unknown"]
    assert raised["rejected_upstream_rows"][0]["reason_codes"] == ["flag_status_unknown"]


def test_scenario_7110_rejects_missing_run_date(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7110-DATE: absent dates cannot enter V624 aggregation."""

    path = _write_artifact(tmp_path, "missing-date.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["run_date"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = ingest_evidence([{"path": path}])

    rejected = result["rejected_upstream_rows"][0]
    assert rejected["reason_codes"] == ["run_date_missing"]
    assert rejected["run_date"] is None
    assert result["missing_run_date_rows"] == [rejected]


@pytest.mark.parametrize("run_date", ["2026/09/07", "20260230", "2026-13-01", 20260907])
def test_scenario_7110_rejects_malformed_run_date(tmp_path: Path, run_date: object) -> None:
    """SCENARIO-REPORT-7110-DATE: shape and calendar validity both matter."""

    path = _write_artifact(tmp_path, "bad-date.json", run_date=run_date)

    result = ingest_evidence([{"path": path}])

    rejected = result["rejected_upstream_rows"][0]
    assert rejected["reason_codes"] == ["run_date_malformed"]
    assert rejected["normalized_run_date"] is None
    assert result["malformed_run_date_rows"] == [rejected]


def test_scenario_7110_rejects_absent_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7110-INTEGRITY: an absent expected input fails closed."""

    missing = tmp_path / "absent.json"

    result = ingest_evidence([{"path": missing}])

    rejected = result["rejected_upstream_rows"][0]
    assert rejected["reason_codes"] == ["artifact_not_found"]
    assert rejected["sha256"] is None
    assert result["classification_complete"] is True


def test_scenario_7110_rejects_duplicate_expected_path(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7110-INTEGRITY: duplicate requests cannot double count evidence."""

    path = _write_artifact(tmp_path, "duplicate.json")

    result = ingest_evidence([{"path": path}, {"path": path}])

    assert result["accepted_upstreams"] == []
    assert len(result["rejected_upstream_rows"]) == 1
    rejected = result["rejected_upstream_rows"][0]
    assert rejected["reason_codes"] == ["duplicate_expected_artifact"]
    assert rejected["expected_occurrences"] == 2


def test_scenario_7110_rejects_hash_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7110-INTEGRITY: changed bytes do not enter aggregation."""

    path = _write_artifact(tmp_path, "drift.json")

    result = ingest_evidence([{"path": path, "expected_sha256": "sha256:" + "0" * 64}])

    rejected = result["rejected_upstream_rows"][0]
    assert rejected["reason_codes"] == ["hash_drift"]
    assert rejected["sha256"] == _sha256(path)
    assert rejected["expected_sha256"] == "sha256:" + "0" * 64


def test_scenario_7110_rejects_unreadable_json_shapes(tmp_path: Path) -> None:
    """REQ-REPORT-7110: unreadable or non-object JSON never supplies evidence."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")

    malformed_result = ingest_evidence([{"path": malformed}])
    shape_result = ingest_evidence([{"path": non_object}])

    assert malformed_result["rejected_upstream_rows"][0]["reason_codes"] == ["artifact_unreadable"]
    assert shape_result["rejected_upstream_rows"][0]["reason_codes"] == ["artifact_not_object"]


def test_scenario_7110_empty_expected_set_is_valid() -> None:
    """SCENARIO-REPORT-7110-EMPTY: zero expected inputs is a complete clean set."""

    result = ingest_evidence([])

    assert result == {
        "classification_complete": True,
        "accepted_upstreams": [],
        "rejected_upstream_rows": [],
        "excluded_flagged_upstreams": [],
        "missing_run_date_rows": [],
        "malformed_run_date_rows": [],
    }


def test_req_7110_can_disable_date_rule_only_for_historical_audit(tmp_path: Path) -> None:
    """REQ-REPORT-7110: historical facts can be read without inventing a date."""

    path = _write_artifact(tmp_path, "historical.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["run_date"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = ingest_evidence([{"path": path}], require_run_date=False)

    assert len(result["accepted_upstreams"]) == 1
    assert result["accepted_upstreams"][0]["run_date"] is None
    assert result["accepted_upstreams"][0]["normalized_run_date"] is None


def test_req_7110_result_is_independent_of_caller_owned_input(tmp_path: Path) -> None:
    """REQ-REPORT-7110: classification does not mutate a consumer expectation."""

    path = _write_artifact(tmp_path, "caller.json")
    expected = [{"path": path, "metadata": {"branch": "audit"}}]
    before = deepcopy(expected)

    ingest_evidence(expected)

    assert expected == before


def test_req_7110_supports_path_inputs_and_rejects_bad_expectation_shapes(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-7110: the reusable API validates every caller input shape."""

    path = _write_artifact(tmp_path, "path-input.json")

    assert len(ingest_evidence([path])["accepted_upstreams"]) == 1
    with pytest.raises(TypeError, match="path or mapping"):
        ingest_evidence([object()])
    with pytest.raises(TypeError, match="expected_sha256"):
        ingest_evidence([{"path": path, "expected_sha256": 7}])


def test_req_7110_understands_explicit_verifier_status_shapes(tmp_path: Path) -> None:
    """REQ-REPORT-7110: explicit verifier stamps and counts are classified exactly."""

    path = _write_artifact(tmp_path, "external-status.json", flagged_adversarial=None)
    flagged = ingest_evidence([path], verifier=lambda _path: {"flagged_adversarial": True})
    clean_stamp = ingest_evidence([path], verifier=lambda _path: {"flagged_adversarial": False})
    clean_count = ingest_evidence([path], verifier=lambda _path: {"loaded": True, "flag_count": 0})
    invalid_report = ingest_evidence([path], verifier=lambda _path: None)  # type: ignore[return-value]

    assert flagged["rejected_upstream_rows"][0]["reason_codes"] == ["verifier_flagged_adversarial"]
    assert len(clean_stamp["accepted_upstreams"]) == 1
    assert len(clean_count["accepted_upstreams"]) == 1
    assert invalid_report["rejected_upstream_rows"][0]["reason_codes"] == ["flag_status_unknown"]


def test_req_7110_duplicate_receipt_survives_unreadable_and_nonobject_inputs(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-7110: duplicate rejection never depends on readable JSON content."""

    malformed = tmp_path / "duplicate-malformed.json"
    malformed.write_text("{", encoding="utf-8")
    nonobject = tmp_path / "duplicate-list.json"
    nonobject.write_text("[]", encoding="utf-8")

    malformed_result = ingest_evidence([malformed, malformed])
    nonobject_result = ingest_evidence([nonobject, nonobject])

    assert malformed_result["rejected_upstream_rows"][0]["run_date"] is None
    assert nonobject_result["rejected_upstream_rows"][0]["run_date"] is None

"""Artifact tests for REQ-REPORT-7110 and SCENARIO-REPORT-7110-ARTIFACT."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from carnot import experiment_7110_v624_evidence_ingress_quarantine as mod


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def valid_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build the expensive 55-capstone replay once for immutable test reuse."""

    output = tmp_path_factory.mktemp("v624") / "experiment_7110.json"
    return mod.build_artifact(ROOT, "20260907", output_path=output)


def test_scenario_7110_census_and_exp7108_regression_replay(
    valid_artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7110-CENSUS: all 55 rows and Exp7099 classify without writes."""

    artifact = valid_artifact

    assert mod.validate_artifact(artifact) == []
    assert len(artifact["capstone_reference_rows"]) == 55
    assert all(
        row["classification"] in mod.HISTORICAL_REFERENCE_CLASSES
        for row in artifact["capstone_reference_rows"]
    )
    assert artifact["historical_artifacts_rewritten"] is False
    regression = artifact["exp7108_exp7099_regression_row"]
    assert regression["consumer_experiment_id"] == 7108
    assert regression["upstream_experiment_id"] == 7099
    assert regression["ingestion_eligible"] is False
    assert "artifact_flagged_adversarial" in regression["reason_codes"]
    assert regression["scientific_honest_verdict"] == (
        "complete_null_adapter_withheld_live_path_not_ready_no_solve_claim"
    )
    assert artifact["evidence_ingress_quarantine_ready_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_positive")


def test_scenario_7110_artifact_has_principles_hashes_and_fixture_receipts(
    valid_artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7110-ARTIFACT: required fields recompute from evidence."""

    artifact = valid_artifact

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact["field_principles"].keys()
    assert all(artifact["field_principles"][field] for field in mod.REQUIRED_ARTIFACT_FIELDS)
    assert {row["fixture"] for row in artifact["fixture_rows"]} == mod.EXPECTED_FIXTURES
    assert all(row["passed"] is True for row in artifact["fixture_rows"])
    assert artifact["accepted_upstreams"]
    assert artifact["rejected_upstream_rows"]
    assert artifact["missing_run_date_rows"]
    assert artifact["malformed_run_date_rows"]
    assert artifact["source_artifact_hashes"]
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert artifact["gate_check_summary"] == {
        "failed_check": None,
        "expected_value": 1,
        "observed_value": 1,
        "passed": True,
    }


def test_scenario_7110_blocked_precondition_is_schema_complete(tmp_path: Path) -> None:
    """REQ-REPORT-7110: unavailable evidence is blocked and never partial."""

    artifact = mod.build_artifact(
        tmp_path,
        "20260907",
        output_path=tmp_path / "results" / "blocked.json",
    )

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["evidence_ingress_quarantine_ready_score"] == 0
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked")
    assert artifact["gate_check_summary"]["failed_check"] == "results_readable"
    assert artifact["gate_check_summary"]["expected_value"] is True
    assert artifact["gate_check_summary"]["observed_value"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_7110_validator_detects_forged_score_verdict_and_checksum(
    valid_artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7110-ARTIFACT: forged terminal fields fail validation."""

    forged = deepcopy(valid_artifact)
    forged["evidence_ingress_quarantine_ready_score"] = 0
    forged["verdict_class"] = "partial"
    forged["honest_verdict"] = "complete_partial_forged"
    forged["reproducibility_checksum"] = "sha256:forged"

    errors = mod.validate_artifact(forged)

    assert "evidence_ingress_quarantine_ready_score_invalid" in errors
    assert "verdict_class_not_derived" in errors
    assert "honest_verdict_not_derived" in errors
    assert "reproducibility_checksum_invalid" in errors


def test_req_7110_failed_fixture_derives_disqualified_not_partial(
    valid_artifact: dict[str, object],
) -> None:
    """REQ-REPORT-7110: an executed test failure maps to disqualified."""

    artifact = deepcopy(valid_artifact)
    artifact["fixture_rows"][0]["passed"] = False
    artifact["evidence_ingress_quarantine_ready_score"] = 0
    artifact["verdict_class"] = "disqualified"
    artifact["honest_verdict"] = mod.DISQUALIFIED_VERDICT
    artifact["gate_check_summary"] = {
        "failed_check": artifact["fixture_rows"][0]["fixture"],
        "expected_value": True,
        "observed_value": False,
        "passed": False,
    }
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)

    assert mod.validate_artifact(artifact) == []


def test_req_7110_streaming_census_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7110: corrupt and missing historical files stay bounded facts."""

    missing = tmp_path / "missing.json"
    invalid_utf8 = tmp_path / "experiment_11_invalid.json"
    invalid_utf8.write_bytes(b'{\n  "run_date": "\xff"\n}')
    malformed = tmp_path / "experiment_4001_capstone_bad.json"
    malformed.write_text("{", encoding="utf-8")
    nonobject = tmp_path / "list.json"
    nonobject.write_text("[]", encoding="utf-8")
    upstream = tmp_path / "experiment_10_flagged.json"
    upstream.write_text(
        json.dumps(
            {"run_date": "20260907", "flagged_adversarial": True},
            indent=2,
        ),
        encoding="utf-8",
    )
    capstone = tmp_path / "experiment_4002_capstone.json"
    capstone.write_text(
        json.dumps({"experiment_id": 4002, "rows": [{"experiment_id": 10}]}, indent=2),
        encoding="utf-8",
    )

    assert mod._load_object(malformed) is None
    assert mod._load_object(nonobject) is None
    assert mod._stream_search(missing, re.compile(b"anything")) is None
    assert mod._streamed_run_date(invalid_utf8) is None
    assert mod._experiment_id(Path("unrelated.json"), {"experiment_id": "none"}) is None
    census = mod.historical_capstone_reference_census(tmp_path)
    assert len(census) == 1
    assert census[0]["flagged_upstream_experiment_ids"] == [10]


def test_req_7110_precondition_replay_exception_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7110: verifier failure is an unavailable-evidence precondition."""

    def unavailable(_path: Path) -> dict[str, object]:
        raise RuntimeError("fixture verifier unavailable")

    monkeypatch.setattr(mod, "_load_adversarial_report", unavailable)
    checks, report = mod._preconditions(ROOT, tmp_path / "result.json")

    assert report is None
    assert checks[-1] == {
        "check": "adversarial_replay_available",
        "expected_value": True,
        "observed_value": False,
        "available": False,
    }


def test_req_7110_date_and_summary_edge_states_are_exact() -> None:
    """REQ-REPORT-7110: malformed history and each gate failure retain exact values."""

    assert mod._date_state(20260907) == "malformed"
    assert mod._date_state("20260230") == "malformed"
    assert mod._valid_run_date(None) is False
    assert mod._valid_run_date("20260230") is False
    assert (
        mod._expected_summary({"preconditions_checked": None, "fixture_rows": None, "rows": None})[
            "passed"
        ]
        is True
    )
    assert mod._expected_summary(
        {
            "preconditions_checked": [object()],
            "fixture_rows": [{"fixture": "bad", "passed": False}],
            "rows": None,
        }
    ) == {
        "failed_check": "bad",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
    }
    assert mod._expected_summary(
        {
            "preconditions_checked": [],
            "fixture_rows": [],
            "rows": [{"check": "count", "expected_value": 55, "observed_value": 54}],
        }
    ) == {
        "failed_check": "count",
        "expected_value": 55,
        "observed_value": 54,
        "passed": False,
    }


def test_req_7110_build_derives_disqualified_when_executed_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7110: executed readiness failure is disqualified, never partial."""

    census = [
        {
            "capstone_path": f"fixture://capstone/{index}",
            "capstone_run_date": "20260907",
            "flagged_upstream_artifacts": [],
            "classification": "reference_only",
            "byte_hash_unchanged": True,
        }
        for index in range(mod.EXPECTED_HISTORICAL_CENSUS_COUNT)
    ]
    monkeypatch.setattr(mod, "historical_capstone_reference_census", lambda _path: census)
    monkeypatch.setattr(mod, "_ready_from_artifact", lambda _artifact: False)

    artifact = mod.build_artifact(ROOT, "20260907", output_path=tmp_path / "result.json")

    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"] == mod.DISQUALIFIED_VERDICT


def test_req_7110_validator_reports_each_independent_schema_violation(
    valid_artifact: dict[str, object],
) -> None:
    """REQ-REPORT-7110: the saved receipt validator fails closed on schema drift."""

    forged = deepcopy(valid_artifact)
    del forged["rows"]
    forged["field_principles"] = {}
    forged["run_date"] = "20260230"
    forged["execution_venue"] = "unknown"
    forged["verifier_is_oracle"] = True
    forged["historical_artifacts_rewritten"] = True
    forged["evidence_ingress_helper_path"] = "wrong.py"
    forged["inference_substrate_class"] = "wrong"
    forged["verdict_class"] = "disqualified"
    forged["honest_verdict"] = mod.DISQUALIFIED_VERDICT
    forged["gate_check_summary"] = {"passed": True}

    errors = mod.validate_artifact(forged)

    assert {
        "required_artifact_fields_missing",
        "field_principles_incomplete",
        "run_date_invalid",
        "execution_venue_invalid",
        "verifier_is_oracle_invalid",
        "historical_artifacts_rewritten_invalid",
        "evidence_ingress_helper_path_invalid",
        "inference_substrate_class_invalid",
        "gate_check_summary_not_derived",
    } <= set(errors)

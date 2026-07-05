"""Tests for Exp 5248 GAP-4 receipt salvage or retirement.

Spec refs: REQ-REPORT-5248,
SCENARIO-REPORT-5248-SALVAGED-CLEAN-NULL,
SCENARIO-REPORT-5248-BLOCKED-OR-RETIRED.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5248_gap4_receipt_salvage_or_retire_v480 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _source_payloads() -> dict[str, dict[str, Any]]:
    return mod.load_source_payloads(REPO)


def _value(artifact: dict[str, Any], field: str) -> Any:
    return artifact[field]["value"]


def _field_rows(row: dict[str, Any], classification: str | None = None) -> list[dict[str, Any]]:
    rows = list(row["field_classifications"])
    if classification is None:
        return rows
    return [item for item in rows if item["classification"] == classification]


def test_req_report_5248_spec_declares_receipt_decision_contract() -> None:
    """REQ-REPORT-5248: OpenSpec anchors the 5248 receipt-only decision."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5248") : spec.index("### REQ-REPORT-5162")]

    for marker in (
        "REQ-REPORT-5248",
        "SCENARIO-REPORT-5248-SALVAGED-CLEAN-NULL",
        "SCENARIO-REPORT-5248-BLOCKED-OR-RETIRED",
        str(mod.RESULT_RELATIVE_PATH),
        "artifact_normalizer_ready=true",
        "cached_fixture_replay_no_llm",
        "salvaged_clean_null",
        "blocked_missing_receipts",
        "retire_current_gap4_pool",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5248_current_gap4_pool_salvages_clean_null() -> None:
    """SCENARIO-REPORT-5248-SALVAGED-CLEAN-NULL: current frozen pool is salvageable."""

    decision = mod.build_decision(root=REPO)

    assert decision.final_decision == "salvaged_clean_null"
    assert decision.wins == 0
    assert decision.losses == 0
    assert decision.ties == 120
    assert decision.unsafe_missing_receipts == []
    assert decision.normalizer_version == mod.EXP5247_VERSION
    assert decision.normalizer_checksum == mod.EXPECTED_EXP5247_CHECKSUM
    assert {row["path"] for row in decision.normalized_artifacts} == {
        str(path) for path in mod.SOURCE_ARTIFACT_RELATIVE_PATHS
    }

    by_path = {Path(row["path"]).name: row for row in decision.normalized_artifacts}
    for name in by_path:
        checksum = by_path[name]["checksum_receipt"]
        assert checksum["classification"] == "safe-normalized"
        assert checksum["claim_critical"] is True
        assert checksum["pre_qa_checksum_matches_stored"] is True

    exp5235 = by_path["experiment_5235_adversarial_qa_null_tautology_calibration_v479.json"]
    assert {
        row["field"] for row in _field_rows(exp5235, "irrelevant-to-claim")
    } >= {"duration_s", "methodology", "flagged_adversarial", "corrigendum_pending"}

    exp5236 = by_path["experiment_5236_gap4_clean_status_after_qa_calibration_v479.json"]
    assert {
        row["field"] for row in _field_rows(exp5236, "safe-normalized")
    } >= {"remaining_blocker", "reproducibility_checksum"}


def test_req_report_5248_terminal_artifact_schema_and_principles() -> None:
    """REQ-REPORT-5248: required fields are principle wrapped and counts are frozen."""

    artifact = mod.build_artifact(decision=mod.build_decision(root=REPO), tests_run=[])

    mod.validate_artifact(artifact)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _value(artifact, "gap4_final_decision") == "salvaged_clean_null"
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "wins") == 0
    assert _value(artifact, "losses") == 0
    assert _value(artifact, "ties") == 120
    assert _value(artifact, "pool_retired") is False
    assert _value(artifact, "no_new_generation") is True
    assert artifact["normalizer_checksum"]["value"] == mod.EXPECTED_EXP5247_CHECKSUM


def test_scenario_report_5248_claim_critical_missing_count_blocks() -> None:
    """SCENARIO-REPORT-5248-BLOCKED-OR-RETIRED: missing frozen counts fail closed."""

    payloads = _source_payloads()
    payloads[str(mod.EXP5225_RELATIVE_PATH)] = copy.deepcopy(
        payloads[str(mod.EXP5225_RELATIVE_PATH)]
    )
    payloads[str(mod.EXP5225_RELATIVE_PATH)].pop("ties")

    decision = mod.build_decision_from_payloads(payloads, normalizer_artifact=mod.load_exp5247(REPO))
    artifact = mod.build_artifact(decision=decision, tests_run=[])

    mod.validate_artifact(artifact)
    assert decision.final_decision == "blocked_missing_receipts"
    assert _value(artifact, "honest_verdict").startswith("blocked_")
    assert {
        "artifact": str(mod.EXP5225_RELATIVE_PATH),
        "field": "ties",
        "reason": "frozen_count_missing_or_not_int",
    } in _value(artifact, "unsafe_missing_receipts")
    assert _value(artifact, "wins") == 0
    assert _value(artifact, "losses") == 0
    assert _value(artifact, "ties") == 0
    assert _value(artifact, "pool_retired") is False


def test_req_report_5248_normalizer_precondition_blocks_without_salvage() -> None:
    """REQ-REPORT-5248: Exp 5247 readiness is a hard precondition."""

    normalizer = copy.deepcopy(mod.load_exp5247(REPO))
    normalizer["artifact_normalizer_ready"] = False

    decision = mod.build_decision_from_payloads(_source_payloads(), normalizer_artifact=normalizer)

    assert decision.final_decision == "blocked_missing_receipts"
    assert decision.unsafe_missing_receipts == [
        {
            "artifact": str(mod.EXP5247_RELATIVE_PATH),
            "field": "artifact_normalizer_ready",
            "reason": "exp5247_not_ready",
        }
    ]
    assert decision.normalized_artifacts == []


def test_req_report_5248_retire_decision_is_explicit_when_requested() -> None:
    """REQ-REPORT-5248: retirement is explicit and does not mutate counts."""

    decision = mod.build_decision_from_payloads(
        _source_payloads(),
        normalizer_artifact=mod.load_exp5247(REPO),
        retire_on_blocked=True,
        forced_missing_receipts=[
            {
                "artifact": str(mod.EXP5224_RELATIVE_PATH),
                "field": "generation_certificate",
                "reason": "fixture_missing_generation_receipt",
            }
        ],
    )

    assert decision.final_decision == "retire_current_gap4_pool"
    assert decision.wins == 0
    assert decision.losses == 0
    assert decision.ties == 120
    artifact = mod.build_artifact(decision=decision, tests_run=[])
    assert _value(artifact, "pool_retired") is True
    assert _value(artifact, "no_new_generation") is True
    mod.validate_artifact(artifact)


def test_req_report_5248_validation_rejects_schema_breaks() -> None:
    """REQ-REPORT-5248: malformed result receipts fail before write."""

    artifact = mod.build_artifact(decision=mod.build_decision(root=REPO), tests_run=[])

    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)
    with pytest.raises(ValueError, match="principle mismatch"):
        mod.validate_artifact(
            artifact
            | {
                "wins": {
                    "value": 0,
                    "principle": "wrong",
                }
            }
        )
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(artifact | {"field_principles": {}})
    with pytest.raises(ValueError, match="principle-wrapped"):
        mod.validate_artifact(artifact | {"wins": 0})
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(
            artifact
            | {
                "honest_verdict": {
                    "value": "salvaged clean null",
                    "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
                }
            }
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(
            artifact
            | {
                "inference_substrate": {
                    "value": "aggregation_from_upstream_artifacts",
                    "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
                }
            }
        )
    with pytest.raises(ValueError, match="gap4_final_decision"):
        mod.validate_artifact(
            artifact
            | {
                "gap4_final_decision": {
                    "value": "maybe",
                    "principle": mod.FIELD_PRINCIPLES["gap4_final_decision"],
                }
            }
        )
    with pytest.raises(ValueError, match="integer"):
        mod.validate_artifact(
            artifact
            | {
                "ties": {
                    "value": True,
                    "principle": mod.FIELD_PRINCIPLES["ties"],
                }
            }
        )
    with pytest.raises(ValueError, match="normalized_artifacts"):
        mod.validate_artifact(
            artifact
            | {
                "normalized_artifacts": {
                    "value": {},
                    "principle": mod.FIELD_PRINCIPLES["normalized_artifacts"],
                }
            }
        )
    with pytest.raises(ValueError, match="unsafe_missing_receipts"):
        mod.validate_artifact(
            artifact
            | {
                "unsafe_missing_receipts": {
                    "value": {},
                    "principle": mod.FIELD_PRINCIPLES["unsafe_missing_receipts"],
                }
            }
        )
    with pytest.raises(ValueError, match="unsafe_missing_receipts"):
        mod.validate_artifact(
            artifact
            | {
                "gap4_final_decision": {
                    "value": "blocked_missing_receipts",
                    "principle": mod.FIELD_PRINCIPLES["gap4_final_decision"],
                },
                "unsafe_missing_receipts": {
                    "value": [],
                    "principle": mod.FIELD_PRINCIPLES["unsafe_missing_receipts"],
                },
            }
        )
    with pytest.raises(ValueError, match="pool_retired"):
        mod.validate_artifact(
            artifact
            | {
                "pool_retired": {
                    "value": True,
                    "principle": mod.FIELD_PRINCIPLES["pool_retired"],
                }
            }
        )
    with pytest.raises(ValueError, match="no_new_generation"):
        mod.validate_artifact(
            artifact
            | {
                "no_new_generation": {
                    "value": False,
                    "principle": mod.FIELD_PRINCIPLES["no_new_generation"],
                }
            }
        )
    with pytest.raises(ValueError, match="normalizer_version"):
        mod.validate_artifact(
            artifact
            | {
                "normalizer_version": {
                    "value": "wrong",
                    "principle": mod.EXTRA_FIELD_PRINCIPLES["normalizer_version"],
                }
            }
        )
    with pytest.raises(ValueError, match="normalizer_checksum"):
        mod.validate_artifact(
            artifact
            | {
                "normalizer_checksum": {
                    "value": "bad",
                    "principle": mod.EXTRA_FIELD_PRINCIPLES["normalizer_checksum"],
                }
            }
        )
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(artifact | {"tests_run": "bad"})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "sha256:bad"})


def test_req_report_5248_defensive_receipt_paths(tmp_path: Path) -> None:
    """REQ-REPORT-5248: defensive branches still fail closed with typed evidence."""

    list_path = tmp_path / "list.json"
    list_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        mod._read_json(list_path)  # noqa: SLF001 - private helper is the defensive seam.
    with pytest.raises(ValueError, match="unknown GAP-4 source"):
        mod._source_checksum("unknown.json", {})  # noqa: SLF001

    payloads = _source_payloads()
    exp5224_pre_qa = {
        key: value
        for key, value in payloads[str(mod.EXP5224_RELATIVE_PATH)].items()
        if key not in mod.QA_ANNOTATION_FIELDS
    }
    checksum = mod._checksum_receipt(str(mod.EXP5224_RELATIVE_PATH), exp5224_pre_qa)  # noqa: SLF001
    assert checksum["classification"] == "safe-normalized"
    assert checksum["reason"] == "stored checksum already matches the full checked-in payload"

    unsafe = mod._field_classification(  # noqa: SLF001
        relative=str(mod.EXP5224_RELATIVE_PATH),
        rejection={"kind": "missing_methodology_receipt", "field": "methodology", "detail": "x"},
    )
    assert unsafe["classification"] == "unsafe-missing"
    assert mod._claim_critical_missing(  # noqa: SLF001
        [
            {
                "path": "fixture.json",
                "field_classifications": [
                    "not-a-mapping",
                    {
                        "field": "methodology",
                        "classification": "unsafe-missing",
                        "claim_critical": True,
                        "kind": "missing_methodology_receipt",
                    },
                ],
            }
        ]
    ) == [
        {
            "artifact": "fixture.json",
            "field": "methodology",
            "reason": "missing_methodology_receipt",
        }
    ]

    contradiction_payloads = copy.deepcopy(payloads)
    contradiction_payloads[str(mod.EXP5236_RELATIVE_PATH)]["wins"] = 99
    contradiction = mod.build_decision_from_payloads(
        contradiction_payloads,
        normalizer_artifact=mod.load_exp5247(REPO),
    )
    assert {
        "artifact": str(mod.EXP5236_RELATIVE_PATH),
        "field": "wins",
        "reason": "frozen_count_contradiction",
    } in contradiction.unsafe_missing_receipts

    positive_payloads = copy.deepcopy(payloads)
    positive_payloads[str(mod.EXP5225_RELATIVE_PATH)]["wins"] = 6
    positive_payloads[str(mod.EXP5225_RELATIVE_PATH)]["losses"] = 0
    positive = mod.build_decision_from_payloads(
        positive_payloads,
        normalizer_artifact=mod.load_exp5247(REPO),
        forced_missing_receipts=[
            {
                "artifact": "duplicate",
                "field": "x",
                "reason": "same",
            },
            {
                "artifact": "duplicate",
                "field": "x",
                "reason": "same",
            },
        ],
    )
    assert positive.unsafe_missing_receipts.count(
        {"artifact": "duplicate", "field": "x", "reason": "same"}
    ) == 1
    assert {
        "artifact": str(mod.EXP5225_RELATIVE_PATH),
        "field": "wins/losses",
        "reason": "frozen_counts_cross_min6_not_clean_null",
    } in positive.unsafe_missing_receipts


def test_req_report_5248_write_artifact_outputs_valid_json(tmp_path: Path) -> None:
    """REQ-REPORT-5248: writer emits the requested stable JSON result."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_artifact(
        output_path=output,
        root=REPO,
        tests_run=[{"command": "pytest fixture", "outcome": "PASS"}],
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == artifact
    mod.validate_artifact(written)


def test_req_report_5248_repository_artifact_matches_schema() -> None:
    """REQ-REPORT-5248: checked-in result artifact remains valid."""

    if not RESULT_PATH.exists():
        pytest.skip("Exp5248 artifact not written yet")
    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(artifact)
    assert _value(artifact, "gap4_final_decision") == "salvaged_clean_null"

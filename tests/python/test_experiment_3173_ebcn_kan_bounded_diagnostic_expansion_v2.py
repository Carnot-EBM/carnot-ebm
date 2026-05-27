"""Tests for Exp 3173 EBCN/KAN bounded diagnostic expansion v2.

Spec refs: REQ-VERIFY-3173, SCENARIO-VERIFY-3173.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import ebcn_kan_bounded_diagnostic_expansion_v2 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def test_req_verify_3173_spec_anchor_exists() -> None:
    """REQ-VERIFY-3173: the bounded panel schema is declared first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3173" in spec
    assert "SCENARIO-VERIFY-3173" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_verify_3173_writes_matrix_v28_bounded_panel(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3173: checked-in exact rows back a nondeployment panel."""

    output = mod.write_artifact(
        REPO_ROOT,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=11.25,
        tests_run=["focused-REQ-VERIFY-3173"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["ebcn_kan_bounded_diagnostic_expansion_v2_ready"] is True
    assert artifact["exact_labeled_row_count"] == 72
    assert artifact["known_false_accept_rows_scored"] == 2
    assert artifact["kan_monitor_record_count"] == 4
    assert artifact["deployed_verifier_claim_allowed"] is False
    assert artifact["live_integration_claim_allowed"] is False
    assert artifact["tests_run"] == ["focused-REQ-VERIFY-3173"]
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete:")

    ebcn = artifact["ebcn_localization_metrics"]
    assert ebcn["scored_row_count"] == 6
    assert ebcn["known_false_accept_rows_scored"] == 2
    assert ebcn["unscored_exact_row_count"] == 66
    assert ebcn["localization_coverage"] == pytest.approx(1.0)
    assert ebcn["false_accept_vs_clean_auc"] == pytest.approx(1.0)
    assert ebcn["false_accept_min_scalar_energy"] > ebcn["clean_accept_max_scalar_energy"]
    assert ebcn["false_accept_energy_margin_over_clean_max"] > 0.0

    kan = artifact["kan_monitor_coverage_metrics"]
    assert kan["monitor_record_count"] == 4
    assert kan["known_false_accept_monitor_record_count"] == 2
    assert kan["known_false_accept_monitor_record_coverage"] == pytest.approx(1.0)
    assert kan["exact_set_monitor_record_coverage"] == pytest.approx(0.055556)

    rerun = artifact["clean_verifier_rerun_status"]
    assert rerun["artifact_present"] is True
    assert rerun["gated_skip"] is True
    assert rerun["live_call_count"] == 0
    assert rerun["rows_contributed"] == 0

    blockers = "\n".join(artifact["promotion_blockers"])
    assert "tiny denominator" in blockers
    assert "No live integration" in blockers
    assert "No deployed verifier" in blockers

    substrate = artifact["inference_substrate"]
    assert substrate["executes_models"] is False
    assert substrate["generation_performed"] is False
    assert substrate["training_performed"] is False
    assert substrate["new_live_model_calls"] == 0
    assert substrate["offline_diagnostic_only"] is True
    mod.validate_artifact(artifact)


def test_req_verify_3173_row_provenance_tracks_exact_sources() -> None:
    """REQ-VERIFY-3173: exact rows keep per-source diagnostic provenance."""

    sources = mod.load_sources(REPO_ROOT)
    rows = mod.collect_exact_rows(sources)
    row_by_id = {row["row_id"]: row for row in rows}

    assert len(rows) == 72
    false_accept = row_by_id["resyn-3084-arith-003"]
    assert false_accept["known_false_accept"] is True
    assert false_accept["exact_label"] == "INVALID"
    assert false_accept["expected_action"] == "reject"
    assert false_accept["ebcn_score"]["scalar_energy"] > 0.0
    assert false_accept["kan_monitor_record"]["record_id"].startswith(
        "kan-proof-monitor-expansion-v1:"
    )
    assert {
        "exp3136_false_accept_autopsy",
        "exp3137_exact_safe_contract",
        "exp3138_canonical_grounding",
        "exp3158_ebcn_energy_sidecar",
        "exp3159_kan_monitor_expansion",
        "exp3167_clean_live_verifier_rerun",
    } <= set(false_accept["source_artifact_ids"])

    exact_only = row_by_id["resyn-3084-smt-023"]
    assert exact_only["exact_label"] == "SAT"
    assert exact_only["known_false_accept"] is False
    assert exact_only["ebcn_score"] is None
    assert exact_only["kan_monitor_record"] is None
    assert "exp3137_exact_safe_contract" in exact_only["source_artifact_ids"]
    assert "exp3167_clean_live_verifier_rerun" in exact_only["source_artifact_ids"]

    synthetic_sources = mod.load_sources(REPO_ROOT)
    synthetic_sources["payloads"]["exp3167_clean_live_verifier_rerun"] = {
        "gated_skip": False,
        "live_call_count": 1,
        "rerun_rows": [
            {
                "row_id": "synthetic-clean-rerun",
                "exact_label": "VALID",
                "expected_action": "accept",
                "live_decision": "accept",
            }
        ],
    }
    synthetic_rows = mod.collect_exact_rows(synthetic_sources)
    synthetic_by_id = {row["row_id"]: row for row in synthetic_rows}
    assert synthetic_by_id["synthetic-clean-rerun"]["clean_rerun_live_row"] is True

    empty_rows: dict[str, dict[str, Any]] = {}
    mod.attach_exp3136_row(empty_rows, {}, set())
    assert empty_rows == {}


def test_req_verify_3173_validation_rejects_overclaims(tmp_path: Path) -> None:
    """REQ-VERIFY-3173: validation blocks missing fields and live/deployed claims."""

    output = mod.write_artifact(
        REPO_ROOT,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=1.0,
        now_s=1.5,
        tests_run=["validation"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")
    invalid_cases = [
        (missing_required, "missing required fields"),
        (artifact | {"honest_verdict": "ready"}, "honest_verdict"),
        (artifact | {"deployed_verifier_claim_allowed": True}, "deployed verifier"),
        (artifact | {"live_integration_claim_allowed": True}, "live integration"),
        (artifact | {"promotion_blockers": []}, "promotion_blockers"),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"new_live_model_calls": 1}
            },
            "new live model calls",
        ),
        (
            artifact
            | {
                "ebcn_localization_metrics": artifact["ebcn_localization_metrics"]
                | {"localization_coverage": 1.5}
            },
            "localization_coverage",
        ),
        (artifact | {"kan_monitor_record_count": 3}, "kan_monitor_record_count"),
    ]

    for bad_artifact, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad_artifact)

    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": "offline"})


def test_req_verify_3173_fails_closed_when_sources_are_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-3173: missing sources produce a blocked nondeployment artifact."""

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=5.0,
        tests_run=["missing-source"],
    )

    assert artifact["ebcn_kan_bounded_diagnostic_expansion_v2_ready"] is False
    assert artifact["exact_labeled_row_count"] == 0
    assert artifact["known_false_accept_rows_scored"] == 0
    assert artifact["kan_monitor_record_count"] == 0
    assert artifact["deployed_verifier_claim_allowed"] is False
    assert artifact["live_integration_claim_allowed"] is False
    assert artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["inference_substrate"]["new_live_model_calls"] == 0
    mod.validate_artifact(artifact)

    relative_output = mod.write_artifact(
        tmp_path,
        output_path=Path("relative-exp3173.json"),
        started_s=6.0,
        now_s=6.25,
        tests_run=["relative-missing-source"],
    )
    assert relative_output == tmp_path / "relative-exp3173.json"
    assert json.loads(relative_output.read_text("utf-8"))["exact_labeled_row_count"] == 0

"""Tests for Exp 3279 evidence matrix v35.

Spec refs: REQ-REPORT-3279, SCENARIO-REPORT-3279.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import evidence_matrix_v35_3279 as mod


REQUIRED_FIELDS = {
    "matrix_v35_ready",
    "clean_row_count",
    "blocked_row_count",
    "flagged_row_count",
    "missing_row_count",
    "sidecar_only_row_count",
    "publication_blocker_count_estimate",
    "next_gap_candidates",
    "rows",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate_payload() -> dict[str, Any]:
    return {
        "experiment": 3276,
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "title": "Repair gate decision v8 after v4, Garak, and clean verifier",
        "blocked_at_layer": "conductor_pre_gate",
        "gate_check_summary": (
            "2 of 3 gate(s) failed; first failure: "
            "exp3274-prompt-injection-v4-garak-dataflip-redteam-eval-v1."
            "garak_redteam_eval_ready (actual=False == expected=True)"
        ),
        "gates_evaluated": [
            {
                "upstream": "exp3273-prompt-injection-kan-full-corpus-delong-eval-v1",
                "artifact_field": "v4_full_eval_ready",
                "expected": True,
                "actual": True,
                "passed": True,
                "reason": "actual=True == expected=True",
            },
            {
                "upstream": "exp3274-prompt-injection-v4-garak-dataflip-redteam-eval-v1",
                "artifact_field": "garak_redteam_eval_ready",
                "expected": True,
                "actual": False,
                "passed": False,
                "reason": "actual=False == expected=True",
            },
            {
                "upstream": "exp3275-clean-local-sota-verifier-rerun-v14",
                "artifact_field": "clean_verifier_rerun_ready",
                "expected": True,
                "actual": False,
                "passed": False,
                "reason": "actual=False == expected=True",
            },
        ],
        "honest_verdict": "blocked_gate_check_failed",
    }


def _write_dot303_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3267_REL_PATH,
        {
            "experiment_id": "exp3267",
            "task_id": "exp3267-close-v302-open-v303-corpus-queue",
            "v302_closed_v303_opened": True,
            "prior_paper_ready": False,
            "prior_publication_blocker_count": 105,
            "prior_next_top_gap": "full_15k_v4_corpus_across_shards_plus_repair_and_garak_gates",
            "honest_verdict": "complete: v302_closed_v303_opened=true",
        },
    )
    _write_json(
        root,
        mod.EXP3268_REL_PATH,
        {
            "experiment_id": "exp3268",
            "task_id": "exp3268-sota-receipt-methodology-supplement-v1",
            "sota_receipt_methodology_supplement_v1_ready": True,
            "clean_sota_receipt_eligible": True,
            "methodology_findings": ["methodology_clean_live_receipt_available"],
            "honest_verdict": "complete: clean_sota_receipt_eligible=true",
        },
    )
    _write_json(
        root,
        mod.EXP3269_REL_PATH,
        {
            "experiment_id": "exp3269",
            "task_id": "exp3269-prompt-injection-v4-full-corpus-split-manifest-v1",
            "full_corpus_manifest_ready": True,
            "target_total_examples": 15000,
            "completed_seed_examples": 2000,
            "planned_new_examples": 13000,
            "manifest_blockers": [],
            "honest_verdict": "complete: full_corpus_manifest_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3270_REL_PATH,
        {
            "experiment_id": "exp3270",
            "task_id": "exp3270-prompt-injection-teacher-label-shards-2-4-v1",
            "teacher_label_shards_2_4_ready": True,
            "cumulative_label_count": 8000,
            "new_label_count": 6000,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=11.875108"}
            ],
            "honest_verdict": "complete: teacher_label_shards_2_4_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3271_REL_PATH,
        {
            "experiment_id": "exp3271",
            "task_id": "exp3271-prompt-injection-teacher-label-shards-5-7-garak-seed-v1",
            "teacher_label_shards_5_7_garak_seed_ready": True,
            "cumulative_label_count": 14000,
            "new_label_count": 6000,
            "garak_seed_count": 1000,
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=11.75169"}
            ],
            "honest_verdict": "complete: teacher_label_shards_5_7_garak_seed_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3272_REL_PATH,
        {
            "experiment_id": "exp3272",
            "task_id": "exp3272-prompt-injection-v4-full-corpus-assembly-leakage-audit-v1",
            "full_15k_corpus_ready": True,
            "leakage_audit_passed": True,
            "assembled_example_count": 15000,
            "train_count": 10000,
            "eval_count": 2000,
            "holdout_count": 2000,
            "garak_count": 1000,
            "flagged_adversarial": True,
            "corrigendum_pending": [{"kind": "TAUTOLOGY", "detail": "counts match exactly"}],
            "honest_verdict": "complete: full_15k_corpus_ready=true",
        },
    )
    _write_json(
        root,
        mod.EXP3273_REL_PATH,
        {
            "experiment_id": "exp3273",
            "task_id": "exp3273-prompt-injection-kan-full-corpus-delong-eval-v1",
            "v4_full_eval_ready": True,
            "sidecar_only": True,
            "full_corpus_auroc": 0.475326,
            "full_corpus_auprc": 0.626269,
            "delong_noninferiority_passed": False,
            "honest_verdict": "complete: v4_full_eval_ready=true; sidecar_only=true",
        },
    )
    _write_json(
        root,
        mod.EXP3274_REL_PATH,
        {
            "experiment_id": "exp3274",
            "task_id": "exp3274-prompt-injection-v4-garak-dataflip-redteam-eval-v1",
            "garak_redteam_eval_ready": False,
            "garak_available": False,
            "garak_gate_passed": False,
            "blocked_reasons": ["blocked_garak_unavailable"],
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=1.138554"}
            ],
            "honest_verdict": "complete: garak_redteam_eval_ready=false",
        },
    )
    _write_json(
        root,
        mod.EXP3275_REL_PATH,
        {
            "experiment_id": "exp3275",
            "clean_verifier_rerun_ready": False,
            "clean_rerun_allowed": False,
            "repair_gate_input_clean_enough": False,
            "gate_reasons": ["abstention_rate_above_threshold"],
            "flagged_adversarial": True,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=9.73802"}
            ],
            "n_eval": 6,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
            "abstention_rate": 1.0,
            "honest_verdict": "complete: clean verifier rerun not ready",
        },
    )
    _write_json(root, mod.EXP3276_REL_PATH, _gate_payload())
    _write_json(
        root,
        mod.EXP3278_REL_PATH,
        {
            "experiment_id": "exp3278",
            "task_id": "exp3278-fr11-full-corpus-continual-self-learning-audit-v1",
            "continuous_self_learning_task": True,
            "fr11_full_corpus_audit_ready": True,
            "controller_memory_only": True,
            "foundation_weight_updates_performed": False,
            "retention_score": 0.982143,
            "adaptation_score": 1.0,
            "forgetting_rate": 0.017857,
            "negative_transfer_rate": 0.0,
            "heldout_trace_count": 2056,
            "honest_verdict": "complete: fr11 full-corpus audit ready=true",
        },
    )


def _row_by_id(artifact: Mapping[str, Any], experiment_id: str) -> Mapping[str, Any]:
    return next(row for row in artifact["rows"] if row["experiment_id"] == experiment_id)


def test_req_report_3279_spec_anchor_exists() -> None:
    """REQ-REPORT-3279: OpenSpec declares matrix v35 before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3279" in spec
    assert "SCENARIO-REPORT-3279" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3279_builds_v35_from_dot303_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3279: v35 records clean, blocked, flagged, and missing rows."""

    _write_dot303_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=8.25)
    statuses = {row["experiment_id"]: row["status"] for row in artifact["rows"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["experiment_id"] == "exp3279"
    assert artifact["task_id"] == "exp3279-evidence-matrix-v35"
    assert artifact["milestone"] == "2026.05.303"
    assert artifact["inference_substrate"] == "artifact_aggregation_only"
    assert artifact["matrix_v35_ready"] is True
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert len(artifact["rows"]) == 12
    assert statuses == {
        "exp3267": "clean",
        "exp3268": "clean",
        "exp3269": "clean",
        "exp3270": "flagged",
        "exp3271": "flagged",
        "exp3272": "flagged",
        "exp3273": "sidecar-only",
        "exp3274": "blocked",
        "exp3275": "blocked",
        "exp3276": "blocked",
        "exp3277": "missing",
        "exp3278": "clean",
    }
    assert artifact["primary_status_counts"] == {
        "clean": 4,
        "blocked": 3,
        "flagged": 3,
        "missing": 1,
        "pilot-only": 0,
        "sidecar-only": 1,
    }
    assert artifact["clean_row_count"] == 4
    assert artifact["blocked_row_count"] == 3
    assert artifact["flagged_row_count"] == 5
    assert artifact["missing_row_count"] == 1
    assert artifact["sidecar_only_row_count"] == 1
    assert artifact["pilot_only_row_count"] == 0

    assert _row_by_id(artifact, "exp3274")["blocker_reasons"] == [
        "blocked_garak_unavailable"
    ]
    assert _row_by_id(artifact, "exp3275")["blocker_reasons"] == [
        "abstention_rate_above_threshold"
    ]
    assert _row_by_id(artifact, "exp3276")["blocker_reasons"] == [
        "2 of 3 gate(s) failed; first failure: "
        "exp3274-prompt-injection-v4-garak-dataflip-redteam-eval-v1."
        "garak_redteam_eval_ready (actual=False == expected=True)",
        "actual=False == expected=True",
        "actual=False == expected=True",
    ]
    assert _row_by_id(artifact, "exp3277")["blocker_reasons"] == [
        "artifact_missing: results/experiment_3277_sota_repair_micro_panel_v9.json"
    ]
    assert _row_by_id(artifact, "exp3270")["quality_flags"] == [
        {"kind": "DURATION_TOO_SHORT", "detail": "duration_s=11.875108"}
    ]
    assert _row_by_id(artifact, "exp3273")["bounded_claims"] == [
        "sidecar_only=true",
        "delong_noninferiority_passed=false",
    ]

    assert artifact["prior_publication_blocker_count"] == 105
    assert artifact["publication_blocker_count_estimate"] == 105
    assert artifact["publication_blocker_delta_from_v302"] == 0
    assert artifact["publication_blocker_movement"] == "unchanged"
    assert artifact["paper_ready"] is False
    assert artifact["publication_readiness"]["paper_ready"] is False
    assert artifact["next_gap_candidates"][0] == {
        "rank": 1,
        "gap": "unblock_garak_redteam_eval",
        "source_experiment_id": "exp3274",
        "reason": "blocked_garak_unavailable",
    }
    assert artifact["source_checksums"][mod.EXP3267_REL_PATH.as_posix()]
    assert artifact["honest_verdict"].startswith("complete:")
    assert "publication_blocker_delta_from_v302=0" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_req_report_3279_writer_and_empty_sources_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3279: writer persists output and missing evidence never becomes ready."""

    _write_dot303_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=3.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["matrix_v35_ready"] is True
    assert saved["duration_s"] == pytest.approx(2.0)

    empty = mod.build_artifact(tmp_path / "empty")
    assert empty["matrix_v35_ready"] is False
    assert empty["missing_row_count"] == 12
    assert empty["blocked_row_count"] == 0
    assert empty["publication_blocker_count_estimate"] == 105
    assert empty["paper_ready"] is False
    assert "exp3267 .303 handoff artifact is missing or not ready" in empty["invariant_violations"]
    mod.validate_artifact(empty)


def test_req_report_3279_defensive_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-3279: helper paths classify malformed evidence without overclaiming."""

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(non_object) == {}
    present = tmp_path / "present.json"
    present.write_text('{"ok": true}\n', encoding="utf-8")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert len(mod.sha256_file(present) or "") == 64
    assert mod._duration(3.0, 2.0) == 0.0
    assert mod._normal_status("sidecar_only") == "sidecar-only"
    assert mod._normal_status("pilot_only") == "pilot-only"
    assert mod._normal_status("weird") == "missing"
    assert mod._bool_value(True) is True
    assert mod._bool_value("true") is False
    assert mod._int_value(3) == 3
    assert mod._int_value(True) == 0
    assert mod._number_value(1.25) == 1.25
    assert mod._number_value(True) == 0.0
    assert mod._as_mapping({"a": 1}) == {"a": 1}
    assert mod._as_mapping([]) == {}
    assert mod._as_list([{"a": 1}]) == [{"a": 1}]
    assert mod._as_list("bad") == []
    assert mod._list_of_strings(["a", 2, None]) == ["a", "2", "None"]
    assert (
        mod._status_for_source(
            mod.EXPECTED_SOURCES[0],
            {"v302_closed_v303_opened": True, "pilot_only": True},
            True,
        )
        == "pilot-only"
    )
    assert mod._blocker_reasons(
        {"present": True, "ready_field": "some_ready_field"},
        {"some_ready_field": False},
    ) == ["some_ready_field=false"]
    assert mod._quality_flags({"flagged_adversarial": True}) == [
        {"kind": "flagged_adversarial", "detail": "flagged_adversarial=true"}
    ]

    _write_dot303_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(artifact | {"experiment_id": "bad"})
    with pytest.raises(ValueError, match="task_id"):
        mod.validate_artifact(artifact | {"task_id": "bad"})
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(artifact | {"milestone": "bad"})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": "bad"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="publication_blocker_count_estimate"):
        mod.validate_artifact(artifact | {"publication_blocker_count_estimate": -1})
    with pytest.raises(ValueError, match="paper_ready cannot be true"):
        mod.validate_artifact(artifact | {"paper_ready": True, "publication_blocker_count_estimate": 1})

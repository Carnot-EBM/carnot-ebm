"""Tests for Exp 3280 milestone .303 capstone.

Spec refs: REQ-REPORT-3280, SCENARIO-REPORT-3280.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import capstone_v303_3280 as mod


REQUIRED_FIELDS = {
    "capstone_v303_ready",
    "paper_ready",
    "publication_blocker_count",
    "publication_blocker_delta",
    "v4_full_corpus_status",
    "garak_gate_status",
    "repair_gate_status",
    "fr11_status",
    "next_top_gap",
    "recommended_next_milestone_title",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capstone_v302() -> dict[str, Any]:
    return {
        "experiment_id": "exp3266",
        "task_id": "exp3266-capstone-v302",
        "capstone_v302_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 105,
        "next_top_gap": "full_15k_v4_corpus_across_shards_plus_repair_and_garak_gates",
        "honest_verdict": "complete: capstone_v302_ready=true",
    }


def _row(
    experiment_id: str,
    *,
    status: str,
    ready: bool,
    summary: Mapping[str, Any] | None = None,
    blocker_reasons: list[str] | None = None,
    quality_flags: list[Mapping[str, str]] | None = None,
    bounded_claims: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "status": status,
        "ready": ready,
        "summary": dict(summary or {}),
        "blocker_reasons": list(blocker_reasons or []),
        "quality_flags": [dict(item) for item in quality_flags or []],
        "bounded_claims": list(bounded_claims or []),
        "path": f"results/{experiment_id}.json",
        "sha256": "a" * 64,
    }


def _matrix_v35(*, ready: bool = True) -> dict[str, Any]:
    rows = [
        _row("exp3267", status="clean", ready=True),
        _row("exp3268", status="clean", ready=True),
        _row("exp3269", status="clean", ready=True),
        _row(
            "exp3270",
            status="flagged",
            ready=True,
            quality_flags=[{"kind": "DURATION_TOO_SHORT", "detail": "duration_s=11.875108"}],
        ),
        _row(
            "exp3271",
            status="flagged",
            ready=True,
            summary={"garak_seed_count": 1000, "cumulative_label_count": 14000},
            quality_flags=[{"kind": "DURATION_TOO_SHORT", "detail": "duration_s=11.75169"}],
        ),
        _row(
            "exp3272",
            status="flagged",
            ready=True,
            summary={
                "full_15k_corpus_ready": True,
                "assembled_example_count": 15000,
                "leakage_audit_passed": True,
            },
            quality_flags=[{"kind": "TAUTOLOGY", "detail": "counts match exactly"}],
        ),
        _row(
            "exp3273",
            status="sidecar-only",
            ready=True,
            summary={
                "v4_full_eval_ready": True,
                "sidecar_only": True,
                "full_corpus_auroc": 0.475326,
                "full_corpus_auprc": 0.626269,
                "delong_noninferiority_passed": False,
            },
            bounded_claims=["sidecar_only=true", "delong_noninferiority_passed=false"],
        ),
        _row(
            "exp3274",
            status="blocked",
            ready=False,
            summary={
                "garak_redteam_eval_ready": False,
                "garak_available": False,
                "garak_gate_passed": False,
                "dataflip_gate_passed": True,
            },
            blocker_reasons=["blocked_garak_unavailable"],
            quality_flags=[{"kind": "DURATION_TOO_SHORT", "detail": "duration_s=1.138554"}],
        ),
        _row(
            "exp3275",
            status="blocked",
            ready=False,
            summary={
                "clean_verifier_rerun_ready": False,
                "repair_gate_input_clean_enough": False,
            },
            blocker_reasons=["abstention_rate_above_threshold"],
            quality_flags=[{"kind": "DURATION_TOO_SHORT", "detail": "duration_s=9.73802"}],
        ),
        _row(
            "exp3276",
            status="blocked",
            ready=False,
            summary={"status": "blocked", "blocked_at_layer": "conductor_pre_gate"},
            blocker_reasons=[
                "2 of 3 gate(s) failed; first failure: "
                "exp3274-prompt-injection-v4-garak-dataflip-redteam-eval-v1."
                "garak_redteam_eval_ready (actual=False == expected=True)",
                "actual=False == expected=True",
                "actual=False == expected=True",
            ],
        ),
        _row(
            "exp3277",
            status="missing",
            ready=False,
            blocker_reasons=[
                "artifact_missing: results/experiment_3277_sota_repair_micro_panel_v9.json"
            ],
        ),
        _row(
            "exp3278",
            status="clean",
            ready=True,
            summary={
                "fr11_full_corpus_audit_ready": True,
                "controller_memory_only": True,
                "foundation_weight_updates_performed": False,
                "retention_score": 0.982143,
                "adaptation_score": 1.0,
                "forgetting_rate": 0.017857,
                "negative_transfer_rate": 0.0,
                "heldout_trace_count": 2056,
            },
        ),
    ]
    return {
        "experiment_id": "exp3279",
        "task_id": "exp3279-evidence-matrix-v35",
        "matrix_v35_ready": ready,
        "paper_ready": False,
        "publication_blocker_count_estimate": 105,
        "publication_blocker_delta_from_v302": 0,
        "rows": rows,
        "publication_readiness": {
            "paper_ready": False,
            "blocking_rows": ["exp3273", "exp3274", "exp3275", "exp3276", "exp3277"],
            "flagged_rows": ["exp3270", "exp3271", "exp3272", "exp3274", "exp3275"],
            "required_gates": {
                "full_15k_corpus": True,
                "kan_full_eval": True,
                "garak_redteam": False,
                "clean_verifier": False,
                "repair_gate": False,
                "repair_micro_panel": False,
                "fr11_full_corpus": True,
            },
        },
        "next_gap_candidates": [
            {
                "rank": 1,
                "gap": "unblock_garak_redteam_eval",
                "source_experiment_id": "exp3274",
                "reason": "blocked_garak_unavailable",
            }
        ],
        "invariant_violations": [] if ready else ["matrix source gate not ready"],
        "honest_verdict": f"complete: matrix_v35_ready={str(ready).lower()}",
    }


def _write_available_dot303_sources(root: Path) -> None:
    _write_json(root, mod.CAPSTONE_V302_REL_PATH, _capstone_v302())
    _write_json(root, mod.MATRIX_V35_REL_PATH, _matrix_v35())
    _write_json(root, mod.EXP3267_REL_PATH, {"experiment_id": "exp3267"})
    _write_json(root, mod.EXP3272_REL_PATH, {"experiment_id": "exp3272", "full_15k_corpus_ready": True})
    _write_json(
        root,
        mod.EXP3274_REL_PATH,
        {
            "experiment_id": "exp3274",
            "garak_redteam_eval_ready": False,
            "blocked_reasons": ["blocked_garak_unavailable"],
        },
    )
    _write_json(
        root,
        mod.EXP3278_REL_PATH,
        {
            "experiment_id": "exp3278",
            "fr11_full_corpus_audit_ready": True,
            "controller_memory_only": True,
        },
    )


def test_req_report_3280_spec_anchor_exists() -> None:
    """REQ-REPORT-3280: OpenSpec declares the .303 capstone first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3280" in spec
    assert "SCENARIO-REPORT-3280" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3280_closes_v303_without_publication_readiness(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3280: .303 narrows the top gap but leaves publication blocked."""

    _write_available_dot303_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=8.25)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3280"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.303"
    assert artifact["capstone_v303_ready"] is True
    assert artifact["gated_skip"] is False
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 105
    assert artifact["publication_blocker_delta"] == 0
    assert artifact["publication_blocker_trend"] == "unchanged"
    assert artifact["v302_next_top_gap_cleared"] is False
    assert artifact["v302_next_top_gap_status"].startswith("partial:")
    assert artifact["v4_full_corpus_status"] == (
        "partial: full_15k_ready_but_flagged_sidecar_noninferiority_failed"
    )
    assert artifact["garak_gate_status"] == "blocked: blocked_garak_unavailable"
    assert artifact["repair_gate_status"] == "blocked: garak_redteam_and_clean_verifier_gates_failed"
    assert artifact["fr11_status"] == (
        "complete: controller_memory_only_retention_0.982143_adaptation_1.0_forgetting_0.017857"
    )
    assert artifact["next_top_gap"] == "unblock_garak_redteam_eval"
    assert artifact["recommended_next_milestone_title"] == (
        "Garak Red-Team Availability + Clean Verifier Repair Gate Reopen"
    )
    assert "full_15k_v4_corpus_materialized" in artifact["changes_since_v302"]
    assert "publication_blocker_count_unchanged" in artifact["changes_since_v302"]
    assert "garak_redteam_blocked_unavailable" in artifact["stayed_blocked"]
    assert "repair_gate_blocked" in artifact["stayed_blocked"]
    assert "prompt_injection_artifact_flags_unresolved" in artifact["stayed_blocked"]
    assert artifact["no_new_garak_run"] is True
    assert artifact["no_new_repair_run"] is True
    assert artifact["ops_status_modified_by_this_task"] is False
    assert artifact["ops_changelog_modified_by_this_task"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert artifact["source_checksums"][mod.MATRIX_V35_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.MATRIX_V35_REL_PATH
    )
    assert artifact["honest_verdict"].startswith("complete:")
    assert "paper_ready=false" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_req_report_3280_writer_and_gated_skip_when_matrix_not_ready(tmp_path: Path) -> None:
    """REQ-REPORT-3280: matrix-not-ready still produces a complete gated-skip capstone."""

    _write_json(root=tmp_path, rel_path=mod.CAPSTONE_V302_REL_PATH, payload=_capstone_v302())
    _write_json(root=tmp_path, rel_path=mod.MATRIX_V35_REL_PATH, payload=_matrix_v35(ready=False))

    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=4.0)
    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=3.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["duration_s"] == 0.0
    assert artifact["capstone_v303_ready"] is True
    assert artifact["gated_skip"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 105
    assert artifact["publication_blocker_delta"] == 0
    assert artifact["v4_full_corpus_status"] == "gated_skip: matrix_v35_not_ready"
    assert artifact["garak_gate_status"] == "gated_skip: matrix_v35_not_ready"
    assert artifact["repair_gate_status"] == "gated_skip: matrix_v35_not_ready"
    assert artifact["fr11_status"] == "gated_skip: matrix_v35_not_ready"
    assert artifact["next_top_gap"] == "produce_ready_evidence_matrix_v35"
    assert artifact["recommended_next_milestone_title"] == (
        "Evidence Matrix V35 Repair Before Milestone Closeout"
    )
    assert "matrix source gate not ready" in artifact["gated_skip_reasons"]
    assert saved["duration_s"] == pytest.approx(2.0)
    assert output == tmp_path / mod.OUTPUT_REL_PATH


def test_req_report_3280_defensive_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-3280: malformed evidence and overclaims fail closed."""

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    listed = tmp_path / "list.json"
    listed.write_text("[]", encoding="utf-8")
    good = tmp_path / "good.json"
    good.write_text('{"ok": true}\n', encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    assert mod.read_json_object(listed) == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert len(mod.sha256_file(good) or "") == 64
    assert mod._as_mapping({"a": 1}) == {"a": 1}
    assert mod._as_mapping([]) == {}
    assert mod._as_list([1, 2]) == [1, 2]
    assert mod._as_list("bad") == []
    assert mod._int_value(True) == 0
    assert mod._int_value(7) == 7
    assert mod._number_value(1.5) == 1.5
    assert mod._number_value(True) == 0.0

    _write_available_dot303_sources(tmp_path)
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
    with pytest.raises(ValueError, match="publication_blocker_count"):
        mod.validate_artifact(artifact | {"publication_blocker_count": -1})
    with pytest.raises(ValueError, match="paper_ready cannot be true"):
        mod.validate_artifact(artifact | {"paper_ready": True, "publication_blocker_count": 1})

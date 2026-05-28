"""Tests for Exp 3293 milestone .304 capstone.

Spec refs: REQ-REPORT-3293, SCENARIO-REPORT-3293.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "carnot"
    / "reporting"
    / "capstone_v304_3293.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("capstone_v304_3293", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
mod = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(mod)


REQUIRED_FIELDS = {
    "capstone_v304_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v303",
    "garak_unblocked",
    "clean_verifier_abstention_unblocked",
    "kan_boundary_resolved",
    "repair_gate_open",
    "repair_micro_panel_ready",
    "fr11_memory_replay_safe",
    "next_top_gap",
    "recommended_next_milestone_title",
    "protected_files_untouched",
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


def _prior_v303(blockers: int = 105) -> dict[str, Any]:
    return {
        "experiment_id": "exp3280",
        "task_id": "exp3280-capstone-v303",
        "capstone_v303_ready": True,
        "paper_ready": False,
        "publication_blocker_count": blockers,
        "next_top_gap": "unblock_garak_redteam_eval",
        "honest_verdict": "complete: capstone_v303_ready=true; paper_ready=false",
    }


def _row(
    experiment_id: str,
    status: str,
    *,
    paper_blocking: bool,
    summary: Mapping[str, Any] | None = None,
    blocker_reasons: list[str] | None = None,
    quality_flags: list[Mapping[str, str]] | None = None,
    bounded_claims: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "status": status,
        "ready": True,
        "paper_blocking": paper_blocking,
        "summary": dict(summary or {}),
        "blocker_reasons": list(blocker_reasons or []),
        "quality_flags": [dict(flag) for flag in quality_flags or []],
        "bounded_claims": list(bounded_claims or []),
    }


def _matrix_v36() -> dict[str, Any]:
    return {
        "experiment_id": "exp3292",
        "task_id": "exp3292-evidence-matrix-v36",
        "matrix_v36_ready": True,
        "paper_ready": False,
        "paper_blocker_count": 10,
        "top_gaps": [
            {
                "rank": 1,
                "gap": "pass_garak_redteam_gate",
                "source_experiment_id": "exp3285",
                "status": "blocked",
                "reason": "garak_attack_success_or_error_gate_failed",
            },
            {
                "rank": 2,
                "gap": "repair_panel_duration_and_scope_boundary",
                "source_experiment_id": "exp3290",
                "status": "flagged",
                "reason": "DURATION_TOO_SHORT",
            },
        ],
        "gate_summary": {
            "garak_toolchain": {
                "status": "clean",
                "ready": True,
                "garak_runner_ready": True,
                "garak_available": True,
            },
            "garak_redteam": {
                "status": "blocked",
                "ready": True,
                "garak_redteam_eval_ready": True,
                "garak_gate_passed": False,
                "dataflip_gate_passed": True,
                "blocker_reasons": ["garak_attack_success_or_error_gate_failed"],
            },
            "clean_verifier": {
                "status": "clean",
                "ready": True,
                "clean_verifier_rerun_ready": True,
                "repair_gate_input_clean_enough": True,
                "abstention_rate": 0.0,
            },
            "kan_boundary": {
                "status": "sidecar-only",
                "ready": True,
                "kan_boundary_decision_ready": True,
                "kan_boundary_decision": "retire_from_prompt_injection_headline",
            },
            "repair_gate": {
                "status": "clean",
                "ready": True,
                "repair_gate_open": True,
            },
            "repair_panel": {
                "status": "flagged",
                "ready": True,
                "repair_panel_ran": True,
                "headline_claim_allowed": False,
            },
            "fr11": {
                "status": "clean",
                "ready": True,
                "controller_memory_only": True,
                "foundation_weight_updates_performed": False,
            },
        },
        "rows": [
            _row("exp3282", "clean", paper_blocking=False),
            _row(
                "exp3285",
                "blocked",
                paper_blocking=True,
                summary={"garak_redteam_eval_ready": True, "garak_gate_passed": False},
                blocker_reasons=["garak_attack_success_or_error_gate_failed"],
            ),
            _row(
                "exp3287",
                "clean",
                paper_blocking=False,
                summary={
                    "clean_verifier_rerun_ready": True,
                    "repair_gate_input_clean_enough": True,
                    "abstention_rate": 0.0,
                },
            ),
            _row(
                "exp3288",
                "sidecar-only",
                paper_blocking=True,
                summary={
                    "kan_boundary_decision_ready": True,
                    "kan_boundary_decision": "retire_from_prompt_injection_headline",
                },
                bounded_claims=["kan_boundary_decision=retire_from_prompt_injection_headline"],
            ),
            _row(
                "exp3289",
                "clean",
                paper_blocking=False,
                summary={"repair_gate_open": True},
            ),
            _row(
                "exp3290",
                "flagged",
                paper_blocking=True,
                summary={
                    "repair_panel_ran": True,
                    "sota_repair_micro_panel_v10_ready": True,
                    "headline_claim_allowed": False,
                },
                quality_flags=[{"kind": "DURATION_TOO_SHORT", "detail": "duration_s=10.893664"}],
            ),
            _row(
                "exp3291",
                "clean",
                paper_blocking=False,
                summary={
                    "fr11_garak_abstention_memory_replay_ready": True,
                    "controller_memory_only": True,
                    "foundation_weight_updates_performed": False,
                    "raw_episodes_preserved": True,
                    "retention_score": 0.982143,
                    "adaptation_score": 1.0,
                    "forgetting_rate": 0.017857,
                },
                bounded_claims=[
                    "controller_memory_only=true",
                    "foundation_weight_updates_performed=false",
                ],
            ),
        ],
        "carried_forward_blockers": [
            {"prior_experiment_id": "exp3270", "reason": ".303 methodology flag remains"},
            {"prior_experiment_id": "exp3271", "reason": ".303 methodology flag remains"},
            {"prior_experiment_id": "exp3272", "reason": ".303 methodology flag remains"},
        ],
        "honest_verdict": "complete: matrix_v36_ready=true; paper_ready=false",
    }


def _write_sources(root: Path) -> None:
    _write_json(root, mod.CAPSTONE_V303_REL_PATH, _prior_v303())
    _write_json(root, mod.MATRIX_V36_REL_PATH, _matrix_v36())


def test_req_report_3293_spec_anchor_exists() -> None:
    """REQ-REPORT-3293: OpenSpec declares the .304 capstone contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3293" in spec
    assert "SCENARIO-REPORT-3293" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_report_3293_closes_v304_without_paper_ready(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3293: .304 closes but only clean headline evidence can publish."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=5.75)
    second = mod.build_artifact(tmp_path, started_s=8.0, now_s=9.0)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3293"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.304"
    assert artifact["inference_substrate"] == "artifact_aggregation_only"
    assert artifact["capstone_v304_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 105
    assert artifact["publication_blocker_count"] == 10
    assert artifact["blocker_delta_from_v303"] == -95
    assert artifact["garak_unblocked"] is True
    assert artifact["garak_gate_passed"] is False
    assert artifact["clean_verifier_abstention_unblocked"] is True
    assert artifact["kan_boundary_resolved"] is True
    assert artifact["kan_boundary_decision"] == "retire_from_prompt_injection_headline"
    assert artifact["repair_gate_open"] is True
    assert artifact["repair_micro_panel_ready"] is True
    assert artifact["repair_micro_panel_headline_eligible"] is False
    assert artifact["fr11_memory_replay_safe"] is True
    assert artifact["next_top_gap"] == "pass_garak_redteam_gate"
    assert artifact["recommended_next_milestone_title"] == (
        "Garak Red-Team Gate Pass + Headline-Eligible Repair Evidence"
    )
    assert "matrix_paper_ready_false" in artifact["paper_ready_blockers"]
    assert "blocked_or_flagged_or_sidecar_rows_present" in artifact["paper_ready_blockers"]
    assert "carried_forward_dot303_blockers_present" in artifact["paper_ready_blockers"]
    assert artifact["protected_files_untouched"] is True
    assert artifact["protected_file_status"]["research-roadmap.yaml"]["modified"] is False
    assert artifact["protected_file_status"]["scripts/research_conductor.py"]["modified"] is False
    assert artifact["no_new_garak_run"] is True
    assert artifact["no_new_repair_run"] is True
    assert artifact["no_next_milestone_activation"] is True
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["duration_s"] == pytest.approx(3.75)
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert artifact["source_checksums"][mod.MATRIX_V36_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.MATRIX_V36_REL_PATH
    )
    assert artifact["honest_verdict"].startswith("complete:")
    assert "paper_ready=false" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_req_report_3293_writer_and_matrix_missing_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3293: writer persists JSON and absent matrix cannot close .304."""

    _write_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=2.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_v304_ready"] is True
    assert saved["duration_s"] == pytest.approx(1.5)

    _write_json(tmp_path / "no_matrix", mod.CAPSTONE_V303_REL_PATH, _prior_v303())
    missing = mod.build_artifact(tmp_path / "no_matrix", started_s=4.0, now_s=3.0)

    assert missing["duration_s"] == 0.0
    assert missing["capstone_v304_ready"] is False
    assert missing["paper_ready"] is False
    assert missing["publication_blocker_count"] == 105
    assert missing["blocker_delta_from_v303"] == 0
    assert missing["next_top_gap"] == "produce_ready_evidence_matrix_v36"
    assert missing["garak_unblocked"] is False
    assert "matrix_v36_missing_or_not_ready" in missing["paper_ready_blockers"]
    mod.validate_artifact(missing)


def test_req_report_3293_headline_rule_and_validation_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3293: flagged or non-headline rows block paper readiness."""

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
    assert mod._as_list([1]) == [1]
    assert mod._as_list("bad") == []
    assert mod._int_value(True) == 0
    assert mod._int_value(7) == 7
    assert mod._number_value(0.25) == 0.25
    assert mod._number_value(True) == 0.0

    optimistic = _matrix_v36()
    optimistic["paper_ready"] = True
    optimistic["paper_blocker_count"] = 0
    _write_json(tmp_path, mod.CAPSTONE_V303_REL_PATH, _prior_v303())
    _write_json(tmp_path, mod.MATRIX_V36_REL_PATH, optimistic)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)

    assert artifact["paper_ready"] is False
    assert "blocked_or_flagged_or_sidecar_rows_present" in artifact["paper_ready_blockers"]

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


def test_req_report_3293_fallback_helpers_and_protected_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-3293: fallback count and protected-file helpers are deterministic."""

    assert (
        mod._prior_publication_blocker_count(
            {},
            {"prior_matrix": {"publication_blocker_count_estimate": 12}},
        )
        == 12
    )
    assert (
        mod._prior_publication_blocker_count(
            {},
            {
                "rows": [
                    {
                        "experiment_id": "exp3281",
                        "summary": {"prior_publication_blocker_count": 9},
                    }
                ]
            },
        )
        == 9
    )
    assert mod._publication_blocker_count({"publication_blocker_count": 3}, 105) == 3
    assert (
        mod._recommended_next_milestone_title("repair_panel_duration_and_scope_boundary")
        == "Repair Panel Methodology Hardening"
    )
    assert (
        mod._recommended_next_milestone_title("resolve_dot303_methodology_flags")
        == "Prompt-Injection Methodology Corrigendum Closure"
    )
    assert (
        mod._recommended_next_milestone_title("unknown_gap")
        == "Publication Blocker Retirement Review"
    )

    def _raise_oserror(*_args: object, **_kwargs: object) -> object:
        raise OSError("git missing")

    monkeypatch.setattr(mod.subprocess, "run", _raise_oserror)
    missing_git = mod._protected_file_status(Path("."))
    assert missing_git["research-roadmap.yaml"]["modified"] is False
    assert missing_git["research-roadmap.yaml"]["git_status_available"] is False

    class _Result:
        returncode = 0
        stdout = " M research-roadmap.yaml\n"

    monkeypatch.setattr(mod.subprocess, "run", lambda *_args, **_kwargs: _Result())
    modified = mod._protected_file_status(Path("."))
    assert modified["research-roadmap.yaml"]["modified"] is True
    assert modified["research-roadmap.yaml"]["status"] == " M"
    assert modified["scripts/research_conductor.py"]["modified"] is False
    assert modified["scripts/research_conductor.py"]["git_status_available"] is True

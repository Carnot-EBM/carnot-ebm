"""Tests for Exp 3306 milestone .305 capstone.

Spec refs: REQ-REPORT-3306, SCENARIO-REPORT-3306.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import capstone_v305_3306 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capstone_v304(blockers: int = 10) -> dict[str, Any]:
    return {
        "artifact": "experiment_3293_capstone_v304",
        "experiment_id": "exp3293",
        "task_id": "exp3293-capstone-v304",
        "capstone_v304_ready": True,
        "paper_ready": False,
        "publication_blocker_count": blockers,
        "garak_gate_passed": False,
        "next_top_gap": "pass_garak_redteam_gate",
        "inference_substrate": "artifact_aggregation_only",
        "honest_verdict": "complete: capstone_v304_ready=true; paper_ready=false",
    }


def _row(
    experiment_id: str,
    *,
    summary: Mapping[str, Any] | None = None,
    claim_boundaries: list[str] | None = None,
    evidence_class: str = "clean-live",
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "evidence_class": evidence_class,
        "summary": dict(summary or {}),
        "claim_boundaries": list(claim_boundaries or []),
        "quality_flags": [],
        "blocker_reasons": [],
    }


def _matrix_v37() -> dict[str, Any]:
    return {
        "artifact": "experiment_3305_evidence_matrix_v37",
        "experiment_id": "exp3305",
        "task_id": "exp3305-evidence-matrix-v37",
        "matrix_v37_ready": True,
        "paper_ready": False,
        "paper_blocker_count": 8,
        "garak_gate_passed": True,
        "repair_headline_claim_allowed": False,
        "fr11_replay_safe": True,
        "top_gap": "clear_garak_dataflip_and_quality_flags",
        "source_checksums": {"results/upstream.json": "abc123"},
        "gate_summary": {
            "garak_gate": {
                "source_experiment_id": "exp3300",
                "garak_gate_passed": True,
                "dataflip_gate_passed": False,
            },
            "repair_headline": {
                "source_experiment_id": "exp3303",
                "repair_headline_claim_allowed": False,
            },
            "fr11_replay": {
                "source_experiment_id": "exp3304",
                "fr11_replay_safe": True,
            },
        },
        "rows": [
            _row(
                "exp3300",
                summary={
                    "garak_redteam_eval_v3_ready": True,
                    "garak_gate_passed": True,
                    "dataflip_gate_passed": False,
                    "attack_success_rate": 0.0,
                },
            ),
            _row(
                "exp3296",
                summary={"kan_prompt_injection_headline_retired": True},
                claim_boundaries=["kan_prompt_injection_headline_retired=true"],
                evidence_class="sidecar-only",
            ),
            _row(
                "exp3303",
                summary={"headline_claim_allowed_after_audit": False},
                evidence_class="blocked",
            ),
            _row(
                "exp3304",
                summary={
                    "fr11_redteam_repair_memory_replay_ready": True,
                    "controller_memory_only": True,
                    "foundation_weight_updates_performed": False,
                },
                evidence_class="sidecar-only",
            ),
        ],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "honest_verdict": "complete: matrix_v37_ready=true; paper_ready=false",
    }


def _write_sources(root: Path) -> None:
    _write_json(root, mod.MATRIX_V37_REL_PATH, _matrix_v37())
    _write_json(root, mod.CAPSTONE_V304_REL_PATH, _capstone_v304())


def test_req_report_3306_spec_anchor_declares_capstone_schema() -> None:
    """REQ-REPORT-3306: OpenSpec declares the .305 capstone contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3306" in spec
    assert "SCENARIO-REPORT-3306" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_3306_closes_v305_from_matrix_v37(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3306: capstone reports Garak pass and repair headline block."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=13.5)
    second = mod.build_artifact(tmp_path, started_s=20.0, now_s=21.0)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3306"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["run_date"] == "20260529"
    assert artifact["milestone"] == "2026.05.305"
    assert artifact["prior_milestone"] == "2026.05.304"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["capstone_v305_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["publication_blocker_count"] == 8
    assert artifact["prior_publication_blocker_count"] == 10
    assert artifact["blocker_delta_from_v304"] == -2
    assert artifact["garak_gate_passed"] is True
    assert artifact["garak_attack_success_rate"] == pytest.approx(0.0)
    assert artifact["repair_headline_claim_allowed"] is False
    assert artifact["fr11_memory_replay_safe"] is True
    assert artifact["kan_headline_retired"] is True
    assert artifact["next_top_gap"] == "clear_garak_dataflip_and_quality_flags"
    assert artifact["recommended_next_milestone_title"] == (
        "DataFlip And Quality-Flag Cleanup Before Publication Readiness"
    )
    assert artifact["protected_files_untouched"] is True
    assert artifact["protected_file_status"]["research-roadmap.yaml"]["modified"] is False
    assert artifact["protected_file_status"]["scripts/research_conductor.py"]["modified"] is False
    assert artifact["no_new_garak_run"] is True
    assert artifact["no_new_repair_run"] is True
    assert artifact["no_push"] is True
    assert artifact["no_next_milestone_activation"] is True
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert artifact["source_checksums"][mod.MATRIX_V37_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.MATRIX_V37_REL_PATH
    )
    assert {source["path"] for source in artifact["source_artifacts"]} == {
        mod.MATRIX_V37_REL_PATH.as_posix(),
        mod.CAPSTONE_V304_REL_PATH.as_posix(),
    }
    assert "matrix_paper_ready_false" in artifact["paper_ready_blockers"]
    assert "publication_blockers_present" in artifact["paper_ready_blockers"]
    assert "repair_headline_claim_not_allowed" in artifact["paper_ready_blockers"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert "garak_gate_passed=true" in artifact["honest_verdict"]
    assert "repair_headline_claim_allowed=false" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_req_report_3306_writer_and_missing_matrix_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3306: writer persists JSON and absent matrix cannot close .305."""

    _write_sources(tmp_path)
    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=2.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_v305_ready"] is True
    assert saved["duration_s"] == pytest.approx(1.25)

    no_matrix = tmp_path / "no_matrix"
    _write_json(no_matrix, mod.CAPSTONE_V304_REL_PATH, _capstone_v304())
    missing = mod.build_artifact(no_matrix, started_s=5.0, now_s=4.0)

    assert missing["duration_s"] == 0.0
    assert missing["capstone_v305_ready"] is False
    assert missing["paper_ready"] is False
    assert missing["publication_blocker_count"] == 10
    assert missing["blocker_delta_from_v304"] == 0
    assert missing["garak_gate_passed"] is False
    assert missing["garak_attack_success_rate"] == pytest.approx(0.0)
    assert missing["kan_headline_retired"] is False
    assert missing["next_top_gap"] == "produce_ready_evidence_matrix_v37"
    assert "matrix_v37_missing_or_not_ready" in missing["paper_ready_blockers"]
    mod.validate_artifact(missing)

    no_prior = tmp_path / "no_prior"
    _write_json(no_prior, mod.MATRIX_V37_REL_PATH, _matrix_v37())
    prior_missing = mod.build_artifact(no_prior, started_s=0.0, now_s=0.0)
    assert prior_missing["capstone_v305_ready"] is False
    assert "capstone_v304_missing_or_not_ready" in prior_missing["paper_ready_blockers"]


def test_req_report_3306_helpers_validation_and_protected_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-3306: helper and validation paths preserve conservative behavior."""

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
    assert mod._as_list(["a"]) == ["a"]
    assert mod._as_list("a") == []
    assert mod._int_value(True) == 0
    assert mod._int_value(7) == 7
    assert mod._number_value(0.25) == 0.25
    assert mod._number_value(True) == 0.0
    assert mod._publication_blocker_count({}, 10) == 10
    assert mod._publication_blocker_count({"publication_blocker_count": 0}, 10) == 0
    assert mod._garak_attack_success_rate({"attack_success_rate": 0.125}) == pytest.approx(0.125)
    assert (
        mod._garak_attack_success_rate(
            {"gate_summary": {"garak_gate": {"attack_success_rate": 0.2}}}
        )
        == pytest.approx(0.2)
    )
    assert (
        mod._recommended_next_milestone_title("clear_repair_headline_evidence_audit")
        == "Repair Headline Evidence Audit Closure"
    )
    assert (
        mod._recommended_next_milestone_title("pass_garak_redteam_gate")
        == "Garak Red-Team Gate Pass"
    )
    assert (
        mod._recommended_next_milestone_title("ready_for_v305_capstone")
        == "Milestone .305 Archive And Handoff"
    )
    assert (
        mod._recommended_next_milestone_title("unknown_gap")
        == "Publication Blocker Retirement Review"
    )

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=0.0, now_s=0.0)
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
    with pytest.raises(ValueError, match="no_push"):
        mod.validate_artifact(artifact | {"no_push": False})
    with pytest.raises(ValueError, match="no_next_milestone_activation"):
        mod.validate_artifact(artifact | {"no_next_milestone_activation": False})

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

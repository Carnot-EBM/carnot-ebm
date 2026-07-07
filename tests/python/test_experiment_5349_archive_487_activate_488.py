"""Tests for Exp 5349 archive .487 / .488 transition artifact.

Spec refs: REQ-REPORT-5349, SCENARIO-REPORT-5349,
SCENARIO-REPORT-5349-BLOCKED-NEXT-ROADMAP.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import yaml

from carnot import experiment_5349_archive_487_activate_488 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _roadmap(milestone: str) -> str:
    return yaml.safe_dump(
        {
            "milestone": milestone,
            "milestone_title": f"fixture {milestone}",
            "milestone_doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
            "tasks": [
                {
                    "id": "exp5349-archive-487-activate-488",
                    "milestone": milestone,
                    "deliverable": str(mod.RESULT_RELATIVE_PATH),
                    "title": "fixture transition",
                    "agent_type": "codex",
                    "model": "gpt-5.5",
                    "prompt": "REQ-REPORT-5349 fixture",
                }
            ],
        },
        sort_keys=False,
    )


def _capstone_payload() -> dict[str, Any]:
    return {
        "experiment_id": _wrap("exp5348-capstone-v487"),
        "milestone": _wrap("2026.07.487"),
        "status": _wrap("complete"),
        "honest_verdict": _wrap(
            "complete: .487 synthesized with runtime_clean=True, "
            "structured_output_protocol_ready=False, bounded_sota_quality_usable=False, "
            "utility_memory_ready=True, bounded_compressor_ready=True, "
            "self_learning_scaleup_ready=False, qstr_solver_kan_clean=True, "
            "internal_energy_corrigendum_clean=False, hardware_speedup_claim=false"
        ),
        "inference_substrate": _wrap("aggregation_from_upstream_artifacts"),
        "runtime_clean": True,
        "structured_output_protocol_ready": False,
        "bounded_sota_quality_usable": False,
        "utility_memory_ready": True,
        "bounded_compressor_ready": True,
        "self_learning_scaleup_ready": False,
        "qstr_fixture_ready": True,
        "solver_guidance_ready": True,
        "kan_constraint_bridge_ready": True,
        "internal_energy_corrigendum_clean": False,
        "hardware_speedup_claim": False,
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "gate_table": _wrap(
            [
                {
                    "gate": "runtime",
                    "ready": True,
                    "classification": "clean_runtime_no_quality_claim",
                    "claim_boundary": "runtime receipt only; no SOTA quality claim",
                    "source_artifacts": [
                        "results/experiment_5337_sota_runtime_corrigendum_multimodel_v487.json"
                    ],
                },
                {
                    "gate": "structured_output_protocol",
                    "ready": False,
                    "classification": "flagged_parse_only_protocol_candidate",
                    "claim_boundary": "parse-only protocol candidate; no quality claim",
                    "source_artifacts": [
                        "results/experiment_5338_structured_output_protocol_calibration_v487.json"
                    ],
                },
                {
                    "gate": "self_learning_scaleup",
                    "ready": False,
                    "classification": "flagged_scaleup_not_claimable",
                    "claim_boundary": "multi-session context-policy scale-up only when not flagged",
                    "source_artifacts": [
                        "results/experiment_5342_provenance_bound_self_learning_scaleup_v487.json"
                    ],
                },
                {
                    "gate": "hardware",
                    "ready": True,
                    "classification": "continuity_workload_receipt_no_speedup",
                    "claim_boundary": "hardware continuity and board-local smoke only; no speedup claim",
                    "source_artifacts": [
                        "results/experiment_5347_hardware_continuity_workload_receipts_v487.json"
                    ],
                },
                "ignored non-map row",
            ]
        ),
        "next_milestone_recommendation": _wrap(
            {
                "do_not_claim": [
                    "headline_sota_quality",
                    "structured_protocol_clean_success",
                    "self_learning_scaleup_clean",
                    "internal_energy_clean",
                    "hardware_speedup",
                ]
            }
        ),
    }


def _make_repo(
    root: Path,
    *,
    active_milestone: str = "2026.07.488",
    vnext_milestone: str = "2026.07.488",
    next_milestone: str | None = "2026.07.488",
    capstone: dict[str, Any] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("CODEX.md", "CLAUDE.md"):
        (root / relative).write_text("fixture\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(_roadmap(active_milestone), encoding="utf-8")
    if next_milestone is not None:
        (root / "research-roadmap-next.yaml").write_text(_roadmap(next_milestone), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "openspec/change-proposals/research-roadmap-vNEXT.md").write_text(
        f"# Research Roadmap vNEXT: {vnext_milestone}\nMilestone: {vnext_milestone}\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    for relative in ("status.md", "changelog.md", "conductor-log.md"):
        (root / "ops" / relative).write_text("fixture 2026.07.488\n", encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    return root


def test_req_report_5349_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5349: OpenSpec anchors the .487 archive and .488 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5349") : spec.index("REQ-REPORT-5336")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5349",
        "SCENARIO-REPORT-5349",
        "SCENARIO-REPORT-5349-BLOCKED-NEXT-ROADMAP",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`roadmap_next_present`",
        "`active_roadmap_modified`",
        "`conductor_modified`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5349_present_next_roadmap_records_no_overwrite(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5349: present .488 roadmap-next emits complete no-overwrite artifact."""

    root = _make_repo(tmp_path, capstone=_capstone_payload())
    roadmap_before = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    conductor_before = (root / "scripts/research_conductor.py").read_text(encoding="utf-8")

    artifact = mod.build_artifact(
        root=root,
        run_date="20260707",
        duration_s=0.5,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    mod.validate_artifact(artifact)
    assert (root / "research-roadmap.yaml").read_text(encoding="utf-8") == roadmap_before
    assert (root / "scripts/research_conductor.py").read_text(encoding="utf-8") == conductor_before
    assert artifact["experiment_id"]["value"] == mod.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == mod.ACTIVATED_MILESTONE
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["archived_milestone"]["value"] == mod.ARCHIVED_MILESTONE
    assert artifact["activated_milestone"]["value"] == mod.ACTIVATED_MILESTONE
    assert artifact["v487_capstone_verdict"]["value"].startswith("complete: .487 synthesized")
    assert artifact["roadmap_next_present"] is True
    assert artifact["milestone_doc_present"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["failed_preconditions"] == []

    preconditions = artifact["preconditions_checked"]["value"]
    assert preconditions["roadmap_next_names_activated_milestone"] is True
    assert preconditions["no_active_roadmap_overwrite_performed"] is True
    assert preconditions["no_conductor_edit_performed"] is True

    summary = artifact["v487_capstone_proves"]["value"]
    assert summary["runtime_clean"] is True
    assert summary["structured_output_protocol_ready"] is False
    assert summary["bounded_sota_quality_usable"] is False
    assert summary["utility_memory_ready"] is True
    assert summary["bounded_compressor_ready"] is True
    assert summary["self_learning_scaleup_ready"] is False
    assert summary["qstr_fixture_ready"] is True
    assert summary["solver_guidance_ready"] is True
    assert summary["kan_constraint_bridge_ready"] is True
    assert summary["internal_energy_corrigendum_clean"] is False
    assert summary["hardware_speedup_claim"] is False
    gates = {row["gate"]: row for row in summary["gate_claim_boundaries"]}
    assert gates["runtime"]["claim_boundary"] == "runtime receipt only; no SOTA quality claim"
    assert gates["structured_output_protocol"]["ready"] is False
    assert gates["self_learning_scaleup"]["classification"] == "flagged_scaleup_not_claimable"
    assert gates["hardware"]["classification"] == "continuity_workload_receipt_no_speedup"
    assert "hardware_speedup" in summary["do_not_claim"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5349_missing_next_roadmap_blocks_without_repair(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5349-BLOCKED-NEXT-ROADMAP: missing next queue fails closed."""

    root = _make_repo(tmp_path, next_milestone=None, capstone=_capstone_payload())
    artifact = mod.build_artifact(
        root=root,
        run_date="20260707",
        duration_s=0.25,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    mod.validate_artifact(artifact)
    assert not (root / "research-roadmap-next.yaml").exists()
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["roadmap_next_present"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert "roadmap_next_missing" in artifact["failed_preconditions"]
    assert artifact["preconditions_checked"]["value"]["active_roadmap_milestone"] == "2026.07.488"


def test_req_report_5349_repository_artifact_matches_schema() -> None:
    """REQ-REPORT-5349: checked-in deliverable is a valid transition artifact."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == mod.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == mod.ACTIVATED_MILESTONE
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["archived_milestone"]["value"] == mod.ARCHIVED_MILESTONE
    assert artifact["activated_milestone"]["value"] == mod.ACTIVATED_MILESTONE
    assert artifact["roadmap_next_present"] is False
    assert artifact["milestone_doc_present"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False


def test_req_report_5349_helper_edges_and_validation_guards(tmp_path: Path) -> None:
    """REQ-REPORT-5349: helpers fail closed on malformed or contradictory data."""

    root = _make_repo(tmp_path / "repo", capstone=_capstone_payload())
    artifact = mod.build_artifact(
        root=root,
        run_date="20260707",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact({key: value for key, value in artifact.items() if key != "schema"})
    with pytest.raises(ValueError, match="schema"):
        mod.validate_artifact(artifact | {"schema": "wrong"})
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(artifact | {"field_principles": {}})
    with pytest.raises(ValueError, match="principle mismatch"):
        mod.validate_artifact(artifact | {"honest_verdict": _wrap("complete: fixture")})
    with pytest.raises(ValueError, match="missing value"):
        mod.validate_artifact(
            artifact | {"honest_verdict": {"principle": mod.FIELD_PRINCIPLES["honest_verdict"]}}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(
            artifact
            | {
                "honest_verdict": {
                    "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
                    "value": "done",
                }
            }
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(
            artifact
            | {
                "inference_substrate": {
                    "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
                    "value": "local_repo_doc_and_artifact_audit",
                }
            }
        )
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(
            artifact
            | {
                "experiment_id": {
                    "principle": mod.FIELD_PRINCIPLES["experiment_id"],
                    "value": "wrong",
                }
            }
        )
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(
            artifact
            | {
                "milestone": {
                    "principle": mod.FIELD_PRINCIPLES["milestone"],
                    "value": "2026.07.487",
                }
            }
        )
    with pytest.raises(ValueError, match="archived_milestone"):
        mod.validate_artifact(
            artifact
            | {
                "archived_milestone": {
                    "principle": mod.FIELD_PRINCIPLES["archived_milestone"],
                    "value": "2026.07.486",
                }
            }
        )
    with pytest.raises(ValueError, match="activated_milestone"):
        mod.validate_artifact(
            artifact
            | {
                "activated_milestone": {
                    "principle": mod.FIELD_PRINCIPLES["activated_milestone"],
                    "value": "2026.07.487",
                }
            }
        )
    with pytest.raises(ValueError, match="roadmap_next_present"):
        mod.validate_artifact(artifact | {"roadmap_next_present": "false"})
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.validate_artifact(artifact | {"active_roadmap_modified": True})
    with pytest.raises(ValueError, match="conductor_modified"):
        mod.validate_artifact(artifact | {"conductor_modified": True})
    with pytest.raises(ValueError, match="preconditions_checked"):
        mod.validate_artifact(
            artifact
            | {
                "preconditions_checked": {
                    "principle": mod.FIELD_PRINCIPLES["preconditions_checked"],
                    "value": "not-a-map",
                }
            }
        )
    with pytest.raises(ValueError, match="cited_upstream_artifacts"):
        mod.validate_artifact(
            artifact
            | {
                "cited_upstream_artifacts": {
                    "principle": mod.FIELD_PRINCIPLES["cited_upstream_artifacts"],
                    "value": "not-a-list",
                }
            }
        )
    with pytest.raises(ValueError, match="complete status"):
        mod.validate_artifact(artifact | {"failed_preconditions": ["still_bad"]})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "sha256:bad"})

    blocked_artifact = mod.build_artifact(
        root=_make_repo(tmp_path / "blocked", next_milestone=None, capstone=_capstone_payload()),
        run_date="20260707",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    with pytest.raises(ValueError, match="blocked status"):
        mod.validate_artifact(blocked_artifact | {"failed_preconditions": []})

    assert mod.value_of(_wrap("x")) == "x"
    assert mod.value_of("x") == "x"
    written = tmp_path / "written.json"
    mod.write_json(written, {"ok": True})
    assert json.loads(written.read_text(encoding="utf-8")) == {"ok": True}
    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_mapping(malformed)[1]["error"] == "malformed_json"
    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(array_json)[1]["error"] == "not_json_object"
    assert mod.yaml_milestone(tmp_path / "missing.yaml") is None
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("bad: [", encoding="utf-8")
    assert mod.yaml_milestone(bad_yaml) is None
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    assert mod.yaml_milestone(list_yaml) is None
    assert mod.document_contains_milestone(tmp_path / "missing.md", "2026.07.488") is False
    assert mod.git_path_modified(tmp_path, mod.ROADMAP_RELATIVE_PATH) is False
    git_repo = tmp_path / "git-repo"
    git_repo.mkdir()
    subprocess.run(("git", "init"), cwd=git_repo, check=True, capture_output=True, text=True)
    (git_repo / "research-roadmap.yaml").write_text("milestone: 2026.07.488\n", encoding="utf-8")
    assert mod.git_path_modified(git_repo, mod.ROADMAP_RELATIVE_PATH) is True
    assert mod._modification_status(tmp_path, mod.ROADMAP_RELATIVE_PATH, None) is False
    assert (
        mod._modification_status(
            tmp_path,
            mod.ROADMAP_RELATIVE_PATH,
            {str(mod.ROADMAP_RELATIVE_PATH): True},
        )
        is True
    )
    assert mod.path_sha256(tmp_path / "missing") is None
    assert mod.capstone_failures({}) == ["capstone_missing_or_unloadable"]
    bad_capstone_failures = mod.capstone_failures(
        _capstone_payload()
        | {
            "milestone": _wrap("2026.07.486"),
            "honest_verdict": _wrap("done"),
            "status": _wrap("blocked"),
        }
    )
    assert "capstone_milestone_expected_2026.07.487_observed_2026.07.486" in (
        bad_capstone_failures
    )
    assert "capstone_status_expected_complete_observed_blocked" in bad_capstone_failures
    assert "capstone_honest_verdict_missing_terminal_prefix" in bad_capstone_failures
    assert mod.capstone_truth_summary({"gate_table": "not-list"})["gate_claim_boundaries"] == []
    assert mod.capstone_truth_summary({"next_milestone_recommendation": "not-map"})[
        "do_not_claim"
    ] is None

    mismatched_next = mod.build_artifact(
        root=_make_repo(
            tmp_path / "mismatch",
            next_milestone="2026.07.487",
            capstone=_capstone_payload(),
        ),
        run_date="20260707",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert "roadmap_next_milestone_expected_2026.07.488_observed_2026.07.487" in (
        mismatched_next["failed_preconditions"]
    )

    _make_repo(
        tmp_path / "dirty",
        vnext_milestone="2026.07.487",
        capstone=_capstone_payload(),
    )
    (tmp_path / "dirty" / "research-roadmap.yaml").unlink()
    dirty_preconditions = mod.precondition_summary(
        root=tmp_path / "dirty",
        capstone_meta={"exists": True, "loadable": True},
        active_roadmap_modified=True,
        conductor_modified=True,
    )
    dirty_failures = mod.failed_preconditions([], dirty_preconditions)
    assert "milestone_doc_missing_or_mismatch_2026.07.488" in dirty_failures
    assert "active_roadmap_missing" in dirty_failures
    assert "active_roadmap_modified" in dirty_failures
    assert "conductor_modified" in dirty_failures

    no_capstone = mod.build_artifact(
        root=_make_repo(tmp_path / "no-capstone"),
        run_date="20260707",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert no_capstone["honest_verdict"]["value"].startswith("blocked_")
    assert "capstone_missing_or_unloadable" in no_capstone["failed_preconditions"]

    out = mod.run(
        root=_make_repo(tmp_path / "run-repo", capstone=_capstone_payload()),
        run_date="20260707",
        duration_s=1.0,
    )
    saved = json.loads(out.read_text(encoding="utf-8"))
    mod.validate_artifact(saved)
    assert saved["experiment_id"]["value"] == mod.EXPERIMENT_ID

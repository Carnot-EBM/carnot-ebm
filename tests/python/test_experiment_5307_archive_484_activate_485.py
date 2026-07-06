"""Tests for Exp 5307 archive .484 / .485 transition artifact.

Spec refs: REQ-REPORT-5307, SCENARIO-REPORT-5307,
SCENARIO-REPORT-5307-BLOCKED-PRECONDITIONS.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import yaml

from carnot import experiment_5307_archive_484_activate_485 as mod


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
                    "id": "exp5307-archive-484-activate-485",
                    "milestone": milestone,
                    "deliverable": str(mod.RESULT_RELATIVE_PATH),
                    "title": "fixture transition",
                    "agent_type": "codex",
                    "model": "gpt-5.5",
                    "prompt": "REQ-REPORT-5307 fixture",
                }
            ],
        },
        sort_keys=False,
    )


def _capstone_payload() -> dict[str, Any]:
    return {
        "experiment_id": "exp5306-capstone-v484",
        "milestone": "2026.07.484",
        "honest_verdict": _wrap(
            "complete: .484 closed with changed-runtime SOTA still blocked and quality "
            "unmeasured, adaptive memory/self-learning cleanly positive, solver/certificate "
            "tracks bounded with LNS and p-bit gates but quarantined EBT telemetry and null "
            "certificate-success lift, and hardware reachability-only with no speedup."
        ),
        "changed_runtime_outcome": _wrap(
            {
                "changed_runtime_sota_ready": False,
                "sota_quality_measured": False,
                "no_quality_claim": True,
            }
        ),
        "continuous_self_learning_outcome": _wrap(
            {
                "adaptive_memory_policy_positive": True,
                "memory_stress_passed": True,
                "unsafe_false_accepts": 0,
            }
        ),
        "solver_energy_certificate_outcome": _wrap(
            {
                "constraint_lns_fixture_ready": True,
                "pbit_gate_ready": True,
                "spectral_control": {"quarantined": True, "headline_eligible": False},
                "kan_dynamic_abstraction": {
                    "diagnostic_tightness_helped": True,
                    "certificate_success_improvement": 0.0,
                },
            }
        ),
        "hardware_speedup_claimed": _wrap(False),
        "hardware_status": {
            "kv260_status": {"status": "blocked_kv260_ssh_unreachable"},
            "polarfire_status": {"status": "reachable_ssh_status_only"},
            "gatemate_status": {"status": "blocked_gatemate_physical_jtag_setup_unchanged"},
        },
        "tasks_summarized": _wrap(
            {
                "expected_count": 11,
                "loadable_count": 11,
                "by_classification": {
                    "clean_positive": 4,
                    "clean_null": 1,
                    "blocked_precondition": 2,
                    "gated_skip": 1,
                    "mixed_positive_with_harmful_class": 1,
                    "quarantined": 2,
                },
            }
        ),
    }


def _make_repo(
    root: Path,
    *,
    active_milestone: str = "2026.07.485",
    vnext_milestone: str = "2026.07.485",
    next_milestone: str | None = None,
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
        (root / "ops" / relative).write_text("fixture 2026.07.485\n", encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    return root


def test_req_report_5307_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5307: OpenSpec anchors the .484 archive and .485 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5307") : spec.index("REQ-REPORT-5296")]

    for marker in (
        "REQ-REPORT-5307",
        "SCENARIO-REPORT-5307",
        "SCENARIO-REPORT-5307-BLOCKED-PRECONDITIONS",
        str(mod.RESULT_RELATIVE_PATH),
        "local_repo_doc_and_artifact_audit",
        "`roadmap_next_present`",
        "`active_roadmap_modified`",
        "`conductor_modified`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5307_active_485_records_no_overwrite(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5307: active .485 emits a complete no-overwrite artifact."""

    root = _make_repo(tmp_path, capstone=_capstone_payload())
    roadmap_before = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    conductor_before = (root / "scripts/research_conductor.py").read_text(encoding="utf-8")

    artifact = mod.build_artifact(
        root=root,
        run_date="20260706",
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
    assert artifact["v484_capstone_verdict"]["value"].startswith("complete: .484 closed")
    assert artifact["roadmap_next_present"] is False
    assert artifact["milestone_doc_present"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert (
        artifact["preconditions_checked"]["value"]["active_roadmap_names_activated_milestone"]
        is True
    )
    assert (
        artifact["preconditions_checked"]["value"]["roadmap_next_absent_after_activation"] is True
    )
    assert artifact["v484_capstone_proves"]["value"]["changed_runtime_sota_ready"] is False
    assert artifact["v484_capstone_proves"]["value"]["sota_quality_measured"] is False
    assert artifact["v484_capstone_proves"]["value"]["adaptive_memory_positive"] is True
    assert artifact["v484_capstone_proves"]["value"]["ebt_telemetry_quarantined"] is True
    assert artifact["v484_capstone_proves"]["value"]["certificate_success_lift"] == 0.0
    assert artifact["v484_capstone_proves"]["value"]["hardware_speedup_claimed"] is False
    assert artifact["failed_preconditions"] == []
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5307_present_next_roadmap_is_recorded(tmp_path: Path) -> None:
    """REQ-REPORT-5307: present research-roadmap-next.yaml is audited, not copied."""

    artifact = mod.build_artifact(
        root=_make_repo(tmp_path, next_milestone="2026.07.485", capstone=_capstone_payload()),
        run_date="20260706",
        duration_s=0.25,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )

    mod.validate_artifact(artifact)
    assert artifact["roadmap_next_present"] is True
    assert (
        artifact["preconditions_checked"]["value"]["roadmap_next_names_activated_milestone"] is True
    )
    assert (
        artifact["preconditions_checked"]["value"]["no_active_roadmap_overwrite_performed"] is True
    )


def test_scenario_report_5307_blocked_preconditions_keep_failures_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5307-BLOCKED-PRECONDITIONS: bad inputs emit blocked_."""

    bad_capstone = _capstone_payload() | {
        "changed_runtime_outcome": _wrap({"changed_runtime_sota_ready": True}),
        "hardware_speedup_claimed": _wrap(True),
    }
    artifact = mod.build_artifact(
        root=_make_repo(
            tmp_path,
            active_milestone="2026.07.484",
            vnext_milestone="2026.07.484",
            capstone=bad_capstone,
        ),
        run_date="20260706",
        duration_s=0.25,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: True,
            mod.CONDUCTOR_RELATIVE_PATH: True,
        },
    )

    mod.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["active_roadmap_modified"] is True
    assert artifact["conductor_modified"] is True
    assert (
        "capstone_changed_runtime_sota_ready_expected_False_observed_True"
        in artifact["failed_preconditions"]
    )
    assert (
        "capstone_sota_quality_measured_expected_False_observed_None"
        in artifact["failed_preconditions"]
    )
    assert (
        "capstone_hardware_speedup_claimed_expected_False_observed_True"
        in artifact["failed_preconditions"]
    )
    assert "milestone_doc_missing_2026.07.485" in artifact["failed_preconditions"]
    assert "active_or_next_roadmap_not_ready_for_2026.07.485" in artifact["failed_preconditions"]
    assert "active_roadmap_modified" in artifact["failed_preconditions"]
    assert "conductor_modified" in artifact["failed_preconditions"]


def test_req_report_5307_repository_artifact_matches_schema() -> None:
    """REQ-REPORT-5307: checked-in deliverable is a valid transition artifact."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == mod.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == mod.ACTIVATED_MILESTONE
    assert artifact["honest_verdict"]["value"].startswith(("complete:", "blocked_"))
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["archived_milestone"]["value"] == mod.ARCHIVED_MILESTONE
    assert artifact["activated_milestone"]["value"] == mod.ACTIVATED_MILESTONE
    assert artifact["milestone_doc_present"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False


def test_req_report_5307_helper_edges_and_validation_guards(tmp_path: Path) -> None:
    """REQ-REPORT-5307: helpers fail closed on malformed or contradictory data."""

    root = _make_repo(tmp_path / "repo", capstone=_capstone_payload())
    artifact = mod.build_artifact(
        root=root,
        run_date="20260706",
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
                    "value": "aggregation_from_upstream_artifacts",
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
                    "value": "2026.07.484",
                }
            }
        )
    with pytest.raises(ValueError, match="archived_milestone"):
        mod.validate_artifact(
            artifact
            | {
                "archived_milestone": {
                    "principle": mod.FIELD_PRINCIPLES["archived_milestone"],
                    "value": "2026.07.483",
                }
            }
        )
    with pytest.raises(ValueError, match="activated_milestone"):
        mod.validate_artifact(
            artifact
            | {
                "activated_milestone": {
                    "principle": mod.FIELD_PRINCIPLES["activated_milestone"],
                    "value": "2026.07.486",
                }
            }
        )
    with pytest.raises(ValueError, match="roadmap_next_present"):
        mod.validate_artifact(artifact | {"roadmap_next_present": "false"})
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.validate_artifact(artifact | {"active_roadmap_modified": "false"})
    with pytest.raises(ValueError, match="conductor_modified"):
        mod.validate_artifact(artifact | {"conductor_modified": "false"})
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
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "sha256:bad"})

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
    assert mod.git_path_modified(tmp_path, mod.ROADMAP_RELATIVE_PATH) is False
    git_repo = tmp_path / "git-repo"
    git_repo.mkdir()
    subprocess.run(("git", "init"), cwd=git_repo, check=True, capture_output=True, text=True)
    (git_repo / "research-roadmap.yaml").write_text("milestone: 2026.07.485\n", encoding="utf-8")
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
            "milestone": "2026.07.483",
            "honest_verdict": _wrap("done"),
        }
    )
    assert "capstone_milestone_expected_2026.07.484_observed_2026.07.483" in bad_capstone_failures
    assert "capstone_honest_verdict_missing_terminal_prefix" in bad_capstone_failures

    no_capstone = mod.build_artifact(
        root=_make_repo(tmp_path / "no-capstone"),
        run_date="20260706",
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
        run_date="20260706",
        duration_s=1.0,
    )
    saved = json.loads(out.read_text(encoding="utf-8"))
    mod.validate_artifact(saved)
    assert saved["experiment_id"]["value"] == mod.EXPERIMENT_ID

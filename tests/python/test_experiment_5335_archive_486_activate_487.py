"""Tests for Exp 5335 archive .486 / .487 transition artifact.

Spec refs: REQ-REPORT-5335, SCENARIO-REPORT-5335,
SCENARIO-REPORT-5335-BLOCKED-NEXT-ROADMAP.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import yaml

from carnot import experiment_5335_archive_486_activate_487 as mod


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
                    "id": "exp5335-archive-486-activate-487",
                    "milestone": milestone,
                    "deliverable": str(mod.RESULT_RELATIVE_PATH),
                    "title": "fixture transition",
                    "agent_type": "codex",
                    "model": "gpt-5.5",
                    "prompt": "REQ-REPORT-5335 fixture",
                }
            ],
        },
        sort_keys=False,
    )


def _capstone_payload() -> dict[str, Any]:
    return {
        "experiment_id": _wrap("exp5334-capstone-v486"),
        "milestone": _wrap("2026.07.486"),
        "status": _wrap("complete"),
        "honest_verdict": _wrap(
            "complete: .486 synthesized with runtime_stable=True, "
            "sota_quality_measured=True, rewrite_smt_self_learning_clean=True, "
            "internal_signal=open_but_flagged, kan_localization_ready=True, "
            "hardware_speedup_claim=false"
        ),
        "inference_substrate": _wrap("artifact_synthesis_and_gate_reconciliation"),
        "runtime_stable": True,
        "sota_quality_measured": True,
        "rewrite_state_ready": True,
        "smt_corrigendum_clean": True,
        "context_lifecycle_ready": True,
        "certificate_self_learning_ready": True,
        "internal_signal_path_open": True,
        "kan_localization_ready": True,
        "hardware_speedup_claim": False,
        "gate_table": _wrap(
            [
                {
                    "gate": "runtime",
                    "ready": True,
                    "classification": "stable_runtime_no_quality_claim",
                    "claim_boundary": "runtime stability only; no quality claim",
                    "source_experiments": [5323, 5324],
                },
                {
                    "gate": "sota_quality",
                    "ready": True,
                    "classification": "bounded_smoke_measured_no_headline_claim",
                    "claim_boundary": "bounded fixture-scored smoke; no headline quality claim",
                    "source_experiments": [5326],
                },
                {
                    "gate": "internal_signal_receipts",
                    "ready": True,
                    "classification": "open_but_flagged",
                    "claim_boundary": "token-probability receipt path only",
                    "source_experiments": [5331],
                },
                {
                    "gate": "hardware",
                    "ready": False,
                    "classification": "reachability_only_no_speedup",
                    "claim_boundary": "reachability receipts only; no authenticated workload or speedup claim",
                    "source_experiments": [5333],
                },
            ]
        ),
        "next_milestone_recommendation": _wrap(
            {
                "do_not_claim": [
                    "hardware_speedup",
                    "headline_sota_quality",
                    "broad_kan_certificate",
                ]
            }
        ),
    }


def _make_repo(
    root: Path,
    *,
    active_milestone: str = "2026.07.487",
    vnext_milestone: str = "2026.07.487",
    next_milestone: str | None = "2026.07.487",
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
        (root / "ops" / relative).write_text("fixture 2026.07.487\n", encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    return root


def test_req_report_5335_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5335: OpenSpec anchors the .486 archive and .487 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5335") : spec.index("REQ-REPORT-5109")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5335",
        "SCENARIO-REPORT-5335",
        "SCENARIO-REPORT-5335-BLOCKED-NEXT-ROADMAP",
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


def test_scenario_report_5335_present_next_roadmap_records_no_overwrite(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5335: present .487 roadmap-next emits complete no-overwrite artifact."""

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
    assert artifact["v486_capstone_verdict"]["value"].startswith("complete: .486 synthesized")
    assert artifact["roadmap_next_present"] is True
    assert artifact["milestone_doc_present"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["failed_preconditions"] == []

    preconditions = artifact["preconditions_checked"]["value"]
    assert preconditions["roadmap_next_names_activated_milestone"] is True
    assert preconditions["no_active_roadmap_overwrite_performed"] is True
    assert preconditions["no_conductor_edit_performed"] is True

    summary = artifact["v486_capstone_proves"]["value"]
    assert summary["runtime_stable"] is True
    assert summary["sota_quality_measured"] is True
    assert summary["internal_signal_path_open"] is True
    assert summary["hardware_speedup_claim"] is False
    gates = {row["gate"]: row for row in summary["gate_claim_boundaries"]}
    assert gates["runtime"]["claim_boundary"] == "runtime stability only; no quality claim"
    assert gates["sota_quality"]["classification"] == "bounded_smoke_measured_no_headline_claim"
    assert gates["internal_signal_receipts"]["classification"] == "open_but_flagged"
    assert gates["hardware"]["ready"] is False
    assert "hardware_speedup" in summary["do_not_claim"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5335_missing_next_roadmap_blocks_without_repair(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5335-BLOCKED-NEXT-ROADMAP: missing next queue fails closed."""

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
    assert artifact["preconditions_checked"]["value"]["active_roadmap_milestone"] == "2026.07.487"


def test_req_report_5335_repository_artifact_matches_schema() -> None:
    """REQ-REPORT-5335: checked-in deliverable is a valid transition artifact."""

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
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False


def test_req_report_5335_helper_edges_and_validation_guards(tmp_path: Path) -> None:
    """REQ-REPORT-5335: helpers fail closed on malformed or contradictory data."""

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
                    "value": "2026.07.486",
                }
            }
        )
    with pytest.raises(ValueError, match="archived_milestone"):
        mod.validate_artifact(
            artifact
            | {
                "archived_milestone": {
                    "principle": mod.FIELD_PRINCIPLES["archived_milestone"],
                    "value": "2026.07.485",
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
    with pytest.raises(ValueError, match="complete status"):
        mod.validate_artifact(artifact | {"failed_preconditions": ["still_bad"]})
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
    assert mod.document_contains_milestone(tmp_path / "missing.md", "2026.07.487") is False
    assert mod.git_path_modified(tmp_path, mod.ROADMAP_RELATIVE_PATH) is False
    git_repo = tmp_path / "git-repo"
    git_repo.mkdir()
    subprocess.run(("git", "init"), cwd=git_repo, check=True, capture_output=True, text=True)
    (git_repo / "research-roadmap.yaml").write_text("milestone: 2026.07.487\n", encoding="utf-8")
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
            "milestone": _wrap("2026.07.485"),
            "honest_verdict": _wrap("done"),
            "status": _wrap("blocked"),
        }
    )
    assert "capstone_milestone_expected_2026.07.486_observed_2026.07.485" in (
        bad_capstone_failures
    )
    assert "capstone_status_expected_complete_observed_blocked" in bad_capstone_failures
    assert "capstone_honest_verdict_missing_terminal_prefix" in bad_capstone_failures

    mismatched_next = mod.build_artifact(
        root=_make_repo(
            tmp_path / "mismatch",
            next_milestone="2026.07.486",
            capstone=_capstone_payload(),
        ),
        run_date="20260707",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert "roadmap_next_milestone_expected_2026.07.487_observed_2026.07.486" in (
        mismatched_next["failed_preconditions"]
    )

    _make_repo(
        tmp_path / "dirty",
        vnext_milestone="2026.07.486",
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
    assert "milestone_doc_missing_or_mismatch_2026.07.487" in dirty_failures
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

"""Tests for Exp 5363 .489 transition artifact.

Spec refs: REQ-REPORT-5363, SCENARIO-REPORT-5363,
SCENARIO-REPORT-5363-BLOCKED-INPUT.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import yaml

from carnot import experiment_5363_transition_v489 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _roadmap(milestone: str, task_ids: list[str] | None = None) -> str:
    tasks = [
        {
            "id": task_id,
            "milestone": milestone,
            "deliverable": f"results/{task_id}.json",
            "title": f"fixture {task_id}",
            "agent_type": "codex",
            "model": "gpt-5.5",
            "prompt": "REQ-REPORT-5363 fixture",
        }
        for task_id in (task_ids or mod.EXPECTED_TASK_IDS)
    ]
    return yaml.safe_dump(
        {
            "milestone": milestone,
            "milestone_title": f"fixture {milestone}",
            "milestone_doc": str(mod.VNEXT_RELATIVE_PATH),
            "tasks": tasks,
        },
        sort_keys=False,
    )


def _vnext_doc(milestone: str = mod.MILESTONE) -> str:
    return f"""# Research Roadmap vNEXT - Milestone {milestone}

**Milestone title:** fixture
**Task range:** Exp 5363-5375
**Pre-staged roadmap:** `research-roadmap-next.yaml`

## Phase Plan

### Phase 0 - Transition and Fresh Source Delta

### Phase 1 - Grammar-Budgeted Structured SOTA

### Phase 2 - Budget-Curated Continuous Self-Learning

### Phase 3 - Solver Guidance, Internal-Feature Preconditions, ARC, and Hardware

### Phase 4 - Capstone
"""


def _capstone_payload() -> dict[str, Any]:
    return {
        "experiment_id": _wrap("exp5362-capstone-v488"),
        "milestone": _wrap(mod.PRIOR_MILESTONE),
        "status": _wrap("complete"),
        "honest_verdict": _wrap("complete: .488 fixture capstone"),
        "structured_protocol_clean": False,
        "constraint_tax_panel_ready": False,
        "tokenprob_feature_rows_ready": True,
        "carry_token_energy_signal_ready": False,
        "dependency_provenance_ready": True,
        "memory_tool_drift_ready": True,
        "self_learning_scaleup_ready": True,
        "solver_projection_ready": True,
        "pbit_schedule_signal_ready": True,
        "arc_new_level_banked": False,
        "hardware_speedup_claim": False,
    }


def _make_repo(
    root: Path,
    *,
    active_milestone: str = mod.MILESTONE,
    doc_milestone: str = mod.MILESTONE,
    next_milestone: str | None = None,
    capstone: dict[str, Any] | None = None,
    task_ids: list[str] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("CODEX.md", "CLAUDE.md"):
        (root / relative).write_text("fixture\n", encoding="utf-8")
    (root / mod.ROADMAP_RELATIVE_PATH).write_text(
        _roadmap(active_milestone, task_ids),
        encoding="utf-8",
    )
    if next_milestone is not None:
        (root / mod.ROADMAP_NEXT_RELATIVE_PATH).write_text(
            _roadmap(next_milestone, task_ids),
            encoding="utf-8",
        )
    (root / mod.VNEXT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(_vnext_doc(doc_milestone), encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    for relative in ("status.md", "conductor-log.md"):
        (root / "ops" / relative).write_text("fixture 2026.07.489\n", encoding="utf-8")
    (root / mod.CONDUCTOR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_RELATIVE_PATH).write_text("# fixture\n", encoding="utf-8")
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    return root


def test_req_report_5363_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5363: OpenSpec anchors the .488 archive and .489 transition."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5363") : spec.index("REQ-REPORT-5336")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5363",
        "SCENARIO-REPORT-5363",
        "SCENARIO-REPORT-5363-BLOCKED-INPUT",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`roadmap_next_present=false`",
        "`active_roadmap_modified=false`",
        "`conductor_modified=false`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5363_already_active_records_transition_context(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5363: already-active .489 records complete transition context."""

    root = _make_repo(tmp_path, capstone=_capstone_payload())
    roadmap_before = (root / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
    conductor_before = (root / mod.CONDUCTOR_RELATIVE_PATH).read_text(encoding="utf-8")

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
    assert (root / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8") == roadmap_before
    assert (root / mod.CONDUCTOR_RELATIVE_PATH).read_text(encoding="utf-8") == conductor_before
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["prior_milestone"] == mod.PRIOR_MILESTONE
    assert artifact["prior_capstone_path"] == str(mod.CAPSTONE_RELATIVE_PATH)
    assert artifact["prior_gate_summary"] == mod.extract_prior_gate_summary(_capstone_payload())
    assert artifact["roadmap_next_present"] is False
    assert artifact["roadmap_doc_present"] is True
    assert artifact["planned_task_source"] == str(mod.ROADMAP_RELATIVE_PATH)
    assert artifact["planned_task_count"] == 13
    assert artifact["planned_task_ids"] == mod.EXPECTED_TASK_IDS
    assert artifact["planned_phase_names"] == mod.EXPECTED_PHASE_NAMES
    assert artifact["roadmap_doc_task_range"] == "Exp 5363-5375"
    assert set(artifact["downstream_gate_expectations"]) == {
        "structured",
        "self_learning",
        "solver",
        "token",
        "ARC",
        "hardware",
    }
    assert artifact["downstream_gate_expectations"]["structured"]["clean_gate"] == (
        "parse_success_rate>=0.95, schema_success_rate>=0.90, "
        "final_json_extraction_rate>=0.95, unsafe_false_accepts=0, "
        "methodology_duration_s>=60"
    )
    assert artifact["downstream_gate_expectations"]["hardware"]["speedup_claim_allowed"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["failed_preconditions"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_report_5363_present_next_roadmap_prefers_literal_source(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5363: present .489 roadmap-next is recorded as the task source."""

    root = _make_repo(tmp_path, next_milestone=mod.MILESTONE, capstone=_capstone_payload())
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
    assert artifact["status"] == "complete"
    assert artifact["roadmap_next_present"] is True
    assert artifact["planned_task_source"] == str(mod.ROADMAP_NEXT_RELATIVE_PATH)
    assert artifact["planned_task_count"] == 13
    assert artifact["planned_task_ids"] == mod.EXPECTED_TASK_IDS


def test_scenario_report_5363_missing_required_source_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5363-BLOCKED-INPUT: missing capstone fails closed."""

    root = _make_repo(tmp_path)
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
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["prior_gate_summary"] == mod.empty_prior_gate_summary()
    assert "capstone_missing_or_unloadable" in artifact["failed_preconditions"]
    assert artifact["roadmap_next_present"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False


def test_req_report_5363_repository_artifact_matches_schema() -> None:
    """REQ-REPORT-5363: checked-in deliverable is a valid transition artifact."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["prior_milestone"] == mod.PRIOR_MILESTONE
    assert artifact["roadmap_next_present"] is False
    assert artifact["roadmap_doc_present"] is True
    assert artifact["planned_task_count"] == 13
    assert artifact["planned_task_ids"] == mod.EXPECTED_TASK_IDS
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False


def test_req_report_5363_helper_edges_and_validation_guards(tmp_path: Path) -> None:
    """REQ-REPORT-5363: helpers fail closed on malformed or contradictory data."""

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
        mod.validate_artifact({key: value for key, value in artifact.items() if key != "status"})
    with pytest.raises(ValueError, match="schema"):
        mod.validate_artifact(artifact | {"schema": "wrong"})
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(artifact | {"field_principles": {}})
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(artifact | {"status": "pending"})
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(artifact | {"milestone": mod.PRIOR_MILESTONE})
    with pytest.raises(ValueError, match="prior_milestone"):
        mod.validate_artifact(artifact | {"prior_milestone": mod.MILESTONE})
    with pytest.raises(ValueError, match="prior_capstone_path"):
        mod.validate_artifact(artifact | {"prior_capstone_path": "wrong.json"})
    with pytest.raises(ValueError, match="prior_gate_summary"):
        mod.validate_artifact(artifact | {"prior_gate_summary": {}})
    with pytest.raises(ValueError, match="roadmap_next_present"):
        mod.validate_artifact(artifact | {"roadmap_next_present": "false"})
    with pytest.raises(ValueError, match="planned_task_count"):
        mod.validate_artifact(artifact | {"planned_task_count": 12})
    with pytest.raises(ValueError, match="planned_task_ids"):
        mod.validate_artifact(artifact | {"planned_task_ids": ["wrong"]})
    with pytest.raises(ValueError, match="downstream_gate_expectations"):
        mod.validate_artifact(artifact | {"downstream_gate_expectations": {}})
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.validate_artifact(artifact | {"active_roadmap_modified": True})
    with pytest.raises(ValueError, match="conductor_modified"):
        mod.validate_artifact(artifact | {"conductor_modified": True})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "done"})
    with pytest.raises(ValueError, match="complete status"):
        mod.validate_artifact(artifact | {"failed_preconditions": ["still_bad"]})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "sha256:bad"})

    blocked = mod.build_artifact(
        root=_make_repo(tmp_path / "blocked"),
        run_date="20260707",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    with pytest.raises(ValueError, match="blocked status"):
        mod.validate_artifact(blocked | {"failed_preconditions": []})

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
    assert mod.read_yaml_mapping(tmp_path / "missing.yaml")[1]["error"] == "missing"
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("bad: [", encoding="utf-8")
    assert mod.read_yaml_mapping(bad_yaml)[1]["error"] == "malformed_yaml"
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    assert mod.read_yaml_mapping(list_yaml)[1]["error"] == "not_yaml_object"
    assert mod.document_contains_milestone(tmp_path / "missing.md", mod.MILESTONE) is False
    assert mod.extract_phase_names("no phases") == []
    assert mod.extract_task_range("no range") is None
    assert mod.extract_roadmap_tasks({"tasks": "bad"}) == []
    assert mod.extract_roadmap_tasks({"tasks": [{"id": "x"}, "bad"]}) == ["x"]
    assert mod.empty_prior_gate_summary() == {
        field: None for field in mod.PRIOR_GATE_FIELDS
    }
    assert mod.path_sha256(tmp_path / "missing") is None
    assert mod.git_path_modified(tmp_path, mod.ROADMAP_RELATIVE_PATH) is False
    git_repo = tmp_path / "git-repo"
    git_repo.mkdir()
    subprocess.run(("git", "init"), cwd=git_repo, check=True, capture_output=True, text=True)
    (git_repo / mod.ROADMAP_RELATIVE_PATH).write_text("milestone: 2026.07.489\n", encoding="utf-8")
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

    wrong_doc = mod.build_artifact(
        root=_make_repo(tmp_path / "wrong-doc", doc_milestone="2026.07.488", capstone=_capstone_payload()),
        run_date="20260707",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert "roadmap_doc_missing_or_mismatch_2026.07.489" in wrong_doc["failed_preconditions"]

    wrong_active = mod.build_artifact(
        root=_make_repo(
            tmp_path / "wrong-active",
            active_milestone="2026.07.488",
            capstone=_capstone_payload(),
        ),
        run_date="20260707",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert "active_roadmap_milestone_expected_2026.07.489_observed_2026.07.488" in (
        wrong_active["failed_preconditions"]
    )

    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.build_artifact(
            root=_make_repo(tmp_path / "dirty", capstone=_capstone_payload()),
            run_date="20260707",
            duration_s=1.0,
            modification_status={
                mod.ROADMAP_RELATIVE_PATH: True,
                mod.CONDUCTOR_RELATIVE_PATH: True,
            },
        )

    out = mod.run(
        root=_make_repo(tmp_path / "run-repo", capstone=_capstone_payload()),
        run_date="20260707",
        duration_s=1.0,
    )
    saved = json.loads(out.read_text(encoding="utf-8"))
    mod.validate_artifact(saved)
    assert saved["milestone"] == mod.MILESTONE

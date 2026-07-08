"""Tests for Exp 5389 .491 transition artifact.

Spec refs: REQ-REPORT-5389, SCENARIO-REPORT-5389,
SCENARIO-REPORT-5389-BLOCKED-INPUT.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest
import yaml

from carnot import experiment_5389_transition_v491 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
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
            "prompt": "REQ-REPORT-5389 fixture",
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
**Previous milestone:** {mod.PRIOR_MILESTONE}
**Task range:** Exp 5389-5401
**Pre-staged roadmap:** `research-roadmap-next.yaml`

## Phase Plan

### Phase 0 - Source Delta and Expanded Local Fixtures

### Phase 1 - Solver Corrigendum and Gated P-bit Ablation

### Phase 2 - Continuous Self-Learning and Live ARC Salience

### Phase 3 - Evidence Surfaces, Certificates, and Capstone
"""


def _capstone_payload() -> dict[str, Any]:
    return {
        "experiment_id": "exp5388-capstone-v490",
        "milestone": _wrap(mod.PRIOR_MILESTONE),
        "status": _wrap("complete"),
        "honest_verdict": _wrap(
            "complete: .490 proved clean structured receipts and real self-learning; "
            "solver guidance is flagged, ARC banked no level, hardware has no speedup, "
            "and token/backend signal stays closed."
        ),
        "structured_methodology_receipt_ready": True,
        "structured_protocol_clean": True,
        "constraint_tax_panel_ready": True,
        "budget_memory_corrigendum_clean": True,
        "continuous_self_learning_real_workflow_ready": True,
        "continuous_self_learning_requirement_satisfied": True,
        "overwrite_guidance_scale_ready": True,
        "pbit_boundary_overwrite_ready": True,
        "arc_new_level_banked": False,
        "hardware_hash_chained_receipt_ready": True,
        "hardware_speedup_claim": False,
        "future_token_signal_allowed": False,
        "phase_summaries": [
            {
                "lane": "solver_guidance",
                "outcome": "ready_flagged",
                "evidence": {
                    "flagged_adversarial": True,
                    "overwrite_guidance_scale_ready": True,
                    "solver_authoritative": True,
                    "unsafe_false_accepts": 0,
                    "corrigendum_pending": [
                        {
                            "kind": "TAUTOLOGY",
                            "severity": "critical",
                            "detail": "forced_hint_harm_rate and validity rate matched",
                        }
                    ],
                },
            },
            {
                "lane": "arc_geometric_salience",
                "outcome": "honest_null_no_level_banked",
                "evidence": {
                    "failure_mode": "bounded_budget_no_levelup",
                    "live_attempt_count": 1,
                    "new_level_banked": False,
                    "offline_reproduced": False,
                    "solve_provenance": "live_agent_self_discovery",
                },
            },
            {
                "lane": "hardware",
                "outcome": "receipt_ready_no_speedup",
                "evidence": {
                    "hardware_hash_chained_receipt_ready": True,
                    "hardware_speedup_claim": False,
                    "repeatability_evidence_present": False,
                    "kv260_status": {
                        "status": "unreachable",
                        "ssh_reachable": False,
                    },
                    "polar_fire_status": {"status": "reachable/workload_receipt"},
                    "gatemate_status": {"status": "blocked_physical_or_jtag"},
                },
            },
            {
                "lane": "token_backend",
                "outcome": "closed_no_backend_signal",
                "evidence": {
                    "future_signal_allowed": False,
                    "backend_reopen_allowed": False,
                    "no_live_signal_claim": True,
                    "logits_available": False,
                    "hidden_states_available": False,
                    "attention_available": False,
                    "intermediate_depth_exits_available": False,
                },
            },
        ],
    }


def _make_repo(
    root: Path,
    *,
    active_milestone: str = mod.MILESTONE,
    doc_milestone: str = mod.MILESTONE,
    next_milestone: str | None = None,
    capstone: dict[str, Any] | None = None,
    task_ids: list[str] | None = None,
    conductor_log: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("AGENTS.md", "CODEX.md", "CLAUDE.md"):
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
    (root / "ops/status.md").write_text("fixture status\n", encoding="utf-8")
    (root / "ops/changelog.md").write_text("fixture changelog\n", encoding="utf-8")
    (root / "ops/conductor-log.md").write_text(
        conductor_log
        or "| 2026-07-08 01:27 UTC | Exp5383 | FLAGGED | "
        "adversarial_verify CRITICAL: TAUTOLOGY |\n",
        encoding="utf-8",
    )
    (root / mod.CONDUCTOR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_RELATIVE_PATH).write_text("# fixture\n", encoding="utf-8")
    if capstone is not None:
        _write_json(root / mod.CAPSTONE_RELATIVE_PATH, capstone)
    return root


def test_req_report_5389_spec_declares_transition_contract() -> None:
    """REQ-REPORT-5389: OpenSpec anchors the .491 transition artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5389") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5389",
        "SCENARIO-REPORT-5389",
        "SCENARIO-REPORT-5389-BLOCKED-INPUT",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.CAPSTONE_RELATIVE_PATH),
        "Exp5383 was adversarially flagged",
        "`roadmap_next_present=false`",
        "`active_roadmap_modified=false`",
        "`conductor_modified=false`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5389_active_491_records_transition_context(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5389: active .491 records complete transition context."""

    root = _make_repo(tmp_path, capstone=_capstone_payload())
    roadmap_before = (root / mod.ROADMAP_RELATIVE_PATH).read_text(encoding="utf-8")
    conductor_before = (root / mod.CONDUCTOR_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(
        root=root,
        run_date="20260708",
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
    assert artifact["roadmap_doc_task_range"] == "Exp 5389-5401"
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["failed_preconditions"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    blockers = artifact["prior_blockers"]
    assert blockers["solver_tautology"]["critical_tautology_flagged"] is True
    assert blockers["solver_tautology"]["conductor_flagged_tautology"] is True
    assert blockers["solver_tautology"]["corrigendum_required"] is True
    assert blockers["ARC"]["arc_new_level_banked"] is False
    assert blockers["ARC"]["failure_mode"] == "bounded_budget_no_levelup"
    assert blockers["token_feature"]["future_token_signal_allowed"] is False
    assert blockers["token_feature"]["backend_reopen_allowed"] is False
    assert blockers["hardware"]["hardware_speedup_claim"] is False
    assert blockers["hardware"]["kv260_reachability"] == "unreachable"

    expectations = artifact["downstream_gate_expectations"]
    assert set(expectations) == {
        "structured",
        "self_learning",
        "solver",
        "ARC",
        "token",
        "hardware",
    }
    assert expectations["structured"]["requires_prior_structured_protocol_clean"] is True
    assert expectations["self_learning"]["no_model_weight_mutation_required"] is True
    assert expectations["solver"]["requires_exp5393_corrigendum_clean_before_exp5394"] is True
    assert expectations["ARC"]["prior_arc_new_level_banked"] is False
    assert expectations["token"]["future_token_signal_allowed_from_prior"] is False
    assert expectations["hardware"]["speedup_claim_allowed_from_prior"] is False


def test_scenario_report_5389_present_next_roadmap_prefers_literal_source(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5389: present .491 roadmap-next is recorded as the task source."""

    root = _make_repo(tmp_path, next_milestone=mod.MILESTONE, capstone=_capstone_payload())
    artifact = mod.build_artifact(
        root=root,
        run_date="20260708",
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


def test_scenario_report_5389_missing_required_source_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5389-BLOCKED-INPUT: missing capstone fails closed."""

    root = _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=root,
        run_date="20260708",
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


def test_req_report_5389_repository_artifact_matches_schema() -> None:
    """REQ-REPORT-5389: checked-in deliverable is a valid transition artifact."""

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
    assert artifact["prior_blockers"]["solver_tautology"]["critical_tautology_flagged"] is True
    assert artifact["prior_blockers"]["hardware"]["kv260_reachability"] == "unreachable"


def test_req_report_5389_helper_edges_and_validation_guards(tmp_path: Path) -> None:
    """REQ-REPORT-5389: helpers fail closed on malformed or contradictory data."""

    root = _make_repo(tmp_path / "repo", capstone=_capstone_payload())
    artifact = mod.build_artifact(
        root=root,
        run_date="20260708",
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
    with pytest.raises(ValueError, match="prior_blockers"):
        mod.validate_artifact(artifact | {"prior_blockers": {}})
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
        run_date="20260708",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    with pytest.raises(ValueError, match="blocked status"):
        mod.validate_artifact(blocked | {"failed_preconditions": []})

    assert mod.value_of(_wrap(False)) is False
    assert mod.value_of(True) is True
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
    assert mod.extract_phase_names("no phases") == []
    assert mod.extract_task_range("no range") is None
    assert mod.extract_roadmap_tasks({"tasks": "bad"}) == []
    assert mod.extract_roadmap_tasks({"tasks": [{"id": "x"}, "bad"]}) == ["x"]
    assert mod.empty_prior_gate_summary() == {field: None for field in mod.PRIOR_GATE_FIELDS}
    assert mod.path_sha256(tmp_path / "missing") is None
    assert mod.git_path_modified(tmp_path, mod.ROADMAP_RELATIVE_PATH) is False
    git_repo = tmp_path / "git-repo"
    git_repo.mkdir()
    subprocess.run(("git", "init"), cwd=git_repo, check=True, capture_output=True, text=True)
    (git_repo / mod.ROADMAP_RELATIVE_PATH).write_text("milestone: 2026.07.491\n", encoding="utf-8")
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
        root=_make_repo(
            tmp_path / "wrong-doc", doc_milestone=mod.PRIOR_MILESTONE, capstone=_capstone_payload()
        ),
        run_date="20260708",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert "roadmap_doc_missing_or_mismatch_2026.07.491" in wrong_doc["failed_preconditions"]

    wrong_active = mod.build_artifact(
        root=_make_repo(
            tmp_path / "wrong-active",
            active_milestone=mod.PRIOR_MILESTONE,
            capstone=_capstone_payload(),
        ),
        run_date="20260708",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert (
        "executable_roadmap_milestone_expected_2026.07.491_observed_2026.07.490"
        in wrong_active["failed_preconditions"]
    )

    wrong_next = mod.build_artifact(
        root=_make_repo(
            tmp_path / "wrong-next",
            next_milestone=mod.PRIOR_MILESTONE,
            capstone=_capstone_payload(),
        ),
        run_date="20260708",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert (
        "roadmap_next_milestone_expected_2026.07.491_observed_2026.07.490"
        in wrong_next["failed_preconditions"]
    )

    wrong_capstone = _capstone_payload() | {
        "milestone": mod.MILESTONE,
        "status": "partial",
        "honest_verdict": "done",
    }
    wrong_capstone_artifact = mod.build_artifact(
        root=_make_repo(tmp_path / "wrong-capstone", capstone=wrong_capstone),
        run_date="20260708",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert (
        "capstone_milestone_expected_2026.07.490_observed_2026.07.491"
        in wrong_capstone_artifact["failed_preconditions"]
    )
    assert "capstone_status_expected_complete_observed_partial" in (
        wrong_capstone_artifact["failed_preconditions"]
    )
    assert "capstone_honest_verdict_missing_terminal_prefix" in (
        wrong_capstone_artifact["failed_preconditions"]
    )

    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.build_artifact(
            root=_make_repo(tmp_path / "dirty", capstone=_capstone_payload()),
            run_date="20260708",
            duration_s=1.0,
            modification_status={
                mod.ROADMAP_RELATIVE_PATH: True,
                mod.CONDUCTOR_RELATIVE_PATH: True,
            },
        )

    no_phase_rows = _capstone_payload() | {"phase_summaries": "bad"}
    no_phase_artifact = mod.build_artifact(
        root=_make_repo(
            tmp_path / "no-phase",
            capstone=no_phase_rows,
            conductor_log="no flag here\n",
        ),
        run_date="20260708",
        duration_s=1.0,
        modification_status={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
        },
    )
    assert no_phase_artifact["prior_blockers"]["solver_tautology"][
        "conductor_flagged_tautology"
    ] is False
    assert no_phase_artifact["prior_blockers"]["hardware"]["kv260_reachability"] is None

    out = mod.run(
        root=_make_repo(tmp_path / "run-repo", capstone=_capstone_payload()),
        run_date="20260708",
        duration_s=1.0,
    )
    saved = json.loads(out.read_text(encoding="utf-8"))
    mod.validate_artifact(saved)
    assert saved["milestone"] == mod.MILESTONE

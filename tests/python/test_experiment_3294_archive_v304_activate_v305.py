"""Tests for Exp 3294 archive .304 and activate .305 handoff.

Spec refs: REQ-REPORT-3294, SCENARIO-REPORT-3294.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import archive_v304_activate_v305_3294 as mod


REQUIRED_FIELDS = {
    "v304_closed_v305_opened",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "prior_next_top_gap",
    "garak_unblocked",
    "prior_garak_gate_passed",
    "prior_attack_success_rate",
    "clean_verifier_abstention_unblocked",
    "kan_headline_retired",
    "repair_gate_open",
    "repair_micro_panel_headline_eligible",
    "protected_files_untouched",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capstone_v304() -> dict[str, Any]:
    return {
        "artifact": "experiment_3293_capstone_v304",
        "experiment_id": "exp3293",
        "task_id": "exp3293-capstone-v304",
        "milestone": "2026.05.304",
        "capstone_v304_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 10,
        "blocker_delta_from_v303": -95,
        "garak_unblocked": True,
        "garak_gate_passed": False,
        "clean_verifier_abstention_unblocked": True,
        "kan_boundary_decision": "retire_from_prompt_injection_headline",
        "repair_gate_open": True,
        "repair_micro_panel_headline_eligible": False,
        "next_top_gap": "pass_garak_redteam_gate",
        "gate_status_details": {
            "garak_redteam": {
                "garak_gate_passed": False,
                "honest_verdict": (
                    "complete: garak_gate_passed=false; "
                    "attack_success_rate=0.311111"
                ),
            },
            "repair_panel": {"headline_claim_allowed": False},
        },
        "honest_verdict": "complete: capstone_v304_ready=true; paper_ready=false",
    }


def _matrix_v36() -> dict[str, Any]:
    return {
        "artifact": "experiment_3292_evidence_matrix_v36",
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
            }
        ],
        "gate_summary": {
            "garak_redteam": {
                "garak_gate_passed": False,
                "honest_verdict": (
                    "complete: garak_gate_passed=false; "
                    "attack_success_rate=0.311111"
                ),
            }
        },
        "evidence_rows": [
            {
                "experiment_id": "exp3285",
                "summary": {"attack_success_rate": 0.311111},
            }
        ],
        "honest_verdict": "complete: matrix_v36_ready=true; paper_ready=false",
    }


def _research_complete_yaml() -> str:
    lines = [
        "milestones:",
        "- id: 2026.05.304",
        "  title: Garak Availability + Abstention-Calibrated Verifier + Repair Gate Reopen",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-05-28'",
        "  tasks:",
    ]
    for task in mod.PRIOR_TASKS:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {task['title']}",
                f"    deliverable: {task['deliverable']}",
                "    result: OK (conductor)",
            ]
        )
    return "\n".join(lines) + "\n"


def _roadmap_yaml(milestone: str = "2026.05.305") -> str:
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "Garak Red-Team Gate Pass + Headline-Eligible Repair Evidence"\n'
        'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        "tasks:\n"
        '  - id: "exp3294-archive-v304-activate-v305"\n'
        f'    milestone: "{milestone}"\n'
        '    deliverable: "results/experiment_3294_archive_v304_activate_v305.json"\n'
    )


def _conductor_log() -> str:
    return (
        "\n".join(
            [
                "| 2026-05-28 17:23 UTC | Close .303 ledger and open .304 blocker queue | OK | 81 passed in 4.19s |",
                "| 2026-05-28 17:36 UTC | Garak install and probe manifest v1 | OK | 81 passed in 3.40s |",
                "| 2026-05-28 17:52 UTC | Prompt-injection corrigendum and duration audit v1 | OK | 81 passed in 4.12s |",
                "| 2026-05-28 18:13 UTC | Gated Garak local smoke against mandated SOTA GGUF | OK | 81 passed in 3.03s |",
                "| 2026-05-28 18:33 UTC | Gated full Garak/DataFlip red-team eval v2 | OK | 81 passed in 4.51s |",
                "| 2026-05-28 19:13 UTC | Clean verifier abstention root-cause audit v1 | OK | 81 passed in 2.72s |",
                "| 2026-05-28 19:31 UTC | Gated abstention-calibrated clean verifier v15 | OK | 81 passed in 3.66s |",
                "| 2026-05-28 19:46 UTC | Gated KAN sidecar failure autopsy and boundary dec | OK | 81 passed in 4.29s |",
                "| 2026-05-28 20:00 UTC | Gated repair gate decision v9 after Garak and abst | OK | 81 passed in 3.34s |",
                "| 2026-05-28 20:14 UTC | Gated SOTA repair micro-panel v10 | OK | 81 passed in 3.08s |",
                "| 2026-05-28 20:26 UTC | FR-11 Garak and abstention memory replay v1 | OK | 81 passed in 2.67s |",
                "| 2026-05-28 20:43 UTC | Evidence matrix v36 | OK | 81 passed in 3.64s |",
                "| 2026-05-28 20:59 UTC | Capstone v304 | OK | 81 passed in 3.93s |",
                "| 2026-05-28 21:47 UTC | Milestone 2026.05.305 activated | OK | 13 tasks queued |",
            ]
        )
        + "\n"
    )


def _write_sources(root: Path, *, archived: bool = True) -> None:
    _write_json(root, mod.CAPSTONE_V304_REL_PATH, _capstone_v304())
    _write_json(root, mod.MATRIX_V36_REL_PATH, _matrix_v36())
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_REL_PATH,
        _research_complete_yaml() if archived else "milestones: []\n",
    )
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.CONDUCTOR_LOG_REL_PATH, _conductor_log())
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")


def test_req_report_3294_spec_anchor_exists() -> None:
    """REQ-REPORT-3294: OpenSpec declares the .304/.305 handoff first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3294" in spec
    assert "SCENARIO-REPORT-3294" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in REQUIRED_FIELDS:
        assert field in spec


def test_scenario_report_3294_existing_archive_opens_v305_without_mutation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3294: existing .304 archive opens Garak-gate .305 queue."""

    _write_sources(tmp_path)
    before_complete = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    before_roadmap = (tmp_path / mod.ACTIVE_ROADMAP_REL_PATH).read_text(encoding="utf-8")
    before_conductor = (tmp_path / mod.CONDUCTOR_REL_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=8.5)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)
    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=2.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3294"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.305"
    assert artifact["prior_milestone"] == "2026.05.304"
    assert artifact["v304_closed_v305_opened"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 10
    assert artifact["prior_next_top_gap"] == "pass_garak_redteam_gate"
    assert artifact["garak_unblocked"] is True
    assert artifact["prior_garak_gate_passed"] is False
    assert artifact["prior_attack_success_rate"] == pytest.approx(0.311111)
    assert artifact["clean_verifier_abstention_unblocked"] is True
    assert artifact["kan_headline_retired"] is True
    assert artifact["repair_gate_open"] is True
    assert artifact["repair_micro_panel_headline_eligible"] is False
    assert artifact["inference_substrate"] == "artifact_aggregation_only"
    assert artifact["research_complete_update"] == {
        "path": "research-complete.yaml",
        "appended": False,
        "already_present": True,
    }
    assert artifact["research_complete_prior_summary"]["task_count"] == len(mod.PRIOR_TASKS)
    assert artifact["v305_queue"]["selected_queue_milestone"] == "2026.05.305"
    assert artifact["v305_queue"]["queue_first_task"] == mod.TASK_ID
    assert artifact["v305_activation_observed"] is True
    assert len(artifact["conductor_log_terminal_rows"]) == len(mod.PRIOR_TASKS)
    assert artifact["conductor_log_terminal_status_counts"] == {"OK": len(mod.PRIOR_TASKS)}
    assert artifact["protected_files_untouched"] is True
    assert artifact["protected_file_checksums"]["research-roadmap.yaml"]["unchanged"] is True
    assert (
        artifact["protected_file_checksums"]["scripts/research_conductor.py"]["unchanged"] is True
    )
    assert "Garak gate pass" in artifact["v305_activation_reason"]
    assert "headline repair evidence" in artifact["v305_activation_reason"]
    assert "not another installation, corpus, or KAN milestone" in artifact[
        "v305_activation_reason"
    ]
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["duration_s"] == pytest.approx(4.5)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert artifact["source_checksums"][mod.CAPSTONE_V304_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V304_REL_PATH
    )
    assert artifact["honest_verdict"].startswith("complete:")
    assert "paper_ready=false" in artifact["honest_verdict"]
    assert saved["duration_s"] == pytest.approx(1.0)
    assert (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(
        encoding="utf-8"
    ) == before_complete
    assert (tmp_path / mod.ACTIVE_ROADMAP_REL_PATH).read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / mod.CONDUCTOR_REL_PATH).read_text(encoding="utf-8") == before_conductor


def test_req_report_3294_appends_missing_archive_once(tmp_path: Path) -> None:
    """REQ-REPORT-3294: missing .304 archive is materialized exactly once."""

    _write_sources(tmp_path, archived=False)

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=4.0)
    saved = json.loads(output.read_text(encoding="utf-8"))
    archive_text = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    ensure_result = mod.ensure_research_complete_entry(tmp_path)

    assert saved["v304_closed_v305_opened"] is True
    assert saved["research_complete_update"]["appended"] is True
    assert saved["research_complete_update"]["already_present"] is False
    assert archive_text.count("- id: 2026.05.304") == 1
    assert ensure_result == {
        "path": "research-complete.yaml",
        "appended": False,
        "already_present": True,
    }


def test_req_report_3294_fail_closed_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3294: malformed inputs remain explicit and non-fabricated."""

    _write_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.CAPSTONE_V304_REL_PATH,
        _capstone_v304()
        | {
            "capstone_v304_ready": False,
            "paper_ready": True,
            "publication_blocker_count": 0,
            "next_top_gap": "wrong",
            "garak_unblocked": False,
            "garak_gate_passed": True,
            "clean_verifier_abstention_unblocked": False,
            "kan_boundary_decision": "promote_to_headline",
            "repair_gate_open": False,
            "repair_micro_panel_headline_eligible": True,
            "gate_status_details": {},
        },
    )
    _write_json(
        tmp_path,
        mod.MATRIX_V36_REL_PATH,
        _matrix_v36() | {"evidence_rows": [], "gate_summary": {}},
    )
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml("2026.05.304"))
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_REL_PATH, "[")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("tasks: [", encoding="utf-8")

    baseline = mod.protected_file_checksums(tmp_path)
    _write_text(tmp_path, mod.CONDUCTOR_REL_PATH, "# modified protected conductor\n")
    artifact = mod.build_artifact(
        tmp_path,
        protected_hash_baseline=baseline,
        started_s=8.0,
        now_s=3.0,
    )

    assert artifact["v304_closed_v305_opened"] is False
    assert artifact["duration_s"] == 0.0
    assert artifact["prior_paper_ready"] is True
    assert artifact["prior_publication_blocker_count"] == 0
    assert artifact["prior_next_top_gap"] == "wrong"
    assert artifact["garak_unblocked"] is False
    assert artifact["prior_garak_gate_passed"] is True
    assert artifact["prior_attack_success_rate"] == 0.0
    assert artifact["clean_verifier_abstention_unblocked"] is False
    assert artifact["kan_headline_retired"] is False
    assert artifact["repair_gate_open"] is False
    assert artifact["repair_micro_panel_headline_eligible"] is True
    assert artifact["protected_files_untouched"] is False
    assert "capstone_v304 authority is not ready" in artifact["blocked_reasons"]
    assert "prior paper_ready must remain false" in artifact["blocked_reasons"]
    assert "selected queue milestone is not 2026.05.305" in artifact["blocked_reasons"]
    assert "protected files changed during handoff" in artifact["blocked_reasons"]
    assert "prior Garak gate must remain failed for .305 targeting" in artifact[
        "blocked_reasons"
    ]
    assert artifact["honest_verdict"].startswith("complete:")

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_yaml_document(bad_yaml) == {}
    assert mod.read_yaml_document(tmp_path / "missing.yaml") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._milestone_entries("bad") == []
    assert mod._milestone_entries([{"id": "x"}, "bad"]) == [{"id": "x"}]
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._int_value(True) == 0
    assert mod._int_value(7) == 7
    assert mod._float_value(True) == 0.0
    assert mod._float_value(1) == 1.0
    assert mod._float_value("bad") == 0.0
    assert mod._file_contains(tmp_path / "missing.log", "needle") is False
    assert mod._terminal_prefix_ok("success: done") is True
    assert mod._terminal_prefix_ok("blocked") is False
    assert mod._parse_conductor_line("not a conductor row") == {}
    assert mod._parse_attack_success_rate("attack_success_rate=0.25") == pytest.approx(0.25)
    assert mod._parse_attack_success_rate("none") == 0.0
    assert (
        mod._prior_attack_success_rate(
            {"gate_status_details": {"garak_redteam": {"attack_success_rate": 0.5}}},
            {},
        )
        == pytest.approx(0.5)
    )
    assert (
        mod._prior_attack_success_rate(
            {},
            {
                "gate_summary": {
                    "garak_redteam": {
                        "honest_verdict": "complete: attack_success_rate=0.4"
                    }
                }
            },
        )
        == pytest.approx(0.4)
    )
    assert (
        mod._prior_attack_success_rate(
            {},
            {"rows": [{"experiment_id": "exp3285", "summary": {"attack_success_rate": 0.3}}]},
        )
        == pytest.approx(0.3)
    )
    assert all(
        row["status"] == "missing"
        for row in mod._conductor_log_terminal_rows(tmp_path / "missing-log-root")
    )

    empty_archive = tmp_path / "empty" / mod.RESEARCH_COMPLETE_REL_PATH
    mod._append_research_complete_entry(empty_archive)
    assert empty_archive.read_text(encoding="utf-8").startswith("milestones:\n- id: 2026.05.304")
    no_newline_archive = tmp_path / "no-newline" / mod.RESEARCH_COMPLETE_REL_PATH
    no_newline_archive.parent.mkdir(parents=True)
    no_newline_archive.write_text("milestones:\n- id: 2026.05.303\n  tasks: []", encoding="utf-8")
    mod._append_research_complete_entry(no_newline_archive)
    assert no_newline_archive.read_text(encoding="utf-8").count("- id: 2026.05.304") == 1

    summary_root = tmp_path / "summary"
    _write_text(
        summary_root,
        mod.RESEARCH_COMPLETE_REL_PATH,
        "milestones:\n- id: 2026.05.303\n  tasks: []\n- id: 2026.05.304\n  tasks: []\n",
    )
    assert mod._research_complete_task_summary(summary_root)["task_count"] == 0

    good_artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(good_artifact | {"experiment_id": "bad"})
    with pytest.raises(ValueError, match="task_id"):
        mod.validate_artifact(good_artifact | {"task_id": "bad"})
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(good_artifact | {"milestone": "bad"})
    with pytest.raises(ValueError, match="random_seed"):
        mod.validate_artifact(good_artifact | {"random_seed": 0})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(good_artifact | {"inference_substrate": "live"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(good_artifact | {"honest_verdict": "blocked"})

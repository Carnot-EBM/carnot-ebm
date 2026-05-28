"""Tests for Exp 3281 archive .303 and activate .304 handoff.

Spec refs: REQ-REPORT-3281, SCENARIO-REPORT-3281.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import archive_v303_activate_v304_3281 as mod


REQUIRED_FIELDS = {
    "v303_closed_v304_opened",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "prior_next_top_gap",
    "full_15k_corpus_materialized",
    "garak_blocker",
    "clean_verifier_abstention_rate",
    "kan_noninferiority_passed",
    "repair_gate_open",
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


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capstone_v303() -> dict[str, Any]:
    return {
        "artifact": "experiment_3280_capstone_v303",
        "experiment_id": "exp3280",
        "task_id": "exp3280-capstone-v303",
        "milestone": "2026.05.303",
        "capstone_v303_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 105,
        "publication_blocker_delta": 0,
        "prior_next_top_gap": "full_15k_v4_corpus_across_shards_plus_repair_and_garak_gates",
        "v4_full_corpus_status": "partial: full_15k_ready_but_flagged_sidecar_noninferiority_failed",
        "garak_gate_status": "blocked: blocked_garak_unavailable",
        "repair_gate_status": "blocked: garak_redteam_and_clean_verifier_gates_failed",
        "fr11_status": "complete: controller_memory_only_retention_0.982143_adaptation_1.0_forgetting_0.017857",
        "next_top_gap": "unblock_garak_redteam_eval",
        "recommended_next_milestone_title": "Garak Red-Team Availability + Clean Verifier Repair Gate Reopen",
        "changes_since_v302": [
            "full_15k_v4_corpus_materialized",
            "fr11_full_corpus_controller_memory_audit_completed",
            "publication_blocker_count_unchanged",
            "top_gap_narrowed_to_garak_redteam_eval",
        ],
        "stayed_blocked": [
            "kan_sidecar_only_noninferiority_failed",
            "garak_redteam_blocked_unavailable",
            "clean_verifier_abstention_gate_failed",
            "repair_gate_blocked",
        ],
        "honest_verdict": "complete: capstone_v303_ready=true; paper_ready=false",
    }


def _row(
    experiment_id: str,
    *,
    status: str,
    ready: bool,
    summary: Mapping[str, Any] | None = None,
    blocker_reasons: list[str] | None = None,
    bounded_claims: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "status": status,
        "ready": ready,
        "summary": dict(summary or {}),
        "blocker_reasons": list(blocker_reasons or []),
        "bounded_claims": list(bounded_claims or []),
        "path": f"results/{experiment_id}.json",
        "sha256": "a" * 64,
    }


def _matrix_v35() -> dict[str, Any]:
    return {
        "artifact": "experiment_3279_evidence_matrix_v35",
        "experiment_id": "exp3279",
        "task_id": "exp3279-evidence-matrix-v35",
        "matrix_v35_ready": True,
        "paper_ready": False,
        "publication_blocker_count_estimate": 105,
        "publication_blocker_delta_from_v302": 0,
        "publication_readiness": {
            "paper_ready": False,
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
        "rows": [
            _row("exp3272", status="flagged", ready=True, summary={"full_15k_corpus_ready": True}),
            _row(
                "exp3273",
                status="sidecar-only",
                ready=True,
                summary={"delong_noninferiority_passed": False, "sidecar_only": True},
                bounded_claims=["sidecar_only=true", "delong_noninferiority_passed=false"],
            ),
            _row(
                "exp3274",
                status="blocked",
                ready=False,
                summary={"garak_available": False, "garak_gate_passed": False},
                blocker_reasons=["blocked_garak_unavailable"],
            ),
            _row(
                "exp3275",
                status="blocked",
                ready=False,
                summary={"clean_verifier_rerun_ready": False},
                blocker_reasons=["abstention_rate_above_threshold"],
            ),
            _row("exp3276", status="blocked", ready=False, summary={"status": "blocked"}),
            _row(
                "exp3278",
                status="clean",
                ready=True,
                summary={"fr11_full_corpus_audit_ready": True, "controller_memory_only": True},
            ),
        ],
        "honest_verdict": "complete: matrix_v35_ready=true",
    }


def _research_complete_yaml() -> str:
    lines = [
        "milestones:",
        "- id: 2026.05.303",
        "  title: Prompt-Injection v4 Full Corpus + Garak Gate + Repair Reopen",
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


def _roadmap_yaml(milestone: str = "2026.05.304") -> str:
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "Garak Availability + Abstention-Calibrated Verifier + Repair Gate Reopen"\n'
        'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        'milestone_provenance: "Authority: results/experiment_3280_capstone_v303.json reports clean verifier abstention_rate=1.0."\n'
        "tasks:\n"
        '  - id: "exp3281-archive-v303-activate-v304"\n'
        f'    milestone: "{milestone}"\n'
        '    deliverable: "results/experiment_3281_archive_v303_activate_v304.json"\n'
    )


def _conductor_log() -> str:
    return (
        "\n".join(
            [
                "| 2026-05-28 12:07 UTC | Close .302 ledger and open .303 corpus queue | OK | 81 passed in 3.36s |",
                "| 2026-05-28 12:22 UTC | SOTA receipt methodology supplement v1 | OK | 81 passed in 3.45s |",
                "| 2026-05-28 12:38 UTC | Prompt-injection v4 full-corpus split manifest | OK | 81 passed in 4.39s |",
                "| 2026-05-28 12:56 UTC | Prompt-injection teacher-label shards 2-4 | OK | 81 passed in 3.60s |",
                "| 2026-05-28 13:13 UTC | Prompt-injection teacher-label shards 5-7 plus Garak seed | OK | 81 passed in 4.16s |",
                "| 2026-05-28 13:32 UTC | Prompt-injection v4 full-corpus assembly and leakage audit | OK | 81 passed in 3.80s |",
                "| 2026-05-28 13:53 UTC | Prompt-injection KAN full-corpus DeLong eval | OK | 81 passed in 3.69s |",
                "| 2026-05-28 14:13 UTC | Prompt-injection v4 Garak and DataFlip red-team eval | OK | 81 passed in 4.67s |",
                "| 2026-05-28 14:55 UTC | Clean local SOTA verifier rerun v14 | OK | 81 passed in 3.79s |",
                "| 2026-05-28 15:01 UTC | Repair gate decision v8 after v4, Garak, and clean verifier | GATE_BLOCK | 2 of 3 gate(s) failed |",
                "| 2026-05-28 15:33 UTC | SOTA repair micro-panel v9 | GATE_BLOCK | Pre-emptive skip: upstream retired |",
                "| 2026-05-28 15:15 UTC | FR-11 full-corpus continual self-learning audit | OK | 81 passed in 3.77s |",
                "| 2026-05-28 15:31 UTC | Evidence matrix v35 for .303 corpus, Garak, repair | OK | 81 passed in 2.98s |",
                "| 2026-05-28 15:45 UTC | Capstone v303 and next-gap decision | OK | 81 passed in 2.78s |",
                "| 2026-05-28 17:06 UTC | Milestone 2026.05.304 activated | OK | 13 tasks queued |",
            ]
        )
        + "\n"
    )


def _write_sources(root: Path, *, archived: bool = True) -> None:
    _write_json(root, mod.CAPSTONE_V303_REL_PATH, _capstone_v303())
    _write_json(root, mod.MATRIX_V35_REL_PATH, _matrix_v35())
    _write_json(
        root,
        mod.EXP3275_REL_PATH,
        {
            "experiment_id": "exp3275",
            "clean_verifier_rerun_ready": False,
            "repair_gate_input_clean_enough": False,
            "abstention_rate": 1.0,
            "abstention_count": 6,
            "gate_reasons": ["abstention_rate_above_threshold"],
        },
    )
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_REL_PATH,
        _research_complete_yaml() if archived else "milestones: []\n",
    )
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.CONDUCTOR_LOG_REL_PATH, _conductor_log())
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")


def test_req_report_3281_spec_anchor_exists() -> None:
    """REQ-REPORT-3281: OpenSpec declares the .303/.304 handoff first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3281" in spec
    assert "SCENARIO-REPORT-3281" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3281_existing_archive_opens_v304_without_mutation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3281: existing .303 archive opens Garak-first .304 queue."""

    _write_sources(tmp_path)
    before_complete = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    before_roadmap = (tmp_path / mod.ACTIVE_ROADMAP_REL_PATH).read_text(encoding="utf-8")
    before_conductor = (tmp_path / mod.CONDUCTOR_REL_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=7.25)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)
    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=2.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3281"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.304"
    assert artifact["prior_milestone"] == "2026.05.303"
    assert artifact["v303_closed_v304_opened"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 105
    assert artifact["prior_next_top_gap"] == "unblock_garak_redteam_eval"
    assert artifact["full_15k_corpus_materialized"] is True
    assert artifact["garak_blocker"] == "blocked_garak_unavailable"
    assert artifact["clean_verifier_abstention_rate"] == pytest.approx(1.0)
    assert artifact["kan_noninferiority_passed"] is False
    assert artifact["repair_gate_open"] is False
    assert artifact["research_complete_update"] == {
        "path": "research-complete.yaml",
        "appended": False,
        "already_present": True,
    }
    assert artifact["research_complete_prior_summary"]["task_count"] == len(mod.PRIOR_TASKS)
    assert artifact["v304_queue"]["selected_queue_milestone"] == "2026.05.304"
    assert artifact["v304_queue"]["queue_first_task"] == mod.TASK_ID
    assert artifact["v304_activation_observed"] is True
    assert len(artifact["conductor_log_terminal_rows"]) == len(mod.PRIOR_TASKS)
    assert artifact["protected_files_untouched"] is True
    assert artifact["protected_file_checksums"]["research-roadmap.yaml"]["unchanged"] is True
    assert (
        artifact["protected_file_checksums"]["scripts/research_conductor.py"]["unchanged"] is True
    )
    assert artifact["blocker_movement"]["publication_blocker_delta"] == 0
    assert "full 15k corpus is already materialized" in artifact["v304_activation_reason"]
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert artifact["source_checksums"][mod.CAPSTONE_V303_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V303_REL_PATH
    )
    assert artifact["honest_verdict"].startswith("complete:")
    assert "paper_ready=false" in artifact["honest_verdict"]
    assert saved["duration_s"] == pytest.approx(1.0)
    assert (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(
        encoding="utf-8"
    ) == before_complete
    assert (tmp_path / mod.ACTIVE_ROADMAP_REL_PATH).read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / mod.CONDUCTOR_REL_PATH).read_text(encoding="utf-8") == before_conductor


def test_req_report_3281_appends_missing_archive_once(tmp_path: Path) -> None:
    """REQ-REPORT-3281: missing .303 archive is materialized exactly once."""

    _write_sources(tmp_path, archived=False)

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=4.0)
    saved = json.loads(output.read_text(encoding="utf-8"))
    archive_text = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    ensure_result = mod.ensure_research_complete_entry(tmp_path)

    assert saved["v303_closed_v304_opened"] is True
    assert saved["research_complete_update"]["appended"] is True
    assert saved["research_complete_update"]["already_present"] is False
    assert archive_text.count("- id: 2026.05.303") == 1
    assert ensure_result == {
        "path": "research-complete.yaml",
        "appended": False,
        "already_present": True,
    }


def test_req_report_3281_fail_closed_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3281: malformed inputs remain explicit and non-fabricated."""

    _write_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.CAPSTONE_V303_REL_PATH,
        _capstone_v303()
        | {
            "capstone_v303_ready": False,
            "paper_ready": True,
            "publication_blocker_count": 0,
            "next_top_gap": "wrong",
        },
    )
    _write_json(
        tmp_path, mod.MATRIX_V35_REL_PATH, _matrix_v35() | {"matrix_v35_ready": False, "rows": []}
    )
    _write_json(
        tmp_path, mod.EXP3275_REL_PATH, {"experiment_id": "exp3275", "abstention_rate": "bad"}
    )
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml("2026.05.303"))
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

    assert artifact["v303_closed_v304_opened"] is False
    assert artifact["duration_s"] == 0.0
    assert artifact["prior_paper_ready"] is True
    assert artifact["prior_publication_blocker_count"] == 0
    assert artifact["prior_next_top_gap"] == "wrong"
    assert artifact["full_15k_corpus_materialized"] is False
    assert artifact["garak_blocker"] == ""
    assert artifact["clean_verifier_abstention_rate"] == 0.0
    assert artifact["kan_noninferiority_passed"] is False
    assert artifact["repair_gate_open"] is False
    assert artifact["protected_files_untouched"] is False
    assert "capstone_v303 authority is not ready" in artifact["blocked_reasons"]
    assert "prior paper_ready must remain false" in artifact["blocked_reasons"]
    assert "selected queue milestone is not 2026.05.304" in artifact["blocked_reasons"]
    assert "protected files changed during handoff" in artifact["blocked_reasons"]
    assert "full 15k corpus materialization evidence is missing" in artifact["blocked_reasons"]
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
    assert all(
        row["status"] == "missing"
        for row in mod._conductor_log_terminal_rows(tmp_path / "missing-log-root")
    )

    empty_archive = tmp_path / "empty" / mod.RESEARCH_COMPLETE_REL_PATH
    mod._append_research_complete_entry(empty_archive)
    assert empty_archive.read_text(encoding="utf-8").startswith("milestones:\n- id: 2026.05.303")
    no_newline_archive = tmp_path / "no-newline" / mod.RESEARCH_COMPLETE_REL_PATH
    no_newline_archive.parent.mkdir(parents=True)
    no_newline_archive.write_text("milestones:\n- id: 2026.05.302\n  tasks: []", encoding="utf-8")
    mod._append_research_complete_entry(no_newline_archive)
    assert no_newline_archive.read_text(encoding="utf-8").count("- id: 2026.05.303") == 1

    summary_root = tmp_path / "summary"
    _write_text(
        summary_root,
        mod.RESEARCH_COMPLETE_REL_PATH,
        "milestones:\n- id: 2026.05.302\n  tasks: []\n- id: 2026.05.303\n  tasks: []\n",
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
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(good_artifact | {"honest_verdict": "blocked"})

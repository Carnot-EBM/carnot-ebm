"""Tests for Exp 2988 milestone .280 archive and .281 activation.

Spec refs: REQ-REPORT-2988, SCENARIO-REPORT-2988.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.reporting import milestone_280_archive_281_activation as exp


def _clock(*values: float):
    times = iter(values)
    return lambda: next(times)


_AUDIT_ROWS = [
    ("exp2975", "projection-only", "results/experiment_2975_archive_v279_activate_v280.json"),
    ("exp2976", "flagged", "results/experiment_2976_dccd_adaptrack_tracecoder_protocol_v1.json"),
    ("exp2977", "blocked", "results/experiment_2977_sota_intent_preserving_code_repair_v1.json"),
    (
        "exp2978",
        "flagged",
        "results/experiment_2978_first_step_semantic_energy_repair_telemetry_v1.json",
    ),
    ("exp2979", "clean", "results/experiment_2979_solver_feedback_mcs_frontier_v1.json"),
    ("exp2980", "flagged", "results/experiment_2980_sota_solver_formalization_feedback_v2.json"),
    ("exp2981", "clean", "results/experiment_2981_interwhen_partial_monitor_promotion_v2.json"),
    ("exp2982", "clean", "results/experiment_2982_fr11_independent_metric_utility_gate_v4.json"),
    ("exp2983", "flagged", "results/experiment_2983_trace_to_skill_repair_memory_pilot_v1.json"),
    ("exp2984", "blocked", "results/experiment_2984_gatemate_readback_smoke_vector_v4.json"),
    ("exp2985", "projection-only", "results/experiment_2985_ssqa_dual_bram_register_map_plan_v1.json"),
    ("exp2986", "flagged", "results/experiment_2986_cross_corpus_matrix_v14.json"),
]


def _audit_payload() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for exp_id, classification, path in _AUDIT_ROWS:
        rows.append(
            {
                "experiment_id": exp_id,
                "classification": classification,
                "path": path,
                "honest_verdict": f"complete: {exp_id} terminal",
                "prior_failure_outcome": f"{exp_id}_outcome",
                "upstream_flags": ["DURATION_TOO_SHORT"] if classification == "flagged" else [],
            }
        )
    rows[2]["honest_verdict"] = "blocked_cached_sota_pair_unavailable_cpu_smoke_only"
    rows[9]["honest_verdict"] = "complete: gatemate_no_readback_no_host_smoke_io"
    return rows


def _write_capstone(root: Path) -> None:
    path = root / exp.CAPSTONE_SOURCE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "experiment_2987_capstone_v280",
        "artifact_audit": _audit_payload(),
        "artifact_classification_counts": {
            "blocked": 2,
            "clean": 3,
            "flagged": 5,
            "gated-skipped": 0,
            "missing": 0,
            "pilot-only": 0,
            "projection-only": 2,
        },
        "blocked_artifacts": ["exp2977", "exp2984"],
        "clean_artifacts": ["exp2979", "exp2981", "exp2982"],
        "flagged_artifacts": ["exp2976", "exp2978", "exp2980", "exp2983", "exp2986"],
        "gaps_remaining": [
            "Repair is not paper-ready.",
            "Solver feedback is not paper-ready.",
            "GateMate is not hardware-ready.",
        ],
        "headline_outcome": "not_paper_ready: repair=blocked, solver=flagged",
        "honest_verdict": (
            "complete: milestone_280_capstone; paper_ready=false; "
            "clean=3; flagged=5; blocked=2; missing=0"
        ),
        "inference_substrate": exp.INFERENCE_SUBSTRATE,
        "milestone": "2026.05.280",
        "missing_artifacts": [],
        "next_milestone_recommendations": ["Repair: restore mandated cached SOTA."],
        "paper_ready": False,
        "pilot_only_artifacts": [],
        "projection_only_artifacts": ["exp2975", "exp2985"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_referenced_results(root: Path, *, skip: str | None = None) -> None:
    for row in _audit_payload():
        path = str(row["path"])
        if path == skip:
            continue
        result_path = root / path
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"honest_verdict": row["honest_verdict"]}), encoding="utf-8")


def _complete_archive_text() -> str:
    task_rows = "\n".join(
        f"  - id: exp{number}\n"
        f"    title: Task {number}\n"
        f"    deliverable: results/experiment_{number}.json\n"
        f"    result: OK (conductor)"
        for number in range(2975, 2988)
    )
    return (
        "- id: 2026.05.280\n"
        "  title: Intent-Preserving Repair + Solver Feedback + Readback-Grounded Self-Learning\n"
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md\n"
        "  completed: '2026-05-24'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        f"{task_rows}\n"
    )


def _write_common_files(
    root: Path,
    *,
    complete_text: str,
    active_roadmap: str = (
        'milestone: "2026.05.281"\n'
        'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        "tasks:\n"
        "  - id: exp2988-archive-v280-activate-v281\n"
        "    deliverable: results/experiment_2988_archive_v280_activate_v281.json\n"
    ),
    next_roadmap: str | None = None,
    write_capstone: bool = True,
    skip_result: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(active_roadmap, encoding="utf-8")
    if next_roadmap is not None:
        (root / "research-roadmap-next.yaml").write_text(next_roadmap, encoding="utf-8")
    conductor = root / "scripts" / "research_conductor.py"
    conductor.parent.mkdir(parents=True, exist_ok=True)
    conductor.write_text("# untouched by Exp 2988\n", encoding="utf-8")
    (root / "ops").mkdir(exist_ok=True)
    (root / "ops" / "status.md").write_text("# status\n", encoding="utf-8")
    (root / "ops" / "changelog.md").write_text("# changelog\n", encoding="utf-8")
    if write_capstone:
        _write_capstone(root)
        _write_referenced_results(root, skip=skip_result)


def _milestone_rows(root: Path) -> list[dict]:
    complete = yaml.safe_load((root / "research-complete.yaml").read_text(encoding="utf-8"))
    rows = complete.get("milestones", []) if isinstance(complete, dict) else complete
    return [row for row in rows if str(row.get("id")) == "2026.05.280"]


def test_req_report_2988_spec_is_declared() -> None:
    """REQ-REPORT-2988: OpenSpec declares the archive/activation contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2988" in spec
    assert "SCENARIO-REPORT-2988" in spec
    assert str(exp.DEFAULT_OUTPUT_PATH) in spec


def test_scenario_report_2988_existing_archive_uses_active_roadmap_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2988: existing .280 archive confirms active .281 roadmap."""

    _write_common_files(tmp_path, complete_text=_complete_archive_text())
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (tmp_path / "scripts/research_conductor.py").read_text(encoding="utf-8")
    before_status = (tmp_path / "ops" / "status.md").read_text(encoding="utf-8")
    before_changelog = (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(4.0, 5.5))
    written = json.loads((tmp_path / exp.DEFAULT_OUTPUT_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"] == exp.COMPLETE_VERDICT
    assert artifact["archive_ready"] is True
    assert artifact["archived_milestone"] == "2026.05.280"
    assert artifact["activated_milestone"] == "2026.05.281"
    assert artifact["research_complete_updated"] is True
    assert artifact["status_updates_written"] is False
    assert artifact["ops_docs_reconciliation_left_to_conductor"] is True
    assert artifact["n_tasks_archived"] == 13
    assert artifact["archive_already_present"] is True
    assert artifact["archive_appended_this_run"] is False
    assert artifact["capstone_source"] == "results/experiment_2987_capstone_v280.json"
    assert artifact["capstone_honest_verdict"].startswith("complete: milestone_280_capstone")
    assert artifact["paper_ready_from_capstone"] is False
    assert artifact["headline_outcome_from_capstone"].startswith("not_paper_ready:")
    assert artifact["artifact_classification_counts_from_capstone"] == {
        "blocked": 2,
        "clean": 3,
        "flagged": 5,
        "gated-skipped": 0,
        "missing": 0,
        "pilot-only": 0,
        "projection-only": 2,
    }
    assert artifact["artifact_classification_counts_from_audit"] == {
        "blocked": 2,
        "clean": 3,
        "flagged": 5,
        "missing": 0,
        "pilot-only": 0,
        "projection-only": 2,
    }
    assert artifact["n_referenced_artifacts"] == 12
    assert artifact["n_source_artifacts_read"] == 12
    assert artifact["missing_referenced_artifacts"] == []
    assert [row["experiment_id"] for row in artifact["blocked_or_flagged_rows_carried_forward"]] == [
        "exp2976",
        "exp2977",
        "exp2978",
        "exp2980",
        "exp2983",
        "exp2984",
        "exp2986",
    ]
    assert artifact["validation_commands"] == exp.DEFAULT_VALIDATION_COMMANDS
    assert artifact["activation"]["roadmap_source"] == "research-roadmap.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is True
    assert artifact["activation"]["research_roadmap_next_exists"] is False
    assert artifact["activation"]["non_empty_tasks"] is True
    assert artifact["roadmap_verification"]["research_roadmap_yaml_modified"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before_status
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before_changelog


def test_req_report_2988_appends_completed_archive_when_absent(tmp_path: Path) -> None:
    """REQ-REPORT-2988: absent .280 archive is appended with all completed tasks."""

    _write_common_files(
        tmp_path,
        complete_text="- id: 2026.05.279\n  completed: '2026-05-24'\n  tasks: []\n",
        next_roadmap=(
            'milestone: "2026.05.281"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
            "tasks:\n"
            "  - id: exp2988-archive-v280-activate-v281\n"
        ),
    )
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(10.0, 10.25))
    rows = _milestone_rows(tmp_path)

    assert artifact["honest_verdict"] == exp.COMPLETE_VERDICT
    assert artifact["archive_ready"] is True
    assert artifact["research_complete_updated"] is True
    assert artifact["archive_already_present"] is False
    assert artifact["archive_appended_this_run"] is True
    assert artifact["activation"]["roadmap_source"] == "research-roadmap-next.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is False
    assert artifact["n_tasks_archived"] == 13
    assert len(rows) == 1
    assert rows[0]["completed"] == "2026-05-24"
    assert len(rows[0]["tasks"]) == 13
    assert rows[0]["tasks"][-1]["deliverable"] == exp.CAPSTONE_SOURCE
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap


def test_req_report_2988_missing_capstone_blocks_without_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-2988: missing capstone exits blocked without archive mutation."""

    _write_common_files(
        tmp_path,
        complete_text="- id: 2026.05.279\n  tasks: []\n",
        write_capstone=False,
    )
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    absolute_output = tmp_path / "custom" / "experiment_2988.json"
    artifact = exp.run(root=tmp_path, output_path=absolute_output, clock=_clock(1.0, 1.1))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["archive_ready"] is False
    assert artifact["research_complete_updated"] is False
    assert artifact["n_tasks_archived"] == 0
    assert artifact["capstone_honest_verdict"] == ""
    assert artifact["artifact_classification_counts_from_audit"] == {
        "blocked": 0,
        "clean": 0,
        "flagged": 0,
        "missing": 0,
        "pilot-only": 0,
        "projection-only": 0,
    }
    assert artifact["blocked_or_flagged_rows_carried_forward"] == []
    assert json.loads(absolute_output.read_text(encoding="utf-8")) == artifact
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap


def test_req_report_2988_wrong_or_empty_roadmap_blocks(tmp_path: Path) -> None:
    """REQ-REPORT-2988: wrong or empty activation roadmap is reported honestly."""

    _write_common_files(
        tmp_path,
        complete_text=_complete_archive_text(),
        active_roadmap='milestone: "2026.05.280"\ntasks: []\n',
    )

    artifact = exp.run(root=tmp_path, clock=_clock(2.0, 2.2))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["archive_ready"] is True
    assert "roadmap milestone is not 2026.05.281" in artifact["blocked_reasons"]
    assert "roadmap has no tasks for 2026.05.281" in artifact["blocked_reasons"]


def test_req_report_2988_helpers_handle_bad_inputs_and_doc_status(tmp_path: Path) -> None:
    """REQ-REPORT-2988: helper readers tolerate malformed inputs and doc status."""

    assert exp._read_text(tmp_path / "missing.txt") == ""
    assert exp._read_json_mapping(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad json", encoding="utf-8")
    assert exp._read_json_mapping(bad_json) == {}
    array_json = tmp_path / "array.json"
    array_json.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp._read_json_mapping(array_json) == {}

    assert exp._load_yaml_mapping(tmp_path / "missing.yaml") == {}
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("milestone: [unterminated\n", encoding="utf-8")
    assert exp._load_yaml_mapping(bad_yaml) == {}
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- not-a-mapping\n", encoding="utf-8")
    assert exp._load_yaml_mapping(list_yaml) == {}

    assert exp._as_str_list(["exp1", 2]) == ["exp1", "2"]
    assert exp._as_str_list("exp1") == []
    assert exp._as_int_mapping({"clean": 1, "flagged": True, "missing": "1"}) == {"clean": 1}
    assert exp._as_int_mapping(["not", "mapping"]) == {}

    complete = (
        "- id: 2026.05.280\n"
        "  tasks: []\n"
        "- id: 2026.05.280\n"
        "  completed: '2026-05-24'\n"
        "  tasks:\n"
        "  - id: exp2975\n"
    )
    assert exp._archive_completed_block_present(complete) is True
    assert exp._archive_completed_block_count(complete) == 1
    assert exp._archive_task_count(complete) == 1
    assert exp._archive_task_count("milestones: not-a-list\n") == 0
    assert exp._archive_completed_block_present("[]\n") is False
    assert exp._audit_rows("not-a-list") == []

    docs = tmp_path / "docs"
    (docs / "ops").mkdir(parents=True)
    (docs / "ops" / "status.md").write_text("2026.05.280 archive and 2026.05.281 activation\n")
    (docs / "ops" / "changelog.md").write_text(
        "archived_milestone=2026.05.280 activated_milestone=2026.05.281\n"
    )
    assert exp._status_updates_written(docs) is True

    counts = exp._classification_counts_from_audit(
        [
            {"classification": "clean"},
            {"classification": "flagged"},
            {"classification": "blocked"},
            {"classification": "projection-only"},
            {"classification": "pilot-only"},
            {"classification": "other"},
            {"classification": ""},
        ]
    )
    assert counts == {
        "blocked": 1,
        "clean": 1,
        "flagged": 1,
        "missing": 0,
        "pilot-only": 1,
        "projection-only": 1,
    }

    _write_common_files(
        tmp_path,
        complete_text=_complete_archive_text(),
        skip_result="results/experiment_2984_gatemate_readback_smoke_vector_v4.json",
    )
    capstone = exp._read_json_mapping(tmp_path / exp.CAPSTONE_SOURCE)
    read_summary = exp._load_all_referenced_artifacts(tmp_path, capstone["artifact_audit"])
    assert read_summary["n_referenced_artifacts"] == 12
    assert read_summary["n_source_artifacts_read"] == 11
    assert read_summary["missing_referenced_artifacts"] == [
        "results/experiment_2984_gatemate_readback_smoke_vector_v4.json"
    ]
    no_path_summary = exp._load_all_referenced_artifacts(tmp_path, [{"experiment_id": "exp2999"}])
    assert no_path_summary["n_referenced_artifacts"] == 0

    verdict, reasons = exp._honest_verdict(
        archive_ready=False,
        research_complete_updated=False,
        roadmap={"milestone_matches": True, "non_empty_tasks": True},
        capstone_loaded=True,
    )
    assert verdict.startswith("blocked:")
    assert reasons == ["research-complete.yaml does not archive 2026.05.280"]

"""Tests for Exp 3012 milestone .282 archive and .283 activation.

Spec refs: REQ-REPORT-3012, SCENARIO-REPORT-3012.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.reporting import milestone_282_archive_283_activation as exp


def _clock(*values: float):
    times = iter(values)
    return lambda: next(times)


_SOURCE_ROWS = [
    ("exp3010", "results/experiment_3010_cross_corpus_matrix_v16.json", True, True),
    ("exp2999", "results/experiment_2999_capstone_v281.json", True, True),
    ("exp3000", "results/experiment_3000_archive_v281_activate_v282.json", True, False),
    (
        "exp3001",
        "results/experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1.json",
        True,
        False,
    ),
    ("exp3002", "results/experiment_3002_metamorphic_repair_oracle_audit_v1.json", True, False),
    (
        "exp3003",
        "results/experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1.json",
        True,
        False,
    ),
    (
        "exp3004",
        "results/experiment_3004_aquaforte_beaver_live_retry_provenance_v2.json",
        True,
        False,
    ),
    ("exp3005", "results/experiment_3005_solver_to_validator_tree_expansion_v1.json", True, False),
    ("exp3006", "results/experiment_3006_eqr_fixed_point_energy_diagnostic_v1.json", True, False),
    (
        "exp3007",
        "results/experiment_3007_fr11_attractor_trace_memory_stability_v1.json",
        True,
        False,
    ),
    ("exp3008", "results/experiment_3008_gatemate_host_visible_io_transport_v2.json", True, False),
    (
        "exp3009",
        "results/experiment_3009_ssqa_dual_bram_rtl_pnr_resource_report_v2.json",
        False,
        False,
    ),
]


def _matrix_rows() -> list[dict[str, object]]:
    return [
        {
            "row_id": "exp3000_archive_activation",
            "source_experiment_id": "exp3000",
            "source_honest_verdict": "complete: archive_ready=true",
            "status": "projection-only",
            "upstream_flags": ["flagged_adversarial=true", "DURATION_TOO_SHORT:critical"],
        },
        {
            "row_id": "exp3001_sota_cache",
            "source_experiment_id": "exp3001",
            "source_honest_verdict": "success: headline cache ready",
            "status": "clean",
            "upstream_flags": [],
        },
        {
            "row_id": "exp3003_metamorphic_repair",
            "source_experiment_id": "exp3003",
            "source_honest_verdict": "flagged: metamorphic repair rerun did not clear gates",
            "status": "flagged",
            "upstream_flags": [
                "flagged_adversarial=true",
                "TAUTOLOGY:critical",
                "METHODOLOGY_MISSING:warn",
            ],
        },
        {
            "row_id": "exp3007_fr11_trace_memory_stability",
            "source_experiment_id": "exp3007",
            "source_honest_verdict": "ready: trace_memory_stability_ready",
            "status": "flagged",
            "upstream_flags": ["flagged_adversarial=true", "TAUTOLOGY:critical"],
        },
        {
            "row_id": "exp3008_gatemate_host_visible_io",
            "source_experiment_id": "exp3008",
            "source_honest_verdict": "blocked_flash_failed",
            "status": "blocked",
            "upstream_flags": [],
        },
        {
            "row_id": "exp3009_ssqa_dual_bram_report",
            "source_experiment_id": "exp3009",
            "source_honest_verdict": "",
            "status": "gated-skipped",
            "upstream_flags": [],
        },
        {
            "row_id": "carry_forward_missing",
            "source_experiment_id": "exp2997",
            "source_honest_verdict": "complete: carried forward missing row",
            "status": "missing",
            "upstream_flags": [],
        },
        {
            "row_id": "carry_forward_pilot",
            "source_experiment_id": "exp2973",
            "source_honest_verdict": "complete: carried forward pilot row",
            "status": "pilot-only",
            "upstream_flags": [],
        },
    ]


def _write_capstone(root: Path) -> None:
    path = root / exp.CAPSTONE_SOURCE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "experiment_3011_capstone_v282",
        "blocked_rows": ["exp3008_gatemate_host_visible_io"],
        "blocked_task_rows": ["exp3008_gatemate_host_visible_io"],
        "capstone_ready": True,
        "clean_task_rows": [
            "exp3001_sota_cache",
            "exp3004_aquaforte_beaver_provenance",
            "exp3005_validator_tree_expansion",
            "exp3006_fixed_point_diagnostic",
            "exp3010_cross_corpus_matrix_v16",
        ],
        "flagged_rows": ["exp3003_metamorphic_repair", "exp3007_fr11_trace_memory_stability"],
        "flagged_task_rows": ["exp3003_metamorphic_repair", "exp3007_fr11_trace_memory_stability"],
        "gated_skipped_rows": ["exp3009_ssqa_dual_bram_report"],
        "gated_skipped_task_rows": ["exp3009_ssqa_dual_bram_report"],
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; repaired=4; "
            "flagged=23; blocked=9; gated_skipped=1; missing=1; next=2026.05.283"
        ),
        "inference_substrate": exp.INFERENCE_SUBSTRATE,
        "matrix_status_counts": {
            "blocked": 9,
            "clean": 38,
            "flagged": 23,
            "gated-skipped": 1,
            "missing": 1,
            "pilot-only": 4,
            "projection-only": 9,
        },
        "milestone": "2026.05.282",
        "missing_rows": ["carry_forward_missing"],
        "missing_task_rows": [],
        "n_tasks_evaluated": 11,
        "next_milestone_recommendation": "2026.05.283 claim-repair-v2",
        "paper_ready": False,
        "pilot_only_task_rows": [],
        "projection_only_task_rows": ["exp3000_archive_activation"],
        "source_artifacts_read": [
            {
                "experiment_id": experiment_id,
                "path": path,
                "present": present,
                "readable_json_object": present,
                "required": required,
                "sha256": "test-sha" if present else None,
            }
            for experiment_id, path, present, required in _SOURCE_ROWS
        ],
        "task_classification_counts": {
            "blocked": 1,
            "clean": 5,
            "flagged": 2,
            "gated-skipped": 1,
            "missing": 0,
            "pilot-only": 0,
            "projection-only": 1,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_referenced_results(root: Path, *, skip: str | None = None) -> None:
    matrix_path = root / exp.MATRIX_SOURCE
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(
        json.dumps(
            {
                "artifact": "experiment_3010_cross_corpus_matrix_v16",
                "honest_verdict": "complete: matrix_v16_ready=true",
                "matrix_v16_ready": True,
                "rows": _matrix_rows(),
            }
        ),
        encoding="utf-8",
    )
    for experiment_id, rel_path, present, _required in _SOURCE_ROWS:
        if rel_path in {skip, exp.MATRIX_SOURCE} or not present:
            continue
        result_path = root / rel_path
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps({"experiment_id": experiment_id, "honest_verdict": "complete: test"}),
            encoding="utf-8",
        )


def _complete_archive_text() -> str:
    task_rows = "\n".join(
        f"  - id: exp{number}\n"
        f"    title: Task {number}\n"
        f"    deliverable: results/experiment_{number}.json\n"
        f"    result: OK (conductor)"
        for number in range(3000, 3012)
    )
    return (
        "- id: 2026.05.282\n"
        "  title: Claim Repair + Metamorphic Validation + Attractor Memory + GateMate IO\n"
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
        'milestone: "2026.05.283"\n'
        'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        "tasks:\n"
        "  - id: exp3012-archive-v282-activate-v283\n"
        "    deliverable: results/experiment_3012_archive_v282_activate_v283.json\n"
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
    conductor.write_text("# untouched by Exp 3012\n", encoding="utf-8")
    ops = root / "ops"
    ops.mkdir(exist_ok=True)
    (ops / "status.md").write_text("# status\n", encoding="utf-8")
    (ops / "changelog.md").write_text("# changelog\n", encoding="utf-8")
    bmad = root / "_bmad"
    bmad.mkdir(exist_ok=True)
    (bmad / "traceability.md").write_text("# traceability\n", encoding="utf-8")
    if write_capstone:
        _write_capstone(root)
        _write_referenced_results(root, skip=skip_result)


def _milestone_rows(root: Path) -> list[dict]:
    complete = yaml.safe_load((root / "research-complete.yaml").read_text(encoding="utf-8"))
    rows = complete.get("milestones", []) if isinstance(complete, dict) else complete
    return [row for row in rows if str(row.get("id")) == "2026.05.282"]


def test_req_report_3012_spec_is_declared() -> None:
    """REQ-REPORT-3012: OpenSpec declares the archive/activation contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-3012" in spec
    assert "SCENARIO-REPORT-3012" in spec
    assert str(exp.DEFAULT_OUTPUT_PATH) in spec


def test_scenario_report_3012_existing_archive_uses_active_roadmap_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3012: existing .282 archive confirms active .283 roadmap."""

    _write_common_files(tmp_path, complete_text=_complete_archive_text())
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )
    before_status = (tmp_path / "ops" / "status.md").read_text(encoding="utf-8")
    before_changelog = (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8")
    before_traceability = (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(4.0, 5.5))
    written = json.loads((tmp_path / exp.DEFAULT_OUTPUT_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert "model_specs" not in artifact
    assert artifact["honest_verdict"] == exp.COMPLETE_VERDICT
    assert artifact["archive_ready"] is True
    assert artifact["archived_milestone"] == "2026.05.282"
    assert artifact["activated_milestone"] == "2026.05.283"
    assert artifact["research_complete_updated"] is True
    assert artifact["status_updates_written"] is False
    assert artifact["ops_docs_reconciliation_left_to_conductor"] is True
    assert artifact["n_tasks_archived"] == 12
    assert artifact["archive_already_present"] is True
    assert artifact["archive_appended_this_run"] is False
    assert artifact["capstone_source"] == "results/experiment_3011_capstone_v282.json"
    assert artifact["capstone_honest_verdict"].startswith("complete: capstone_ready=true")
    assert artifact["paper_ready_from_capstone"] is False
    assert artifact["artifact_classification_counts_from_capstone"] == {
        "blocked": 9,
        "clean": 38,
        "flagged": 23,
        "gated-skipped": 1,
        "missing": 1,
        "pilot-only": 4,
        "projection-only": 9,
    }
    assert artifact["task_classification_counts_from_capstone"] == {
        "blocked": 1,
        "clean": 5,
        "flagged": 2,
        "gated-skipped": 1,
        "missing": 0,
        "pilot-only": 0,
        "projection-only": 1,
    }
    assert artifact["artifact_classification_counts_from_matrix"] == {
        "blocked": 1,
        "clean": 1,
        "flagged": 2,
        "gated-skipped": 1,
        "missing": 1,
        "pilot-only": 1,
        "projection-only": 1,
    }
    assert artifact["n_milestone_282_source_artifacts_referenced"] == 11
    assert artifact["n_milestone_282_source_artifacts_read"] == 10
    assert artifact["missing_referenced_artifacts"] == [
        "results/experiment_3009_ssqa_dual_bram_rtl_pnr_resource_report_v2.json"
    ]
    assert [row["row_id"] for row in artifact["blocked_or_flagged_rows_carried_forward"]] == [
        "exp3003_metamorphic_repair",
        "exp3007_fr11_trace_memory_stability",
        "exp3008_gatemate_host_visible_io",
        "exp3009_ssqa_dual_bram_report",
        "carry_forward_missing",
    ]
    assert [row["row_id"] for row in artifact["adversarial_flags_carried_forward"]] == [
        "exp3000_archive_activation",
        "exp3003_metamorphic_repair",
        "exp3007_fr11_trace_memory_stability",
    ]
    assert artifact["adversarial_flag_count"] == 3
    assert artifact["validation_commands"] == exp.DEFAULT_VALIDATION_COMMANDS
    assert artifact["activation"]["roadmap_source"] == "research-roadmap.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is True
    assert artifact["activation"]["research_roadmap_next_exists"] is False
    assert artifact["activation"]["non_empty_tasks"] is True
    assert artifact["roadmap_verification"]["research_roadmap_yaml_modified"] is False
    assert artifact["ops_doc_verification"]["ops_status_modified"] is False
    assert artifact["ops_doc_verification"]["ops_changelog_modified"] is False
    assert artifact["ops_doc_verification"]["bmad_traceability_modified"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before_status
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before_changelog
    assert (tmp_path / "_bmad" / "traceability.md").read_text(
        encoding="utf-8"
    ) == before_traceability


def test_req_report_3012_appends_completed_archive_when_absent(tmp_path: Path) -> None:
    """REQ-REPORT-3012: absent .282 archive is appended with all completed tasks."""

    _write_common_files(
        tmp_path,
        complete_text="- id: 2026.05.281\n  completed: '2026-05-24'\n  tasks: []\n",
        next_roadmap=(
            'milestone: "2026.05.283"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
            "tasks:\n"
            "  - id: exp3012-archive-v282-activate-v283\n"
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
    assert artifact["n_tasks_archived"] == 12
    assert len(rows) == 1
    assert rows[0]["completed"] == "2026-05-24"
    assert len(rows[0]["tasks"]) == 12
    assert rows[0]["tasks"][-1]["deliverable"] == exp.CAPSTONE_SOURCE
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap


def test_req_report_3012_missing_capstone_blocks_without_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-3012: missing capstone exits blocked without archive mutation."""

    _write_common_files(
        tmp_path,
        complete_text="- id: 2026.05.281\n  tasks: []\n",
        write_capstone=False,
    )
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    absolute_output = tmp_path / "custom" / "experiment_3012.json"
    artifact = exp.run(root=tmp_path, output_path=absolute_output, clock=_clock(1.0, 1.1))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["archive_ready"] is False
    assert artifact["research_complete_updated"] is False
    assert artifact["n_tasks_archived"] == 0
    assert artifact["capstone_honest_verdict"] == ""
    assert artifact["artifact_classification_counts_from_matrix"] == {
        "blocked": 0,
        "clean": 0,
        "flagged": 0,
        "gated-skipped": 0,
        "missing": 0,
        "pilot-only": 0,
        "projection-only": 0,
    }
    assert artifact["blocked_or_flagged_rows_carried_forward"] == []
    assert artifact["adversarial_flags_carried_forward"] == []
    assert json.loads(absolute_output.read_text(encoding="utf-8")) == artifact
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap


def test_req_report_3012_wrong_or_empty_roadmap_blocks(tmp_path: Path) -> None:
    """REQ-REPORT-3012: wrong or empty activation roadmap is reported honestly."""

    _write_common_files(
        tmp_path,
        complete_text=_complete_archive_text(),
        active_roadmap='milestone: "2026.05.282"\ntasks: []\n',
    )

    artifact = exp.run(root=tmp_path, clock=_clock(2.0, 2.2))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["archive_ready"] is True
    assert "roadmap milestone is not 2026.05.283" in artifact["blocked_reasons"]
    assert "roadmap has no tasks for 2026.05.283" in artifact["blocked_reasons"]


def test_req_report_3012_helpers_handle_bad_inputs_and_doc_status(tmp_path: Path) -> None:
    """REQ-REPORT-3012: helper readers tolerate malformed inputs and doc status."""

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
        "- id: 2026.05.282\n"
        "  tasks: []\n"
        "- id: 2026.05.282\n"
        "  completed: '2026-05-24'\n"
        "  tasks:\n"
        "  - id: exp3000\n"
    )
    assert exp._archive_completed_block_present(complete) is True
    assert exp._archive_completed_block_count(complete) == 1
    assert exp._archive_task_count(complete) == 1
    assert exp._archive_task_count("milestones: not-a-list\n") == 0
    assert exp._archive_completed_block_present("[]\n") is False
    assert exp._matrix_rows("not-a-list") == []

    docs = tmp_path / "docs"
    (docs / "ops").mkdir(parents=True)
    (docs / "_bmad").mkdir(parents=True)
    (docs / "ops" / "status.md").write_text("2026.05.282 archive and 2026.05.283 activation\n")
    (docs / "ops" / "changelog.md").write_text(
        "archived_milestone=2026.05.282 activated_milestone=2026.05.283\n"
    )
    assert exp._status_updates_written(docs) is True

    counts = exp._classification_counts_from_rows(
        [
            {"status": "clean"},
            {"status": "flagged"},
            {"status": "blocked"},
            {"status": "projection-only"},
            {"status": "pilot-only"},
            {"status": "missing"},
            {"status": "gated-skipped"},
            {"status": "other"},
            {"status": ""},
        ]
    )
    assert counts == {
        "blocked": 1,
        "clean": 1,
        "flagged": 1,
        "gated-skipped": 1,
        "missing": 1,
        "pilot-only": 1,
        "projection-only": 1,
    }

    _write_common_files(
        tmp_path,
        complete_text=_complete_archive_text(),
        skip_result="results/experiment_3008_gatemate_host_visible_io_transport_v2.json",
    )
    capstone = exp._read_json_mapping(tmp_path / exp.CAPSTONE_SOURCE)
    read_summary = exp._load_all_referenced_artifacts(tmp_path, capstone)
    assert read_summary["n_milestone_282_source_artifacts_referenced"] == 11
    assert read_summary["n_milestone_282_source_artifacts_read"] == 9
    assert read_summary["missing_referenced_artifacts"] == [
        "results/experiment_3008_gatemate_host_visible_io_transport_v2.json",
        "results/experiment_3009_ssqa_dual_bram_rtl_pnr_resource_report_v2.json",
    ]
    no_path_summary = exp._load_all_referenced_artifacts(
        tmp_path, {"source_artifacts_read": [{"experiment_id": "exp3000"}]}
    )
    assert no_path_summary["n_milestone_282_source_artifacts_referenced"] == 0

    verdict, reasons = exp._honest_verdict(
        archive_ready=False,
        research_complete_updated=False,
        roadmap={"milestone_matches": True, "non_empty_tasks": True},
        capstone_loaded=True,
    )
    assert verdict.startswith("blocked:")
    assert reasons == ["research-complete.yaml does not archive 2026.05.282"]

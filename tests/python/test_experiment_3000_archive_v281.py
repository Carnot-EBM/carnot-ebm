"""Tests for Exp 3000 milestone .281 archive and .282 activation.

Spec refs: REQ-REPORT-3000, SCENARIO-REPORT-3000.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.reporting import milestone_281_archive_282_activation as exp


def _clock(*values: float):
    times = iter(values)
    return lambda: next(times)


_SOURCE_ROWS = [
    ("exp2998", "results/experiment_2998_cross_corpus_matrix_v15.json", True, True),
    ("exp2987", "results/experiment_2987_capstone_v280.json", True, True),
    ("exp2988", "results/experiment_2988_archive_v280_activate_v281.json", True, False),
    ("exp2989", "results/experiment_2989_sota_gguf_cache_provenance_preflight_v1.json", True, False),
    ("exp2990", "results/experiment_2990_verifier_backed_hard_code_stress_manifest_v1.json", True, False),
    (
        "exp2991",
        "results/experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1.json",
        True,
        False,
    ),
    (
        "exp2992",
        "results/experiment_2992_sota_solver_formalization_provenance_reproduction_v1.json",
        True,
        False,
    ),
    (
        "exp2993",
        "results/experiment_2993_aquaforte_beaver_substrate_corrigendum_v1.json",
        True,
        False,
    ),
    ("exp2994", "results/experiment_2994_prompt_validator_dialogue_schema_v1.json", True, False),
    ("exp2995", "results/experiment_2995_fr11_verifier_grounded_trace_memory_v2.json", True, False),
    (
        "exp2996",
        "results/experiment_2996_gatemate_host_visible_readback_smoke_v1.json",
        True,
        False,
    ),
    (
        "exp2997",
        "results/experiment_2997_ssqa_dual_bram_rtl_pnr_resource_report_v1.json",
        False,
        False,
    ),
]


def _matrix_rows() -> list[dict[str, object]]:
    return [
        {
            "row_id": "exp2988_archive_activation",
            "source_experiment_id": "exp2988",
            "source_honest_verdict": "complete: archive_ready=true",
            "status": "projection-only",
            "upstream_flags": [],
        },
        {
            "row_id": "exp2991_intent_preserving_repair",
            "source_experiment_id": "exp2991",
            "source_honest_verdict": "flagged: hard-set repair did not clear promotion gates",
            "status": "flagged",
            "upstream_flags": ["TAUTOLOGY:critical", "METHODOLOGY_MISSING:warn"],
        },
        {
            "row_id": "exp2993_aquaforte_beaver_substrate",
            "source_experiment_id": "exp2993",
            "source_honest_verdict": "complete: live retry measured separately",
            "status": "flagged",
            "upstream_flags": ["DURATION_TOO_SHORT:critical"],
        },
        {
            "row_id": "exp2995_fr11_trace_memory",
            "source_experiment_id": "exp2995",
            "source_honest_verdict": "ready: verifier_grounded_trace_memory_ready",
            "status": "clean",
            "upstream_flags": [],
        },
        {
            "row_id": "exp2996_gatemate_readback_smoke",
            "source_experiment_id": "exp2996",
            "source_honest_verdict": "blocked_flash_failed",
            "status": "blocked",
            "upstream_flags": [],
        },
        {
            "row_id": "exp2997_ssqa_dual_bram_rtl_pnr",
            "source_experiment_id": "exp2997",
            "source_honest_verdict": "",
            "status": "missing",
            "upstream_flags": [],
        },
        {
            "row_id": "carry_forward_pilot",
            "source_experiment_id": "exp2973",
            "source_honest_verdict": "complete: carried forward",
            "status": "pilot-only",
            "upstream_flags": [],
        },
    ]


def _write_capstone(root: Path) -> None:
    path = root / exp.CAPSTONE_SOURCE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "experiment_2999_capstone_v281",
        "artifact_classification_counts": {
            "blocked": 8,
            "clean": 34,
            "flagged": 20,
            "gated-skipped": 0,
            "missing": 1,
            "pilot-only": 4,
            "projection-only": 8,
        },
        "blocked_artifacts": ["exp2996_gatemate_readback_smoke"],
        "capstone_ready": True,
        "clean_artifacts": ["exp2989_sota_cache", "exp2995_fr11_trace_memory"],
        "flagged_artifacts": [
            "exp2991_intent_preserving_repair",
            "exp2993_aquaforte_beaver_substrate",
        ],
        "gaps_remaining": ["GateMate remains blocked.", "SSQA remains missing."],
        "gated_skipped_artifacts": [],
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; clean=34; "
            "flagged=20; blocked=8; missing=1; gated_skipped=0"
        ),
        "inference_substrate": exp.INFERENCE_SUBSTRATE,
        "milestone": "2026.05.281",
        "missing_artifacts": ["exp2997_ssqa_dual_bram_rtl_pnr"],
        "next_milestone_recommendations": ["Repair flagged rows before publication."],
        "paper_ready": False,
        "pilot_only_artifacts": ["carry_forward_pilot"],
        "projection_only_artifacts": ["exp2988_archive_activation"],
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
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_referenced_results(root: Path, *, skip: str | None = None) -> None:
    matrix_path = root / exp.MATRIX_SOURCE
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(
        json.dumps(
            {
                "artifact": "experiment_2998_cross_corpus_matrix_v15",
                "honest_verdict": "complete: matrix_v15_ready=true",
                "matrix_v15_ready": True,
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
        for number in range(2988, 3000)
    )
    return (
        "- id: 2026.05.281\n"
        "  title: SOTA Cache Recovery + Verifier-Backed Repair + Provenance-Grounded Self-Learning\n"
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
        'milestone: "2026.05.282"\n'
        'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        "tasks:\n"
        "  - id: exp3000-archive-v281-activate-v282\n"
        "    deliverable: results/experiment_3000_archive_v281_activate_v282.json\n"
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
    conductor.write_text("# untouched by Exp 3000\n", encoding="utf-8")
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
    return [row for row in rows if str(row.get("id")) == "2026.05.281"]


def test_req_report_3000_spec_is_declared() -> None:
    """REQ-REPORT-3000: OpenSpec declares the archive/activation contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-3000" in spec
    assert "SCENARIO-REPORT-3000" in spec
    assert str(exp.DEFAULT_OUTPUT_PATH) in spec


def test_scenario_report_3000_existing_archive_uses_active_roadmap_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3000: existing .281 archive confirms active .282 roadmap."""

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
    assert artifact["honest_verdict"] == exp.COMPLETE_VERDICT
    assert artifact["archive_ready"] is True
    assert artifact["archived_milestone"] == "2026.05.281"
    assert artifact["activated_milestone"] == "2026.05.282"
    assert artifact["research_complete_updated"] is True
    assert artifact["status_updates_written"] is False
    assert artifact["ops_docs_reconciliation_left_to_conductor"] is True
    assert artifact["n_tasks_archived"] == 12
    assert artifact["archive_already_present"] is True
    assert artifact["archive_appended_this_run"] is False
    assert artifact["capstone_source"] == "results/experiment_2999_capstone_v281.json"
    assert artifact["capstone_honest_verdict"].startswith("complete: capstone_ready=true")
    assert artifact["paper_ready_from_capstone"] is False
    assert artifact["artifact_classification_counts_from_capstone"] == {
        "blocked": 8,
        "clean": 34,
        "flagged": 20,
        "gated-skipped": 0,
        "missing": 1,
        "pilot-only": 4,
        "projection-only": 8,
    }
    assert artifact["artifact_classification_counts_from_matrix"] == {
        "blocked": 1,
        "clean": 1,
        "flagged": 2,
        "gated-skipped": 0,
        "missing": 1,
        "pilot-only": 1,
        "projection-only": 1,
    }
    assert artifact["n_milestone_281_source_artifacts_referenced"] == 11
    assert artifact["n_milestone_281_source_artifacts_read"] == 10
    assert artifact["missing_referenced_artifacts"] == [
        "results/experiment_2997_ssqa_dual_bram_rtl_pnr_resource_report_v1.json"
    ]
    assert [row["row_id"] for row in artifact["blocked_or_flagged_rows_carried_forward"]] == [
        "exp2991_intent_preserving_repair",
        "exp2993_aquaforte_beaver_substrate",
        "exp2996_gatemate_readback_smoke",
        "exp2997_ssqa_dual_bram_rtl_pnr",
    ]
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


def test_req_report_3000_appends_completed_archive_when_absent(tmp_path: Path) -> None:
    """REQ-REPORT-3000: absent .281 archive is appended with all completed tasks."""

    _write_common_files(
        tmp_path,
        complete_text="- id: 2026.05.280\n  completed: '2026-05-24'\n  tasks: []\n",
        next_roadmap=(
            'milestone: "2026.05.282"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
            "tasks:\n"
            "  - id: exp3000-archive-v281-activate-v282\n"
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


def test_req_report_3000_missing_capstone_blocks_without_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-3000: missing capstone exits blocked without archive mutation."""

    _write_common_files(
        tmp_path,
        complete_text="- id: 2026.05.280\n  tasks: []\n",
        write_capstone=False,
    )
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    absolute_output = tmp_path / "custom" / "experiment_3000.json"
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
    assert json.loads(absolute_output.read_text(encoding="utf-8")) == artifact
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap


def test_req_report_3000_wrong_or_empty_roadmap_blocks(tmp_path: Path) -> None:
    """REQ-REPORT-3000: wrong or empty activation roadmap is reported honestly."""

    _write_common_files(
        tmp_path,
        complete_text=_complete_archive_text(),
        active_roadmap='milestone: "2026.05.281"\ntasks: []\n',
    )

    artifact = exp.run(root=tmp_path, clock=_clock(2.0, 2.2))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["archive_ready"] is True
    assert "roadmap milestone is not 2026.05.282" in artifact["blocked_reasons"]
    assert "roadmap has no tasks for 2026.05.282" in artifact["blocked_reasons"]


def test_req_report_3000_helpers_handle_bad_inputs_and_doc_status(tmp_path: Path) -> None:
    """REQ-REPORT-3000: helper readers tolerate malformed inputs and doc status."""

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
        "- id: 2026.05.281\n"
        "  tasks: []\n"
        "- id: 2026.05.281\n"
        "  completed: '2026-05-24'\n"
        "  tasks:\n"
        "  - id: exp2988\n"
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
    (docs / "ops" / "status.md").write_text("2026.05.281 archive and 2026.05.282 activation\n")
    (docs / "ops" / "changelog.md").write_text(
        "archived_milestone=2026.05.281 activated_milestone=2026.05.282\n"
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
        skip_result="results/experiment_2996_gatemate_host_visible_readback_smoke_v1.json",
    )
    capstone = exp._read_json_mapping(tmp_path / exp.CAPSTONE_SOURCE)
    read_summary = exp._load_all_referenced_artifacts(tmp_path, capstone)
    assert read_summary["n_milestone_281_source_artifacts_referenced"] == 11
    assert read_summary["n_milestone_281_source_artifacts_read"] == 9
    assert read_summary["missing_referenced_artifacts"] == [
        "results/experiment_2996_gatemate_host_visible_readback_smoke_v1.json",
        "results/experiment_2997_ssqa_dual_bram_rtl_pnr_resource_report_v1.json",
    ]
    no_path_summary = exp._load_all_referenced_artifacts(
        tmp_path, {"source_artifacts_read": [{"experiment_id": "exp2998"}]}
    )
    assert no_path_summary["n_milestone_281_source_artifacts_referenced"] == 0

    verdict, reasons = exp._honest_verdict(
        archive_ready=False,
        research_complete_updated=False,
        roadmap={"milestone_matches": True, "non_empty_tasks": True},
        capstone_loaded=True,
    )
    assert verdict.startswith("blocked:")
    assert reasons == ["research-complete.yaml does not archive 2026.05.281"]

"""Tests for Exp 3026 milestone .283 archive and .284 activation.

Spec refs: REQ-REPORT-3026, SCENARIO-REPORT-3026.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import milestone_283_archive_284_activation as exp


def _clock(*values: float):
    times = iter(values)
    return lambda: next(times)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _capstone_payload(*, capstone_ready: bool = True) -> dict[str, object]:
    return {
        "artifact": "experiment_3025_capstone_v283",
        "blocked_rows": ["exp3021_gatemate_transport_shim"],
        "blocked_task_rows": ["exp3021_gatemate_transport_shim"],
        "capstone_ready": capstone_ready,
        "clean_task_rows": [
            "exp3017_instruction_validator_tree",
            "exp3020_fr11_verifier_feedback_controller",
            "exp3024_cross_corpus_matrix_v17",
        ],
        "flagged_adversarial": True,
        "flagged_rows": ["exp3016_repair_acceptance_controller"],
        "flagged_task_rows": ["exp3016_repair_acceptance_controller"],
        "gated_skipped_rows": [
            "exp3022_gatemate_transport_flash_smoke",
            "exp3023_ssqa_explicit_gate_artifact",
        ],
        "gated_skipped_task_rows": [
            "exp3022_gatemate_transport_flash_smoke",
            "exp3023_ssqa_explicit_gate_artifact",
        ],
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; repaired=3; "
            "flagged=29; blocked=10; gated_skipped=3; missing=1"
        ),
        "matrix_recommended_next_actions": [
            "Exp3016: do not promote the repair row until adversarial flags are cleared.",
            "Exp3022: obtain a physical GateMate pinout or supported host-visible transport.",
        ],
        "matrix_status_counts": {
            "blocked": 10,
            "clean": 40,
            "flagged": 29,
            "gated-skipped": 3,
            "missing": 1,
            "pilot-only": 4,
            "projection-only": 10,
        },
        "matrix_still_blocked_claims": [
            "exp3016_repair_acceptance_controller_flagged",
            "exp3022_gatemate_transport_flash_smoke_gated_skipped",
        ],
        "milestone": "2026.05.283",
        "missing_rows": ["carry_forward_v16:exp2997_ssqa_dual_bram_rtl_pnr"],
        "missing_task_rows": [],
        "paper_ready": False,
        "paper_ready_blockers": [
            "repair row exp3016_repair_acceptance_controller is flagged",
            "matrix contains non-clean rows: flagged=29, blocked=10, gated_skipped=3, missing=1",
        ],
        "pilot_only_rows": ["corpus:MBPP"],
        "pilot_only_task_rows": [],
        "projection_only_rows": ["exp3012_archive_activation"],
        "projection_only_task_rows": ["exp3012_archive_activation"],
        "task_classification_counts": {
            "blocked": 1,
            "clean": 3,
            "flagged": 6,
            "gated-skipped": 2,
            "missing": 0,
            "pilot-only": 0,
            "projection-only": 1,
        },
    }


def _matrix_payload() -> dict[str, object]:
    return {
        "artifact": "experiment_3024_cross_corpus_matrix_v17",
        "blocked_count": 10,
        "clean_count": 40,
        "flagged_adversarial": True,
        "flagged_count": 29,
        "gated_skipped_count": 3,
        "honest_verdict": (
            "complete: matrix_v17_ready=true; clean=40; flagged=29; blocked=10; "
            "gated_skipped=3; projection_only=10; pilot_only=4; missing=1"
        ),
        "matrix_v17_ready": True,
        "milestone": "2026.05.283",
        "missing_count": 1,
        "pilot_only_count": 4,
        "projection_only_count": 10,
        "recommended_next_actions": [
            "Exp3016: do not promote the repair row until adversarial flags are cleared.",
        ],
        "rows": [
            {
                "row_id": "exp3016_repair_acceptance_controller",
                "source_experiment_id": "exp3016",
                "source_honest_verdict": "complete: acceptance-controlled repair gates passed",
                "status": "flagged",
                "upstream_flags": [
                    "flagged_adversarial=true",
                    "TAUTOLOGY:critical",
                    "METHODOLOGY_MISSING:warn",
                ],
            },
            {
                "row_id": "exp3021_gatemate_transport_shim",
                "source_experiment_id": "exp3021",
                "source_honest_verdict": "blocked: GateMate pinout unavailable",
                "status": "blocked",
                "upstream_flags": [],
            },
            {
                "row_id": "exp3022_gatemate_transport_flash_smoke",
                "source_experiment_id": "exp3022",
                "source_honest_verdict": "gated-skipped: host_visible_io_ready=false",
                "status": "gated-skipped",
                "upstream_flags": [],
            },
            {
                "row_id": "exp3020_fr11_verifier_feedback_controller",
                "source_experiment_id": "exp3020",
                "source_honest_verdict": "complete: bounded verifier-feedback utility",
                "status": "clean",
                "upstream_flags": [],
            },
        ],
        "still_blocked_claims": [
            "exp3016_repair_acceptance_controller_flagged",
            "exp3022_gatemate_transport_flash_smoke_gated_skipped",
        ],
    }


def _archive_text() -> str:
    task_rows = "\n".join(
        f"  - id: exp{number}\n"
        f"    title: Task {number}\n"
        f"    deliverable: results/experiment_{number}.json\n"
        f"    result: OK (conductor)"
        for number in range(3012, 3026)
    )
    return (
        "milestones:\n"
        "- id: 2026.05.283\n"
        "  title: Claim Repair v2 + Feasibility-Gated Self-Learning + GateMate IO Boundary\n"
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md\n"
        "  completed: '2026-05-25'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        f"{task_rows}\n"
    )


def _active_roadmap(milestone: str = "2026.05.284") -> str:
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "Repair Corrigendum + FR-11 Held-Out Learning + GateMate Output Contract"\n'
        'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        "tasks:\n"
        "  - id: exp3026-archive-v283-activate-v284\n"
        "    deliverable: results/experiment_3026_archive_v283_activate_v284.json\n"
        "  - id: exp3027-adversarial-flag-methodology-corrigendum\n"
        "    deliverable: results/experiment_3027_adversarial_flag_methodology_corrigendum_v1.json\n"
    )


def _write_common_files(
    root: Path,
    *,
    capstone_ready: bool = True,
    archive_text: str | None = None,
    active_roadmap: str | None = None,
    next_roadmap: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(
        _archive_text() if archive_text is None else archive_text,
        encoding="utf-8",
    )
    (root / "research-roadmap.yaml").write_text(
        _active_roadmap() if active_roadmap is None else active_roadmap,
        encoding="utf-8",
    )
    if next_roadmap is not None:
        (root / "research-roadmap-next.yaml").write_text(next_roadmap, encoding="utf-8")
    conductor = root / "scripts" / "research_conductor.py"
    conductor.parent.mkdir(parents=True, exist_ok=True)
    conductor.write_text("# protected conductor\n", encoding="utf-8")
    _write_json(root / exp.CAPSTONE_SOURCE, _capstone_payload(capstone_ready=capstone_ready))
    _write_json(root / exp.MATRIX_SOURCE, _matrix_payload())


def test_req_report_3026_spec_is_declared() -> None:
    """REQ-REPORT-3026: OpenSpec declares the archive/activation contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-3026" in spec
    assert "SCENARIO-REPORT-3026" in spec
    assert str(exp.DEFAULT_OUTPUT_PATH) in spec


def test_scenario_report_3026_uses_active_roadmap_fallback(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3026: completed .283 archive confirms active .284 roadmap."""

    _write_common_files(tmp_path)
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )

    artifact = exp.run(root=tmp_path, clock=_clock(2.0, 3.25))
    written = json.loads((tmp_path / exp.DEFAULT_OUTPUT_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert "model_specs" not in artifact
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["milestone_archived"] is True
    assert artifact["next_milestone"] == "2026.05.284"
    assert artifact["next_roadmap_path"] == "research-roadmap.yaml"
    assert artifact["capstone_ready"] is True
    assert artifact["previous_paper_ready"] is False
    assert artifact["protected_files_unchanged"] is True
    assert artifact["inference_substrate"] == {
        "mode": "aggregation",
        "source": "checked_in_artifacts",
        "live_inference": False,
        "llm_calls": False,
        "gpu_required": False,
    }
    assert artifact["milestone_283_summary"]["matrix_status_summary"] == {
        "blocked": 10,
        "clean": 40,
        "flagged": 29,
        "gated_skipped": 3,
        "missing": 1,
        "pilot_only": 4,
        "projection_only": 10,
        "paper_ready": False,
    }
    assert artifact["milestone_283_summary"]["adversarially_flagged_count"] == 1
    assert artifact["roadmap_activation"]["used_active_roadmap_fallback"] is True
    assert artifact["roadmap_activation"]["requested_staged_roadmap_exists"] is False
    assert artifact["next_execution_order"] == [
        "exp3026-archive-v283-activate-v284",
        "exp3027-adversarial-flag-methodology-corrigendum",
    ]
    assert [blocker["kind"] for blocker in artifact["carry_forward_blockers"]] == [
        "paper_ready_blockers",
        "status_bucket",
        "status_bucket",
        "status_bucket",
        "status_bucket",
        "status_bucket",
        "status_bucket",
        "adversarial_flags",
        "matrix_still_blocked_claims",
        "recommended_next_actions",
    ]
    assert artifact["carry_forward_blockers"][1]["status"] == "flagged"
    assert artifact["carry_forward_blockers"][1]["rows"] == [
        "exp3016_repair_acceptance_controller"
    ]
    assert artifact["carry_forward_blockers"][7]["rows"][0]["row_id"] == (
        "exp3016_repair_acceptance_controller"
    )
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_3026_prefers_staged_roadmap_when_present(tmp_path: Path) -> None:
    """REQ-REPORT-3026: research-roadmap-next.yaml is preferred when present."""

    _write_common_files(
        tmp_path,
        active_roadmap=_active_roadmap("2026.05.283"),
        next_roadmap=_active_roadmap("2026.05.284"),
    )

    artifact = exp.run(root=tmp_path, clock=_clock(10.0, 10.1))

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["next_roadmap_path"] == "research-roadmap-next.yaml"
    assert artifact["roadmap_activation"]["used_active_roadmap_fallback"] is False
    assert artifact["roadmap_activation"]["active_roadmap_milestone"] == "2026.05.283"


def test_req_report_3026_blocks_when_preconditions_fail(tmp_path: Path) -> None:
    """REQ-REPORT-3026: blocked preconditions are reported without protected mutations."""

    _write_common_files(
        tmp_path,
        capstone_ready=False,
        archive_text="milestones:\n- id: 2026.05.282\n  tasks: []\n",
        active_roadmap='milestone: "2026.05.284"\ntasks: []\n',
    )
    output = tmp_path / "custom" / "exp3026.json"

    artifact = exp.run(root=tmp_path, output_path=output, clock=_clock(1.0, 1.5))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["milestone_archived"] is False
    assert artifact["capstone_ready"] is False
    assert artifact["protected_files_unchanged"] is True
    assert "Exp 3025 capstone is not capstone_ready=true" in artifact["blocked_reasons"]
    assert "research-complete.yaml does not contain completed 2026.05.283 archive" in artifact[
        "blocked_reasons"
    ]
    assert "roadmap for 2026.05.284 has no tasks" in artifact["blocked_reasons"]
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_req_report_3026_helpers_handle_bad_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-3026: helper readers and summaries tolerate absent or malformed input."""

    assert exp._read_text(tmp_path / "missing.txt") == ""
    assert exp._read_json_mapping(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad json", encoding="utf-8")
    assert exp._read_json_mapping(bad_json) == {}
    array_json = tmp_path / "array.json"
    array_json.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp._read_json_mapping(array_json) == {}

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("milestone: [unterminated\n", encoding="utf-8")
    assert exp._load_yaml_mapping(bad_yaml) == {}
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("\n", encoding="utf-8")
    assert exp._load_yaml_mapping(empty_yaml) == {}
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    assert exp._load_yaml_mapping(list_yaml) == {}

    assert exp._as_str_list(["x", 2]) == ["x", "2"]
    assert exp._as_str_list("x") == []
    assert exp._as_int_mapping({"clean": 1, "flagged": True, "missing": "1"}) == {"clean": 1}
    assert exp._as_int_mapping(["not", "mapping"]) == {}
    assert exp._task_ids({"tasks": [{"id": "a"}, {"missing": "id"}, "bad"]}) == ["a"]
    assert exp._task_ids({"tasks": "not-a-list"}) == []
    assert exp._matrix_rows({"rows": [{"row_id": "r"}, "bad"]}) == [{"row_id": "r"}]
    assert exp._matrix_rows(["bad"]) == []
    assert exp._count_for_status(
        "clean",
        matrix={"clean_count": True},
        capstone_counts={"clean": 7},
    ) == 7

    before = {"research-roadmap.yaml": {"exists": True, "sha256": "a"}}
    after = {"research-roadmap.yaml": {"exists": True, "sha256": "b"}}
    assert exp._protected_unchanged(before, after) is False

    verdict, reasons = exp._honest_verdict(
        capstone_present=False,
        capstone_ready=False,
        archive={"milestone_archived": True},
        roadmap={
            "milestone_matches": False,
            "milestone_doc_matches": True,
            "non_empty_tasks": True,
        },
        protected_files_unchanged=False,
    )
    assert verdict.startswith("blocked:")
    assert reasons == [
        "Exp 3025 capstone source missing or invalid",
        "roadmap milestone is not 2026.05.284",
        "protected files changed during archive activation",
    ]

"""Tests for Exp 2962 milestone .278 archive and .279 activation.

Spec refs: REQ-REPORT-2962, SCENARIO-REPORT-2962.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.reporting import milestone_278_archive_279_activation as exp


def _clock(*values: float):
    times = iter(values)
    return lambda: next(times)


def _write_capstone(root: Path) -> None:
    path = root / exp.CAPSTONE_SOURCE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "experiment_2961_capstone_v278",
        "artifact_classification_counts": {
            "aggregation-only": 2,
            "blocked": 1,
            "clean": 5,
            "flagged": 5,
            "gated-skipped": 0,
            "missing": 0,
            "pilot-only": 0,
        },
        "blocked_artifacts": ["exp2957"],
        "clean_artifacts": ["exp2949", "exp2953", "exp2955", "exp2956", "exp2958"],
        "flagged_artifacts": ["exp2950", "exp2951", "exp2952", "exp2954", "exp2959"],
        "gaps_remaining": [
            "Taxonomy-guided repair delta remains flagged; rerun or audit before paper use.",
            "FR-11 utility-gated self-learning remains flagged despite positive utility.",
            "GateMate flash/timing smoke remains blocked by board detection.",
        ],
        "headline_outcome": (
            "partial: materialization and threshold-policy evidence improved, "
            "but 5 unresolved gaps keep paper readiness unchanged"
        ),
        "honest_verdict": (
            "complete: milestone_278_capstone; paper_ready=false; "
            "clean=5; flagged=5; blocked=1; missing=0"
        ),
        "inference_substrate": exp.INFERENCE_SUBSTRATE,
        "milestone": "2026.05.278",
        "missing_artifacts": [],
        "next_milestone_recommendations": [
            "GateMate .279: resolve board detection and capture flash/timing/output-hash smoke."
        ],
        "paper_ready": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_common_files(
    root: Path,
    *,
    complete_text: str,
    active_roadmap: str = (
        'milestone: "2026.05.279"\n'
        'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
    ),
    next_roadmap: str | None = None,
    write_capstone: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(active_roadmap, encoding="utf-8")
    if next_roadmap is not None:
        (root / "research-roadmap-next.yaml").write_text(next_roadmap, encoding="utf-8")
    conductor = root / "scripts" / "research_conductor.py"
    conductor.parent.mkdir(parents=True, exist_ok=True)
    conductor.write_text("# untouched by Exp 2962\n", encoding="utf-8")
    if write_capstone:
        _write_capstone(root)


def test_req_report_2962_spec_is_declared() -> None:
    """REQ-REPORT-2962: OpenSpec declares the archive/activation contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2962" in spec
    assert "SCENARIO-REPORT-2962" in spec
    assert str(exp.DEFAULT_OUTPUT_PATH) in spec


def test_scenario_report_2962_existing_archive_uses_active_roadmap_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2962: existing .278 archive confirms active .279 roadmap."""

    complete_text = """milestones:
- id: 2026.05.277
  completed: '2026-05-23'
  tasks: []
- id: 2026.05.278
  title: Structured Code Repair + Utility-Gated Self-Learning + GateMate Materialization
  doc: openspec/change-proposals/research-roadmap-vNEXT.md
  completed: '2026-05-24'
  tasks:
  - id: exp2961
    deliverable: results/experiment_2961_capstone_v278.json
    result: OK (conductor)
"""
    _write_common_files(tmp_path, complete_text=complete_text)
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (tmp_path / "scripts/research_conductor.py").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(4.0, 5.25))
    written = json.loads((tmp_path / exp.DEFAULT_OUTPUT_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"] == exp.COMPLETE_VERDICT
    assert artifact["archive_ready"] is True
    assert artifact["archived_milestone"] == "2026.05.278"
    assert artifact["activated_milestone"] == "2026.05.279"
    assert artifact["archive_already_present"] is True
    assert artifact["archive_appended_this_run"] is False
    assert artifact["capstone_source"] == "results/experiment_2961_capstone_v278.json"
    assert artifact["capstone_honest_verdict"].startswith("complete: milestone_278_capstone")
    assert artifact["paper_ready_from_capstone"] is False
    assert artifact["headline_outcome_from_capstone"].startswith("partial:")
    assert artifact["clean_artifacts_from_capstone"] == [
        "exp2949",
        "exp2953",
        "exp2955",
        "exp2956",
        "exp2958",
    ]
    assert artifact["flagged_artifacts_from_capstone"] == [
        "exp2950",
        "exp2951",
        "exp2952",
        "exp2954",
        "exp2959",
    ]
    assert artifact["blocked_artifacts_from_capstone"] == ["exp2957"]
    assert artifact["missing_artifacts_from_capstone"] == []
    assert artifact["artifact_classification_counts_from_capstone"] == {
        "aggregation-only": 2,
        "blocked": 1,
        "clean": 5,
        "flagged": 5,
        "gated-skipped": 0,
        "missing": 0,
        "pilot-only": 0,
    }
    assert artifact["next_gaps_from_capstone"] == [
        "Taxonomy-guided repair delta remains flagged; rerun or audit before paper use.",
        "FR-11 utility-gated self-learning remains flagged despite positive utility.",
        "GateMate flash/timing smoke remains blocked by board detection.",
    ]
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["run_date"] == "20260524"
    assert artifact["duration_s"] == 1.25
    assert artifact["activation"]["roadmap_source"] == "research-roadmap.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is True
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_2962_appends_minimal_archive_when_absent(tmp_path: Path) -> None:
    """REQ-REPORT-2962: absent .278 archive is appended without touching roadmap."""

    _write_common_files(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.277\n  tasks: []\n",
        next_roadmap=(
            'milestone: "2026.05.279"\n'
            'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        ),
    )
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(10.0, 10.5))
    complete = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    row = [m for m in complete["milestones"] if str(m.get("id")) == "2026.05.278"]

    assert artifact["honest_verdict"] == exp.COMPLETE_VERDICT
    assert artifact["archive_ready"] is True
    assert artifact["archive_already_present"] is False
    assert artifact["archive_appended_this_run"] is True
    assert artifact["activation"]["roadmap_source"] == "research-roadmap-next.yaml"
    assert artifact["activation"]["used_active_roadmap_fallback"] is False
    assert len(row) == 1
    assert row[0]["completed"] == "2026-05-24"
    assert row[0]["tasks"][0]["deliverable"] == exp.CAPSTONE_SOURCE
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap


def test_req_report_2962_missing_capstone_blocks_without_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-2962: missing capstone exits with blocked_capstone_missing."""

    _write_common_files(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.277\n  tasks: []\n",
        active_roadmap='milestone: "2026.05.279"\n',
        write_capstone=False,
    )
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (tmp_path / "scripts/research_conductor.py").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(1.0, 1.1))

    assert artifact["honest_verdict"] == "blocked_capstone_missing"
    assert artifact["archive_ready"] is False
    assert artifact["archived_milestone"] == "2026.05.278"
    assert artifact["activated_milestone"] == "2026.05.279"
    assert artifact["capstone_source"] == "results/experiment_2961_capstone_v278.json"
    assert artifact["capstone_honest_verdict"] == ""
    assert artifact["paper_ready_from_capstone"] is False
    assert artifact["headline_outcome_from_capstone"] == ""
    assert artifact["clean_artifacts_from_capstone"] == []
    assert artifact["flagged_artifacts_from_capstone"] == []
    assert artifact["blocked_artifacts_from_capstone"] == []
    assert artifact["missing_artifacts_from_capstone"] == []
    assert artifact["next_gaps_from_capstone"] == []
    assert artifact["artifact_classification_counts_from_capstone"] == {}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == 0.1
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (tmp_path / "scripts/research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_2962_blocks_on_wrong_roadmap(tmp_path: Path) -> None:
    """REQ-REPORT-2962: wrong activation milestone is reported honestly."""

    _write_common_files(
        tmp_path,
        complete_text=("milestones:\n- id: 2026.05.278\n  completed: '2026-05-24'\n  tasks: []\n"),
        active_roadmap='milestone: "2026.05.278"\n',
    )

    artifact = exp.run(root=tmp_path, clock=_clock(2.0, 2.2))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["archive_ready"] is True
    assert "roadmap milestone is not 2026.05.279" in artifact["blocked_reasons"]


def test_req_report_2962_helpers_handle_bad_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-2962: helper readers tolerate missing or malformed inputs."""

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
        "milestones:\n"
        "- id: 2026.05.278\n"
        "  tasks: []\n"
        "- id: 2026.05.278\n"
        "  completed: '2026-05-24'\n"
    )
    assert exp._archive_completed_block_present(complete) is True
    assert exp._archive_completed_block_present("milestones: []\n") is False

    assert exp._headline_outcome({}) == ""
    assert exp._headline_outcome({"headline_outcome": "partial"}) == "partial"
    assert exp._next_gaps({"gaps_remaining": ["gap"]}) == ["gap"]
    assert exp._next_gaps({"next_milestone_recommendations": ["action"]}) == ["action"]

    verdict, reasons = exp._honest_verdict(
        archive_ready=False,
        roadmap={"milestone_matches": True},
    )
    assert verdict.startswith("blocked:")
    assert reasons == ["research-complete.yaml does not archive 2026.05.278"]

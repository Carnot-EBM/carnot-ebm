"""Tests for Exp5637 V509 execution source-delta ingestion.

Spec refs: REQ-REPORT-5637, SCENARIO-REPORT-5637-APPEND-DELTA,
SCENARIO-REPORT-5637-BLOCKED-MARKER,
SCENARIO-REPORT-5637-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5637_v509_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _roadmap() -> str:
    tasks = [
        {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": task_id,
            "deliverable": f"results/{task_id}.json",
        }
        for task_id in (
            "exp5636-transition-v509",
            "exp5637-v509-source-delta-ingestion",
            "exp5638-fr11-gate-schema-corrigendum",
            "exp5639-anytime-valid-csl-independent-audit",
            "exp5640-fr11-shadow-pipeline-integration",
            "exp5641-arc-counterexample-executable-model",
            "exp5642-arc-executable-model-live-ab",
            "exp5643-arc-live-self-discovery-levelup-v509",
            "exp5644-two-axis-parallel-tempering-exact-audit",
            "exp5645-two-axis-tempering-hard-constraint-quality",
            "exp5646-two-axis-tempering-rust-parity",
            "exp5647-v509-capstone-reconciliation",
        )
    ]
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V509 Planner Refresh - 20260714\n\n"
        "- **PatchWorld: Gradient-Free Optimization of Executable World Models** - "
        "arXiv:2605.30880.\n"
        "- **Learning Explicit Behavioral Models with Adaptive Questions and "
        "World-Model Probes** - arXiv:2606.07127.\n"
        "<!-- V509-PLANNER-REFRESH-20260714-END -->\n"
    )


def _make_repo(root: Path, references_text: str, *, roadmap_next: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("AGENTS.md", "CODEX.md", "CLAUDE.md", "research-program.md"):
        (root / relative).write_text("fixture\n", encoding="utf-8")
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(
        references_text,
        encoding="utf-8",
    )
    (root / mod.RESEARCH_COMPLETE_RELATIVE_PATH).write_text(
        (
            "experiments:\n"
            "- id: exp5626-v508-source-delta-ingestion\n"
            "  title: Execution-time V508 source delta\n"
            "  source: arXiv:2607.08897\n"
        ),
        encoding="utf-8",
    )
    (root / mod.ROADMAP_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    if roadmap_next:
        (root / mod.ROADMAP_NEXT_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Research Roadmap vNEXT\n"
        "**Milestone:** 2026.07.509\n"
        "**Task range:** Exp5636-Exp5647\n"
        "Exp5637 searches after the V509 planner marker.\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        "retired: []\nretired_extras: []\n", encoding="utf-8"
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "External generated-text scorers, native runtime certificates, "
        "solver-guidance reruns, ARC epistemic-object probes, hardware speedup, "
        "board, SNN, TSU, Kona, and Aleph scopes remain closed.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / mod.EXP5636_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "experiment_id": "exp5636-transition-v509",
                "current_task_range": "exp5636-exp5647",
                "dependency_map": {
                    "executable_arc_model_to_live_attempt": {
                        "tasks": [
                            "exp5641-arc-counterexample-executable-model",
                            "exp5642-arc-executable-model-live-ab",
                            "exp5643-arc-live-self-discovery-levelup-v509",
                        ]
                    }
                },
                "retired_scopes": [
                    {"key": "arc_epistemic_object_probe", "closed": True},
                    {"key": "timing_hardware_claims", "closed": True},
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_req_report_5637_spec_declares_v509_source_delta_contract() -> None:
    """REQ-REPORT-5637: OpenSpec anchors V509 source-delta fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5637") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5637",
        "SCENARIO-REPORT-5637-APPEND-DELTA",
        "SCENARIO-REPORT-5637-BLOCKED-MARKER",
        "SCENARIO-REPORT-5637-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`search_timestamp_utc`",
        "`reproducibility_checksum`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5637_append_delta_updates_references_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5637-APPEND-DELTA: one new exact hook appends once."""

    root = _make_repo(tmp_path, _planner_references(), roadmap_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T18:38:01Z",
        duration_s=0.5,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert result_text.endswith("\n")
    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert references.index(mod.EXECUTION_REFRESH_HEADING) > references.index(mod.PLANNER_MARKER)
    assert "Baba in Wonderland" in references
    assert "arXiv:2605.16725" in references
    assert artifact["planner_marker_found"] is True
    assert artifact["research_references_updated"] is True
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["search_timestamp_utc"] == "2026-07-14T18:38:01Z"
    assert artifact["roadmap_context"]["source"] == "research-roadmap-next.yaml"
    assert artifact["honest_verdict"].startswith("complete:")

    accepted_ids = {row["source_id"] for row in artifact["new_references_added"]}
    assert accepted_ids == {"baba_in_wonderland_2605_16725"}
    mapping_ids = {
        experiment_id
        for row in artifact["experiment_mappings"]
        for experiment_id in row["experiment_ids"]
    }
    assert mapping_ids <= mod.ALLOWED_MAPPING_IDS
    assert {
        "exp5641-arc-counterexample-executable-model",
        "exp5642-arc-executable-model-live-ab",
    }.issubset(mapping_ids)
    assert "exp5647-v509-capstone-reconciliation" not in mapping_ids

    artifact_second = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T18:38:02Z",
        duration_s=0.5,
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert artifact_second["research_references_updated"] is False


def test_scenario_report_5637_missing_planner_marker_blocks_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5637-BLOCKED-MARKER: missing marker blocks mutation."""

    references_text = "## Earlier Refresh\n\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T18:38:01Z",
        duration_s=0.5,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert references == references_text
    assert artifact["planner_marker_found"] is False
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5637_source_checks_and_watch_boundaries(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5637-APPEND-DELTA: checks classify non-local sources."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T18:38:01Z",
        duration_s=0.5,
        research_references_updated=True,
    )

    mod.validate_artifact(artifact)
    assert artifact["marker_checks"]["planner_marker"] == mod.PLANNER_MARKER
    assert artifact["marker_checks"]["planner_marker_found"] is True
    assert artifact["duplicate_checks"]["candidate_count"] == len(mod.CANDIDATE_FINDINGS)
    assert artifact["duplicate_checks"]["accepted_count"] == len(mod.CANDIDATE_FINDINGS)
    assert artifact["duplicate_checks"]["duplicates_suppressed_count"] == len(
        artifact["duplicates_suppressed"]
    )
    assert any(
        row["path"] == mod.RESEARCH_COMPLETE_RELATIVE_PATH.as_posix()
        for row in artifact["dedupe_corpus_checked"]
    )
    assert any(
        row["source_id"] == "semantic_scholar_ebt_route"
        and row["status"] == "http_429_no_fresh_count_claim"
        for row in artifact["source_link_checks"]
    )
    assert any(
        row["source_id"] == "confidently_wrong_internal_states_2607_11414"
        and row["classification"] == "watch_only_external_internal_state_probe"
        for row in artifact["watch_only_items"]
    )
    assert any(
        row["source_id"] == "scratchworld_2606_31689"
        and row["classification"] == "watch_only_external_benchmark"
        for row in artifact["watch_only_items"]
    )
    surfaces = {row["surface"] for row in artifact["sources_checked"]}
    assert {
        "arXiv",
        "OpenReview",
        "Semantic Scholar",
        "Hugging Face Papers",
        "GitHub discovery/trending",
        "Extropic writing",
        "Logical Intelligence public pages",
        "local Carnot ledgers",
    }.issubset(surfaces)


def test_scenario_report_5637_sparse_repo_defaults_to_blocked_noop(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5637-BLOCKED-MARKER: absent ledgers stay terminal."""

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.5)

    mod.validate_artifact(artifact)
    assert artifact["planner_marker_found"] is False
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    assert artifact["roadmap_context"] == {
        "source": "research-roadmap.yaml",
        "milestone": "",
        "task_ids": [],
    }
    assert artifact["search_timestamp_utc"].endswith("Z")
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5637_field_principle_validation_rejects_bad_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5637-FIELD-PRINCIPLES: malformed fields fail closed."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T18:38:01Z",
        duration_s=0.5,
        research_references_updated=True,
    )
    mod.validate_artifact(artifact)

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("reproducibility_checksum")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["closed_scopes_reopened"] = True
    with pytest.raises(ValueError, match="closed_scopes_reopened"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["search_timestamp_utc"] = "2026-07-14T18:38:01"
    with pytest.raises(ValueError, match="timestamp"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["experiment_mappings"] = [
        {"lane": "bad", "experiment_ids": ["exp5647-v509-capstone-reconciliation"]}
    ]
    with pytest.raises(ValueError, match="mapping"):
        mod.validate_artifact(broken)


def test_scenario_report_5637_defensive_helpers_cover_edge_cases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-5637-FIELD-PRINCIPLES: helper boundaries stay explicit."""

    root = _make_repo(tmp_path, _planner_references())
    original = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert mod._relative_path(tmp_path, Path("/outside/root.txt")) == "/outside/root.txt"
    assert mod._honest_verdict(False, []).startswith("blocked:")
    assert mod._honest_verdict(True, []).startswith("complete: no new")
    assert mod._honest_verdict(True, [{"source_id": "accepted"}]).startswith(
        "complete: accepted 1 non-duplicate actionable V509"
    )
    assert mod._append_execution_refresh_if_needed(root, False, mod._accepted_findings()) is False
    assert (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8") == original
    fallback_insert = mod._insert_after_planner_block(
        "## V509 Planner Refresh - 20260714\nNo end marker.\n",
        "## V509 Execution Refresh - 20260714\n\nBody.\n",
    )
    assert fallback_insert.endswith("Body.\n")

    cli_root = _make_repo(tmp_path / "cli", _planner_references())
    assert (
        mod.main(
            [
                "--root",
                str(cli_root),
                "--search-timestamp-utc",
                "2026-07-14T18:38:03Z",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert mod.RESULT_RELATIVE_PATH.as_posix() in captured.out

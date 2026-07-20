"""Tests for Exp5744 V513 source-delta ingestion.

Spec refs: REQ-REPORT-5744, SCENARIO-REPORT-5744-ZERO-FINDING,
SCENARIO-REPORT-5744-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5744-BLOCKED-MARKER,
SCENARIO-REPORT-5744-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5744_v513_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
START = "2026-07-20T14:00:00Z"
FINISH = "2026-07-20T14:00:42Z"


def _roadmap() -> str:
    tasks = [
        {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": task_id,
            "deliverable": f"results/{task_id}.json",
        }
        for task_id in (
            "exp5743-transition-v513",
            "exp5744-v513-source-delta-ingestion",
            "exp5745-arc-causal-gate-schema-corrigendum",
            "exp5746-exact-proposal-utility-benchmark",
            "exp5747-sota-exact-proposal-utility-panel",
            "exp5748-selective-exact-feedback-search",
            "exp5749-csl-render-matched-mechanism-audit",
            "exp5750-dependent-task-continuous-self-learning",
            "exp5751-rust-restart-parity-repair",
            "exp5752-one-axis-allocation-free-10x-crossover",
            "exp5753-arc-generic-primitive-live-registry-ab",
            "exp5754-v513-capstone-reconciliation",
        )
    ]
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V513 Planner Refresh - 20260720\n\n"
        "- **Opt-Verifier** - arXiv:2605.29556.\n"
        "- **Think Again or Think Longer?** - arXiv:2606.19808.\n"
        "- **CerCE** - OpenReview ICLR 2026.\n"
        "- **ARM-EBM** - arXiv:2512.15605 v4.\n"
        "<!-- V513-PLANNER-REFRESH-20260720-END -->\n"
    )


def _accepted_fixture() -> mod.JsonDict:
    return {
        "source_id": "post_marker_exact_fixture_2607_19999",
        "title": "Post-Marker Exact Fixture For Dual Receipts",
        "url": "https://arxiv.org/abs/2607.19999",
        "arxiv_id": "2607.19999",
        "timestamp_utc": "2026-07-20T14:01:00Z",
        "post_marker_basis": "observed after V513-PLANNER-REFRESH-20260720-END in a fixture query",
        "target_experiments": ["exp5746-exact-proposal-utility-benchmark"],
        "local_substrate": "sealed finite-domain exact benchmark rows",
        "substrate": "sealed finite-domain exact benchmark rows",
        "authority_boundary": "structure receipts and exact validators admit rows; model text is advisory only",
        "carnot_hook": "Add a bounded omitted-constraint control to the existing Exp5746 benchmark plan.",
        "falsifiable_metric": "structure_receipt_failure_count=0 and validator_disagreement_count=0",
    }


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
            "- id: exp5732-v512-source-delta-ingestion\n"
            "  title: V512 source delta\n"
            "- id: exp5743-transition-v513\n"
            "  title: V513 transition\n"
        ),
        encoding="utf-8",
    )
    (root / mod.ROADMAP_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    if roadmap_next:
        (root / mod.ROADMAP_NEXT_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Research Roadmap vNEXT\n"
        "**Milestone:** 2026.07.513\n"
        "**Task range:** Exp5743-Exp5754\n"
        "Exp5744 is bounded bibliographic work with no compute claim.\n",
        encoding="utf-8",
    )
    (root / "openspec/capabilities/research-reporting").mkdir(parents=True, exist_ok=True)
    (root / mod.SPEC_RELATIVE_PATH).write_text(
        "REQ-REPORT-5744\n"
        "SCENARIO-REPORT-5744-ZERO-FINDING\n"
        "SCENARIO-REPORT-5744-ACCEPT-BOUNDED-DELTA\n"
        "SCENARIO-REPORT-5744-BLOCKED-MARKER\n"
        "SCENARIO-REPORT-5744-FIELD-PRINCIPLES\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        (
            "retired: []\n"
            "retired_extras:\n"
            "- id: external_generated_text_scoring_closed\n"
            "  reason: closed for V513\n"
            "- id: unauthenticated_hardware_claims_closed\n"
            "  reason: no local timing receipt\n"
        ),
        encoding="utf-8",
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "Graph redesign, external text scoring, LLM judges, model-weight writes, "
        "broad RL, ARC value scopes, headline scorers, and unauthenticated "
        "hardware claims remain closed.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def test_req_report_5744_spec_declares_bibliographic_contract() -> None:
    """REQ-REPORT-5744: OpenSpec anchors V513 source-delta fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5744") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5744",
        "SCENARIO-REPORT-5744-ZERO-FINDING",
        "SCENARIO-REPORT-5744-ACCEPT-BOUNDED-DELTA",
        "SCENARIO-REPORT-5744-BLOCKED-MARKER",
        "SCENARIO-REPORT-5744-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.PLANNER_MARKER,
        mod.INFERENCE_SUBSTRATE,
        "`search_started_at_utc`",
        "`search_finished_at_utc`",
        "`bibliographic_elapsed_s`",
        "`accepted_findings`",
        "`semantic_scholar_status`",
        "`github_status`",
        "`roadmap_change_required`",
        "`benchmark_compute_claimed`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5744_zero_finding_completes_without_reference_append(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5744-ZERO-FINDING: zero accepted findings are complete."""

    root = _make_repo(tmp_path, _planner_references(), roadmap_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at_utc=START,
        search_finished_at_utc=FINISH,
        accepted_findings=[],
        duplicate_findings=[],
        watch_only_findings=[],
        excluded_findings=[],
        inaccessible_findings=[],
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert result_text.endswith("\n")
    assert mod.EXECUTION_REFRESH_HEADING not in references
    assert artifact["accepted_findings"] == []
    assert artifact["target_experiment_map"] == []
    assert artifact["bibliographic_elapsed_s"] == pytest.approx(42.0)
    assert artifact["references_updated"] is False
    assert artifact["references_mutated_this_run"] is False
    assert artifact["roadmap_change_required"] is False
    assert artifact["benchmark_compute_claimed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["roadmap_context"]["source"] == "research-roadmap-next.yaml"
    assert artifact["honest_verdict"].startswith("complete: no new")


def test_scenario_report_5744_accept_delta_appends_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5744-ACCEPT-BOUNDED-DELTA: hooks append once."""

    root = _make_repo(tmp_path, _planner_references())

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at_utc=START,
        search_finished_at_utc=FINISH,
        accepted_findings=[_accepted_fixture()],
        duplicate_findings=[
            {
                "source_id": "opt_verifier_2605_29556",
                "title": "Opt-Verifier",
                "url": "https://arxiv.org/abs/2605.29556",
                "reason": "Already indexed by the V513 planner.",
            }
        ],
        watch_only_findings=[],
        excluded_findings=[],
        inaccessible_findings=[],
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert "Post-Marker Exact Fixture For Dual Receipts" in references
    assert "exp5746-exact-proposal-utility-benchmark" in references
    assert references.index(mod.EXECUTION_REFRESH_HEADING) > references.index(
        mod.PLANNER_HEADING
    )
    assert artifact["accepted_findings"][0]["target_experiments"] == [
        "exp5746-exact-proposal-utility-benchmark"
    ]
    assert artifact["target_experiment_map"][0]["authority_boundary"].startswith(
        "structure receipts"
    )
    assert artifact["references_updated"] is True
    assert artifact["references_mutated_this_run"] is True
    assert artifact["honest_verdict"].startswith("complete: accepted 1")

    artifact_second = mod.build_and_write_artifact(
        root=root,
        search_started_at_utc="2026-07-20T14:02:00Z",
        search_finished_at_utc="2026-07-20T14:02:01Z",
        accepted_findings=[_accepted_fixture()],
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert artifact_second["references_updated"] is True
    assert artifact_second["references_mutated_this_run"] is False


def test_scenario_report_5744_missing_planner_marker_blocks_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5744-BLOCKED-MARKER: missing marker blocks mutation."""

    references_text = "## V512 Planner Refresh - 20260719\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at_utc=START,
        search_finished_at_utc=FINISH,
        accepted_findings=[_accepted_fixture()],
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert references == references_text
    assert artifact["planner_marker_found"] is False
    assert artifact["references_updated"] is False
    assert artifact["accepted_findings"] == []
    assert artifact["roadmap_change_required"] is False
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5744_statuses_boundaries_and_validation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5744-FIELD-PRINCIPLES: dispositions are explicit."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at_utc=START,
        search_finished_at_utc=FINISH,
        accepted_findings=[_accepted_fixture()],
    )

    mod.validate_artifact(artifact)
    surfaces = {row["surface"] for row in artifact["sources_checked"]}
    assert {
        "arXiv",
        "OpenReview",
        "Semantic Scholar",
        "Hugging Face Papers",
        "GitHub discovery",
        "Extropic writing",
        "Logical Intelligence public pages",
        "local Carnot ledgers",
    }.issubset(surfaces)
    assert artifact["semantic_scholar_status"]["papers"] == [
        "arXiv:2507.02092",
        "arXiv:2512.15605",
    ]
    assert artifact["extropic_status"]["local_execution_available"] is False
    assert artifact["logical_intelligence_status"]["local_execution_available"] is False
    assert artifact["huggingface_status"]["roadmap_delta"] is False
    assert artifact["github_status"]["accepted_support_repository"] is None
    assert artifact["closed_scope_review"] == mod.closed_scope_review()
    assert any(
        row["source_id"] == "arxiv_cs_ai_recent"
        for row in artifact["source_link_checks"]
    )

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("github_status")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["benchmark_compute_claimed"] = True
    with pytest.raises(ValueError, match="benchmark"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["search_finished_at_utc"] = START
    with pytest.raises(ValueError, match="timestamp"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["bibliographic_elapsed_s"] = 99.0
    with pytest.raises(ValueError, match="elapsed"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["target_experiment_map"] = [
        {"source_id": "bad", "target_experiments": ["exp5754-v513-capstone-reconciliation"]}
    ]
    with pytest.raises(ValueError, match="target experiment"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["accepted_findings"] = [
        {
            "source_id": "incomplete",
            "target_experiments": ["exp5746-exact-proposal-utility-benchmark"],
        }
    ]
    with pytest.raises(ValueError, match="accepted finding"):
        mod.validate_artifact(broken)


def test_scenario_report_5744_helpers_scope_block_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-5744-FIELD-PRINCIPLES: helper boundaries stay explicit."""

    root = _make_repo(tmp_path, _planner_references())
    assert mod.read_text_if_present(root / "missing.md") == ""
    assert mod.relative_path(tmp_path, Path("/outside/root.txt")) == "/outside/root.txt"
    assert mod.normalize_timestamp("2026-07-20T14:00:00+00:00").endswith("Z")
    assert mod.planner_marker_line("missing") is None
    assert mod.honest_verdict(False, []).startswith("blocked:")
    assert mod.honest_verdict(True, []).startswith("complete: no new")
    block = mod.execution_refresh_block([_accepted_fixture()])
    assert mod.insert_after_planner_block("no marker", block).endswith(block)
    fallback_insert = mod.insert_after_planner_block(
        f"{mod.PLANNER_HEADING}\nbody",
        block,
    )
    assert fallback_insert.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert fallback_insert.endswith(block)

    malformed_root = _make_repo(tmp_path / "malformed", _planner_references())
    (malformed_root / mod.ROADMAP_RELATIVE_PATH).write_text("- just\n- a list\n")
    context = mod.roadmap_context(malformed_root)
    assert context["milestone"] == ""
    assert context["task_ids"] == []

    artifact = mod.build_artifact(
        root=root,
        search_started_at_utc=START,
        search_finished_at_utc=FINISH,
    )
    tampered = dict(artifact)
    tampered["honest_verdict"] = "complete: tampered"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(tampered)

    blocked = dict(artifact)
    blocked["roadmap_change_required"] = True
    blocked["status"] = "blocked"
    blocked["honest_verdict"] = "blocked: scope expansion requires operator review"
    blocked["reproducibility_checksum"] = mod.payload_checksum(blocked)
    mod.validate_artifact(blocked)

    cli_root = _make_repo(tmp_path / "cli", _planner_references())
    assert (
        mod.main(
            [
                "--root",
                str(cli_root),
                "--search-started-at-utc",
                START,
                "--search-finished-at-utc",
                FINISH,
                "--zero-findings",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert mod.RESULT_RELATIVE_PATH.as_posix() in captured.out


def test_scenario_report_5744_module_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-5744-FIELD-PRINCIPLES: module entry exits cleanly."""

    root = _make_repo(tmp_path, _planner_references())
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_5744_v513_source_delta_ingestion",
            "--root",
            str(root),
            "--search-started-at-utc",
            START,
            "--search-finished-at-utc",
            FINISH,
            "--zero-findings",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(
            "carnot.experiment_5744_v513_source_delta_ingestion",
            run_name="__main__",
        )

    assert exc_info.value.code == 0
    assert mod.RESULT_RELATIVE_PATH.as_posix() in capsys.readouterr().out

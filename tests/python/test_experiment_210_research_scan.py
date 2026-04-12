"""Spec: REQ-REPORT-005, REQ-REPORT-006, REQ-REPORT-007, REQ-REPORT-008,
SCENARIO-REPORT-004, SCENARIO-REPORT-005
"""

# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import json
import os
import runpy
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_210_research_scan.py"
    spec = importlib.util.spec_from_file_location("experiment_210_research_scan", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    (repo / "research-references.md").write_text(
        "# Research References & Future Considerations\n\nExisting notes.\n",
        encoding="utf-8",
    )
    (repo / "research-studying.md").write_text(
        "# Research Studying — Ranked Ideas for Future Experiments\n\nExisting queue.\n",
        encoding="utf-8",
    )
    return repo


# REQ-REPORT-005, SCENARIO-REPORT-004
def test_build_results_payload_contains_ranked_findings_and_follow_on_experiments():
    module = load_module()

    payload = module.build_results("2026-04-12T12:00:00+00:00")

    assert payload["experiment"] == "Exp 210"
    assert payload["scan_date"] == "2026-04-12T12:00:00+00:00"
    assert "constraint extraction on instruction-tuned models" in payload["focus_problem"]
    assert payload["queries_run"][0] == "constraint extraction from chain-of-thought responses"
    assert (
        payload["papers"][0]["title"]
        == "Neuro-Symbolic Verification on Instruction Following of LLMs"
    )
    assert payload["papers"][0]["rank"] == 1
    assert (
        payload["risk_evidence"][0]["title"]
        == "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation"
    )
    assert [experiment["id"] for experiment in payload["proposed_experiments_2026_04_15"]] == [
        "EXP-211",
        "EXP-212",
        "EXP-213",
    ]


# REQ-REPORT-006, SCENARIO-REPORT-004
def test_upsert_marked_section_appends_when_markers_are_missing(tmp_path: Path):
    module = load_module()
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nBaseline text.\n", encoding="utf-8")

    module.upsert_marked_section(
        path,
        module.REFERENCES_START_MARKER,
        module.REFERENCES_END_MARKER,
        "## Added Section\n\nBody.\n",
    )

    updated = path.read_text(encoding="utf-8")
    assert updated.count(module.REFERENCES_START_MARKER) == 1
    assert updated.count(module.REFERENCES_END_MARKER) == 1
    assert "Baseline text." in updated
    assert "## Added Section" in updated


# REQ-REPORT-008, SCENARIO-REPORT-005
def test_upsert_marked_section_replaces_existing_block_without_duplication(tmp_path: Path):
    module = load_module()
    path = tmp_path / "notes.md"
    path.write_text(
        "\n".join(
            [
                "# Notes",
                "",
                module.STUDYING_START_MARKER,
                "old body",
                module.STUDYING_END_MARKER,
                "",
            ]
        ),
        encoding="utf-8",
    )

    module.upsert_marked_section(
        path,
        module.STUDYING_START_MARKER,
        module.STUDYING_END_MARKER,
        "new body\n",
    )

    updated = path.read_text(encoding="utf-8")
    assert "old body" not in updated
    assert "new body" in updated
    assert updated.count(module.STUDYING_START_MARKER) == 1
    assert updated.count(module.STUDYING_END_MARKER) == 1


# REQ-REPORT-005, REQ-REPORT-006, REQ-REPORT-007, REQ-REPORT-008,
# SCENARIO-REPORT-004, SCENARIO-REPORT-005
def test_main_updates_docs_and_writes_results_idempotently(tmp_path: Path, monkeypatch):
    module = load_module()
    repo = make_repo(tmp_path)

    monkeypatch.setattr(module, "REPO_ROOT", repo)
    monkeypatch.setattr(module, "RESULTS_PATH", repo / "results" / "experiment_210_results.json")
    monkeypatch.setattr(module, "REFERENCES_PATH", repo / "research-references.md")
    monkeypatch.setattr(module, "STUDYING_PATH", repo / "research-studying.md")
    monkeypatch.setattr(module, "get_scan_timestamp", lambda: "2026-04-12T15:00:00+00:00")

    assert module.main() == 0
    assert module.main() == 0

    payload = json.loads(
        (repo / "results" / "experiment_210_results.json").read_text(encoding="utf-8")
    )
    references = (repo / "research-references.md").read_text(encoding="utf-8")
    studying = (repo / "research-studying.md").read_text(encoding="utf-8")

    assert payload["scan_date"] == "2026-04-12T15:00:00+00:00"
    assert references.count(module.REFERENCES_START_MARKER) == 1
    assert studying.count(module.STUDYING_START_MARKER) == 1
    assert "Exp 210: Constraint Extraction for Instruction-Tuned Models" in references
    assert "Study Run 2026-04-12 - Constraint Extraction for Instruction-Tuned Models" in studying
    assert "EXP-212" in studying


# REQ-REPORT-005, REQ-REPORT-006, REQ-REPORT-007, SCENARIO-REPORT-004
def test_cli_entrypoint_honors_repo_override(tmp_path: Path, monkeypatch):
    repo = make_repo(tmp_path)
    module_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "experiment_210_research_scan.py"
    )

    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
    monkeypatch.setattr("sys.argv", [str(module_path)])

    try:
        runpy.run_path(str(module_path), run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0

    payload = json.loads(
        (repo / "results" / "experiment_210_results.json").read_text(encoding="utf-8")
    )
    assert payload["experiment"] == "Exp 210"
    assert os.environ["CARNOT_REPO_ROOT"] == str(repo)

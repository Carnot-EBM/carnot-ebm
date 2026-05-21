"""Tests for Exp 2827 milestone .267 archive and .268 activation.

Spec: REQ-REPORT-2827, SCENARIO-REPORT-2827.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.reporting import milestone_267_archive_268_activation as exp


def _clock(*values: float):
    times = iter(values)
    return lambda: next(times)


def _write_common_inputs(
    root: Path,
    *,
    roadmap_milestone: str = "2026.05.268",
    complete_text: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{roadmap_milestone}"\n'
        "tasks:\n"
        "  - id: exp2827\n"
        '    deliverable: "results/experiment_2827_archive_v267.json"\n',
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    (root / "legacy" / "fabricated").mkdir(parents=True)
    (root / "legacy" / "fabricated" / "README.md").write_text(
        "experiment_2823_truthfulqa_ensemble_eval.json was moved here.",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True)
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired:\n"
        "  - experiment_id: 2823\n"
        '    completed_milestone: "2026.05.267"\n'
        "    reason: fabricated\n",
        encoding="utf-8",
    )


def _generic_267_complete_text() -> str:
    return """milestones:
- id: 2026.05.266
  tasks:
  - id: exp2818
    result: OK (conductor)
- id: 2026.05.267
  title: ''
  doc: ''
  completed: '2026-05-21'
  finding: See conductor log for per-experiment results.
  tasks:
  - id: exp2819
    title: Archive .266 + Activate .267
    deliverable: results/experiment_2819_archive_v266.json
    result: OK (conductor)
  - id: exp2820
    title: FoVer Memory-Leakage Isolation
    deliverable: results/experiment_2820_fover_memory_leakage_isolation.json
    result: OK (conductor)
  - id: exp2821
    title: MBPP Corpus
    deliverable: results/experiment_2821_mbpp_ensemble_eval.json
    result: OK (conductor)
  - id: exp2822
    title: HumanEval Full
    deliverable: results/experiment_2822_humaneval_full_ensemble_eval.json
    result: OK (conductor)
  - id: exp2823
    title: TruthfulQA Corpus
    deliverable: results/experiment_2823_truthfulqa_ensemble_eval.json
    result: OK (conductor)
  - id: exp2824
    title: Cross-Corpus Per-Verifier Dual-Condition Discriminative Matrix
    deliverable: results/experiment_2824_cross_corpus_verifier_matrix.json
    result: OK (conductor)
  - id: exp2825
    title: Paper v6 Section 5 table
    deliverable: results/experiment_2825_paper_v6_multicorpus_table.json
    result: OK (conductor)
  - id: exp2826
    title: Capstone v267
    deliverable: results/experiment_2826_capstone_v267.json
    result: OK (conductor)
"""


def test_scenario_report_2827_corrects_existing_generic_archive_without_duplicate(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2827: existing .267 archive is corrected, not duplicated."""

    _write_common_inputs(
        tmp_path,
        complete_text=_generic_267_complete_text(),
    )
    before_conductor = (Path.cwd() / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )

    artifact = exp.run(root=tmp_path, clock=_clock(10.0, 14.25))

    written = json.loads((tmp_path / "results/experiment_2827_archive_v267.json").read_text())
    complete = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    row = [m for m in complete["milestones"] if str(m.get("id")) == "2026.05.267"]
    task_results = {task["id"]: task["result"] for task in row[0]["tasks"]}

    assert artifact == written
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["archived_milestone"] == "2026.05.267"
    assert artifact["archived_milestone_experiments_completed"] == 3
    assert artifact["activated_milestone"] == "2026.05.268"
    assert artifact["duration_s"] == 4.25
    assert artifact["archive"]["existing_entry_corrected"] is True
    assert artifact["archive"]["appended_this_run"] is False
    assert len(row) == 1
    assert row[0]["partial_status_note"].startswith(".267 was a partial milestone")
    assert task_results["exp2819"].startswith("SKIP")
    assert task_results["exp2820"].startswith("SKIP")
    assert task_results["exp2821"].startswith("SKIP")
    assert task_results["exp2822"].startswith("SKIP")
    assert "fabrication" in task_results["exp2823"].lower()
    assert task_results["exp2824"].startswith("OK")
    assert task_results["exp2825"].startswith("OK")
    assert task_results["exp2826"].startswith("OK")
    assert (Path.cwd() / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_2827_appends_honest_partial_archive_when_absent(tmp_path: Path) -> None:
    """REQ-REPORT-2827: absent .267 archive row is appended with partial statuses."""

    _write_common_inputs(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.266\n  tasks: []\n",
    )

    artifact = exp.run(root=tmp_path, clock=_clock(1.0, 2.0))

    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    complete = yaml.safe_load(complete_text)
    row = [m for m in complete["milestones"] if str(m.get("id")) == "2026.05.267"]

    assert artifact["archive_ready"] is True
    assert artifact["archive"]["appended_this_run"] is True
    assert artifact["archive"]["existing_entry_corrected"] is False
    assert artifact["preconditions_checked"]["roadmap_milestone"]["observed"] == "2026.05.268"
    assert artifact["preconditions_checked"]["fabricated_retirement"]["passed"] is True
    assert artifact["archived_milestone_experiments_completed"] == 3
    assert complete_text.count("id: 2026.05.267") == 1
    assert len(row[0]["tasks"]) == 8


def test_req_report_2827_unexpected_roadmap_reports_without_archive_mutation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2827: unexpected active milestone is terminal but does not mutate archives."""

    _write_common_inputs(
        tmp_path,
        roadmap_milestone="2026.05.269",
        complete_text="milestones:\n- id: 2026.05.266\n  tasks: []\n",
    )
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(5.0, 5.5))

    assert artifact["honest_verdict"].startswith("complete:")
    assert "unexpected_active_milestone" in artifact["honest_verdict"]
    assert artifact["archive_ready"] is False
    assert artifact["activated_milestone"] == "2026.05.269"
    assert artifact["archive"]["appended_this_run"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete


def test_req_report_2827_helpers_cover_missing_and_following_archive_rows(tmp_path: Path) -> None:
    """REQ-REPORT-2827: helper parsing handles missing YAML and bounded archive rows."""

    assert exp._load_first_yaml_mapping(tmp_path / "missing.yaml") == {}
    text = (
        "milestones:\n"
        "- id: 2026.05.267\n"
        "  tasks: []\n"
        "- id: 2026.05.268\n"
        "  tasks: []\n"
    )
    assert exp._find_milestone_block(text, "2026.05.267") == (1, 3)

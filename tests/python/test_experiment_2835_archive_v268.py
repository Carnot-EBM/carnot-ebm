"""Tests for Exp 2835 milestone .268 archive and .269 activation.

Spec: REQ-REPORT-2835, SCENARIO-REPORT-2835.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.reporting import milestone_268_archive_269_activation as exp


def _clock(*values: float):
    times = iter(values)
    return lambda: next(times)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_source_artifacts(root: Path) -> None:
    _write_json(
        root / "results/experiment_2827_archive_v267.json",
        {
            "honest_verdict": "complete: archive_ready=true",
            "archived_milestone": "2026.05.267",
            "activated_milestone": "2026.05.268",
        },
    )
    _write_json(
        root / "results/experiment_2828_fover_memory_leakage_isolation.json",
        {
            "honest_verdict": "blocked_cuda: ModuleNotFoundError: No module named 'torch'",
            "blocked_resources": ["python3_torch_cuda", "qwen36_35b_a3b_gguf_cache"],
        },
    )
    for experiment_id, blocked_resources in {
        "2829": ["cuda", "hf_mbpp", "qwen36_gguf_cache"],
        "2830": ["cuda", "hf_openai_humaneval", "qwen36_gguf_cache"],
        "2831": [
            "cuda",
            "hf_truthfulqa_generation",
            "qwen36_gguf_cache",
            "bleurt_base_128",
        ],
    }.items():
        _write_json(
            root / f"results/experiment_{experiment_id}_{exp.SOURCE_ARTIFACT_STEMS[experiment_id]}.json",
            {
                "honest_verdict": "blocked_cuda_unavailable",
                "blocked_resources": blocked_resources,
            },
        )
    _write_json(
        root / "results/experiment_2832_cross_corpus_verifier_matrix_v2.json",
        {
            "honest_verdict": (
                "complete: upstream artifacts loaded but no measured per-verifier AUROC "
                "rows were present"
            ),
            "verifier_corpus_dual_matrix": {},
        },
    )
    _write_json(
        root / "results/experiment_2833_paper_v6_multicorpus_table_v2.json",
        {
            "honest_verdict": (
                "complete: exp2828-2832 artifacts integrated honestly; at least one "
                "dual-condition AUROC remains unmeasured, so arxiv_ready_v7=false"
            ),
            "arxiv_ready_v7": False,
        },
    )
    _write_json(
        root / "results/experiment_2834_capstone_v268.json",
        {
            "experiment": "exp2834",
            "milestone": "2026.05.268",
            "honest_verdict": (
                "complete: .268 capstone synthesised - all 4 corpus evaluation tasks "
                "blocked; FoVer-overfit thesis UNCONFIRMED; FR-11 hypothesis "
                "UNCONFIRMED"
            ),
            "blocked_experiments": ["exp2828", "exp2829", "exp2830", "exp2831"],
            "n_experiments_blocked": 4,
            "fover_shape_overfit_confirmed": False,
            "self_learning_contribution_confirmed": False,
            "acceptance_criteria_met": 5,
        },
    )


def _write_common_inputs(
    root: Path,
    *,
    roadmap_milestone: str = "2026.05.269",
    complete_text: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "research-roadmap.yaml").write_text(
        f'milestone: "{roadmap_milestone}"\n'
        "tasks:\n"
        "  - id: exp2835-archive-v268-activate-v269\n"
        '    deliverable: "results/experiment_2835_archive_v268.json"\n',
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(complete_text, encoding="utf-8")
    _write_source_artifacts(root)


def _generic_268_complete_text() -> str:
    return """milestones:
- id: 2026.05.267
  tasks: []
- id: 2026.05.268
  title: ''
  doc: ''
  completed: '2026-05-22'
  finding: See conductor log for per-experiment results.
  tasks:
  - id: exp2827
    title: Archive .267 + Activate .268
    deliverable: results/experiment_2827_archive_v267.json
    result: OK (conductor)
  - id: exp2828
    title: FoVer Memory-Leakage Isolation
    deliverable: results/experiment_2828_fover_memory_leakage_isolation.json
    result: OK (conductor)
  - id: exp2829
    title: MBPP Corpus
    deliverable: results/experiment_2829_mbpp_ensemble_eval.json
    result: OK (conductor)
  - id: exp2830
    title: HumanEval Full
    deliverable: results/experiment_2830_humaneval_full_ensemble_eval.json
    result: OK (conductor)
  - id: exp2831
    title: TruthfulQA Corpus
    deliverable: results/experiment_2831_truthfulqa_ensemble_eval.json
    result: OK (conductor)
  - id: exp2832
    title: Cross-Corpus Per-Verifier Dual-Condition Matrix v2
    deliverable: results/experiment_2832_cross_corpus_verifier_matrix_v2.json
    result: OK (conductor)
  - id: exp2833
    title: Paper v6 Section 5 table v2
    deliverable: results/experiment_2833_paper_v6_multicorpus_table_v2.json
    result: OK (conductor)
  - id: exp2834
    title: Capstone v268
    deliverable: results/experiment_2834_capstone_v268.json
    result: OK (conductor)
"""


def test_scenario_report_2835_corrects_generic_archive_without_duplicate(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2835: generic .268 archive is corrected without duplication."""

    _write_common_inputs(tmp_path, complete_text=_generic_268_complete_text())
    before_roadmap = (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8")
    before_conductor = (Path.cwd() / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    )

    artifact = exp.run(root=tmp_path, clock=_clock(10.0, 12.5))

    written = json.loads((tmp_path / "results/experiment_2835_archive_v268.json").read_text())
    complete = yaml.safe_load((tmp_path / "research-complete.yaml").read_text(encoding="utf-8"))
    row = [m for m in complete["milestones"] if str(m.get("id")) == "2026.05.268"]
    task_results = {task["id"]: task["result"] for task in row[0]["tasks"]}

    assert artifact == written
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["archived_milestone"] == "2026.05.268"
    assert artifact["activated_milestone"] == "2026.05.269"
    assert artifact["blocked_task_count"] == 4
    assert artifact["duration_s"] == 2.5
    assert artifact["archive"]["existing_entry_corrected"] is True
    assert artifact["archive"]["appended_this_run"] is False
    assert len(row) == 1
    assert task_results["exp2827"].startswith("OK")
    assert task_results["exp2828"].startswith("blocked_cuda")
    assert task_results["exp2829"].startswith("blocked_cuda_unavailable")
    assert task_results["exp2830"].startswith("blocked_cuda_unavailable")
    assert task_results["exp2831"].startswith("blocked_cuda_unavailable")
    assert "empty matrix" in task_results["exp2832"]
    assert "not cite-ready" in task_results["exp2833"]
    assert "FoVer-overfit unconfirmed" in task_results["exp2834"]
    assert artifact["archived_task_summary"]["exp2828"]["blocked"] is True
    assert artifact["archived_task_summary"]["exp2834"]["acceptance_criteria_met"] == 5
    assert "system python3" in artifact["runtime_root_cause"]
    assert "GGUF" in artifact["runtime_root_cause"]
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == before_roadmap
    assert (Path.cwd() / "scripts" / "research_conductor.py").read_text(
        encoding="utf-8"
    ) == before_conductor


def test_req_report_2835_appends_honest_blocked_archive_when_absent(tmp_path: Path) -> None:
    """REQ-REPORT-2835: absent .268 archive row is appended with blocked statuses."""

    _write_common_inputs(
        tmp_path,
        complete_text="milestones:\n- id: 2026.05.267\n  tasks: []\n",
    )

    artifact = exp.run(root=tmp_path, clock=_clock(1.0, 1.75))

    complete_text = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")
    complete = yaml.safe_load(complete_text)
    row = [m for m in complete["milestones"] if str(m.get("id")) == "2026.05.268"]

    assert artifact["archive_ready"] is True
    assert artifact["archive"]["appended_this_run"] is True
    assert artifact["archive"]["existing_entry_corrected"] is False
    assert artifact["preconditions_checked"]["roadmap_milestone"]["observed"] == "2026.05.269"
    assert artifact["preconditions_checked"]["research_complete_archive"]["observed_before_count"] == 0
    assert complete_text.count("id: 2026.05.268") == 1
    assert len(row[0]["tasks"]) == 8


def test_req_report_2835_unexpected_roadmap_reports_without_archive_mutation(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2835: unexpected active milestone reports terminally without mutation."""

    _write_common_inputs(
        tmp_path,
        roadmap_milestone="2026.05.268",
        complete_text="milestones:\n- id: 2026.05.267\n  tasks: []\n",
    )
    before_complete = (tmp_path / "research-complete.yaml").read_text(encoding="utf-8")

    artifact = exp.run(root=tmp_path, clock=_clock(5.0, 5.5))

    assert artifact["honest_verdict"].startswith("complete:")
    assert "unexpected_active_milestone" in artifact["honest_verdict"]
    assert artifact["archive_ready"] is False
    assert artifact["activated_milestone"] == "2026.05.269"
    assert artifact["activation"]["observed_active_milestone_before"] == "2026.05.268"
    assert artifact["archive"]["appended_this_run"] is False
    assert (tmp_path / "research-complete.yaml").read_text(encoding="utf-8") == before_complete


def test_req_report_2835_helpers_cover_missing_and_following_archive_rows(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2835: helpers handle missing YAML, missing JSON, and bounded rows."""

    assert exp._load_first_yaml_mapping(tmp_path / "missing.yaml") == {}
    assert exp._load_json(tmp_path / "missing.json") == {}
    assert exp._archive_row_matches(None) is False
    text = (
        "milestones:\n"
        "- id: 2026.05.268\n"
        "  tasks: []\n"
        "- id: 2026.05.269\n"
        "  tasks: []\n"
    )
    assert exp._find_milestone_block(text, "2026.05.268") == (1, 3)

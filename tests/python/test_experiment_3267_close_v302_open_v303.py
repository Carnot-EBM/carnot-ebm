"""Tests for Exp 3267 close .302 and open .303 handoff.

Spec refs: REQ-REPORT-3267, SCENARIO-REPORT-3267.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import close_v302_open_v303_3267 as mod


REQUIRED_FIELDS = {
    "v302_closed_v303_opened",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "prior_next_top_gap",
    "v4_shard_label_count",
    "v4_shard_auroc",
    "protected_files_untouched",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capstone_payload(
    *,
    ready: bool = True,
    paper_ready: bool = False,
    blockers: int = 105,
    next_gap: str = mod.FULL_V4_CORPUS_REPAIR_GAP,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_3266_capstone_v302",
        "experiment_id": "exp3266",
        "task_id": "exp3266-capstone-v302",
        "milestone": "2026.05.302",
        "capstone_v302_ready": ready,
        "paper_ready": paper_ready,
        "publication_blocker_count": blockers,
        "next_top_gap": next_gap,
        "cuda_recovery_unblocked_sota_receipt": ready,
        "v4_shard_status": {
            "label_count_total": 2000,
            "label_counts": {"benign": 1459, "injection": 541},
            "shard_auroc": 0.791096,
        },
        "honest_verdict": (
            "complete: capstone_v302_ready=true; paper_ready=false; "
            "publication_blocker_count=105; next_top_gap="
            "full_15k_v4_corpus_across_shards_plus_repair_and_garak_gates"
        ),
    }


def _exp3264_payload(ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3264_prompt_injection_teacher_label_shard_v3",
        "experiment_id": "exp3264",
        "teacher_label_shard_ready": ready,
        "teacher_label_shard_v3_ready": ready,
        "shard_size": 2000 if ready else 0,
        "label_counts": {"benign": 1459, "injection": 541} if ready else {},
        "honest_verdict": f"complete: teacher_label_shard_ready={str(ready).lower()}",
    }


def _exp3265_payload(ready: bool = True) -> dict[str, Any]:
    return {
        "artifact": "experiment_3265_prompt_injection_kan_train_eval_shard_v3",
        "experiment_id": "exp3265",
        "kan_train_eval_shard_ready": ready,
        "kan_train_eval_shard_v3_ready": ready,
        "shard_auroc": 0.791096 if ready else None,
        "n_train": 1600 if ready else 0,
        "n_eval": 400 if ready else 0,
        "non_headline_note": "single-shard AUROC is a viability check only",
        "honest_verdict": f"complete: kan_train_eval_shard_ready={str(ready).lower()}",
    }


def _research_complete_yaml() -> str:
    lines = [
        "- id: 2026.05.302",
        "  title: CUDA-Recovered SOTA Receipt + v4 Teacher-Label Shard + Capstone",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-05-28'",
        "  finding: See conductor log for per-experiment results.",
        "  tasks:",
    ]
    for task in mod.PRIOR_TASKS:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {task['title']}",
                f"    deliverable: {task['deliverable']}",
                "    result: OK (conductor)",
            ]
        )
    return "\n".join(lines) + "\n"


def _roadmap_yaml(*, milestone: str = "2026.05.303", first_task: str = mod.TASK_ID) -> str:
    tasks = [
        first_task,
        "exp3268-sota-receipt-methodology-supplement-v1",
        "exp3269-prompt-injection-v4-full-corpus-split-manifest-v1",
    ]
    lines = [
        f'milestone: "{milestone}"',
        'milestone_title: "Prompt-Injection v4 Full Corpus + Garak Gate + Repair Reopen"',
        f'milestone_doc: "{mod.VNEXT_DOC_REL_PATH.as_posix()}"',
        "tasks:",
    ]
    for task_id in tasks:
        lines.extend(
            [
                f'  - id: "{task_id}"',
                f'    milestone: "{milestone}"',
                f'    deliverable: "results/{task_id}.json"',
            ]
        )
    return "\n".join(lines) + "\n"


def _write_sources(root: Path, *, include_research_complete: bool = True) -> None:
    _write_json(root, mod.CAPSTONE_V302_REL_PATH, _capstone_payload())
    _write_json(root, mod.EXP3264_REL_PATH, _exp3264_payload())
    _write_json(root, mod.EXP3265_REL_PATH, _exp3265_payload())
    if include_research_complete:
        _write_text(root, mod.RESEARCH_COMPLETE_REL_PATH, _research_complete_yaml())
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "| 2026-05-28 11:55 UTC | Milestone 2026.05.303 activated | OK | 14 tasks queued |\n",
    )
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT\n")


def test_req_report_3267_spec_anchor_exists() -> None:
    """REQ-REPORT-3267: OpenSpec declares the .303 handoff before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3267" in spec
    assert "SCENARIO-REPORT-3267" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3267_builds_ready_handoff_without_duplicate_archive(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3267: present .302 archive opens .303 corpus queue."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.5)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)
    sources = {row["role"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3267"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.303"
    assert artifact["prior_milestone"] == "2026.05.302"
    assert artifact["v302_closed_v303_opened"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 105
    assert artifact["prior_next_top_gap"] == mod.FULL_V4_CORPUS_REPAIR_GAP
    assert artifact["v4_shard_label_count"] == 2000
    assert artifact["v4_shard_auroc"] == pytest.approx(0.791096)
    assert artifact["research_complete_update"]["appended"] is False
    assert artifact["research_complete_update"]["already_present"] is True
    assert artifact["research_complete_prior_summary"]["task_count"] == len(mod.PRIOR_TASKS)
    assert artifact["v303_queue"]["selected_queue_milestone"] == "2026.05.303"
    assert artifact["v303_queue"]["queue_first_task"] == mod.TASK_ID
    assert artifact["v303_activation_observed"] is True
    assert artifact["protected_files_untouched"] is True
    assert artifact["protected_file_checksums"]["research-roadmap.yaml"]["unchanged"] is True
    assert (
        artifact["protected_file_checksums"]["scripts/research_conductor.py"]["unchanged"] is True
    )
    assert artifact["no_push"] is True
    assert artifact["no_conductor_execution"] is True
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "paper_ready=true" not in artifact["honest_verdict"]
    assert sources["capstone_v302"]["sha256"] == _sha256(tmp_path / mod.CAPSTONE_V302_REL_PATH)
    assert sources["teacher_label_shard_v3"]["ready"] is True
    assert sources["kan_train_eval_shard_v3"]["ready"] is True


def test_req_report_3267_writer_appends_missing_research_complete_entry_once(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3267: writer materializes the missing .302 summary once."""

    _write_sources(tmp_path, include_research_complete=False)
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_REL_PATH, "- id: 2026.05.301\n  tasks: []\n")

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=6.0)
    saved = json.loads(output.read_text(encoding="utf-8"))
    archive_text = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    ensure_result = mod.ensure_research_complete_entry(tmp_path)

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["v302_closed_v303_opened"] is True
    assert saved["research_complete_update"]["appended"] is True
    assert saved["research_complete_update"]["already_present"] is False
    assert archive_text.count("- id: 2026.05.302") == 1
    assert ensure_result == {
        "path": mod.RESEARCH_COMPLETE_REL_PATH.as_posix(),
        "appended": False,
        "already_present": True,
    }


def test_req_report_3267_fail_closed_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3267: malformed inputs remain explicit and non-fabricated."""

    _write_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.CAPSTONE_V302_REL_PATH,
        _capstone_payload(ready=False, paper_ready=True, blockers=106, next_gap="wrong"),
    )
    _write_json(tmp_path, mod.EXP3264_REL_PATH, _exp3264_payload(False))
    _write_json(tmp_path, mod.EXP3265_REL_PATH, _exp3265_payload(False))
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone="2026.05.304"))
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_REL_PATH, "[")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("tasks: [", encoding="utf-8")

    baseline = mod.protected_file_checksums(tmp_path)
    _write_text(tmp_path, mod.CONDUCTOR_REL_PATH, "# modified protected conductor\n")
    artifact = mod.build_artifact(
        tmp_path,
        protected_hash_baseline=baseline,
        started_s=3.0,
        now_s=2.0,
    )
    no_verdict = dict(artifact)
    no_verdict["honest_verdict"] = "blocked"
    missing = dict(artifact)
    missing.pop("v302_closed_v303_opened")
    bad_experiment = dict(artifact)
    bad_experiment["experiment_id"] = "exp0000"
    bad_task = dict(artifact)
    bad_task["task_id"] = "wrong"
    bad_milestone = dict(artifact)
    bad_milestone["milestone"] = "2026.05.304"
    bad_seed = dict(artifact)
    bad_seed["random_seed"] = 0

    assert artifact["v302_closed_v303_opened"] is False
    assert artifact["duration_s"] == 0.0
    assert artifact["prior_paper_ready"] is True
    assert artifact["prior_publication_blocker_count"] == 106
    assert artifact["prior_next_top_gap"] == "wrong"
    assert artifact["v4_shard_label_count"] == 0
    assert artifact["v4_shard_auroc"] == 0.0
    assert artifact["protected_files_untouched"] is False
    assert "capstone_v302 authority is not ready" in artifact["blocked_reasons"]
    assert "prior paper_ready must remain false" in artifact["blocked_reasons"]
    assert "prior publication blocker count is not 105" in artifact["blocked_reasons"]
    assert (
        "prior next_top_gap does not preserve the .303 corpus queue anchor"
        in artifact["blocked_reasons"]
    )
    assert (
        "research-complete.yaml does not contain the .302 task summary"
        in artifact["blocked_reasons"]
    )
    assert "selected queue milestone is not 2026.05.303" in artifact["blocked_reasons"]
    assert "protected files changed during handoff" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("complete:")

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_yaml_document(bad_yaml) == {}
    assert mod.read_yaml_document(tmp_path / "missing.yaml") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._milestone_entries("not-yaml-shape") == []
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._int_value(True) == 0
    assert mod._int_value(7) == 7
    assert mod._int_value("7") == 7
    assert mod._int_value("bad") == 0
    assert mod._float_value(0.5) == 0.5
    assert mod._float_value("0.25") == 0.25
    assert mod._float_value("bad") == 0.0
    assert mod._file_contains(tmp_path / "missing.log", "needle") is False
    assert mod._terminal_prefix_ok("success: done") is True
    assert mod._terminal_prefix_ok("blocked") is False

    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(no_verdict)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(bad_experiment)
    with pytest.raises(ValueError, match="task_id"):
        mod.validate_artifact(bad_task)
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(bad_milestone)
    with pytest.raises(ValueError, match="random_seed"):
        mod.validate_artifact(bad_seed)

"""Tests for Exp 3246 archive .300 and activate .301.

Spec refs: REQ-REPORT-3246, SCENARIO-REPORT-3246.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v300_activate_v301_3246 as mod


REQUIRED_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "archive_v300_activate_v301_ready",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "prior_local_sota_receipt_status",
    "prior_prompt_injection_v4_status",
    "prior_fr11_failure_memory_status",
    "next_top_gap",
    "queue_first_task",
    "queue_last_task",
    "protected_files_untouched",
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
    blockers: int = 106,
    local_status: str = "blocked",
    prompt_status: str = "gate_blocked",
    fr11_status: str = "complete",
    next_gap: str = mod.NEXT_TOP_GAP,
) -> dict[str, Any]:
    return {
        "schema_version": "carnot.milestone_capstone.v300_matrix_v33_closeout.v1",
        "experiment_id": "exp3245",
        "task_id": "exp3245-capstone-v300",
        "milestone": "2026.05.300",
        "capstone_v300_ready": ready,
        "paper_ready": paper_ready,
        "publication_blocker_count": blockers,
        "local_sota_receipt_state": {
            "status": local_status,
            "state": "blocked_selected_python_cuda_smoke_failed",
            "cuda_python_smoke_passed": False,
            "selected_python_torch_cuda_available": False,
            "blocking_artifacts": ["exp3236", "exp3237", "exp3238"],
        },
        "prompt_injection_v4_state": {
            "status": prompt_status,
            "manifest_ready": True,
            "teacher_label_shard_status": "gate_blocked",
            "train_eval_shard_status": "gate_blocked",
            "blocking_artifacts": ["exp3240", "exp3241"],
        },
        "fr11_failure_memory_state": {
            "status": fr11_status,
            "completed": fr11_status == "complete",
            "model_weight_update_claimed": False,
        },
        "next_top_gap": next_gap,
        "honest_verdict": (
            "complete: capstone_v300_ready=true; paper_ready=false; "
            "publication_blocker_count=106; local_sota_receipt_status=blocked; "
            "prompt_injection_v4_status=gate_blocked; "
            "fr11_failure_memory_status=complete; "
            "next_top_gap=repair_selected_python_torch_cuda_before_exp3237"
        ),
    }


def _roadmap_yaml(
    *,
    milestone: str = "2026.05.301",
    first_task: str = mod.TASK_ID,
    last_task: str = mod.EXPECTED_QUEUE_LAST_TASK,
) -> str:
    return "\n".join(
        [
            f'milestone: "{milestone}"',
            'milestone_title: "Selected-Python CUDA Repair"',
            f'milestone_doc: "{mod.VNEXT_DOC_REL_PATH.as_posix()}"',
            "tasks:",
            f'  - id: "{first_task}"',
            f'    milestone: "{milestone}"',
            f'    deliverable: "{mod.OUTPUT_REL_PATH.as_posix()}"',
            '  - id: "exp3247-selected-python-cuda-root-cause-surgery-v1"',
            f'    milestone: "{milestone}"',
            '    deliverable: "results/experiment_3247_selected_python_cuda_root_cause_surgery_v1.json"',
            f'  - id: "{last_task}"',
            f'    milestone: "{milestone}"',
            '    deliverable: "results/experiment_3258_capstone_v301.json"',
            "",
        ]
    )


def _conductor_log(*, activated: bool = True) -> str:
    lines = [
        "| 2026-05-28 04:44 UTC | Isolated CUDA and selected-Python smoke receipt | OK | 81 passed in 4.32s |\n",
        "| 2026-05-28 04:46 UTC | llama.cpp CUDA receipt smoke v2 gated on selected- | GATE_BLOCK | first failure: exp3236-isolated-cuda-python-smoke-v1.cuda |\n",
        "| 2026-05-28 04:52 UTC | Mandated local SOTA GGUF receipt v7 gated on llama | GATE_BLOCK | upstream retired exp3237 |\n",
        "| 2026-05-28 05:05 UTC | Prompt-injection KAN v4 resource manifest and shar | OK | 81 passed in 3.02s |\n",
        "| 2026-05-28 05:11 UTC | Prompt-injection KAN v4 shard train/eval with non- | GATE_BLOCK | exp3240 gate failed |\n",
        "| 2026-05-28 05:13 UTC | DCCD exact-row structured proposal preflight gated | GATE_BLOCK | upstream retired exp3238 |\n",
        "| 2026-05-28 05:25 UTC | FR-11 failure-memory controller for continuous sel | OK | 81 passed in 4.18s |\n",
        "| 2026-05-28 05:40 UTC | Cross-corpus matrix v33 for runtime, prompt-inject | OK | 81 passed in 3.33s |\n",
        "| 2026-05-28 05:53 UTC | Capstone .300 publication readiness and next-gap d | OK | 81 passed in 4.65s |\n",
    ]
    if activated:
        lines.append(
            "| 2026-05-28 06:40 UTC | Milestone 2026.05.301 activated | OK | 13 tasks queued |\n"
        )
    return "".join(lines)


def _status_text() -> str:
    return (
        "Milestone 2026.05.301 Research Planning Staged\n"
        "paper_ready=false publication_blocker_count=106 "
        "local_sota_receipt_status=blocked "
        "next_top_gap=repair_selected_python_torch_cuda_before_exp3237\n"
    )


def _changelog_text() -> str:
    return (
        "Planned milestone 2026.05.301 after .300 remained paper_ready=false "
        "with publication_blocker_count=106, local_sota_receipt_status=blocked, "
        "prompt-injection v4 and DCCD work gate-blocked behind the missing local "
        "SOTA receipt, FR-11 failure memory completed controller-side.\n"
    )


def _write_sources(
    tmp_path: Path,
    *,
    staged: bool = False,
    activated: bool = True,
    capstone: dict[str, Any] | None = None,
) -> None:
    _write_json(tmp_path, mod.CAPSTONE_V300_REL_PATH, capstone or _capstone_payload())
    _write_text(tmp_path, mod.CONDUCTOR_LOG_REL_PATH, _conductor_log(activated=activated))
    _write_text(tmp_path, mod.STATUS_REL_PATH, _status_text())
    _write_text(tmp_path, mod.CHANGELOG_REL_PATH, _changelog_text())
    _write_text(tmp_path, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT\n")
    _write_text(tmp_path, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml())
    if staged:
        _write_text(tmp_path, mod.STAGED_ROADMAP_REL_PATH, _roadmap_yaml())


def test_req_report_3246_spec_anchor_exists() -> None:
    """REQ-REPORT-3246: OpenSpec declares the archive/activation contract."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3246" in spec
    assert "SCENARIO-REPORT-3246" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3246_builds_active_queue_handoff(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3246: active .301 roadmap can confirm the handoff."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.25)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)
    sources = {row["role"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3246"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.301"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["archive_v300_activate_v301_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 106
    assert artifact["prior_local_sota_receipt_status"] == "blocked"
    assert artifact["prior_prompt_injection_v4_status"] == "gate_blocked"
    assert artifact["prior_fr11_failure_memory_status"] == "complete"
    assert artifact["next_top_gap"] == mod.NEXT_TOP_GAP
    assert artifact["queue_first_task"] == mod.TASK_ID
    assert artifact["queue_last_task"] == mod.EXPECTED_QUEUE_LAST_TASK
    assert artifact["queue_task_count"] == 3
    assert artifact["queue_paths"]["selected_queue_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_pre_activation_shape_observed"] is False
    assert artifact["activation_already_observed"] is True
    assert artifact["selected_python_cuda_boundary_summary"]["blocked_local_sota_receipts"] is True
    assert artifact["selected_python_cuda_boundary_summary"][
        "blocked_downstream_prompt_injection_tasks"
    ] is True
    assert artifact["selected_python_cuda_boundary_summary"][
        "blocked_downstream_structured_repair_tasks"
    ] is True
    assert "exp3242" in artifact["selected_python_cuda_boundary_summary"]["downstream_blocked_tasks"]
    assert artifact["protected_files"] == [
        "research-roadmap.yaml",
        "scripts/research_conductor.py",
    ]
    assert artifact["protected_files_untouched"] == {
        "research-roadmap.yaml": True,
        "scripts/research_conductor.py": True,
    }
    assert artifact["principle_annotations"]["honest_verdict"].startswith("Terminal")
    assert artifact["no_new_kan_training"] is True
    assert artifact["no_conductor_execution"] is True
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "paper_ready=true" not in artifact["honest_verdict"]
    assert sources["capstone_v300"]["sha256"] == _sha256(tmp_path / mod.CAPSTONE_V300_REL_PATH)


def test_req_report_3246_prefers_staged_queue_when_present(tmp_path: Path) -> None:
    """REQ-REPORT-3246: staged .301 roadmap is accepted before activation."""

    _write_sources(tmp_path, staged=True, activated=False)
    _write_text(
        tmp_path,
        mod.ACTIVE_ROADMAP_REL_PATH,
        _roadmap_yaml(
            milestone="2026.05.300",
            first_task="exp3233-archive-v299-activate-v300",
            last_task="exp3245-capstone-v300",
        ),
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["archive_v300_activate_v301_ready"] is True
    assert artifact["queue_paths"]["selected_queue_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_pre_activation_shape_observed"] is True
    assert artifact["activation_already_observed"] is False
    assert artifact["queue_first_task"] == mod.TASK_ID
    assert artifact["queue_last_task"] == mod.EXPECTED_QUEUE_LAST_TASK


def test_req_report_3246_writer_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3246: writer persists JSON and malformed inputs stay explicit."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=6.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v300_activate_v301_ready"] is True
    assert saved["duration_s"] == pytest.approx(2.0)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("tasks: [", encoding="utf-8")

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_yaml_document(bad_yaml) == {}
    assert mod.read_yaml_document(tmp_path / "missing.yaml") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._int_value(True) == 0
    assert mod._int_value("106") == 106
    assert mod._status_from_capstone({"flat_status": "flat"}, "missing", "flat_status") == "flat"
    assert mod._duration(3.0, 2.0) == 0.0

    empty = mod.build_artifact(tmp_path / "empty", started_s=3.0, now_s=2.0)
    assert empty["archive_v300_activate_v301_ready"] is False
    assert empty["honest_verdict"].startswith("complete:")
    assert "capstone_v300 authority is missing or malformed" in empty["blocked_reasons"]

    wrong = _capstone_payload(
        ready=False,
        paper_ready=True,
        blockers=105,
        local_status="complete",
        prompt_status="complete",
        fr11_status="blocked",
        next_gap="wrong_gap",
    )
    _write_sources(tmp_path, capstone=wrong)
    not_ready = mod.build_artifact(tmp_path)
    assert "capstone_v300 authority is not ready" in not_ready["blocked_reasons"]
    assert "prior paper_ready must remain false" in not_ready["blocked_reasons"]
    assert "prior publication blocker count is not 106" in not_ready["blocked_reasons"]
    assert "local SOTA receipt status is not blocked" in not_ready["blocked_reasons"]
    assert "prompt-injection v4 status is not gate_blocked" in not_ready["blocked_reasons"]
    assert "FR-11 failure-memory status is not complete" in not_ready["blocked_reasons"]
    assert "next_top_gap does not preserve the selected-Python CUDA gap" in not_ready[
        "blocked_reasons"
    ]


def test_req_report_3246_blocks_on_wrong_queue_shape(tmp_path: Path) -> None:
    """REQ-REPORT-3246: queue activation facts cannot name the wrong boundary."""

    _write_sources(tmp_path)
    _write_text(
        tmp_path,
        mod.ACTIVE_ROADMAP_REL_PATH,
        _roadmap_yaml(
            milestone="2026.05.301",
            first_task="exp9999-wrong-first",
            last_task="exp9999-wrong-last",
        ),
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["archive_v300_activate_v301_ready"] is False
    assert "selected queue first task is not exp3246-archive-v300-activate-v301" in artifact[
        "blocked_reasons"
    ]
    assert "selected queue last task is not exp3258-capstone-v301" in artifact["blocked_reasons"]

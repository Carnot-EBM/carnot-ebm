"""Tests for Exp 3219 archive .297 and activate .298.

Spec refs: REQ-REPORT-3219, SCENARIO-REPORT-3219.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v297_activate_v298_3219 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "prior_milestone",
    "prior_task_range",
    "capstone_artifact",
    "matrix_artifact",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "prior_next_top_gap",
    "activation_ready",
    "research_roadmap_next_exists",
    "inference_substrate",
    "conductor_file_modified",
    "active_roadmap_modified",
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


def _upstream_row(exp_id: str, status: str, *, gate_blocked: bool = False) -> dict[str, Any]:
    path = {
        "exp3205": "results/experiment_3205_archive_v296_activate_v297.json",
        "exp3206": "results/experiment_3206_cuda_env_forensics_ledger_v1.json",
        "exp3207": "results/experiment_3207_llama_cpp_cuda_rebuild_clean_subprocess_v1.json",
        "exp3208": "results/experiment_3208_full_local_sota_receipt_v5.json",
        "exp3209": "results/experiment_3209_clean_live_sota_verifier_rerun_v12.json",
        "exp3210": (
            "results/experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.json"
        ),
        "exp3211": "results/experiment_3211_constraintbench_feasibility_objective_pilot_v1.json",
        "exp3212": "results/experiment_3212_structured_repair_proposal_preflight_v1.json",
        "exp3213": "results/experiment_3213_repair_gate_decision_v6.json",
        "exp3214": "results/experiment_3214_multi_turn_repair_ladder_v7.json",
        "exp3215": "results/experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.json",
        "exp3216": "results/experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.json",
    }[exp_id]
    row: dict[str, Any] = {
        "experiment_id": exp_id,
        "path": path,
        "role": f"{exp_id}_role",
        "source_field": f"{exp_id}_field",
        "status": status,
        "status_rationale": f"{status} rationale",
        "present": status != "missing",
        "readable_json_object": status != "missing",
        "honest_verdict": "complete: ok" if status == "clean" else f"{status}: terminal",
    }
    if gate_blocked:
        row["gated_skip_evidence"] = {
            "source": "ops/conductor-log.md",
            "status": "gated_skipped",
            "line": f"| gate blocked {exp_id} |",
        }
    return row


def _matrix_payload(
    *,
    ready: bool = True,
    blocker_count: int = 92,
    next_gap: str = mod.TOP_UNRESOLVED_GAP,
) -> dict[str, Any]:
    return {
        "schema_version": "carnot.cross_corpus_matrix.v31_297_artifact_aggregation.v1",
        "experiment_id": "exp3217",
        "milestone": "2026.05.297",
        "cross_corpus_matrix_v31_ready": ready,
        "paper_ready": False,
        "publication_blocker_count": blocker_count,
        "blocker_delta_from_v30": 7,
        "next_top_gap": next_gap,
        "honest_verdict": "complete: cross_corpus_matrix_v31_ready=true",
        "upstream_artifacts": [
            _upstream_row("exp3205", "clean"),
            _upstream_row("exp3206", "blocked"),
            _upstream_row("exp3207", "blocked"),
            _upstream_row("exp3208", "gated_skipped", gate_blocked=True),
            _upstream_row("exp3209", "missing", gate_blocked=True),
            _upstream_row("exp3210", "clean"),
            _upstream_row("exp3211", "clean"),
            _upstream_row("exp3212", "missing", gate_blocked=True),
            _upstream_row("exp3213", "blocked"),
            _upstream_row("exp3214", "gated_skipped", gate_blocked=True),
            _upstream_row("exp3215", "clean"),
            _upstream_row("exp3216", "diagnostic_only"),
        ],
    }


def _capstone_payload(
    *,
    ready: bool = True,
    blocker_count: int = 92,
    next_gap: str = mod.TOP_UNRESOLVED_GAP,
) -> dict[str, Any]:
    return {
        "schema_version": "carnot.milestone_capstone.v297_matrix_v31_terminal_aggregation.v1",
        "experiment_id": "exp3218",
        "milestone": "2026.05.297",
        "capstone_v297_ready": ready,
        "paper_ready": False,
        "publication_blocker_count": blocker_count,
        "blocker_delta_from_v30": 7,
        "next_top_gap": next_gap,
        "honest_verdict": "complete: capstone_v297_ready=true",
    }


def _task_slug(exp_id: str) -> str:
    slugs = {
        "exp3205": "archive-v296-activate-v297",
        "exp3206": "cuda-env-forensics-ledger-v1",
        "exp3207": "llama-cpp-cuda-rebuild-clean-subprocess-v1",
        "exp3208": "full-local-sota-receipt-v5",
        "exp3209": "clean-live-sota-verifier-rerun-v12",
        "exp3210": "context-cot-clbench-parametric-shortcut-fixtures-v1",
        "exp3211": "constraintbench-feasibility-objective-pilot-v1",
        "exp3212": "structured-repair-proposal-preflight-v1",
        "exp3213": "repair-gate-decision-v6",
        "exp3214": "multi-turn-repair-ladder-v7",
        "exp3215": "fr11-evidence-gated-trace-replay-controller-v2",
        "exp3216": "fr11-grounded-continuation-nonforgetting-queue-v1",
        "exp3217": "cross-corpus-matrix-v31",
        "exp3218": "capstone-v297",
    }
    return slugs[exp_id]


def _research_complete_yaml(*, missing_last: bool = False) -> str:
    task_lines: list[str] = []
    exp_ids = list(mod.EXPECTED_PRIOR_EXPERIMENT_IDS)
    if missing_last:
        exp_ids = exp_ids[:-1]
    for exp_id in exp_ids:
        task_lines.extend(
            [
                f"  - id: {exp_id}-{_task_slug(exp_id)}",
                f"    title: {exp_id} terminal task",
                f"    deliverable: results/{exp_id}.json",
                "    result: OK (conductor)",
            ]
        )
    return "\n".join(
        [
            "- id: 2026.05.297",
            "  title: CUDA Receipt Recovery",
            "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
            "  completed: '2026-05-27'",
            "  tasks:",
            *task_lines,
            "",
        ]
    )


def _roadmap_yaml(*, milestone: str = "2026.05.298", first_task: str | None = None) -> str:
    first = first_task or mod.FIRST_V298_TASK_ID
    task_ids = [
        first,
        "exp3220-hermetic-cuda-runtime-repair-ledger-v1",
        "exp3221-llama-cpp-cuda-offload-receipt-smoke-v1",
        "exp3222-full-local-sota-receipt-v6",
        "exp3225-clean-live-sota-verifier-rerun-v13",
        "exp3226-structured-repair-proposal-preflight-v2",
        "exp3227-repair-gate-decision-v7",
        "exp3228-multi-turn-repair-ladder-v8",
        "exp3231-cross-corpus-matrix-v32",
        "exp3232-capstone-v298",
    ]
    lines = [
        f'milestone: "{milestone}"',
        'milestone_title: "Hermetic CUDA Receipt Repair"',
        f'milestone_doc: "{mod.VNEXT_DOC_REL_PATH.as_posix()}"',
        "tasks:",
    ]
    for task_id in task_ids:
        lines.extend(
            [
                f"  - id: {task_id}",
                f'    milestone: "{milestone}"',
                f'    deliverable: "results/{task_id}.json"',
            ]
        )
    return "\n".join(lines) + "\n"


def _write_common_sources(
    root: Path,
    *,
    staged_roadmap: bool = False,
    active_milestone: str = "2026.05.298",
    capstone_ready: bool = True,
    matrix_ready: bool = True,
    missing_last_task: bool = False,
) -> None:
    _write_json(root, mod.CAPSTONE_V297_REL_PATH, _capstone_payload(ready=capstone_ready))
    _write_json(root, mod.MATRIX_V31_REL_PATH, _matrix_payload(ready=matrix_ready))
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_REL_PATH,
        _research_complete_yaml(missing_last=missing_last_task),
    )
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone=active_milestone))
    if staged_roadmap:
        _write_text(root, mod.STAGED_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "| 2026-05-27 23:24 UTC | Milestone 2026.05.298 activated | OK | 14 tasks queued |\n",
    )
    _write_text(root, mod.CHANGELOG_REL_PATH, "## 2026-05-27 (Milestone 2026.05.297)\n")
    _write_text(root, mod.STATUS_REL_PATH, "# status\n")
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT - Milestone 2026.05.298\n")
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")


def test_req_report_3219_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3219: OpenSpec declares the archive/activation contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3219" in spec
    assert "SCENARIO-REPORT-3219" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3219_builds_ready_artifact_from_active_queue(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3219: active .298 roadmap can confirm activation."""

    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    statuses = {row["experiment_id"]: row for row in artifact["prior_terminal_statuses"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3219"
    assert artifact["milestone"] == "2026.05.298"
    assert artifact["prior_milestone"] == "2026.05.297"
    assert artifact["prior_task_range"] == ["exp3205", "exp3218"]
    assert artifact["capstone_artifact"] == mod.CAPSTONE_V297_REL_PATH.as_posix()
    assert artifact["matrix_artifact"] == mod.MATRIX_V31_REL_PATH.as_posix()
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 92
    assert artifact["prior_next_top_gap"] == mod.TOP_UNRESOLVED_GAP
    assert artifact["activation_ready"] is True
    assert artifact["research_roadmap_next_exists"] is False
    assert artifact["inference_substrate"] == "artifact_aggregation_only"
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(statuses) == set(mod.EXPECTED_PRIOR_EXPERIMENT_IDS)
    assert statuses["exp3208"]["artifact_status"] == "gated_skipped"
    assert statuses["exp3209"]["gate_blocked"] is True
    assert statuses["exp3212"]["artifact_status"] == "missing"
    assert statuses["exp3217"]["artifact_status"] == "clean"
    assert statuses["exp3218"]["artifact_status"] == "clean"
    assert artifact["terminal_status_counts"] == {
        "blocked": 3,
        "clean": 6,
        "diagnostic_only": 1,
        "gated_skipped": 2,
        "missing": 2,
    }
    assert artifact["gate_blocked_prior_tasks"] == ["exp3208", "exp3209", "exp3212", "exp3214"]
    assert artifact["queue_paths"] == {
        "staged_roadmap_path": mod.STAGED_ROADMAP_REL_PATH.as_posix(),
        "staged_roadmap_exists": False,
        "active_roadmap_path": mod.ACTIVE_ROADMAP_REL_PATH.as_posix(),
        "active_roadmap_exists": True,
        "selected_queue_path": mod.ACTIVE_ROADMAP_REL_PATH.as_posix(),
        "selected_queue_milestone": "2026.05.298",
        "selected_queue_first_task": mod.FIRST_V298_TASK_ID,
        "selected_queue_task_count": 10,
        "milestone_doc": mod.VNEXT_DOC_REL_PATH.as_posix(),
        "milestone_doc_exists": True,
    }
    assert artifact["critical_path"]["top_unresolved_gap"] == mod.TOP_UNRESOLVED_GAP
    assert artifact["critical_path"]["unblock_sequence"][:3] == [
        "exp3220-hermetic-cuda-runtime-repair-ledger-v1",
        "exp3221-llama-cpp-cuda-offload-receipt-smoke-v1",
        "exp3222-full-local-sota-receipt-v6",
    ]
    assert artifact["source_checksums"][mod.CAPSTONE_V297_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V297_REL_PATH
    )
    assert artifact["conductor_activation_observed"] is True
    assert artifact["new_milestone_document_exists"] is True
    assert artifact["no_new_model_execution"] is True
    assert artifact["no_conductor_execution"] is True


def test_req_report_3219_records_staged_queue_when_present(tmp_path: Path) -> None:
    """REQ-REPORT-3219: staged roadmap materialization is reported directly."""

    _write_common_sources(tmp_path, staged_roadmap=True, active_milestone="2026.05.297")

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["activation_ready"] is True
    assert artifact["research_roadmap_next_exists"] is True
    assert artifact["queue_paths"]["selected_queue_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["queue_paths"]["selected_queue_milestone"] == "2026.05.298"
    assert artifact["queue_paths"]["active_roadmap_exists"] is True


def test_req_report_3219_blocks_on_unready_sources_or_queue(tmp_path: Path) -> None:
    """REQ-REPORT-3219: source and queue mismatches fail closed."""

    _write_common_sources(tmp_path, capstone_ready=False, active_milestone="2026.05.297")

    artifact = mod.build_artifact(tmp_path)

    assert artifact["activation_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_activation_not_ready:")
    assert "capstone_v297 authority is not ready" in artifact["blocked_reasons"]
    assert "selected queue milestone is not 2026.05.298" in artifact["blocked_reasons"]

    _write_common_sources(tmp_path, missing_last_task=True)
    missing_task = mod.build_artifact(tmp_path)

    assert missing_task["activation_ready"] is False
    assert (
        "prior terminal statuses do not cover exp3205 through exp3218"
        in missing_task["blocked_reasons"]
    )


def test_req_report_3219_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3219: writer persists stable JSON and helpers fail closed."""

    _write_common_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=3.0, now_s=3.125)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["activation_ready"] is True
    assert saved["duration_s"] == pytest.approx(0.125)

    bad_json = tmp_path / "bad.json"
    list_json = tmp_path / "list.json"
    bad_yaml = tmp_path / "bad.yaml"
    list_yaml = tmp_path / "list.yaml"
    null_yaml = tmp_path / "null.yaml"
    bad_json.write_text("{", encoding="utf-8")
    list_json.write_text("[1]\n", encoding="utf-8")
    bad_yaml.write_text(": bad\n", encoding="utf-8")
    list_yaml.write_text("- just\n- a\n- list\n", encoding="utf-8")
    null_yaml.write_text("null\n", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_yaml_mapping(tmp_path / "missing.yaml") == {}
    assert mod.read_yaml_mapping(bad_yaml) == {}
    assert mod.read_yaml_mapping(list_yaml) == {"_root_list": ["just", "a", "list"]}
    assert mod.read_yaml_mapping(null_yaml) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.expected_next_milestone("2026.05.297") == "2026.05.298"
    assert mod.expected_next_milestone("bad") == ""
    assert mod._duration(5.0, 4.0) == 0.0
    assert mod._task_ids({"tasks": [{"id": "x"}, {"bad": "y"}, "bad"]}) == ["x"]
    assert mod._task_ids({"tasks": "bad"}) == []
    assert mod._terminal_status_counts([]) == {}
    assert mod._prior_task_range_complete([]) is False
    assert (
        mod._find_milestone_entry({"id": "2026.05.297", "tasks": []}, "2026.05.297")
        == {"id": "2026.05.297", "tasks": []}
    )
    assert mod._find_milestone_entry({"_root_list": []}, "2026.05.297") == {}
    assert mod._file_contains(tmp_path / "missing.txt", "needle") is False
    assert mod._file_contains(tmp_path / mod.CONDUCTOR_LOG_REL_PATH, "needle") is False
    assert mod._int_value(True) == 0
    assert mod._int_value("7") == 7
    assert mod._int_value("bad") == 0
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list(["x"]) == ["x"]
    assert mod._as_list("x") == []
    assert mod._conductor_status({"exp3205": {"result": "OK"}}, "exp3205") == "OK"
    assert mod._conductor_status({}, "exp3205") == ""
    assert mod._artifact_status({}, False) == "blocked"
    weird_matrix = _matrix_payload()
    weird_matrix["upstream_artifacts"][0]["status"] = "unexpected"
    weird_rows = mod._prior_terminal_statuses(
        weird_matrix,
        _capstone_payload(),
        {"_root_list": [{"id": "2026.05.297", "tasks": ["not-a-mapping"]}]},
    )
    assert weird_rows[0]["artifact_status"] == "missing"
    assert weird_rows[0]["terminal"] is False

    missing_reasons = mod._blocked_reasons(
        capstone={},
        matrix={},
        capstone_ready=False,
        matrix_ready=False,
        prior_next_gap="wrong-gap",
        prior_terminal_statuses=[],
        queue_paths={
            "selected_queue_milestone": "",
            "selected_queue_first_task": "",
            "milestone_doc": "",
            "selected_queue_task_count": 0,
        },
        vnext_doc_exists=False,
    )
    assert missing_reasons == [
        "capstone_v297 authority is missing or malformed",
        "matrix_v31 authority is missing or malformed",
        "prior next_top_gap does not match the .298 critical path",
        "prior terminal statuses do not cover exp3205 through exp3218",
        "selected queue milestone is not 2026.05.298",
        "selected queue first task is not exp3219-archive-v297-activate-v298",
        "selected queue milestone_doc is not the vNEXT document",
        "selected queue has no tasks",
        "openspec/change-proposals/research-roadmap-vNEXT.md is missing",
    ]

    mismatch_reasons = mod._blocked_reasons(
        capstone={"publication_blocker_count": 92},
        matrix={"publication_blocker_count": 93, "next_top_gap": "different"},
        capstone_ready=True,
        matrix_ready=False,
        prior_next_gap=mod.TOP_UNRESOLVED_GAP,
        prior_terminal_statuses=[
            {"experiment_id": exp_id, "terminal": True}
            for exp_id in mod.EXPECTED_PRIOR_EXPERIMENT_IDS
        ],
        queue_paths={
            "selected_queue_milestone": mod.MILESTONE,
            "selected_queue_first_task": mod.FIRST_V298_TASK_ID,
            "milestone_doc": mod.VNEXT_DOC_REL_PATH.as_posix(),
            "selected_queue_task_count": 1,
        },
        vnext_doc_exists=True,
    )
    assert mismatch_reasons == [
        "matrix_v31 authority is not ready",
        "capstone and matrix publication blocker counts disagree",
        "matrix_v31 next_top_gap disagrees with capstone authority",
    ]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(mod, "MILESTONE", "2026.05.299")
        assert "expected CalVer sequence does not produce 2026.05.298" in mod._blocked_reasons(
            capstone={"publication_blocker_count": 92},
            matrix={"publication_blocker_count": 92, "next_top_gap": mod.TOP_UNRESOLVED_GAP},
            capstone_ready=True,
            matrix_ready=True,
            prior_next_gap=mod.TOP_UNRESOLVED_GAP,
            prior_terminal_statuses=[
                {"experiment_id": exp_id, "terminal": True}
                for exp_id in mod.EXPECTED_PRIOR_EXPERIMENT_IDS
            ],
            queue_paths={
                "selected_queue_milestone": "2026.05.299",
                "selected_queue_first_task": mod.FIRST_V298_TASK_ID,
                "milestone_doc": mod.VNEXT_DOC_REL_PATH.as_posix(),
                "selected_queue_task_count": 1,
            },
            vnext_doc_exists=True,
        )

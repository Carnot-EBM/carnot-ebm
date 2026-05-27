"""Tests for Exp 3191 archive .295 and activate .296.

Spec refs: REQ-REPORT-3191, SCENARIO-REPORT-3191.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v295_activate_v296_3191 as mod


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


def _capstone_payload(
    *, ready: bool = True, next_gap: str = mod.TOP_UNRESOLVED_GAP
) -> dict[str, Any]:
    return {
        "schema": "carnot.milestone_capstone.v295_matrix_v29_aggregation.v1",
        "artifact": "experiment_3190_capstone_v295",
        "milestone": "2026.05.295",
        "capstone_v295_ready": ready,
        "paper_ready": False,
        "publication_blocker_count": 80,
        "blocker_delta_from_v28": 7,
        "missing_artifact_count": 1,
        "next_top_gap": next_gap,
        "local_sota_receipt_status": ("cpu_fallback_receipt_only_non_headline_clean_rerun_blocked"),
        "verifier_status": (
            "gated_skip_cpu_fallback_receipt_only_flagged_adversarial_"
            "controlled_invariance_passed_exact_authority_only"
        ),
        "repair_gate_status": "blocked_receipt_precondition",
        "repair_ladder_status": (
            "materialized_gated_skip_repair_gate_blocked_no_live_repair_attempts"
        ),
        "fr11_self_learning_status": (
            "controller_memory_promotion_allowed_cross_environment_replay_passed_"
            "no_model_weight_update"
        ),
        "hardware_sampler_status": (
            "diagnostic_only_thrml_local_api_smoke_no_kv260_speedup_no_tsu_kona_execution"
        ),
        "honest_verdict": "complete: capstone_v295_ready=true",
    }


def _matrix_payload(
    *, ready: bool = True, next_gap: str = mod.TOP_UNRESOLVED_GAP
) -> dict[str, Any]:
    return {
        "schema": "carnot.cross_corpus_matrix.v29_295_artifact_aggregation.v1",
        "artifact": "experiment_3189_cross_corpus_matrix_v29",
        "milestone": "2026.05.295",
        "cross_corpus_matrix_v29_ready": ready,
        "paper_ready": False,
        "publication_blocker_count": 80,
        "blocker_delta_from_v28": 7,
        "next_top_gap": next_gap,
        "honest_verdict": "complete: cross_corpus_matrix_v29_ready=true",
    }


def _task_slug(exp_id: str) -> str:
    slugs = {
        "exp3177": "archive-v294-activate-v295",
        "exp3178": "receipt-backed-authenticity-contract-v3",
        "exp3179": "local-sota-receipt-smoke-v3",
        "exp3180": "controlled-invariance-executor-v2",
        "exp3181": "clean-live-sota-verifier-rerun-v10",
        "exp3182": "distributional-ebm-exact-row-sidecar-v1",
        "exp3183": "counterexample-certificate-expansion-v3",
        "exp3184": "repair-gate-decision-v4",
        "exp3185": "multi-turn-repair-ladder-v5",
        "exp3186": "fr11-controller-memory-promotion-pack-v1",
        "exp3187": "fr11-cross-environment-drift-replay-v1",
        "exp3188": "thrml-factor-graph-api-boundary-v1",
        "exp3189": "cross-corpus-matrix-v29",
        "exp3190": "capstone-v295",
    }
    return slugs[exp_id]


def _research_complete_yaml(*, missing_last: bool = False) -> str:
    task_lines: list[str] = []
    exp_ids = list(mod.EXPECTED_PRIOR_EXPERIMENT_IDS)
    if missing_last:
        exp_ids = exp_ids[:-1]
    for exp_id in exp_ids:
        task_id = f"{exp_id}-{_task_slug(exp_id)}"
        task_lines.extend(
            [
                f"  - id: {task_id}",
                f"    title: {exp_id} terminal task",
                f"    deliverable: results/{exp_id}.json",
                "    result: OK (conductor)",
            ]
        )
    return "\n".join(
        [
            "- id: 2026.05.295",
            "  title: Receipt-Backed Live SOTA Clearance",
            "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
            "  completed: '2026-05-27'",
            "  tasks:",
            *task_lines,
            "",
        ]
    )


def _roadmap_yaml(*, milestone: str = "2026.05.296", first_task: str | None = None) -> str:
    first = first_task or mod.FIRST_V296_TASK_ID
    return "\n".join(
        [
            f'milestone: "{milestone}"',
            'milestone_title: "CUDA-Backed SOTA Receipt Recovery"',
            f'milestone_doc: "{mod.VNEXT_DOC_REL_PATH.as_posix()}"',
            "tasks:",
            f"  - id: {first}",
            f'    milestone: "{milestone}"',
            f'    deliverable: "{mod.OUTPUT_REL_PATH.as_posix()}"',
            "  - id: exp3192-receipt-adversarial-contract-v4",
            f'    milestone: "{milestone}"',
            '    deliverable: "results/experiment_3192_receipt_adversarial_contract_v4.json"',
            "  - id: exp3193-llama-cpp-cuda-offload-health-probe-v1",
            f'    milestone: "{milestone}"',
            '    deliverable: "results/experiment_3193_llama_cpp_cuda_offload_health_probe_v1.json"',
            "  - id: exp3194-clean-live-sota-verifier-rerun-v11",
            f'    milestone: "{milestone}"',
            '    deliverable: "results/experiment_3194_clean_live_sota_verifier_rerun_v11.json"',
            "  - id: exp3198-repair-gate-decision-v5",
            f'    milestone: "{milestone}"',
            '    deliverable: "results/experiment_3198_repair_gate_decision_v5.json"',
            "  - id: exp3199-multi-turn-repair-ladder-v6",
            f'    milestone: "{milestone}"',
            '    deliverable: "results/experiment_3199_multi_turn_repair_ladder_v6.json"',
            "",
        ]
    )


def _write_common_sources(
    root: Path,
    *,
    staged_roadmap: bool = False,
    active_milestone: str = "2026.05.296",
    capstone_ready: bool = True,
    matrix_ready: bool = True,
    missing_last_task: bool = False,
) -> None:
    _write_json(root, mod.CAPSTONE_V295_REL_PATH, _capstone_payload(ready=capstone_ready))
    _write_json(root, mod.MATRIX_V29_REL_PATH, _matrix_payload(ready=matrix_ready))
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
        "| 2026-05-27 14:59 UTC | Milestone 2026.05.296 activated | OK | 14 tasks queued |\n",
    )
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT - Milestone 2026.05.296\n")
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")


def test_req_report_3191_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3191: OpenSpec declares the archive/activation contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3191" in spec
    assert "SCENARIO-REPORT-3191" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3191_builds_ready_artifact_from_active_queue(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3191: active .296 roadmap can confirm activation."""

    _write_common_sources(tmp_path, staged_roadmap=False)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.25)
    statuses = {row["experiment_id"]: row for row in artifact["prior_terminal_statuses"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3191"
    assert artifact["milestone"] == "2026.05.296"
    assert artifact["prior_milestone"] == "2026.05.295"
    assert artifact["prior_task_range"] == ["exp3177", "exp3190"]
    assert artifact["capstone_artifact"] == mod.CAPSTONE_V295_REL_PATH.as_posix()
    assert artifact["matrix_artifact"] == mod.MATRIX_V29_REL_PATH.as_posix()
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 80
    assert artifact["prior_next_top_gap"] == mod.TOP_UNRESOLVED_GAP
    assert artifact["activation_ready"] is True
    assert artifact["research_roadmap_next_exists"] is False
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["prior_terminal_statuses"]) == 14
    assert set(statuses) == set(mod.EXPECTED_PRIOR_EXPERIMENT_IDS)
    assert statuses["exp3177"]["terminal_status"] == "OK (conductor)"
    assert statuses["exp3190"]["terminal_status"] == "OK (conductor)"
    assert artifact["terminal_status_counts"] == {"OK (conductor)": 14}
    assert artifact["queue_paths"] == {
        "staged_roadmap_path": mod.STAGED_ROADMAP_REL_PATH.as_posix(),
        "staged_roadmap_exists": False,
        "active_roadmap_path": mod.ACTIVE_ROADMAP_REL_PATH.as_posix(),
        "active_roadmap_exists": True,
        "selected_queue_path": mod.ACTIVE_ROADMAP_REL_PATH.as_posix(),
        "selected_queue_milestone": "2026.05.296",
        "selected_queue_first_task": mod.FIRST_V296_TASK_ID,
        "selected_queue_task_count": 6,
        "milestone_doc": mod.VNEXT_DOC_REL_PATH.as_posix(),
        "milestone_doc_exists": True,
    }
    assert artifact["critical_path"]["top_unresolved_gap"] == mod.TOP_UNRESOLVED_GAP
    assert artifact["critical_path"]["unblock_sequence"][:4] == [
        "exp3192-receipt-adversarial-contract-v4",
        "exp3193-llama-cpp-cuda-offload-health-probe-v1",
        "exp3194-clean-live-sota-verifier-rerun-v11",
        "exp3198-repair-gate-decision-v5",
    ]
    assert artifact["source_checksums"][mod.CAPSTONE_V295_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V295_REL_PATH
    )
    assert artifact["no_new_model_execution"] is True
    assert artifact["no_conductor_execution"] is True


def test_req_report_3191_records_staged_queue_when_present(tmp_path: Path) -> None:
    """REQ-REPORT-3191: staged roadmap materialization is reported directly."""

    _write_common_sources(tmp_path, staged_roadmap=True, active_milestone="2026.05.295")

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["activation_ready"] is True
    assert artifact["research_roadmap_next_exists"] is True
    assert artifact["queue_paths"]["selected_queue_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["queue_paths"]["selected_queue_milestone"] == "2026.05.296"
    assert artifact["queue_paths"]["active_roadmap_exists"] is True


def test_req_report_3191_blocks_on_unready_sources_or_queue(tmp_path: Path) -> None:
    """REQ-REPORT-3191: source and queue mismatches fail closed."""

    _write_common_sources(tmp_path, capstone_ready=False, active_milestone="2026.05.295")

    artifact = mod.build_artifact(tmp_path)

    assert artifact["activation_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_activation_not_ready:")
    assert "capstone_v295 authority is not ready" in artifact["blocked_reasons"]
    assert "selected queue milestone is not 2026.05.296" in artifact["blocked_reasons"]

    _write_common_sources(tmp_path, missing_last_task=True)
    missing_task = mod.build_artifact(tmp_path)

    assert missing_task["activation_ready"] is False
    assert (
        "prior terminal statuses do not cover exp3177 through exp3190"
        in missing_task["blocked_reasons"]
    )


def test_req_report_3191_write_artifact_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3191: writer persists stable JSON and helpers fail closed."""

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
    assert mod.expected_next_milestone("2026.05.295") == "2026.05.296"
    assert mod.expected_next_milestone("bad") == ""
    assert mod._duration(5.0, 4.0) == 0.0
    assert mod._task_ids({"tasks": [{"id": "x"}, {"bad": "y"}, "bad"]}) == ["x"]
    assert mod._task_ids({"tasks": "bad"}) == []
    assert mod._terminal_status_counts([]) == {}
    assert mod._prior_task_range_complete([]) is False
    assert (
        mod._prior_terminal_statuses(
            {"_root_list": [{"id": "2026.05.295", "tasks": ["not-a-mapping"]}]}
        )[0]["terminal_status"]
        == "missing"
    )
    assert mod._find_milestone_entry({"id": "2026.05.295", "tasks": []}, "2026.05.295") == {
        "id": "2026.05.295",
        "tasks": [],
    }
    assert mod._find_milestone_entry({"_root_list": []}, "2026.05.295") == {}
    assert mod._int_value(True) == 0
    assert mod._int_value("7") == 7
    assert mod._int_value("bad") == 0
    assert mod._as_mapping({"x": 1}) == {"x": 1}
    assert mod._as_mapping(None) == {}
    assert mod._as_list(["x"]) == ["x"]
    assert mod._as_list("x") == []

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
        "capstone_v295 authority is missing or malformed",
        "matrix_v29 authority is missing or malformed",
        "prior next_top_gap does not match the .296 critical path",
        "prior terminal statuses do not cover exp3177 through exp3190",
        "selected queue milestone is not 2026.05.296",
        "selected queue first task is not exp3191-archive-v295-activate-v296",
        "selected queue milestone_doc is not the vNEXT document",
        "selected queue has no tasks",
        "openspec/change-proposals/research-roadmap-vNEXT.md is missing",
    ]

    mismatch_reasons = mod._blocked_reasons(
        capstone={"publication_blocker_count": 80},
        matrix={"publication_blocker_count": 81},
        capstone_ready=True,
        matrix_ready=False,
        prior_next_gap=mod.TOP_UNRESOLVED_GAP,
        prior_terminal_statuses=[
            {"experiment_id": exp_id, "terminal": True}
            for exp_id in mod.EXPECTED_PRIOR_EXPERIMENT_IDS
        ],
        queue_paths={
            "selected_queue_milestone": mod.MILESTONE,
            "selected_queue_first_task": mod.FIRST_V296_TASK_ID,
            "milestone_doc": mod.VNEXT_DOC_REL_PATH.as_posix(),
            "selected_queue_task_count": 1,
        },
        vnext_doc_exists=True,
    )
    assert mismatch_reasons == [
        "matrix_v29 authority is not ready",
        "capstone and matrix publication blocker counts disagree",
    ]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(mod, "MILESTONE", "2026.05.297")
        assert "expected CalVer sequence does not produce 2026.05.296" in mod._blocked_reasons(
            capstone={"publication_blocker_count": 80},
            matrix={"publication_blocker_count": 80},
            capstone_ready=True,
            matrix_ready=True,
            prior_next_gap=mod.TOP_UNRESOLVED_GAP,
            prior_terminal_statuses=[
                {"experiment_id": exp_id, "terminal": True}
                for exp_id in mod.EXPECTED_PRIOR_EXPERIMENT_IDS
            ],
            queue_paths={
                "selected_queue_milestone": "2026.05.297",
                "selected_queue_first_task": mod.FIRST_V296_TASK_ID,
                "milestone_doc": mod.VNEXT_DOC_REL_PATH.as_posix(),
                "selected_queue_task_count": 1,
            },
            vnext_doc_exists=True,
        )

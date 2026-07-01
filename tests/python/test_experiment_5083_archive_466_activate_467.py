"""Tests for Exp 5083 .466-to-.467 transition recording.

Spec refs: REQ-CAPSTONE-5083, SCENARIO-CAPSTONE-5083,
SCENARIO-CAPSTONE-5083-BLOCKED-YAML,
SCENARIO-CAPSTONE-5083-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5083_archive_466_activate_467 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_active_roadmap(root: Path, milestone: str = mod.MILESTONE_TO) -> None:
    (root / mod.ROADMAP_ACTIVE_REL_PATH).write_text(
        "\n".join(
            [
                f'milestone: "{milestone}"',
                'milestone_title: "LOCAL SOTA RUNTIME REPAIR + EXACT-VERIFIER PIVOT + GOVERNED FR-11"',
                f'milestone_doc: "{mod.NEXT_MILESTONE_DOC}"',
                "tasks:",
                "  - id: exp5083-phase0-archive-466-activate-467",
                f"    deliverable: {mod.RESULT_RELATIVE_PATH}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_v466_artifacts(root: Path) -> None:
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5071],
        {
            "honest_verdict": "complete_gguf_logprob_preflight_partial_ready",
            "duration_s": 2.003356,
            "inference_substrate": "deterministic_verifier",
            "flagged_adversarial": False,
            "completion_endpoint_ready": False,
            "logprob_endpoint_ready": False,
            "top_logprob_or_confidence_ready": False,
            "live_completion_invoked": False,
            "usable_sota_models": [{"hf_id": "model-a"}, {"hf_id": "model-b"}, {"hf_id": "model-c"}],
        },
    )
    _write_json(
        root / mod.BLOCKED_ARTIFACTS[5072],
        {
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "duration_s": 0.0,
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp5071-gguf-logprob-preflight-v466.logprob_endpoint_ready "
                "(actual=False == expected=True)"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp5071-gguf-logprob-preflight-v466",
                    "artifact_field": "logprob_endpoint_ready",
                    "expected": True,
                    "actual": False,
                    "passed": False,
                }
            ],
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5075],
        {
            "experiment_id": 5075,
            "honest_verdict": "complete_dccd_guided_frontier_no_headline_underpowered",
            "duration_s": 0.278656,
            "flagged_adversarial": False,
            "n_questions": 200,
            "dccd_accuracy": 0.515,
            "unguided_accuracy": 0.515,
            "rerank_only_accuracy": 0.665,
            "delta_dccd_vs_rerank": -0.15,
            "beats_rerank_only": False,
            "ci95_delta": [-0.240844, -0.059156],
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5076],
        {
            "experiment_id": 5076,
            "honest_verdict": "complete_d6_replication_no_pareto_win",
            "duration_s": 0.016732,
            "flagged_adversarial": False,
            "cascade_accuracy": 0.665,
            "judge_only_accuracy": 0.585,
            "delta_vs_judge_only": 0.08,
            "ci95_delta": [-0.005, 0.16],
            "efficiency_win": False,
            "accuracy_headline_allowed": False,
            "judge_call_fraction": 0.0,
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5077],
        {
            "experiment_id": 5077,
            "honest_verdict": "complete_fr11_group_sc_memory_guarded_no_promote_delta_minus_0p050",
            "duration_s": 0.005109,
            "flagged_adversarial": False,
            "fr11_attempt_completed": True,
            "heldout_delta": -0.05,
            "nonforgetting_delta": -0.142857,
            "promoted_count": 0,
            "quarantined_count": 3,
            "promotion_decision": {
                "promoted": False,
                "no_promote_reason": "heldout_delta_negative;nonforgetting_regressed",
            },
            "rollback_guard_passed": True,
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5079],
        {
            "experiment_id": 5079,
            "honest_verdict": "success_board_continuity_matrix_written_no_speedup_claim",
            "duration_s": 6.323637,
            "flagged_adversarial": False,
            "kv260_ssh_ready": True,
            "kv260_speedup_claim_allowed": False,
            "polarfire_detected": True,
            "gatemate_detected": False,
            "gatemate_terminal_state": "blocked_gatemate_usb_undetected",
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5080],
        {
            "experiment": 5080,
            "honest_verdict": "success_kan_pwa_milp_property_verified_tiny",
            "duration_s": 0.348543,
            "flagged_adversarial": False,
            "milp_solver_available": True,
            "pwa_abstraction_built": True,
            "property_holds": True,
            "property_checked": True,
            "binary_variable_count": 3,
            "error_bound": 0.0,
            "solver_status": "optimal",
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5082],
        {
            "experiment": 5082,
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "duration_s": 0.0,
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp5081-phase-d-fr11-decision-gate-v466.decision_ready "
                "(upstream artifact not found for task id 'exp5081-phase-d-fr11-decision-gate-v466')"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp5081-phase-d-fr11-decision-gate-v466",
                    "artifact_field": "decision_ready",
                    "expected": True,
                    "actual": None,
                    "passed": False,
                }
            ],
        },
    )


def test_req_capstone_5083_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-5083: OpenSpec anchors the .466 truth record."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-5083",
        "SCENARIO-CAPSTONE-5083",
        "SCENARIO-CAPSTONE-5083-BLOCKED-YAML",
        "SCENARIO-CAPSTONE-5083-FIELD-PRINCIPLES",
        "experiment_5083_archive_466_activate_467.py",
        "results/experiment_5083_archive_466_activate_467.json",
        "missing live completion/logprob endpoint",
        "KAN tiny-only proof",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_5083_records_transition_truth(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5083: blockers are carried without success laundering."""

    _write_active_roadmap(tmp_path)
    _write_v466_artifacts(tmp_path)

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["duration_s"] >= 0.0001
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "live_llm_inference" not in json.dumps(artifact)
    assert artifact["milestone_from"] == "2026.07.466"
    assert artifact["milestone_to"] == "2026.07.467"
    assert artifact["next_milestone_doc"] == mod.NEXT_MILESTONE_DOC
    assert artifact["docs_updated"] == []
    assert artifact["flagged_adversarial"] is False
    assert artifact["preconditions_checked"]["roadmaps"]["pre_staged"]["status"] == (
        "absent_already_promoted"
    )

    assert len(artifact["source_artifacts"]) == len(mod.SOURCE_ARTIFACTS)
    assert {row["experiment_id"] for row in artifact["blocked_artifacts"]} == {5072, 5082}
    assert {row["experiment_id"] for row in artifact["missing_artifacts"]} == {5073, 5074, 5081}
    assert {row["status"] for row in artifact["missing_artifacts"]} == {"skipped_or_absent"}

    close = artifact["close_state"]
    assert close["scientific_decision_completed"] is False
    assert close["endpoint_state"]["completion_endpoint_ready"] is False
    assert close["endpoint_state"]["logprob_endpoint_ready"] is False
    assert close["endpoint_state"]["top_logprob_or_confidence_ready"] is False
    assert close["uprm_vpr_state"]["uprm_cache_blocked"] is True
    assert close["uprm_vpr_state"]["uprm_process_skipped"] is True
    assert close["uprm_vpr_state"]["vpr_skipped"] is True
    assert close["dccd_state"]["delta_dccd_vs_rerank"] == -0.15
    assert close["dccd_state"]["beats_rerank_only"] is False
    assert close["d6_state"]["efficiency_win"] is False
    assert close["d6_state"]["accuracy_headline_allowed"] is False
    assert close["fr11_state"]["promoted"] is False
    assert close["fr11_state"]["heldout_delta"] == -0.05
    assert close["hardware_state"]["kv260_speedup_claim_allowed"] is False
    assert close["hardware_state"]["gatemate_detected"] is False
    assert close["kan_state"]["tiny_only_proof"] is True
    assert close["capstone_state"]["blocked"] is True

    assert [row["blocker_id"] for row in artifact["blockers_carried_forward"]] == mod.BLOCKER_IDS
    assert all(row["must_not_be_laundered_into_success"] is True for row in artifact["blockers_carried_forward"])
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_5083_blocks_bad_present_roadmap_yaml(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5083-BLOCKED-YAML: unreadable roadmaps fail closed."""

    _write_active_roadmap(tmp_path)
    (tmp_path / mod.ROADMAP_NEXT_REL_PATH).write_text("milestone: [", encoding="utf-8")
    _write_v466_artifacts(tmp_path)

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["honest_verdict"] == "blocked_yaml_parse"
    assert artifact["blockers_carried_forward"] == []
    assert artifact["close_state"] == {}
    assert artifact["blocked_artifacts"] == []
    assert artifact["missing_artifacts"] == []
    assert artifact["flagged_adversarial"] is False
    assert artifact["preconditions_checked"]["roadmaps"]["pre_staged"]["parse_ok"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_5083_resource_edges_and_validation(tmp_path: Path, capsys: Any) -> None:
    """REQ-CAPSTONE-5083: helper edge states and schema failures are explicit."""

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    empty_payload, empty_status = mod._parse_yaml_status(
        tmp_path,
        Path("empty.yaml"),
        absent_status="missing",
    )
    assert empty_payload == {}
    assert empty_status["parse_ok"] is True

    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    assert mod._parse_yaml_status(tmp_path, Path("list.yaml"), absent_status="missing")[1][
        "error"
    ] == "yaml_not_mapping"
    assert mod.roadmap_blocker({"active": {"parse_ok": None}, "pre_staged": {}}) == (
        "blocked_yaml_parse"
    )
    assert mod.roadmap_blocker(
        {"active": {"parse_ok": True, "milestone": "2026.07.999"}, "pre_staged": {}}
    ) == "blocked_active_milestone_mismatch"

    missing_payload, missing_status = mod.read_json_mapping(tmp_path / "missing.json")
    assert missing_payload == {}
    assert missing_status["error"] == "missing"

    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod.read_json_mapping(malformed)[1]["loadable"] is False

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(non_object)[1]["error"] == "json_not_object"

    _write_active_roadmap(tmp_path)
    _write_v466_artifacts(tmp_path)
    (tmp_path / mod.SOURCE_ARTIFACTS[5075]).write_text("{", encoding="utf-8")
    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)
    assert any(row["experiment_id"] == 5075 for row in artifact["missing_artifacts"])

    missing_block_root = tmp_path / "missing-block"
    _write_v466_artifacts(missing_block_root)
    (missing_block_root / mod.BLOCKED_ARTIFACTS[5072]).unlink()
    _, _, missing_rows, _ = mod.load_v466_artifacts(missing_block_root)
    assert any(
        row["experiment_id"] == 5072 and row["status"] == "missing_gate_block_artifact"
        for row in missing_rows
    )

    present_skip_root = tmp_path / "present-skip"
    _write_v466_artifacts(present_skip_root)
    _write_json(
        present_skip_root / mod.SKIPPED_OR_MISSING_ARTIFACTS[5073]["path"],
        {"honest_verdict": "complete_unexpected_present", "duration_s": 0.1},
    )
    source_rows, _, _, payloads = mod.load_v466_artifacts(present_skip_root)
    assert 5073 in payloads
    assert any(row["experiment_id"] == 5073 for row in source_rows)

    assert mod._mapping([]) == {}
    assert mod._list("bad") == []
    assert mod._bool(None) is False
    assert mod._number(True) is None
    assert mod._number("bad") is None

    invalid = dict(artifact)
    invalid["honest_verdict"] = "bad"
    invalid["inference_substrate"] = "live_llm_inference"
    invalid["milestone_from"] = "wrong"
    invalid["milestone_to"] = "wrong"
    invalid["next_milestone_doc"] = "wrong"
    invalid["docs_updated"] = ["ops/status.md"]
    invalid["flagged_adversarial"] = True
    invalid["blockers_carried_forward"] = []
    invalid["close_state"] = {"scientific_decision_completed": True}
    invalid["reproducibility_checksum"] = "bad"
    del invalid["duration_s"]
    errors = mod.artifact_schema_errors(invalid)
    assert "duration_s" in errors
    assert "honest_verdict" in errors
    assert "inference_substrate" in errors
    assert "milestone_from" in errors
    assert "milestone_to" in errors
    assert "next_milestone_doc" in errors
    assert "docs_updated" in errors
    assert "flagged_adversarial" in errors
    assert "blockers_carried_forward" in errors
    assert "close_state" in errors
    assert "reproducibility_checksum" in errors

    assert mod.main(root=tmp_path, artifact_path=tmp_path / "out.json") == 0
    captured = capsys.readouterr()
    assert "experiment_5083_archive_466_activate_467" in captured.out

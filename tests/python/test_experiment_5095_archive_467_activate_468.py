"""Tests for Exp 5095 .467-to-.468 transition recording.

Spec refs: REQ-CAPSTONE-5095, SCENARIO-CAPSTONE-5095,
SCENARIO-CAPSTONE-5095-BLOCKED-YAML,
SCENARIO-CAPSTONE-5095-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5095_archive_467_activate_468 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


class FlatClock:
    """SCENARIO-CAPSTONE-5095 clock keeps duration deterministic."""

    def __call__(self) -> float:
        return 5095.0


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_active_roadmap(root: Path, milestone: str = mod.MILESTONE_TO) -> None:
    (root / mod.ROADMAP_ACTIVE_REL_PATH).write_text(
        "\n".join(
            [
                f'milestone: "{milestone}"',
                'milestone_title: "EXACT-VERIFIER SCALE-UP + EVIDENCE ENERGY + FORMAL FR-11"',
                f'milestone_doc: "{mod.NEXT_MILESTONE_DOC}"',
                "tasks:",
                "  - id: exp5095-phase0-archive-467-activate-468",
                f"    deliverable: {mod.RESULT_RELATIVE_PATH}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_default_upstreams(root: Path) -> None:
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5085],
        {
            "honest_verdict": "success_llamacpp_logprob_endpoint_ready",
            "flagged_adversarial": True,
            "duration_s": 18.88467,
            "completion_endpoint_ready": True,
            "logprob_endpoint_ready": True,
            "top_logprob_or_confidence_ready": True,
            "live_completion_invoked": True,
            "usable_sota_models": [
                {"hf_id": "model-a"},
                {"hf_id": "model-b"},
                {"hf_id": "model-c"},
            ],
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5086],
        {
            "experiment": "experiment_5086_uprm_logprob_cache_retry",
            "experiment_id": 5086,
            "honest_verdict": "blocked_uprm_logprob_cache_retry_endpoint_failed",
            "flagged_adversarial": False,
            "duration_s": 0.289489,
            "logprob_cache_ready": False,
            "step_cache_ready": False,
            "endpoint_used": "http://127.0.0.1:46097/completion",
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5087],
        {
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: "
                "exp5086-uprm-logprob-cache-retry-v467.logprob_cache_ready "
                "(actual=False == expected=True)"
            ),
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5088],
        {
            "experiment": "experiment_5088_temporal_consistency_prm",
            "experiment_id": 5088,
            "honest_verdict": "complete_temporal_consistency_prm_no_win",
            "flagged_adversarial": True,
            "duration_s": 1.102074,
            "beats_one_pass": False,
            "delta_vs_one_pass": 0.0,
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5089],
        {
            "honest_verdict": "complete_pbit_guided_cdcl_distribution_sensitive_no_win",
            "flagged_adversarial": True,
            "duration_s": 0.056744,
            "correctness_preserved": True,
            "helps_declared_family": False,
            "delta_effort_vs_pure": {"pbit_guided": 0, "random_assumption": 3},
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5090],
        {
            "experiment": "experiment_5090_static_csr_constrained_decoding",
            "experiment_id": 5090,
            "honest_verdict": "success_static_csr_masks_speedup_and_validity_win",
            "flagged_adversarial": True,
            "duration_s": 35.628679,
            "beats_cpu_trie": True,
            "beats_rerank_only_on_validity_or_cost": True,
            "mask_equivalence_rate": 1.0,
            "mask_speedup": 77.420013,
            "validity_rate": 1.0,
            "rerank_only_validity_rate": 0.666667,
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5091],
        {
            "experiment": 5091,
            "honest_verdict": "success_kan_pwa_milp_scale_property_verified_small",
            "flagged_adversarial": False,
            "duration_s": 0.368275,
            "property_holds": True,
            "property_status": "verified",
            "abstraction_built": True,
            "solver_available": True,
            "solver_status": "optimal",
            "binary_variable_count": 6,
            "pwa_piece_count": 6,
            "constraint_count": 43,
            "global_error_bound": 0.0,
            "solve_time_s": 0.008194,
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5092],
        {
            "experiment": "experiment_5092_fr11_budgeted_onpolicy_memory",
            "experiment_id": 5092,
            "honest_verdict": "complete_fr11_budgeted_onpolicy_memory_guarded_no_promote_delta_plus_0p000",
            "flagged_adversarial": False,
            "duration_s": 0.002725,
            "fr11_attempt_completed": True,
            "heldout_delta": 0.0,
            "nonforgetting_delta": 0.0,
            "contamination_guard_passed": True,
            "poison_guard_passed": True,
            "rollback_guard_passed": True,
            "promoted_count": 0,
            "promotion_decision": {
                "gate_conditions": {
                    "positive_utility_gt_zero": False,
                    "heldout_delta_gte_zero": True,
                    "nonforgetting_delta_gte_zero": True,
                    "poison_guard_passed": True,
                    "contamination_guard_passed": True,
                    "rollback_guard_passed": True,
                },
                "no_promote_reason": "positive_utility_not_observed",
                "promoted": False,
            },
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5093],
        {
            "experiment": "experiment_5093_hardware_continuity",
            "experiment_id": 5093,
            "honest_verdict": "complete_hardware_continuity_v467_partial_board_blockers",
            "flagged_adversarial": False,
            "duration_s": 8.882491,
            "kv260_ssh_ready": True,
            "kv260_uio_transcript_path": None,
            "kv260_speedup_claim_allowed": False,
            "gatemate_detected": False,
            "gatemate_terminal_state": "blocked_gatemate_dirtyjtag_cable_seen_no_gatemate_idcode_terminal",
            "polarfire_detected": True,
            "polarfire_dispatch_precheck_ready": True,
            "destructive_actions_taken": [],
        },
    )
    _write_json(
        root / mod.SOURCE_ARTIFACTS[5094],
        {
            "experiment": "experiment_5094_capstone_v467",
            "experiment_id": 5094,
            "honest_verdict": "complete_capstone_v467_exact_verifier_pivot_positive_runtime_process_blocked",
            "flagged_adversarial": False,
            "duration_s": 0.001286,
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "milestone_decision": "exact_verifier_pivot_positive",
        },
    )


def test_req_capstone_5095_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-5095: OpenSpec anchors the .467 truth record."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-5095",
        "SCENARIO-CAPSTONE-5095",
        "SCENARIO-CAPSTONE-5095-BLOCKED-YAML",
        "SCENARIO-CAPSTONE-5095-FIELD-PRINCIPLES",
        "experiment_5095_archive_467_activate_468.py",
        "results/experiment_5095_archive_467_activate_468.json",
        "complete_467_archived_468_activated_exact_verifier_pivot_carried_forward",
        "flagged endpoint/live-runtime claim",
        "hardware continuity without speedup",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_5095_records_transition_truth(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5095: .467 blockers and positives are not laundered."""

    _write_active_roadmap(tmp_path)
    _write_default_upstreams(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        clock=FlatClock(),
    )

    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["duration_s"] == 0.0001
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "live_llm_inference" not in json.dumps(artifact)
    assert artifact["milestone_from"] == "2026.07.467"
    assert artifact["milestone_to"] == "2026.07.468"
    assert artifact["next_milestone_doc"] == mod.NEXT_MILESTONE_DOC
    assert artifact["docs_updated"] == []
    assert artifact["flagged_adversarial"] is False
    assert artifact["preconditions_checked"]["roadmaps"]["pre_staged"]["status"] == (
        "absent_already_promoted"
    )

    assert {row["experiment_id"] for row in artifact["source_artifacts"]} == set(
        mod.SOURCE_ARTIFACTS
    )
    assert {row["experiment_id"] for row in artifact["blocked_artifacts"]} == {5086, 5087}
    assert {row["experiment_id"] for row in artifact["flagged_artifacts"]} == {
        5085,
        5088,
        5089,
        5090,
    }
    assert {row["experiment_id"] for row in artifact["clean_positive_artifacts"]} == {5091}
    assert {
        row["path"]
        for row in artifact["missing_artifacts"]
        if row["status"] == "prompt_listed_absent"
    } == {"results/experiment_5088_temporal_consistency_process_verifier_v467.json"}

    close = artifact["close_state"]
    assert close["transition_record_only"] is True
    assert close["capstone_state"]["milestone_decision"] == "exact_verifier_pivot_positive"
    assert close["runtime_state"]["endpoint_live_runtime_claim_flagged"] is True
    assert close["runtime_state"]["reported_completion_endpoint_ready"] is True
    assert close["runtime_state"]["headline_runtime_ready"] is False
    assert close["uprm_cache_state"]["logprob_cache_ready"] is False
    assert close["uprm_cache_state"]["step_cache_ready"] is False
    assert close["process_verifier_state"]["process_verifier_win"] is False
    assert close["process_verifier_state"]["temporal_fallback_reported_win"] is False
    assert close["kan_state"]["clean_positive"] is True
    assert close["kan_state"]["binary_variable_count"] == 6
    assert close["static_csr_state"]["flagged_toy_result"] is True
    assert close["static_csr_state"]["headline_allowed"] is False
    assert close["pbit_cdcl_state"]["effort_win"] is False
    assert close["fr11_state"]["promoted"] is False
    assert close["fr11_state"]["heldout_delta"] == 0.0
    assert close["hardware_state"]["speedup_claim_allowed"] is False
    assert close["hardware_state"]["clean_continuity_without_speedup"] is True

    assert [row["blocker_id"] for row in artifact["blockers_carried_forward"]] == mod.BLOCKER_IDS
    assert all(
        row["must_not_be_laundered_into_success"] is True
        for row in artifact["blockers_carried_forward"]
    )

    pivot = artifact["exact_verifier_pivot"]
    assert pivot["pivot_decision"] == "exact_verifier_pivot_positive"
    assert pivot["driver_experiment_id"] == 5091
    assert pivot["clean_positive"] is True
    assert pivot["scale_boundary"] == "small_multi_unit_property_not_architecture_scale_claim"
    assert pivot["carried_to_milestone"] == "2026.07.468"

    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_5095_blocks_bad_present_roadmap_yaml(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5095-BLOCKED-YAML: unreadable roadmaps fail closed."""

    _write_active_roadmap(tmp_path)
    (tmp_path / mod.ROADMAP_NEXT_REL_PATH).write_text("milestone: [", encoding="utf-8")
    _write_default_upstreams(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        clock=FlatClock(),
    )

    assert artifact["honest_verdict"] == "blocked_yaml_parse"
    assert artifact["source_artifacts"] == []
    assert artifact["blocked_artifacts"] == []
    assert artifact["flagged_artifacts"] == []
    assert artifact["clean_positive_artifacts"] == []
    assert artifact["missing_artifacts"] == []
    assert artifact["close_state"] == {}
    assert artifact["blockers_carried_forward"] == []
    assert artifact["exact_verifier_pivot"] == {}
    assert artifact["flagged_adversarial"] is False
    assert artifact["preconditions_checked"]["roadmaps"]["pre_staged"]["parse_ok"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_5095_resource_edges_and_validation(
    tmp_path: Path, capsys: Any, monkeypatch: Any
) -> None:
    """SCENARIO-CAPSTONE-5095-FIELD-PRINCIPLES: schema drift fails closed."""

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
    assert (
        mod._parse_yaml_status(tmp_path, Path("list.yaml"), absent_status="missing")[1]["error"]
        == "yaml_not_mapping"
    )
    assert mod.roadmap_blocker({"active": {"parse_ok": None}, "pre_staged": {}}) == (
        "blocked_yaml_parse"
    )
    assert (
        mod.roadmap_blocker(
            {"active": {"parse_ok": True, "milestone": "2026.07.999"}, "pre_staged": {}}
        )
        == "blocked_active_milestone_mismatch"
    )

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
    _write_default_upstreams(tmp_path)
    (tmp_path / mod.SOURCE_ARTIFACTS[5091]).write_text("{", encoding="utf-8")
    source_rows, _, missing_rows, _, _, payloads = mod.load_v467_artifacts(tmp_path)
    assert any(row["experiment_id"] == 5091 for row in missing_rows)
    assert 5091 not in payloads
    assert all(row["experiment_id"] != 5091 for row in source_rows)
    _write_default_upstreams(tmp_path)

    prompt_path = tmp_path / mod.PROMPT_LISTED_MISSING_ARTIFACTS[5088]["path"]
    _write_json(prompt_path, {"honest_verdict": "complete_prompt_listed_path_present"})
    source_rows, _, _, _, _, _ = mod.load_v467_artifacts(tmp_path)
    assert any(row.get("status") == "prompt_listed_path_present" for row in source_rows)
    prompt_path.unlink()

    assert mod._mapping([]) == {}
    assert mod._list("bad") == []
    assert mod._bool(None) is False
    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod.build_exact_verifier_pivot({}) == {}

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        clock=FlatClock(),
    )
    invalid = json.loads(json.dumps(artifact))
    invalid["honest_verdict"] = "bad"
    invalid["inference_substrate"] = "wrong"
    invalid["milestone_from"] = "wrong"
    invalid["milestone_to"] = "wrong"
    invalid["next_milestone_doc"] = "wrong"
    invalid["docs_updated"] = ["ops/status.md"]
    invalid["flagged_adversarial"] = True
    invalid["blockers_carried_forward"] = []
    invalid["exact_verifier_pivot"] = {"clean_positive": False}
    invalid["close_state"]["capstone_state"]["milestone_decision"] = "bounded_no_headline"
    invalid["reproducibility_checksum"] = "bad"
    invalid["note"] = "live_llm_inference"
    del invalid["duration_s"]
    errors = mod.artifact_schema_errors(invalid)
    assert "missing.duration_s" in errors
    assert "honest_verdict.not_terminal" in errors
    assert "inference_substrate.not_aggregation" in errors
    assert "milestone_from.invalid" in errors
    assert "milestone_to.invalid" in errors
    assert "next_milestone_doc.invalid" in errors
    assert "docs_updated.not_deferred" in errors
    assert "flagged_adversarial.must_be_false" in errors
    assert "forbidden.live_llm_inference_claim" in errors
    assert "blockers_carried_forward.invalid" in errors
    assert "exact_verifier_pivot.invalid" in errors
    assert "close_state.invalid" in errors
    assert "reproducibility_checksum.invalid" in errors

    assert mod.main(root=tmp_path, artifact_path=tmp_path / "out.json", clock=FlatClock()) == 0
    captured = capsys.readouterr()
    assert "experiment_5095_archive_467_activate_468" in captured.out

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced_schema_error"])
    assert mod.main(root=tmp_path, artifact_path=tmp_path / "bad-out.json", clock=FlatClock()) == 1
    captured = capsys.readouterr()
    assert "forced_schema_error" in captured.out

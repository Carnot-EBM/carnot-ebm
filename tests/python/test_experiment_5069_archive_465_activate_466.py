"""Tests for Exp 5069 .465-to-.466 transition recording.

Spec refs: REQ-CAPSTONE-5069, SCENARIO-CAPSTONE-5069,
SCENARIO-CAPSTONE-5069-BLOCKED-YAML,
SCENARIO-CAPSTONE-5069-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5069_archive_465_activate_466 as mod


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
                'milestone_title: "PHASE D FINAL PROCESS-VERIFIER CHECK + GUARDED FR-11 MEMORY EVOLUTION"',
                f'milestone_doc: "{mod.NEXT_MILESTONE_DOC}"',
                "tasks:",
                "  - id: exp5069-phase0-archive-465-activate-466",
                f"    deliverable: {mod.RESULT_RELATIVE_PATH}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _capstone() -> JsonDict:
    return {
        "schema": "carnot.experiment_5068_capstone_v465.v1",
        "experiment": "experiment_5068_capstone_v465",
        "experiment_id": 5068,
        "milestone": mod.MILESTONE_FROM,
        "honest_verdict": (
            "complete_capstone_v465_execution_incomplete_"
            "fr11_no_credible_positive_evidence_missing_sota"
        ),
        "capstone_ready": False,
        "moat_state": "execution_incomplete",
        "best_verifier_evidence": {
            "best_arm": "D6",
            "best_arm_delta": 0.08,
            "best_arm_ci": [0.0, 0.165],
            "headline_countable": False,
            "cascade_efficiency_win": True,
            "second_corpus_audit_clean": False,
            "guided_decoding_frontier_state": "guided_gain_observed_plus_0p111",
            "execution_incomplete_reasons": [
                "D1 flagged: complete_d1_sota_refresh_audit_no_proper_win_plus_0p080",
                "D4 audit not clean: second_corpus_audit_clean=false",
                "D6 efficiency observed but accuracy CI does not exclude zero",
            ],
            "source_row": {
                "artifact_id": "D6",
                "efficiency_win": True,
                "ci95": [0.0, 0.165],
            },
        },
        "fr11_self_learning_result": {
            "state": "no_credible_positive_evidence",
            "honest_verdict": "complete_guarded_no_promote_minus_0p050",
            "promoted": False,
            "heldout_delta": -0.05,
            "no_promote_reason": "heldout_delta_nonpositive;nonforgetting_regressed",
        },
        "hardware_result": {
            "state": "packet_built",
            "claim_boundary": "local_kv260_transcript_backed_parity_timing_only",
            "honest_verdict": "success_kv260_testbench_timing_packet_built",
            "timing_ratio_packet_built": True,
            "no_general_speedup_claim": True,
        },
        "sota_result": {
            "state": "missing",
            "claim_boundary": "missing_artifact_no_sota_ingestion_claim",
            "honest_verdict": "",
        },
        "arc_result": {
            "state": "no_bank",
            "honest_verdict": "complete_re86_no_new_level_residual_duplicate_depth",
            "new_levels_banked": 0,
        },
        "missing_upstream_artifacts": [
            {
                "source": "sota",
                "experiment_id": 5066,
                "path": "results/experiment_5066_sota_ingestion_v466.json",
            }
        ],
        "next_milestone_pointer": {
            "selected_state": "execution_incomplete",
            "selected": {
                "experiment_class": "execution_repair_before_claim_or_retirement",
                "blocked_dependency": "exp5066_sota_ingestion_missing",
            },
            "sota_ingestion_missing": True,
        },
    }


def _phase_payload(exp_id: int, verdict: str, **extra: Any) -> JsonDict:
    return {
        "experiment": f"experiment_{exp_id}",
        "experiment_id": exp_id,
        "honest_verdict": verdict,
        "duration_s": 1.0,
        **extra,
    }


def _write_phase_artifacts(root: Path) -> None:
    payloads = {
        5057: _phase_payload(5057, "complete_gate_state_preflight_partial_ready", flagged_adversarial=True),
        5058: _phase_payload(5058, "complete_sota_candidate_refresh_ready_d1_d6"),
        5059: _phase_payload(
            5059,
            "complete_d1_sota_refresh_audit_no_proper_win_plus_0p080",
            flagged_adversarial=True,
        ),
        5060: _phase_payload(
            5060,
            "retired_d4_second_corpus_audit_failed_constraintbench_exact_v1_plus_0p370",
            second_corpus_audit_clean=False,
        ),
        5061: _phase_payload(
            5061,
            "success_tool_first_cascade_parity_at_0pct_judge_calls",
            efficiency_win=True,
        ),
        5062: _phase_payload(
            5062,
            "complete_guided_decoding_cost_frontier_guided_gain_plus_0p111",
            sample_power="underpowered",
        ),
        5063: _phase_payload(
            5063,
            "complete_moat_execution_incomplete_v465_blocked_flagged_or_unclean",
            moat_state="execution_incomplete",
        ),
        5064: _phase_payload(
            5064,
            "complete_guarded_no_promote_minus_0p050",
            promoted=False,
        ),
        5065: _phase_payload(
            5065,
            "success_kv260_testbench_timing_packet_built",
            timing_ratio_packet_built=True,
        ),
        5067: _phase_payload(
            5067,
            "complete_re86_no_new_level_residual_duplicate_depth",
            new_levels_banked=0,
        ),
    }
    for exp_id, payload in payloads.items():
        _write_json(root / mod.PHASE_ARTIFACTS[exp_id], payload)


def test_req_capstone_5069_spec_declares_transition_contract() -> None:
    """REQ-CAPSTONE-5069: OpenSpec anchors the transition artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-5069",
        "SCENARIO-CAPSTONE-5069",
        "SCENARIO-CAPSTONE-5069-BLOCKED-YAML",
        "SCENARIO-CAPSTONE-5069-FIELD-PRINCIPLES",
        "experiment_5069_archive_465_activate_466.py",
        "results/experiment_5069_archive_465_activate_466.json",
        "D1 bounded/no proper win",
        "Exp5066 missing due unavailable Gemini routing",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_5069_records_transition_truth(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5069: blockers are carried without success laundering."""

    _write_active_roadmap(tmp_path)
    _write_json(tmp_path / mod.CAPSTONE_REL_PATH, _capstone())
    _write_phase_artifacts(tmp_path)

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["honest_verdict"] == (
        "complete_465_archived_466_activated_execution_incomplete_carried_forward"
    )
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "live_llm_inference" not in json.dumps(artifact)
    assert artifact["milestone_from"] == mod.MILESTONE_FROM
    assert artifact["milestone_to"] == mod.MILESTONE_TO
    assert artifact["source_capstone_path"] == str(mod.CAPSTONE_REL_PATH)
    assert artifact["next_milestone_doc"] == mod.NEXT_MILESTONE_DOC
    assert artifact["docs_updated"] == []
    assert artifact["flagged_adversarial"] is False
    assert artifact["preconditions_checked"]["roadmaps"]["pre_staged"]["status"] == (
        "absent_already_promoted"
    )
    assert artifact["close_state"]["capstone_ready"] is False
    assert artifact["close_state"]["moat_state"] == "execution_incomplete"
    assert artifact["close_state"]["sota_result"]["state"] == "missing"
    assert artifact["close_state"]["next_milestone_pointer"]["sota_ingestion_missing"] is True
    assert [row["blocker_id"] for row in artifact["blockers_carried_forward"]] == [
        "d1_bounded_no_proper_win",
        "d4_duplicate_audit_retirement",
        "d6_efficiency_only",
        "guided_decoding_underpowered",
        "fr11_guarded_no_promote",
        "kv260_parity_packet_only",
        "exp5066_missing_unavailable_gemini_routing",
    ]
    assert all(row["must_not_be_laundered_into_success"] is True for row in artifact["blockers_carried_forward"])
    assert artifact["phase_artifacts_loaded"]["results/experiment_5066_sota_ingestion_v466.json"][
        "present"
    ] is False
    assert artifact["phase_artifacts_loaded"]["results/experiment_5059_d1_sota_refresh_audit.json"][
        "flagged_adversarial"
    ] is True
    assert {row["path"] for row in artifact["cited_upstream_artifacts"]} >= {
        str(mod.ROADMAP_ACTIVE_REL_PATH),
        str(mod.CAPSTONE_REL_PATH),
    }
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_5069_blocks_bad_present_roadmap_yaml(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5069-BLOCKED-YAML: unreadable roadmaps fail closed."""

    _write_active_roadmap(tmp_path)
    (tmp_path / mod.ROADMAP_NEXT_REL_PATH).write_text("milestone: [", encoding="utf-8")
    _write_json(tmp_path / mod.CAPSTONE_REL_PATH, _capstone())

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["honest_verdict"] == "blocked_yaml_parse"
    assert artifact["blockers_carried_forward"] == []
    assert artifact["close_state"] == {}
    assert artifact["flagged_adversarial"] is False
    assert artifact["preconditions_checked"]["roadmaps"]["pre_staged"]["parse_ok"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_5069_resource_edges_and_validation(tmp_path: Path, capsys: Any) -> None:
    """REQ-CAPSTONE-5069: helper edge states and schema failures are explicit."""

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
        "blocked_missing_active_roadmap"
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
    _write_json(tmp_path / mod.CAPSTONE_REL_PATH, _capstone())
    (tmp_path / mod.PHASE_ARTIFACTS[5057]).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.PHASE_ARTIFACTS[5057]).write_text("{", encoding="utf-8")
    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)
    assert artifact["phase_artifacts_loaded"][str(mod.PHASE_ARTIFACTS[5057])]["loadable"] is False

    capstone_status = artifact["preconditions_checked"]["source_capstone"]
    assert capstone_status["loadable"] is True
    assert capstone_status["sha256"].startswith("sha256:")
    assert mod._mapping([]) == {}
    assert mod._list("bad") == []
    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod._int("bad") == 0

    invalid = dict(artifact)
    invalid["honest_verdict"] = "bad"
    invalid["inference_substrate"] = "live_llm_inference"
    invalid["milestone_from"] = "wrong"
    invalid["milestone_to"] = "wrong"
    invalid["source_capstone_path"] = "wrong"
    invalid["next_milestone_doc"] = "wrong"
    invalid["docs_updated"] = ["ops/status.md"]
    invalid["flagged_adversarial"] = True
    invalid["blockers_carried_forward"] = []
    invalid["close_state"] = {}
    invalid["reproducibility_checksum"] = "bad"
    errors = mod.artifact_schema_errors(invalid)
    for error in (
        "honest_verdict",
        "inference_substrate",
        "milestone_from",
        "milestone_to",
        "source_capstone_path",
        "next_milestone_doc",
        "docs_updated",
        "flagged_adversarial",
        "blockers_carried_forward",
        "close_state",
        "reproducibility_checksum",
    ):
        assert error in errors
    assert "schema" in mod.artifact_schema_errors({})

    blocked_missing = mod.run(root=tmp_path / "empty", artifact_path=tmp_path / "blocked.json")
    assert blocked_missing["honest_verdict"] == "blocked_missing_active_roadmap"

    missing_capstone_root = tmp_path / "missing_capstone"
    missing_capstone_root.mkdir()
    _write_active_roadmap(missing_capstone_root)
    missing_capstone = mod.run(
        root=missing_capstone_root,
        artifact_path=tmp_path / "blocked_missing_capstone.json",
    )
    assert missing_capstone["honest_verdict"] == "blocked_missing_source_capstone"

    bad_capstone_root = tmp_path / "bad_capstone"
    bad_capstone_root.mkdir()
    _write_active_roadmap(bad_capstone_root)
    (bad_capstone_root / mod.CAPSTONE_REL_PATH).parent.mkdir(parents=True, exist_ok=True)
    (bad_capstone_root / mod.CAPSTONE_REL_PATH).write_text("{", encoding="utf-8")
    bad_capstone = mod.run(
        root=bad_capstone_root,
        artifact_path=tmp_path / "blocked_bad_capstone.json",
    )
    assert bad_capstone["honest_verdict"] == "blocked_unloadable_source_capstone"

    exit_code = mod.main(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert exit_code == 0
    assert printed["experiment"] == mod.EXPERIMENT
    assert printed["reproducibility_checksum"] == mod.payload_checksum(printed)

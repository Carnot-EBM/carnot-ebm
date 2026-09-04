"""Focused tests for REQ-REPORT-6983 and its V611 capstone scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_6983_v611_capstone as mod


REPO_ROOT = Path(__file__).resolve().parents[2]


def _payload(
    verdict_class: str = "positive",
    honest_verdict: str = "complete_positive_fixture",
    **updates: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "fixture.v1",
        "status": "complete",
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "field_principles": {"rows": "Rows support independent arithmetic."},
        "rows": [{"terminal": True}],
    }
    payload.update(updates)
    return payload


def _copy_capstone_inputs(target: Path) -> None:
    """Copy only immutable inputs so integration tests never write the repository."""

    for relative in (mod.DESIGN_PATH, mod.ROADMAP_PATH, mod.EXCLUSION_PATH):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    for task in mod.EXPECTED_TASKS[:-1]:
        relative = Path(str(task["deliverable"]))
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)


def test_req_report_6983_spec_precedes_implementation() -> None:
    """REQ-REPORT-6983 names every field and all required scenarios."""

    text = (REPO_ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-REPORT-6983", 1)[1]
    assert {
        "SCENARIO-REPORT-6983-CONTRACT",
        "SCENARIO-REPORT-6983-ABSENCE",
        "SCENARIO-REPORT-6983-EXCLUSIONS",
        "SCENARIO-REPORT-6983-VERDICTS",
        "SCENARIO-REPORT-6983-ROWS",
        "SCENARIO-REPORT-6983-BLOCKED",
        "SCENARIO-REPORT-6983-HANDOFF",
        "SCENARIO-REPORT-6983-ARTIFACT",
    } <= set(mod.spec_anchors(section))
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_6983_contract_matches_both_primary_sources() -> None:
    """SCENARIO-REPORT-6983-CONTRACT checks exact parity without Exp6972."""

    design = (REPO_ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = mod.load_yaml(REPO_ROOT / mod.ROADMAP_PATH)
    document_rows = mod.parse_document_contract(design)
    yaml_rows = mod.parse_yaml_contract(roadmap)
    gate_rows = mod.build_gate_contract_rows(document_rows, yaml_rows)

    assert [row["task_id"] for row in document_rows] == mod.EXPECTED_TASK_IDS
    assert [row["task_id"] for row in yaml_rows] == mod.EXPECTED_TASK_IDS
    assert len(gate_rows) == 13
    assert all(row["passed"] for row in gate_rows)
    assert mod.contract_conforms(document_rows, yaml_rows, gate_rows)

    changed_yaml = deepcopy(roadmap)
    changed_yaml["tasks"][3]["title"] = "Changed title"
    changed_yaml["tasks"][4]["gated_on"][0]["artifact_field"] = "misspelled_field"
    changed_rows = mod.parse_yaml_contract(changed_yaml)
    changed_gates = mod.build_gate_contract_rows(document_rows, changed_rows)
    assert not mod.contract_conforms(document_rows, changed_rows, changed_gates)
    assert any(row["broken_contract"] for row in changed_gates)


@pytest.mark.parametrize(
    ("payload", "live_flags", "branch", "verdict_class"),
    [
        (None, [], "absent", "blocked"),
        (
            {
                "status": "blocked",
                "honest_verdict": "blocked_gate_check_failed",
                "blocked_at_layer": "conductor_pre_gate",
                "gates_evaluated": [],
            },
            [],
            "pre_gate_blocked",
            "blocked",
        ),
        (
            _payload(
                "blocked",
                "blocked_local_fixture",
                gate_check_summary={
                    "failed_check": "fixture",
                    "expected_value": True,
                    "observed_value": False,
                },
            ),
            [],
            "artifact_blocked",
            "blocked",
        ),
        (_payload("null", "complete_null_fixture"), [], "null", "null"),
        (_payload(), [], "positive", "positive"),
        (
            _payload(verifier_is_oracle=True),
            [],
            "circular_positive",
            "circular_positive",
        ),
        (
            _payload(flagged_adversarial=True, corrigendum_pending=[{"kind": "TAUTOLOGY"}]),
            [],
            "disqualified",
            "disqualified",
        ),
        (_payload(), [{"kind": "LIVE_FLAG", "severity": "critical"}], "disqualified", "disqualified"),
        (
            _payload("unknown", "unfinished_fixture"),
            [],
            "partial",
            "partial",
        ),
    ],
)
def test_scenario_report_6983_verdict_and_branch_propagation(
    payload: dict[str, object] | None,
    live_flags: list[dict[str, object]],
    branch: str,
    verdict_class: str,
) -> None:
    """SCENARIO-REPORT-6983-VERDICTS preserves each evidence class."""

    row = mod.classify_task(mod.EXPECTED_TASKS[4], payload, live_flags)
    assert row["branch_status"] == branch
    assert row["verdict_class"] == verdict_class
    assert row["terminal"] is True


def test_scenario_report_6983_flagged_exclusion_names_exact_flags() -> None:
    """SCENARIO-REPORT-6983-EXCLUSIONS excludes stamped and live flag kinds."""

    payload = _payload(
        flagged_adversarial=True,
        corrigendum_pending=[
            {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
            {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
        ],
    )
    live = [{"kind": "CURRENT_FLAG", "severity": "critical"}]
    row = mod.classify_task(mod.EXPECTED_TASKS[0], payload, live)
    exclusions = mod.build_flagged_exclusion_rows([row])

    assert exclusions[0]["stamped_flag_kinds"] == [
        "DURATION_TOO_SHORT",
        "METHODOLOGY_MISSING",
    ]
    assert exclusions[0]["live_flag_kinds"] == ["CURRENT_FLAG"]
    assert exclusions[0]["excluded"] is True


def test_scenario_report_6983_runtime_recomputes_from_family_rows() -> None:
    """SCENARIO-REPORT-6983-ROWS ignores a forged runtime headline."""

    payload = {
        "lease_aware_runtime_ready_score": 0,
        "live_generation_rows": [
            {
                "model_id": model,
                "terminal_state": "complete",
                "completion_tokens": 8,
                "live_cuda": True,
                "teardown_complete": True,
                "vram_release_passed": True,
                "process_owned": True,
                "owned_process_absent": True,
            }
            for model in mod.REQUIRED_MODELS
        ],
    }
    result = mod.recompute_runtime(payload)
    assert result["family_count"] == 3
    assert result["completed_family_count"] == 3
    assert result["recomputed_ready_score"] == 1
    assert result["reported_ready_score"] == 0
    assert result["reported_matches"] is False


def test_scenario_report_6983_constraint_metrics_and_headroom_use_rows() -> None:
    """SCENARIO-REPORT-6983-ROWS recomputes exact mapping and headroom rows."""

    bank = {"rows": [{"terminal": True}, {"terminal": True}]}
    candidates = [
        {
            "split": "calibration",
            "pair_id": "a",
            "hf_id": "m",
            "schedule_id": "direct",
            "terminal": True,
            "parse_success": True,
            "schema_outcome": "pass",
            "domain_correspondence_outcome": "pass",
            "objective_direction_outcome": "pass",
            "objective_order_outcome": "pass",
            "optimum_outcome": "pass",
            "exact_semantic_success": False,
            "authorities_agree": True,
        },
        {
            "split": "heldout",
            "pair_id": "b",
            "hf_id": "m",
            "schedule_id": "direct",
            "terminal": True,
            "parse_success": True,
            "schema_outcome": "pass",
            "domain_correspondence_outcome": "pass",
            "objective_direction_outcome": "pass",
            "objective_order_outcome": "pass",
            "optimum_outcome": "pass",
            "exact_semantic_success": True,
            "authorities_agree": True,
        },
    ]
    groups = [
        {"terminal": True, "has_heldout_headroom": True, "within_group_exact_headroom": 1},
        {"terminal": True, "has_heldout_headroom": False, "within_group_exact_headroom": 0},
    ]
    result = mod.recompute_constraints(
        bank,
        {"per_candidate_rows": candidates, "per_group_results": groups},
    )
    assert result["bank_candidate_count"] == 2
    assert result["certified_candidate_count"] == 2
    assert result["parse_success_count"] == 2
    assert result["exact_semantic_success_count"] == 1
    assert result["heldout_headroom_group_count"] == 1
    assert result["calibration_label_values"] == [False]


def test_scenario_report_6983_pwa_certificate_state_uses_upstream_labels() -> None:
    """SCENARIO-REPORT-6983-ROWS blocks PWA readiness on constant labels."""

    pwa = {
        "feature_rows": [{"split": "calibration"}],
        "pwa_unit_rows": [],
        "milp_query_rows": [],
        "invariant_certificate_rows": [],
        "heldout_comparison_rows": [],
        "certified_pwa_energy_ready_score": 1,
    }
    certification = {
        "per_candidate_rows": [
            {"split": "calibration", "exact_semantic_success": False},
            {"split": "calibration", "exact_semantic_success": False},
        ]
    }
    result = mod.recompute_pwa(pwa, certification)
    assert result["calibration_labels_nonconstant"] is False
    assert result["recomputed_ready_score"] == 0
    assert result["reported_matches"] is False


def test_scenario_report_6983_learning_recomputes_gain_and_forgetting() -> None:
    """SCENARIO-REPORT-6983-ROWS recomputes each learning arm and held future."""

    rows = []
    for arm, successes in (("frozen", 1), ("read_only", 1), ("transactional_write", 2)):
        rows.extend(
            {"arm": arm, "event_id": f"e{i}", "terminal": True, "exact_success": i < successes}
            for i in range(3)
        )
    payload = {
        "rows": rows,
        "held_future_rows": [{"paired_delta": 1}, {"paired_delta": -1}],
        "rollback_rows": [{"passed": True}],
        "chronological_gain_over_readonly": 99,
        "max_forgetting": 99,
    }
    result = mod.recompute_self_learning(payload)
    assert result["success_count_by_arm"] == {
        "frozen": 1,
        "read_only": 1,
        "transactional_write": 2,
    }
    assert result["chronological_gain_over_readonly"] == 1
    assert result["max_forgetting"] == 1
    assert result["positive_score"] == 0


def test_scenario_report_6983_spilled_energy_recomputes_auc_and_retirement() -> None:
    """SCENARIO-REPORT-6983-ROWS computes signal metrics from eligible spans."""

    payload = {
        "per_span_results": [
            {
                "split": "heldout",
                "eligible": True,
                "terminal": True,
                "error_label": 0,
                "signals": {"spilled_energy": 0.1, "entropy": 0.4},
            },
            {
                "split": "heldout",
                "eligible": True,
                "terminal": True,
                "error_label": 1,
                "signals": {"spilled_energy": 0.9, "entropy": 0.4},
            },
            {"split": "heldout", "eligible": False, "terminal": True, "error_label": 1},
        ],
        "retirement_recommendation": {"retire": True, "propose_retry": False},
        "spilled_energy_requalified_score": 1,
    }
    result = mod.recompute_spilled_energy(payload)
    assert result["eligible_heldout_count"] == 2
    assert result["signal_metrics"]["spilled_energy"]["auroc"] == 1.0
    assert result["signal_metrics"]["entropy"]["auroc"] == 0.5
    assert result["recomputed_requalified_score"] == 0
    assert result["retired"] is True


def test_scenario_report_6983_arc_and_hybrid_recomputation() -> None:
    """SCENARIO-REPORT-6983-ROWS requires ARC rows and hybrid group evidence."""

    arc = mod.recompute_arc(
        {
            "per_transition_rows": [
                {"exact": True, "changing": True},
                {"exact": False, "changing": False},
            ],
            "paired_control_delta_rows": [{"delta": 0.2}],
            "live_path_trace_rows": [{"reachable": True}],
        }
    )
    assert arc["transition_count"] == 2
    assert arc["heldout_exact_accuracy"] == 0.5
    assert arc["strongest_control_delta"] == 0.2
    assert arc["live_path_reachable_score"] == 1

    hybrid = mod.recompute_hybrid_selection(
        {
            "per_group_results": [
                {
                    "terminal": True,
                    "likelihood_success": False,
                    "hybrid_success": True,
                    "exact_filter_used": True,
                },
                {
                    "terminal": True,
                    "likelihood_success": True,
                    "hybrid_success": True,
                    "exact_filter_used": True,
                },
            ]
        }
    )
    assert hybrid["group_count"] == 2
    assert hybrid["hybrid_gain"] == 1
    assert hybrid["verdict_class"] == "circular_positive"
    assert mod.recompute_hybrid_selection({})["available"] is False


def test_scenario_report_6983_blocked_causes_normalize_artifact_and_pre_gate() -> None:
    """SCENARIO-REPORT-6983-BLOCKED keeps exact local and upstream values."""

    local = _payload(
        "blocked",
        "blocked_local",
        gate_check_summary={
            "failed_checks": [
                {"failed_check": "labels_nonconstant", "expected_value": True, "observed_value": False}
            ]
        },
    )
    local_rows = mod.blocked_causes(mod.EXPECTED_TASKS[5], local)
    assert local_rows[0]["failed_check"] == "labels_nonconstant"
    assert local_rows[0]["expected_value"] is True
    assert local_rows[0]["observed_value"] is False

    pre_gate = {
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "blocked_at_layer": "conductor_pre_gate",
        "gates_evaluated": [
            {
                "upstream": "exp6977-certified-pwa-kan-energy",
                "artifact_field": "certified_pwa_energy_ready_score",
                "expected": 1,
                "actual": 0,
                "passed": False,
            }
        ],
    }
    row = mod.blocked_causes(mod.EXPECTED_TASKS[10], pre_gate)[0]
    assert row["failed_check"].endswith("certified_pwa_energy_ready_score")
    assert row["expected_value"] == 1
    assert row["observed_value"] == 0


def test_scenario_report_6983_builds_actual_terminal_capstone() -> None:
    """SCENARIO-REPORT-6983-ARTIFACT validates the checked-in V611 evidence."""

    artifact = mod.build_artifact(REPO_ROOT, "20260904", run_commands=False)
    assert mod.validate_artifact(artifact) == []
    assert artifact["expected_task_ids"] == mod.EXPECTED_TASK_IDS
    assert artifact["observed_task_ids"] == mod.EXPECTED_TASK_IDS
    assert artifact["v611_capstone_complete_score"] == 1
    assert artifact["v611_task_contract_conforms_score"] == 1
    assert artifact["v611_science_positive_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert artifact["pwa_certificate_recomputation"]["recomputed_ready_score"] == 0
    assert artifact["self_learning_recomputation"]["chronological_gain_over_readonly"] == 0
    assert artifact["spilled_energy_recomputation"]["retired"] is True
    assert artifact["v612_handoff_rows"][0]["boundary_task_id"].startswith("exp6977-")
    excluded = {row["number"]: row for row in artifact["flagged_exclusion_rows"]}
    assert excluded[6972]["stamped_flag_kinds"] == [
        "DURATION_TOO_SHORT",
        "METHODOLOGY_MISSING",
    ]
    assert excluded[6979]["stamped_flag_kinds"] == ["DURATION_TOO_SHORT"]
    branches = {row["number"]: row["branch_status"] for row in artifact["branch_status_rows"]}
    assert branches[6977] == "artifact_blocked"
    assert branches[6982] == "pre_gate_blocked"


def test_scenario_report_6983_absent_artifact_is_terminal_branch_evidence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6983-ABSENCE records absence once without partial retry."""

    _copy_capstone_inputs(tmp_path)
    (tmp_path / mod.EXPECTED_TASKS[8]["deliverable"]).unlink()
    artifact = mod.build_artifact(tmp_path, "20260904", run_commands=False)
    row = next(row for row in artifact["branch_status_rows"] if row["number"] == 6980)
    assert row["branch_status"] == "absent"
    assert artifact["v611_capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert "partial" not in artifact["honest_verdict"]


def test_scenario_report_6983_only_missing_contract_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6983-ABSENCE blocks only on a missing contract file."""

    _copy_capstone_inputs(tmp_path)
    (tmp_path / mod.DESIGN_PATH).unlink()
    artifact = mod.build_artifact(tmp_path, "20260904", run_commands=False)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] == "contract_preconditions"
    assert artifact["gate_check_summary"]["expected_value"] == "both contract files present"
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_6983_validator_rejects_forged_terminal_fields() -> None:
    """SCENARIO-REPORT-6983-ARTIFACT rejects score and checksum changes."""

    artifact = mod.build_artifact(REPO_ROOT, "20260904", run_commands=False)
    artifact["v611_science_positive_score"] = 1
    assert "v611_science_positive_score does not match eligible rows" in mod.validate_artifact(
        artifact
    )
    artifact["v611_science_positive_score"] = 0
    artifact["field_principles"].pop("rows")
    assert "field_principles missing rows" in mod.validate_artifact(artifact)
    artifact["field_principles"]["rows"] = "restored"
    artifact["reproducibility_checksum"] = "sha256:forged"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(artifact)


def test_req_report_6983_wrapper_writes_only_requested_output(tmp_path: Path) -> None:
    """REQ-REPORT-6983 exposes a strict dated CLI and temporary output path."""

    _copy_capstone_inputs(tmp_path)
    output = tmp_path / "result.json"
    assert mod.main(
        [
            "--repo-root",
            str(tmp_path),
            "--date",
            "20260904",
            "--output",
            str(output),
            "--no-commands",
        ]
    ) == 0
    assert mod.validate_artifact(json.loads(output.read_text(encoding="utf-8"))) == []
    assert mod.main(["--date", "2026-09-04", "--output", str(output)]) == 2


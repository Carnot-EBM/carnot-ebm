"""Tests for Exp 5244 V479 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5244, SCENARIO-CAPSTONE-5244,
SCENARIO-CAPSTONE-5244-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import carnot.experiment_5244_capstone_v479 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _base(experiment: str | int, verdict: str, *, flagged: bool | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": experiment,
        "honest_verdict": verdict,
        "duration_s": 1.0,
        "inference_substrate": "fixture",
    }
    if flagged is not None:
        payload["flagged_adversarial"] = flagged
    return payload


def _payloads() -> dict[int, dict[str, Any]]:
    return {
        5233: {
            **_base(
                "experiment_5233_archive_478_activate_479",
                "complete: .478 archived and .479 activated.",
            ),
            "research_roadmap_yaml_activated": _wrap(True),
        },
        5234: _base(
            "experiment_5234_sota_ingestion_v479",
            "complete: V479 SOTA execution refresh found no new actionable findings.",
        ),
        5235: {
            **_base(
                "experiment_5235_adversarial_qa_null_tautology_calibration_v479",
                "complete: QA calibration passed for structured nulls.",
                flagged=True,
            ),
            "qa_calibration_passed": True,
            "gap4_reclassification_ready": True,
            "duration_methodology_checks_preserved": True,
            "corrigendum_pending": [{"severity": "critical", "kind": "DURATION_TOO_SHORT"}],
        },
        5236: {
            **_base(
                "experiment_5236_gap4_clean_status_after_qa_calibration_v479",
                "complete: GAP-4 is still blocked after QA calibration.",
                flagged=True,
            ),
            "gap4_status_decision": "blocked_missing_receipts",
            "gap4_headline_eligible": False,
            "remaining_blocker": "artifact_schema_errors: reproducibility_checksum",
            "qa_calibration_passed": True,
            "wins": 0,
            "losses": 0,
            "ties": 120,
            "corrigendum_pending": [{"severity": "critical", "kind": "TAUTOLOGY"}],
        },
        5237: {
            **_base(
                "experiment_5237_gap1_stability_freeze_or_retire_v479",
                "complete: GAP-1 blocked_instability.",
            ),
            "gap1_registry_promoted": _wrap(False),
            "gap1_stability_decision": _wrap("blocked_instability"),
            "stability_audit": {"exact_subset_stability_passed": False},
        },
        5238: {
            **_base(
                "experiment_5238_veribmc_methodology_correct_rerun_or_retire_v479",
                "complete: solver feedback stayed null; retired current VerIbmc path.",
            ),
            "methodology_receipts_complete": _wrap(True),
            "retire_current_veribmc_path": _wrap(True),
            "solver_feedback_uplift": _wrap(0.0),
            "solver_only_solved": _wrap(1),
            "llm_only_solved": _wrap(2),
            "llm_solver_feedback_solved": _wrap(2),
        },
        5239: {
            **_base(
                "experiment_5239_continuous_self_learning_controlled_memory_ablation_v479",
                "complete: typed memory shows controlled useful reuse.",
            ),
            "continuous_self_learning_task": _wrap(True),
            "aligned_vs_shuffled_delta": _wrap(1.0),
            "aligned_vs_no_memory_delta": _wrap(0.666667),
            "degradation_detected": _wrap(True),
            "retention_check_passed": _wrap(True),
            "rollback_policy_exercised": _wrap(True),
            "broad_self_distillation_used": _wrap(False),
        },
        5240: {
            **_base(
                "experiment_5240_arc_rubric_to_patch_synthesis_v479",
                "success: provenance-routing live patch is gated for exp5241 without a solve claim.",
            ),
            "recommended_live_patch_available": True,
            "patch_test_ready": True,
            "arc_level_solve_claimed": False,
            "registry_precheck_done": True,
        },
        5241: {
            **_base(
                "experiment_5241_arc_gated_live_patch_attempt_v479",
                "complete: level_delta=0 provenance=live_agent_self_discovery.",
                flagged=True,
            ),
            "preconditions_checked": True,
            "solve_provenance": "live_agent_self_discovery",
            "reproducible_total_levels_delta": 0,
            "solve_claim": {"claimed": False, "residual": "no_level_banked"},
            "arc_validation_commands": [{"command": "arc_artifact_lint", "passed": False}],
            "corrigendum_pending": [{"severity": "critical", "kind": "DURATION_TOO_SHORT"}],
        },
        5242: {
            **_base(
                "experiment_5242_kan_certificate_abstraction_scale_v479",
                "success: bounded KAEM certificate extended.",
            ),
            "kan_certificate_extended": _wrap(True),
            "kan_certificate_baseline_reproduced": _wrap(True),
            "max_pwa_segments_verified": _wrap(10),
            "false_property_rejected": _wrap(True),
            "blocked_reason": None,
        },
        5243: {
            **_base(
                "experiment_5243_hardware_continuity_kan_pbit_boundary",
                "complete: kv260=reachable polarfire=reachable gatemate=blocked_physical_jtag no_speedup_claim",
            ),
            "kv260_status": _wrap("reachable"),
            "kv260_ssh_only_confirmed": _wrap(True),
            "polarfire_status": _wrap("reachable"),
            "gatemate_status": _wrap("blocked_physical_jtag"),
            "speedup_claimed": _wrap(False),
        },
    }


def _make_repo(root: Path, *, omit: set[int] | None = None) -> None:
    omit = omit or set()
    payloads = _payloads()
    for source in mod.UPSTREAM_SOURCES:
        if source.experiment_number not in omit:
            _write_json(root / source.relative_path, payloads[source.experiment_number])
    for context_path in mod.SOURCE_CONTEXT_PATHS:
        path = root / context_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"context for {context_path}\n", encoding="utf-8")


def _validation() -> list[dict[str, str]]:
    return [{"command": "focused pytest", "status": "PASS", "notes": "fixture"}]


def _field(field: str, value: Any) -> dict[str, Any]:
    return _wrap(value, mod.FIELD_PRINCIPLES[field])


def test_req_capstone_5244_spec_declares_v479_contract() -> None:
    """REQ-CAPSTONE-5244: OpenSpec declares the V479 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5244") :]

    for marker in (
        "REQ-CAPSTONE-5244",
        "SCENARIO-CAPSTONE-5244",
        "SCENARIO-CAPSTONE-5244-FIELD-PRINCIPLES",
        mod.EXPERIMENT_ID,
        str(mod.RESULT_RELATIVE_PATH),
        "gap4_final_status",
        "ops_docs_updated=false",
        "hardware_speedup_claimed=false",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5244_excludes_flagged_and_preserves_bounded_claims(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5244: flagged rows stay out of headline decisions."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260704",
        duration_s=1.25,
        validation_commands_run=_validation(),
        conductor_untouched=True,
        ops_docs_updated=False,
    )

    mod.validate_artifact(artifact)
    assert mod.value_of(artifact["milestone"]) == mod.MILESTONE
    assert mod.value_of(artifact["tasks_seen"]) == 11
    assert mod.value_of(artifact["missing_artifacts"]) == []
    assert mod.value_of(artifact["gap4_final_status"]) == "blocked"
    assert mod.value_of(artifact["gap1_final_status"]) == "blocked"
    assert mod.value_of(artifact["veribmc_final_status"]) == "retired"
    assert mod.value_of(artifact["continuous_self_learning_status"]) == "controlled_positive"
    assert mod.value_of(artifact["arc_level_delta"]) == 0
    assert mod.value_of(artifact["kan_certificate_status"]) == "extended"
    assert mod.value_of(artifact["hardware_speedup_claimed"]) is False
    assert mod.value_of(artifact["ops_docs_updated"]) is False
    assert mod.value_of(artifact["research_conductor_py_untouched_confirmed"]) is True
    assert mod.value_of(artifact["inference_substrate"]) == mod.INFERENCE_SUBSTRATE
    assert artifact["flagged_adversarial"] is False

    headline = mod.value_of(artifact["headline_eligible_artifacts"])
    blocked = mod.value_of(artifact["gated_or_blocked_artifacts"])
    excluded = artifact["excluded_from_headline_artifacts"]
    summary = artifact["per_task_summary"]

    assert "exp5235-adversarial-qa-null-tautology-calibration-v479" not in headline
    assert "exp5236-gap4-clean-status-after-qa-calibration-v479" not in headline
    assert "exp5241-arc-gated-live-patch-attempt-v479" not in headline
    assert "exp5242-kan-certificate-abstraction-scale-v479" in headline
    assert any(
        row["task_id"] == "exp5236-gap4-clean-status-after-qa-calibration-v479" for row in blocked
    )
    assert any(row["task_id"] == "exp5237-gap1-stability-freeze-or-retire-v479" for row in blocked)
    assert excluded["exp5241-arc-gated-live-patch-attempt-v479"]["exclusion_reasons"] == [
        "flagged_adversarial",
        "critical_corrigendum_pending",
        "failed_validation_command",
    ]
    assert summary["exp5242-kan-certificate-abstraction-scale-v479"]["headline_eligible"] is True
    assert summary["exp5236-gap4-clean-status-after-qa-calibration-v479"]["status"] == "blocked"
    assert "GAP-4 blocked" in mod.value_of(artifact["honest_verdict"])
    assert "VerIbmc retired" in mod.value_of(artifact["honest_verdict"])
    assert "hardware no-speedup" in mod.value_of(artifact["honest_verdict"])


def test_req_capstone_5244_validation_and_helper_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5244: schema validation rejects capstone overclaims."""

    _make_repo(tmp_path, omit={5243})
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260704",
        duration_s=2.0,
        validation_commands_run=_validation(),
        conductor_untouched=True,
        ops_docs_updated=False,
    )

    assert mod.value_of(artifact["missing_artifacts"]) == [
        "exp5243-hardware-continuity-kan-pbit-boundary-v479"
    ]
    assert artifact["status_decisions"]["hardware"] == "hardware continuity evidence missing"
    mod.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(
            {key: value for key, value in artifact.items() if key != "duration_s"}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": _field("honest_verdict", "done")})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(
            artifact | {"inference_substrate": _field("inference_substrate", "live_llm")}
        )
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(artifact | {"milestone": _field("milestone", "2026.07.480")})
    with pytest.raises(ValueError, match="field principle mismatch"):
        mod.validate_artifact(artifact | {"field_principles": {"milestone": "wrong"}})
    with pytest.raises(ValueError, match="gap4_final_status field principle mismatch"):
        mod.validate_artifact(artifact | {"gap4_final_status": _wrap("blocked")})
    with pytest.raises(ValueError, match="flagged_adversarial"):
        mod.validate_artifact(artifact | {"flagged_adversarial": True})
    with pytest.raises(ValueError, match="research_conductor"):
        mod.validate_artifact(
            artifact
            | {
                "research_conductor_py_untouched_confirmed": _field(
                    "research_conductor_py_untouched_confirmed", False
                )
            }
        )
    with pytest.raises(ValueError, match="gap4_final_status"):
        mod.validate_artifact(artifact | {"gap4_final_status": _field("gap4_final_status", "open")})
    with pytest.raises(ValueError, match="gap1_final_status"):
        mod.validate_artifact(artifact | {"gap1_final_status": _field("gap1_final_status", "open")})
    with pytest.raises(ValueError, match="veribmc_final_status"):
        mod.validate_artifact(
            artifact | {"veribmc_final_status": _field("veribmc_final_status", "null")}
        )
    with pytest.raises(ValueError, match="continuous_self_learning_status"):
        mod.validate_artifact(
            artifact
            | {
                "continuous_self_learning_status": _field(
                    "continuous_self_learning_status", "positive"
                )
            }
        )
    with pytest.raises(ValueError, match="kan_certificate_status"):
        mod.validate_artifact(
            artifact | {"kan_certificate_status": _field("kan_certificate_status", "produced")}
        )
    with pytest.raises(ValueError, match="hardware_speedup_claimed"):
        mod.validate_artifact(
            artifact | {"hardware_speedup_claimed": _field("hardware_speedup_claimed", True)}
        )
    with pytest.raises(ValueError, match="ops_docs_updated"):
        mod.validate_artifact(artifact | {"ops_docs_updated": _field("ops_docs_updated", True)})
    with pytest.raises(ValueError, match="headline eligible"):
        mod.validate_artifact(
            artifact
            | {
                "headline_eligible_artifacts": _field(
                    "headline_eligible_artifacts",
                    [
                        *mod.value_of(artifact["headline_eligible_artifacts"]),
                        "exp5241-arc-gated-live-patch-attempt-v479",
                    ],
                )
            }
        )
    with pytest.raises(ValueError, match="arc_level_delta"):
        mod.validate_artifact(artifact | {"arc_level_delta": _field("arc_level_delta", "0")})
    with pytest.raises(ValueError, match="tasks_seen"):
        mod.validate_artifact(artifact | {"tasks_seen": _field("tasks_seen", "11")})
    with pytest.raises(ValueError, match="missing_artifacts"):
        mod.validate_artifact(artifact | {"missing_artifacts": _field("missing_artifacts", {})})
    with pytest.raises(ValueError, match="gated_or_blocked_artifacts"):
        mod.validate_artifact(
            artifact | {"gated_or_blocked_artifacts": _field("gated_or_blocked_artifacts", {})}
        )
    with pytest.raises(ValueError, match="headline_eligible_artifacts"):
        mod.validate_artifact(
            artifact | {"headline_eligible_artifacts": _field("headline_eligible_artifacts", {})}
        )
    with pytest.raises(ValueError, match="validation_commands_run"):
        mod.validate_artifact(
            artifact | {"validation_commands_run": _field("validation_commands_run", [])}
        )
    with pytest.raises(ValueError, match="excluded_from_headline_artifacts"):
        mod.validate_artifact(artifact | {"excluded_from_headline_artifacts": []})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "stale"})

    assert mod.source_by_number(5233).task_id == "exp5233-archive-478-activate-479"
    with pytest.raises(KeyError):
        mod.source_by_number(9999)
    assert mod.value_of(_wrap("x")) == "x"
    assert mod.value_of("x") == "x"
    assert mod.honest_verdict_text({"value": "complete_wrapped"}) == "complete_wrapped"
    assert mod.honest_verdict_text(None) == ""
    assert mod.as_bool(_wrap(True)) is True
    assert mod.as_bool(False) is False
    assert mod.as_number("3.5") == 3.5
    assert mod.as_number("bad") is None
    assert mod.as_number(True) is None
    assert mod.has_critical_corrigendum({"corrigendum_pending": [{"severity": "critical"}]})
    assert not mod.has_critical_corrigendum({"corrigendum_pending": [3, {"severity": "warn"}]})
    assert mod.has_methodology_gap({"corrigendum_pending": [{"kind": "METHODOLOGY_MISSING"}]})
    assert mod.failed_validation_command({"arc_validation_commands": [{"passed": False}]})
    assert mod.failed_validation_command({"validation_commands_run": [{"status": "FAIL"}]})
    assert not mod.failed_validation_command({"arc_validation_commands": [3, {"passed": True}]})
    assert not mod.failed_validation_command({"arc_validation_commands": [{"passed": True}]})
    assert mod.exclusion_reasons(
        {
            "flagged_adversarial": True,
            "adversarial_verify_passed": False,
            "corrigendum_pending": [{"kind": "METHODOLOGY_MISSING"}],
            "status": "blocked",
        }
    ) == [
        "flagged_adversarial",
        "adversarial_verify_failed",
        "methodology_incomplete",
        "gate_blocked",
    ]
    assert mod.is_gate_blocked({"status": "blocked"})
    assert mod.is_gate_blocked({"honest_verdict": "blocked_gate_check_failed"})
    assert not mod.is_gate_blocked({"honest_verdict": "complete: ok"})
    assert mod._status_for({"blocked_at_layer": "conductor_pre_gate"}) == "gate_blocked"

    assert mod.gap4_final_status({}, set())[0] == "unknown"
    assert mod.gap4_final_status({5236: {"gap4_status_decision": "clean_positive"}}, {5236})[0] == (
        "clean_positive"
    )
    assert mod.gap4_final_status({5236: {"gap4_status_decision": "clean_null"}}, {5236})[0] == (
        "clean_null"
    )
    assert mod.gap4_final_status({5236: {"gap4_status_decision": "blocked_missing"}}, {5236})[
        0
    ] == ("blocked")
    assert mod.gap4_final_status({5236: {"gap4_status_decision": "strange"}}, {5236})[0] == (
        "unknown"
    )
    assert mod.gap4_final_status({5236: {"gap4_status_decision": "clean_null"}}, set())[0] == (
        "blocked"
    )
    assert mod.gap1_final_status({}, set())[0] == "unknown"
    assert (
        mod.gap1_final_status({5237: {"gap1_registry_promoted": _wrap(True)}}, {5237})[0]
        == "promoted"
    )
    assert mod.gap1_final_status({5237: {"gap1_stability_decision": "retired"}}, {5237})[0] == (
        "retired"
    )
    assert (
        mod.gap1_final_status({5237: {"gap1_stability_decision": "blocked_instability"}}, {5237})[0]
        == "blocked"
    )
    assert mod.gap1_final_status({5237: {"gap1_stability_decision": "other"}}, {5237})[0] == (
        "unknown"
    )
    assert mod.gap1_final_status({5237: {"gap1_stability_decision": "other"}}, set())[0] == (
        "blocked"
    )
    assert mod.veribmc_final_status({}, set())[0] == "unknown"
    assert mod.veribmc_final_status({5238: {"solver_feedback_uplift": 0.25}}, {5238})[0] == (
        "positive"
    )
    assert (
        mod.veribmc_final_status(
            {5238: {"solver_feedback_uplift": 0.0, "retire_current_veribmc_path": True}},
            {5238},
        )[0]
        == "retired"
    )
    assert mod.veribmc_final_status({5238: {"solver_feedback_uplift": 0.0}}, {5238})[0] == (
        "clean_null"
    )
    assert mod.veribmc_final_status({5238: {"solver_feedback_uplift": 0.0}}, set())[0] == (
        "blocked"
    )
    assert mod.continuous_self_learning_status({}, set())[0] == "unknown"
    assert (
        mod.continuous_self_learning_status({5239: {"continuous_self_learning_task": True}}, set())[
            0
        ]
        == "blocked"
    )
    assert (
        mod.continuous_self_learning_status(
            {
                5239: {
                    "continuous_self_learning_task": True,
                    "aligned_vs_shuffled_delta": 0.1,
                    "aligned_vs_no_memory_delta": 0.1,
                    "retention_check_passed": True,
                    "rollback_policy_exercised": True,
                    "broad_self_distillation_used": False,
                }
            },
            {5239},
        )[0]
        == "controlled_positive"
    )
    assert (
        mod.continuous_self_learning_status(
            {5239: {"continuous_self_learning_task": True, "broad_self_distillation_used": True}},
            {5239},
        )[0]
        == "degraded"
    )
    assert (
        mod.continuous_self_learning_status(
            {
                5239: {
                    "continuous_self_learning_task": True,
                    "aligned_vs_shuffled_delta": 0.0,
                    "aligned_vs_no_memory_delta": 0.0,
                    "broad_self_distillation_used": False,
                }
            },
            {5239},
        )[0]
        == "controlled_null"
    )
    assert (
        mod.continuous_self_learning_status(
            {
                5239: {
                    "continuous_self_learning_task": True,
                    "aligned_vs_shuffled_delta": 0.2,
                    "aligned_vs_no_memory_delta": 0.0,
                    "broad_self_distillation_used": False,
                }
            },
            {5239},
        )[0]
        == "degraded"
    )
    assert mod.arc_level_delta({}) == (0, "no ARC live-patch artifact was present")
    assert mod.arc_level_delta({5241: {"reproducible_total_levels_delta": 2}}, {5241})[0] == 2
    assert mod.kan_certificate_status({}, set())[0] == "unknown"
    assert mod.kan_certificate_status({5242: {"kan_certificate_extended": True}}, set())[0] == (
        "blocked"
    )
    assert (
        mod.kan_certificate_status({5242: {"kan_certificate_extended": True}}, {5242})[0]
        == "extended"
    )
    assert (
        mod.kan_certificate_status({5242: {"kan_certificate_baseline_reproduced": True}}, {5242})[0]
        == "tiny_only"
    )
    assert mod.kan_certificate_status({5242: {"blocked_reason": "solver_missing"}}, {5242})[0] == (
        "blocked"
    )
    assert mod.kan_certificate_status({5242: {"blocked_reason": None}}, {5242})[0] == "unknown"
    assert mod.hardware_speedup_claimed({}) == (False, "hardware continuity evidence missing")
    assert mod.hardware_speedup_claimed({5243: {"speedup_claimed": True}})[0] is True

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_mapping(malformed)[1]["error"] == "malformed_json"
    not_mapping = tmp_path / "array.json"
    not_mapping.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(not_mapping)[1]["error"] == "not_json_object"

    out_path = mod.run(
        root=tmp_path,
        run_date="20260704",
        duration_s=2.5,
        validation_commands_run=_validation(),
        conductor_untouched=True,
        ops_docs_updated=False,
    )
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["duration_s"] == 2.5
    assert mod.value_of(saved["validation_commands_run"]) == _validation()
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)


def test_req_capstone_5244_repository_artifact_matches_schema() -> None:
    """REQ-CAPSTONE-5244: checked-in artifact is the stable deliverable."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert mod.value_of(artifact["gap4_final_status"]) == "blocked"
    assert mod.value_of(artifact["gap1_final_status"]) == "blocked"
    assert mod.value_of(artifact["veribmc_final_status"]) == "retired"
    assert mod.value_of(artifact["continuous_self_learning_status"]) == "controlled_positive"
    assert mod.value_of(artifact["arc_level_delta"]) == 0
    assert mod.value_of(artifact["kan_certificate_status"]) == "extended"
    assert mod.value_of(artifact["hardware_speedup_claimed"]) is False
    assert mod.value_of(artifact["ops_docs_updated"]) is False
    assert mod.value_of(artifact["research_conductor_py_untouched_confirmed"]) is True
    assert mod.value_of(artifact["inference_substrate"]) == mod.INFERENCE_SUBSTRATE
    assert mod.value_of(artifact["honest_verdict"]).startswith("complete:")

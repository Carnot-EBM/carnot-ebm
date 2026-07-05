"""Tests for Exp 5256 V480 capstone synthesis.

Spec refs: REQ-CAPSTONE-5256, SCENARIO-CAPSTONE-5256,
SCENARIO-CAPSTONE-5256-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import carnot.experiment_5256_capstone_v480 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _base(verdict: str, substrate: str = mod.INFERENCE_SUBSTRATE) -> dict[str, Any]:
    return {
        "duration_s": 1.0,
        "flagged_adversarial": False,
        "honest_verdict": verdict,
        "inference_substrate": substrate,
        "reproducibility_checksum": "sha256:fixture",
    }


def _fixture_payloads() -> dict[int, dict[str, Any]]:
    return {
        5245: _base("blocked_archive_479_activate_480: full tests blocked activation"),
        5247: {
            **_base("complete: normalizer ready"),
            "artifact_normalizer_ready": True,
            "duration_policy_preserved": True,
        },
        5248: {
            **_base("complete: GAP-4 final decision salvaged_clean_null"),
            "gap4_final_decision": _wrap("salvaged_clean_null"),
            "wins": _wrap(0),
            "losses": _wrap(0),
            "ties": _wrap(120),
            "pool_retired": _wrap(False),
            "unsafe_missing_receipts": _wrap([]),
        },
        5249: {
            **_base(
                "blocked_precondition_cross_model_memory_not_measured",
                "precondition_check_only",
            ),
            "cross_model_memory_eligible": False,
            "model_specs": _wrap(
                {"precondition_audit": {"blockers": ["blocked_llama_cpp_gpu_offload"]}}
            ),
            "retention_check_passed": _wrap(False),
            "rollback_exercised": _wrap(False),
            "no_model_training": _wrap(True),
        },
        5250: {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "cross_model_memory_eligible false",
            "blocked_at_layer": "conductor_pre_gate",
            "duration_s": 0.0,
        },
        5251: {
            **_base(
                "complete: fragment self-checking was harmful", "live_llm_inference_local_gguf_sota"
            ),
            "accuracy_change": _wrap(-0.25),
            "unsupported_claim_delta": _wrap(1.0),
            "deterministic_violation_delta": _wrap(4.0),
            "consumer_recommendation": _wrap("redesign_or_retire"),
            "schema_errors": [],
        },
        5252: {
            **_base(
                "complete: typed provenance memory did not reduce hallucination errors",
                "live_llm_inference_local_gguf_sota",
            ),
            "citation_support_delta": _wrap(0.0),
            "repeated_error_delta": _wrap(0.0),
            "unsupported_claim_rate_no_memory": _wrap(0.0),
            "unsupported_claim_rate_typed_memory": _wrap(0.0),
            "schema_errors": [],
        },
        5253: {
            **_base(
                "complete: level_delta=0 patch_decision=retire_current_provenance_patch",
                "offline_arcade_live_agent_runtime_self_discovery_no_llm",
            ),
            "level_delta": _wrap(0),
            "retire_current_provenance_patch": _wrap(True),
            "solve_provenance": _wrap("live_agent_self_discovery"),
            "duplicate_solve_claimed": _wrap(False),
        },
        5254: {
            **_base(
                "complete: bounded two-variable convex-envelope certificate",
                "offline_deterministic_certificate_no_llm",
            ),
            "variables_verified": _wrap(2),
            "max_segments_or_envelopes_verified": _wrap(2),
            "true_property_certified": _wrap(True),
            "false_property_rejected": _wrap(True),
            "no_hardware_speedup_claim": _wrap(True),
        },
        5255: {
            **_base(
                "complete: kv260=reachable polarfire=reachable gatemate=blocked_physical_jtag no_speedup_claim",
                "hardware_probe_no_speedup_claim",
            ),
            "kv260_status": _wrap("reachable"),
            "polarfire_status": _wrap("reachable"),
            "gatemate_status": _wrap("blocked_physical_jtag"),
            "speedup_claimed": _wrap(False),
        },
    }


def _make_repo(root: Path, *, omit: set[int] | None = None) -> None:
    omit = omit or set()
    for source in mod.UPSTREAM_SOURCES:
        payload = _fixture_payloads().get(source.experiment_number)
        if payload is not None and source.experiment_number not in omit:
            _write_json(root / source.relative_path, payload)
    for context_path in mod.SOURCE_CONTEXT_PATHS:
        path = root / context_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"context for {context_path}\n", encoding="utf-8")


def _validation() -> list[dict[str, str]]:
    return [{"command": "focused pytest", "status": "PASS", "notes": "fixture"}]


def test_req_capstone_5256_spec_declares_v480_contract() -> None:
    """REQ-CAPSTONE-5256: OpenSpec declares the required V480 fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5256") :]

    for marker in (
        "REQ-CAPSTONE-5256",
        "SCENARIO-CAPSTONE-5256",
        "SCENARIO-CAPSTONE-5256-FIELD-PRINCIPLES",
        mod.EXPERIMENT_ID,
        str(mod.RESULT_RELATIVE_PATH),
        "Exp5246 as missing",
        "hardware_speedup_claimed=false",
        "ops_docs_updated=false",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5256_fixture_preserves_missing_gated_and_negative_states(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5256: missing and gated tasks are explicit non-wins."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260705",
        duration_s=1.0,
        validation_commands_run=_validation(),
        conductor_untouched=True,
        ops_docs_updated=False,
    )

    mod.validate_artifact(artifact)
    assert mod.value_of(artifact["tasks_seen"]) == 10
    assert mod.value_of(artifact["artifact_normalizer_status"]).startswith("ready:")
    assert mod.value_of(artifact["gap4_final_status"]).startswith("salvaged_clean_null")
    assert "blocked_precondition" in mod.value_of(artifact["continuous_self_learning_status"])
    assert mod.value_of(artifact["verifier_dose_status"]).startswith("blocked_gate")
    assert mod.value_of(artifact["token_guard_status"]).startswith("clean_negative")
    assert mod.value_of(artifact["halluhard_status"]).startswith("clean_null")
    assert mod.value_of(artifact["arc_level_delta"]) == 0
    assert mod.value_of(artifact["kan_certificate_status"]).startswith("bounded_positive")
    assert mod.value_of(artifact["hardware_speedup_claimed"]) is False
    assert mod.value_of(artifact["ops_docs_updated"]) is False
    assert len(mod.value_of(artifact["recommended_next_tasks"])) == 5
    assert artifact["flagged_adversarial"] is False

    missing_or_skipped = mod.value_of(artifact["tasks_missing_or_skipped"])
    by_id = {row["experiment_number"]: row for row in missing_or_skipped}
    assert by_id[5245]["status"] == "blocked"
    assert by_id[5246]["status"] == "missing"
    assert by_id[5249]["status"] == "blocked"
    assert by_id[5250]["status"] == "gated_skipped"
    assert "GAP-4 salvaged clean null" in mod.value_of(artifact["honest_verdict"])
    assert "ARC delta 0" in mod.value_of(artifact["honest_verdict"])


def test_req_capstone_5256_current_repo_deliverable_shape() -> None:
    """REQ-CAPSTONE-5256: current repo synthesis keeps observed upstream states."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260705",
        duration_s=1.0,
        validation_commands_run=_validation(),
        conductor_untouched=True,
        ops_docs_updated=False,
    )

    mod.validate_artifact(artifact)
    assert mod.value_of(artifact["tasks_seen"]) == 10
    assert mod.value_of(artifact["gap4_final_status"]) == (
        "salvaged_clean_null: wins=0 losses=0 ties=120 unsafe_missing_receipts=0 pool_retired=false"
    )
    assert "blocked_llama_cpp_gpu_offload" in mod.value_of(
        artifact["continuous_self_learning_status"]
    )
    assert "accuracy_change=-0.25" in mod.value_of(artifact["token_guard_status"])
    assert mod.value_of(artifact["arc_level_delta"]) == 0
    assert mod.value_of(artifact["hardware_speedup_claimed"]) is False


def test_req_capstone_5256_validation_rejects_overclaiming_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5256: validation fails closed on schema and claim drift."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260705",
        duration_s=1.0,
        validation_commands_run=_validation(),
        conductor_untouched=True,
        ops_docs_updated=False,
    )

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = mod.wrap_field("inference_substrate", "live_llm")
    with pytest.raises(AssertionError, match="inference_substrate"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["hardware_speedup_claimed"] = mod.wrap_field("hardware_speedup_claimed", True)
    with pytest.raises(AssertionError, match="hardware speedup"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["tasks_seen"] = mod.wrap_field("tasks_seen", "10")
    with pytest.raises(AssertionError, match="tasks_seen"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    del broken["honest_verdict"]
    with pytest.raises(AssertionError, match="missing required field"):
        mod.validate_artifact(broken)


def test_req_capstone_5256_edge_branches_stay_conservative(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5256: unusual source and validation edges do not inflate claims."""

    _make_repo(tmp_path)
    _write_json(tmp_path / mod.UPSTREAM_SOURCES[1].relative_path, _base("complete: exp5246 landed"))
    payloads = _fixture_payloads()
    payloads[5247]["artifact_normalizer_ready"] = False
    payloads[5247]["flagged_adversarial"] = True
    _write_json(tmp_path / mod.UPSTREAM_SOURCES[2].relative_path, payloads[5247])
    _write_json(
        tmp_path / mod.UPSTREAM_SOURCES[5].relative_path,
        {
            **_base("complete: scheduler fixture", "cached_fixture_replay_no_llm"),
            "status": "complete",
        },
    )
    _write_json(
        tmp_path / mod.UPSTREAM_SOURCES[6].relative_path, _base("complete: sparse token guard")
    )
    _write_json(tmp_path / "results/not_object.json", ["not", "an", "object"])

    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260705",
        duration_s=1.0,
        validation_commands_run=_validation(),
        conductor_untouched=True,
        ops_docs_updated=False,
    )

    assert artifact["status_decisions"]["unexpected_5246_state"] == "loaded"
    assert mod.value_of(artifact["artifact_normalizer_status"]).startswith("blocked_or_missing")
    assert mod.value_of(artifact["verifier_dose_status"]).startswith("not_blocked")
    assert "accuracy_change=0.00" in mod.value_of(artifact["token_guard_status"])
    assert (
        artifact["per_task_summary"]["exp5247-slot-artifact-normalizer-v480"]["status"] == "flagged"
    )
    payload, meta = mod.read_json_mapping(tmp_path / "results/not_object.json")
    assert payload == {}
    assert meta["error"] == "not_json_object"

    out_path = tmp_path / "nested" / "capstone.json"
    mod.write_artifact(out_path, artifact)
    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact


def test_req_capstone_5256_validation_rejects_remaining_shape_edges(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5256: every required wrapped field remains typed."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260705",
        duration_s=1.0,
        validation_commands_run=_validation(),
        conductor_untouched=True,
        ops_docs_updated=False,
    )

    broken = copy.deepcopy(artifact)
    broken["honest_verdict"] = mod.wrap_field("honest_verdict", "partial without prefix")
    with pytest.raises(AssertionError, match="honest_verdict"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["tasks_missing_or_skipped"] = mod.wrap_field("tasks_missing_or_skipped", "none")
    with pytest.raises(AssertionError, match="tasks_missing_or_skipped"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["arc_level_delta"] = mod.wrap_field("arc_level_delta", "0")
    with pytest.raises(AssertionError, match="arc_level_delta"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["ops_docs_updated"] = mod.wrap_field("ops_docs_updated", True)
    with pytest.raises(AssertionError, match="ops_docs_updated"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["recommended_next_tasks"] = mod.wrap_field("recommended_next_tasks", ["only one"])
    with pytest.raises(AssertionError, match="recommended_next_tasks"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["flagged_adversarial"] = True
    with pytest.raises(AssertionError, match="flagged_adversarial"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["tasks_seen"] = {"principle": "wrong", "value": 10}
    with pytest.raises(AssertionError, match="tasks_seen"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["duration_s"] = 99.0
    with pytest.raises(AssertionError, match="checksum"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_req_capstone_5256() -> None:
    """SCENARIO-CAPSTONE-5256: committed deliverable JSON is the V480 capstone."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert mod.value_of(artifact["tasks_seen"]) == 10
    assert any(
        row["experiment_number"] == 5246
        for row in mod.value_of(artifact["tasks_missing_or_skipped"])
    )
    assert mod.value_of(artifact["inference_substrate"]) == mod.INFERENCE_SUBSTRATE
    assert mod.value_of(artifact["hardware_speedup_claimed"]) is False

"""Tests for Exp 3064 SSQA host-visible readback boundary ledger.

Spec refs: REQ-HW-090, SCENARIO-HW-090.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import ssqa_host_visible_readback_boundary_ledger_3064 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"
REQUIRED_FIELDS = {
    "ssqa_boundary_ledger_ready",
    "ssqa_readback_allowed",
    "ssqa_status",
    "required_host_visible_fields",
    "current_allowed_claim",
    "hardware_execution_claim_made",
    "speedup_claim_made",
    "source_artifacts",
    "inference_substrate",
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


def _exp3051() -> dict[str, Any]:
    return {
        "experiment": 3051,
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "gate_check_summary": (
            "1 of 1 gate(s) failed; first failure: "
            "exp3050-gatemate-host-visible-flash-smoke-v5.gatemate_host_visible_smoke_passed "
            "(upstream artifact not found for task id 'exp3050-gatemate-host-visible-flash-smoke-v5')"
        ),
        "gates_evaluated": [
            {
                "upstream": "exp3050-gatemate-host-visible-flash-smoke-v5",
                "artifact_field": "gatemate_host_visible_smoke_passed",
                "expected": True,
                "actual": None,
                "passed": False,
            }
        ],
        "blocked_at_layer": "conductor_pre_gate",
    }


def _capstone(ssqa_status: str = "gated_skipped_host_visible_smoke_missing") -> dict[str, Any]:
    return {
        "artifact": "experiment_3053_capstone_v285",
        "capstone_ready": True,
        "paper_ready": False,
        "gatemate_status": "blocked_output_contract",
        "ssqa_status": ssqa_status,
        "missing_source_artifacts": [mod.EXP3050_REL_PATH.as_posix()],
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; "
            f"ssqa_status={ssqa_status}"
        ),
    }


def _exp3063() -> dict[str, Any]:
    return {
        "artifact": "experiment_3063_gatemate_no_rerun_operator_action_ledger_v1",
        "gatemate_no_rerun_ledger_ready": True,
        "gatemate_rerun_allowed": False,
        "downstream_tasks_blocked": [
            {
                "task_id": "exp3051-ssqa-readback-eligibility-bounded-gate-v3",
                "branch_type": "ssqa_readback",
                "allowed_to_rerun": False,
                "matrix_status": "gate_skipped",
                "upstream_blocker": "exp3050.gatemate_host_visible_smoke_passed is missing or false",
            }
        ],
        "hardware_execution_claim_made": False,
        "speedup_claim_made": False,
        "honest_verdict": "complete: gatemate_no_rerun_ledger_ready=true; downstream_blocked=3",
    }


def _write_sources(root: Path, *, smoke: dict[str, Any] | None = None) -> None:
    _write_json(root, mod.EXP3051_BOUNDED_REL_PATH, _exp3051())
    _write_json(root, mod.EXP3053_REL_PATH, _capstone())
    _write_json(root, mod.EXP3063_REL_PATH, _exp3063())
    if smoke is not None:
        _write_json(root, mod.EXP3050_REL_PATH, smoke)
    _write_text(
        root,
        mod.HARDWARE_WISHLIST_REL_PATH,
        "No GateMate latency or speedup claim until host-visible sample-level timing exists.\n",
    )
    _write_text(
        root,
        mod.CHANGELOG_REL_PATH,
        "SSQA stayed gate-skipped because host-visible GateMate smoke was missing.\n",
    )


def _passing_smoke(root: Path) -> dict[str, Any]:
    transcript = root / "logs" / "exp3050" / "host_visible_smoke.txt"
    return {
        "gatemate_host_visible_smoke_passed": True,
        "host_visible_transcript_path": str(transcript),
        "transcript_sha256": "0" * 64,
        "host_reader_command": ".venv/bin/python scripts/gatemate_done_reader.py --expect done=1",
        "expected_transcript": ["done=1 PASS"],
        "observed_transcript": ["done=1 PASS"],
        "transcript_matched": True,
        "selected_output_signal": "done",
        "ccf_binding": {"signal_name": "done", "pin": "IO_EB_B7"},
        "flash_succeeded": True,
        "flash_command": "openFPGALoader -c dirtyJtag -b olimex_gatemateevb bitstream.bit",
        "readback_supported": True,
        "readback_attempted": True,
        "readback_hash": "1" * 64,
        "sample_count": 16,
        "wall_clock_duration_s": 0.25,
        "per_sample_latency_s": 0.015625,
        "sampler_configuration": {"n_spins": 16, "top": "ising_n16_gatemate"},
        "hardware_execution_claim_made": False,
        "speedup_claim_made": False,
        "honest_verdict": "complete: gatemate_host_visible_smoke_passed",
    }


def test_req_hw_090_spec_entry_present() -> None:
    """REQ-HW-090: OpenSpec declares the SSQA readback boundary ledger."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-090" in spec
    assert "SCENARIO-HW-090" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_hw_090_builds_gate_skipped_boundary_ledger(tmp_path: Path) -> None:
    """SCENARIO-HW-090: missing smoke transcript keeps SSQA readback gated."""
    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path)
    source_by_id = {row["experiment_id"]: row for row in artifact["source_artifacts"]}
    required_by_id = {row["field_id"]: row for row in artifact["required_host_visible_fields"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["ssqa_boundary_ledger_ready"] is True
    assert artifact["ssqa_readback_allowed"] is False
    assert artifact["ssqa_status"] == "gated_skipped_host_visible_smoke_missing"
    assert artifact["current_allowed_claim"] == "gated_skipped_host_visible_smoke_missing"
    assert artifact["hardware_execution_claim_made"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["hardware_performance_claim_made"] is False
    assert artifact["sampler_behavior_claim_made"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["host_visible_smoke_evidence"]["present"] is False
    assert artifact["host_visible_smoke_evidence"]["readback_unlocks_ssqa"] is False
    assert artifact["source_summary"]["missing_host_visible_smoke_artifact"] is True

    assert artifact["inference_substrate"] == {
        "kind": "ssqa_host_visible_readback_boundary_ledger",
        "source": "checked_in_local_artifacts",
        "model_inference": False,
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "hardware_readback_attempted": False,
        "flash_attempted": False,
        "rtl_run": False,
        "local_repo_only": True,
        "timing_or_speedup_claim": False,
    }
    assert source_by_id["exp3051"]["path"] == mod.EXP3051_BOUNDED_REL_PATH.as_posix()
    assert source_by_id["exp3053"]["present"] is True
    assert source_by_id["exp3063"]["present"] is True
    assert source_by_id["exp3050"]["present"] is False
    assert required_by_id["host_visible_transcript_path"]["present"] is False
    assert required_by_id["transcript_sha256"]["claim_scope"] == "readback"
    assert required_by_id["per_sample_latency_s"]["claim_scope"] == "acceleration"
    assert required_by_id["sampler_configuration"]["claim_scope"] == "sampler_behavior"


def test_req_hw_090_passing_smoke_evidence_allows_readback_only(tmp_path: Path) -> None:
    """REQ-HW-090: complete host-visible smoke evidence opens readback, not speedup."""
    _write_sources(tmp_path, smoke=_passing_smoke(tmp_path))

    artifact = mod.build_artifact(tmp_path)
    required = {row["field_id"]: row for row in artifact["required_host_visible_fields"]}

    assert artifact["ssqa_boundary_ledger_ready"] is True
    assert artifact["ssqa_readback_allowed"] is True
    assert artifact["ssqa_status"] == "clean_host_visible_smoke_transcript_present"
    assert artifact["current_allowed_claim"] == (
        "host_visible_smoke_transcript_present_ssqa_readback_may_run_no_performance_claim"
    )
    assert artifact["host_visible_smoke_evidence"]["present"] is True
    assert artifact["host_visible_smoke_evidence"]["path"] == mod.EXP3050_REL_PATH.as_posix()
    assert artifact["host_visible_smoke_evidence"]["readback_unlocks_ssqa"] is True
    assert all(row["present"] is True for row in required.values())
    assert artifact["hardware_execution_claim_made"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["acceleration_claim_made"] is False
    assert artifact["sampler_behavior_claim_made"] is False


def test_req_hw_090_nonpassing_smoke_or_missing_source_fails_closed(tmp_path: Path) -> None:
    """REQ-HW-090: non-passing smoke stays skipped; missing capstone blocks readiness."""
    nonpassing = _passing_smoke(tmp_path)
    nonpassing["gatemate_host_visible_smoke_passed"] = False
    nonpassing["transcript_matched"] = False
    _write_sources(tmp_path, smoke=nonpassing)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["ssqa_boundary_ledger_ready"] is True
    assert artifact["ssqa_readback_allowed"] is False
    assert artifact["ssqa_status"] == "gated_skipped_host_visible_smoke_missing"
    assert artifact["host_visible_smoke_evidence"]["present"] is True
    assert artifact["host_visible_smoke_evidence"]["readback_unlocks_ssqa"] is False
    assert "gatemate_host_visible_smoke_passed" in artifact["host_visible_smoke_evidence"][
        "missing_required_fields"
    ]

    missing_source_root = tmp_path / "missing-source"
    _write_json(missing_source_root, mod.EXP3051_BOUNDED_REL_PATH, _exp3051())
    _write_json(missing_source_root, mod.EXP3063_REL_PATH, _exp3063())
    blocked = mod.build_artifact(missing_source_root)

    assert blocked["ssqa_boundary_ledger_ready"] is False
    assert blocked["ssqa_readback_allowed"] is False
    assert blocked["honest_verdict"].startswith("blocked_precondition:")
    assert blocked["missing_source_artifacts"] == [mod.EXP3053_REL_PATH.as_posix()]


def test_req_hw_090_write_artifact_and_helpers_are_stable(tmp_path: Path) -> None:
    """REQ-HW-090: helpers use fallback paths and write stable JSON."""
    _write_json(tmp_path, mod.EXP3051_REQUESTED_REL_PATH, _exp3051())
    _write_json(tmp_path, mod.EXP3053_REL_PATH, _capstone())
    _write_json(tmp_path, mod.EXP3063_REL_PATH, _exp3063())
    _write_text(tmp_path, mod.HARDWARE_WISHLIST_REL_PATH, "hardware boundary\n")
    _write_text(tmp_path, mod.CHANGELOG_REL_PATH, "changelog boundary\n")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")

    output = mod.write_artifact(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["ssqa_boundary_ledger_ready"] is True
    assert payload["ssqa_readback_allowed"] is False
    assert mod.read_json_object(bad) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod._existing_path(tmp_path, (mod.EXP3051_REQUESTED_REL_PATH, mod.EXP3051_BOUNDED_REL_PATH)) == (
        mod.EXP3051_REQUESTED_REL_PATH
    )
    source = mod._source_payload(tmp_path, mod.EXP3051_SOURCE)
    assert source["path"] == mod.EXP3051_REQUESTED_REL_PATH.as_posix()
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._as_text_list("done=1") == ["done=1"]
    assert mod._field_present({"items": ["done=1"]}, "items") is True

"""Tests for Exp 3119 GateMate/SSQA operator evidence ingestion v4.

Spec refs: REQ-HW-094, SCENARIO-HW-094.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.reporting import gatemate_ssqa_operator_evidence_ingestion_3119 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"
TRANSCRIPT_REL_PATH = Path("logs/experiment_3050/host_visible_smoke.txt")
REQUIRED_FIELDS = {
    "operator_evidence_ingestion_v4_ready",
    "gatemate_rerun_allowed",
    "ssqa_readback_allowed",
    "missing_operator_actions",
    "evidence_files_seen",
    "hardware_commands_run",
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _v3_artifact() -> dict[str, Any]:
    missing_actions = [
        {
            "missing_item": "authoritative_pinout_ccf_binding",
            "operator_action": "Provide an authoritative GateMate CCF Pin_out binding.",
            "present": False,
            "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            "checked_paths": [
                mod.EXP3048_REL_PATH.as_posix(),
                mod.GATEMATE_CCF_REL_PATH.as_posix(),
            ],
        },
        {
            "missing_item": "host_reader_command",
            "operator_action": "Commit a concrete host reader command for done.",
            "present": False,
            "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            "checked_paths": [mod.EXP3048_REL_PATH.as_posix()],
        },
        {
            "missing_item": "expected_transcript",
            "operator_action": "Record the expected pass/fail transcript.",
            "present": False,
            "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            "checked_paths": [mod.EXP3048_REL_PATH.as_posix()],
        },
        {
            "missing_item": "safety_limits",
            "operator_action": "Open the downstream flash safety gate.",
            "present": False,
            "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            "checked_paths": [mod.EXP3048_REL_PATH.as_posix()],
        },
        {
            "missing_item": "host_visible_smoke_evidence",
            "operator_action": "Commit a passing host-visible smoke transcript.",
            "present": False,
            "source_artifact": mod.EXP3050_REL_PATH.as_posix(),
            "checked_paths": [mod.EXP3050_REL_PATH.as_posix()],
            "missing_required_fields": [field for field, _scope in mod.HOST_VISIBLE_REQUIRED_FIELDS],
        },
    ]
    source_artifacts = [
        {
            "path": mod.EXP3048_REL_PATH.as_posix(),
            "role": "gatemate_output_contract_operator_package",
            "required": False,
            "present": True,
            "readable": True,
        },
        {
            "path": mod.EXP3050_REL_PATH.as_posix(),
            "role": "host_visible_gatemate_smoke_transcript",
            "required": False,
            "present": False,
            "readable": False,
        },
        {
            "path": mod.GATEMATE_CCF_REL_PATH.as_posix(),
            "role": "checked_in_gatemate_constraints",
            "required": False,
            "present": True,
            "readable": True,
        },
    ]
    return {
        "artifact": "experiment_3106_gatemate_ssqa_operator_evidence_ingestion_v3",
        "operator_evidence_ingestion_v3_ready": True,
        "gatemate_rerun_allowed": False,
        "ssqa_readback_allowed": False,
        "missing_operator_actions": missing_actions,
        "checked_paths": [
            {
                "path": row["path"],
                "evidence_class": row["role"],
                "present": row["present"],
                "readable": row["readable"],
            }
            for row in source_artifacts
        ],
        "source_artifacts": source_artifacts,
        "hardware_commands_run": [],
        "speedup_claim_made": False,
        "honest_verdict": "complete: operator_evidence_ingestion_v3_ready=true",
    }


def _exp3048(*, ready: bool = False) -> dict[str, Any]:
    return {
        "artifact": "experiment_3048_gatemate_output_contract_operator_package_v1",
        "gatemate_output_contract_ready": ready,
        "host_visible_io_plan_ready": ready,
        "selected_output_signal": "done",
        "ccf_binding": {"signal_name": "done", "pin": "IO_EB_B7"} if ready else {},
        "host_reader_command": (
            ".venv/bin/python scripts/gatemate_done_reader.py --expect done=1" if ready else ""
        ),
        "expected_transcript": ["done=1 PASS"] if ready else [],
        "safety_limits": {
            "downstream_flash_gate_open": ready,
            "max_flash_attempts_without_operator_review": 1 if ready else 0,
        },
    }


def _smoke_payload(*, transcript_sha256: str) -> dict[str, Any]:
    return {
        "gatemate_host_visible_smoke_passed": True,
        "host_visible_transcript_path": TRANSCRIPT_REL_PATH.as_posix(),
        "transcript_sha256": transcript_sha256,
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
    }


def _write_documented_sources(
    root: Path,
    *,
    gatemate_ready: bool = False,
    smoke_ready: bool = False,
    transcript_file: bool = False,
) -> None:
    _write_json(root, mod.EXP3106_REL_PATH, _v3_artifact())
    _write_json(root, mod.EXP3048_REL_PATH, _exp3048(ready=gatemate_ready))
    _write_text(
        root,
        mod.GATEMATE_CCF_REL_PATH,
        "Pin_out done Loc = IO_EB_B7\n" if gatemate_ready else "# no physical Pin_out\n",
    )
    if transcript_file:
        _write_text(root, TRANSCRIPT_REL_PATH, "done=1 PASS\n")
    if smoke_ready:
        sha = _sha256_text("done=1 PASS\n") if transcript_file else "0" * 64
        _write_json(root, mod.EXP3050_REL_PATH, _smoke_payload(transcript_sha256=sha))


def test_req_hw_094_spec_entry_present() -> None:
    """REQ-HW-094: OpenSpec declares the v4 operator evidence contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-094" in spec
    assert "SCENARIO-HW-094" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_hw_094_missing_evidence_keeps_boundaries_blocked(tmp_path: Path) -> None:
    """SCENARIO-HW-094: incomplete documented evidence keeps hardware blocked."""
    _write_documented_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path)
    missing = {row["missing_item"]: row for row in artifact["missing_operator_actions"]}
    seen = {row["path"]: row for row in artifact["evidence_files_seen"]}
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["operator_evidence_ingestion_v4_ready"] is True
    assert artifact["gatemate_rerun_allowed"] is False
    assert artifact["ssqa_readback_allowed"] is False
    assert artifact["hardware_commands_run"] == []
    assert artifact["speedup_claim_made"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"]["no_live_model_inference"] is True
    assert artifact["inference_substrate"]["executes_hardware"] is False
    assert artifact["inference_substrate"]["hardware_readback_attempted"] is False

    assert set(missing) == {
        "authoritative_pinout_ccf_binding",
        "host_reader_command",
        "expected_transcript",
        "safety_limits",
        "host_visible_smoke_evidence",
    }
    assert missing["authoritative_pinout_ccf_binding"]["missing_required_fields"]
    assert "host_visible_transcript_path" in missing["host_visible_smoke_evidence"][
        "missing_required_fields"
    ]
    assert mod.EXP3106_REL_PATH.as_posix() in seen
    assert mod.EXP3048_REL_PATH.as_posix() in seen
    assert mod.EXP3050_REL_PATH.as_posix() not in seen
    assert sources[mod.EXP3050_REL_PATH.as_posix()]["present"] is False


def test_req_hw_094_complete_evidence_only_recommends_future_operator_task(
    tmp_path: Path,
) -> None:
    """REQ-HW-094: complete evidence opens only future operator-owned scope."""
    _write_documented_sources(
        tmp_path,
        gatemate_ready=True,
        smoke_ready=True,
        transcript_file=True,
    )

    artifact = mod.build_artifact(tmp_path)
    seen = {row["path"]: row for row in artifact["evidence_files_seen"]}

    assert artifact["operator_evidence_ingestion_v4_ready"] is True
    assert artifact["gatemate_rerun_allowed"] is True
    assert artifact["ssqa_readback_allowed"] is True
    assert artifact["missing_operator_actions"] == []
    assert artifact["hardware_commands_run"] == []
    assert artifact["speedup_claim_made"] is False
    assert artifact["hardware_execution_claim_made"] is False
    assert artifact["allowed_next_experiment_scope"] == (
        "operator_allowed_future_task: documented evidence supports an "
        "operator-owned SSQA readback task only; Exp 3119 ran no hardware "
        "command and makes no timing or speedup claim"
    )
    assert artifact["host_visible_smoke_evidence"]["ready"] is True
    assert artifact["host_visible_smoke_evidence"]["transcript_file_sha256_matched"] is True
    assert seen[TRANSCRIPT_REL_PATH.as_posix()]["sha256"] == _sha256_text("done=1 PASS\n")


def test_req_hw_094_gatemate_only_keeps_ssqa_readback_blocked(tmp_path: Path) -> None:
    """REQ-HW-094: GateMate evidence alone cannot authorize SSQA readback."""
    _write_documented_sources(tmp_path, gatemate_ready=True)

    artifact = mod.build_artifact(tmp_path)
    missing = {row["missing_item"]: row for row in artifact["missing_operator_actions"]}

    assert artifact["gatemate_rerun_allowed"] is True
    assert artifact["ssqa_readback_allowed"] is False
    assert list(missing) == ["host_visible_smoke_evidence"]
    assert artifact["allowed_next_experiment_scope"] == (
        "operator_allowed_future_task: documented GateMate rerun evidence is "
        "complete, but SSQA readback remains blocked until host-visible smoke "
        "transcript evidence is complete"
    )


def test_req_hw_094_untraceable_transcript_keeps_ssqa_blocked(tmp_path: Path) -> None:
    """REQ-HW-094: SSQA evidence needs a local transcript file trace."""
    _write_documented_sources(
        tmp_path,
        gatemate_ready=True,
        smoke_ready=True,
        transcript_file=False,
    )

    artifact = mod.build_artifact(tmp_path)
    missing = {row["missing_item"]: row for row in artifact["missing_operator_actions"]}

    assert artifact["gatemate_rerun_allowed"] is True
    assert artifact["ssqa_readback_allowed"] is False
    assert missing["host_visible_smoke_evidence"]["missing_required_fields"] == [
        "host_visible_transcript_file"
    ]
    assert artifact["host_visible_smoke_evidence"]["transcript_file_present"] is False


def test_req_hw_094_missing_v3_blocks_precondition(tmp_path: Path) -> None:
    """REQ-HW-094: missing v3 evidence artifact honestly blocks v4 readiness."""
    artifact = mod.build_artifact(tmp_path)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert artifact["operator_evidence_ingestion_v4_ready"] is False
    assert artifact["gatemate_rerun_allowed"] is False
    assert artifact["ssqa_readback_allowed"] is False
    assert sources[mod.EXP3106_REL_PATH.as_posix()]["required"] is True
    assert artifact["honest_verdict"].startswith("blocked_precondition:")


def test_req_hw_094_write_artifact_and_helpers_are_stable(tmp_path: Path) -> None:
    """REQ-HW-094: writing and fail-closed helpers stay deterministic."""
    _write_documented_sources(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")

    output = mod.write_artifact(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["operator_evidence_ingestion_v4_ready"] is True
    assert mod.read_json_object(bad) == {}
    assert mod.read_json_object(scalar) == {}
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._as_text_list("done=1") == ["done=1"]
    assert mod._coerce_rel_path(" results/example.json ") == Path("results/example.json")
    assert mod._ccf_binding_from_text("Pin_out done Loc = IO_EB_B7\n", "done") == {
        "signal_name": "done",
        "pin": "IO_EB_B7",
        "line": "Pin_out done Loc = IO_EB_B7",
        "line_number": 1,
        "source_path": mod.GATEMATE_CCF_REL_PATH.as_posix(),
    }

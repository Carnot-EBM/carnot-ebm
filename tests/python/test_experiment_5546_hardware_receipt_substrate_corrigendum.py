"""Tests for Exp5546 hardware receipt substrate corrigendum.

Spec refs: REQ-VERIFY-5546, SCENARIO-VERIFY-5546.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5546_hardware_receipt_substrate_corrigendum as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path(
    "tests/python/test_experiment_5546_hardware_receipt_substrate_corrigendum.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _receipt(
    *,
    device: str,
    status: str = "reachable",
    classification: str = "timing_blocked",
    blocked_reason: str | None = None,
    command_kinds: list[str] | None = None,
    device_names: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "device": device,
        "status": status,
        "classification": classification,
        "parseable": True,
        "blocked_reason": blocked_reason,
        "parser_version": "hardware_receipt_parser_repeatability.v1",
        "device_names": device_names or [device],
        "driver_versions": {"nvidia_driver": "610.43.02"} if device == "cuda" else {},
        "runtime_versions": {"cuda": "12.8"} if device == "cuda" else {},
        "versions": {"machine": "aarch64"} if device == "kv260" else {},
        "memory": {"device_memory": [{"index": 0, "total_mib": 24576}]}
        if device == "cuda"
        else {},
        "metadata": metadata or {},
        "command_kinds": command_kinds or [f"{device}_identity"],
    }


def _clean_5532_payload() -> dict[str, object]:
    return {
        "experiment_id": "exp5532-hardware-receipt-parser-repeatability",
        "run_date": "2026-07-10",
        "inference_substrate": "hardware_receipt_parser_repeatability",
        "matched_timing_available": False,
        "hardware_speedup_claim": False,
        "parser_failures": {},
        "device_receipts": {
            "cpu": _receipt(
                device="cpu",
                command_kinds=["cpu_info"],
                device_names=["AMD Ryzen AI 9 HX 370"],
            ),
            "cuda": _receipt(
                device="cuda",
                classification="workload_blocked",
                command_kinds=["cuda_runtime_info", "cuda_driver_info"],
                device_names=["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090"],
            ),
            "polarfire": _receipt(
                device="polarfire",
                command_kinds=["polarfire_ssh_identity"],
                device_names=["Microchip PolarFire-SoC Discovery Kit"],
                metadata={"hostname": "mpfs-disco-kit", "machine": "riscv64"},
            ),
            "kv260": _receipt(
                device="kv260",
                status="blocked_identity",
                classification="identity_blocked",
                blocked_reason="blocked_kv260_ssh_identity",
                command_kinds=["kv260_ssh_identity"],
                device_names=[],
            ),
            "gatemate": _receipt(
                device="gatemate",
                status="blocked_identity",
                classification="identity_blocked",
                blocked_reason="blocked_gatemate_dirtyjtag_identity",
                command_kinds=[
                    "gatemate_dirtyjtag_detect",
                    "gatemate_toolchain_yosys",
                    "gatemate_toolchain_nextpnr_himbaechel",
                    "gatemate_toolchain_gmpack",
                ],
                device_names=[],
            ),
        },
    }


def _fallback_5519_payload() -> dict[str, object]:
    return {
        "experiment_id": "exp5519-hardware-continuity-methodology-receipts",
        "run_date": "2026-07-10",
        "inference_substrate": "hardware_receipts",
        "matched_timing_available": False,
        "hardware_speedup_claim": False,
        "cpu_receipt": _receipt(device="cpu", command_kinds=["cpu_info"]),
        "cuda_receipt": _receipt(
            device="cuda",
            command_kinds=["cuda_runtime_info", "cuda_driver_info"],
            device_names=["NVIDIA GeForce RTX 3090"],
        ),
        "polar_fire_receipt": _receipt(
            device="polar_fire",
            command_kinds=["polarfire_ssh_identity_hash"],
            device_names=["Microchip PolarFire-SoC Discovery Kit"],
        ),
        "kv260_receipt": _receipt(
            device="kv260",
            status="blocked_identity",
            classification="identity_blocked",
            blocked_reason="blocked_kv260_ssh_identity",
            command_kinds=["kv260_ssh_identity"],
            device_names=[],
        ),
        "gatemate_receipt": _receipt(
            device="gatemate",
            status="blocked_identity",
            classification="identity_blocked",
            blocked_reason="blocked_gatemate_dirtyjtag_identity",
            command_kinds=["gatemate_dirtyjtag_detect"],
            device_names=[],
        ),
    }


def _write_sources(root: Path, exp5532: dict[str, object] | None = None) -> None:
    _write_json(
        root / mod.SOURCE_RELATIVE_PATHS[0],
        exp5532 if exp5532 is not None else _clean_5532_payload(),
    )
    _write_json(root / mod.SOURCE_RELATIVE_PATHS[1], _fallback_5519_payload())


def test_req_verify_5546_spec_declares_no_llm_corrigendum_contract() -> None:
    """REQ-VERIFY-5546: OpenSpec anchors fields, source parsing, and no-speedup."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5546") : spec.index("### REQ-VERIFY-5532")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-VERIFY-5546",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp 5532",
        "Exp 5519",
        "hardware_receipt_methodology_no_llm",
        "`model_specs`",
        "`target_model`",
        "`/dev/mmcblk*`",
        "`/dev/disk`",
        "`xmutil`",
        "`/dev/uio*`",
        "hardware_speedup_claim",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5546_builds_clean_no_llm_corrigendum(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5546: receipt parsing emits clean methodology gates."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    saved_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert saved["llm_invoked"] is False
    assert saved["no_model_specs_required"] is True
    assert isinstance(saved["random_seed"], int)
    assert saved["random_seed"] == mod.derive_random_seed(
        saved["source_input_checksums"], mod.PARSER_VERSION
    )
    assert len(saved["reproducibility_checksum"]) == 64
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert saved["compute_bound_markers_absent"] is True
    assert saved["parser_rows_valid"] is True
    assert saved["kv260_safe_path_used"] is True
    assert saved["matched_timing_available"] is False
    assert saved["hardware_speedup_claim"] is False
    assert saved["hardware_receipt_corrigendum_clean"] is True
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert [row["device"] for row in saved["device_receipts"]] == list(mod.DEVICE_ORDER)
    assert all(row["parser_outcome"] == "parsed" for row in saved["device_receipts"])
    assert saved["device_receipts"][0]["source_artifact"] == str(mod.SOURCE_RELATIVE_PATHS[0])
    assert saved["device_receipts"][1]["device_identities"] == [
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 3090",
    ]
    assert {
        (row["device"], row["blocked_reason"])
        for row in saved["blockers"]
        if row["kind"] == "device_blocker"
    } == {
        ("kv260", "blocked_kv260_ssh_identity"),
        ("gatemate", "blocked_gatemate_dirtyjtag_identity"),
    }
    assert any(row["kind"] == "matched_timing_missing" for row in saved["blockers"])

    artifact_text = mod.canonical_json(saved)
    for forbidden in ('"model_specs":', '"target_model":', "GGUF", "live_llm_inference"):
        assert forbidden not in artifact_text
    mod.validate_artifact(saved)


def test_req_verify_5546_fallback_rows_and_unsafe_kv260_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5546: malformed rows and unsafe KV260 evidence are explicit blockers."""

    exp5532 = _clean_5532_payload()
    device_receipts = exp5532["device_receipts"]
    assert isinstance(device_receipts, dict)
    device_receipts.pop("cpu")
    kv260 = device_receipts["kv260"]
    assert isinstance(kv260, dict)
    kv260["command_kinds"] = ["kv260_host_mmcblk_probe"]
    kv260["status"] = "reachable"
    kv260["classification"] = "workload_blocked"
    kv260["blocked_reason"] = None
    gatemate = device_receipts["gatemate"]
    assert isinstance(gatemate, dict)
    gatemate.pop("status")
    _write_sources(tmp_path, exp5532)

    artifact = mod.build_artifact(root=tmp_path, tests_added_or_reused=[TEST_PATH.as_posix()])

    assert artifact["device_receipts"][0]["source_artifact"] == str(mod.SOURCE_RELATIVE_PATHS[1])
    assert artifact["device_receipts"][0]["parser_outcome"] == "parsed"
    assert artifact["device_receipts"][3]["parser_outcome"] == "parsed"
    assert artifact["device_receipts"][4]["parser_outcome"] == "malformed"
    assert artifact["parser_rows_valid"] is False
    assert artifact["kv260_safe_path_used"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["hardware_receipt_corrigendum_clean"] is False
    assert any(row["kind"] == "unsafe_kv260_command" for row in artifact["blockers"])
    assert any(row["kind"] == "parser_blocker" and row["device"] == "gatemate" for row in artifact["blockers"])
    mod.validate_artifact(artifact)

    speedup = deepcopy(artifact)
    speedup["hardware_speedup_claim"] = True
    speedup["reproducibility_checksum"] = mod.payload_checksum(speedup)
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(speedup)

    model_specs = deepcopy(artifact)
    model_specs["model_specs"] = {"headline_required_any_of": ["Qwen3.5-35B-A3B-GGUF"]}
    model_specs["compute_bound_markers_absent"] = False
    model_specs["reproducibility_checksum"] = mod.payload_checksum(model_specs)
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(model_specs)

    llm = deepcopy(artifact)
    llm["llm_invoked"] = True
    llm["reproducibility_checksum"] = mod.payload_checksum(llm)
    with pytest.raises(ValueError, match="llm_invoked"):
        mod.validate_artifact(llm)

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "0" * 64
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum)

    marker = deepcopy(artifact)
    marker["notes"] = "live_llm_inference"
    marker["reproducibility_checksum"] = mod.payload_checksum(marker)
    with pytest.raises(ValueError, match="compute_bound_markers_absent"):
        mod.validate_artifact(marker)

    assert mod.compute_bound_markers_absent({"target_model": "not-used"}) is False
    assert mod.kv260_command_kinds_safe("kv260_ssh_identity") is False
    assert mod.normalize_tests(None) == [TEST_PATH.as_posix()]

    source_without_experiment_id = {
        "source_artifact": "results/source.json",
        "payload": {"device_receipts": {"cpu": "not-a-receipt"}},
    }
    assert mod.source_experiment_id(source_without_experiment_id) is None
    assert mod.normalize_device_receipt(
        device="cpu",
        sources=[source_without_experiment_id],
    )["blocked_reason"] == "receipt_not_mapping"


def test_req_verify_5546_missing_sources_are_blocked_without_speedup(tmp_path: Path) -> None:
    """REQ-VERIFY-5546: absent inputs create blocked parser rows, never speedup."""

    _write_json(tmp_path / mod.SOURCE_RELATIVE_PATHS[0], [])
    non_mapping_sources = mod.load_source_inputs(tmp_path)
    assert non_mapping_sources[0]["error"] == "source_not_mapping"
    (tmp_path / mod.SOURCE_RELATIVE_PATHS[0]).unlink()

    artifact = mod.run_experiment(
        repo_root=tmp_path,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    saved = json.loads(artifact.read_text(encoding="utf-8"))

    assert artifact == tmp_path / mod.RESULT_RELATIVE_PATH
    assert saved["parser_rows_valid"] is False
    assert saved["hardware_receipt_corrigendum_clean"] is False
    assert saved["hardware_speedup_claim"] is False
    assert {row["parser_outcome"] for row in saved["device_receipts"]} == {"missing"}
    assert {row["kind"] for row in saved["blockers"]} >= {
        "source_missing",
        "parser_blocker",
        "matched_timing_missing",
    }
    mod.validate_artifact(saved)

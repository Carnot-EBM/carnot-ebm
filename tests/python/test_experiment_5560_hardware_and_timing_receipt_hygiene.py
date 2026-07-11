"""Tests for Exp5560 hardware and timing receipt hygiene.

Spec refs: REQ-VERIFY-5560, SCENARIO-VERIFY-5560.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5546_hardware_receipt_substrate_corrigendum as exp5546
from carnot import experiment_5560_hardware_and_timing_receipt_hygiene as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5560_hardware_and_timing_receipt_hygiene.py")


class StepClock:
    """Deterministic monotonic clock for launch/finish receipt assertions."""

    def __init__(self) -> None:
        self.value = 5560.0

    def __call__(self) -> float:
        self.value += 0.125
        return self.value


def _write_json(path: Path, payload: dict[str, object] | list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _device_row(
    *,
    device: str,
    safe_command_kinds: list[str],
    status: str = "reachable",
    classification: str = "timing_blocked",
    blocked_reason: str | None = None,
    identities: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "device": device,
        "source_artifact": str(exp5546.SOURCE_RELATIVE_PATHS[0]),
        "source_experiment_id": "exp5532-hardware-receipt-parser-repeatability",
        "parser_outcome": "parsed",
        "status": status,
        "classification": classification,
        "blocked_reason": blocked_reason,
        "source_parser_version": "hardware_receipt_parser_repeatability.v1",
        "device_identities": identities or [device],
        "driver_versions": {},
        "memory": {},
        "metadata": metadata or {},
        "safe_command_kinds": safe_command_kinds,
    }


def _upstream_corrigendum() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": exp5546.SCHEMA,
        "experiment": 5546,
        "experiment_id": exp5546.EXPERIMENT_ID,
        "parser_version": exp5546.PARSER_VERSION,
        "llm_invoked": False,
        "no_model_specs_required": True,
        "compute_bound_markers_absent": True,
        "device_receipts": [
            _device_row(device="cpu", safe_command_kinds=["cpu_info"], identities=["cpu"]),
            _device_row(
                device="cuda",
                safe_command_kinds=["cuda_runtime_info", "cuda_driver_info"],
                classification="workload_blocked",
                identities=["NVIDIA GeForce RTX 3090"],
            ),
            _device_row(
                device="polarfire",
                safe_command_kinds=["polarfire_ssh_identity"],
                identities=["Microchip PolarFire-SoC Discovery Kit"],
            ),
            _device_row(
                device="kv260",
                safe_command_kinds=["kv260_ssh_identity"],
                status="blocked_identity",
                classification="identity_blocked",
                blocked_reason="blocked_kv260_ssh_identity",
                identities=[],
            ),
            _device_row(
                device="gatemate",
                safe_command_kinds=[
                    "gatemate_dirtyjtag_detect",
                    "gatemate_toolchain_yosys",
                ],
                status="blocked_identity",
                classification="identity_blocked",
                blocked_reason="blocked_gatemate_dirtyjtag_identity",
                identities=[],
            ),
        ],
        "parser_rows_valid": True,
        "kv260_safe_path_used": True,
        "matched_timing_available": False,
        "hardware_speedup_claim": False,
        "reproducibility_checksum": "",
        "inference_substrate": exp5546.INFERENCE_SUBSTRATE,
        "honest_verdict": "complete: clean upstream",
    }
    payload["reproducibility_checksum"] = exp5546.payload_checksum(payload)
    return payload


def _write_upstream(root: Path, payload: dict[str, object] | None = None) -> None:
    _write_json(root / mod.UPSTREAM_HARDWARE_CORRIGENDUM, payload or _upstream_corrigendum())


def test_req_verify_5560_spec_declares_timing_hygiene_contract() -> None:
    """REQ-VERIFY-5560: OpenSpec anchors safe receipts and timing gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5560") : spec.index("### REQ-VERIFY-5546")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-VERIFY-5560",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.UPSTREAM_HARDWARE_CORRIGENDUM),
        "hardware_receipt_and_timing_hygiene_no_llm",
        "`/dev/mmcblk*`",
        "`/dev/disk`",
        "`xmutil`",
        "`/dev/uio*`",
        "monotonic clock",
        "artifact checksum",
        "hardware_speedup_claim",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5560_builds_clean_hygiene_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5560: clean upstream receipts keep timing explicit."""

    _write_upstream(tmp_path)

    artifact = mod.build_artifact(
        root=tmp_path,
        clock=StepClock(),
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    saved_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert saved["llm_invoked"] is False
    assert saved["no_model_specs_required"] is True
    assert saved["upstream_hardware_corrigendum"] == str(mod.UPSTREAM_HARDWARE_CORRIGENDUM)
    assert [row["device"] for row in saved["device_receipts"]] == list(exp5546.DEVICE_ORDER)
    assert saved["kv260_safe_path_used"] is True
    assert saved["forbidden_block_device_paths_used"] is False
    assert saved["parser_rows_valid"] is True
    assert saved["launch_finish_receipt_ready"] is True
    assert saved["monotonic_clock_used"] is True
    assert saved["artifact_checksum_linked"] is True
    assert saved["matched_timing_available"] is False
    assert saved["repeated_timing_pairs"] == 0
    assert saved["hardware_speedup_claim"] is False
    assert saved["conductor_modified"] is False
    assert saved["roadmap_yaml_unchanged"] is True
    assert saved["tests_added_or_reused"] == [TEST_PATH.as_posix()]
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert len(saved["reproducibility_checksum"]) == 64
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)

    receipt = saved["launch_finish_receipt_examples"][0]
    assert receipt["clock_source"] == "time.perf_counter"
    assert receipt["finish_monotonic_s"] > receipt["launch_monotonic_s"]
    assert receipt["duration_s"] == pytest.approx(0.125)
    assert receipt["artifact_path"] == str(mod.UPSTREAM_HARDWARE_CORRIGENDUM)
    assert receipt["artifact_checksum_matches"] is True
    assert receipt["artifact_file_sha256"]

    mod.validate_artifact(saved)


def test_req_verify_5560_blocks_unsafe_paths_and_overclaims(tmp_path: Path) -> None:
    """REQ-VERIFY-5560: unsafe paths and speedup claims fail closed."""

    upstream = _upstream_corrigendum()
    rows = upstream["device_receipts"]
    assert isinstance(rows, list)
    kv260 = rows[3]
    assert isinstance(kv260, dict)
    kv260["safe_command_kinds"] = ["kv260_host_mmcblk_probe"]
    kv260["metadata"] = {"bad_path": "/dev/mmcblk0"}
    upstream["kv260_safe_path_used"] = False
    upstream["reproducibility_checksum"] = exp5546.payload_checksum(upstream)
    _write_upstream(tmp_path, upstream)

    artifact = mod.build_artifact(
        root=tmp_path,
        clock=StepClock(),
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )

    assert artifact["kv260_safe_path_used"] is False
    assert artifact["forbidden_block_device_paths_used"] is True
    assert artifact["parser_rows_valid"] is True
    assert artifact["launch_finish_receipt_ready"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert any(blocker["kind"] == "unsafe_kv260_command" for blocker in artifact["blockers"])
    assert any(blocker["kind"] == "forbidden_block_device_path" for blocker in artifact["blockers"])
    mod.validate_artifact(artifact)

    speedup = deepcopy(artifact)
    speedup["hardware_speedup_claim"] = True
    speedup["reproducibility_checksum"] = mod.payload_checksum(speedup)
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(speedup)

    conductor = deepcopy(artifact)
    conductor["conductor_modified"] = True
    conductor["reproducibility_checksum"] = mod.payload_checksum(conductor)
    with pytest.raises(ValueError, match="conductor_modified"):
        mod.validate_artifact(conductor)


def test_req_verify_5560_missing_upstream_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5560: missing upstream and helper edge cases stay explicit."""

    output = mod.run_experiment(repo_root=tmp_path, tests_added_or_reused=[TEST_PATH.as_posix()])
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["parser_rows_valid"] is False
    assert saved["kv260_safe_path_used"] is False
    assert saved["launch_finish_receipt_ready"] is False
    assert saved["artifact_checksum_linked"] is False
    assert saved["hardware_speedup_claim"] is False
    assert saved["honest_verdict"].startswith("blocked:")
    assert {blocker["kind"] for blocker in saved["blockers"]} >= {
        "upstream_hardware_corrigendum_missing",
        "parser_rows_invalid",
        "matched_timing_missing",
    }
    mod.validate_artifact(saved)

    assert mod.contains_forbidden_block_device_path({"safe": ["ssh", 7]}) is False
    assert mod.contains_forbidden_block_device_path(["ssh", {"bad": "/dev/disk/by-id/x"}]) is True
    assert mod.contains_forbidden_block_device_path(7) is False
    assert mod.count_repeated_timing_pairs(
        [
            {"device": "cpu", "workload_hash": "a", "duration_s": 1.0},
            {"device": "polarfire", "workload_hash": "a", "duration_s": 2.0},
            {"device": "kv260", "workload_hash": "b", "duration_s": 3.0},
            {"device": 7, "workload_hash": "bad-device", "duration_s": 1.0},
            {"device": "cpu", "workload_hash": "missing-timing"},
            "not-a-row",
        ]
    ) == 1
    assert mod.count_repeated_timing_pairs(None) == 0
    assert mod.normalize_tests(None) == [TEST_PATH.as_posix()]

    blocker_artifact = dict(saved, conductor_modified=True, roadmap_yaml_unchanged=False)
    blockers = mod.collect_blockers(
        blocker_artifact,
        {
            "path": str(mod.UPSTREAM_HARDWARE_CORRIGENDUM),
            "blocked_reason": None,
        },
    )
    assert {"kind": "conductor_modified"} in blockers
    assert {"kind": "roadmap_yaml_changed"} in blockers

    _write_json(tmp_path / mod.UPSTREAM_HARDWARE_CORRIGENDUM, [])
    loaded = mod.load_upstream_corrigendum(tmp_path)
    assert loaded["present"] is False
    assert loaded["blocked_reason"] == "upstream_hardware_corrigendum_not_mapping"

    malformed_upstream = _upstream_corrigendum()
    malformed_upstream["device_receipts"] = ["not-a-row"]
    malformed_upstream["reproducibility_checksum"] = exp5546.payload_checksum(malformed_upstream)
    _write_upstream(tmp_path, malformed_upstream)
    malformed = mod.build_artifact(
        root=tmp_path,
        clock=StepClock(),
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    assert malformed["device_receipts"] == []
    assert malformed["parser_rows_valid"] is False

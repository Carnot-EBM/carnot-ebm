"""Tests for Exp5930 adaptive-state ABI v2 board mapping.

Spec refs: REQ-HW-5930, SCENARIO-HW-5930,
REQ-FPGA-5930, SCENARIO-FPGA-5930.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5930_adaptive_state_board_mapping as mod


REPO = Path(__file__).resolve().parents[2]
HW_SPEC = REPO / "openspec/capabilities/hardware/spec.md"
FPGA_SPEC = REPO / "openspec/capabilities/fpga/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5930_adaptive_state_board_mapping.py")


class StaticToolRunner:
    """REQ-HW-5930 fake local runner; board probes must never reach it."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float) -> mod.CommandReceipt:
        assert timeout_s > 0.0
        rendered = mod.command_to_string(command)
        forbidden = (
            "ssh -o ConnectTimeout=5 -o BatchMode=yes kria",
            "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire",
            "openFPGALoader -c dirtyJtag --detect",
            "xmutil",
            "/dev/mmcblk",
            "/dev/disk",
            "program",
        )
        assert not any(token in rendered for token in forbidden), rendered
        self.commands.append(command)
        stdout = f"{command[0]} exp5930 fixture\n"
        if command[0] == "yosys":
            stdout += "Number of cells: 42\nEstimated LUTs: 42\n"
        if command[0] == "vvp":
            stdout += "EXP5930 ABI v2 RTL smoke PASS\n"
        return mod.CommandReceipt(command=command, exit_code=0, stdout=stdout, stderr="", duration_s=0.001)


def _test_exit_codes() -> dict[str, int]:
    return {
        ".venv/bin/pytest tests/python/test_experiment_5930_adaptive_state_board_mapping.py -q --no-cov -n 0": 0,
        ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5930_adaptive_state_board_mapping.py -m pytest tests/python/test_experiment_5930_adaptive_state_board_mapping.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5930_adaptive_state_board_mapping.py --fail-under=100": 0,
        ".venv/bin/pytest tests/python -q": 0,
        ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_5930_adaptive_state_board_mapping.py": 0,
        ".venv/bin/python scripts/adversarial_verify.py results/experiment_5930_adaptive_state_board_mapping.json": 0,
        ".venv/bin/python scripts/root_clutter_sweep.py": 0,
        "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md _bmad/traceability.md": 0,
    }


def _artifact(runner: StaticToolRunner | None = None) -> dict[str, object]:
    return mod.build_artifact(
        root=REPO,
        run_date="20260726",
        duration_s=1.25,
        command_runner=runner or StaticToolRunner(),
        test_exit_codes=_test_exit_codes(),
    )


def test_req_hw_fpga_5930_specs_declare_board_mapping_contract() -> None:
    """REQ-HW-5930/REQ-FPGA-5930: OpenSpec anchors the static mapping work."""

    hw = HW_SPEC.read_text(encoding="utf-8")
    fpga = FPGA_SPEC.read_text(encoding="utf-8")
    hw_section = hw[hw.index("### REQ-HW-5930") : hw.index("### SCENARIO-HW-5930")]
    fpga_section = fpga[fpga.index("### REQ-FPGA-5930") : fpga.index("### SCENARIO-FPGA-5930")]
    normalized_hw = " ".join(hw_section.split())

    for marker in (
        "REQ-HW-5930",
        "SCENARIO-HW-5930",
        "REQ-FPGA-5930",
        "SCENARIO-FPGA-5930",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "physical_probe_executed=false",
        "board_abi_mapping_ready_score=1.0",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in hw_section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in hw_section
        assert " ".join(principle.split()) in normalized_hw
    for marker in (
        mod.RTL_RELATIVE_PATH.as_posix(),
        "valid/ready",
        "stale state-version requests",
        "resource reports as estimates",
    ):
        assert marker in fpga_section


def test_req_fpga_5930_fixed_width_contract_and_rtl_source_are_backend_neutral() -> None:
    """REQ-FPGA-5930: every ABI v2 op has fixed-width request/response semantics."""

    contract = mod.fixed_width_contract()
    mapping = mod.abi_operation_mapping(REPO)
    rtl = (REPO / mod.RTL_RELATIVE_PATH).read_text(encoding="utf-8")

    assert set(mapping) == set(mod.ABI_V2_OPERATIONS)
    assert len({row["opcode"] for row in mapping.values()}) == len(mod.ABI_V2_OPERATIONS)
    assert contract["request_bits"] == 1936
    assert contract["response_bits"] == 1640
    assert contract["request_fields"]["expected_state_version"] == "u32"
    assert contract["request_fields"]["validator_receipt_hash"] == "u256"
    assert contract["response_fields"]["error_code"] == "u8"
    assert contract["status_codes"]["OK"] == 0
    assert contract["error_codes"]["STALE_STATE_VERSION"] != contract["error_codes"]["OK"]
    assert "assign req_ready = (!resp_valid) || resp_ready;" in rtl
    assert "MODEL_WEIGHT" not in rtl.upper()
    assert "SPEEDUP" not in rtl.upper()
    assert "POWER" not in rtl.upper()


def test_scenario_hw_fpga_5930_simulator_reference_trace_and_adversarial_parity() -> None:
    """SCENARIO-HW-5930/SCENARIO-FPGA-5930: trace replay and unsafe sequences match."""

    trace = mod.replay_exp5926_trace()
    adversarial = mod.run_adversarial_matrix()
    recovery = adversarial["cases"]["crash_recovery"]

    assert trace["trace_count"] == 1
    assert trace["operation_count"] == 37
    assert trace["state_hash_parity"] is True
    assert trace["status_error_parity"] is True
    assert trace["parity_failures"] == []
    assert adversarial["all_rejected_or_recovered"] is True
    assert adversarial["state_hash_unchanged_for_all_rejections"] is True
    assert adversarial["cases"]["backpressure_stall"]["mutation_observed"] is False
    assert adversarial["cases"]["stale_version"]["error_code"] == "STALE_STATE_VERSION"
    assert adversarial["cases"]["replayed_commit"]["error_code"] == "REPLAYED_COMMIT"
    assert adversarial["cases"]["tamper_validator"]["error_code"] == "INVALID_VALIDATOR_RECEIPT"
    assert recovery["recovered_state_hash"] == recovery["checkpoint_state_hash"]


def test_req_hw_5930_artifact_skips_unchanged_physical_routes_and_preserves_board_states() -> None:
    """REQ-HW-5930: unchanged Exp5861 routes prevent physical board commands."""

    runner = StaticToolRunner()
    artifact = _artifact(runner)

    assert artifact["status"] == "complete_static_mapping_no_physical_probe"
    assert artifact["spec_refs"] == [
        "REQ-HW-5930",
        "SCENARIO-HW-5930",
        "REQ-FPGA-5930",
        "SCENARIO-FPGA-5930",
    ]
    assert artifact["gate_replay_receipt"]["exp5926_ready_score"] == 1.0
    assert artifact["authenticated_route_state_diff"]["materially_new_authenticated_route"] is False
    assert artifact["physical_probe_executed"] is False
    assert artifact["bounded_physical_trace_and_teardown_if_any"] == []
    assert artifact["kv260_polarfire_and_gatemate_state_receipts"] == {
        "kv260": "programmed_image_poc",
        "polarfire": "prior_physical_workload_only",
        "gatemate": "blocked_idcode",
    }
    assert {row["board"] for row in artifact["no_unchanged_probe_receipt"]["avoided"]} == {
        "kv260",
        "polarfire",
        "gatemate",
    }
    assert artifact["no_speedup_power_energy_thermalization_convergence_tsu_kona_or_sovereignty_claim"] is True
    assert isinstance(artifact["board_abi_mapping_ready_score"], float)
    assert artifact["board_abi_mapping_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["honest_verdict"].startswith("complete_static_mapping:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert all("ssh" not in command[0] for command in runner.commands)
    mod.validate_artifact(artifact)


def test_req_hw_5930_static_tool_receipts_are_hashed_and_labelled_as_estimates() -> None:
    """REQ-HW-5930: lint, simulation, synthesis, timing, and resource receipts are auditable."""

    artifact = _artifact()
    receipts = artifact["installed_toolchain_target_command_exit_and_hash_receipts"]
    reports = artifact["static_synthesis_timing_estimate_and_resource_reports"]

    assert {"iverilog_lint", "iverilog_sim_build", "rtl_simulation", "yosys_generic_synth"} <= set(receipts)
    for receipt in receipts.values():
        assert receipt["scope"] == "local_static_tool_no_board_probe"
        assert receipt["exit_code"] == 0
        assert mod.is_sha256(receipt["stdout_sha256"])
        assert mod.is_sha256(receipt["stderr_sha256"])
        assert receipt["measurement_type"] == "static_estimate"
        assert receipt["physical_measurement"] is False
    assert reports["resource_report"]["measurement_type"] == "static_estimate"
    assert reports["timing_estimate"]["physical_measurement"] is False
    assert reports["resource_report"]["tool_receipt"] == "yosys_generic_synth"
    mod.validate_artifact(artifact)


def test_req_hw_5930_schema_rejects_overclaims_probe_laundering_and_checksum_drift() -> None:
    """REQ-HW-5930: validation blocks fake physical execution and prohibited claims."""

    base = _artifact()
    mutations = [
        (lambda a: a.update(physical_probe_executed=True), "physical_probe_executed"),
        (
            lambda a: a.update(
                no_speedup_power_energy_thermalization_convergence_tsu_kona_or_sovereignty_claim=False
            ),
            "prohibited claim",
        ),
        (lambda a: a.update(board_abi_mapping_ready_score=0.5), "ready score"),
        (lambda a: a.update(inference_substrate="hardware_smoke"), "inference_substrate"),
        (lambda a: a.update(verifier_is_oracle=False), "verifier_is_oracle"),
        (lambda a: a["field_provenance"].pop("status"), "field provenance"),  # type: ignore[index,union-attr]
        (lambda a: a.update(honest_verdict="complete_static_mapping: speedup=true"), "honest_verdict"),
        (lambda a: a.update(field_principles={}), "field_principles"),
        (
            lambda a: a.update(bounded_physical_trace_and_teardown_if_any=[{"command": "fixture"}]),
            "physical trace",
        ),
        (
            lambda a: a["protected_files_unchanged"].update(unchanged=False),  # type: ignore[index,union-attr]
            "protected files",
        ),
    ]

    for mutate, needle in mutations:
        artifact = deepcopy(base)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        assert any(needle in error for error in mod.artifact_schema_errors(artifact)), needle

    checksum_drift = deepcopy(base)
    checksum_drift["status"] = "tampered"
    assert any("checksum" in error for error in mod.artifact_schema_errors(checksum_drift))

    missing = deepcopy(base)
    del missing["status"]
    assert any("missing required fields" in error for error in mod.artifact_schema_errors(missing))

    with pytest.raises(ValueError, match="physical_probe_executed"):
        invalid = deepcopy(base)
        invalid["physical_probe_executed"] = True
        invalid["reproducibility_checksum"] = mod.payload_checksum(invalid)
        mod.validate_artifact(invalid)


def test_req_hw_5930_helper_error_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-HW-5930: helper error edges stay deterministic and auditable."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object expected"):
        mod.read_json(bad_json)

    kernel = mod.abi_v2.AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    root_hash = kernel.canonical_state_hash()
    rollback = mod._apply_operation(
        kernel,
        {"event_id": "rollback-fixture", "op": "rollback", "target_state_hash": root_hash},
        {},
        {},
        expected_state_version=0,
    )
    assert rollback["accepted"] is True
    with pytest.raises(ValueError, match="unsupported operation"):
        mod._apply_operation(kernel, {"event_id": "bad", "op": "bad-op"}, {}, {})

    assert mod._parse_cell_estimate({"stdout_excerpt": ["no resource marker"]}) is None
    assert mod._parse_cell_estimate({"stdout_tail_excerpt": ["     7651 cells"]}) == 7651

    real_apply = mod._apply_operation
    flipped = {"done": False}

    def skew_once(
        kernel_obj: object,
        operation: dict[str, object],
        snapshots: dict[str, str],
        proposals: dict[str, str],
        *,
        expected_state_version: int | None = None,
    ) -> dict[str, object]:
        result = real_apply(
            kernel_obj,
            operation,
            snapshots,
            proposals,
            expected_state_version=expected_state_version,
        )
        if expected_state_version is None and not flipped["done"]:
            result = dict(result)
            result["code"] = "SKEWED_REFERENCE"
            flipped["done"] = True
        return result

    monkeypatch.setattr(mod, "_apply_operation", skew_once)
    parity = mod.replay_exp5926_trace()
    assert parity["status_error_parity"] is False
    assert parity["parity_failures"]


def test_req_hw_5930_run_experiment_writes_valid_json(tmp_path: Path) -> None:
    """REQ-HW-5930: run_experiment writes the deliverable artifact atomically."""

    artifact = _artifact()
    out_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert saved == artifact
    mod.validate_artifact(saved)

    live_path = mod.run_experiment(
        repo_root=REPO,
        output_root=tmp_path,
        run_date="20260726",
        duration_s=1.25,
        command_runner=StaticToolRunner(),
        test_exit_codes=_test_exit_codes(),
    )
    live = json.loads(live_path.read_text(encoding="utf-8"))
    assert live_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert live["reproducibility_checksum"] == mod.payload_checksum(live)
    assert live["test_exit_codes"] == _test_exit_codes()
    mod.validate_artifact(live)

"""Tests for Exp5573 matched CPU/CUDA sampler and board continuity receipts.

Spec refs: REQ-VERIFY-5573, SCENARIO-VERIFY-5573.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot import experiment_5573_matched_sampler_hardware_continuity as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5573_matched_sampler_hardware_continuity.py")


class StepClock:
    """Deterministic clock that makes timing receipts stable in tests."""

    def __init__(self) -> None:
        self.value = 5573.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


class FakeCuda:
    """Tiny CUDA facade for precondition tests."""

    def __init__(self, *, available: bool, memory_raises: bool = False) -> None:
        self.available = available
        self.memory_raises = memory_raises
        self.sync_calls = 0

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1 if self.available else 0

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "Fake RTX 3090"

    def mem_get_info(self, index: int) -> tuple[int, int]:
        assert index == 0
        if self.memory_raises:
            raise RuntimeError("memory unavailable")
        return (12 * 1024**3, 24 * 1024**3)

    def memory_reserved(self, index: int) -> int:
        assert index == 0
        if self.memory_raises:
            raise RuntimeError("reserved unavailable")
        return 128 * 1024**2

    def synchronize(self) -> None:
        self.sync_calls += 1


class FakeTorch:
    """Torch-shaped object exposing just enough CUDA metadata for Exp5573."""

    __version__ = "2.9.0+cu128"

    def __init__(self, *, cuda_available: bool = True, memory_raises: bool = False) -> None:
        self.cuda = FakeCuda(available=cuda_available, memory_raises=memory_raises)
        self.version = type("Version", (), {"cuda": "12.8"})()


class RecordingRunner:
    """SCENARIO-VERIFY-5573 fake command runner with safe command assertions."""

    def __init__(self, probes: dict[tuple[str, ...], mod.CommandProbe]) -> None:
        self.probes = dict(probes)
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float) -> mod.CommandProbe:
        assert timeout_s > 0.0
        rendered = mod.command_to_string(command)
        assert "/dev/mmcblk" not in rendered
        assert "/dev/disk" not in rendered
        assert "--write" not in rendered
        assert "flash" not in rendered.lower()
        self.commands.append(command)
        if command not in self.probes:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command]


def _probe(
    command: tuple[str, ...],
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _descriptor_payload() -> dict[str, object]:
    descriptors: list[dict[str, object]] = []
    for row_index, row_id in enumerate(("row_a", "row_b", "row_c")):
        variables = [f"fact:{row_id}:v{i}" for i in range(8)]
        target = {
            variables[row_index]: "present",
            variables[(row_index + 3) % len(variables)]: "absent",
        }
        descriptors.append(
            {
                "descriptor_id": f"desc:{row_id}",
                "row_id": row_id,
                "all_repair_variables": variables,
                "repair_block_variables": list(target),
                "target_repair_assignment": target,
                "active_constraints": [
                    {
                        "kind": "unit-test",
                        "repair_variable": variable,
                        "target": value,
                    }
                    for variable, value in target.items()
                ],
            }
        )
    return {
        "schema": "unit.descriptor.bundle",
        "descriptor_count": len(descriptors),
        "asp_repair_descriptors": descriptors,
    }


def _write_descriptor(root: Path, payload: dict[str, object] | None = None) -> None:
    path = root / mod.DESCRIPTOR_SOURCE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or _descriptor_payload(), sort_keys=True), encoding="utf-8")


def _runner(*, kv260_reached: bool = True, polarfire_reached: bool = True) -> RecordingRunner:
    kv_stdout = (
        "board_identity=kv260\nhostname=kria\nmachine=aarch64\n"
        "app=carnot_ising_v2_n64 loaded\nuio=/dev/uio0 /dev/uio1\n"
    )
    pf_stdout = (
        "board_identity=polarfire\nhostname=mpfs-disco-kit\nmachine=riscv64\n"
        "kernel=6.18.17\nworkload_sha256="
        + "a" * 64
        + "\n"
    )
    return RecordingRunner(
        {
            mod.KV260_CONTINUITY_COMMAND: _probe(
                mod.KV260_CONTINUITY_COMMAND,
                exit_code=0 if kv260_reached else 255,
                stdout=kv_stdout if kv260_reached else "",
                stderr="" if kv260_reached else "ssh timeout",
            ),
            mod.POLARFIRE_WORKLOAD_COMMAND: _probe(
                mod.POLARFIRE_WORKLOAD_COMMAND,
                exit_code=0 if polarfire_reached else 255,
                stdout=pf_stdout if polarfire_reached else "",
                stderr="" if polarfire_reached else "ssh timeout",
            ),
            mod.GATEMATE_JTAG_COMMAND: _probe(
                mod.GATEMATE_JTAG_COMMAND,
                exit_code=1,
                stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
                stderr="no idcode found\n",
            ),
        }
    )


def _fake_sampler_rows(
    instances: list[mod.IsingInstance],
    seeds: tuple[int, ...],
    samples_per_pair: int,
    matched_schedule: dict[str, object],
    torch_module: object,
    clock: object,
) -> list[dict[str, object]]:
    del torch_module, clock
    rows: list[dict[str, object]] = []
    for instance in instances:
        for seed in seeds:
            for backend, wall_scale in (("cpu", 2.0), ("cuda", 1.0)):
                base = -float(instance.size) - float(seed % 7)
                rows.append(
                    {
                        "pair_id": f"{instance.instance_id}:seed{seed}",
                        "backend": backend,
                        "instance_id": instance.instance_id,
                        "size": instance.size,
                        "seed": seed,
                        "samples": samples_per_pair,
                        "temperature": matched_schedule["temperature"],
                        "warmup_steps": matched_schedule["warmup_steps"],
                        "thinning": matched_schedule["thinning"],
                        "precision": matched_schedule["precision"],
                        "best_energy": base - (0.1 if backend == "cuda" else 0.0),
                        "energy_mean": base + 1.5,
                        "energy_std": 0.5,
                        "energy_min": base - 0.1,
                        "energy_max": base + 3.0,
                        "energy_quantiles": {"p05": base, "p50": base + 1.0, "p95": base + 2.0},
                        "constraint_satisfaction_rate": 0.9,
                        "autocorrelation_time": 1.5,
                        "effective_sample_size": samples_per_pair / 1.5,
                        "wall_time_s": wall_scale + float(instance.size) / 1000.0,
                        "warmup_time_s": wall_scale / 10.0,
                        "sample_time_s": wall_scale,
                        "result_hash": f"{backend}-{instance.instance_id}-{seed}",
                    }
                )
    return rows


def test_req_verify_5573_spec_declares_matched_sampler_contract() -> None:
    """REQ-VERIFY-5573: OpenSpec anchors sampler matching and board gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5573") : spec.index("### REQ-VERIFY-5546")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-VERIFY-5573",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.DESCRIPTOR_SOURCE_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`/dev/mmcblk*`",
        "`/dev/disk`",
        "at least 10000 post-warmup samples",
        "current physical/JTAG visibility",
        "without inventing matched timing",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5573_builds_matched_artifact_and_raw_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5573: successful CPU/CUDA pairs produce bounded speedups."""

    _write_descriptor(tmp_path)
    runner = _runner(kv260_reached=True, polarfire_reached=True)
    artifact = mod.build_artifact(
        root=tmp_path,
        command_runner=runner,
        torch_module=FakeTorch(cuda_available=True),
        sampler_runner=_fake_sampler_rows,
        clock=StepClock(),
        timestamp=lambda: "2026-07-11T12:00:00Z",
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    saved_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert runner.commands == [
        mod.KV260_CONTINUITY_COMMAND,
        mod.POLARFIRE_WORKLOAD_COMMAND,
        mod.GATEMATE_JTAG_COMMAND,
    ]
    assert saved["descriptor_source"]["path"] == str(mod.DESCRIPTOR_SOURCE_RELATIVE_PATH)
    assert saved["descriptor_source"]["available"] is True
    assert saved["instance_sizes"] == [32, 64]
    assert saved["seeds"] == list(mod.DEFAULT_SEEDS)
    assert saved["samples_per_pair"] == 10000
    assert saved["matched_schedule"]["warmup_steps"] == mod.DEFAULT_WARMUP_STEPS
    assert saved["matched_schedule"]["precision"] == "float32"
    assert saved["cpu_device_receipt"]["status"] == "reachable"
    assert saved["cuda_device_receipt"]["status"] == "reachable"
    assert saved["successful_matched_pairs"] == 6
    assert len(saved["speedup_by_pair"]) == 6
    assert all(row["speedup"] > 1.0 for row in saved["speedup_by_pair"])
    assert saved["hardware_speedup_claim_allowed"] is True
    assert saved["kv260_receipt"]["lane_status"] == "reached"
    assert saved["kv260_mmcblk_accessed"] is False
    assert saved["polarfire_receipt"]["lane_status"] == "reached"
    assert saved["gatemate_receipt"]["lane_status"] == "unchanged_blocker"
    assert saved["board_speedup_claimed"] is False
    assert saved["raw_rows_path"] == mod.RAW_ROWS_RELATIVE_PATH.as_posix()
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)

    raw_rows = json.loads((tmp_path / mod.RAW_ROWS_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert len(raw_rows) == 12
    assert {row["samples"] for row in raw_rows} == {10000}
    assert {row["backend"] for row in raw_rows} == {"cpu", "cuda"}
    mod.validate_artifact(saved)


def test_scenario_verify_5573_blocks_cuda_absence_without_sampler_fallback(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5573: absent CUDA emits a blocked receipt, not CPU fallback."""

    _write_descriptor(tmp_path)

    def forbidden_sampler(*args: Any, **kwargs: Any) -> list[dict[str, object]]:
        raise AssertionError("sampler must not run when CUDA is unavailable")

    artifact = mod.build_artifact(
        root=tmp_path,
        command_runner=_runner(kv260_reached=False, polarfire_reached=False),
        torch_module=FakeTorch(cuda_available=False),
        sampler_runner=forbidden_sampler,
        clock=StepClock(),
        timestamp=lambda: "2026-07-11T12:00:00Z",
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )

    assert artifact["preconditions"]["cuda_available"] is False
    assert artifact["cuda_device_receipt"]["status"] == "blocked"
    assert artifact["successful_matched_pairs"] == 0
    assert artifact["speedup_by_pair"] == []
    assert artifact["hardware_speedup_claim_allowed"] is False
    assert artifact["kv260_receipt"]["lane_status"] == "unchanged_blocker"
    assert artifact["polarfire_receipt"]["lane_status"] == "changed_blocker"
    assert artifact["gatemate_receipt"]["lane_status"] == "unchanged_blocker"
    assert artifact["kv260_mmcblk_accessed"] is False
    assert artifact["board_speedup_claimed"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_verify_5573_derives_instances_from_descriptors() -> None:
    """REQ-VERIFY-5573: ASP/FSM descriptor rows lift into deterministic Ising sizes."""

    instances = mod.build_ising_instances(
        _descriptor_payload(),
        instance_sizes=(16, 64),
    )

    assert [instance.size for instance in instances] == [16, 64]
    assert all(instance.couplings.shape == (instance.size, instance.size) for instance in instances)
    assert all(instance.biases.shape == (instance.size,) for instance in instances)
    assert all(instance.target_spins.shape == (instance.size,) for instance in instances)
    assert all(instance.constraint_indices for instance in instances)
    assert all(instance.checksum for instance in instances)
    assert instances[1].descriptor_ids


def test_req_verify_5573_validation_rejects_overclaims_and_bad_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-5573: validation fails closed on unsafe or unsupported claims."""

    _write_descriptor(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        command_runner=_runner(),
        torch_module=FakeTorch(cuda_available=True),
        sampler_runner=_fake_sampler_rows,
        clock=StepClock(),
        timestamp=lambda: "2026-07-11T12:00:00Z",
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )

    bad_mmc = deepcopy(artifact)
    bad_mmc["kv260_mmcblk_accessed"] = True
    bad_mmc["reproducibility_checksum"] = mod.payload_checksum(bad_mmc)
    with pytest.raises(ValueError, match="kv260_mmcblk_accessed"):
        mod.validate_artifact(bad_mmc)

    bad_board = deepcopy(artifact)
    bad_board["board_speedup_claimed"] = True
    bad_board["reproducibility_checksum"] = mod.payload_checksum(bad_board)
    with pytest.raises(ValueError, match="board_speedup_claimed"):
        mod.validate_artifact(bad_board)

    bad_samples = deepcopy(artifact)
    bad_samples["samples_per_pair"] = 9999
    bad_samples["reproducibility_checksum"] = mod.payload_checksum(bad_samples)
    with pytest.raises(ValueError, match="samples_per_pair"):
        mod.validate_artifact(bad_samples)

    bad_speedup = deepcopy(artifact)
    bad_speedup["successful_matched_pairs"] = 0
    bad_speedup["hardware_speedup_claim_allowed"] = True
    bad_speedup["reproducibility_checksum"] = mod.payload_checksum(bad_speedup)
    with pytest.raises(ValueError, match="hardware_speedup_claim_allowed"):
        mod.validate_artifact(bad_speedup)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

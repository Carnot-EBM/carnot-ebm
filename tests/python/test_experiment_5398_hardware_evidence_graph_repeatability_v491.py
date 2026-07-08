"""Tests for Exp 5398 hardware evidence graph repeatability receipts.

Spec refs: REQ-HW-5398, SCENARIO-HW-5398.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5398_hardware_evidence_graph_repeatability_v491 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"


class RecordingRunner:
    """SCENARIO-HW-5398 runner with explicit safe command receipts."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        assert timeout_s > 0.0
        command = tuple(command)
        self.commands.append(command)
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


class StepClock:
    """Deterministic clock for REQ-HW-5398 duration and checksum assertions."""

    def __init__(self) -> None:
        self.value = 5398.0

    def __call__(self) -> float:
        self.value += 0.125
        return self.value


def _probe(
    command: tuple[str, ...],
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _workload_stdout(
    *,
    input_sha256: str | None = None,
    output_sha256: str | None = None,
    wall_time_s: float = 0.1,
) -> str:
    return (
        json.dumps(
            {
                "hostname": "mpfs-disco-kit",
                "input_sha256": input_sha256 or mod.POLARFIRE_EXPECTED_INPUT_SHA256,
                "output_sha256": output_sha256 or mod.POLARFIRE_EXPECTED_OUTPUT_SHA256,
                "python_version": "3.12.12",
                "uname": "Linux mpfs-disco-kit 6.18.17-linux4microchip-2026.04.1 riscv64",
                "wall_time_s": wall_time_s,
            },
            sort_keys=True,
        )
        + "\n"
    )


def _base_probes(*, openfpga_present: bool = True) -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    openfpga_path = "/opt/oss-cad-suite/bin/openFPGALoader" if openfpga_present else ""
    openfpga_version = "openFPGALoader v1.1.1" if openfpga_present else ""
    return {
        mod.HOST_DATE_COMMAND: [
            _probe(
                mod.HOST_DATE_COMMAND,
                stdout=(
                    "host=carnot-host\n"
                    "date_utc=2026-07-08T14:00:00Z\n"
                    "date_local=2026-07-08T10:00:00-0400\n"
                ),
            )
        ],
        mod.HARDWARE_ENV_COMMAND: [
            _probe(
                mod.HARDWARE_ENV_COMMAND,
                stdout=(
                    "CARNOT_MODE=live\n"
                    "EXTROPIC_API_KEY=do-not-record\n"
                    "KONA_API_KEY=also-hidden\n"
                    "PRIVATE_TOKEN=hidden\n"
                ),
            )
        ],
        mod.TOOL_VERSION_COMMAND: [
            _probe(
                mod.TOOL_VERSION_COMMAND,
                stdout=(
                    "ssh_path=/usr/bin/ssh\nssh_version=OpenSSH_10.0p1\n"
                    f"openFPGALoader_path={openfpga_path}\n"
                    f"openFPGALoader_version={openfpga_version}\n"
                    "yosys_path=/opt/oss-cad-suite/bin/yosys\n"
                    "yosys_version=Yosys 0.64\n"
                    "nextpnr-himbaechel_path=/opt/oss-cad-suite/bin/nextpnr-himbaechel\n"
                    "nextpnr-himbaechel_version=nextpnr-himbaechel 0.8\n"
                    "gmpack_path=/opt/oss-cad-suite/bin/gmpack\n"
                    "gmpack_version=gmpack 2026.04\n"
                    "lsusb_path=/usr/bin/lsusb\nlsusb_version=lsusb (usbutils) 018\n"
                ),
            )
        ],
        mod.GATEMATE_USB_COMMAND: [
            _probe(mod.GATEMATE_USB_COMMAND, stdout="1209:c0ca DirtyJTAG\n")
        ],
        mod.POLARFIRE_USB_COMMAND: [
            _probe(mod.POLARFIRE_USB_COMMAND, stdout="1514:2008 FlashPro5\n")
        ],
        mod.GPU_CONTEXT_COMMAND: [
            _probe(
                mod.GPU_CONTEXT_COMMAND,
                stdout=("NVIDIA GeForce RTX 3090, 24576 MiB\nNVIDIA GeForce RTX 3090, 24576 MiB\n"),
            )
        ],
    }


def _runner(
    *,
    kv260_exit: int = 255,
    kv260_stdout: str = "",
    kv260_stderr: str = "ssh: Could not resolve hostname kv260.local: Name or service not known\n",
    polarfire_status_exit: int = 0,
    polarfire_status_stdout: str = (
        "hostname=mpfs-disco-kit\n"
        "uname=Linux mpfs-disco-kit 6.18.17-linux4microchip-2026.04.1 riscv64\n"
        "python=Python 3.12.12\n"
    ),
    polarfire_status_stderr: str = "",
    polarfire_workload_stdout: list[str] | None = None,
    polarfire_workload_exit: int = 0,
    gatemate_path_available: bool = False,
    gatemate_detect_exit: int = 0,
    gatemate_detect_stdout: str = "GateMate Series GM1Ax IDCODE 0x20000001\n",
    openfpga_present: bool = True,
) -> RecordingRunner:
    probes = _base_probes(openfpga_present=openfpga_present)
    probes[mod.KV260_SSH_CONFIG_COMMAND] = [
        _probe(
            mod.KV260_SSH_CONFIG_COMMAND,
            stdout="hostname kv260.local\nuser xilinx\nport 22\n",
        )
    ]
    probes[mod.KV260_DNS_COMMAND] = [
        _probe(
            mod.KV260_DNS_COMMAND,
            exit_code=2,
            stderr="getent: ahosts kria: Name or service not known\n",
        )
    ]
    probes[mod.KV260_SSH_TRUE_COMMAND] = [
        _probe(
            mod.KV260_SSH_TRUE_COMMAND,
            exit_code=kv260_exit,
            stdout=kv260_stdout,
            stderr=kv260_stderr,
        )
    ]
    probes[mod.POLARFIRE_STATUS_COMMAND] = [
        _probe(
            mod.POLARFIRE_STATUS_COMMAND,
            exit_code=polarfire_status_exit,
            stdout=polarfire_status_stdout,
            stderr=polarfire_status_stderr,
        )
    ]
    if polarfire_status_exit == 0:
        outputs = polarfire_workload_stdout or [
            _workload_stdout(wall_time_s=0.10),
            _workload_stdout(wall_time_s=0.12),
            _workload_stdout(wall_time_s=0.11),
        ]
        probes[mod.POLARFIRE_WORKLOAD_COMMAND] = [
            _probe(
                mod.POLARFIRE_WORKLOAD_COMMAND,
                exit_code=polarfire_workload_exit,
                stdout=stdout,
                stderr="" if polarfire_workload_exit == 0 else "workload failed\n",
            )
            for stdout in outputs
        ]
    if gatemate_path_available and openfpga_present:
        probes[mod.GATEMATE_DETECT_COMMAND] = [
            _probe(
                mod.GATEMATE_DETECT_COMMAND,
                exit_code=gatemate_detect_exit,
                stdout=gatemate_detect_stdout,
                stderr="" if gatemate_detect_exit == 0 else "dirtyJtag open failed\n",
            )
        ]
    return RecordingRunner(probes)


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def _tests_run() -> list[dict[str, object]]:
    return [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5398_hardware_evidence_graph_repeatability_v491.py -q",
            "outcome": "passed in test fixture",
        }
    ]


def test_req_hw_5398_spec_declares_evidence_graph_contract() -> None:
    """REQ-HW-5398: OpenSpec anchors the v491 evidence graph contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5398") : spec.index("### SCENARIO-HW-4910")]

    for marker in (
        "REQ-HW-5398",
        "SCENARIO-HW-5398",
        str(mod.RESULT_RELATIVE_PATH),
        "command nodes",
        "observation nodes",
        "verification nodes",
        mod.KV260_REQUIRED_COMMAND_FORM,
        "evidence_graph_path",
        "evidence_graph_hash",
        "offline_verifier_path",
        "offline_verifier_passed",
        "polar_fire_repeat_count",
        "polar_fire_timing_variance",
        "kv260_reachability",
        "gatemate_workload_path_available",
        "repeatability_evidence_present",
        "hardware_speedup_claim",
        "destructive_action_taken",
        "honest_verdict",
    ):
        assert marker in section


def test_scenario_hw_5398_builds_verified_repeatability_graph() -> None:
    """SCENARIO-HW-5398: repeated PolarFire workload creates verified evidence."""

    runner = _runner()
    artifact, graph = mod.build_evidence_bundle(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert runner.commands == [
        mod.HOST_DATE_COMMAND,
        mod.HARDWARE_ENV_COMMAND,
        mod.TOOL_VERSION_COMMAND,
        mod.GATEMATE_USB_COMMAND,
        mod.POLARFIRE_USB_COMMAND,
        mod.GPU_CONTEXT_COMMAND,
        mod.KV260_SSH_CONFIG_COMMAND,
        mod.KV260_DNS_COMMAND,
        mod.KV260_SSH_TRUE_COMMAND,
        mod.POLARFIRE_STATUS_COMMAND,
        mod.POLARFIRE_WORKLOAD_COMMAND,
        mod.POLARFIRE_WORKLOAD_COMMAND,
        mod.POLARFIRE_WORKLOAD_COMMAND,
    ]
    assert mod.GATEMATE_DETECT_COMMAND not in runner.commands
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "milestone") == "2026.07.491"
    assert _value(artifact, "boards_checked") == ["KV260", "PolarFire", "GateMate"]
    assert _value(artifact, "evidence_graph_path") == mod.EVIDENCE_GRAPH_RELATIVE_PATH.as_posix()
    assert _value(artifact, "evidence_graph_hash") == mod.graph_content_hash(graph)
    assert _value(artifact, "offline_verifier_path") == mod.OFFLINE_VERIFIER_PATH
    assert _value(artifact, "offline_verifier_passed") is True
    assert _value(artifact, "polar_fire_repeat_count") == 3
    assert _value(artifact, "polar_fire_timing_variance") == pytest.approx(0.000066666667)
    assert _value(artifact, "repeatability_evidence_present") is True
    assert _value(artifact, "hardware_speedup_claim") is False
    assert _value(artifact, "destructive_action_taken") is False
    assert _value(artifact, "gatemate_workload_path_available") is False
    assert _value(artifact, "kv260_reachability")["status"] == "unreachable"
    assert _value(artifact, "kv260_reachability")["ssh_alias"]["hostname"] == "kv260.local"
    assert "complete:" in _value(artifact, "honest_verdict")
    assert "hardware_speedup_claim=false" in _value(artifact, "honest_verdict")
    assert graph["schema"] == mod.GRAPH_SCHEMA
    assert graph["graph_hash"] == mod.graph_content_hash(graph)
    assert {node["node_type"] for node in graph["nodes"]} == {
        "command",
        "observation",
        "verification",
    }
    assert any(
        node["node_type"] == "observation"
        and node["board"] == "PolarFire"
        and node["reproducibility_class"] == "repeatable_board_local_same_output_timing"
        for node in graph["nodes"]
    )
    assert mod.verify_evidence_graph(graph).passed is True
    assert "do-not-record" not in json.dumps(artifact)
    assert "also-hidden" not in json.dumps(artifact)
    assert "PRIVATE_TOKEN" not in json.dumps(artifact)
    assert "/dev/mmcblk" not in json.dumps(artifact).lower()
    mod.validate_artifact(artifact, graph)


def test_polarfire_unreachable_keeps_complete_receipt_without_repeatability() -> None:
    """REQ-HW-5398: complete graph receipts can honestly preserve blockers."""

    runner = _runner(
        polarfire_status_exit=255,
        polarfire_status_stdout="",
        polarfire_status_stderr="ssh: connect to host polarfire port 22: No route to host\n",
    )
    artifact, graph = mod.build_evidence_bundle(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert mod.POLARFIRE_WORKLOAD_COMMAND not in runner.commands
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "polar_fire_repeat_count") == 0
    assert _value(artifact, "polar_fire_timing_variance") is None
    assert _value(artifact, "repeatability_evidence_present") is False
    assert artifact["board_details"]["PolarFire"]["reproducibility_class"] == (
        "blocked_polarfire_ssh_unreachable"
    )
    assert _value(artifact, "hardware_speedup_claim") is False
    mod.validate_artifact(artifact, graph)


def test_mismatched_polarfire_output_hash_blocks_repeatability_class() -> None:
    """REQ-HW-5398: repeat timing is not reproducible when output hashes drift."""

    runner = _runner(
        polarfire_workload_stdout=[
            _workload_stdout(wall_time_s=0.10),
            _workload_stdout(output_sha256="0" * 64, wall_time_s=0.12),
            _workload_stdout(wall_time_s=0.11),
        ]
    )
    artifact, graph = mod.build_evidence_bundle(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert _value(artifact, "polar_fire_repeat_count") == 3
    assert _value(artifact, "repeatability_evidence_present") is False
    assert artifact["board_details"]["PolarFire"]["reproducibility_class"] == (
        "non_reproducible_output_hash_drift"
    )
    assert artifact["blocked_reason"]["PolarFire"]["reason"] == "output_sha256 mismatch"
    assert _value(artifact, "hardware_speedup_claim") is False
    mod.validate_artifact(artifact, graph)


def test_invalid_polarfire_repeats_are_classified_without_repeatability() -> None:
    """REQ-HW-5398: malformed board-local receipts do not become repeat evidence."""

    bad_receipt, bad_error = mod.parse_polarfire_workload_stdout(
        "\nnot json\n"
        + json.dumps(
            {
                "hostname": "",
                "input_sha256": "1" * 64,
                "output_sha256": mod.POLARFIRE_EXPECTED_OUTPUT_SHA256,
                "python_version": 312,
                "uname": "Linux mpfs-disco-kit riscv64",
                "wall_time_s": -1.0,
            },
            sort_keys=True,
        )
    )
    assert isinstance(bad_receipt, dict)
    assert bad_error is not None
    assert "hostname missing" in bad_error
    assert "input_sha256 mismatch" in bad_error
    assert "wall_time_s invalid" in bad_error
    assert "python_version invalid" in bad_error
    assert mod.parse_polarfire_workload_stdout("not json\n") == (
        None,
        "workload stdout is not valid JSON",
    )

    runner = _runner(polarfire_workload_stdout=["not json\n", "not json\n", "not json\n"])
    artifact, graph = mod.build_evidence_bundle(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert _value(artifact, "polar_fire_repeat_count") == 3
    assert _value(artifact, "polar_fire_timing_variance") is None
    assert _value(artifact, "repeatability_evidence_present") is False
    assert artifact["board_details"]["PolarFire"]["reproducibility_class"] == (
        "insufficient_valid_board_local_repeats"
    )
    assert artifact["blocked_reason"]["PolarFire"]["reason"] == (
        "workload stdout is not valid JSON"
    )
    mod.validate_artifact(artifact, graph)


def test_gatemate_detect_is_non_destructive_and_still_not_a_workload_path() -> None:
    """SCENARIO-HW-5398: GateMate detect is separate from workload availability."""

    runner = _runner(kv260_exit=0, kv260_stderr="", gatemate_path_available=True)
    artifact, graph = mod.build_evidence_bundle(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        gatemate_physical_path_available=True,
        tests_run=_tests_run(),
    )

    assert mod.GATEMATE_DETECT_COMMAND in runner.commands
    assert _value(artifact, "kv260_reachability")["status"] == "reachable"
    assert artifact["board_details"]["GateMate"]["status"] == "blocked_physical_or_jtag"
    assert artifact["board_details"]["GateMate"]["jtag_detect_status"] == "detected"
    assert _value(artifact, "gatemate_workload_path_available") is False
    assert _value(artifact, "destructive_action_taken") is False
    assert any(
        node["node_type"] == "command"
        and node["board"] == "GateMate"
        and "openFPGALoader" in node["command"]
        for node in graph["nodes"]
    )
    mod.validate_artifact(artifact, graph)


def test_helpers_and_verifier_reject_tampered_graphs() -> None:
    """REQ-HW-5398: offline verifier fails closed on graph drift and unsafe evidence."""

    artifact, graph = mod.build_evidence_bundle(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    assert mod.parse_ssh_alias_config("hostname kria\nport 22\nuser xilinx\n") == {
        "hostname": "kria",
        "port": "22",
        "user": "xilinx",
    }
    assert mod.timing_variance([0.10, 0.12, 0.11]) == pytest.approx(0.000066666667)
    assert mod.timing_variance([0.10]) is None

    tampered = json.loads(json.dumps(graph))
    tampered["nodes"][0]["node_hash"] = "f" * 64
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert "node_hash mismatch" in result.errors[0]

    tampered = json.loads(json.dumps(graph))
    tampered["edges"].append({"from": "missing", "to": "also-missing", "relation": "bad"})
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert "unknown edge endpoint" in result.errors[0]

    tampered = json.loads(json.dumps(graph))
    first_command = next(node for node in tampered["nodes"] if node["node_type"] == "command")
    first_command["command"] = "openFPGALoader --write flash.bit"
    first_command["node_hash"] = mod.node_content_hash(first_command)
    tampered["graph_hash"] = mod.graph_content_hash(tampered)
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert "destructive command" in result.errors[0]

    tampered = json.loads(json.dumps(graph))
    kv_obs = next(
        node
        for node in tampered["nodes"]
        if node["node_type"] == "observation" and node["board"] == "KV260"
    )
    kv_obs["board_state"]["bad_evidence"] = "/dev/mmcblk0"
    kv_obs["board_state_hash"] = mod.sha256_json(kv_obs["board_state"])
    kv_obs["node_hash"] = mod.node_content_hash(kv_obs)
    tampered["graph_hash"] = mod.graph_content_hash(tampered)
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert "host block-device" in result.errors[0]

    empty = {"schema": "bad", "nodes": [], "edges": "bad", "graph_hash": "bad"}
    result = mod.verify_evidence_graph(empty)
    assert result.passed is False
    assert "schema mismatch" in result.errors
    assert "nodes missing" in result.errors
    assert "edges missing" in result.errors
    assert "command node missing" in result.errors

    tampered = json.loads(json.dumps(graph))
    tampered["nodes"][0] = "not a node"
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert "node[0] not mapping" in result.errors

    tampered = json.loads(json.dumps(graph))
    tampered["nodes"][0].pop("id")
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert "node[0] missing id" in result.errors

    tampered = json.loads(json.dumps(graph))
    tampered["nodes"][1]["id"] = tampered["nodes"][0]["id"]
    tampered["nodes"][1]["node_hash"] = mod.node_content_hash(tampered["nodes"][1])
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert any("duplicate node id" in error for error in result.errors)

    tampered = json.loads(json.dumps(graph))
    obs = next(node for node in tampered["nodes"] if node["node_type"] == "observation")
    obs.pop("board_state")
    obs["node_hash"] = mod.node_content_hash(obs)
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert any("board_state missing" in error for error in result.errors)

    tampered = json.loads(json.dumps(graph))
    obs = next(node for node in tampered["nodes"] if node["node_type"] == "observation")
    obs["board_state_hash"] = "0" * 64
    obs["node_hash"] = mod.node_content_hash(obs)
    tampered["graph_hash"] = mod.graph_content_hash(tampered)
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert any("board_state_hash mismatch" in error for error in result.errors)

    tampered = json.loads(json.dumps(graph))
    cmd = next(node for node in tampered["nodes"] if node["node_type"] == "command")
    cmd["input_hash"] = "short"
    cmd["node_hash"] = mod.node_content_hash(cmd)
    tampered["graph_hash"] = mod.graph_content_hash(tampered)
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert any("input_hash invalid" in error for error in result.errors)

    tampered = json.loads(json.dumps(graph))
    tampered["nodes"] = [
        node for node in tampered["nodes"] if node["node_type"] != "verification"
    ]
    tampered["edges"] = []
    tampered["graph_hash"] = mod.graph_content_hash(tampered)
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert "verification node missing" in result.errors

    tampered = json.loads(json.dumps(graph))
    tampered["edges"].append("not an edge")
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert "edge not mapping" in result.errors

    tampered = json.loads(json.dumps(graph))
    tampered["graph_hash"] = "0" * 64
    result = mod.verify_evidence_graph(tampered)
    assert result.passed is False
    assert "graph_hash mismatch" in result.errors

    artifact["hardware_speedup_claim"]["value"] = True
    with pytest.raises(AssertionError, match="hardware_speedup_claim"):
        mod.validate_artifact(artifact, graph)


def test_run_experiment_writes_artifact_and_graph(tmp_path: Path) -> None:
    """SCENARIO-HW-5398: run_experiment writes stable v491 JSON artifacts."""

    artifact_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    graph_path = tmp_path / mod.EVIDENCE_GRAPH_RELATIVE_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    assert artifact_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert graph_path.exists()
    assert _value(artifact, "evidence_graph_hash") == mod.graph_content_hash(graph)
    assert artifact["spec_refs"] == list(mod.SPEC_REFS)
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert _value(artifact, "offline_verifier_passed") is True
    mod.validate_artifact(artifact, graph)


def test_default_tests_run_keeps_cli_bundle_valid() -> None:
    """REQ-HW-5398: CLI-style artifacts still record verification provenance."""

    artifact, graph = mod.build_evidence_bundle(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
    )

    assert _value(artifact, "tests_run") == [
        {
            "command": "verification not yet attached at artifact generation",
            "outcome": "pending_external_test_run",
        }
    ]
    mod.validate_artifact(artifact, graph)

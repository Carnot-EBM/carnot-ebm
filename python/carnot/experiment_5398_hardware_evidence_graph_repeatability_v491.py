#!/usr/bin/env python3
"""Exp 5398: hash-linked hardware evidence graph with repeatability classes.

Spec refs: REQ-HW-5398, SCENARIO-HW-5398.

This module records what the bench can honestly prove about the active FPGA
boards. The graph form is intentionally more explicit than a flat receipt:
commands, board observations, and verification status are separate hash-linked
nodes. That makes each blocked board and each repeated board-local timing run
auditable without turning continuity evidence into an acceleration claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5293_hardware_continuity_reachability_v483 as base
from carnot import experiment_5374_hardware_continuity_receipts_v489 as receipts
from carnot import experiment_5386_hardware_hashchain_receipts_v490 as prev


JsonDict = dict[str, Any]
Clock = prev.Clock
CommandProbe = prev.CommandProbe
CommandRunner = prev.CommandRunner

RUN_DATE = "20260708"
EXPERIMENT_ID = "exp5398-hardware-evidence-graph-repeatability-v491"
EXPERIMENT_NAME = "experiment_5398_hardware_evidence_graph_repeatability"
MILESTONE = "2026.07.491"
SCHEMA = "carnot.experiment_5398.hardware_evidence_graph_repeatability.v491"
GRAPH_SCHEMA = "carnot.hardware.evidence_graph.v1"
SPEC_REFS = ("REQ-HW-5398", "SCENARIO-HW-5398")
RANDOM_SEED = 5398
INFERENCE_SUBSTRATE = "hardware_evidence_graph_repeatability_no_speedup"
HARDWARE_EVIDENCE_LEVEL = "hash_linked_evidence_graph_repeatability_no_speedup"
GENESIS_NODE_HASH = "0" * 64
OFFLINE_VERIFIER_PATH = (
    "python/carnot/experiment_5398_hardware_evidence_graph_repeatability_v491.py"
)

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5398_hardware_evidence_graph_repeatability_v491.json"
)
EVIDENCE_GRAPH_RELATIVE_PATH = Path(
    "results/experiment_5398_hardware_evidence_graph_repeatability_v491.graph.json"
)
PRIOR_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5386_hardware_hashchain_receipts_v490.json"
)

HOST_DATE_COMMAND = prev.HOST_DATE_COMMAND
HARDWARE_ENV_COMMAND = prev.HARDWARE_ENV_COMMAND
TOOL_VERSION_COMMAND = prev.TOOL_VERSION_COMMAND
GATEMATE_USB_COMMAND = prev.GATEMATE_USB_COMMAND
POLARFIRE_USB_COMMAND = prev.POLARFIRE_USB_COMMAND
GPU_CONTEXT_COMMAND = prev.GPU_CONTEXT_COMMAND
KV260_SSH_TRUE_COMMAND = prev.KV260_SSH_TRUE_COMMAND
KV260_REQUIRED_COMMAND_FORM = prev.KV260_REQUIRED_COMMAND_FORM
POLARFIRE_STATUS_COMMAND = prev.POLARFIRE_STATUS_COMMAND
GATEMATE_DETECT_COMMAND = prev.GATEMATE_DETECT_COMMAND

KV260_SSH_CONFIG_COMMAND = ("ssh", "-G", "kria")
KV260_DNS_COMMAND = ("getent", "hosts", "kria")
LOCAL_TIMEOUT_S = prev.LOCAL_TIMEOUT_S
SSH_TIMEOUT_S = prev.SSH_TIMEOUT_S
GATEMATE_TIMEOUT_S = prev.GATEMATE_TIMEOUT_S
POLARFIRE_REPEAT_TARGET = 3
BOARDS_CHECKED = ("KV260", "PolarFire", "GateMate")
TERMINAL_PREFIXES = ("complete:", "blocked:")

POLARFIRE_WORKLOAD_INPUT = b"carnot-exp5398-polarfire-repeatability-v491\n"
POLARFIRE_WORKLOAD_OUTPUT_SUFFIX = b"|polarfire-v491-output"
POLARFIRE_EXPECTED_INPUT_SHA256 = hashlib.sha256(POLARFIRE_WORKLOAD_INPUT).hexdigest()
POLARFIRE_EXPECTED_OUTPUT_SHA256 = hashlib.sha256(
    POLARFIRE_WORKLOAD_INPUT + POLARFIRE_WORKLOAD_OUTPUT_SUFFIX
).hexdigest()
POLARFIRE_WORKLOAD_PYTHON = (
    "import hashlib,json,platform,socket,time;"
    "started=time.perf_counter();"
    'payload=b"carnot-exp5398-polarfire-repeatability-v491\\n";'
    'out=hashlib.sha256(payload+b"|polarfire-v491-output").hexdigest();'
    "receipt={"
    '"hostname":socket.gethostname(),'
    '"uname":" ".join(platform.uname()),'
    '"python_version":platform.python_version(),'
    '"input_sha256":hashlib.sha256(payload).hexdigest(),'
    '"output_sha256":out,'
    '"wall_time_s":round(time.perf_counter()-started,6)'
    "};"
    "print(json.dumps(receipt,sort_keys=True))"
)
POLARFIRE_WORKLOAD_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    f"python3 -c '{POLARFIRE_WORKLOAD_PYTHON}'",
)

REQUIRED_WRAPPED_FIELDS = (
    "status",
    "milestone",
    "boards_checked",
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
    "tests_run",
    "preconditions_checked",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Complete if the v491 receipt and graph were emitted, even when boards are blocked honestly.",
    "milestone": "Pins this receipt to 2026.07.491 so hardware state is not treated as floating.",
    "boards_checked": "Lists the active hardware lanes classified by this receipt.",
    "evidence_graph_path": "Names the separate hash-linked evidence graph emitted with the artifact.",
    "evidence_graph_hash": "SHA-256 of graph content, excluding only its self-hash field.",
    "offline_verifier_path": "Identifies the offline verifier used to recompute graph hashes and safety gates.",
    "offline_verifier_passed": "True only if the graph verifier passed on the emitted graph.",
    "polar_fire_repeat_count": "Counts safe PolarFire board-local workload attempts when SSH is reachable.",
    "polar_fire_timing_variance": "Measured variance across repeated board-local timings, or null when unavailable.",
    "kv260_reachability": "Classifies KV260 as reachable, unreachable, or not_checked with SSH/DNS reason.",
    "gatemate_workload_path_available": "True only when GateMate has an actual non-destructive workload path.",
    "repeatability_evidence_present": "True only if repeated board-local timing and stable output hashes exist.",
    "hardware_speedup_claim": "False unless repeated board-local timing and a same-workload speedup comparison exist.",
    "destructive_action_taken": "False because this experiment runs no flash, program, dd, or storage-write command.",
    "honest_verdict": "One-line summary starting with complete: or blocked: for conductor reconciliation.",
    "tests_run": "Records verification commands without treating them as hardware evidence.",
    "preconditions_checked": "Records date, sanitized environment, local tools, USB visibility, and SSH targets.",
}


@dataclass(frozen=True)
class GraphVerification:
    """Offline graph verifier result.

    The verifier reports all discovered errors instead of failing on the first
    one because evidence graphs are audit artifacts. A full error list tells a
    future operator whether the problem is a broken hash link, unsafe command,
    missing node, or graph-level hash mismatch.
    """

    passed: bool
    errors: list[str]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wrap_field(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def sha256_text(text: str) -> str:
    return receipts.sha256_text(text)


def sha256_json(payload: Mapping[str, Any]) -> str:
    return base.sha256_json(payload)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def combined_output_sha256(command: Mapping[str, Any]) -> str:
    return prev.combined_output_sha256(command)


def node_content_hash(node: Mapping[str, Any]) -> str:
    stable = dict(node)
    stable["node_hash"] = ""
    return sha256_json(stable)


def graph_content_hash(graph: Mapping[str, Any]) -> str:
    stable = dict(graph)
    stable["graph_hash"] = ""
    return sha256_json(stable)


def timing_variance(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return round(sum((value - mean) ** 2 for value in values) / len(values), 12)


def parse_ssh_alias_config(stdout: str) -> JsonDict:
    parsed: JsonDict = {}
    for line in stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] in {"hostname", "user", "port"}:
            parsed[parts[0]] = parts[1]
    return parsed


def parse_polarfire_workload_stdout(stdout: str) -> tuple[JsonDict | None, str | None]:
    parsed: Any | None = None
    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            break
    if not isinstance(parsed, Mapping):
        return None, "workload stdout is not valid JSON"

    receipt: JsonDict = {
        "hostname": parsed.get("hostname"),
        "uname": parsed.get("uname"),
        "python_version": parsed.get("python_version"),
        "input_sha256": parsed.get("input_sha256"),
        "output_sha256": parsed.get("output_sha256"),
        "wall_time_s": parsed.get("wall_time_s"),
        "workload_class": "deterministic_sha256_transform",
        "board_local": True,
        "speedup_claimed": False,
    }
    errors: list[str] = []
    for key in ("hostname", "uname", "input_sha256", "output_sha256"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            errors.append(f"{key} missing")
    if receipt.get("input_sha256") != POLARFIRE_EXPECTED_INPUT_SHA256:
        errors.append("input_sha256 mismatch")
    if receipt.get("output_sha256") != POLARFIRE_EXPECTED_OUTPUT_SHA256:
        errors.append("output_sha256 mismatch")
    if not isinstance(receipt.get("wall_time_s"), int | float) or receipt["wall_time_s"] < 0:
        errors.append("wall_time_s invalid")
    if receipt.get("python_version") is not None and not isinstance(
        receipt.get("python_version"), str
    ):
        errors.append("python_version invalid")
    return receipt, "; ".join(errors) if errors else None


def default_tests_run() -> list[JsonDict]:
    return [
        {
            "command": "verification not yet attached at artifact generation",
            "outcome": "pending_external_test_run",
        }
    ]


def _command_receipt(
    *,
    probe: CommandProbe,
    timeout_s: float,
    kind: str,
    outcome: str,
    command_override: str | None = None,
) -> JsonDict:
    return receipts.command_receipt(
        probe=probe,
        timeout_s=timeout_s,
        kind=kind,
        outcome=outcome,
        command_override=command_override,
    )


def kv260_reachability(
    *, command_runner: CommandRunner
) -> tuple[JsonDict, JsonDict | None, list[JsonDict]]:
    alias_probe = command_runner(KV260_SSH_CONFIG_COMMAND, LOCAL_TIMEOUT_S)
    dns_probe = command_runner(KV260_DNS_COMMAND, LOCAL_TIMEOUT_S)
    ssh_probe = command_runner(KV260_SSH_TRUE_COMMAND, SSH_TIMEOUT_S)
    reachable = ssh_probe.exit_code == 0
    reason = None if reachable else (ssh_probe.stderr.strip() or "ssh reachability failed")
    blocker = None
    if not reachable:
        blocker = receipts.blocker_from_probe(
            "unreachable",
            ssh_probe,
            SSH_TIMEOUT_S,
            command_override=KV260_REQUIRED_COMMAND_FORM,
        )
    detail: JsonDict = {
        "board": "KV260",
        "status": "reachable" if reachable else "unreachable",
        "reason": reason,
        "check_method": "ssh_batchmode_true_only",
        "command_form": KV260_REQUIRED_COMMAND_FORM,
        "ssh_alias": parse_ssh_alias_config(alias_probe.stdout),
        "dns": {
            "command": base.command_to_string(KV260_DNS_COMMAND),
            "exit_code": int(dns_probe.exit_code),
            "resolved": dns_probe.exit_code == 0,
            "identifier": base.remote_identifier(dns_probe.combined_output)
            if dns_probe.combined_output
            else None,
            "stdout_sha256": sha256_text(dns_probe.stdout),
            "stderr_sha256": sha256_text(dns_probe.stderr),
        },
        "ssh": {
            "exit_code": int(ssh_probe.exit_code),
            "reachable": reachable,
            "identifier": base.remote_identifier(ssh_probe.combined_output)
            if ssh_probe.combined_output
            else None,
            "stdout_sha256": sha256_text(ssh_probe.stdout),
            "stderr_sha256": sha256_text(ssh_probe.stderr),
        },
        "blocked_reason": blocker,
        "speedup_claimed": False,
    }
    commands = [
        _command_receipt(
            probe=alias_probe,
            timeout_s=LOCAL_TIMEOUT_S,
            kind="kv260_ssh_alias_config",
            outcome="recorded" if alias_probe.exit_code == 0 else "alias_unavailable",
        ),
        _command_receipt(
            probe=dns_probe,
            timeout_s=LOCAL_TIMEOUT_S,
            kind="kv260_dns_alias_lookup",
            outcome="resolved" if dns_probe.exit_code == 0 else "unresolved",
        ),
        _command_receipt(
            probe=ssh_probe,
            timeout_s=SSH_TIMEOUT_S,
            kind="kv260_ssh_true_reachability_probe",
            outcome=detail["status"],
            command_override=KV260_REQUIRED_COMMAND_FORM,
        ),
    ]
    return detail, blocker, commands


def polarfire_repeatability(
    *, command_runner: CommandRunner
) -> tuple[JsonDict, JsonDict | None, list[JsonDict]]:
    status_probe = command_runner(POLARFIRE_STATUS_COMMAND, SSH_TIMEOUT_S)
    reachable = status_probe.exit_code == 0
    commands = [
        _command_receipt(
            probe=status_probe,
            timeout_s=SSH_TIMEOUT_S,
            kind="polarfire_authenticated_status_probe",
            outcome="reachable" if reachable else "unreachable",
        )
    ]
    base_detail: JsonDict = {
        "board": "PolarFire",
        "status": "reachable" if reachable else "unreachable",
        "ssh_reachable": reachable,
        "probe_exit_code": int(status_probe.exit_code),
        "remote_identifier": base.remote_identifier(status_probe.combined_output)
        if reachable
        else None,
        "repeat_target": POLARFIRE_REPEAT_TARGET,
        "repeat_count": 0,
        "valid_repeat_count": 0,
        "timing_variance": None,
        "output_hashes": [],
        "workload_attempts": [],
        "reproducibility_class": "blocked_polarfire_ssh_unreachable",
        "speedup_claimed": False,
    }
    if not reachable:
        return (
            base_detail,
            receipts.blocker_from_probe("unreachable", status_probe, SSH_TIMEOUT_S),
            commands,
        )

    attempts: list[JsonDict] = []
    blocker: JsonDict | None = None
    for index in range(POLARFIRE_REPEAT_TARGET):
        probe = command_runner(POLARFIRE_WORKLOAD_COMMAND, SSH_TIMEOUT_S)
        receipt, parse_error = parse_polarfire_workload_stdout(probe.stdout)
        valid = probe.exit_code == 0 and receipt is not None and parse_error is None
        attempts.append(
            {
                "index": index + 1,
                "exit_code": int(probe.exit_code),
                "valid": valid,
                "parse_error": parse_error,
                "receipt": receipt,
                "wall_time_s": receipt.get("wall_time_s") if isinstance(receipt, Mapping) else None,
                "input_sha256": receipt.get("input_sha256") if isinstance(receipt, Mapping) else None,
                "output_sha256": receipt.get("output_sha256") if isinstance(receipt, Mapping) else None,
            }
        )
        if not valid and blocker is None:
            blocker = receipts.blocker_from_probe(
                parse_error or probe.stderr.strip() or "workload command failed",
                probe,
                SSH_TIMEOUT_S,
            )
        commands.append(
            _command_receipt(
                probe=probe,
                timeout_s=SSH_TIMEOUT_S,
                kind=f"polarfire_board_local_workload_repeat_{index + 1}",
                outcome="valid_repeat" if valid else "invalid_repeat",
            )
        )

    timing_values = [
        float(attempt["wall_time_s"])
        for attempt in attempts
        if isinstance(attempt.get("wall_time_s"), int | float)
    ]
    output_hashes = [
        str(attempt["output_sha256"])
        for attempt in attempts
        if isinstance(attempt.get("output_sha256"), str)
    ]
    valid_attempts = [attempt for attempt in attempts if attempt["valid"] is True]
    stable_outputs = len(set(output_hashes)) == 1 and output_hashes == [
        POLARFIRE_EXPECTED_OUTPUT_SHA256
    ] * POLARFIRE_REPEAT_TARGET
    repeatability_present = len(valid_attempts) >= POLARFIRE_REPEAT_TARGET and stable_outputs
    if repeatability_present:
        repeatability_class = "repeatable_board_local_same_output_timing"
    elif len(output_hashes) >= 2 and len(set(output_hashes)) > 1:
        repeatability_class = "non_reproducible_output_hash_drift"
    else:
        repeatability_class = "insufficient_valid_board_local_repeats"
    base_detail.update(
        {
            "status": "reachable/repeated_workload",
            "repeat_count": len(attempts),
            "valid_repeat_count": len(valid_attempts),
            "timing_variance": timing_variance(timing_values),
            "output_hashes": output_hashes,
            "workload_attempts": attempts,
            "workload_validated": repeatability_present,
            "reproducibility_class": repeatability_class,
        }
    )
    return base_detail, blocker, commands


def gatemate_status(
    *,
    command_runner: CommandRunner,
    context: Mapping[str, Any],
    gatemate_physical_path_available: bool | None,
) -> tuple[JsonDict, JsonDict | None, list[JsonDict], bool]:
    path_available = receipts.gatemate_path_available_from_context(
        context, gatemate_physical_path_available
    )
    detail_status, detail, blocker, commands = receipts.gatemate_status_from_context(
        command_runner=command_runner,
        context=context,
        physical_path_available=path_available,
    )
    detail = dict(detail)
    workload_path_available = False
    detail["real_workload_path_available"] = False
    if detail_status == "detected":
        detail["jtag_detect_status"] = "detected"
        detail["status"] = "blocked_physical_or_jtag"
        detail["reason"] = "dirtyjtag_detected_but_no_real_workload_path"
        blocker = {"reason": "dirtyjtag_detected_but_no_real_workload_path"}
    return detail, blocker, commands, workload_path_available


def command_board(kind: str) -> str:
    if kind.startswith("kv260"):
        return "KV260"
    if kind.startswith("polarfire"):
        return "PolarFire"
    if kind.startswith("gatemate"):
        return "GateMate"
    return "Host"


def append_node(nodes: list[JsonDict], node: JsonDict) -> JsonDict:
    node["previous_node_hash"] = nodes[-1]["node_hash"] if nodes else GENESIS_NODE_HASH
    node["node_hash"] = node_content_hash(node)
    nodes.append(node)
    return node


def command_node(index: int, command: Mapping[str, Any]) -> JsonDict:
    command_text = str(command.get("command", ""))
    return {
        "id": f"cmd:{index}:{command.get('kind')}",
        "node_type": "command",
        "board": command_board(str(command.get("kind", ""))),
        "kind": command.get("kind"),
        "command": command_text,
        "command_sha256": sha256_text(command_text),
        "input_hash": sha256_text(command_text),
        "output_hash": combined_output_sha256(command),
        "stdout_sha256": command.get("stdout_sha256"),
        "stderr_sha256": command.get("stderr_sha256"),
        "exit_code": command.get("exit_code"),
        "outcome": command.get("outcome"),
        "node_hash": "",
    }


def observation_node(
    *,
    node_id: str,
    board: str,
    board_state: Mapping[str, Any],
    command_node_ids: Sequence[str],
    reproducibility_class: str,
    input_hash: str,
    output_hash: str,
) -> JsonDict:
    return {
        "id": node_id,
        "node_type": "observation",
        "board": board,
        "observed_command_node_ids": list(command_node_ids),
        "board_state": dict(board_state),
        "board_state_hash": sha256_json(dict(board_state)),
        "input_hash": input_hash,
        "output_hash": output_hash,
        "reproducibility_class": reproducibility_class,
        "node_hash": "",
    }


def verification_node(*, checked_node_count: int, checked_edge_count: int) -> JsonDict:
    return {
        "id": "verify:offline:0",
        "node_type": "verification",
        "board": "offline",
        "verifier_path": OFFLINE_VERIFIER_PATH,
        "offline_verifier_status": "passed",
        "checked_node_count": checked_node_count,
        "checked_edge_count": checked_edge_count,
        "graph_hash_algorithm": "sha256_json_canonical_excluding_graph_hash",
        "node_hash": "",
    }


def build_evidence_graph(
    *,
    commands_run: Sequence[Mapping[str, Any]],
    kv260_detail: Mapping[str, Any],
    polarfire_detail: Mapping[str, Any],
    gatemate_detail: Mapping[str, Any],
) -> JsonDict:
    nodes: list[JsonDict] = []
    edges: list[JsonDict] = []
    command_ids_by_board: dict[str, list[str]] = {"KV260": [], "PolarFire": [], "GateMate": []}
    for index, command in enumerate(commands_run):
        node = append_node(nodes, command_node(index, command))
        board = str(node["board"])
        if board in command_ids_by_board:
            command_ids_by_board[board].append(str(node["id"]))

    observations = [
        observation_node(
            node_id="obs:kv260:reachability",
            board="KV260",
            board_state=kv260_detail,
            command_node_ids=command_ids_by_board["KV260"],
            reproducibility_class=str(kv260_detail.get("status", "not_checked")),
            input_hash=sha256_text(KV260_REQUIRED_COMMAND_FORM),
            output_hash=sha256_json(kv260_detail),
        ),
        observation_node(
            node_id="obs:polarfire:repeatability",
            board="PolarFire",
            board_state=polarfire_detail,
            command_node_ids=command_ids_by_board["PolarFire"],
            reproducibility_class=str(polarfire_detail.get("reproducibility_class")),
            input_hash=POLARFIRE_EXPECTED_INPUT_SHA256,
            output_hash=sha256_json(polarfire_detail),
        ),
        observation_node(
            node_id="obs:gatemate:workload_path",
            board="GateMate",
            board_state=gatemate_detail,
            command_node_ids=command_ids_by_board["GateMate"],
            reproducibility_class=str(gatemate_detail.get("status", "blocked_physical_or_jtag")),
            input_hash=sha256_text("gatemate_dirtyjtag_toolchain_state"),
            output_hash=sha256_json(gatemate_detail),
        ),
    ]
    for observation in observations:
        append_node(nodes, observation)
        for command_id in observation["observed_command_node_ids"]:
            edges.append(
                {"from": command_id, "to": observation["id"], "relation": "supports_observation"}
            )
    checked_edge_count = len(edges) + len(observations)
    verifier = append_node(
        nodes,
        verification_node(checked_node_count=len(nodes) + 1, checked_edge_count=checked_edge_count),
    )
    for observation in observations:
        edges.append({"from": observation["id"], "to": verifier["id"], "relation": "verified_by"})

    graph: JsonDict = {
        "schema": GRAPH_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "nodes": nodes,
        "edges": edges,
        "graph_hash": "",
    }
    graph["graph_hash"] = graph_content_hash(graph)
    return graph


def verify_evidence_graph(graph: Mapping[str, Any]) -> GraphVerification:
    errors: list[str] = []
    if graph.get("schema") != GRAPH_SCHEMA:
        errors.append("schema mismatch")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not nodes:
        errors.append("nodes missing")
        nodes = []
    if not isinstance(edges, list):
        errors.append("edges missing")
        edges = []
    previous_hash = GENESIS_NODE_HASH
    node_ids: set[str] = set()
    node_types: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            errors.append(f"node[{index}] not mapping")
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            errors.append(f"node[{index}] missing id")
            continue
        if node_id in node_ids:
            errors.append(f"duplicate node id {node_id}")
        node_ids.add(node_id)
        node_types.add(str(node.get("node_type")))
        if node.get("previous_node_hash") != previous_hash:
            errors.append(f"previous_node_hash mismatch at {node_id}")
        if node.get("node_hash") != node_content_hash(node):
            errors.append(f"node_hash mismatch at {node_id}")
        previous_hash = str(node.get("node_hash"))
        if node.get("node_type") == "command" and receipts.command_is_destructive(
            str(node.get("command", ""))
        ):
            errors.append(f"destructive command at {node_id}")
        if node.get("node_type") == "observation":
            board_state = node.get("board_state")
            if not isinstance(board_state, Mapping):
                errors.append(f"board_state missing at {node_id}")
            elif node.get("board_state_hash") != sha256_json(board_state):
                errors.append(f"board_state_hash mismatch at {node_id}")
            if node.get("board") == "KV260" and not receipts.no_host_mmcblk_kv260_evidence(
                {"board_state": board_state}
            ):
                errors.append(f"host block-device KV260 evidence at {node_id}")
        for key in ("input_hash", "output_hash"):
            if node.get("node_type") in {"command", "observation"}:
                value = node.get(key)
                if not isinstance(value, str) or len(value) != 64:
                    errors.append(f"{key} invalid at {node_id}")
    for required_type in ("command", "observation", "verification"):
        if required_type not in node_types:
            errors.append(f"{required_type} node missing")
    for edge in edges:
        if not isinstance(edge, Mapping):
            errors.append("edge not mapping")
            continue
        if edge.get("from") not in node_ids or edge.get("to") not in node_ids:
            errors.append("unknown edge endpoint")
    if graph.get("graph_hash") != graph_content_hash(graph):
        errors.append("graph_hash mismatch")
    return GraphVerification(passed=not errors, errors=errors)


def destructive_action_taken(commands: Sequence[Mapping[str, Any]]) -> bool:
    return any(receipts.command_is_destructive(str(command.get("command", ""))) for command in commands)


def honest_verdict(
    *,
    kv260_status: str,
    polarfire_repeat_count: int,
    gatemate_workload_path_available: bool,
    repeatability_evidence_present: bool,
    hardware_speedup_claim: bool,
) -> str:
    return (
        "complete: "
        f"kv260={kv260_status} "
        f"polar_fire_repeat_count={polarfire_repeat_count} "
        f"gatemate_workload_path_available={str(gatemate_workload_path_available).lower()} "
        f"repeatability_evidence_present={str(repeatability_evidence_present).lower()} "
        f"hardware_speedup_claim={str(hardware_speedup_claim).lower()}"
    )


def build_evidence_bundle(
    *,
    command_runner: CommandRunner = base.run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    gatemate_physical_path_available: bool | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[JsonDict, JsonDict]:
    started = clock()
    context, commands_run = receipts.collect_preconditions(command_runner)
    context["prior_receipt"] = str(PRIOR_RESULT_RELATIVE_PATH)
    context["ssh_targets"] = {"KV260": "kria", "PolarFire": "polarfire"}

    kv260_detail, kv260_blocker, kv260_commands = kv260_reachability(
        command_runner=command_runner
    )
    commands_run.extend(kv260_commands)
    polarfire_detail, polarfire_blocker, polarfire_commands = polarfire_repeatability(
        command_runner=command_runner
    )
    commands_run.extend(polarfire_commands)
    gatemate_detail, gatemate_blocker, gatemate_commands, gatemate_workload_path = (
        gatemate_status(
            command_runner=command_runner,
            context=context,
            gatemate_physical_path_available=gatemate_physical_path_available,
        )
    )
    commands_run.extend(gatemate_commands)

    graph = build_evidence_graph(
        commands_run=commands_run,
        kv260_detail=kv260_detail,
        polarfire_detail=polarfire_detail,
        gatemate_detail=gatemate_detail,
    )
    verification = verify_evidence_graph(graph)
    repeatability_present = polarfire_detail.get("workload_validated") is True
    hardware_speedup_claim = False
    tests = [dict(item) for item in (tests_run if tests_run is not None else default_tests_run())]

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "duration_s": base.round_duration(clock() - started),
        "commit": commit,
        "status": wrap_field("status", "complete"),
        "milestone": wrap_field("milestone", MILESTONE),
        "boards_checked": wrap_field("boards_checked", list(BOARDS_CHECKED)),
        "evidence_graph_path": wrap_field(
            "evidence_graph_path", EVIDENCE_GRAPH_RELATIVE_PATH.as_posix()
        ),
        "evidence_graph_hash": wrap_field("evidence_graph_hash", graph_content_hash(graph)),
        "offline_verifier_path": wrap_field("offline_verifier_path", OFFLINE_VERIFIER_PATH),
        "offline_verifier_passed": wrap_field("offline_verifier_passed", verification.passed),
        "polar_fire_repeat_count": wrap_field(
            "polar_fire_repeat_count", int(polarfire_detail.get("repeat_count", 0))
        ),
        "polar_fire_timing_variance": wrap_field(
            "polar_fire_timing_variance", polarfire_detail.get("timing_variance")
        ),
        "kv260_reachability": wrap_field("kv260_reachability", kv260_detail),
        "gatemate_workload_path_available": wrap_field(
            "gatemate_workload_path_available", gatemate_workload_path
        ),
        "repeatability_evidence_present": wrap_field(
            "repeatability_evidence_present", repeatability_present
        ),
        "hardware_speedup_claim": wrap_field("hardware_speedup_claim", hardware_speedup_claim),
        "destructive_action_taken": wrap_field(
            "destructive_action_taken", destructive_action_taken(commands_run)
        ),
        "honest_verdict": wrap_field(
            "honest_verdict",
            honest_verdict(
                kv260_status=str(kv260_detail["status"]),
                polarfire_repeat_count=int(polarfire_detail.get("repeat_count", 0)),
                gatemate_workload_path_available=gatemate_workload_path,
                repeatability_evidence_present=repeatability_present,
                hardware_speedup_claim=hardware_speedup_claim,
            ),
        ),
        "tests_run": wrap_field("tests_run", tests),
        "preconditions_checked": wrap_field("preconditions_checked", context),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "hardware_evidence_level": HARDWARE_EVIDENCE_LEVEL,
        "commands_run": commands_run,
        "board_details": {
            "KV260": kv260_detail,
            "PolarFire": polarfire_detail,
            "GateMate": gatemate_detail,
        },
        "blocked_reason": {
            "KV260": kv260_blocker,
            "PolarFire": polarfire_blocker,
            "GateMate": gatemate_blocker,
        },
        "offline_verifier_errors": verification.errors,
        "docs_update_decision": {
            "ops_changelog_updated": False,
            "ops_status_updated": False,
            "reason": "task stop rule delegates docs/status reconciliation to the conductor",
        },
        "conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact, graph)
    return artifact, graph


def validate_wrapped_field(artifact: Mapping[str, Any], field: str) -> Any:
    wrapped = artifact.get(field)
    require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
    require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
    require("value" in wrapped, f"{field} missing value")
    return wrapped["value"]


def validate_tests_run(tests_run: Any) -> None:
    require(isinstance(tests_run, list) and tests_run, "tests_run must be non-empty")
    for index, item in enumerate(tests_run):
        require(isinstance(item, Mapping), f"tests_run[{index}] must be mapping")
        require(isinstance(item.get("command"), str) and item["command"], "test command missing")
        require(isinstance(item.get("outcome"), str) and item["outcome"], "test outcome missing")


def validate_artifact(artifact: Mapping[str, Any], graph: Mapping[str, Any]) -> None:
    for field in REQUIRED_WRAPPED_FIELDS:
        validate_wrapped_field(artifact, field)
    require(artifact.get("schema") == SCHEMA, "schema mismatch")
    require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id mismatch")
    require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs mismatch")
    require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    require(validate_wrapped_field(artifact, "status") == "complete", "status mismatch")
    require(validate_wrapped_field(artifact, "milestone") == MILESTONE, "milestone mismatch")
    require(
        validate_wrapped_field(artifact, "boards_checked") == list(BOARDS_CHECKED),
        "boards_checked mismatch",
    )
    require(
        validate_wrapped_field(artifact, "evidence_graph_path")
        == EVIDENCE_GRAPH_RELATIVE_PATH.as_posix(),
        "evidence_graph_path mismatch",
    )
    require(
        validate_wrapped_field(artifact, "evidence_graph_hash") == graph_content_hash(graph),
        "evidence_graph_hash mismatch",
    )
    require(
        validate_wrapped_field(artifact, "offline_verifier_path") == OFFLINE_VERIFIER_PATH,
        "offline_verifier_path mismatch",
    )
    verification = verify_evidence_graph(graph)
    require(verification.passed is True, f"offline verifier failed: {verification.errors}")
    require(
        validate_wrapped_field(artifact, "offline_verifier_passed") is True,
        "offline_verifier_passed mismatch",
    )
    repeat_count = validate_wrapped_field(artifact, "polar_fire_repeat_count")
    require(isinstance(repeat_count, int) and repeat_count >= 0, "bad repeat count")
    variance = validate_wrapped_field(artifact, "polar_fire_timing_variance")
    require(variance is None or isinstance(variance, int | float), "bad timing variance")
    kv260 = validate_wrapped_field(artifact, "kv260_reachability")
    require(
        isinstance(kv260, Mapping) and kv260.get("status") in {"reachable", "unreachable", "not_checked"},
        "bad kv260_reachability",
    )
    require(
        validate_wrapped_field(artifact, "gatemate_workload_path_available") is False,
        "gatemate workload path unexpectedly available",
    )
    require(
        validate_wrapped_field(artifact, "hardware_speedup_claim") is False,
        "hardware_speedup_claim must be false",
    )
    require(
        validate_wrapped_field(artifact, "destructive_action_taken") is False,
        "destructive_action_taken must be false",
    )
    verdict = validate_wrapped_field(artifact, "honest_verdict")
    require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "bad verdict")
    require("hardware_speedup_claim=false" in verdict, "verdict missing speedup boundary")
    validate_tests_run(validate_wrapped_field(artifact, "tests_run"))
    require(receipts.no_host_mmcblk_kv260_evidence(artifact), "host block-device evidence present")
    require(artifact.get("conductor_modified") is False, "conductor_modified mismatch")
    require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "checksum mismatch",
    )


def write_outputs(repo_root: str | Path, artifact: Mapping[str, Any], graph: Mapping[str, Any]) -> Path:
    validate_artifact(artifact, graph)
    root = Path(repo_root)
    graph_path = root / EVIDENCE_GRAPH_RELATIVE_PATH
    artifact_path = root / RESULT_RELATIVE_PATH
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(dict(graph), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = base.run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str | None = None,
    gatemate_physical_path_available: bool | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    artifact, graph = build_evidence_bundle(
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        commit=commit or base.get_git_commit(repo_root),
        gatemate_physical_path_available=gatemate_physical_path_available,
        tests_run=tests_run,
    )
    return write_outputs(repo_root, artifact, graph)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument(
        "--gatemate-physical-path-available",
        action="store_true",
        help="Run bounded GateMate detect because the current physical/JTAG path is available.",
    )
    args = parser.parse_args(argv)
    print(
        run_experiment(
            repo_root=Path("."),
            run_date=args.date,
            gatemate_physical_path_available=args.gatemate_physical_path_available or None,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""REQ-VERIFY-8231: portable host state does not establish board performance.

The complete probability tree and causal state retain their original ordering.
All measurements here run on the host; device transfer remains a future contract.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.hardware_workload_obligations_8216 import authenticate, storage_probe
from carnot.reporting.kv260_workload_boundary_8230 import freeze as freeze
from carnot.reporting.request_trace_inventory_8200 import operand
from carnot.verify import restricted_action_rule_8207 as rule
from carnot.verify.delayed_utility_learning_8225 import predict

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8231_v711_polarfire_state_boundary"
CLI = "scripts/experiments/" + NAME + ".py"
CONFIG: Json = dict(seed=7118231, envelope_version=1, encoding="canonical UTF-8 JSON, finite FP64")
PRODUCERS = {
    "historical": (8216, "v709_hardware_workload_obligations", "hardware_boundary_ready_score"),
    "kernel": (8221, "v711_utility_kernel", "static_kernel_ready_score"),
    "learning": (8225, "v711_delayed_utility_learning", "utility_trajectory_ready_score"),
}
PINS = {
    8216: "sha256:5b31ca2cca90b76d47f7c1b9575264e352a3990ba303992c8083f97c0fff823f",
    8221: "sha256:5ff6f09312a81ce95b752d38dec94d8a1c57bbf894cde785ebe7494ca7c639e5",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts expose finite work and keep the supervisor informed."""
    print(f"[exp8231] phase={phase} completed={completed} pending={pending}", flush=True)


def load(root: Path, raw: Path) -> Json:
    """Authenticate independent producers so blocked learning loses no board rows."""
    data: Json = dict(checks=[], references=[], cited=[], branches={}, board={}, cases=[])
    for index, (name, (eid, suffix, score)) in enumerate(PRODUCERS.items()):
        progress("authenticate_before_" + name, index, 3 - index)
        path = root / "results" / f"experiment_{eid}_{suffix}.json"
        begin = len(data["checks"])
        try:
            value, valid = authenticate(path, eid, score, raw, data)
        except (AttributeError, TypeError) as error:
            value, valid = {}, False
            data["checks"].append(operand("object_schema", path, "object", str(error)))
        for field, wanted, observed in [
            (
                "sha256",
                PINS.get(eid, sha256_file(path) if path.is_file() else None),
                sha256_file(path) if path.is_file() else None,
            ),
            (
                "task_id",
                f"exp{eid}-" + suffix.split("_", 1)[1].replace("_", "-"),
                value.get("task_id"),
            ),
            ("fixture_mode", False, bool(value.get("fixture_mode") or value.get("fixture"))),
        ]:
            data["checks"].append(operand(field, path, wanted, observed))
        valid = valid and all(c["passed"] for c in data["checks"][begin:])
        try:
            if valid and name == "historical":
                board = next(b for b in value["board_rows"] if b["board"] == "PolarFire")
                for source, pin in [
                    (root / board["source_path"], board["source_hash"]),
                    (Path(board["source_transcript"]), board["source_transcript_sha256"]),
                ]:
                    freeze(checked(dict(path=str(source), sha256=pin)), raw, data)
                for field in ["custody_valid", "actual_substrate"]:
                    data["checks"].append(
                        operand(
                            "board." + field,
                            path,
                            True if field == "custody_valid" else "linux_cpu",
                            board.get(field),
                        )
                    )
                data["board"] = board
            if valid and name == "kernel":
                work = json.loads(
                    freeze(checked(value["measurement_reference"]), raw, data).read_bytes()
                )
                restart = next(
                    r
                    for r in value["raw_shard_hashes"]
                    if Path(r["path"]).name == "restart-input.json"
                )
                rows = json.loads(freeze(checked(restart), raw, data).read_bytes())["rows"]
                final = next(
                    r
                    for r in value["raw_shard_hashes"]
                    if Path(r["path"]).parts[-2:] == ("uninterrupted", "final.json")
                )
                state = json.loads(freeze(checked(final), raw, data).read_bytes())
                data["cases"] += [
                    dict(
                        scope="private_static_fixture",
                        arm="static",
                        state=dict(schema_version=1, model=work["static"]["model"]),
                        rows=rows,
                    ),
                    dict(
                        scope="private_causal_fixture", arm="causal_restart", state=state, rows=rows
                    ),
                ]
            if valid and name == "learning":
                ref = next(
                    r for r in value["raw_shard_hashes"] if r["path"] == value["final_states_path"]
                )
                states = json.loads(freeze(checked(ref), raw, data).read_bytes())
                restart = next(
                    r
                    for r in value["raw_shard_hashes"]
                    if Path(r["path"]).name == "restart-input.json"
                )
                rows = json.loads(freeze(checked(restart), raw, data).read_bytes())["rows"]
                data["cases"] += [
                    dict(scope="natural_learning", arm=f"seed{index}_{arm}", state=s, rows=rows)
                    for index, bundle in enumerate(states)
                    for arm, s in bundle.items()
                ]
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            data["checks"].append(
                operand("primitive_contract", path, "authenticated fields", str(error))
            )
        checks = data["checks"][begin:]
        data["branches"][name] = dict(
            ready=valid and all(c["passed"] for c in checks),
            verdict=value.get("honest_verdict"),
            verdict_class=value.get("verdict_class"),
            original_reason=dict(
                owned_failure=value.get("owned_failure"),
                failed_validation_receipts=[
                    r for r in value.get("validation_receipts", []) if not r.get("passed")
                ],
            ),
            failed_operands=[c for c in checks if not c["passed"]],
        )
        data["cited"].append(
            dict(
                path=str(path),
                sha256=sha256_file(path) if path.is_file() else None,
                imported_fields=[
                    score,
                    "honest_verdict",
                    "board_rows",
                    "measurement_reference",
                    "raw_shard_hashes",
                    "final_states_path",
                    "validation_receipts",
                ],
            )
        )
        progress("authenticate_after_" + name, index + 1, 2 - index)
    return data


def inventory(state: Json) -> Json:
    """List every tree node without flattening ordered clipping or exact mixtures."""
    result: Json = dict(
        predicates=[],
        ordered_corrections=[],
        exact_mixtures=[],
        global_parameters=dict(
            clip_bounds=[1e-6, 1 - 1e-6],
            input="authenticated base probability supplied by host",
            acceptance_permission="original baseline_action",
            fit_parameters=[],
        ),
        restart_fields=sorted(state),
        checksum_fields=sorted(k for k in state if "hash" in k or "sha256" in k),
    )

    def walk(model: Json, location: str) -> None:
        kind = model["kind"]
        if kind not in {"input", "global", "patch", "mixture"}:
            raise ValueError("model_kind")
        result["global_parameters"]["fit_parameters"].append(
            dict(
                path=location,
                fields={
                    k: v
                    for k, v in model.items()
                    if k not in {"base", "candidate", "patches", "kind"}
                },
            )
        )
        if kind in {"input", "global"}:
            return
        walk(model["base"], location + ".base")
        if kind == "mixture":
            result["exact_mixtures"].append(
                dict(
                    path=location,
                    step=model["step"],
                    fp64_hex=float(model["step"]).hex(),
                    formula="(1-step)*predict(base)+step*predict(candidate)",
                )
            )
            walk(model["candidate"], location + ".candidate")
        else:
            for index, op in enumerate(model["patches"]):
                result["predicates"].append(
                    dict(path=f"{location}.patches[{index}]", **op["group"])
                )
                result["ordered_corrections"].append(
                    dict(
                        path=location,
                        index=index,
                        delta=op["delta"],
                        fp64_hex=float(op["delta"]).hex(),
                        clip_after_each=True,
                    )
                )

    walk(state["model"], "model")
    if state.get("candidate"):
        walk(state["candidate"]["model"], "candidate.model")
    return result


def encode(state: Json) -> bytes:
    """Canonical JSON retains exact Python float values and binds the entire state."""
    if state.get("schema_version") != 1:
        raise ValueError("state_version")
    inventory(state)
    return json.dumps(
        dict(version=1, payload=state, payload_sha256=canonical_hash(state)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def decode(payload: bytes) -> Json:
    """Reject unsupported versions before accepting state with an equal checksum."""
    envelope = json.loads(payload)
    if envelope["version"] != 1:
        raise ValueError("version")
    state: Json = envelope["payload"]
    if envelope["payload_sha256"] != canonical_hash(state):
        raise ValueError("payload_hash")
    encode(state)
    return state


def predictions(state: Json, rows: list[Json]) -> list[Json]:
    """Record missing probabilities and original permissions alongside scalar parity."""
    result = []
    for r in rows:
        p = predict(state["model"], r)
        result.append(
            dict(
                unit_id=r["unit_id"],
                p=p,
            action=rule.action(p, r["baseline_action"]),
                baseline_action=r["baseline_action"],
            )
        )
    return result


def measure(data: Json, raw: Path) -> list[Json]:
    """Time host encoding, decoding and local fsync, never a board operation."""
    raw.mkdir(parents=True, exist_ok=True)
    result = []
    for index, case in enumerate(data["cases"]):
        progress("benchmark_before_" + case["arm"], index, len(data["cases"]) - index)
        start = time.perf_counter_ns()
        payload = encode(case["state"])
        encoded = time.perf_counter_ns()
        restored = decode(payload)
        decoded = time.perf_counter_ns()
        path = raw / f"state-{index}.json"
        storage = storage_probe(payload, path, case["arm"])
        expected = predictions(case["state"], case["rows"])
        actual = predictions(restored, case["rows"])
        result.append(
            dict(
                unit_id=f"state-{index}",
                source_cluster_id=case["scope"],
                arm=case["arm"],
                condition=case["scope"],
                scope=case["scope"],
                metric="host_state_parity",
                numerator=int(restored == case["state"] and actual == expected),
                denominator=1,
                status="completed",
                parity=restored == case["state"] and actual == expected,
                serialized_bytes=len(payload),
                payload_sha256=canonical_hash(case["state"]),
                envelope_sha256=sha256_file(path),
                payload_path=str(path),
                inventory=inventory(restored),
                predictions=actual,
                prediction_available_count=sum(p["p"] is not None for p in actual),
                prediction_missing_count=sum(p["p"] is None for p in actual),
                encode_ns=encoded - start,
                decode_ns=decoded - encoded,
                host_storage=storage,
                started_perf_counter_ns=start,
                encoded_perf_counter_ns=encoded,
                decoded_perf_counter_ns=decoded,
                clock="perf_counter_ns",
                measurement_scope="current host only; no network or board timing",
            )
        )
        progress("benchmark_after_" + case["arm"], index + 1, len(data["cases"]) - index - 1)
    return result


def verify_primitives(data: Json, measured: list[Json]) -> None:
    """Fresh-process disk reload verifies exact envelopes and deterministic parity."""
    if len(data["cases"]) != len(measured):
        raise ValueError("primitive_count")
    for case, row in zip(data["cases"], measured, strict=True):
        payload = Path(row["payload_path"]).read_bytes()
        restored = decode(payload)
        expected = dict(
            serialized_bytes=len(encode(case["state"])),
            payload_sha256=canonical_hash(case["state"]),
            envelope_sha256=sha256_file(Path(row["payload_path"])),
            inventory=inventory(case["state"]),
            predictions=predictions(case["state"], case["rows"]),
            parity=restored == case["state"]
            and predictions(restored, case["rows"]) == row["predictions"],
        )
        if payload != encode(case["state"]) or any(row[k] != v for k, v in expected.items()):
            raise ValueError("primitive_drift")
        if (
            row["encode_ns"] != row["encoded_perf_counter_ns"] - row["started_perf_counter_ns"]
            or row["decode_ns"] != row["decoded_perf_counter_ns"] - row["encoded_perf_counter_ns"]
        ):
            raise ValueError("clock_drift")


def contract() -> Json:
    """Require a future durable board transaction with all costs charged once."""
    return dict(
        version=1,
        substrate="PolarFire Linux CPU; fabric unsupported",
        authorization="future separately authorized device work only",
        payload="exact versioned canonical UTF-8 JSON envelope; no tree flattening",
        payload_hash_equality="sender envelope and state SHA256 equal receiver before activation and after restart",
        version_rejection="reject unknown envelope/state versions before writing active state",
        durable_commit_steps=[
            "private_temporary_write",
            "flush",
            "file_fsync",
            "atomic_rename",
            "directory_fsync",
            "acknowledge_after_commit",
        ],
        restart="fresh receiver process reads durable bytes and reproduces tree, pending/release/RNG state and frozen prediction/action parity",
        cost_fields=[
            "host_encode_ns",
            "host_hash_ns",
            "host_queue_ns",
            "connection_setup_ns",
            "network_send_ns",
            "network_receive_ns",
            "device_decode_ns",
            "device_hash_ns",
            "device_write_ns",
            "device_file_fsync_ns",
            "device_rename_ns",
            "device_directory_fsync_ns",
            "restart_read_ns",
            "restart_decode_ns",
            "readout_ns",
            "host_validation_ns",
        ],
        accounting="same-request disjoint clocks, acknowledgement and retransmits; charge acquisition/model work and all persistence outside eligible device work",
        symbolic_transport="T_network >= (payload_bytes + protocol_bytes + readout_bytes)/B_measured + L_setup + L_ack + T_queue + T_retries",
        measured_board_bandwidth_bytes_s=None,
        measured_board_latency_ns=None,
        bandwidth_domain="B_measured > 0, unknown; no advertised link rate substituted",
        device_speedup=None,
        current_ssh_reachability="not_probed",
    )


def reduce(data: Json, measured: list[Json]) -> Json:
    """Readiness certifies an inventory even when natural learning remains blocked."""
    parity = all(r["parity"] for r in measured)
    ready = int(
        bool(measured)
        and parity
        and data["board"].get("custody_valid") is True
        and data["board"].get("actual_substrate") == "linux_cpu"
        and data.get("branches", {}).get("historical", {}).get("ready", True)
    )
    blocked = [dict(branch=k, **v) for k, v in data["branches"].items() if not v["ready"]]
    kind = (
        "disqualified"
        if not parity
        else "blocked"
        if blocked or not ready
        else "circular_positive"
        if data.get("fixture")
        else "null"
    )
    rows = measured + [
        dict(
            unit_id=b["branch"] + "_obligation",
            source_cluster_id=b["branch"],
            arm=b["branch"],
            condition="upstream_branch",
            metric="branch_available",
            numerator=None,
            denominator=1,
            status="excluded",
            exclusion_reason=b["failed_operands"],
        )
        for b in blocked
    ]
    return dict(
        honest_verdict="complete_"
        + kind
        + ("_" + blocked[0]["branch"] + "_operand" if blocked else "_polarfire_state_boundary"),
        verdict_class=kind,
        polarfire_boundary_ready_score=ready,
        serialized_state_rows=measured,
        branch_readiness=data["branches"],
        completed_obligations=[
            "authenticated board inventory",
            "versioned host state and local durable parity",
        ]
        if ready
        else [],
        blocked_obligations=blocked,
        future_device_obligations=[
            "payload equality",
            "version rejection",
            "durable commit",
            "fresh restart",
            "measured transfer-inclusive clocks",
        ],
        polarfire_obligation=dict(
            historical=data["board"],
            actual_substrate="linux_cpu",
            current_reachability="not_probed",
            current_board_execution=False,
            fabric_use=False,
            fresh_device_timing_ns=None,
            benefit_score=0,
            device_work_authorized=False,
        ),
        rows=rows,
        intended_count=len(rows),
        completed_count=len(measured),
        failed_count=0,
        censored_count=0,
        excluded_count=len(blocked),
        independent_count=0,
        verifier_is_oracle=True,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        acceptance_gates=dict(
            host_parity=parity, qualified_inventory=bool(ready), measured_device_benefit=False
        ),
        current_device_execution_count=0,
    )

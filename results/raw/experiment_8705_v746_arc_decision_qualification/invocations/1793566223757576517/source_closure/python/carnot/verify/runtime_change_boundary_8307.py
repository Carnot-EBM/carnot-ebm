"""REQ-VERIFY-8307: compare runtime evidence before permitting another CUDA attempt.

A repeated failure is not new evidence. This reader authenticates the earlier
failure and admits probes only when a recorded runtime operand changes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v709_execution import child
from carnot.reporting.v710_contract_replay import snapshot, require_reference
from carnot.verify import cuda_primitive_8290 as primitive
from carnot.verify.runtime_localization_8290 import cuda_ready

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8307_v717_runtime_change_boundary"
TASK = "exp8307-runtime-change-boundary"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8290_v716_runtime_localization.json"
PIN = "sha256:62875d880623865338f7d6b85e1caa82ed388cb0b94b4402f0457245361a0b1f"
REPAIR = "ops/cuda-runtime-repair-receipt.json"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush each boundary so a supervisor can see measured progress."""
    print(f"[exp8307] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Name absent operands separately from measured failures, with exact bytes."""
    return dict(
        upstream_id="exp8290",
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


def historical_identity(value: Json, work: Json, binding: Json) -> Json:
    """Recover causal operands from authenticated primitive rows, not file dates."""
    diag = work["diagnostic"]
    libraries = {r["path"]: r for row in diag["rows"] for r in row["primitive"]["libraries"]}
    libraries.update({r["path"]: r for r in binding["native_library_hashes"]})
    libraries[binding["binary"]] = dict(path=binding["binary"], sha256=binding["binary_sha256"])
    return dict(
        driver=value["device_inventory"][0]["driver_version"],
        kernel=diag["driver_module_version"],
        devices=[
            {k: r[k] for k in ("uuid", "index", "name", "driver_version")}
            for r in value["device_inventory"]
        ],
        masks=binding["inherited"],
        libraries=list(libraries.values()),
        nodes=diag["device_nodes"],
        permitted_uuid=binding["permitted_uuid"],
        binary=binding["binary"],
        driver_library=binding["driver_library"],
        runtime_library=binding["runtime_library"],
    )


def inventory(previous: Json, raw: Path) -> tuple[Json, list[Json]]:
    """Read hashes and inventory without loading a CUDA library or native binary."""
    receipt = child(
        "read_only_inventory",
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        raw,
        deadline=15,
        scope="external_inventory",
    )
    devices = [
        dict(zip(("index", "uuid", "name", "driver_version"), map(str.strip, cols), strict=True))
        for line in Path(receipt["stdout_path"]).read_text().splitlines()
        if len(cols := line.split(",")) == 4
    ]
    libraries, resolutions = [], []
    for item in previous["libraries"]:
        path = Path(item["path"])
        if path.name.startswith("libcuda.so"):
            path = path.with_name("libcuda.so.1")
        resolved = path.resolve()
        libraries.append(
            dict(path=item["path"], sha256=sha256_file(resolved) if resolved.is_file() else None)
        )
        resolutions.append(dict(operand_path=item["path"], resolved_path=str(resolved)))
    kernel = Path("/proc/driver/nvidia/version")
    current = dict(
        driver=devices[0]["driver_version"] if devices else None,
        kernel=kernel.read_text() if kernel.is_file() else None,
        devices=devices,
        masks={k: os.environ[k] for k in primitive.VARIABLES if k in os.environ},
        libraries=libraries,
        resolved_library_rows=resolutions,
        nodes=primitive.node_access(sorted(Path("/dev").glob("nvidia*"))),
        permitted_uuid=previous["permitted_uuid"],
        binary=previous["binary"],
        driver_library=previous.get("driver_library", "libcuda.so.1"),
        runtime_library=previous.get("runtime_library", "libcudart.so.12"),
    )
    return current, [receipt]


def load(root: Path, raw: Path, *, observe: bool = True) -> Json:
    """Authenticate the declared historical primary and its terminal byte custody."""
    data: Json = dict(
        previous={},
        current={},
        checks=[],
        refs=[],
        historical={},
        diagnostic=dict(rows=[], checks=[], lease_receipt={}, cleanup_receipt={}),
        receipt_rows=[],
        fixture=False,
        observation_receipts=[],
    )
    path = root / UPSTREAM
    progress("authenticate_before", 0, 1)
    try:
        check = gate(path, "sha256", PIN, sha256_file(path) if path.is_file() else None)
        data["checks"].append(check)
        if not check["passed"]:
            raise ValueError("upstream_primary_bytes")
        ref = snapshot(path, raw / "inputs", "primary")
        data["refs"].append(ref)
        value = json.loads(Path(ref["snapshot_path"]).read_bytes())
        if value["required_checks_passed"] is not True or value["flagged_adversarial"] is not False:
            raise ValueError("upstream_owned_checks")
        side = path.parent / "raw" / path.stem / "validators" / (PIN.split(":")[1] + ".json")
        bound = read_bound_sidecar(path, side)
        if bound["report"]["passed"] is not True:
            raise ValueError("upstream_terminal_auditors")
        data["refs"].append(snapshot(side, raw / "inputs", "validator"))
        terminal = Path(value["terminal_validation_sidecar_path"])
        terminal_value = json.loads(terminal.read_bytes())
        if (
            terminal_value["publication"]["primary_sha256"] != PIN
            or not terminal_value["normal_process_exit"]
        ):
            raise ValueError("upstream_terminal_binding")
        data["refs"].append(snapshot(terminal, raw / "inputs", "terminal"))
        for item in [
            value["work_reference"],
            dict(path=value["runtime_binding_path"], sha256=value["runtime_binding_sha256"]),
        ]:
            require_reference(item)
            data["refs"].append(snapshot(Path(item["path"]), raw / "inputs", "primitive"))
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        binding = json.loads(Path(value["runtime_binding_path"]).read_bytes())
        for row in work["diagnostic"]["rows"]:
            for prefix in ("stdout", "stderr"):
                item = dict(
                    path=row["receipt"][prefix + "_path"], sha256=row["receipt"][prefix + "_sha256"]
                )
                require_reference(item)
                data["refs"].append(snapshot(Path(item["path"]), raw / "inputs", prefix))
            if (
                json.loads(Path(row["receipt"]["stdout_path"]).read_text().splitlines()[-1])
                != row["primitive"]
            ):
                raise ValueError("upstream_probe_primitive")
        for receipt_check in bound["report"]["checks"]:
            if not receipt_check["passed"] or not receipt_check["normal_exit"]:
                raise ValueError("upstream_validator_exit")
            for prefix in ("stdout", "stderr"):
                ref = dict(
                    path=receipt_check[prefix + "_path"], sha256=receipt_check[prefix + "_sha256"]
                )
                require_reference(ref)
                data["refs"].append(
                    snapshot(Path(ref["path"]), raw / "inputs", "terminal_" + prefix)
                )
        data.update(historical=value, previous=historical_identity(value, work, binding))
        if observe:
            data["current"], data["observation_receipts"] = inventory(
                data["previous"], raw / "inventory"
            )
        data["observed_wall_ns"] = time.time_ns()
        repair = root / REPAIR
        if repair.is_file():
            data["refs"].append(snapshot(repair, raw / "inputs", "operator_repair"))
            receipt = json.loads(repair.read_bytes())
            eligible = eligible_repair(receipt, value, data["observed_wall_ns"])
            data["receipt_rows"].append(
                dict(
                    path=str(repair), sha256=sha256_file(repair), receipt=receipt, eligible=eligible
                )
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        data["checks"].append(gate(path, "authenticated_runtime_evidence", True, str(error)))
    progress("authenticate_after", 1, 0)
    return data


def frontier_wall_ns(value: Json) -> int:
    """Use recorded phase clocks so a repair predating the failure cannot reopen it."""
    return max(
        int(r["started_wall_ns"] + r["duration_s"] * 1e9)
        for r in value["phase_spans"]
        if "started_wall_ns" in r
    )


def eligible_repair(receipt: Json, value: Json, observed_wall_ns: int) -> bool:
    """A dated operator action must follow the authenticated failure evidence frontier."""
    return bool(
        receipt.get("authority") == "operator"
        and receipt.get("previous_primary_sha256") == PIN
        and frontier_wall_ns(value) < receipt.get("executed_at_ns", 0) <= observed_wall_ns
        and receipt.get("action") in {"driver_repair", "device_repair", "lease_rebind", "reboot"}
        and receipt.get("description")
    )


def normalized(value: Any) -> Any:
    """UUID case changes do not alter which physical resource is named."""
    if isinstance(value, str) and value.lower().startswith("gpu-"):
        return value.lower()
    if isinstance(value, list):
        return [normalized(v) for v in value]
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    return value


def delta_rows(data: Json) -> list[Json]:
    """Only present causal identities or authenticated repairs can reopen the gate."""
    rows = []
    for field in ("driver", "kernel", "devices", "masks", "libraries", "nodes", "permitted_uuid"):
        old, new = data["previous"].get(field), data["current"].get(field)
        available = old is not None and new is not None
        changed = available and normalized(old) != normalized(new)
        complete_libraries = field != "libraries" or all(r.get("sha256") for r in new or [])
        rows.append(
            dict(
                unit_id=field,
                field=field,
                previous=old,
                observed=new,
                available=available,
                changed=changed,
                potentially_causal=bool(changed and complete_libraries),
                status="completed" if available else "censored",
            )
        )
    rows.append(
        dict(
            unit_id="operator_repair",
            field="operator_repair",
            previous=None,
            observed=data["receipt_rows"],
            available=True,
            changed=any(r["eligible"] for r in data["receipt_rows"]),
            potentially_causal=any(r["eligible"] for r in data["receipt_rows"]),
            status="completed",
        )
    )
    return rows


def probe_changed(data: Json, raw: Path) -> Json:
    """One lease fixes one UUID; children cannot search alternative device mappings."""
    current = data["current"]
    uuid = current["permitted_uuid"]
    diag: Json = dict(rows=[], checks=[], lease_receipt={}, cleanup_receipt={})
    lease: Any = None
    began = time.monotonic_ns()
    try:
        if not any(
            normalized(r["uuid"]) == normalized(uuid)
            and all(
                mask is None
                or (key == "NVIDIA_VISIBLE_DEVICES" and mask == "all")
                or str(r["index"]) in mask.split(",")
                or r["uuid"] in mask.split(",")
                for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
                for mask in [current["masks"].get(key)]
            )
            for r in current["devices"]
        ):
            raise ValueError("leased_uuid_not_visible")
        lease = GpuLease.acquire(
            runtime_dir="/tmp/carnot-gpu-leases",
            task_id=TASK,
            device_uuid=uuid,
            expected_model="no_model_load:cuda_byte_copy",
            vram_before_mb=0,
            ttl_s=420,
        )
        diag["lease_receipt"] = lease.owner_receipt()
        lease.transition("admitted")
        for index, layer in enumerate(("driver", "runtime", "native")):
            progress("changed_probe_before_" + layer, index, 3 - index)
            argv = [
                "/usr/bin/env",
                "CUDA_VISIBLE_DEVICES=" + uuid,
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--cuda-probe",
                layer,
                "--uuid",
                uuid,
                "--binary",
                current["binary"],
                "--library",
                current[layer + "_library"] if layer != "native" else "libcuda.so.1",
            ]
            receipt = child(
                layer,
                argv,
                raw,
                deadline=min(60, max(0.01, 360 - (time.monotonic_ns() - began) / 1e9)),
                heartbeat=20,
                scope="external_cuda",
            )
            text = Path(receipt["stdout_path"]).read_text().splitlines()
            primitive_value = (
                json.loads(text[-1]) if receipt["passed"] and text else dict(error="child_exit")
            )
            diag["rows"].append(
                dict(
                    layer=layer, binding="explicit_uuid", receipt=receipt, primitive=primitive_value
                )
            )
            progress("changed_probe_after_" + layer, index + 1, 2 - index)
        lease.transition("terminal_blocked")
    except (OSError, ValueError, LeaseError) as error:
        diag["checks"].append(gate(raw, "exclusive_gpu_lease_and_children", True, str(error)))
    finally:
        if lease:
            if lease.document["phase"] not in {"terminal_blocked", "terminal_complete"}:
                lease.transition("terminal_blocked")
            diag["cleanup_receipt"] = lease.release()
            diag["lease_receipt"]["release"] = diag["cleanup_receipt"]
    return diag


def reduce(data: Json, checks_passed: bool) -> Json:
    """A changed precondition earns progress; execution still requires measured parity."""
    rows = delta_rows(data)
    changed = any(r["potentially_causal"] for r in rows)
    diag = data["diagnostic"]
    required_driver = (
        "cuInit",
        "cuCtxCreate_v2",
        "cuMemAlloc_v2",
        "cuMemcpyHtoD_v2",
        "cuMemcpyDtoH_v2",
    )
    ready = changed and cuda_ready(diag) and checks_passed
    ready &= diag["lease_receipt"].get("device_uuid") == data["current"].get("permitted_uuid")
    for row in diag["rows"]:
        p = row["primitive"]
        ready &= p.get("environment") == dict(
            data["current"]["masks"], CUDA_VISIBLE_DEVICES=data["current"]["permitted_uuid"]
        )
        if row["layer"] == "driver":
            ready &= (
                all(p.get("api_returns", {}).get(k) == 0 for k in required_driver)
                and p.get("byte_copy_parity") is True
                and p.get("allocation_bytes") == 32
                and p.get("cleanup_passed") is True
                and p.get("copy_source_hex") == bytes(range(32)).hex() == p.get("copy_target_hex")
            )
    missing = [g for g in data["checks"] if not g["passed"]]
    missing += [
        gate(Path(r["stdout_path"]), "inventory.normal_exit", True, r["passed"])
        for r in data["observation_receipts"]
        if not r["passed"]
    ]
    missing += diag.get("checks", [])
    missing += [
        gate(Path(r["path"]), "library.sha256", "readable SHA-256", None)
        for r in data["current"].get("libraries", [])
        if r.get("sha256") is None
    ]
    ready &= not missing
    verdict = "disqualified" if not checks_passed else "blocked" if missing or not ready else "null"
    gates = list(missing)
    if verdict == "blocked" and not gates:
        gates.append(
            gate(
                Path(data.get("boundary_path", ROOT / UPSTREAM)),
                "authenticated_causal_delta"
                if not changed
                else "leased_context_copy_and_native_ready",
                True,
                changed if not changed else bool(ready),
            )
        )
    return dict(
        honest_verdict="complete_"
        + verdict
        + "_"
        + ("upstream_evidence" if missing else "cuda_runtime"),
        verdict_class=verdict,
        gate_check_summary=gates,
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["available"] for r in rows),
        failed_count=0,
        censored_count=sum(not r["available"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(intended_operands=len(rows), independent_scientific_samples=0),
        runtime_changed_score=int(changed),
        cuda_context_ready_score=int(ready),
        environment_delta_rows=rows,
        current_probe_count=len(diag["rows"]),
        current_model_calls=0,
        failure_layer=data["historical"].get("failure_layer", []),
        root_cause_status="unproved",
        lease_receipt=diag["lease_receipt"],
        cuda_probe_rows=diag["rows"],
        acceptance_gates=dict(
            authenticated_change=changed, cuda_context_ready=bool(ready), owned_checks=checks_passed
        ),
        reopen_contract=dict(
            previous_primary_sha256=PIN,
            eligible=changed and not missing,
            required_delta="Byte-bound corrective driver/library, kernel, device visibility or lease condition, or dated operator repair receipt.",
            probe_budget=dict(per_child_s=60, total_s=360, attempts_per_layer=1),
            required_measurements=[
                "driver initialization",
                "actual context",
                "32-byte host/device/host parity",
                "runtime enumeration",
                "native compatibility",
                "owned cleanup",
            ],
            next_consumer="exp8313-changed-runtime-canary",
            revalidate_immediately_before_load=True,
        ),
    )


def verify_primitives(data: Json) -> None:
    """Rebuild imported identities and compare raw child output before reducing scores."""
    for ref in data["refs"]:
        require_reference(ref)
    for row in data["receipt_rows"]:
        ref = next(r for r in data["refs"] if r["path"] == row["path"])
        receipt = json.loads(Path(ref["snapshot_path"]).read_bytes())
        if (
            receipt != row["receipt"]
            or eligible_repair(receipt, data["historical"], data["observed_wall_ns"])
            != row["eligible"]
        ):
            raise ValueError("operator_repair_drift")
    if data["historical"] and not data["fixture"]:
        refs = data["refs"]
        value = json.loads(Path(refs[0]["snapshot_path"]).read_bytes())
        work = json.loads(Path(refs[3]["snapshot_path"]).read_bytes())
        binding = json.loads(Path(refs[4]["snapshot_path"]).read_bytes())
        if (
            refs[0]["sha256"] != PIN
            or value != data["historical"]
            or historical_identity(value, work, binding) != data["previous"]
        ):
            raise ValueError("historical_identity_drift")
    if data.get("current_reference"):
        require_reference(data["current_reference"])
        if json.loads(Path(data["current_reference"]["path"]).read_bytes()) != data["current"]:
            raise ValueError("current_identity_drift")
    for row in data["diagnostic"]["rows"]:
        r = row["receipt"]
        for prefix in ("stdout", "stderr"):
            require_reference(dict(path=r[prefix + "_path"], sha256=r[prefix + "_sha256"]))
        if (
            r["passed"]
            and json.loads(Path(r["stdout_path"]).read_text().splitlines()[-1]) != row["primitive"]
        ):
            raise ValueError("current_probe_drift")

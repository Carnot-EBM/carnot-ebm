"""REQ-VERIFY-8216: available evidence cannot erase an unmeasured obligation.

Numeric conversion is host arithmetic on frozen development heads. It does not
execute a device kernel or create independent learning evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.request_trace_inventory_8200 import copy_bytes, operand
from carnot.reporting import radial_hardware_8108 as custody
from carnot.reporting.hardware_service_8190 import REOPEN
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import restricted_action_rule_8207 as rule
from carnot.verify import prospective_service_8214 as service

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8216_v709_hardware_workload_obligations"
CLI = "scripts/experiments/" + NAME + ".py"
CONFIG: Json = dict(
    seed=7098216,
    precisions=["float64", "float32", "fixed16"],
    fixed_max=32767,
    roundoff_allowance=1e-12,
    target_speedup=100,
)
PRODUCERS = {
    "action": (8208, "restricted_energy_fit", "action_fit_ready_score"),
    "learning": (8211, "calibrated_memory_trajectory", "learning_trajectory_ready_score"),
    "service": (8214, "prospective_service_measurement", "service_measurement_ready_score"),
}
PINS = {
    8208: "sha256:0063bc79bf9be8d3a7739603f105a263a83d9d9e599c57e4f9ff427263262aaa",
    8211: "sha256:51fb7180404aff65936949496e4a9a16eddf19f2423dc6613d4cc1b4deba249f",
    8214: "sha256:6e64b515e25598844d287b06aa2ba7324caf2bb4603b925179457438ccf3233a",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts expose real progress before the supervisor's deadline."""
    print(f"[exp8216] phase={phase} completed={completed} pending={pending}", flush=True)


def load(root: Path, raw: Path) -> Json:
    """Authenticate each producer independently so a missing sibling loses no rows."""
    data: Json = dict(
        heads=[],
        samples=[],
        baseline={},
        learning={},
        service={},
        boards=[],
        checks=[],
        references=[],
        cited=[],
        branches={},
        fixture=False,
    )
    policy_path = root / "ops/exclusion_manifest.yaml"
    policy = yaml.safe_load(policy_path.read_text()) if policy_path.is_file() else {}
    retired = {
        r.get("experiment_id")
        for k in ("retired", "retired_experiments")
        for r in policy.get(k, [])
    }
    for i, (name, (eid, suffix, score)) in enumerate(PRODUCERS.items()):
        progress("authenticate_before_" + name, i, 3 - i)
        path = root / "results" / f"experiment_{eid}_v709_{suffix}.json"
        begin = len(data["checks"])
        value, valid = authenticate(path, eid, score, raw, data)
        for field, wanted, observed in [
            ("sha256", PINS[eid], sha256_file(path) if path.is_file() else None),
            ("task_id", f"exp{eid}-{suffix.replace('_', '-')}", value.get("task_id")),
            ("fixture_mode", False, bool(value.get("fixture_mode") or value.get("fixture"))),
            ("not_retired", True, eid not in retired),
        ]:
            data["checks"].append(operand(field, path, wanted, observed))
        valid = valid and all(c["passed"] for c in data["checks"][begin:])
        try:
            if valid and name == "action":
                frozen = dict(path=value["frozen_heads_path"], sha256=value["frozen_heads_sha256"])
                heads = json.loads(checked(frozen).read_bytes())
                measured = json.loads(checked(value["measurement_reference"]).read_bytes())
                data["references"] += [
                    copy_bytes(checked(r), raw) for r in [frozen, value["measurement_reference"]]
                ]
                data.update(
                    heads=heads["heads"],
                    baseline=heads["baseline"],
                    samples=measured["evidence"]["rows"],
                )
            if valid and name == "learning":
                ref = dict(path=value["trajectory_path"], sha256=value["trajectory_sha256"])
                data["references"].append(copy_bytes(checked(ref), raw))
                data["learning"] = dict(value, storage_source=ref)
            if valid and name == "service":
                ref = next(
                    r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "result.json"
                )
                work = json.loads(checked(ref).read_bytes())["work"]
                if work["requests"] != value["request_rows"]:
                    raise ValueError("service_request_primitive_drift")
                data["references"].append(copy_bytes(checked(ref), raw))
                data["service"] = value
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            valid = False
            data["checks"].append(
                operand("primitive_contract", path, "authenticated fields", str(error))
            )
        data["branches"][name] = valid
        data["cited"].append(
            dict(
                experiment_id=eid,
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                fields_imported=[score],
                eligible=valid,
            )
        )
        progress("authenticate_after_" + name, i + 1, 2 - i)
    history = root / "results/experiment_8203_v708_hardware_decision_boundary.json"
    if history.is_file():
        data["references"].append(copy_bytes(history, raw))
        for board in json.loads(history.read_bytes())["board_rows"][:3]:
            board = dict(board)
            begin = len(data["checks"])
            for p, pin in [
                (root / board["source_path"], board["source_hash"]),
                (Path(board["source_transcript"]), board["source_transcript_sha256"]),
            ]:
                data["checks"].append(
                    operand("board_source_sha256", p, pin, sha256_file(p) if p.is_file() else None)
                )
                if p.is_file():
                    data["references"].append(copy_bytes(p, raw))
            board.update(
                custody_valid=all(c["passed"] for c in data["checks"][begin:]),
                current_hardware_execution=False,
                current_reachability="not_probed",
                actual_substrate=custody.CONTRACTS[board["board"]][0],
                reopen_condition=REOPEN[board["board"]],
            )
            data["boards"].append(board)
    return data


def precision(data: Json) -> list[Json]:
    """Reapply acceptance permission after every conversion and measured fallback.

    The original float64 basis stays shared. Bounds cover converted operands and
    accumulation, not a future fixed-point exponential implementation.
    """
    rows: list[Json] = []
    for head in data["heads"]:
        progress("benchmark_before_" + head["arm"], 0, len(data["samples"]))
        train = [
            r for r in data["samples"] if r["unit_id"] in head["fit_ids"] and r["x"] is not None
        ]
        x = np.asarray([r["x"] for r in train], dtype=float)
        lower, upper = x.min(axis=0), x.max(axis=0)
        phi_train = rule.design(head["arm"], x, head["geometry"])
        weights = np.asarray(head["weights"], dtype=float)
        ws = max(float(np.max(np.abs(weights))), 1e-30) / 32767
        ps = max(float(np.max(np.abs(phi_train))), 1e-30) / 32767
        wi = np.rint(weights / ws).astype(np.int16)
        for i, source in enumerate(data["samples"]):
            query = {k: source[k] for k in ("unit_id", "source_cluster_id", "x", "historical_x")}
            reference_prediction = rule.predict(head, query, data["baseline"])
            for kind in CONFIG["precisions"]:
                row = dict(
                    unit_id=source["unit_id"],
                    source_cluster_id=source["source_cluster_id"],
                    arm=head["arm"],
                    precision=kind,
                    condition="current_exposed_head",
                    metric="final_decision_mismatch",
                    denominator=1,
                    source_experiment_id=8208,
                    status="excluded",
                    numerator=None,
                    exclusion_reason="missing_paired_features",
                )
                if reference_prediction["p"] is None:
                    rows.append(row)
                    continue
                began = time.perf_counter_ns()
                phi = rule.design(head["arm"], np.asarray([source["x"]]), head["geometry"])[0]
                p = reference_prediction["p"]
                outside = bool(np.any(source["x"] < lower) or np.any(source["x"] > upper))
                approximate, radius = p, 0.0
                if kind != "float64":
                    if kind == "float32":
                        a, b = phi.astype(np.float32), weights.astype(np.float32)
                        z = float(np.sum(a * b, dtype=np.float32))
                        u = float(np.finfo(np.float32).eps)
                        rounding = (
                            2 * len(phi) * u / (1 - 2 * len(phi) * u) * float(np.sum(abs(a * b)))
                        )
                    else:
                        ai = np.clip(np.rint(phi / ps), -32767, 32767).astype(np.int16)
                        a, b = ai.astype(float) * ps, wi.astype(float) * ws
                        z = int(np.sum(ai.astype(np.int64) * wi.astype(np.int64))) * ps * ws
                        rounding = 1e-12 * (1 + abs(z))
                    residual = float(
                        np.sum(abs(phi - a) * abs(weights) + abs(a) * abs(weights - b))
                    )
                    radius = (residual + rounding) / (4 * head["temperature"]) + 1e-12
                    approximate = rule.probability(0, -z, head["temperature"])
                ambiguous = min(abs(approximate - t) for t in [0.1, 1 / 6, 0.5]) <= radius
                fallback = kind != "float64" and (ambiguous or outside)
                start = time.perf_counter_ns()
                final_p = (
                    rule.predict(head, query, data["baseline"])["p"] if fallback else approximate
                )
                fallback_ns = time.perf_counter_ns() - start if fallback else 0
                permission = reference_prediction["baseline_action"]
                final = rule.action(final_p, permission)
                row.update(
                    status="completed",
                    exclusion_reason=None,
                    numerator=int(final != reference_prediction["action"]),
                    probability_reference=p,
                    probability_approximate=approximate,
                    probability_error=abs(p - approximate),
                    interval_radius=radius,
                    baseline_action=permission,
                    raw_action=rule.action(approximate, permission),
                    reference_action=reference_prediction["action"],
                    final_action=final,
                    fallback=fallback,
                    ambiguous_margin=ambiguous,
                    outside_domain=outside,
                    fallback_ns=fallback_ns,
                    elapsed_ns=time.perf_counter_ns() - began,
                    weight_scale=ws,
                    operand_scale=ps,
                )
                rows.append(row)
            if i % 32 == 0 or i + 1 == len(data["samples"]):
                progress("precision_" + head["arm"], i + 1, len(data["samples"]) - i - 1)
        progress("benchmark_after_" + head["arm"], len(data["samples"]), 0)
    return rows


def storage_probe(payload: bytes, path: Path, branch: str) -> Json:
    """A local byte round trip measures storage only, not the original learning run."""
    progress("benchmark_before_storage_" + branch, 0, 1)
    start = time.perf_counter_ns()
    transferred = memoryview(payload).tobytes()
    transfer_end = time.perf_counter_ns()
    with path.open("wb") as stream:
        stream.write(transferred)
        stream.flush()
        write_end = time.perf_counter_ns()
        os.fsync(stream.fileno())
        fsync_end = time.perf_counter_ns()
    read = path.read_bytes()
    end = time.perf_counter_ns()
    if read != payload:
        raise ValueError("storage_parity")
    progress("benchmark_after_storage_" + branch, 1, 0)
    return dict(
        branch=branch,
        bytes=len(payload),
        transfer_ns=transfer_end - start,
        storage_ns=write_end - transfer_end,
        fsync_ns=fsync_end - write_end,
        readout_ns=end - fsync_end,
        total_ns=end - start,
        clock="perf_counter_ns",
        payload_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        scope="current local storage primitive; not original workload or device transfer",
    )


def service_costs(value: Json) -> tuple[list[Json], list[Json]]:
    """Remove only the observed scoring envelope; keep acquisition and durable work."""
    rows: list[Json] = []
    for request in value.get("request_rows", []):
        for arm in request["arms"]:
            stages = (
                service.stages(arm)
                if request["status"] == "completed"
                else {
                    k: None
                    for k in [
                        "conversion_ns",
                        "serialization_ns",
                        "commit_fsync_ns",
                        "readout_ns",
                        "scoring_ns",
                    ]
                }
            )
            if any(v is not None and v < 0 for k, v in stages.items() if k.endswith("_ns")):
                raise ValueError("clock_partition")
            total = arm["end_ns"] - arm["start_ns"]
            rows.append(
                dict(
                    unit_id=request["request_id"],
                    source_cluster_id=request["source_cluster_id"],
                    arm=arm["arm"],
                    condition=arm["arm"],
                    metric="downstream_ns",
                    numerator=total,
                    denominator=1,
                    status=request["status"],
                    exclusion_reason=None,
                    stages=stages,
                    acquisition_ns=request["acquisition_ns"],
                    transfer_ns=stages["conversion_ns"],
                    storage_ns=stages["serialization_ns"],
                    fsync_ns=stages["commit_fsync_ns"],
                    readout_ns=stages["readout_ns"],
                    measurement_scope="Exp8214 matched shared acquisition; no new inference",
                )
            )
    bounds = []
    acquisition = sum(r["acquisition_ns"] for r in value.get("request_rows", []))
    startup = value.get("cold_start_cost_s", 0) * 1e9
    overhead = value.get("shared_recording_and_checkpoint_overhead_s", 0) * 1e9
    for arm in sorted({r["arm"] for r in rows}):
        chosen = [r for r in rows if r["arm"] == arm]
        total = startup + acquisition + overhead + sum(r["numerator"] for r in chosen)
        scoring = sum(r["stages"]["scoring_ns"] for r in chosen if r["status"] == "completed")
        unknown = sum(r["numerator"] for r in chosen if r["status"] != "completed")
        serial = (total - scoring - unknown) / total
        bounds.append(
            dict(
                branch="service",
                arm=arm,
                total_ns=total,
                acquisition_ns=acquisition,
                model_startup_ns=startup,
                recording_overhead_ns=overhead,
                removable_scoring_ns=scoring,
                unpartitioned_failed_downstream_ns=unknown,
                serial_fraction=serial,
                maximum_speedup=1 / serial,
                supports_100x=serial <= 0.01,
                required_serial_share_reduction=max(0, serial - 0.01),
                required_workload_change="Acquisition/model startup and durable storage must change until retained share <=0.01.",
                scope="complete designed research workload; scoring envelope is optimistic, includes crossing",
            )
        )
    return rows, bounds


def reduce(data: Json, measured: list[Json], probes: list[Json]) -> Json:
    """Reducer readiness means scoped evidence, not completion of every science branch."""
    workloads, bounds = service_costs(data["service"]) if data["branches"]["service"] else ([], [])
    dispositions = {}
    cost_obligations = []
    missing = []
    for name, ready in data["branches"].items():
        failures = [
            c
            for c in data["checks"]
            if not c["passed"] and f"experiment_{PRODUCERS[name][0]}_" in c["path"]
        ]
        dispositions[name] = dict(
            ready=int(ready), status="qualified" if ready else "blocked", failed_operands=failures
        )
        if not ready:
            missing.append(
                dict(
                    unit_id=name + "_obligation",
                    source_cluster_id=name,
                    arm=name,
                    condition="optional_branch",
                    metric="branch_available",
                    numerator=None,
                    denominator=1,
                    status="excluded",
                    exclusion_reason=failures or "optional_producer_unavailable",
                )
            )
    if data["branches"]["learning"]:
        path = Path(
            data["learning"]
            .get("storage_source", {})
            .get(
                "path", str(ROOT / "results/experiment_8211_v709_calibrated_memory_trajectory.json")
            )
        )
        cost_obligations.append(
            dict(
                operand(
                    "trajectory.states.component_clocks", path, "disjoint component clocks", None
                ),
                upstream="exp8211",
            )
        )
        dispositions["learning"]["component_cost_status"] = "blocked_unrecorded_component_clocks"
        bounds.append(
            dict(
                branch="learning",
                status="blocked",
                total_ns=None,
                serial_fraction=None,
                maximum_speedup=None,
                supports_100x=False,
                exact_field="trajectory.states.component_clocks",
                observed=None,
                expected="disjoint fit/acquisition/transfer/storage/fsync/readout clocks",
                historical_child_duration_s=[
                    r["duration_s"] for r in data["learning"].get("child_exit_rows", [])
                ],
                scope="cached causal trajectory; local storage probes cannot fill original timers",
            )
        )
    board_rows = []
    for name, (substrate, kmax, idcode) in custody.CONTRACTS.items():
        original = next((b for b in data["boards"] if b["board"] == name), {})
        board_rows.append(
            dict(
                original,
                board=name,
                actual_substrate=substrate,
                k_max=kmax,
                blocked_idcode=idcode,
                evidence_status="historical_authenticated"
                if original.get("custody_valid")
                else "blocked_absent_evidence",
                current_hardware_execution=False,
                current_reachability="not_probed",
                reopen_condition=REOPEN[name],
            )
        )
    rows = [
        dict(
            r,
            **(
                {"decision_mismatch_rate": r["numerator"]}
                if r["metric"] == "final_decision_mismatch"
                else {}
            ),
        )
        for r in measured + workloads + missing
    ]
    completed = [r for r in rows if r["status"] == "completed"]
    numeric = [r for r in measured if r["status"] == "completed"]
    decisions = all(r["numerator"] == 0 for r in numeric)
    envelopes = all(r["probability_error"] <= r["interval_radius"] for r in numeric)
    permissions = all(
        r["final_action"] != "accept" or r["baseline_action"] == "accept" for r in numeric
    )
    parity = decisions and envelopes and permissions
    ready = int(parity and any(data["branches"].values()))
    kind = (
        "disqualified"
        if not parity
        else "circular_positive"
        if ready and data.get("fixture")
        else "null"
        if ready
        else "blocked"
    )
    fallback = [
        dict(
            unit_id=r["unit_id"],
            source_cluster_id=r["source_cluster_id"],
            arm=r["arm"],
            precision=r["precision"],
            fallback=r["fallback"],
            measured_cpu_fallback_ns=r["fallback_ns"],
        )
        for r in measured
        if r["status"] == "completed" and r["precision"] != "float64"
    ]
    return dict(
        honest_verdict="complete_" + kind + "_hardware_workload_obligations",
        verdict_class=kind,
        hardware_boundary_ready_score=ready,
        branch_readiness=dispositions,
        cost_obligation_checks=cost_obligations,
        precision_rows=measured,
        fallback_cost_rows=fallback,
        exact_fallback_count=sum(r["fallback"] for r in fallback),
        measured_cpu_fallback_ns=sum(r["measured_cpu_fallback_ns"] for r in fallback),
        workload_rows=workloads,
        amdahl_bounds=bounds,
        storage_probe_rows=probes,
        board_rows=board_rows,
        reopen_conditions=REOPEN,
        current_device_execution_count=0,
        rows=rows,
        intended_count=len(rows),
        completed_count=len(completed),
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        independent_count=0,
        verifier_is_oracle=bool(data.get("fixture") or data["learning"].get("verifier_is_oracle")),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        acceptance_gates=dict(
            final_decision_parity=decisions,
            numeric_envelopes=envelopes,
            acceptance_permission=permissions,
            qualified_reducer=bool(ready),
        ),
        hardware_decisions=[
            dict(tier=1, substrate="CPU", obligation="integer counters and updates"),
            dict(
                tier=2,
                substrate="CPU/Rust",
                obligation="radial memory; FPGA batching requires demand and compatible measured dispatch",
            ),
            dict(
                tier=3,
                substrate="GPU/NPU only with access",
                obligation="prediction; NPU local SDK/operator/dispatch gap remains",
            ),
            dict(
                tier=4,
                substrate="future fabric reconfiguration",
                obligation="structure changes require new authenticated dispatch",
            ),
        ],
        access_obligations=[
            dict(
                device=n,
                status="blocked_authenticated_access",
                vendor_evidence_is_local=False,
                reopen_condition=REOPEN[n],
            )
            for n in ["NPU", "TSU"]
        ],
    )


def authenticate(path: Path, eid: int, score: str, raw: Path, data: Json) -> tuple[Json, bool]:
    """Respect the prospective producer's receipt location without rewriting its primary."""
    if eid != 8214:
        return custody.authenticate(path, eid, score, raw, data)
    begin = len(data["checks"])
    value: Json = {}
    data["checks"].append(operand("exists", path, True, path.is_file()))
    try:
        frozen = copy_bytes(path, raw)
        data["references"].append(frozen)
        value = json.loads(Path(frozen["frozen_path"]).read_bytes())
        for field, wanted in [
            ("experiment_id", eid),
            (score, 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            data["checks"].append(operand(field, path, wanted, value.get(field)))
        receipt = Path(value["raw_path"]) / "publication_receipt.json"
        publication = json.loads(receipt.read_bytes())["publication"]
        sidecar = read_bound_sidecar(path, Path(publication["sidecar_path"]))
        for field, wanted, observed in [
            ("publication.primary_path", str(path.absolute()), publication["primary_path"]),
            ("publication.primary_sha256", sha256_file(path), publication["primary_sha256"]),
            ("report.passed", True, sidecar["report"]["passed"]),
        ]:
            data["checks"].append(operand(field, path, wanted, observed))
        data["references"] += [
            copy_bytes(p, raw) for p in [receipt, Path(publication["sidecar_path"])]
        ]
    except (OSError, ValueError, KeyError, TypeError) as error:
        data["checks"].append(operand("publication_receipt", path, "byte-bound pass", str(error)))
    return value, all(c["passed"] for c in data["checks"][begin:])

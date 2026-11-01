"""REQ-REPORT-8081: historical custody does not depend on cache or learning success."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_feature_8068 as old
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = old.ROOT
CUSTODY = old.CUSTODY
MODES = {8078: ("cold", "warm", "all_miss"), 8079: ("changed_10pct", "eviction", "restart")}
ARMS = ("python_cached", "native_cached", "python_uncached", "native_uncached")
CELLS = (
    ("feedback_constrained", "accepted"),
    ("unconstrained", "accepted"),
    ("feedback_constrained", "rejected"),
)
INPUTS = {
    8068: (
        "results/experiment_8068_v698_hardware_feature_boundary.json",
        "hardware_custody_ready_score",
    ),
    8078: ("results/experiment_8078_v699_feature_cache_core.json", "cache_core_ready_score"),
    8079: (
        "results/experiment_8079_v699_feature_cache_lifecycle.json",
        "cache_lifecycle_ready_score",
    ),
    8075: (
        "results/experiment_8075_v699_constraint_projection_kernel.json",
        "projection_kernel_ready_score",
    ),
    8076: (
        "results/experiment_8076_v699_projected_online_learning.json",
        "learning_trajectory_ready_score",
    ),
}
RESOURCES = list(
    dict.fromkeys(
        [
            *old.RESOURCES,
            "python/carnot/reporting/hardware_feature_8068.py",
            "python/carnot/experiment_8068_v698_hardware_feature_boundary.py",
            "results/experiment_8068_v698_hardware_feature_boundary.json",
            "python/carnot/reporting/hardware_workload_8081.py",
            "python/carnot/experiment_8081_v699_hardware_workload_boundary.py",
            "scripts/experiments/experiment_8081_v699_hardware_workload_boundary.py",
            "tests/python/test_hardware_workload_8081.py",
        ]
    )
)


def authenticate(root: Path, eid: int, raw: Path, data: Json) -> tuple[Json, bool]:
    """A clean terminal binding grants timing use only in the producer's own scope."""
    path = root / INPUTS[eid][0]
    before = len(data["checks"])
    saved = old.seal(path, None, raw, data, f"exp{eid}")
    if saved is None:
        return {}, False
    value = json.loads(saved.read_bytes())
    classes = {"positive", "null"} if eid in MODES else {"positive", "null", "circular_positive"}
    if eid == 8068:
        classes = {"positive", "null", "blocked"}
    expected = dict(
        experiment_id=eid,
        required_checks_passed=True,
        flagged_adversarial=False,
        **{INPUTS[eid][1]: 1},
    )
    for field, wanted in expected.items():
        data["checks"].append(old.operand(f"exp{eid}", path, field, wanted, value.get(field)))
    data["checks"].append(
        old.operand(
            f"exp{eid}",
            path,
            "verdict_class",
            sorted(classes),
            sorted(classes)
            if value.get("verdict_class") in classes
            else value.get("verdict_class"),
        )
    )
    try:
        terminal = Path(value["terminal_validation_sidecar_path"])
        binding = json.loads(terminal.read_bytes())["publication"]
        report = read_bound_sidecar(path, Path(binding["sidecar_path"]))
        for field, wanted, observed in (
            ("publication.primary_sha256", sha256_file(path), binding["primary_sha256"]),
            ("publication.primary_path", str(path), binding["primary_path"]),
            ("report.passed", True, report["report"]["passed"]),
        ):
            data["checks"].append(old.operand(f"exp{eid}", path, field, wanted, observed))
        for p in (terminal, Path(binding["sidecar_path"])):
            old.seal(p, None, raw, data, f"exp{eid}")
        for ref in value["raw_shard_hashes"]:
            p = Path(ref["path"])
            old.seal(p if p.is_absolute() else root / p, ref["sha256"], raw, data, f"exp{eid}")
    except (OSError, KeyError, ValueError) as error:
        data["checks"].append(
            old.operand(f"exp{eid}", path, "terminal_contract", "byte-bound pass", str(error))
        )
    return value, all(r["passed"] for r in data["checks"][before:])


def load(root: Path, raw: Path) -> Json:
    """Read immutable inputs without invoking models, devices or benchmark kernels."""
    data: Json = dict(
        boards=[],
        partitions={},
        checks=[],
        references=[],
        fixture=False,
        projection_receipts=[],
        trained_head_specs=[],
    )
    for resource in RESOURCES:
        old.seal(root / resource, None, raw, data, "local_resource")
    prior, qualified = old.authenticate(root, CUSTODY, 8055, raw, data)
    for original in prior.get("board_rows", []):
        board = deepcopy(original)
        before = len(data["checks"])
        path = root / board["source_path"]
        saved = old.seal(path, board["source_hash"], raw, data, board["board"])
        receipt = json.loads(saved.read_bytes()) if saved else {}
        transcript = receipt.get("kv260_terminal_transcript_path") or receipt.get(
            "raw_dispatch_transcript_path"
        )
        digest = receipt.get("kv260_terminal_transcript_sha256") or next(
            (
                r.get("latest_receipt_hash")
                for r in receipt.get("board_rows", [])
                if r.get("board") == board["board"]
            ),
            None,
        )
        if transcript and digest:
            old.seal(
                root / transcript,
                digest if digest.startswith("sha256:") else "sha256:" + digest,
                raw,
                data,
                board["board"],
            )
        board["custody_valid"] = qualified and all(r["passed"] for r in data["checks"][before:])
        data["boards"].append(board)
    for eid in INPUTS:
        print(
            f"[exp8081] authenticate exp{eid} completed={len(data['partitions'])} pending=exp{eid}",
            flush=True,
        )
        value, valid = authenticate(root, eid, raw, data)
        if eid in MODES:
            data["partitions"][str(eid)] = dict(
                qualified=valid,
                path=str(root / INPUTS[eid][0]),
                rows=[
                    deepcopy(r)
                    for r in value.get("rows", [])
                    if r["status"] == "completed" and r.get("repetition", -1) >= 0
                ]
                if valid
                else [],
                population_rows=value.get("population_rows", []) if valid else [],
            )
        elif valid and eid in {8075, 8076}:
            data["projection_receipts"].append(
                dict(
                    upstream=f"exp{eid}",
                    claim_scope=value["claim_scope"],
                    verifier_is_oracle=value["verifier_is_oracle"],
                    operation_counts=value.get("hardware_operation_counts", {}),
                    timing_rows=value.get(
                        "projection_cost_rows", value.get("cpu_update_costs", [])
                    ),
                )
            )
            data["trained_head_specs"].extend(value.get("trained_head_specs", []))
    return data


def reduce(data: Json) -> Json:
    """Charge complete host intervals before applying a hypothetical arithmetic gain."""
    checks, boards = deepcopy(data["checks"]), deepcopy(data["boards"])
    contracts = {
        "KV260": ("fpga_fabric", 5, None),
        "PolarFire": ("linux_cpu", None, None),
        "GateMate": ("none", None, "0xffffffff"),
    }
    for board in boards:
        valid = (
            board.get("processor_class"),
            board.get("k_max"),
            board.get("blocker"),
        ) == contracts.get(board["board"])
        valid = (
            valid and board.get("current_hardware_execution") is False and board["custody_valid"]
        )
        board["custody_valid"] = bool(valid)
        checks.append(
            old.operand(
                board["board"], Path(board["source_path"]), "board_scope", True, bool(valid)
            )
        )
    custody = {b["board"] for b in boards} == set(contracts) and all(
        b["custody_valid"] for b in boards
    )
    if not custody and not boards:
        checks.append(old.operand("custody", ROOT / CUSTODY, "board_rows", sorted(contracts), []))
    modes, bounds, fractions = [], [], []
    required = {
        "gradient_arithmetic_ns",
        "cache_extraction_ns",
        "cache_hashing_ns",
        "cache_storage_ns",
        "storage_fsync_ns",
        "guard_scans_ns",
    }
    expected = {
        (arm, condition, kind, rep)
        for arm in ARMS
        for condition, kind in CELLS
        for rep in range(30)
    }
    for eid, names in MODES.items():
        partition = data["partitions"].get(str(eid), {})
        path = Path(partition.get("path", str(ROOT / INPUTS[eid][0])))
        for mode in names:
            rows = [r for r in partition.get("rows", []) if r["mode"] == mode]
            identities = {
                (r["arm"], r["condition"], r["transaction_class"], r["repetition"]) for r in rows
            }
            valid = (
                partition.get("qualified", False)
                and identities == expected
                and len(rows) == len(expected)
            )
            groups: dict[tuple[str, str, str], list[Json]] = defaultdict(list)
            for row in rows:
                components = row.get("components", {})
                amounts = [row.get("transaction_ns"), *components.values()]
                cost_valid = (
                    required <= components.keys()
                    and all(
                        type(v) in {int, float} and math.isfinite(v) and v >= 0 for v in amounts
                    )
                    and row["transaction_ns"] > 0
                    and sum(components.values()) == row["transaction_ns"]
                )
                if not cost_valid:
                    checks.append(
                        old.operand(
                            f"exp{eid}",
                            path,
                            f"rows[{row['unit']}].cost_contract",
                            "finite nonnegative disjoint components sum to positive transaction",
                            dict(transaction_ns=row.get("transaction_ns"), components=components),
                        )
                    )
                valid = valid and cost_valid
                groups[(row["arm"], row["condition"], row["transaction_class"])].append(row)
            modes.append(
                dict(
                    unit=mode,
                    source=f"exp{eid}",
                    arm="qualification",
                    seed=None,
                    condition=mode,
                    numerator=int(valid),
                    denominator=1,
                    qualified=bool(valid),
                    status="completed" if valid else "excluded",
                    exclusion_reason=None
                    if valid
                    else "missing_or_unqualified_complete_cost_cells",
                    observed_count=len(rows),
                    required_count=len(expected),
                )
            )
            if not valid:
                checks.append(
                    old.operand(
                        f"exp{eid}",
                        path,
                        f"completed_cost_cells[{mode}]",
                        dict(qualified=True, unique_count=len(expected)),
                        dict(
                            qualified=bool(partition.get("qualified")),
                            unique_count=len(identities),
                            cost_contract_passed=bool(valid),
                        ),
                    )
                )
                continue
            for (arm, condition, kind), members in sorted(groups.items()):
                components = {
                    key: sum(r["components"].get(key, 0) for r in members)
                    for key in set().union(*(r["components"] for r in members))
                }
                population = sum(
                    r["elapsed_ns"]
                    for r in partition.get("population_rows", [])
                    if (r["mode"], r["condition"], r["transaction_class"], r["arm"] + "_cached")
                    == (mode, condition, kind, arm)
                )
                components["separate_population_shutdown_ns"] = population
                total = sum(r["transaction_ns"] for r in members) + population
                arithmetic = components["gradient_arithmetic_ns"]
                f = arithmetic / total
                identity = f"{mode}/{condition}/{kind}/{arm}"
                fractions.append(
                    dict(
                        unit=identity,
                        source=f"exp{eid}",
                        arm=arm,
                        seed=None,
                        condition=condition,
                        transaction_class=kind,
                        mode=mode,
                        numerator=arithmetic,
                        denominator=total,
                        eligible_fraction=f,
                        component_fractions={k: v / total for k, v in components.items()},
                        measured_components_ns=components,
                        projection_ns=None,
                        primitive_units=[r["unit"] for r in members],
                        status="completed",
                        exclusion_reason=None,
                    )
                )
                bounds.append(
                    dict(
                        unit=identity,
                        source=f"exp{eid}",
                        arm=arm,
                        seed=None,
                        condition=condition,
                        mode=mode,
                        transaction_class=kind,
                        eligible_fraction=f,
                        numerator=total,
                        denominator=total - arithmetic + arithmetic / 100,
                        arithmetic_only100x_ceiling=1 / ((1 - f) + f / 100),
                        infinite_arithmetic_ceiling=1 / (1 - f) if f < 1 else None,
                        break_even_transfer_queue_budget_ns=arithmetic * 0.99,
                        device_transfer_ns=None,
                        queue_ns=None,
                        projection_ns=None,
                        status="completed",
                        exclusion_reason=None,
                        symbolic_speedup="T/(T-A+A/100+transfer+queue)",
                        claim_scope="hypothetical arithmetic-only host interval; no complete-service or device speed claim",
                    )
                )
    combined = all(r["qualified"] for r in modes)
    blocked = not custody or not combined or any(not c["passed"] for c in checks)
    rows = [
        dict(
            unit="historical_custody",
            source=b["board"],
            arm="read_only",
            seed=None,
            condition="historical",
            numerator=int(b["custody_valid"]),
            denominator=1,
            status="completed" if b["custody_valid"] else "excluded",
            exclusion_reason=None if b["custody_valid"] else "unqualified_custody",
        )
        for b in boards
    ] + modes
    completed = sum(r["status"] == "completed" for r in rows)
    return dict(
        honest_verdict="complete_blocked_external_cost_or_custody"
        if blocked
        else "complete_circular_positive_private_fixture"
        if data["fixture"]
        else "complete_null_conditional_hardware_workload_boundary",
        verdict_class="blocked" if blocked else "circular_positive" if data["fixture"] else "null",
        verifier_is_oracle=bool(data["fixture"]),
        hardware_custody_ready_score=int(custody),
        combined_workload_bound_ready_score=int(combined),
        current_device_execution_count=0,
        board_rows=boards,
        mode_qualification_rows=modes,
        acceleration_bounds=bounds,
        compatible_component_fractions=fractions,
        rows=rows,
        gate_check_summary=[r for r in checks if not r["passed"]],
        intended_count=len(rows),
        eligible_count=completed,
        completed_count=completed,
        excluded_count=len(rows) - completed,
        independent_count=0,
        censored_count=0,
        failed_count=0,
        sample_size_budget=dict(
            independent_unit="historical obligations and one exposed host workload",
            current_scientific_samples=0,
            per_cell_repetitions=30,
        ),
        generalized_learning_benefit_score=0,
        ideal_f1_hypothetical100x=100,
        projection_receipts=data["projection_receipts"],
        trained_head_specs=data["trained_head_specs"],
        projection_precision_obligations=dict(
            device_accelerator_qualified=False,
            mapping="Sparse row dots and bound checks: CPU/Rust; optional GPU batching requires measured transfer and residual parity.",
            policy="Float64 full residual checks <=1e-8; frozen augmented coefficient/calibration; overflow and threshold uncertainty checks; authoritative CPU fallback before commit.",
            timing_scope="Projection receipts are separate host workloads; never splice into cache fractions.",
        ),
        compatibility_map=dict(
            KV260="SSH kria; quadratic Ising fabric k_max<=5; no projection accelerator",
            PolarFire="Linux CPU dispatch only",
            GateMate="physical/JTAG0xffffffff blocker; dated physical change before probe",
            NPU="unqualified",
            TSU="unqualified; vendor throughput is not a local measurement",
        ),
        missing_cost_components=[
            "source acquisition",
            "original model inference",
            "external feedback",
            "device transfer",
            "device queue",
            "projection within cache transactions",
        ],
        purchase_recommendation="none",
    )

"""REQ-REPORT-8550: sealed scope evidence keeps future hardware separate from observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import nine_input_board_evidence_8538 as old
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
failure = old.failure
COST = "results/experiment_8546_v734_complete_native_cost.json"
HISTORY = "results/experiment_8538_v733_nine_input_board_boundary.json"
HISTORY_PIN = "sha256:ab5d6cadea6a98f92746403efb46edc49661d9c556016aba17e715ba856952eb"
OPERATIONS = old.COST_OPERATIONS
BOARDS = ["KV260", "PolarFire", "GateMate", "NPU", "TSU"]
ARITHMETIC = ["scaling", "raw_logit_offset", "cubic_residual"]
VENDOR = [
    dict(
        source="https://extropic.ai/writing/z1t/",
        inspected_date="20261015",
        scope="vendor_theoretical_Z1_and_FPGA_energy_estimates",
        included_operations=["sparse_tanh_sampling", "FPGA_digital_work"],
        excluded_operations=["dense_final_logits", "inter_device_movement"],
        local_measurement=False,
        carnot_efficiency_ratio=None,
        comparison="scope_and_exclusions_only",
    )
]
LITERATURE = [
    dict(
        source=url,
        scope=scope,
        status="future_hardware_hypothesis",
        local_measurement=False,
        installed_nine_input_kernel=False,
    )
    for url, scope in [
        ("https://arxiv.org/abs/2607.21077", "pipelined_quadratic_Ising_and_memory_architecture"),
        ("https://www.nature.com/articles/s41467-026-72020-8", "250_STT_MTJs_FPGA_sampling"),
    ]
]


def history(root: Path) -> Json:
    """Independent pins prevent a repaired summary hash from rewriting board history."""
    result = dict(old.history(root))
    path = root / HISTORY
    try:
        if old.reference(path)["sha256"] != HISTORY_PIN:
            raise ValueError("original_board_pin")
        value = json.loads(path.read_bytes())
        if value["run_date"] != "20261015" or value["required_checks_passed"] is not True:
            raise ValueError("original_board_date_or_checks")
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        if (
            read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))["report"][
                "passed"
            ]
            is not True
        ):
            raise ValueError("original_board_terminal")
        result["refs"].append(old.reference(path))
        result["readers"].append(dict(path=str(path), sha256=HISTORY_PIN, run_date="20261015"))
    except (OSError, ValueError, KeyError, TypeError):
        result["qualified"] = False
        result["checks"].append(
            old.failure(
                path,
                "original_board_seal",
                HISTORY_PIN,
                old.reference(path)["sha256"] if path.is_file() else None,
                "exp8538-nine-input-board-boundary",
            )
        )
    return result


def ceiling(spans: list[Json]) -> Json:
    """Zero arithmetic cost is a conditional upper bound; all fixed host work remains charged."""
    return dict(old.ceiling(spans))


def costs(root: Path) -> Json:
    """A future producer must supply bound terminal bytes and reconciled complete primitive clocks."""
    path = root / COST
    result: Json = dict(
        path=str(path), qualified=False, checks=[], refs=[], spans=[], conditional_ceiling=None
    )
    expected: Any
    observed: Any
    field, expected, observed = "complete_native_cost_ready_score", 1, None
    try:
        value = json.loads(path.read_bytes())
        result["refs"].append(old.reference(path))
        for field, expected in dict(
            complete_native_cost_ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
            experiment_id=8546,
            task_id="exp8546-complete-native-cost",
            run_date="20261015",
        ).items():
            observed = value.get(field)
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(field)
        field, expected, observed = "bound_terminal_bytes", True, False
        validate_primary(value, path)
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        if (
            read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))["report"][
                "passed"
            ]
            is not True
        ):
            raise ValueError(field)
        field, expected, observed = "reconciled_complete_operation_clocks", True, False
        ref = value["disjoint_complete_spans_reference"]
        primitive = json.loads(old.base.checked(ref).read_bytes())
        bound = ceiling(primitive["spans"])
        if (
            primitive.get("input_count") != 9
            or primitive.get("coefficient_count") != 109
            or primitive.get("reconciled") is not True
            or primitive.get("complete_clock_ns") != bound["complete_clock_ns"]
        ):
            raise ValueError(field)
        result.update(qualified=True, spans=primitive["spans"], conditional_ceiling=bound)
        result["refs"].append(ref)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        result["checks"].append(
            old.failure(path, field, expected, observed, "exp8546-complete-native-cost")
        )
    return result


def ceilings(cost: Json) -> list[Json]:
    """Unknown device costs remain separate from an ideal upper bound, never a measurement."""
    rows = []
    for subset in [ARITHMETIC, ["cubic_residual"]]:
        total = cost["conditional_ceiling"]["complete_clock_ns"] if cost["qualified"] else None
        eligible = (
            sum(r["end_ns"] - r["start_ns"] for r in cost["spans"] if r["operation"] in subset)
            if cost["qualified"]
            else None
        )
        fixed = total - eligible if total is not None else None
        gain = total / fixed if fixed else None
        rows.append(
            dict(
                candidate_operations=subset,
                complete_clock_ns=total,
                offloaded_clock_ns=eligible,
                fixed_host_clock_ns=fixed,
                zero_cost_ceiling=gain,
                measured_acceleration=None,
                nfr01_boundary=("insufficient_for_NFR01" if gain < 10 else "conditional_only")
                if gain is not None
                else "unknown_missing_qualified_cost",
                unknown_costs=dict(transfer=None, precision=None, update=None, sampling=None),
                vendor_exclusions=VENDOR,
                conditional=True,
                missing_reason=None if cost["qualified"] else "Exp8546_not_qualified",
            )
        )
    return rows

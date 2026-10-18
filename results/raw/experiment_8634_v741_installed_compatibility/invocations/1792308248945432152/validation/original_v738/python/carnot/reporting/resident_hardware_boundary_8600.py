"""REQ-REPORT-8600: explicit adapters preserve board history and fresh resident operands."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import resident_hardware_boundary_8589 as base
from carnot.reporting import resident_hardware_evidence_8600 as evidence
from carnot.reporting import v738_contract_methods as authority

Json = dict[str, Any]
ROOT, reference, checksum = base.ROOT, base.reference, base.checksum
NAME, TASK = "experiment_8600_v738_resident_hardware_boundary", "exp8600-resident-hardware-boundary"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_resident_hardware_boundary_8600.py"
OWNED = [
    f"python/carnot/reporting/resident_hardware_{s}_8600.py"
    for s in ["evidence", "boundary", "runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
_measure, _build, _replay = base.measure, base.build, base.replay


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase counts because no device or model command is needed."""
    print(f"[exp8600] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def bindings() -> Iterator[None]:
    """Limit identity changes to this invocation so historical readers retain their authority."""
    with patch.multiple(
        base,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        authority=authority,
        evidence=evidence,
        build=build,
        progress=progress,
    ):
        yield


def measure(root: Path, raw: Path, private: Path, run_date: str | None = None) -> Json:
    """The original bounded resource and evidence checks run before any reduction."""
    with bindings():
        work = dict(_measure(root, raw, private))
    work["run_date"] = run_date or datetime.now().strftime("%Y%m%d")
    work["code_config_hashes"] += [
        reference(ROOT / "docs/research-notes/v738-resident-hardware-boundary.md")
    ]
    work["code_config_hashes"] += [
        reference(Path(module.__file__))
        for module in [base, evidence.base, authority, evidence.resident, evidence.resident.base]
    ]
    literature = ROOT / "results/raw" / NAME / "literature/ingestion.json"
    work["refs"] += [reference(literature), *json.loads(literature.read_bytes())]
    for row in work["checks"]:
        row["field"] = row["field"].replace("twelve_object", "eleven_object")
        row["check"] = row["check"].replace("twelve_object", "eleven_object")
    base.atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Mapping readiness and CPU residence never imply an installed compatible device kernel."""
    with bindings():
        value: Json = deepcopy(_build(work, receipts, raw, output))
    cost = work["evidence"]["cost"]
    original = cost["value"] or {}
    value.update(
        experiment_id=8600,
        milestone="2026.10.738",
        run_date=work["run_date"],
        honest_verdict=f"complete_{value['verdict_class']}_v738_resident_hardware_boundary",
        target_model=original.get("target_model") if cost["qualified"] else None,
        numerical_inference_this_invocation=bool(cost["qualified"]),
    )
    gates = value["acceptance_gates"]
    gates["eleven_object_v738_contract"] = gates.pop("activated_twelve_object_contract")
    value["future_dependencies"] = [
        dict(
            experiment_id=8596,
            status="qualified_current_evidence"
            if cost["qualified"]
            else "future_dependency_not_qualified_observation",
        )
    ]
    value["acquisition_decision"].update(
        cpu_residence_removes_bottleneck=bool(
            original["lifecycle_contrast"]["throughput_ratio_ci95"][0] > 1
            and original["lifecycle_contrast"]["p95_request_latency_ratio"] < 1
        )
        if cost["qualified"]
        else "unmeasured_without_qualified_complete_resident_cost",
        measured_resident_acceleration=original.get("lifecycle_contrast")
        if cost["qualified"]
        else None,
        missing_reason="no_compatible_installed_kernel",
        cpu_scope="Qualified Exp8596 removes repeated cold startup; it does not establish hardware acceleration or durable learning.",
    )
    value["preserved_historical_outcomes"] += [
        "V737 completed twelve scheduling slots, not twelve successful studies",
        "Exp8580 current validator qualified",
        "Exp8581 causal accounting qualified",
        "Exp8588 qualified ARC null; no refinement support",
        "Completed archive ends at V736; V737 primaries and conductor receipts are newer evidence",
    ]
    value["preserved_quarantines"].update(
        Exp8582="runner-clock regression; numbers quarantined",
        Exp8584="methodology provenance failure; numbers quarantined",
        original_v738_protocol=authority.PROTOCOL,
        original_v737_protocol="openspec/change-proposals/v737-methods-protocol.json",
    )
    value["literature_operation_mapping"] += [
        dict(
            source="https://arxiv.org/abs/2610.03972",
            operations=["distribution_sampling", "argmin_optimization"],
            deterministic_spline_evaluation=False,
            local_speedup=None,
            distinction="Distribution quality requires sampling tests; finding a minimum does not establish a distribution or deterministic spline parity.",
        ),
        dict(
            source="https://github.com/Duchstf/KANELE",
            operations=["deterministic_spline_evaluation"],
            devices=["xcvu9p-flgb2104-2-i", "xczu7ev-ffvc1156-2-e"],
            toolchain="Vivado 2024.1 RTL simulation and device-specific build",
            installed_KV260_spline_evaluator=False,
        ),
    ]
    value["field_principles"]["qualified_cost_reference"] = (
        "Only original qualified Exp8596 complete rows, parity and immutable producer replay authorize resident operands."
    )
    value["field_principles"]["target_model"] = (
        "Imported authenticated small numerical head and checkpoint; this invocation loads no model and performs no inference."
    )
    value["field_principles"]["numerical_inference_this_invocation"] = (
        "True records original small-head numerical parity reconstruction; this invocation makes zero language-model calls."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh reductions compare independently pinned original evidence with every summary field."""
    with bindings():
        return bool(_replay(path))

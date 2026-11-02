"""REQ-REPORT-8625: accounting readiness never grants physical acceleration or learning benefit."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import hardware_operation_boundary_8613 as base
from carnot.reporting import hardware_operation_evidence_8625 as evidence
from carnot.reporting import v740_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json

Json = dict[str, Any]
ROOT, reference, checksum = base.ROOT, base.reference, base.checksum
NAME, TASK = (
    "experiment_8625_v740_hardware_operation_boundary",
    "exp8625-hardware-operation-boundary",
)
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hardware_operation_boundary_8625.py"
OWNED = [
    f"python/carnot/reporting/hardware_operation_{s}_8625.py"
    for s in ["evidence", "boundary", "runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts because read-only work must still remain visible to its supervisor."""
    print(f"[exp8625] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def bindings() -> Iterator[None]:
    """Temporary identity bindings let historical reducers keep their original public authority."""
    with patch.multiple(
        base,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        authority=authority,
        evidence=evidence,
        progress=progress,
    ):
        yield


def measure(root: Path, raw: Path, private: Path, run_date: str) -> Json:
    """Authenticate resources and evidence before reducing existing clocks; no model is loaded."""
    with bindings():
        work = dict(base.measure(root, raw, private, run_date))
    methods = evidence.sealed(root, evidence.METHODS, "methods_ready_score")
    work["checks"] += methods["checks"]
    work["checks"].append(
        evidence.failure(
            root / evidence.METHODS,
            "methods_ready_score",
            1,
            1 if methods["qualified"] else None,
            "exp8615-contract-methods",
        )
    )
    work["contract"]["ready"] = work["contract"]["ready"] and methods["qualified"]
    work["refs"] += methods["refs"]
    for row in work["checks"]:
        for key in ["check", "field"]:
            row[key] = row[key].replace("thirteen_object_v739", "twelve_object_v740")
    work["code_config_hashes"] = [
        reference(ROOT / p)
        for p in [
            *OWNED,
            TEST,
            "ops/exclusion_manifest.yaml",
            authority.PROTOCOL,
            "docs/research-notes/v740-hardware-operation-boundary.md",
            "python/carnot/verify/sparse_energy_7996.py",
            "crates/carnot-core/src/direct_spline_8379.rs",
            "crates/carnot-core/src/nine_input_8533.rs",
        ]
    ]
    work["code_config_hashes"] += [
        reference(Path(m.__file__)) for m in [base, evidence.old, evidence.clocks, authority]
    ]
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Keep lifecycle savings, native arithmetic and hypothetical offload on separate boundaries."""
    ordered = dict(work, evidence={key: work["evidence"][key] for key in sorted(work["evidence"])})
    with bindings():
        value: Json = deepcopy(base.build(ordered, receipts, raw, output))
    cost = work["evidence"]["cost"]
    current = bool(cost["qualified"] and work["evidence"]["lean"]["qualified"])
    original = cost["value"] or {}
    value.update(
        experiment_id=8625,
        milestone="2026.10.740",
        honest_verdict=f"complete_{value['verdict_class']}_v740_hardware_operation_boundary",
        NPU_qualified=0,
        TSU_qualified=0,
        random_seed=7408625,
    )
    value["acceptance_gates"]["twelve_object_authority"] = value["acceptance_gates"].pop(
        "thirteen_object_authority"
    )
    value["qualified_cost_reference"].update(
        reference=reference(Path(work["root"]) / (evidence.LEAN if current else evidence.COST))
        if cost["qualified"]
        else None,
        historical_boundary=not current,
        boundary="complete lean resident launch-through-clean-exit sessions"
        if current
        else "historical Exp8596 complete resident sessions",
    )
    value["source_artifact_hashes"] += work["refs"]
    value["native_arithmetic_reference"] = original.get(
        "arithmetic_contrast" if current else "primary_contrast"
    )
    value["lifecycle_savings_reference"] = original.get(
        "primary_contrast" if current else "lifecycle_contrast"
    )
    value["learning_path"] = (
        "Released delayed feedback drives the 109-coefficient CPU spline head: at most37 data-gradient touches and one lazy global-decay write; checkpoint copies and feedback drain remain host work. Rust supports native arithmetic; the separate34-coefficient update kernel is not interchangeable with the109-coefficient head. No qualified Exp8618 update clocks exist."
    )
    value["learning_implementation_references"].append(
        reference(ROOT / "crates/carnot-core/src/nine_input_8533.rs")
    )
    value["future_dependencies"] = [
        dict(
            experiment_id=n,
            qualified=work["evidence"][key]["qualified"],
            status="qualified_current_evidence"
            if work["evidence"][key]["qualified"]
            else "future_dependency_not_qualified_observation",
        )
        for n, key in [(8622, "lean"), (8618, "learning")]
    ]
    value["preserved_historical_outcomes"] = (value["preserved_historical_outcomes"] or []) + [
        "V739: thirteen scheduling slots, six producer artifacts; live V738 design failed methods authority",
        "Exp8603 remains disqualified: original51 tests and three deadline cases passed;83 executable runner statements covered; continuation94 maps to92",
        "V733 source, V735 cold-cost and V737 ARC nulls remain closed; Exp8608 prototype qualified; repaired Exp8609 remained methods-blocked",
    ]
    value["branch_gate_values"] = {
        key: value[key]
        for key in [
            "hardware_boundary_ready_score",
            "physical_hardware_ready_score",
            "NPU_qualified",
            "TSU_qualified",
            "nfr01_score",
        ]
    }
    value["newly_owned_coverage"] = work.get("owned_coverage")
    value["acquisition_decision"]["measured_dominant_operations"] = (
        sorted(
            evidence.OPERATIONS,
            key=lambda op: sum(c["seconds"][op] for c in cost["cells"]),
            reverse=True,
        )[:3]
        if cost["qualified"]
        else []
    )
    value["field_principles"].update(
        qualified_cost_reference="Use pinned qualified Exp8622 clocks; otherwise explicitly retain historical Exp8596. No boundary is combined with missing learning clocks.",
        lifecycle_savings_reference="Matched legacy_native versus lean_native measures lifecycle work; it does not measure an accelerator.",
        native_arithmetic_reference="Matched lean_python versus lean_native retains its own lifecycle and does not meet10x NFR01.",
        NPU_qualified="Numeric0: no compatible physical NPU evidence.",
        TSU_qualified="Numeric0: no compatible physical TSU evidence.",
        branch_gate_values="Exact numeric0/1 mirror of top-level readiness; null costs are never zeros.",
    )
    value["field_principles"] = {
        key: value["field_principles"].get(
            key, "Invocation-bound original evidence; missing is distinct from zero."
        )
        for key in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reauthentication against independent producer pins rejects self-consistently rehashed summaries."""
    try:
        value = json.loads(path.read_bytes())
        ref = value["work_reference"]
        work = json.loads(Path(evidence.checked(ref)["path"]).read_bytes())
        if evidence.authenticate(Path(work["root"])) != work["evidence"]:
            return False
        for code in work["code_config_hashes"] + work["refs"]:
            evidence.checked(code)
        if work.get("validation_reference"):
            validation = json.loads(
                Path(evidence.checked(work["validation_reference"])["path"]).read_bytes()
            )
            if validation["receipts"] != value["validation_receipts"]:
                return False
        contract = authority.authority(Path(work["root"]))
        methods = evidence.sealed(Path(work["root"]), evidence.METHODS, "methods_ready_score")
        contract["ready"] = contract["ready"] and methods["qualified"]
        if contract != work["contract"]:
            return False
        return bool(
            build(
                work,
                value["validation_receipts"],
                Path(ref["path"]).parent,
                Path(value["publication_output"]),
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False

"""REQ-REPORT-8596: reuse complete costs after authenticating numerical head provenance."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import resident_complete_cost_8584 as base
from carnot.reporting import resident_provenance_8593 as provenance
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
__all__ = ["base"]
ROOT = base.ROOT
NAME = "experiment_8596_v738_resident_complete_cost"
TASK = "exp8596-resident-complete-cost"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_resident_complete_cost_8596.py"
OWNED = [
    "python/carnot/reporting/resident_complete_cost_8596.py",
    "python/carnot/reporting/resident_complete_runner_8596.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PINS = {
    "results/experiment_8591_v738_contract_methods.json": "sha256:d0fc7e60a2054ecebac3b21eb6cb452a4671b8b1a24b5ce20881cbb4147ad837",
    "results/experiment_8593_v738_resident_provenance_qualification.json": "sha256:d19461a13336e5a9563013d49a9b14dfd78aebec8e48064e751c3f7bef44a4eb",
}
_admit, _artifact, _measure = base.admit, base.artifact, base.measure
REUSED = [
    "python/carnot/reporting/resident_complete_cost_8584.py",
    "python/carnot/reporting/resident_complete_worker_8584.py",
    "python/carnot/reporting/resident_complete_runner_8584.py",
    "python/carnot/reporting/resident_validation_8572.py",
    "python/carnot/reporting/complete_native_worker_8559.py",
    "python/carnot/verify/nine_input_8533.py",
    "python/carnot/verify/native_nine_input_8533.py",
]


def admit(root: Path) -> Json:
    """Original producer seals and bound publication receipts qualify fresh clocks only."""
    values, refs = [], []
    for name, digest in PINS.items():
        path = root / name
        ref = dict(path=str(path), sha256=digest)
        value = base.read(ref)
        for field, expected in dict(required_checks_passed=True, flagged_adversarial=False).items():
            base.require(path, field, expected, value.get(field))
        terminal = Path(value["terminal_validation_sidecar_path"])
        sidecar = Path(json.loads(terminal.read_bytes())["publication"]["sidecar_path"])
        base.require(
            path, "bound_publication", True, read_bound_sidecar(path, sidecar)["report"]["passed"]
        )
        for dependency in value["code_config_hashes"]:
            base.require(
                Path(dependency["path"]),
                "qualified_dependency_bytes",
                dependency["sha256"],
                sha256_file(Path(dependency["path"])),
            )
        refs.extend([ref, base.reference(terminal), base.reference(sidecar)])
        refs.extend(value["code_config_hashes"])
        values.append(value)
    methods, qualified = values
    for field in ["resident_inputs_ready_score", "methods_ready_score"]:
        base.require(root / list(PINS)[0], field, 1, methods.get(field))
    base.require(
        root / list(PINS)[1],
        "resident_measurement_ready_score",
        1,
        qualified.get("resident_measurement_ready_score"),
    )
    # Descriptor reconstruction uses the historical consumer under its original identity.
    with patch.object(base, "admit", _admit):
        bundle = _admit(root)
        target = provenance.descriptor(root)
    base.require(root, "authenticated_target_model", target, qualified.get("target_model"))
    protocol = base.read(methods["methods_protocol_reference"])
    base.require(
        root, "unchanged_resident_protocol", bundle["protocol"], protocol["resident_session"]
    )
    refs.extend([methods["methods_protocol_reference"], qualified["head_descriptor_reference"]])
    refs.extend(base.reference(ROOT / name) for name in REUSED)
    return dict(
        bundle,
        refs=bundle["refs"] + refs,
        target_model=target,
        preserved_quarantines=methods["preserved_quarantines"],
        provenance_qualification_reference=base.reference(root / list(PINS)[1]),
        parity_revalidation="No affected dependency changed; original parity receipts retained.",
    )


def measure(root: Path, raw: Path, run_date: str) -> Json:
    """The unchanged fixed roster runs only after admission; every pair is checkpointed."""
    work = dict(_measure(root, raw))
    work["run_date"] = run_date
    for row in work["gate_check_summary"]:
        row["upstream"] = next((name for name in PINS if row["path"].endswith(name)), TASK)
    atomic_json(raw / "measurement.json", work)
    return work


def artifact(work: Json, receipts: list[Json], output: Path) -> Json:
    """Keep the registered reduction and benefit gates; add authenticated invocation identity."""
    value = dict(_artifact(work, receipts, output))
    bundle = work["bundle"]
    value.update(
        experiment_id=8596,
        task_id=TASK,
        milestone="2026.10.738",
        run_date=work["run_date"],
        target_model=bundle["target_model"] if bundle else None,
        honest_verdict=f"complete_{value['verdict_class']}_v738_resident_complete_cost",
    )
    missing = []
    for slot in base.roster()[len(work["rows"]) :]:
        missing.append(
            dict(
                slot,
                unit_id=f"L{slot['length']}-B{slot['batch']}-R{slot['repeat']}-{slot['arm']}",
                status="censored",
                intended_count=64 * slot["length"],
                completed_count=0,
                failed_count=0,
                censored_count=64 * slot["length"],
                primitive_reference=None,
                missing_reason="upstream_or_resource_gate",
            )
        )
    value["rows"].extend(missing)
    value["censoring_rows"].extend(missing)
    value["field_principles"]["target_model"] = (
        "Exact Exp8593-authenticated Exp8431 checkpoint, scaler, basis, extension and typed policy; no language-model load."
    )
    value["reproducibility_checksum"] = base.checksum(value)
    return value


@contextmanager
def bindings(run_date: str, input_root: Path | None = None) -> Iterator[None]:
    """Scoped adapters preserve historical files while existing consumers validate new bytes."""
    with patch.multiple(
        base,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        admit=admit,
        artifact=artifact,
        measure=lambda root, raw: measure(input_root or root, raw, run_date),
    ):
        yield

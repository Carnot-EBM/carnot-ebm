"""REQ-REPORT-8665: map authenticated cost operands without executing their workloads."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting import hardware_operation_evidence_8651 as previous
from carnot.reporting import installed_cost_8661 as installed
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
ROOT, reference, checked, checksum = (
    previous.ROOT,
    previous.reference,
    previous.checked,
    previous.checksum,
)
NAME, TASK = (
    "experiment_8665_v743_hardware_operation_evidence",
    "exp8665-hardware-operation-evidence",
)
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hardware_operation_evidence_8665.py"
OWNED = [
    "python/carnot/reporting/hardware_operation_evidence_8665.py",
    "python/carnot/reporting/hardware_evidence_runner_8665.py",
    "python/carnot/reporting/hardware_evidence_validation_8665.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PRODUCERS = [
    (8657, "causal_brier_stream", "learning_cost_ready_score", "delayed_learning"),
    (8661, "installed_complete_cost", "installed_cost_ready_score", "installed_inference"),
    (8659, "complete_source_comparison", "source_comparison_ready_score", "source_extraction"),
]
PINS = {
    "results/experiment_8651_v742_hardware_operation_evidence.json": "sha256:a96652bec42b3dfa0a93acf218fc105599fdb6ebc67d36263e2bff81f41b7187",
    previous.old.BOARD: previous.old.BOARD_PIN,
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts expose real phase boundaries without padding the run duration."""
    print(f"[exp8665] phase={phase} completed={completed} pending={pending}", flush=True)


def receipt(path: Path) -> Json:
    """Use the unchanged path-bound reader before trusting any producer's readiness."""
    side = (
        path.parent / "raw" / path.stem / "validators" / (reference(path)["sha256"][7:] + ".json")
    )
    value = read_bound_sidecar(path, side)
    if value["primary_path"] != str(path.absolute()) or value["report"]["passed"] is not True:
        raise ValueError("bound_publication_receipt")
    return reference(side)


def installed_cells(value: Json, refs: list[Json]) -> list[Json]:
    """Rebuild original frame partitions; opaque bindings cannot isolate spline execution."""
    authority = value["authority_reference"]
    work = json.loads(checked(authority).read_bytes())
    refs.append(authority)
    refs.append(value["target_model"]["checkpoint"])
    checked(refs[-1])
    cells = []
    for index, row in enumerate(work["rows"]):
        ref = row["primitive_reference"]
        primitive = json.loads(checked(ref).read_bytes())
        refs.extend([ref, primitive["config_reference"]])
        _, _, spans = installed.summarize(primitive, work["bundle"])
        seconds = {
            op: sum(s["end_ns"] - s["start_ns"] for s in spans if s["operation"] == op) / 1e9
            for op in sorted({s["operation"] for s in spans})
        }
        cells.append(
            dict(
                boundary="installed_inference",
                cohort=f"L{row['length']}-B{row['batch']}-R{row['repeat']}",
                arm=row["arm"],
                total_s=sum(seconds.values()),
                seconds=seconds,
                compatible_s=None,
                unaccelerated_share=None,
                conditional_ceiling=None,
                conditional_limit="missing_compatible_operation_span",
                achieved_speedup=None,
                primitive_reference=ref,
                missing_reason="binding_residual_does_not_isolate_spline_arithmetic",
            )
        )
        if (index + 1) % 20 == 0:
            progress("reduce_installed_primitives", index + 1, len(work["rows"]) - index - 1)
    if len(cells) != 120:
        raise ValueError("original_installed_120_cell_roster")
    return cells


def admit(root: Path, producer: tuple[int, str, str, str]) -> Json:
    """A sibling's absence never vetoes a qualified source; missing and zero stay distinct."""
    number, suffix, field, boundary = producer
    path = root / f"results/experiment_{number}_v743_{suffix}.json"
    result: Json = dict(
        experiment_id=number,
        boundary=boundary,
        qualified=False,
        cells=[],
        refs=[],
        checks=[],
        missing_reason="no_qualified_complete_primitive_closure",
    )
    observed, expected, state = None, 1, "missing_file"
    try:
        value = json.loads(path.read_bytes())
        result["refs"].append(reference(path))
        validate_primary(value, path)
        observed = value.get(field)
        state = (
            "missing_field"
            if field not in value
            else "observed_zero"
            if type(observed) is int and observed == 0
            else "observed"
        )
        if type(observed) is not int or observed != 1:
            raise ValueError(field)
        field, expected = "qualified_producer_checks", True
        gates = value["acceptance_gates"]
        findings = value.get("adversarial_findings", [])
        if isinstance(findings, dict):
            findings = [flag for report in findings["reports"] for flag in report["flags"]]
        observed = (
            value["required_checks_passed"] is True
            and value["task_id"] == f"exp{number}-{suffix.replace('_', '-')}"
            and value["verdict_class"] in {"null", "positive", "circular_positive"}
            and all(f["severity"].lower() == "info" for f in findings)
            and (
                any(
                    g["name"] == "qualified_complete_measurement" and g["passed"] is True
                    for g in gates
                )
                if isinstance(gates, list)
                else bool(gates) and all(v is True for v in gates.values())
            )
        )
        state = "observed"
        if not observed:
            raise ValueError(field)
        field, expected, observed = "bound_publication_receipt", True, False
        result["refs"].append(receipt(path))
        field, expected, observed = (
            "operation_cost_closure",
            "nonempty complete primitive references",
            value.get("operation_cost_closure"),
        )
        if observed:
            for ref in observed:
                primitive = checked(ref)
                if not primitive.resolve().is_relative_to(
                    (path.parent / "raw" / path.stem).resolve()
                ):
                    raise ValueError("foreign_primitive")
                result["cells"].append(
                    dict(
                        previous.reduce_spans(json.loads(primitive.read_bytes()), boundary),
                        primitive_reference=ref,
                    )
                )
                result["refs"].append(ref)
        elif number == 8661 and "authority_reference" in value:
            result["cells"] = installed_cells(value, result["refs"])
        else:
            result["reported_extraction_cost_count"] = len(
                value.get("complete_extraction_cost_rows", [])
            )
            state = "observed_empty" if field in value else "missing_field"
            raise ValueError("missing_complete_startup_transport_shutdown_operation_spans")
        if len({(c["cohort"], c["arm"]) for c in result["cells"]}) != len(result["cells"]):
            raise ValueError("duplicate_cohort_arm")
        result.update(qualified=True, missing_reason=None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(cells=[], missing_reason=str(error))
        result["checks"].append(
            dict(
                previous.failure(path, f"exp{number}", field, expected, observed),
                passed=False,
                observed_state=state,
                reason=str(error),
            )
        )
    return result


def authenticate(root: Path) -> Json:
    """Independent original pins retain historical authority without running old controllers."""
    board: Json = dict(qualified=False, value=None, refs=[], checks=[])
    for relative, digest in PINS.items():
        path = root / relative
        try:
            checked(dict(path=str(path), sha256=digest))
            value = json.loads(path.read_bytes())
            validate_primary(value, path)
            board["refs"].extend([reference(path), receipt(path)])
            if value["required_checks_passed"] is not True:
                raise ValueError("historical_owned_checks")
            if value["experiment_id"] == 8651:
                board["value"] = value
            checked(value["target_model"]["checkpoint"])
            board["refs"].append(value["target_model"]["checkpoint"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            board["checks"].append(
                dict(
                    previous.failure(
                        path,
                        path.stem,
                        "original_sha256",
                        digest,
                        reference(path)["sha256"] if path.is_file() else None,
                    ),
                    passed=False,
                    reason=str(error),
                )
            )
    board["qualified"] = not board["checks"]
    branches = [admit(root, producer) for producer in PRODUCERS]
    return dict(board=board, branches=branches)


def measure(root: Path, raw: Path, private: Path, run_date: str) -> Json:
    """Freeze exact source operands before validation; optional devices are not prerequisites."""
    progress("authenticate_originals_once_zero_model_device_calls", 0, 1)
    start = time.monotonic_ns()
    evidence = authenticate(root)
    auth_s = (time.monotonic_ns() - start) / 1e9
    private.mkdir(parents=True, exist_ok=True)
    probe = private / "disk_probe"
    probe.write_bytes(b"disk-backed scratch")
    probe.unlink()
    refs = [*evidence["board"]["refs"], *(r for b in evidence["branches"] for r in b["refs"])]
    snapshots = []
    for index, ref in enumerate(refs):
        destination = raw / "inputs" / f"{index:03d}-{Path(ref['path']).name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked(ref), destination)
        destination.chmod(0o444)
        snapshots.append(dict(reference(destination), original_reference=ref))
    work: Json = dict(
        root=str(root.absolute()),
        run_date=run_date,
        evidence=evidence,
        snapshots=snapshots,
        preconditions_checked=dict(
            python=sys.executable,
            tools={
                p: reference(ROOT / ".venv/bin" / p) for p in ["python", "pytest", "ruff", "mypy"]
            },
            memory_available=next(
                s
                for s in Path("/proc/meminfo").read_text().splitlines()
                if s.startswith("MemAvailable:")
            ),
            scratch_path=str(private),
            scratch_writable=True,
            scratch_device=private.stat().st_dev,
            scratch_free_bytes=shutil.disk_usage(private).free,
            scratch_filesystem="disk-backed /var/tmp on host root filesystem",
            optional_gpu_required=False,
            optional_board_required=False,
            board_authentication_s=auth_s,
            board_authenticated_within_60_seconds=auth_s <= 60,
        ),
        phase_spans=[
            dict(
                phase="authenticate_and_snapshot",
                start_ns=start,
                end_ns=time.monotonic_ns(),
                duration_s=(time.monotonic_ns() - start) / 1e9,
            )
        ],
        duration_s=(time.monotonic_ns() - start) / 1e9,
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "docs/research-notes/v743-hardware-operation-evidence.md",
                "ops/exclusion_manifest.yaml",
                "python/carnot/reporting/hardware_operation_evidence_8651.py",
                "python/carnot/reporting/installed_cost_8661.py",
                "python/carnot/reporting/primary_publication.py",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete_no_workload_rerun", len(refs), 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Keep report readiness independent of missing science and physical execution."""
    value = previous.build(deepcopy(work), receipts, raw, output)
    board, branches = work["evidence"]["board"], work["evidence"]["branches"]
    original = board["value"] or {}
    cells = [
        dict(c, producer=b["experiment_id"]) for b in branches if b["qualified"] for c in b["cells"]
    ]
    value.update(
        experiment_id=8665,
        task_id=TASK,
        milestone="2026.10.743",
        random_seed=7438665,
        honest_verdict=f"complete_{value['verdict_class']}_v743_hardware_operation_evidence",
        per_board_results=original.get("board_status_rows", []),
        kv260_evidence=dict(
            scope="quadratic Ising k<=5 only",
            reference=board["refs"][:1],
            native_current_head_support=False,
        ),
        polarfire_evidence=dict(
            scope="board-local Linux CPU dispatch only",
            reference=board["refs"][:1],
            native_current_head_support=False,
        ),
        gatemate_blocker=dict(
            required_order=[
                "dated physical cable/port/power change",
                "GM1Ax IDCODE0x20000001",
                "n16 flash and sample/hash evidence",
            ],
            probe_performed=False,
        ),
        npu_status="unqualified",
        tsu_status="unqualified",
        complete_boundary_rows=cells,
        conditional_acceleration_bounds=value["conditional_ceiling_rows"],
        acquisition_priority_delta=dict(
            changed=False,
            purchase_authorized=False,
            reason="Qualified installed costs retain opaque spline/binding time; no compatible physical kernel or new complete learning boundary changes acquisition priority.",
        ),
        rows=[
            dict(
                unit_id=f"exp{b['experiment_id']}",
                arm=b["boundary"],
                metric="qualified_complete_cost_closure",
                value=1 if b["qualified"] else None,
                status="completed" if b["qualified"] else "censored",
                missing_reason=b["missing_reason"],
                aggregation_scope="readiness only; costs and ceilings remain separate per cohort and arm",
            )
            for b in branches
        ],
        historical_adversarial_findings=original.get("adversarial_findings", []),
    )
    for b in branches:
        for c in b["cells"]:
            if c.get("missing_reason"):
                value["gate_check_summary"].append(
                    dict(
                        previous.failure(
                            Path(c["primitive_reference"]["path"]),
                            f"exp{b['experiment_id']}",
                            "spans.deterministic_spline_109",
                            "isolated compatible arithmetic endpoints",
                            None,
                        ),
                        observed_state="missing_field",
                        reason=c["missing_reason"],
                    )
                )
    value["conditional_acceleration_bounds"] = [
        dict(
            boundary=b["boundary"],
            producer=b["experiment_id"],
            cohort=c.get("cohort"),
            arm=c.get("arm"),
            conditional_ceiling=c.get("conditional_ceiling"),
            conditional_limit=c.get("conditional_limit"),
            total_s=c.get("total_s"),
            compatible_s=c.get("compatible_s"),
            achieved_speedup=None,
            missing_reason=c.get("missing_reason", b["missing_reason"]),
            interpretation="Conditional arithmetic-removal bound only; never achieved speed.",
        )
        for b in branches
        for c in (b["cells"] or [{}])
    ]
    value["conditional_ceiling_rows"] = value["conditional_acceleration_bounds"]
    value["source_admission_rows"] = [
        {k: b[k] for k in b if k not in {"refs", "cells"}} for b in branches
    ]
    value["hardware_100x_requirements"]["observed_unaccelerated_shares"] = [
        c["unaccelerated_share"] for c in cells
    ]
    value["hardware_100x_requirements"]["host_fraction_measured"] = all(
        c["compatible_s"] is not None for c in cells
    ) and bool(cells)
    value["methodology_note"] = (
        "Authenticate original board facts and cold-reduce existing complete cost primitives. No current numerical inference, language-model calls, device probes or timing reruns. Opaque binding spans cannot establish arithmetic-removal ceilings."
    )
    value["preserved_historical_outcomes"] = [
        *original.get("preserved_historical_outcomes", []),
        "V742 completed thirteen scheduling slots, not thirteen scientific studies. Exp8640 input gates qualified; Exp8641 exhausted measurement budget before original union/deadline mutations; Exp8642 failed driver separation and required unrelated producer statements from consumers; Exp8646 numerical checks passed but historical teardown hit the tracked-results guard. Learning, complete-source comparison and installed costs never ran. Exp8650 action parity survived but reset TypeError left236 resets without authenticated causes.",
    ]
    for field, observed in value["branch_gate_values"].items():
        value["gate_check_summary"].append(
            dict(
                previous.failure(output, TASK, field, observed, observed),
                check="computed_" + field,
                sha256=None,
            )
        )
    value["field_principles"] = {
        k: "Exact invocation evidence; missing operands remain null. Report readiness, physical support and benefit are distinct."
        for k in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reauthenticate original pins and source operands rather than trusting local rehashing."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(checked(value["work_reference"]).read_bytes())
        if authenticate(Path(work["root"])) != work["evidence"]:
            return False
        for snapshot in work["snapshots"]:
            if snapshot["sha256"] != snapshot["original_reference"]["sha256"]:
                return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            checked(ref)
        return bool(
            build(
                work,
                value["validation_receipts"],
                Path(value["work_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False

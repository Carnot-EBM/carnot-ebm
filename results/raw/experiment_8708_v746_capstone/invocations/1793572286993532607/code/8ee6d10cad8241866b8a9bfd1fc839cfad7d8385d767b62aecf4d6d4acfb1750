"""REQ-REPORT-8386: keep historical custody and future device work independently visible.

A receipt can queue a changed setup. It cannot recreate lost source bytes or
turn an old failed device observation into a successful measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import gatemate_missing_evidence_8372 as old
from carnot.reporting import kv260_local_cost_boundary_8315 as base  # Explicit adapter export.
from carnot.reporting import v721_contract_methods as previous
from carnot.reporting import v722_contract_methods as authority
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
ROOT = base.ROOT
NAME = "experiment_8386_v722_gatemate_obligation_delta"
TASK = "exp8386-gatemate-obligation-delta"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_gatemate_obligation_delta_8386.py"
OWNED = [
    "python/carnot/reporting/gatemate_obligation_delta_8386.py",
    "python/carnot/reporting/gatemate_obligation_runner_8386.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:4baacae3c2767bc2b06356c35b4562cb29f7a345906672ebe604c0cacb38ec2b"
HISTORY = "results/experiment_8372_v721_gatemate_missing_evidence.json"
HISTORY_PIN = "sha256:ff1ec8bc18ab2a3d600d398ae2cc1289b1f0f2a60dd9825d1b2ee307f4665227"
TERMINAL = "results/raw/" + old.NAME + "/terminal_validation.json"
SIDECAR = "results/raw/" + old.NAME + "/validators/" + HISTORY_PIN[7:] + ".json"
CUTOFF_NS = 1791608068368056749
ACTIVE, DESIGN = authority.ACTIVE, authority.DESIGN
NOTE = "docs/research-notes/v722-gatemate-obligation.md"
PINS = {
    HISTORY: HISTORY_PIN,
    TERMINAL: "sha256:08b9940ac6e0f6dfe7784eb35ffc9907aaa9fbb748e3785c0dd9b4890c9c46bb",
    SIDECAR: "sha256:734aaed39a1dcd13a79d2831462eb491e1a10f612008c5d928bb9d6780e44448",
    old.TRANSCRIPT: old.TRANSCRIPT_PIN,
    previous.legacy.PROTOCOL: old.PROTOCOL_PIN,
    previous.PROTOCOL: previous.DEPLOYMENT_PIN,
}
CONDITIONS: list[Json] = [
    dict(id="physical_change", expected="dated operator cable/port/power change", requires=[]),
    dict(id="idcode", expected="GM1Ax IDCODE 0x20000001", requires=["physical_change"]),
    dict(id="n16_flash", expected="authenticated n16 bitstream flash", requires=["idcode"]),
    dict(id="device_smoke", expected="device sample/hash smoke parity", requires=["n16_flash"]),
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so a bounded reader never appears to be a stalled job."""
    print(f"[exp8386] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Keep absent operands null, because zero would claim an actual observation."""
    return dict(
        check=field,
        upstream=path.stem,
        path=str(path),
        hash=None,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


failure = gate


def scan(paths: list[Path], raw: Path) -> list[Json]:
    """Read explicit receipts once; neither repository searches nor device probes are needed."""
    started = time.monotonic()
    rows = []
    for index, path in enumerate(paths):
        progress("delta_receipt", index, len(paths) - index)
        if time.monotonic() - started > 300:
            raise TimeoutError("delta_scan_deadline")
        refs: list[Json] = []
        try:
            frozen = base.pin(path, raw, refs)
            value = json.loads(frozen.read_bytes())
            if (
                value.get("exp8372_primary_sha256") != HISTORY_PIN
                or type(value.get("received_wall_ns")) is not int
                or not CUTOFF_NS < value["received_wall_ns"] <= time.time_ns()
            ):
                raise ValueError("receipt_after_exp8372_cutoff")
            imported = old.consume_receipt(frozen, raw, old.MISSING)
            if not imported or any(r["disposition"] == "rejected" for r in imported):
                raise ValueError("supplied_receipt_contract")
            reopen = [
                "physical_change"
                if r["disposition"] == "queued_next_hardware_task"
                else "source_custody:" + r["original_path"]
                for r in imported
            ]
            rows.append(
                dict(
                    path=str(path),
                    authenticated=True,
                    could_reopen=reopen,
                    reference=refs[0],
                    imported=imported,
                    reason=None,
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            rows.append(
                dict(
                    path=str(path),
                    authenticated=False,
                    could_reopen=[],
                    reference=refs[0] if refs else None,
                    imported=[],
                    reason=str(error),
                )
            )
    progress("delta_scan_complete", len(rows), 0)
    return rows


def measure(root: Path, raw: Path, supplied: list[Path] | None = None) -> Json:
    """Seal the exact invocation before reducing any historical or newly supplied claims."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(
        root=str(root.absolute()),
        inputs=[],
        supplied=[],
        started_monotonic_ns=time.monotonic_ns(),
        code_config_hashes=[],
    )
    names = [ACTIVE, DESIGN, *PINS]
    for index, name in enumerate(names):
        progress("authenticate_input", index, len(names) - index)
        refs: list[Json] = []
        try:
            base.pin(root / name, raw, refs)
            work["inputs"].append(
                dict(name=name, reference=refs[0], observed_sha256=refs[0]["sha256"], reason=None)
            )
        except (OSError, ValueError) as error:
            work["inputs"].append(
                dict(name=name, reference=None, observed_sha256=None, reason=str(error))
            )
    work["supplied"] = scan(supplied or [], raw)
    for name in [
        *OWNED,
        "python/carnot/reporting/gatemate_missing_evidence_8372.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/v709_execution.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "ops/exclusion_manifest.yaml",
        NOTE,
    ]:
        base.pin(ROOT / name, raw / "code", work["code_config_hashes"])
    work["ended_monotonic_ns"] = time.monotonic_ns()
    progress("measurement_complete", len(names), 0)
    return work


def derive(work: Json) -> Json:
    """Recompute obligations from frozen bytes instead of trusting a saved readiness claim."""
    checks, loaded, refs = [], {}, []
    root = Path(work["root"])
    for row in work["inputs"]:
        ref = row["reference"]
        if ref:
            path = base.checked(ref)
            if ref["sha256"] != row["observed_sha256"]:
                raise ValueError("input_observation_hash")
            refs.append(ref)
        check = gate(
            root / row["name"],
            "input." + row["name"],
            PINS.get(row["name"], "present"),
            row["observed_sha256"] if row["name"] in PINS else ("present" if ref else None),
        )
        check["hash"] = row["observed_sha256"]
        checks.append(check)
        if check["passed"]:
            loaded[row["name"]] = path.read_bytes()
    task_hash = None
    try:
        plan = yaml.safe_load(loaded[ACTIVE])
        task = next(t for t in plan["tasks"] if t["id"] == TASK)
        task_hash = canonical_hash(task)
    except (KeyError, TypeError, StopIteration, yaml.YAMLError):
        pass
    checks.append(gate(root / ACTIVE, "full_task_sha256", TASK_PIN, task_hash))
    independent = None
    try:
        tasks = parse_design(loaded[DESIGN].decode(), milestone="2026.10.722")[1]
        independent = canonical_hash(next(t for t in tasks if t["id"] == TASK))
    except (KeyError, ValueError, IndexError, StopIteration):
        pass
    checks.append(gate(root / DESIGN, "independent_design.full_task_sha256", TASK_PIN, independent))
    historical: Json = {}
    if all(name in loaded for name in PINS):
        historical = json.loads(loaded[HISTORY])
        validate_primary(historical, root / HISTORY)
        terminal, sidecar = json.loads(loaded[TERMINAL]), json.loads(loaded[SIDECAR])
        cutoff = max(
            r["started_wall_ns"] + round(r["duration_s"] * 1e9) for r in terminal["checks"]
        )
        if (
            historical["missing_source_hashes"] != old.MISSING
            or historical["history_authentication"] is not False
            or terminal["publication"]["primary_sha256"] != HISTORY_PIN
            or sidecar["report"]["passed"] is not True
            or cutoff != CUTOFF_NS
        ):
            raise ValueError("sealed_history_contract")
    sources, physical = {}, None
    from tempfile import TemporaryDirectory

    for row in work["supplied"]:
        if row["reference"]:
            base.checked(row["reference"])
        if row["authenticated"]:
            receipt = json.loads(base.checked(row["reference"]).read_bytes())
            if (
                receipt.get("exp8372_primary_sha256") != HISTORY_PIN
                or type(receipt.get("received_wall_ns")) is not int
                or receipt["received_wall_ns"] <= CUTOFF_NS
            ):
                raise ValueError("receipt_frontier")
            with TemporaryDirectory(prefix="exp8386-reader-", dir="/var/tmp") as directory:
                imported = old.consume_receipt(
                    base.checked(row["reference"]), Path(directory), old.MISSING
                )
            keys = {"disposition", "original_path", "sha256", "physical_change"}
            if [{k: v for k, v in r.items() if k in keys} for r in imported] != [
                {k: v for k, v in r.items() if k in keys} for r in row["imported"]
            ]:
                raise ValueError("receipt_reduction")
            for imported_row in row["imported"]:
                base.checked(imported_row["receipt_reference"])
                if imported_row["disposition"] == "supplied_exact_bytes":
                    base.checked(imported_row["reference"])
                    sources[imported_row["original_path"]] = imported_row["sha256"]
                else:
                    physical = imported_row["physical_change"]
            reopened = [
                "physical_change"
                if r["disposition"] == "queued_next_hardware_task"
                else "source_custody:" + r["original_path"]
                for r in imported
            ]
            if row["could_reopen"] != reopened:
                raise ValueError("receipt_reopening_scope")
        checks.append(
            gate(
                Path(row["path"]),
                "supplied_receipt.authentication",
                True,
                True if row["authenticated"] else row["reason"],
            )
        )
    for missing in old.MISSING:
        checks.append(
            gate(
                root / missing["path"],
                "missing_source." + missing["path"],
                missing["sha256"],
                sources.get(missing["path"]),
            )
        )
    for condition in CONDITIONS:
        checks.append(
            gate(
                root / old.TRANSCRIPT,
                "future." + condition["id"],
                condition["expected"],
                condition["expected"]
                if condition["id"] == "physical_change" and physical
                else None,
            )
        )
    checks.append(gate(root / HISTORY, "history_authenticated", True, False))
    for check in checks:
        ref = next((r for r in refs if r["original_path"] == check["path"]), None)
        check["hash"] = ref["sha256"] if ref else check["hash"]
    return dict(
        checks=checks,
        refs=refs,
        historical=historical,
        sources=sources,
        physical=physical,
        recorded=bool(historical and task_hash == TASK_PIN),
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """A documented obligation earns no hardware, learning or generalization claim."""
    evidence = derive(work)
    owned = bool(receipts) and all(r["passed"] for r in receipts if r.get("scope") != "global")
    owned = owned and work.get("preflight_passed", True)
    units = [("continuity_record", evidence["recorded"], "sealed obligation unavailable")]
    units += [
        (r["path"], r["path"] in evidence["sources"], "exact historical bytes not supplied")
        for r in old.MISSING
    ]
    units += [
        (
            r["id"],
            bool(evidence["physical"]) if r["id"] == "physical_change" else False,
            "physical change not supplied"
            if r["id"] == "physical_change"
            else "future device observation not supplied",
        )
        for r in CONDITIONS
    ]
    rows = [
        dict(
            source_id=name,
            arm="read_only_obligation",
            intended=1,
            completed=bool(done),
            failed=bool(index == 0 and not owned),
            censored=not done,
            excluded=False,
            independent=0,
            numerator=int(done),
            denominator=1,
            missing_reason=None if done else reason,
        )
        for index, (name, done, reason) in enumerate(units)
    ]
    kind = "blocked" if owned else "disqualified"
    value: Json = dict(
        experiment_id=8386,
        task_id=TASK,
        milestone="2026.10.722",
        run_date="20261010",
        honest_verdict="complete_" + kind + "_gatemate_obligation_delta",
        verdict_class=kind,
        gate_check_summary=evidence["checks"] + work.get("preflight_checks", []),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=evidence["historical"].get("historical_model_provenance", []),
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["completed"] for r in rows),
        failed_count=sum(r["failed"] for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            continuity=1,
            missing_sources=2,
            physical_obligations=4,
            timing_repeats_are_independent=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed_cached_history_and_constructed_private_controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates=dict(
            owned_validation=owned,
            obligation_recorded=evidence["recorded"],
            history_authenticated=False,
            hardware_execution=False,
            scientific_benefit=False,
        ),
        validation_receipts=receipts,
        adversarial_findings=work.get("adversarial_findings", []),
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        preconditions_checked=True,
        precondition_receipt=work.get("preflight_checks", []),
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(
                phase="bounded_delta_and_validation",
                started_monotonic_ns=work["started_monotonic_ns"],
                ended_monotonic_ns=work["ended_monotonic_ns"],
            )
        ],
        random_seed=7228386,
        source_artifact_hashes=evidence["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[base.reference(raw / "measurement.json")],
        cited_upstream_artifacts=[
            dict(
                r,
                imported_fields=[
                    "task authority, terminal custody, historical missing sources and physical frontier"
                ],
            )
            for r in evidence["refs"]
        ],
        obligation_recorded=evidence["recorded"],
        obligation_recorded_score=int(evidence["recorded"] and owned),
        supplied_evidence_delta=work["supplied"],
        exact_missing_hashes=old.MISSING,
        history_authenticated=False,
        physical_change_receipt=evidence["physical"],
        next_evidence_conditions=CONDITIONS,
        device_command_count=0,
        original_idcode="0xffffffff",
        required_idcode="0x20000001",
        evidence_cutoff_wall_ns=CUTOFF_NS,
        evidence_delta_scan_limit_s=300,
        future_producer_dependencies=[],
        execution_ready_score=0,
        current_contract_ready_score=0,
        work_reference=base.reference(raw / "measurement.json"),
        publication_output=str(output),
        invocation_argv=work.get("invocation_argv", []),
        historical_verdict=evidence["historical"].get("honest_verdict"),
        methodology="Seal Exp8372 and its original transcript; inspect only explicit post-cutoff receipts; independently retain two source and four physical obligations; no LLM, source recovery, JTAG, flash or benchmark.",
    )
    value["field_principles"] = {
        k: "Bind the field to sealed operands; continuity documentation grants no semantic or device benefit."
        for k in value
    }
    value["field_principles"].update(
        field_principles="Explain each field's audit purpose.",
        reproducibility_checksum="Bind the full reduced record, independently of semantic validity.",
        history_authenticated="Historical source custody remains blocked pending exact bytes and future replay.",
        physical_change_receipt="Independent present setup evidence queues only its physical condition.",
        supplied_evidence_delta="Preserve authentication, rejection and reopening scope for each supplied receipt.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Cold reduction rejects new hashes that conceal changed task or historical meaning."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            base.checked(ref)
        for ref in work["code_config_hashes"]:
            if base.sha256_file(Path(ref["original_path"])) != ref["sha256"]:
                return False
        for row in work["inputs"]:
            if row["reference"] and row["name"] == ACTIVE:
                plan = yaml.safe_load(base.checked(row["reference"]).read_bytes())
                task = next(t for t in plan["tasks"] if t["id"] == TASK)
                if canonical_hash(task) != TASK_PIN:
                    return False
        for field in ["execution_manifest_reference", "owned_coverage_reference"]:
            if work.get(field):
                base.checked(work[field])
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if stream + "_path" in receipt:
                    base.checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    )
        raw = Path(value["work_reference"]["path"]).parent
        return bool(
            value
            == build(work, value["validation_receipts"], raw, Path(value["publication_output"]))
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration, yaml.YAMLError):
        return False

"""REQ-VERIFY-8722: freeze one current invocation before reducing retained evidence."""

from __future__ import annotations

from datetime import datetime, UTC
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import v747_capstone_inputs as e
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def measure(raw: Path, date: str, private: bool) -> Json:
    """Authenticate resources and original producer receipts without running old studies."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    e.progress("preconditions_before", 0, 14)
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="capstone8722-", dir="/var/tmp") as directory:
        pre = resources(Path(directory))
        pre["write_probe_passed"] = (
            Path(directory) / "write_probe"
        ).read_bytes() == b"private disk scratch"
    pre.update(
        optional_devices_required=False,
        language_model_calls=0,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        total_cap_s=4800,
    )
    ready = bool(
        pre["disk_backed"]
        and pre["tools_available"]
        and pre["write_probe_passed"]
        and min(pre["free_disk_bytes"], pre["memory_available_bytes"]) >= 2**30
    )
    authority_root = raw / "authority"
    names = [e.DESIGN, "research-roadmap.yaml"]
    if (e.ROOT / e.contract_source.STAGED).is_file():
        names.append(e.contract_source.STAGED)
    for name in names:
        target = authority_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(e.ROOT / name, target)
    contract = e.contract(authority_root)
    log = e.freeze(e.ROOT / "ops/conductor-log.md", raw / "sources")
    lines = Path(log["snapshot_reference"]["path"]).read_text().splitlines()
    slots, refs, gates = [], [log], []
    for i, task in enumerate(contract["tasks"][:-1]):
        e.progress("slot_before_" + task["id"], i, 14 - i)
        row = e.slot(task, raw / "primaries", lines)
        slots.append(row)
        refs += [row["declared_reference"], row["primary_reference"], *row["evidence_references"]]
        e.progress("slot_after_" + task["id"], i + 1, 13 - i)
    for task in contract["tasks"]:
        for gate in task.get("gated_on", []):
            upstream = next(r for r in slots if r["task_id"] == gate["upstream"])
            row = e.observe(
                upstream["declared_reference"],
                gate["upstream"],
                gate["artifact_field"],
                gate["value"],
            )
            row.update(downstream=task["id"], producer_qualified=upstream["producer_qualified"])
            row["passed"] = bool(row["passed"] and upstream["producer_qualified"])
            gates.append(row)
    methods = e.previous.original_inputs()
    refs += methods["references"]
    prior = e.previous.prior_rows(contract["tasks"], raw / "history")
    refs += [r["primary_reference"] for r in prior] + [
        ref for r in prior for ref in r["references"]
    ]
    history = e.ROOT / "results/experiment_8708_v746_capstone.json"
    e.verify(dict(path=str(history), sha256="sha256:" + e.PINS[8708]))
    history_ref = e.freeze(history, raw / "history")
    refs.append(history_ref)
    code = sorted(
        set(
            e.OWNED
            + e.TESTS
            + [
                str(Path(str(m.__file__)).resolve().relative_to(e.ROOT))
                for m in list(sys.modules.values())
                if getattr(m, "__file__", None)
                and Path(str(m.__file__)).suffix == ".py"
                and Path(str(m.__file__)).resolve().is_relative_to(e.ROOT)
                and ".venv" not in Path(str(m.__file__)).parts
            ]
        )
    )
    dependencies = [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "pyproject.toml",
        *names,
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
        "ops/exclusion_manifest.yaml",
        "ops/e2e-test-plan.md",
        "ops/verifier_gaps.md",
        "research-references.md",
        "research-studying.md",
        "docs/research-notes/v747-capstone.md",
        "scripts/experiment_template.py",
        "scripts/publication_gate.py",
        "scripts/summarize_artifact.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "scripts/check_spec_coverage.py",
        "ops/publication_gate_state.json",
        "docs/technical-report.md",
        "docs/arxiv-paper/main.tex",
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        "tests/python/test_restricted_decision_audit_8210.py",
    ]
    refs += [e.freeze(e.ROOT / name, raw / "sources") for name in sorted(set(dependencies))]
    refs += [e.reference(p) for p in authority_root.rglob("*") if p.is_file()]
    code_refs = [e.freeze(e.ROOT / name, raw / "code") for name in code]
    e.atomic_json(
        raw / "configuration.json",
        dict(
            owned=e.OWNED,
            tests=e.TESTS,
            required_statement_coverage=100,
            retained_driver="v746_capstone",
            language_model_calls=0,
        ),
    )
    code_refs.append(e.reference(raw / "configuration.json"))
    e.atomic_json(raw / "premeasurement_seal.json", dict(source_refs=refs, code_refs=code_refs))
    e.progress("sealed_before_reduction", 13, 1)
    publication = child(
        "publication_gate",
        [sys.executable, "-u", "scripts/publication_gate.py", "--json"],
        raw / "publication",
        deadline=60,
    )
    if not publication["passed"]:
        raise ValueError("publication_gate_execution")
    external = [
        dict(
            upstream=r["task_id"],
            check="qualified_producer",
            **r["declared_reference"],
            field="producer_qualified",
            operator="==",
            expected=True,
            observed=r["producer_qualified"],
            passed=False,
            operand_state="producer_disqualified"
            if r["producer_executed"]
            else "producer_not_executed",
        )
        for r in slots
        if not r["producer_qualified"]
    ]
    external += [r for r in methods["gate_check_summary"] if not r["passed"]]
    end = time.monotonic_ns()
    if end - start >= 3600_000_000_000:
        raise TimeoutError("measurement_deadline")
    work = dict(
        raw=str(raw),
        invocation_id=raw.name,
        run_date=date,
        private_control=private,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(
                phase="authenticate_freeze_reconcile",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
        contract=contract,
        authority_root=str(authority_root),
        slots=slots,
        prior_rows=prior,
        gate_rows=gates,
        external_checks=external,
        history_reference=history_ref,
        preconditions=pre,
        preconditions_passed=ready,
        source_refs=refs,
        code_refs=code_refs,
        operand_refs=[ref for r in slots for ref in r["evidence_references"]],
        publication_receipt=publication,
        publication=json.loads(Path(publication["stdout_path"]).read_bytes()),
    )
    e.atomic_json(raw / "measurement.json", work)
    e.atomic_json(
        raw.parents[1] / "seals" / (raw.name + ".json"),
        dict(measurement=e.reference(raw / "measurement.json"), code_refs=code_refs),
    )
    (raw / "measurement.json").chmod(0o444)
    return work

"""REQ-REPORT-9181-V782: qualify single-call structural work without model claims.

The retained V781 result remains failure evidence. Current rows qualify only the
small changed algorithm and its target checks; the full union belongs to Exp9182.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import causal_validation_9167 as old
from carnot.reporting.current_work_receipt import canonical_hash, build_current_work_receipt
from carnot.reporting.forward_evidence_runner_9111 import bound
from carnot.reporting.natural_label_eligibility_8975 import ref
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.spline_basis_methods_8986 import gate
from carnot.reporting.validation_diagnostics_9181 import immutable, run, probe
from carnot.reporting.validation_schedule_runner_9139 import foreign_argv

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_9181_v782_causal_validation"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_causal_validation_9181.py"
OWNED = [
    "python/carnot/reporting/causal_validation_9181.py",
    "python/carnot/reporting/validation_diagnostics_9181.py",
    CLI,
]
CHANGED = [
    "scripts/adversarial_verify.py",
    old.SEAM,
    "scripts/experiments/experiment_9111_v777_forward_evidence.py",
    "scripts/verdict_row_consistency_lint.py",
]
BASE = ROOT / "results/raw" / NAME / "baseline"
BASE_PINS = {
    "verdict_row_consistency_lint.py": "sha256:74df9c4e62692c2c1f37b5772856676f0cb2af404bc77d5627b6bc509fe03698",
    "adversarial_verify.py": "sha256:232a887591efdb1f2b3e39e8f2c3b199bb10a9458ec713f7191ef33827f22a1f",
    "forward_evidence_runner_9111.py": "sha256:779bbcb6f44c928e07d6111595974cc5cdf69693fe0252c82280eb9dad621bcb",
    "experiment_9111_v777_forward_evidence.py": "sha256:df472898780fb30373596f5b5cbba225202900f0b051119f44ff799147ff5458",
}
SEED = 7829181
MODEL_SPECS: list[Json] = []
INPUTS = {
    "results/experiment_9180_v782_contract.json": (
        "sha256:42c35a172e548872dbbc82c737ce2c3ee3a624d4a0fec2495225c2f7f38e92f2",
        "contract_ready_score",
        1,
    ),
    "results/experiment_9167_v781_causal_validation.json": (
        "sha256:4776dc4139c8c65d3f477fc9d37a434265cf5b67d2d2797d0dee26f8b7723ebe",
        "causal_fix_ready_score",
        0,
    ),
}


def prepare(root: Path, raw: Path, date: str) -> Json:
    """Retain original obligations and exact authorities before any dependent work."""
    work = old.prepare(root, raw, date)
    work.update(
        start_ns=time.monotonic_ns(),
        commands=[],
        validation_receipts=[],
        rows=[],
        diagnostic_attempt_rows=[],
        structural_scan_rows=[],
        diagnostic_elapsed_s=0,
    )
    for name, (pin, field, expected) in INPUTS.items():
        path = root / name
        observed = ref(path)["sha256"] or "missing_file"
        work["checks"].append(gate(path, "sha256", pin, observed, "input_bytes", name))
        if observed != pin:
            continue
        value = json.loads(bound(dict(path=str(path), sha256=pin)).read_bytes())
        work["checks"].append(
            gate(path, field, expected, value.get(field, "missing_field"), "input_gate", name)
        )
        target = raw / "inputs" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        target.chmod(0o444)
        reference = dict(path=str(target), sha256=pin)
        bound(reference)
        work["references"].append(
            dict(reference, producer_path=str(path), imported_fields=list(value))
        )
        work["inputs"][name] = value
        if "9167" in name:
            work["historical_failures"] = [
                c for c in value["gate_check_summary"] if not c["passed"]
            ]
            work["historical_operation_rows"] = value["causal_evidence"][
                "controlled_qualification"
            ]["after_operation_rows"]
            for trace_ref in value["causal_evidence"]["trace_references"]:
                original = bound(trace_ref)
                target = (
                    raw / "retained_traces" / str(trace_ref["sha256"]).split(":")[1] / original.name
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(original.read_bytes())
                target.chmod(0o444)
                work["references"].append(
                    dict(
                        ref(target),
                        producer_path=str(original),
                        imported_fields=["retained_v781_trace_no_current_qualification_credit"],
                    )
                )
    union = {r["node_id"]: r for r in work["remaining_obligations"]}
    for value in work["inputs"].values():
        obligations = value.get("validation_obligations", {})
        for key in [
            "remaining_v781_obligations",
            "affected_node_assertions",
            "affected_v780_node_assertions",
        ]:
            for row in obligations.get(key, []):
                node = row if isinstance(row, str) else row["node_id"]
                union.setdefault(
                    node,
                    dict(
                        node_id=node,
                        reasons=[key],
                        status="pending",
                        current_qualification_credit=0,
                    ),
                )
    work["remaining_obligations"] = list(union.values())
    work["code_config_hashes"] = [ref(ROOT / p) for p in [*OWNED, *CHANGED, TEST, "pyproject.toml"]]
    current_sources = []
    for name in [*OWNED, *CHANGED, TEST, "pyproject.toml"]:
        target = raw / "revision_v782" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((ROOT / name).read_bytes())
        target.chmod(0o444)
        current_sources.append(
            dict(
                ref(target),
                producer_path=str(ROOT / name),
                imported_fields=["executed_revision_bytes"],
            )
        )
    work["references"].extend(current_sources)
    work["revision_reference"].update(
        owned_code_config_hashes=work["code_config_hashes"],
        exact_obligation_union=work["remaining_obligations"],
        full_qualification_owner="exp9182-forward-qualification",
        exact_revised_source_manifest=current_sources,
    )
    work["baseline_references"] = [
        dict(path=str(BASE / n), sha256=pin) for n, pin in BASE_PINS.items()
    ]
    for reference in work["baseline_references"]:
        observed = ref(Path(reference["path"]))["sha256"] or "missing_file"
        work["checks"].append(
            gate(
                Path(reference["path"]),
                "sha256",
                reference["sha256"],
                observed,
                "baseline_bytes",
                "pre_change_snapshot",
            )
        )
    origins = {Path(name).name: str(ROOT / name) for name in CHANGED}
    work["references"].extend(
        dict(
            reference,
            producer_path=origins[Path(reference["path"]).name],
            imported_fields=["exact_pre_change_source_bytes"],
        )
        for reference in work["baseline_references"]
    )
    work["commands"] = commands(raw)
    command_union = {
        canonical_hash(row["command"]["argv"]): row
        for row in work["revision_reference"]["exact_validation_command_union"]
    }
    for command in work["commands"]:
        identity = canonical_hash(command["argv"])
        command_union.setdefault(
            identity,
            dict(unit_id="command:" + identity, command=command, estimated_remaining_s=None),
        )
    work["revision_reference"]["exact_validation_command_union"] = list(command_union.values())
    work["expected_target_nodes"] = sorted(
        n for n in union if n.startswith("tests/python/test_forward_evidence_runner_9111.py::")
    )
    if all(c["passed"] for c in work["checks"]):
        paths = sorted(
            {
                arg
                for spec in work["commands"]
                for arg in spec["argv"]
                if arg.startswith("tests/python/")
            }
        )
        known = sorted(n for n in union if n.split("::")[0] in paths)
        unknown_paths = [p for p in paths if not any(n.startswith(p + "::") for n in known)]
        receipt = run(
            "freeze_collection",
            [
                sys.executable,
                "-m",
                "pytest",
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "--collect-only",
                "-q",
                *unknown_paths,
            ],
            raw,
            time.monotonic() + 120,
        )
        work["collection_receipt"] = receipt
        work["checks"].append(
            gate(
                Path(receipt["stdout_path"]),
                "exit_code",
                0,
                receipt["exit_code"],
                "scope_collection",
                "current_revision",
            )
        )
        work["expected_nodes"] = sorted(
            set(known)
            | {
                line
                for line in Path(receipt["stdout_path"]).read_text().splitlines()
                if line.startswith("tests/python/") and "::" in line
            }
        )
        for node in work["expected_nodes"]:
            union.setdefault(
                node,
                dict(
                    node_id=node,
                    reasons=["v782_affected_assertion"],
                    status="pending",
                    current_qualification_credit=0,
                ),
            )
        work["remaining_obligations"] = list(union.values())
        work["revision_reference"]["exact_obligation_union"] = work["remaining_obligations"]
    else:
        work["expected_nodes"] = []
    immutable(
        raw / "scope.json",
        dict(
            commands=work["commands"],
            revision=work["revision_reference"],
            seed=SEED,
            exclusions=[],
            diagnosis_slices_s=[300, 450, 150],
            repository_health_command=[str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
            repository_health_ceiling_s=120,
            formulas="exact flag equality; changed scans<=1; hashes=0; unique node phase outcomes",
        ),
    )
    return work


def commands(raw: Path) -> list[Json]:
    """Freeze checks once; the complete target module keeps its original fixture lifetime."""
    py = str(ROOT / ".venv/bin/python")
    pytest = [py, "-m", "pytest", "-n0", "-o", "addopts=", "--no-cov", "-q"]
    specs = [
        dict(name=n, argv=a)
        for n, a in [
            ("target_module", [*pytest, "tests/python/test_forward_evidence_runner_9111.py"]),
            (
                "owned_tests",
                [
                    py,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + str(raw / "coverage.ini"),
                    "-m",
                    "pytest",
                    "-n0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    TEST,
                ],
            ),
            (
                "original_controls",
                [
                    *pytest,
                    old.TEST,
                    "tests/python/test_forward_evidence_9111.py",
                    "tests/python/test_verdict_row_consistency_lint.py",
                ],
            ),
            (
                "direct_consumers",
                [
                    *pytest,
                    *sorted(
                        str(p.relative_to(ROOT))
                        for p in (ROOT / "tests/python").glob("test_adversarial_verify*.py")
                    ),
                ],
            ),
            (
                "e2e018_021",
                [
                    *pytest,
                    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                    "tests/python/test_restricted_decision_audit_8210.py",
                ],
            ),
            ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, *CHANGED, TEST]),
            (
                "ruff_format",
                [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, *CHANGED, TEST],
            ),
            (
                "strict_mypy",
                [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED],
            ),
            ("spec_coverage", [py, "scripts/check_spec_coverage.py", TEST]),
        ]
    ]
    return sorted(
        specs,
        key=lambda s: (
            s["name"]
            not in {"target_module", "original_controls", "direct_consumers", "e2e018_021"}
        ),
    )


def diagnose(work: Json, stop: float) -> None:
    """Use one bounded diagnostic allocation and measure instrumentation separately."""
    raw = Path(work["raw"]) / "diagnosis"
    start = time.monotonic()
    deadline = min(stop, start + 900)
    for arm, slice_end in [
        ("original", start + 300),
        ("v781_wrapper", start + 300),
        ("changed", start + 750),
        ("overhead", start + 900),
    ]:
        output = raw / (arm + ".json")
        argv = foreign_argv(
            CLI, "--probe", "changed" if arm == "overhead" else arm, "--probe-output", str(output)
        )
        env = {} if arm == "overhead" else {"CARNOT_9181_TRACE": str(raw / arm)}
        row = run(arm, argv, raw, min(deadline, slice_end), extra_env=env)
        row["trace_references"] = [ref(p) for p in sorted((raw / arm).glob("*.jsonl"))]
        events = [
            json.loads(line)
            for reference in row["trace_references"]
            for line in bound(reference).read_text().splitlines()
        ]
        row["child_identities"] = [e for e in events if e["event"] == "start"]
        row["measured_hook_callback_s"] = (
            sum(e["hook_callback_ns"] for e in events if e["event"] == "summary") / 1e9
        )
        row["total_diagnostic_deadline_monotonic_s"] = deadline
        row["diagnostic_ceiling_s"] = 900
        row["probe_reference"] = ref(output)
        work["diagnostic_attempt_rows"].append(row)
        if row["passed"]:
            work["structural_scan_rows"].append(
                json.loads(bound(row["probe_reference"]).read_bytes())
            )
    work["diagnostic_elapsed_s"] = time.monotonic() - start


def coverage_rows(tracer: Coverage) -> list[Json]:
    """Measure every new statement, comparing shared files to their retained originals."""
    rows = []
    for name in OWNED + CHANGED:
        _, statements, excluded, missing, _ = tracer.analysis2(str(ROOT / name))
        selected = set(statements)
        if name in CHANGED:
            before = (BASE / Path(name).name).read_text().splitlines()
            after = (ROOT / name).read_text().splitlines()
            selected &= {
                i + 1
                for tag, _, _, start, end in difflib.SequenceMatcher(
                    a=before, b=after
                ).get_opcodes()
                if tag in {"insert", "replace"}
                for i in range(start, end)
            }
        rows.append(
            dict(
                path=name,
                statements=sorted(selected),
                missing=sorted(selected & set(missing)),
                excluded=sorted(selected & set(excluded)),
            )
        )
    return rows


def verify(work: Json, stop: float, verification_stop: float | None = None) -> None:
    """Run each frozen command once and combine coverage only after all children exit."""
    raw = Path(work["raw"])
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED + CHANGED)
        + "[report]\nexclude_lines=\nomit=\n"
    )
    for spec in work["commands"]:
        measurement = spec["name"] in {
            "target_module",
            "original_controls",
            "direct_consumers",
            "e2e018_021",
        }
        deadline = stop if measurement else min(verification_stop or stop, time.monotonic() + 1200)
        if time.monotonic() >= deadline:
            continue
        events = raw / (spec["name"] + "-nodes.jsonl")
        argv = spec["argv"]
        if "pytest" in argv:
            argv = [*argv, "-p", "carnot.reporting.receipt_qualification_9125"]
        row = run(
            spec["name"],
            argv,
            raw / "validation",
            deadline,
            extra_env=dict(CARNOT_9125_NODE_LOG=str(events)),
        )
        row["events_reference"] = ref(events)
        if events.exists():
            events.chmod(0o444)
        work["validation_receipts"].append(row)
    tracer = Coverage(config_file=str(config))
    tracer.combine(strict=True, keep=True)
    tracer.save()
    rows = coverage_rows(tracer)
    total = sum(len(r["statements"]) for r in rows)
    work["coverage"] = dict(
        files=rows,
        data_reference=ref(raw / ".coverage"),
        statement_percent=100 * (total - sum(len(r["missing"]) for r in rows)) / total,
        excluded_statements=sum(len(r["excluded"]) for r in rows),
    )


def qualified(work: Json) -> bool:
    """Reopen operation and child receipts so a summary label cannot grant readiness."""
    if not all(c["passed"] for c in work["checks"]):
        return False
    scans = work["structural_scan_rows"]
    if len(scans) != 4 or work["diagnostic_elapsed_s"] > 900:
        return False
    before, wrapper, changed, overhead = scans
    if not (
        before["input"] == wrapper["input"] == changed["input"] == overhead["input"]
        and before["flags"] == wrapper["flags"] == changed["flags"] == overhead["flags"]
        and before["full_scans"] > 1
        and wrapper["hashes"] > 1
        and changed["full_scans"] == 1
        and changed["hashes"] == 0
    ):
        return False
    for row in work["diagnostic_attempt_rows"] + work["validation_receipts"]:
        saved = json.loads(bound(row["receipt_reference"]).read_bytes())
        if not row["passed"] or any(row[k] != v for k, v in saved.items()):
            return False
        for stream in ["stdout", "stderr"]:
            bound(dict(path=row[stream + "_path"], sha256=row[stream + "_sha256"]))
    for attempt, scan in zip(work["diagnostic_attempt_rows"], scans, strict=True):
        if scan != json.loads(bound(attempt["probe_reference"]).read_bytes()):
            return False
        for reference in attempt["trace_references"]:
            events = [json.loads(line) for line in bound(reference).read_text().splitlines()]
            if not (
                events[0]["pid"] == attempt["pid"]
                and events[0]["pythonpath"] is None
                and events[-1]["event"] == "summary"
                and events[-1]["counts"].get("check_tautology") == 1
            ):
                return False
        if attempt["name"] != "overhead" and not attempt["trace_references"]:
            return False
    actual = work["validation_receipts"]
    if len(actual) != len(work["commands"]):
        return False
    for row, spec in zip(actual, commands(Path(work["raw"])), strict=True):
        expected = spec["argv"] + (
            ["-p", "carnot.reporting.receipt_qualification_9125"]
            if "pytest" in spec["argv"]
            else []
        )
        if row["name"] != spec["name"] or row["argv"] != expected:
            return False
    events = [
        json.loads(line)
        for row in actual
        if row["events_reference"]["sha256"]
        for line in bound(row["events_reference"]).read_text().splitlines()
    ]
    target = work["expected_target_nodes"]
    if not target or len([n for n in target if "test_rehashed_primitive_tampering[" in n]) != 7:
        return False
    for node in work["expected_nodes"]:
        phases = [
            (e["when"], e["outcome"])
            for e in events
            if e["nodeid"] == node and e["when"] != "start"
        ]
        if sorted(phases) != [("call", "passed"), ("setup", "passed"), ("teardown", "passed")]:
            return False
    coverage = work["coverage"]
    tracer = Coverage(config_file=False, data_file=str(bound(coverage["data_reference"])))
    tracer.set_option("report:exclude_lines", [])
    tracer.load()
    rows = coverage_rows(tracer)
    return rows == coverage["files"] and all(not r["missing"] and not r["excluded"] for r in rows)


def build(work: Json) -> Json:
    """Reduce each unique executed node once and keep scientific authority closed."""
    value = old.build(dict(work, validation_receipts=[], diagnosis={}, node_events=[]))
    ready = qualified(work)
    blocked = any(not c["passed"] and c["check"] != "scope_collection" for c in work["checks"])
    cls = "blocked" if blocked else "null" if ready else "disqualified"
    events = [
        json.loads(line)
        for r in work["validation_receipts"]
        if r["events_reference"]["sha256"]
        for line in bound(r["events_reference"]).read_text().splitlines()
    ]
    nodes = sorted(set(work["expected_nodes"]) | {e["nodeid"] for e in events})
    rows: list[Json] = []
    for node in nodes:
        phases = [e for e in events if e["nodeid"] == node and e["when"] != "start"]
        passed = sorted((e["when"], e["outcome"]) for e in phases) == [
            ("call", "passed"),
            ("setup", "passed"),
            ("teardown", "passed"),
        ]
        failed = any(e["outcome"] == "failed" for e in phases)
        rows.append(
            dict(
                unit_id=node,
                qid=None,
                group_id=node,
                sibling=None,
                arm="original_assertions",
                condition="corrected_revision",
                seed=SEED,
                absolute_metric=int(passed),
                numerator=int(passed),
                denominator=1,
                status="completed" if passed else "failed" if failed else "censored",
                exclusion_reason=None,
                censor_reason=None if passed or failed else "unfinished_phases",
            )
        )
    counts = {
        key: sum(r["status"] == key for r in rows) for key in ["completed", "failed", "censored"]
    }
    value.update(
        schema="carnot.causal_validation.v2",
        experiment_id=9181,
        task_id="exp9181-causal-validation",
        milestone="2026.11.782",
        honest_verdict="complete_" + cls + "_causal_validation",
        verdict_class=cls,
        rows=rows,
        intended_count=len(nodes),
        attempted_count=len(nodes),
        completed_count=counts["completed"],
        failed_count=counts["failed"],
        censored_count=counts["censored"],
        sample_size_budget=dict(
            intended=len(nodes), attempted=len(nodes), **counts, effective_independent_groups=0
        ),
        causal_fix_ready_score=int(ready),
        required_checks_passed=ready,
        owned_checks_passed=ready,
        validation_receipts=work["validation_receipts"],
        owned_statement_coverage=work.get("coverage", {}),
        structural_scan_rows=work["structural_scan_rows"],
        diagnostic_attempt_rows=work["diagnostic_attempt_rows"],
        causal_evidence=dict(
            operation="invariant ARC-null structural predicate",
            trace_references=[
                r for a in work["diagnostic_attempt_rows"] for r in a["trace_references"]
            ],
            competing_hypotheses=[
                "per-pair structural traversal",
                "per-pair canonical hashing",
                "hook overhead",
            ],
            before_after_rows=work["structural_scan_rows"],
            diagnostic_elapsed_s=work["diagnostic_elapsed_s"],
            instrumentation_overhead=work["structural_scan_rows"][-2:],
            censoring="No speedup ratio from censored baseline",
            historical_operation_rows=work.get("historical_operation_rows", []),
            historical_failures=work.get("historical_failures", []),
        ),
        inference_substrate_class="no_model_load"
        if work["diagnostic_attempt_rows"]
        else "blocked_no_run",
        random_seed=SEED,
        work_reference=ref(Path(work["raw"]) / "work.json"),
    )
    value["attempted_count"] = len({e["nodeid"] for e in events})
    value["sample_size_budget"]["attempted"] = value["attempted_count"]
    value["cost_model_rows"] = [
        dict(
            unit_id=r["name"],
            measured_elapsed_s=r["duration_s"],
            kill_timeout_s=r["deadline_s"],
            uncertainty="right_censored" if r["timed_out"] else "observed_process_only",
        )
        for r in work["diagnostic_attempt_rows"] + work["validation_receipts"]
    ]
    if len(work["structural_scan_rows"]) == 4:
        changed, overhead = work["structural_scan_rows"][-2:]
        value["causal_evidence"]["instrumentation_overhead"] = dict(
            traced_operation_s=changed["measured_duration_s"],
            untraced_operation_s=overhead["measured_duration_s"],
            signed_operation_delta_s=changed["measured_duration_s"]
            - overhead["measured_duration_s"],
            scope="One paired operation observation; callback counters retained in immutable traces; no general speedup claim",
        )
    value["acceptance_gates"]["scope"]["condition"] = (
        "Full exact obligation union belongs to Exp9182"
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> Json:
    """Rebuild claims from bound work rather than trusting rehashed aggregate edits."""
    value = json.loads(path.read_bytes())
    work = json.loads(bound(value["work_reference"]).read_bytes())
    for reference in work["references"] + work["code_config_hashes"]:
        bound(reference)
    if value != build(work):
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True)


def terminal(path: Path, raw: Path, stop: float) -> Json:
    """Only actual checker receipts may expose candidate bytes through publication."""
    replay(path)
    receipts = [
        run(name, [sys.executable, str(ROOT / script), flag, str(path)], raw, stop)
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    return dict(passed=all(r["passed"] for r in receipts), checks=receipts)


def main(argv: list[str] | None = None) -> int:
    """Share deadlines across descendants and reserve verification and publication once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261117")
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--probe", choices=["original", "v781_wrapper", "changed"])
    parser.add_argument("--probe-output", type=Path)
    args = parser.parse_args(argv)
    start = time.monotonic()
    try:
        if args.date != "20261117":
            raise ValueError("run_date")
        if args.probe:
            immutable(args.probe_output, probe(args.probe, BASE / "adversarial_verify.py"))
            return 0
        if args.replay:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        if args.from_work and args.output.absolute().is_relative_to(ROOT):
            raise ValueError("private_work_requires_external_output")
        raw = args.output.absolute().parent / "raw" / NAME / "invocations" / str(time.time_ns())
        print("[exp9181] phase=authentication_before", flush=True)
        work = (
            json.loads(args.from_work.read_bytes())
            if args.from_work
            else prepare(args.input_root, raw, args.date)
        )
        print("[exp9181] phase=authentication_after", flush=True)
        if not args.from_work:
            if all(c["passed"] for c in work["checks"]):
                print("[exp9181] phase=diagnosis_before", flush=True)
                diagnose(work, start + 900)
                print("[exp9181] phase=diagnosis_after", flush=True)
                verify(work, start + 3300, start + 4500)
                work["repository_health"] = run(
                    "repository_health",
                    [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
                    raw / "repository_health",
                    min(start + 4500, time.monotonic() + 120),
                )
            work["duration_s"] = time.monotonic() - start
            work["current_work"] = build_current_work_receipt(
                run_id=raw.name,
                owner_pid=os.getpid(),
                events=[],
                inference_substrate="deterministic_no_model_load",
                inference_substrate_details=dict(
                    work="CPU structural scans and actual child validation"
                ),
                inference_substrate_class="no_model_load",
                execution_venue="host",
                started_monotonic_ns=work["start_ns"],
                ended_monotonic_ns=time.monotonic_ns(),
            )
            immutable(raw / "work.json", work)
        value = build(work)
        print("[exp9181] phase=publication_before", flush=True)
        receipt = publish_primary(
            args.output,
            value,
            lambda path: terminal(
                path, raw / "terminal", min(start + 4800, time.monotonic() + 300)
            ),
        )
        print(
            json.dumps(
                dict(
                    receipt,
                    honest_verdict=value["honest_verdict"],
                    causal_fix_ready_score=value["causal_fix_ready_score"],
                )
            ),
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError) as error:
        print("[exp9181] refused:" + str(error), flush=True)
        return 1

"""REQ-VERIFY-8708: freeze inputs, validate owned work and publish checked final bytes."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import v746_capstone_inputs as e
from carnot.reporting import v746_capstone_report as report
from carnot.reporting import v746_retained_evidence as retained
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
E2E = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py::test_cli_private_success_failure_and_replay",
    "tests/python/test_restricted_decision_audit_8210.py::test_private_cli_and_rehashed_mutations",
]


def measure(raw: Path, date: str, private: bool) -> Json:
    """Freeze exact authority and original operands; optional hardware cannot veto CPU work."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    e.progress("preconditions_before", 0, 14)
    raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="capstone8708-", dir="/var/tmp") as directory:
        pre = resources(Path(directory))
        pre.update(
            write_probe_passed=(Path(directory) / "write_probe").read_bytes()
            == b"private disk scratch",
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
    for source, target in [
        (e.DESIGN, e.authority.DESIGN),
        ("research-roadmap.yaml", "research-roadmap.yaml"),
    ]:
        destination = authority_root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(e.ROOT / source, destination)
    contract = e.authority.authority(authority_root)
    methods = e.original_inputs()
    checkpoint = methods["protocol"].get("head_checkpoint_reference")
    log = e.freeze(e.ROOT / "ops/conductor-log.md", raw / "sources")
    lines = Path(log["snapshot_reference"]["path"]).read_text().splitlines()
    slots, values, refs, gates = [], {}, [*methods["references"], log], []
    for index, task in enumerate(contract["tasks"][:-1]):
        e.progress("slot_before_" + task["id"], index, 14 - index)
        row = e.resolve(task, raw / "primaries", lines)
        slots.append(row)
        refs += [row["declared_reference"], row["primary_reference"], *row["evidence_references"]]
        if row["producer_executed"]:
            values[str(row["experiment_id"])] = e.projection(e.load(row["primary_reference"]))
        e.progress("slot_after_" + task["id"], index + 1, 13 - index)
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
    history_path = e.ROOT / "results/experiment_8694_v745_capstone.json"
    history_ref = dict(path=str(history_path), sha256="sha256:" + e.PINS[8694])
    history = e.load(history_ref)
    refs.append(history_ref)
    stamps = {}
    for number in (8681, 8691, 8692, 8693, 8694):
        path = next((e.ROOT / "results").glob(f"experiment_{number}_*.json"))
        e.verify(dict(path=str(path), sha256="sha256:" + e.PINS[number]))
        ref = e.freeze(path, raw / "history")
        value = e.load(ref)
        stamps[f"exp{number}"] = {
            k: value.get(k)
            for k in (
                "honest_verdict",
                "verdict_class",
                "flagged_adversarial",
                "required_checks_passed",
            )
        }
        refs.append(ref)
    prior = e.prior_rows(contract["tasks"], raw / "history")
    refs += [r["primary_reference"] for r in prior]
    refs += [ref for r in prior for ref in r["references"]]
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
    dependencies = code + [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "pyproject.toml",
        e.DESIGN,
        e.authority.DESIGN,
        "research-roadmap.yaml",
        "openspec/change-proposals/research-roadmap-v745-preserved-20261101.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
        "ops/exclusion_manifest.yaml",
        "ops/e2e-test-plan.md",
        "ops/verifier_gaps.md",
        "research-references.md",
        "research-studying.md",
        "docs/research-notes/v746-capstone.md",
        "scripts/experiment_template.py",
        "scripts/publication_gate.py",
        "scripts/summarize_artifact.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "scripts/check_spec_coverage.py",
        "ops/publication_gate_state.json",
        "docs/technical-report.md",
        "docs/arxiv-paper/main.tex",
        "tests/python/conftest.py",
        *[p.split("::")[0] for p in E2E],
    ]
    refs += [e.freeze(e.ROOT / name, raw / "sources") for name in sorted(set(dependencies))]
    refs += [e.reference(p) for p in authority_root.rglob("*") if p.is_file()]
    code_refs = [e.freeze(e.ROOT / name, raw / "code") for name in code]
    e.atomic_json(
        raw / "configuration.json",
        dict(
            owned=e.OWNED,
            tests=e.TESTS,
            consumers=E2E,
            flags=COMMON,
            required_statement_coverage=100,
        ),
    )
    code_refs.append(e.reference(raw / "configuration.json"))
    e.atomic_json(raw / "premeasurement_seal.json", dict(source_refs=refs, code_refs=code_refs))
    e.progress("inputs_sources_sealed_before_reduction", 13, 1)
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
            **r["primary_reference"],
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
    external += [
        g
        for r in slots
        for g in r["failed_or_missing_gate_operands"]
        if isinstance(g, dict) and not g.get("passed")
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
        values=values,
        prior_rows=prior,
        gate_rows=gates,
        external_checks=external,
        history={
            k: history[k]
            for k in (
                "historical_quarantines",
                "preserved_nulls",
                "preserved_v737_supervisor_null",
                "preserved_v743_batch_one_null",
                "historical_stamped_outcomes",
            )
        },
        historical_stamps=stamps,
        checkpoint_reference=checkpoint,
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


def validate(raw: Path) -> list[Json]:
    """Collect once and preserve each child's SQLite, JUnit and source before combination."""
    logs = raw / "validation"
    collected = child(
        "collect_once",
        [sys.executable, "-m", "pytest", *COMMON, "--collect-only", *e.TESTS, *E2E],
        logs,
        deadline=90,
    )
    nodes = [
        line.strip()
        for line in Path(collected["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    if not collected["passed"] or not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("single_exact_collection_failed")
    receipts, saved = [collected], []
    with tempfile.TemporaryDirectory(prefix="capstone8708-coverage-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        for index, suite in enumerate([*e.TESTS, *E2E]):
            selected = [n for n in nodes if n == suite or n.startswith(suite + "::")]
            if not selected:
                raise ValueError("empty_required_shard:" + suite)
            folder = scratch / str(index)
            folder.mkdir()
            config = logs / f"coverage-{index}.ini"
            role = "owned" if suite in e.TESTS else "consumer"
            include = e.OWNED if role == "owned" else [suite.split("::")[0]]
            config.write_text(
                "[run]\nparallel=true\npatch=subprocess\ndata_file="
                + str(folder / ".coverage")
                + "\ninclude=\n"
                + "".join("    " + str(e.ROOT / p) + "\n" for p in include)
                + "[report]\nexclude_lines=\n"
            )
            receipt = child(
                f"tests_{index}",
                [
                    "env",
                    "COVERAGE_RCFILE=" + str(config),
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + str(config),
                    "-m",
                    "pytest",
                    *COMMON,
                    *selected,
                    "--basetemp=" + str(folder / "tmp"),
                    "--junitxml=" + str(folder / "junit.xml"),
                ],
                logs,
                deadline=480,
            )
            ref = retained.retain(
                receipt,
                selected,
                folder,
                logs / "children" / str(index),
                [e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS],
                role,
            )
            saved.append(ref)
            receipt["retained_child_reference"] = ref
            receipts.append(receipt)
        union = retained.aggregate(saved, nodes)
        receipts.append(dict(name="exact_disjoint_node_union", **union))
        owned = [r for r in saved if retained.load(r)["role"] == "owned"]
        combined = retained.combine(owned, logs / "combined")
        receipts.append(
            dict(
                name="retained_owned_sqlite",
                passed=bool(combined["combined_reference"]),
                **combined,
            )
        )
        config = logs / "report.ini"
        config.write_text(
            "[run]\ndata_file="
            + str(logs / "combined/.coverage")
            + "\ninclude=\n"
            + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
            + "[report]\nexclude_lines=\n"
        )
        for name, args in [
            ("coverage_report", ["report", "--fail-under=100", "--show-missing"]),
            ("coverage_json", ["json", "-o", str(raw / "coverage.json")]),
        ]:
            receipts.append(
                child(
                    name,
                    [
                        sys.executable,
                        "-m",
                        "coverage",
                        args[0],
                        "--rcfile=" + str(config),
                        *args[1:],
                    ],
                    logs,
                    deadline=60,
                )
            )
    for name, args in [
        ("ruff_check", ["-m", "ruff", "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", ["-m", "ruff", "format", "--check", *e.OWNED, *e.TESTS]),
        (
            "strict_mypy",
            [
                "-m",
                "mypy",
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=skip",
                *e.OWNED,
            ],
        ),
        ("spec_coverage", ["scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]:
        receipts.append(child(name, [sys.executable, *args], logs, deadline=120))
    return receipts


def replay(path: Path) -> None:
    """Rebuild from independently sealed primitives; a rehashed candidate owns no authority."""
    value = json.loads(path.read_bytes())
    anchor = Path(value["publication_output"]).parent / "raw" / e.NAME
    raw = anchor / "invocations" / value["invocation_id"]
    seal = e.load(e.reference(anchor / "seals" / (raw.name + ".json")))
    if seal["measurement"] != value["primitive_reference"]:
        raise ValueError("independent_measurement_seal")
    work = e.load(value["primitive_reference"])
    if work["raw"] != str(raw) or work["code_refs"] != seal["code_refs"]:
        raise ValueError("independent_source_seal")
    for ref in work["source_refs"] + work["code_refs"] + work["operand_refs"]:
        if ref["sha256"] is not None:
            e.verify(ref)
        elif Path(ref["path"]).exists():
            raise ValueError("absent_operand_appeared")
        if "snapshot_reference" in ref:
            e.verify(ref["snapshot_reference"])
    if (
        work["contract"] != e.authority.authority(Path(work["authority_root"]))
        or work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
    ):
        raise ValueError("authority_or_clock_drift")
    validation_seal = e.load(e.reference(anchor / "validation_seals" / (raw.name + ".json")))
    if validation_seal["validation"] != value["validation_reference"]:
        raise ValueError("independent_validation_seal")
    validation = e.load(value["validation_reference"])
    for receipt in validation["receipts"] + [work["publication_receipt"]]:
        for stream in ("stdout", "stderr"):
            if (
                stream + "_path" in receipt
                and e.sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]
            ):
                raise ValueError("child_log_drift")
    children = [
        r["retained_child_reference"]
        for r in validation["receipts"]
        if "retained_child_reference" in r
    ]
    if children:
        union = next(r for r in validation["receipts"] if r["name"] == "exact_disjoint_node_union")
        if (
            dict(
                name="exact_disjoint_node_union", **retained.aggregate(children, union["node_ids"])
            )
            != union
        ):
            raise ValueError("retained_child_evidence")
    if report.build(work, validation["receipts"], Path(value["publication_output"])) != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed", 14, 0)


def terminal(path: Path, raw: Path) -> Json:
    """Keep unchanged scanners' actual severities on the exact final candidate bytes."""
    receipts = [
        child(name, [sys.executable, "-u", *args], raw, deadline=180)
        for name, args in [
            ("cold_replay", [e.CLI, "--cold-replay", str(path)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)]),
            ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)]),
        ]
    ]
    findings = json.loads(Path(receipts[1]["stdout_path"]).read_bytes())
    return dict(
        passed=all(r["passed"] for r in receipts) and findings["flagged_count"] == 0,
        receipts=receipts,
        actual_findings=findings,
    )


def run(output: Path, date: str, private: bool, failed_child: bool) -> int:
    """Private controls cannot escape scratch or count as production qualification."""
    if (private or failed_child) and (
        output.absolute().is_relative_to(e.ROOT) or output.resolve().is_relative_to(e.ROOT)
    ):
        raise ValueError("private controls require output outside checkout")
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, date, private)
    receipts = (
        [
            dict(
                name="private_control_only",
                passed=True,
                scope="private_fixture_not_production_validation",
            )
        ]
        if private
        else validate(raw)
    )
    if failed_child:
        receipts.append(
            child(
                "failed_child",
                [sys.executable, "-c", "raise SystemExit(1)"],
                raw / "validation",
                deadline=10,
            )
        )
    e.atomic_json(raw / "validation.json", dict(receipts=receipts))
    e.atomic_json(
        raw.parents[1] / "validation_seals" / (raw.name + ".json"),
        dict(validation=e.reference(raw / "validation.json")),
    )
    value = report.build(work, receipts, output)
    publication = publish_primary(
        output, value, lambda candidate: terminal(candidate, raw / "terminal")
    )
    reader = reader_receipt(
        e.TASK,
        output.parent,
        field="reconciliation_ready_score",
        expected=value["reconciliation_ready_score"],
    )
    e.atomic_json(
        raw / "terminal_validation.json",
        dict(publication=publication, consumer=reader, passed=reader["passed"]),
    )
    e.progress("complete_" + value["verdict_class"], 14, 0)
    return 0 if value["required_checks_passed"] and reader["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Reject malformed dates and private aliases before writing any measurement evidence."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero", 0, 14)
    os.chdir(e.ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261101")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if len(args.date) != 8:
            raise ValueError("calendar_date")
        datetime.strptime(args.date, "%Y%m%d")
        with budget(4800):
            return run(args.output.absolute(), args.date, args.private_control, args.failed_child)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        print(f"[exp8708] error={error}", file=sys.stderr, flush=True)
        return 1

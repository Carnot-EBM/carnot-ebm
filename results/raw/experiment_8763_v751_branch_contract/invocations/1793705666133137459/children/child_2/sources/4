"""REQ-VERIFY-8749: preserve real scoped checks independently of repository health."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import v750_archive_bound as a
from carnot.reporting import v746_retained_evidence as evidence
from carnot.reporting.collection_contract import collect
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace
from carnot.reporting import v749_contract_runner

Json = dict[str, Any]
terminal_runner = namespace(v749_contract_runner, "v750_terminal")
terminal_runner.CLI = a.CLI
terminal = terminal_runner.terminal


def build(work: Json, validation: Json, output: Path) -> Json:
    """Reduce independent branch custody and required checks without upgrading old verdicts."""
    checked = a.scoped_ready(validation) and not work["private_control"]
    contract = work["contract"]
    gates = dict(
        contract=contract["ready"],
        learning_inputs=work["learning"]["ready"],
        exclusion=work["exclusion_check"]["passed"],
        archived_authority=len(work["historical_fixture_rows"]) == 3
        and all(r["passed"] for r in work["historical_fixture_rows"]),
        mutations=bool(work["mutation_rows"]) and all(r["passed"] for r in work["mutation_rows"]),
        owned_failure_controls=len(work["owned_failure_controls"]) == 2
        and all(r["passed"] for r in work["owned_failure_controls"]),
        preconditions=work["preconditions"]["private_scratch_writable"]
        and all(work["preconditions"]["tools"].values()),
        owned_validation=checked,
    )
    verdict = (
        "blocked"
        if not gates["contract"] or not gates["learning_inputs"]
        else "positive"
        if all(gates.values())
        else "disqualified"
    )
    raw = Path(work["raw"])
    failures = [
        dict(
            upstream=a.TASK,
            path=str(raw / "work.json"),
            sha256=a.reference(raw / "work.json")["sha256"],
            check=k,
            field=k,
            operator="==",
            expected=True,
            observed=v,
            passed=False,
        )
        for k, v in gates.items()
        if not v
    ]
    failures += work["learning"].get("gate_check_summary", [])
    end = validation["ended_at_utc"]
    duration = (
        datetime.fromisoformat(end) - datetime.fromisoformat(work["started_at_utc"])
    ).total_seconds()
    tasks = [r.get("task_object") for r in contract["rows"]]
    ready = checked and all(
        gates[k]
        for k in ["archived_authority", "mutations", "owned_failure_controls", "preconditions"]
    )
    value = dict(
        experiment_id=8749,
        task_id=a.TASK,
        milestone=a.MILESTONE,
        run_date="20261103",
        invocation_id=work["invocation_id"],
        honest_verdict="complete_" + verdict + "_v750_archive_bound_contract",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        rows=contract["rows"],
        intended_count=14,
        completed_count=sum(r["absolute_metric"] for r in contract["rows"]),
        failed_count=sum(not r["absolute_metric"] for r in contract["rows"]),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        verifier_is_oracle=False,
        exposure_scope="administrative custody only; constructed and oracle-dependent science remains separately gated",
        independent_generalization_score=0,
        acceptance_gates=gates,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=end,
        duration_s=duration,
        phase_spans=[
            dict(
                phase="custody_and_scoped_validation",
                start_utc=work["started_at_utc"],
                end_utc=end,
                duration_s=duration,
            )
        ],
        random_seed=7508749,
        source_artifact_hashes=work["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[a.reference(raw / "work.json"), a.reference(raw / "validation.json")],
        cited_upstream_artifacts=work["references"],
        repository_health_reference=work["repository_health_reference"],
        scoped_validation_scope=dict(
            required_suites=a.TESTS + a.HISTORICAL + a.CONSUMERS,
            exact_node_ids=validation["union"].get("node_ids", []),
            repository_health_is_dependency=False,
            original_assertions_unchanged=True,
            e2e=["E2E-018", "E2E-021"],
        ),
        contract_ready_score=int(gates["contract"] and ready),
        learning_protocol_ready_score=int(gates["learning_inputs"] and ready),
        archived_authority_ready_score=int(gates["archived_authority"] and ready),
        full_task_objects=tasks,
        canonical_tasks_sha256=a.PIN,
        historical_fixture_rows=work["historical_fixture_rows"],
        owned_failure_controls=work["owned_failure_controls"],
        branch_method_references=work["branch_method_references"],
        preserved_primary_rows=work["preserved_primary_rows"],
        archived_validation_recipe=work["archived_validation_recipe"],
        contract_mutation_rows=work["mutation_rows"],
        visible_task_table=work["visible_table"],
        learning_protocol=work["learning"],
        learning_execution_ready_score=0,
        learning_execution_authorized=False,
        new_scientific_measurements=0,
        primitive_reference=a.reference(raw / "work.json"),
        validation_reference=a.reference(raw / "validation.json"),
        frozen_input_reference=work["frozen_input_reference"],
        publication_output=str(output.absolute()),
        methodology_note="Bind fourteen complete tasks to an independent digest. Freeze original V744 objectives, partitions and criteria. Replay unchanged V748 assertions under an archived private root. Require exact scoped nodes and actual coverage independently of the retained timed-out full-suite diagnostic. No new learning, verifier transfer, cost or hardware result is measured.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Bind producer bytes and current execution; administrative custody never grants scientific promotion.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """A separate seal and fresh reconstruction reject consistently rehashed report edits."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["primitive_reference"]["path"]).parent
        seal = json.loads((raw / "independent_seal.json").read_bytes())
        if seal != {
            k: value[k]
            for k in ["primitive_reference", "validation_reference", "frozen_input_reference"]
        }:
            return False
        work, validation = (
            a.load(value["primitive_reference"]),
            a.load(value["validation_reference"]),
        )
        frozen = a.load(value["frozen_input_reference"])
        mutable = {"mutation_rows", "historical_fixture_rows", "owned_failure_controls"}
        if any(work[k] != v for k, v in frozen.items() if k not in mutable):
            return False
        refs = work["sources"] + work["references"] + validation.get("raw_references", [])
        if any(a.reference(Path(r["path"]))["sha256"] != r["sha256"] for r in refs):
            return False
        if a.authority(Path(work["root"])) != work["contract"]:
            return False
        return bool(value == build(work, validation, Path(value["publication_output"])))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def coverage_summary(raw: Path, children: list[Json]) -> Json:
    """Count executable statements from actual parent and child SQLite without exclusions."""
    combined = evidence.combine(children, raw / "combined")
    target = Path(combined["combined_reference"]["path"])
    shutil.copyfile(raw / "producer.sqlite", target.parent / ".coverage.producer")
    cov = Coverage(data_file=str(target), config_file=False)
    cov.load()
    cov.combine(keep=True)
    cov.save()
    files = []
    for name in a.OWNED:
        _, statements, excluded, missing, _ = cov.analysis2(str(a.ROOT / name))
        files.append(
            dict(path=name, statements=len(statements), excluded=excluded, missing=missing)
        )
    total = sum(r["statements"] for r in files)
    missing_count = sum(len(r["missing"]) for r in files)
    return dict(
        complete=total > 0 and missing_count == 0 and not any(r["excluded"] for r in files),
        percent=100 * (total - missing_count) / total,
        files=files,
        combined_reference=a.reference(target),
    )


def validate(raw: Path, tracer: Coverage, archived_suites: list[str]) -> Json:
    """Every original scoped test remains required while global health stays separate."""
    with TemporaryDirectory(prefix="exp8749-checks-") as directory:
        suites = a.TESTS + a.CONSUMERS + [os.path.relpath(s, a.ROOT) for s in archived_suites]
        plan, collected = collect(Path(directory), raw, suites, a.TESTS, child)
        atomic_json(raw / "frozen_test_plan.json", plan)
        receipts, children = [collected], []
        for spec in plan:
            receipt = child(spec["name"], spec["argv"], raw / "checks", deadline=600)
            receipts.append(receipt)
            children.append(
                evidence.retain(
                    receipt,
                    spec["nodes"],
                    Path(spec["scratch"]),
                    raw / "children" / spec["name"],
                    [a.reference(a.ROOT / p) for p in a.OWNED + a.TESTS],
                    spec["role"],
                )
            )
        for name, argv in [
            ("ruff", [".venv/bin/ruff", "check", *a.OWNED, *a.TESTS]),
            ("format", [".venv/bin/ruff", "format", "--check", *a.OWNED, *a.TESTS]),
            ("types", [".venv/bin/mypy", "--strict", "--follow-imports=silent", *a.OWNED]),
            ("spec", [sys.executable, "scripts/check_spec_coverage.py", *a.TESTS]),
        ]:
            receipts.append(child(name, argv, raw / "checks", deadline=120))
        union = evidence.aggregate(children, [n for s in plan for n in s["nodes"]])
        tracer.stop()
        tracer.save()
        coverage = coverage_summary(raw, children)
        refs = children + [ref for r in children for ref in evidence.load(r)["sqlite_references"]]
        refs += [
            dict(path=r[s + "_path"], sha256=r[s + "_sha256"])
            for r in receipts
            for s in ["stdout", "stderr"]
        ]
        return dict(
            receipts=receipts,
            retained_child_rows=children,
            union=union,
            coverage=coverage,
            raw_references=refs,
            ended_at_utc=datetime.now(UTC).isoformat(),
        )


def main(argv: list[str] | None = None) -> int:
    """Publish only terminal checked bytes; private controls cannot grant readiness."""
    a.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261103"], default="20261103")
    parser.add_argument("--output", type=Path, default=a.ROOT / "results" / (a.NAME + ".json"))
    parser.add_argument("--root", type=Path, default=a.ROOT)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        a.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.resolve().is_relative_to(a.ROOT.resolve()):
        parser.error("private outputs must remain outside the checkout")
    raw = output.parent / "raw" / a.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    started = datetime.now(UTC).isoformat()
    tracer = Coverage(data_file=str(raw / "producer.sqlite"), config_file=False)
    if not args.private_control:
        tracer.start()
    with TemporaryDirectory(prefix="exp8749-fixture-") as directory:
        private = Path(directory)
        probe = private / "writable"
        probe.write_text("private scratch")
        a.progress("authenticate_and_freeze")
        work = a.freeze(raw, args.root)
        work.update(
            root=str(args.root.absolute()),
            invocation_id=raw.name,
            started_at_utc=started,
            private_control=args.private_control,
            preconditions=dict(
                private_scratch_writable=probe.read_text() == "private scratch",
                tools={
                    name: (a.ROOT / ".venv/bin" / name).is_file()
                    for name in ["python", "pytest", "ruff", "mypy"]
                },
                no_model_needed=True,
            ),
            mutation_rows=[],
            historical_fixture_rows=[],
            owned_failure_controls=[],
            visible_table=parse_design((args.root / a.DESIGN).read_text(), milestone=a.MILESTONE)[
                0
            ],
        )
        # Keep actual fixture bytes after private scratch is removed.
        dependencies = []
        archived_suites = []
        for version in [748, 749]:
            a.progress("freeze_archived_fixture_" + str(version))
            fixture = a.archived_root(private / f"archive_{version}", version)
            suite = a.historical_suite(fixture, raw / f"original_tests_{version}", version)
            archived_suites.append(suite)
            cli = (
                a.HISTORICAL_CLI
                if version == 748
                else "scripts/experiments/experiment_8737_v749_contract_methods.py"
            )
            for index, path in enumerate(
                [
                    fixture / a.DESIGN,
                    fixture / a.ACTIVE,
                    fixture / a.STAGED,
                    fixture / cli,
                    Path(suite),
                    Path(suite).parent / "conftest.py",
                ]
            ):
                target = raw / f"archived_dependency_{version}_{index}"
                shutil.copyfile(path, target)
                dependencies.append(a.reference(target))
        work["references"] += dependencies
        work["archived_validation_recipe"]["dependencies"] = dependencies
        atomic_json(raw / "frozen_inputs.json", work)
        work["frozen_input_reference"] = a.reference(raw / "frozen_inputs.json")
        if args.private_control:
            validation = dict(
                receipts=[dict(name="private_control", passed=False)],
                union=dict(passed=False),
                coverage=dict(complete=False),
                ended_at_utc=datetime.now(UTC).isoformat(),
            )
        else:
            a.progress("reproduce_original_child5_failure")
            receipt = child(
                "moving_roadmap_failure",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    "tests/python/test_v748_contract_8723.py::test_conductor_prompt_encoding[active]",
                    "--basetemp=" + str(private / "failure_tmp"),
                    "--junitxml=" + str(raw / "moving_failure.xml"),
                ],
                raw / "checks",
                expected=1,
                deadline=120,
            )
            work["historical_fixture_rows"].append(
                dict(
                    check="original_mutable_vnext_failure",
                    passed=receipt["passed"]
                    and "V748 JSON contract missing" in Path(receipt["stdout_path"]).read_text(),
                    receipt=receipt,
                )
            )
            a.progress("full_task_mutation_controls")
            work["mutation_rows"] = a.mutations(private / "mutations")
            a.progress("scoped_validation")
            validation = validate(raw, tracer, archived_suites)
            for index, version in enumerate([748, 749]):
                historical = validation["receipts"][len(a.TESTS + a.CONSUMERS) + 1 + index]
                work["historical_fixture_rows"].append(
                    dict(
                        check=f"unchanged_archived_v{version}_tests",
                        passed=historical["passed"],
                        receipt=historical,
                    )
                )
            failure_node = private / "test_owned_failure.py"
            failure_node.write_text("def test_owned_failure():\n    assert False\n")
            failed_receipt = child(
                "deliberate_owned_failure",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-c",
                    "/dev/null",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    str(failure_node),
                    "--basetemp=" + str(private / "owned_failure_tmp"),
                    "--junitxml=" + str(raw / "owned_failure.xml"),
                ],
                raw / "checks",
                deadline=60,
            )
            failed = dict(validation, receipts=[failed_receipt])
            work["owned_failure_controls"] = [
                dict(
                    check="owned_failure_zeros_readiness",
                    passed=failed_receipt["exit_code"] == 1 and not a.scoped_ready(failed),
                    receipt=failed_receipt,
                ),
                dict(
                    check="unrelated_global_health_stays_red",
                    passed=work["repository_health_reference"]["full_suite_pass"] is False,
                ),
            ]
            validation["ended_at_utc"] = datetime.now(UTC).isoformat()
        atomic_json(raw / "work.json", work)
        atomic_json(raw / "validation.json", validation)
        atomic_json(
            raw / "independent_seal.json",
            dict(
                primitive_reference=a.reference(raw / "work.json"),
                validation_reference=a.reference(raw / "validation.json"),
                frozen_input_reference=work["frozen_input_reference"],
            ),
        )
        value = build(work, validation, output)
        a.progress("before_terminal_validation")
        published = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
        atomic_json(raw / "publication_receipt.json", published)
        atomic_json(
            raw / "reader_receipt.json",
            reader_receipt(
                a.TASK,
                output.parent,
                field="contract_ready_score",
                expected=value["contract_ready_score"],
            ),
        )
        a.progress("published_" + value["verdict_class"], 14, 0)
    return 0

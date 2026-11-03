"""REQ-REPORT-8754: thin orchestration reuses private validation and atomic publication."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from coverage import Coverage
import yaml

from carnot.reporting import verifier_surface_methods_8754 as m
from carnot.reporting import verifier_surface_report_8754 as report
from carnot.reporting import v750_archive_bound as archive
from carnot.reporting import v750_archive_runner as original
from carnot.reporting import v749_contract_runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
validation_runner = namespace(original, "surface_validation")
validation_runner.a = m
terminal_runner = namespace(v749_contract_runner, "surface_terminal")
terminal_runner.CLI = m.CLI
terminal = terminal_runner.terminal
validate = validation_runner.validate


def validation_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """New modules use strict type checking without legacy repository relaxations."""
    if name == "types":
        argv = [
            argv[0],
            "--config-file=/dev/null",
            "--strict",
            "--follow-imports=silent",
            "--ignore-missing-imports",
            *m.OWNED,
        ]
    return child(name, argv, logs, **kwargs)


validation_runner.child = validation_child


def prepare(raw: Path, private: Path, protocol: Path, private_control: bool) -> Json:
    """Freeze exact inputs and source closure before any measurement or test child."""
    started = datetime.now(UTC).isoformat()
    raw.mkdir(parents=True, exist_ok=True)
    scratch = private / "writable"
    scratch.write_text("private scratch")
    gates, refs = [], []
    protocol_ref = dict(path=str(protocol), sha256=m.PIN)
    try:
        canonical = m.retained(protocol)
        transformed, transforms, truth_rows = m.surface(canonical)
        mutations = m.mutations(canonical, transformed)
        refs.append(protocol_ref)
    except (OSError, ValueError, KeyError, TypeError) as error:
        canonical, transformed, transforms, truth_rows, mutations = {}, {}, [], [], []
        gates.append(
            dict(
                upstream="exp8737-contract-methods",
                **protocol_ref,
                check="retained_protocol_authentication",
                field="sha256_and_protocol",
                operator="==",
                expected=m.PIN,
                observed="missing_file" if not protocol.exists() else str(error),
                passed=False,
            )
        )
    workloads = []
    for label, value in [
        ("canonical", canonical),
        ("transformed", transformed),
        ("surface_transform", dict(random_seed=m.SEED, rows=transforms)),
    ]:
        atomic_json(raw / (label + ".json"), value)
        workloads.append(m.reference(raw / (label + ".json")))
    primary = m.ROOT / "results/experiment_8737_v749_contract_methods.json"
    prior = m.load(m.reference(primary))
    previous_validation = m.load(prior["validation_reference"])
    health = next(r for r in previous_validation["receipts"] if r["name"] == "full_python_suite")
    current_health = m.ROOT / "results/raw" / m.NAME / "startup/full_python_suite.receipt.json"
    if current_health.is_file():
        current = m.load(m.reference(current_health))
        refs += [m.reference(current_health)] + [
            dict(path=current[s + "_path"], sha256=current[s + "_sha256"])
            for s in ["stdout", "stderr"]
        ]
    refs += [
        m.reference(primary),
        prior["validation_reference"],
        m.reference(m.ROOT / "results/experiment_8729_v748_equal_budget_verifier.json"),
    ]
    source_files = (
        m.OWNED
        + m.TESTS
        + archive.HISTORICAL
        + m.CONSUMERS
        + [
            "python/carnot/reporting/v749_verifier_protocol.py",
            "python/carnot/reporting/verification_methods_8726.py",
            "python/carnot/reporting/equal_budget_verifier_8729.py",
            "python/carnot/reporting/v750_archive_bound.py",
            "python/carnot/reporting/v750_archive_runner.py",
            "python/carnot/reporting/collection_contract.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
    )
    for name in [*archive.ARCHIVES, "ops/exclusion_manifest.yaml"]:
        refs.append(m.reference(m.ROOT / name))
    excluded = yaml.safe_load((m.ROOT / "ops/exclusion_manifest.yaml").read_bytes())
    retired = [
        r
        for r in excluded["retired"] + excluded.get("retired_experiments", [])
        if isinstance(r, dict) and str(r.get("experiment_id")) == "8754"
    ]
    assert not retired, "task_retired"
    analysis = dict(
        correctness="all three truth values and valid alternatives; reject_all must not qualify",
        transfer=dict(
            primary_surface="canonical",
            evaluation_worlds=30,
            defective_units_per_arm=150,
            new_family_units_per_arm=60,
            criterion="new-family detection delta lower95 > 0 and no extra correct-verifier false rejection",
        ),
        invariance="pair identical world/variant/arm; zero truth discrepancies and zero correct-verifier decision disagreements",
        cluster_reduction=dict(
            cluster="world_id",
            paired=True,
            bootstrap_draws=10000,
            random_seed=m.SEED,
            old_and_new_families_separate=True,
        ),
        timing=dict(
            repetitions=10,
            early_exit=False,
            boundary="all five calls including predicate work",
            repeats_increase_sample_size=False,
            equal_calls_imply_equal_time=False,
        ),
        probe_design=dict(
            excluded_defect_families=list(m.historical.NEW),
            source="unchanged Exp8726 schedules; no outcomes used",
        ),
    )
    summaries = [
        dict(
            reference=r,
            intended_rows=720,
            intended_calls=3600,
            evaluation_worlds=30,
            independent_worlds=60,
            status="frozen_unexecuted",
        )
        for r in workloads[:2]
    ]
    return dict(
        raw=str(raw),
        invocation_id=raw.name,
        started_at_utc=started,
        frozen_at_utc=datetime.now(UTC).isoformat(),
        private_control=private_control,
        input_ready=not gates,
        gate_check_summary=gates,
        sources=[m.reference(m.ROOT / p) for p in source_files],
        references=refs,
        verifier_protocol_reference=protocol_ref,
        workload_references=workloads,
        surface_transform_reference=workloads[2],
        canonical_workload=summaries[0],
        transformed_workload=summaries[1],
        truth_equivalence_rows=truth_rows,
        method_mutation_rows=mutations,
        analysis_protocol=analysis,
        preconditions=dict(
            private_scratch_writable=scratch.read_text() == "private scratch",
            tools={
                name: (m.ROOT / ".venv/bin" / name).is_file()
                for name in ["python", "pytest", "ruff", "mypy"]
            },
            no_model_needed=True,
            exclusion_check_passed=not retired,
        ),
        repository_health_reference=dict(
            primary_reference=m.reference(primary),
            receipt=health,
            full_suite_pass=False,
            historical_producer_promoted=False,
            current_invocation_diagnostic=m.reference(current_health)
            if current_health.is_file()
            else None,
        ),
    )


def replay(path: Path) -> bool:
    """Fresh reconstruction binds every imported byte and rejects rehashed aggregates."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["primitive_reference"]["path"]).parent
        seal = json.loads((raw / "independent_seal.json").read_bytes())
        if seal != {k: value[k] for k in ["primitive_reference", "validation_reference"]}:
            return False
        work, validation = (
            m.load(value["primitive_reference"]),
            m.load(value["validation_reference"]),
        )
        for ref in (
            work["sources"]
            + work["references"]
            + work["workload_references"]
            + validation.get("raw_references", [])
        ):
            m.load(ref) if Path(ref["path"]).suffix == ".json" else None
            if m.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                return False
        if work["input_ready"]:
            canonical = m.retained(Path(work["verifier_protocol_reference"]["path"]))
            transformed, controls, checks = m.surface(canonical)
            if (
                m.load(work["canonical_workload"]["reference"]) != canonical
                or m.load(work["transformed_workload"]["reference"]) != transformed
                or checks != work["truth_equivalence_rows"]
                or m.load(work["surface_transform_reference"])["rows"] != controls
            ):
                return False
        return bool(value == report.build(work, validation, Path(value["publication_output"])))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv: list[str] | None = None) -> int:
    """Only terminal validated bytes become visible; private controls never qualify."""
    m.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261103"], default="20261103")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--protocol", type=Path, default=m.PROTOCOL)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.resolve().is_relative_to(m.ROOT.resolve()):
        parser.error("private outputs must remain outside the checkout")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    tracer = Coverage(data_file=str(raw / "producer.sqlite"), config_file=False)
    if not args.private_control:
        tracer.start()
    with TemporaryDirectory(prefix="exp8754-private-") as directory:
        private = Path(directory)
        m.progress("authenticate_and_freeze")
        work = prepare(raw, private, args.protocol, args.private_control)
        suites = []
        if args.private_control:
            validation = dict(
                receipts=[dict(name="private_control", passed=False)],
                union=dict(passed=False),
                coverage=dict(complete=False),
                ended_at_utc=datetime.now(UTC).isoformat(),
            )
        else:
            for version in [748, 749]:
                m.progress("freeze_archived_tests_" + str(version))
                fixture = archive.archived_root(private / str(version), version)
                suite = archive.historical_suite(
                    fixture, raw / f"original_tests_{version}", version
                )
                suites.append(suite)
                for name in [archive.DESIGN, archive.ACTIVE, archive.STAGED]:
                    target = raw / f"archive_{version}_{Path(name).name}"
                    shutil.copyfile(fixture / name, target)
                    work["references"].append(m.reference(target))
                work["sources"] += [
                    m.reference(Path(suite)),
                    m.reference(Path(suite).parent / "conftest.py"),
                ]
            atomic_json(raw / "frozen_inputs.json", work)
            m.progress("before_scoped_validation")
            validation = validate(raw, tracer, suites)
            m.progress("after_scoped_validation")
        atomic_json(raw / "work.json", work)
        atomic_json(raw / "validation.json", validation)
        atomic_json(
            raw / "independent_seal.json",
            dict(
                primitive_reference=m.reference(raw / "work.json"),
                validation_reference=m.reference(raw / "validation.json"),
            ),
        )
        value = report.build(work, validation, output)
        m.progress("before_terminal_validation")
        published = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
        atomic_json(raw / "publication_receipt.json", published)
        atomic_json(
            raw / "reader_receipt.json",
            reader_receipt(
                m.TASK,
                output.parent,
                field="verifier_transfer_methods_ready_score",
                expected=value["verifier_transfer_methods_ready_score"],
            ),
        )
        m.progress("published_" + value["verdict_class"], 480, 0)
    return 0

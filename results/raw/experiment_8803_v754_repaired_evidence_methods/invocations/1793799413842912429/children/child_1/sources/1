"""REQ-VERIFY-8803: retain genuine validation and publish only replayed terminal bytes."""

import argparse
from datetime import UTC, datetime
import importlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

import yaml

from carnot.reporting import repaired_evidence_8803 as m
from carnot.reporting import evidence_evaluation_runner_8779 as shared
from carnot.reporting import v754_repaired_memory as authority
from carnot.reporting import v753_qualification_runner as historical
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]


def now() -> str:
    """UTC clocks document real phases rather than duration padding."""
    return datetime.now(UTC).isoformat()


def preflight(private: Path) -> Json:
    """Charge imports and custody once; only the measured cheap unit scales with work."""
    start = time.monotonic()
    private.mkdir(parents=True, exist_ok=True)
    probe = private / "writable"
    probe.write_text("private8803")
    imports = [importlib.import_module(name).__name__ for name in ["coverage", "yaml"]]
    tools = all(
        shutil.which(str(m.ROOT / ".venv/bin" / name))
        for name in ["python", "pytest", "ruff", "mypy"]
    )
    current = authority.authority(m.ROOT)
    preserved = m.ROOT / m.ARCHIVE
    tasks = parse_design(preserved.read_text(), milestone="2026.11.753")[1]
    history: Json = dict(
        milestone="2026.11.753", source=reference(preserved), task_digest=authority.digest(tasks)
    )
    source = reference(m.HISTORY)
    auth = m.old.authenticate(m.old.UPSTREAM_METHOD, m.old.UPSTREAM_ROWS)
    manifest = yaml.safe_load((m.ROOT / "ops/exclusion_manifest.yaml").read_bytes())
    retired = any(
        str(r.get("experiment_id")) in {"8803", m.TASK}
        for key in ["retired", "retired_experiments"]
        for r in manifest.get(key, [])
        if isinstance(r, dict)
    )
    setup = time.monotonic() - start
    cheap = time.monotonic()
    sample = m.views(m.variants(((0, 1), (0, 2), (1, 2), (2, 3))), "projection_probe")
    unit = time.monotonic() - cheap
    memory = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    free = shutil.disk_usage(private).free
    checks = dict(
        imports=bool(imports),
        tools=tools,
        private_scratch=probe.read_text() == "private8803",
        current_authority=current["ready"],
        historical_authority=history["source"]["sha256"] == m.ARCHIVE_PIN
        and history["task_digest"]
        == "472ded8826d2026a662403a8908a9c6fb1a9ef90c3d52b5d3547a65b05b1432b",
        historical_primary=source["sha256"] == m.HISTORY_PIN,
        original_sources=auth["passed"],
        not_retired=not retired,
        memory=memory > 512 * 1024**2,
        disk=free > 256 * 1024**2,
        projected_time=setup + unit * 312 + 900 < 4800,
    )
    diagnostics = [
        dict(
            check=name,
            upstream=m.TASK,
            artifact_field=name,
            expected=True,
            observed=value,
            path=str(m.ROOT),
            sha256=None,
        )
        for name, value in checks.items()
        if not value
    ]
    return dict(
        passed=all(checks.values()),
        checks=checks,
        diagnostics=diagnostics,
        dependencies=imports,
        historical_authority=history,
        current_authority=current,
        authentication=auth,
        historical_primary=source,
        available_memory_bytes=memory,
        free_disk_bytes=free,
        resource_projection=dict(
            setup_seconds_once=setup,
            measured_probe_seconds=unit,
            probe_units=len(sample),
            projected_work_seconds=unit * 312,
            reserved_validation_seconds=900,
            projected_total_seconds=setup + unit * 312 + 900,
            scientific_acceptance_separate=True,
        ),
    )


def unqualified() -> Json:
    """Private CLI controls cannot manufacture production validation receipts."""
    return dict(
        receipts=[dict(name="unqualified_private_control", passed=False)],
        union=dict(passed=False),
        coverage=dict(complete=False),
        original_coverage=dict(complete=False, statements=0),
        raw_references=[],
    )


def validate(raw: Path, private: Path) -> Json:
    """The shipped collector retains original nodes, assertions and actual child SQLite."""
    validator = namespace(shared, "repaired8803_validation")
    validator.e = SimpleNamespace(
        OWNED=m.OWNED + m.old.OWNED, TESTS=m.TESTS + m.old.TESTS, ROOT=m.ROOT
    )
    result = dict(validator.validate(raw, private))
    database = Path(result["coverage"]["combined_reference"]["path"])
    result["original_coverage"] = historical.coverage_summary(database, m.old.OWNED)
    result["original_coverage"]["statements"] = sum(
        f["statements"] for f in result["original_coverage"]["files"]
    )
    result["coverage"] = historical.coverage_summary(database, m.OWNED)
    return result


def consensus_ok(plan: Json) -> bool:
    """Consensus unanimity cannot certify truth when every constructed answer is wrong."""
    rows = plan["consensus"]
    return (
        len(rows) == 6480
        and all(
            len(r["candidates"]) == 3
            and (
                (all(x == r["expected"] for x in r["candidates"]))
                if r["group"] == "unanimous_correct"
                else (len(set(r["candidates"])) == 1 and r["candidates"][0] != r["expected"])
                if r["group"] == "unanimous_wrong"
                else len(set(r["candidates"])) == 3
            )
            and r["sufficient"] == (r["expected"] != "insufficient")
            for r in rows
        )
        and all(
            {r["sufficient"] for r in rows if r["group"] == group} == {True, False}
            for group in m.GROUPS
        )
    )


def build(work: Json, validation: Json) -> Json:
    """Independent protocol gates grant no learned extraction or population benefit claim."""
    runnable = work["preconditions_checked"]["passed"]
    plan = load(work["plan_reference"])
    decisions = load(work["decisions_reference"])["rows"]
    fixtures = m.controls() if runnable else []
    truth = [
        dict(
            unit_id=f"control/{i}",
            expected=f["expected"],
            boolean_completion=m.oracle.enumerate_truth(f["input"]),
            strong_decision=m.old.strong(f["input"]),
            renamed_decision=m.old.strong(m.c.rename(f["input"])),
        )
        for i, f in enumerate(fixtures)
    ]
    rows = [
        dict(
            r,
            unit_id=f"control/{r['input_index']}",
            cluster_id="tiny_control_cluster",
            arm=r["arm"],
            seed=m.SEED,
            metric=int(r["decision"] == fixtures[r["input_index"]]["expected"]),
            denominator=1,
            raw_numerator=int(r["decision"] == fixtures[r["input_index"]]["expected"]),
            raw_denominator=1,
            status="completed",
            missing_reason=None,
        )
        for r in decisions
    ]
    checks = (
        bool(validation["receipts"])
        and all(r["passed"] for r in validation["receipts"])
        and validation["union"]["passed"]
        and validation["coverage"]["complete"]
        and validation["original_coverage"]["complete"]
        and validation["original_coverage"]["statements"] == 377
    )
    gates = dict(
        preconditions=runnable,
        scoped_validation=checks,
        original_repair=bool(work["repair_controls"])
        and all(r["passed"] for r in work["repair_controls"]),
        independent_truth=bool(truth)
        and all(
            r["expected"]
            == r["boolean_completion"]["decision"]
            == r["strong_decision"]
            == r["renamed_decision"]
            for r in truth
        ),
        input_only_child=all(r["passed"] for r in work["measurement_receipts"]) and bool(decisions),
        disjoint=plan["split_manifest"].get("disjoint", False),
        length_balance=bool(plan["length_balance_rows"])
        and all(r["passed"] for r in plan["length_balance_rows"]),
    )
    evidence_ready = all(gates.values())
    consensus_ready = (
        checks
        and runnable
        and gates["disjoint"]
        and gates["independent_truth"]
        and work["consensus_controls_passed"]
    )
    verdict = (
        "blocked"
        if not runnable
        else "circular_positive"
        if evidence_ready and consensus_ready
        else "disqualified"
    )
    if verdict == "disqualified":
        evidence_ready = consensus_ready = False
    value: Json = dict(
        experiment_id=8803,
        task_id=m.TASK,
        milestone="2026.11.754",
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_constructed_protocol_qualification",
        verdict_class=verdict,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        verifier_is_oracle=True,
        duration_s=work["duration_s"],
        random_seed=m.SEED,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"] + validation["raw_references"],
        preconditions_checked=work["preconditions_checked"],
        phase_spans=work["phase_spans"],
        rows=rows,
        intended_count=len(m.controls()) * 4,
        completed_count=len(rows),
        censored_count=len(m.controls()) * 4 - len(rows),
        failed_count=0,
        excluded_count=0,
        independent_count=int(bool(rows)),
        n_samples=int(bool(rows)),
        gate_check_summary=work["preconditions_checked"].get("diagnostics", []),
        acceptance_gates=gates,
        required_checks_passed=bool(checks),
        owned_statement_coverage=validation["coverage"],
        validation_receipts=validation["receipts"],
        original_assertion_receipts=dict(
            coverage=validation["original_coverage"],
            original_tests=m.old.TESTS,
            test_identity=work["original_test_reference"],
            repair_controls=work["repair_controls"],
            preserved_authority=work["preconditions_checked"].get("historical_authority"),
        ),
        exposure_scope="V751-V753 primaries, numerical pilot and 120-world roster are exposed history. New cohorts are sealed, unmeasured future inputs for Exp8805/8806. Tiny constructed controls grant no natural-language or learned generalization claim. Canary worlds are reserved; no Qwen work.",
        independent_generalization_score=0,
        evidence_interventions_ready_score=int(evidence_ready),
        consensus_protocol_ready_score=int(consensus_ready),
        split_manifest=plan["split_manifest"],
        truth_control_rows=truth,
        length_balance_rows=plan["length_balance_rows"],
        strong_reference=dict(
            implementation="unchanged evidence_intervention_methods_8791.strong",
            independent_check="evidence_set_corpus_8768.enumerate_truth plus hand-inspected counterexamples",
            unlimited_rows=[r for r in rows if r["arm"] == "unlimited_exact_reference"],
        ),
        primitive_budget_definition=dict(
            primitive="Python LINE events in unchanged reporting solver code, including validation and repeated scans; wrapper and external library initialization excluded",
            limit=m.BUDGET,
            equal_for=list(m.old.ARMS),
            excludes="C builtin internals; not a hardware FLOP or time equivalence",
            unlimited_reference_separate=True,
        ),
        consensus_protocol=dict(
            constructed_candidates=True,
            groups=list(m.GROUPS),
            crossed_with=["support", "refute", "insufficient"],
            scorer_fields=["facts", "query", "candidates"],
            sealed_reference=work["plan_reference"],
            input_reference=work["cohort_references"]["consensus_inputs"],
            label_reference=work["cohort_references"]["consensus_labels"],
            ready=bool(consensus_ready),
        ),
        cohort_references=work["cohort_references"],
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        validation_reference=reference(Path(work["raw"]) / "validation.json"),
        cited_upstream_artifacts=[
            dict(
                experiment_id=8791,
                fields_imported=["verdict_class", "exposure_scope"],
                **reference(m.HISTORY),
            )
        ],
        methodology_note="Deterministic qualification and future corpus sealing only. Compared local and set-level validators are unchanged. All exact reference gains are circular; no inference, numerical training or independent population benefit is claimed. Equal Python line budgets do not imply equal machine instructions. Original assertions and historical primary bytes remain unchanged.",
        repository_health_reference=work["repository_health_reference"],
    )
    value["field_principles"] = {
        k: "Measured or reconstructed from exact sealed raw operands; protocol readiness is distinct from generalization."
        for k in value
    }
    value["field_principles"].update(
        field_principles="Definitions apply to every added top-level field.",
        reproducibility_checksum="Canonical hash of all other artifact fields.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct decisions and corpus bytes, rejecting repaired local hashes and aggregates."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["primitive_reference"]["path"]).parent
        seal = json.loads((raw / "independent_seal.json").read_bytes())
        if any(value[k] != seal[k] for k in ["primitive_reference", "validation_reference"]):
            return False
        work, validation = load(value["primitive_reference"]), load(value["validation_reference"])
        for ref in (
            work["code_config_hashes"]
            + work["source_artifact_hashes"]
            + work["raw_shard_hashes"]
            + validation["raw_references"]
        ):
            if reference(Path(ref["path"])) != ref:
                return False
        if work["preconditions_checked"]["passed"]:
            if (
                reference(m.HISTORY)["sha256"] != m.HISTORY_PIN
                or reference(m.ROOT / m.ARCHIVE)["sha256"] != m.ARCHIVE_PIN
            ):
                return False
            plan = load(work["plan_reference"])
            if canonical_hash(plan) != canonical_hash(m.freeze()):
                return False
            actual = load(work["decisions_reference"])["rows"]
            if actual != m.measure([r["input"] for r in m.controls()]):
                return False
            if work["consensus_controls_passed"] != consensus_ok(plan):
                return False
        return bool(value == build(work, validation))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(candidate: Path, raw: Path) -> Json:
    """Unchanged scanners and an outside-checkout interpreter must accept these exact bytes."""
    receipts = []
    for name, argv in [
        (
            "adversarial",
            [sys.executable, "scripts/adversarial_verify.py", str(candidate), "--json"],
        ),
        (
            "row_consistency",
            [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
        ),
        (
            "cold_replay",
            [
                "env",
                "-u",
                "PYTHONPATH",
                sys.executable,
                "-c",
                "import os,runpy,sys; os.chdir('/tmp'); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",
                str(m.ROOT / m.CLI),
                "--cold-replay",
                str(candidate),
            ],
        ),
    ]:
        receipts.append(child(name, argv, raw / "terminal_checks", deadline=120))
    result = dict(passed=all(r["passed"] for r in receipts), receipts=receipts)
    atomic_json(raw / "terminal_validation.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Seal future cohorts, trace original controls and publish an honest terminal verdict."""
    m.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261104"], default="20261104")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--decide-input", type=Path)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    if args.decide_input:
        if args.decisions is None:
            parser.error("--decisions required")
        payloads = json.loads(
            args.decide_input.read_bytes(),
            object_pairs_hook=m.original.unique_object,
            parse_constant=m.original.reject_constant,
        )["rows"]
        atomic_json(args.decisions, dict(rows=m.measure(payloads)))
        return 0
    output = args.output.absolute()
    if args.private_control and output.resolve().is_relative_to(m.ROOT.resolve()):
        parser.error("private output must be outside the checkout")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    started, clock = now(), time.monotonic()
    with TemporaryDirectory(prefix="exp8803-private-", dir="/var/tmp") as directory:
        private = Path(directory)
        m.progress("preconditions_before")
        try:
            pre = preflight(private)
        except (OSError, ImportError, ValueError, KeyError) as error:
            pre = dict(
                passed=False,
                diagnostics=[
                    dict(
                        check="missing_file" if isinstance(error, OSError) else "rejected_producer",
                        upstream=m.TASK,
                        artifact_field="preconditions",
                        expected=True,
                        observed=str(error),
                        path=str(private),
                        sha256=None,
                    )
                ],
            )
        m.progress("preconditions_after")
        ready = pre["passed"]
        plan = (
            m.freeze()
            if ready
            else dict(split_manifest={}, intervention=[], consensus=[], length_balance_rows=[])
        )
        atomic_json(raw / "plan.json", plan)
        cohort_references = {}
        for cohort in ["intervention", "consensus"]:
            for projection in ["inputs", "labels"]:
                path = raw / f"{cohort}_{projection}.json"
                selected = (
                    [
                        dict(
                            unit_id=canonical_hash(r["unit_id"]),
                            input=r["input"],
                            **({"candidates": r["candidates"]} if cohort == "consensus" else {}),
                        )
                        for r in plan[cohort]
                    ]
                    if projection == "inputs"
                    else [{k: v for k, v in r.items() if k != "input"} for r in plan[cohort]]
                )
                atomic_json(path, dict(rows=selected))
                cohort_references[f"{cohort}_{projection}"] = reference(path)
            m.progress("sealed_" + cohort, len(plan[cohort]), 0)
        atomic_json(
            raw / "inputs.json", dict(rows=[r["input"] for r in m.controls()] if ready else [])
        )
        sealed = now()
        m.progress(
            "cohorts_sealed_before_evaluation", len(plan["split_manifest"].get("worlds", [])), 0
        )
        measurements = []
        if ready:
            measurements.append(
                child(
                    "input_only_controls",
                    [
                        sys.executable,
                        str(m.ROOT / m.CLI),
                        "--decide-input",
                        str(raw / "inputs.json"),
                        "--decisions",
                        str(raw / "decisions.json"),
                    ],
                    raw / "measurement",
                    deadline=60,
                )
            )
        if not (raw / "decisions.json").exists():
            atomic_json(raw / "decisions.json", dict(rows=[]))
        measured = now()
        repair = m.repair_controls(private) if ready else []
        # Retain private JSON boundary receipts before temporary scratch is removed.
        for r in repair:
            destination = raw / "repair_controls" / Path(r["source"]["path"]).name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(r["source"]["path"], destination)
            r["source"] = reference(destination)
        m.progress("validation_before")
        validation = unqualified() if args.private_control or not ready else validate(raw, private)
        m.progress("validation_after")
        sources = [
            reference(m.ROOT / p)
            for p in m.OWNED
            + m.TESTS
            + m.old.OWNED
            + m.old.TESTS
            + [
                "python/carnot/reporting/primary_publication.py",
                "scripts/experiment_template.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ]
        health = m.ROOT / "results/raw" / m.NAME / "startup/full_python_suite.receipt.json"
        work = dict(
            raw=str(raw),
            run_date=args.date,
            preconditions_checked=pre,
            plan_reference=reference(raw / "plan.json"),
            decisions_reference=reference(raw / "decisions.json"),
            measurement_receipts=measurements,
            repair_controls=repair,
            consensus_controls_passed=consensus_ok(plan) if ready else False,
            cohort_references=cohort_references,
            duration_s=time.monotonic() - clock,
            phase_spans=[
                dict(phase=p, start_utc=a, end_utc=b)
                for p, a, b in [
                    ("preconditions_and_seal", started, sealed),
                    ("tiny_controls", sealed, measured),
                    ("validation", measured, now()),
                ]
            ],
            code_config_hashes=sources,
            source_artifact_hashes=(
                [reference(m.HISTORY), reference(m.ROOT / m.ARCHIVE)]
                + pre["authentication"]["references"]
                + pre["current_authority"]["references"]
            )
            if ready
            else [],
            raw_shard_hashes=[
                reference(raw / n) for n in ["plan.json", "inputs.json", "decisions.json"]
            ]
            + [r["source"] for r in repair]
            + list(cohort_references.values()),
            original_test_reference=reference(m.ROOT / m.old.TESTS[0]),
            repository_health_reference=reference(health) if health.is_file() else None,
        )
        atomic_json(raw / "work.json", work)
        atomic_json(raw / "validation.json", validation)
        value = build(work, validation)
        atomic_json(
            raw / "independent_seal.json",
            {k: value[k] for k in ["primitive_reference", "validation_reference"]},
        )
        m.progress("terminal_publication_before")
        receipt = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
        atomic_json(raw / "publication_receipt.json", receipt)
        m.progress("published_" + value["verdict_class"], len(value["rows"]), 0)
    return 0

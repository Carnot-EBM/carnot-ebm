"""REQ-REPORT-8932: freeze, validate and atomically publish the retrospective audit."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import pilot_label_audit_8932 as a
from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import graph_source_methods_8931 as methods
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured, verify_measured
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]


def freeze(root: Path, raw: Path) -> Json:
    """Snapshot authority and source bytes so later queue changes do not rewrite this audit."""
    started = time.monotonic()
    a.progress("before_authentication", 0, 50)
    inputs = raw / "inputs"
    for name in [
        *methods.INPUT_PINS,
        methods.branch.DESIGN,
        methods.branch.ACTIVE,
        methods.branch.STAGED,
        a.NOTE,
    ]:
        source = root / name
        if source.is_file():
            target = inputs / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    evidence = a.authenticate(inputs)
    result = a.reduce(evidence["pilot"], evidence["runtime"]) if not evidence["failures"] else {}
    sources = []
    for name in [
        *a.OWNED,
        a.TEST,
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        "research-references.md",
        "openspec/capabilities/research-reporting/spec.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "python/carnot/reporting/graph_source_methods_8931.py",
        "python/carnot/reporting/graph_source_8931.py",
        "python/carnot/reporting/v764_branch_contract.py",
        "python/carnot/reporting/v756_branch_qualification.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "ops/e2e-test-plan.md",
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((a.ROOT / name).read_bytes())
        sources.append(
            dict(
                methods.reference(target),
                executed_source=name in a.OWNED,
                source_path=str(a.ROOT / name),
            )
        )
    a.progress("after_authentication_and_reduction", result.get("completed_count", 0), 0)
    return dict(
        raw=str(raw.absolute()),
        evidence=evidence,
        result=result,
        sources=sources,
        input_references=[methods.reference(p) for p in sorted(inputs.rglob("*")) if p.is_file()],
        started=started,
        ended=time.monotonic(),
        repository_health={},
        phase_spans=[dict(phase="authenticate_and_reduce", duration_s=time.monotonic() - started)],
        template=ExperimentTemplate(
            8932, "Pilot label taxonomy audit", f"results/{a.NAME}.json"
        ).build_result({}, status="success"),
    )


def validation_plan(private: Path) -> list[CommandSpec]:
    """Reuse bounded historical checks while measuring only the three new source files."""
    commands = pilot.validation_plan(private)
    substitutions = dict(zip(pilot.OWNED + [pilot.TEST], a.OWNED + [a.TEST], strict=True))
    config = private / "coverage.ini"
    text = config.read_text()
    for before, after in substitutions.items():
        text = text.replace(before, after)
    config.write_text(text)
    return [replace(c, argv=tuple(substitutions.get(arg, arg) for arg in c.argv)) for c in commands]


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Readiness proves label custody only; fixtures and failed controls cannot qualify it."""
    evidence, result = work["evidence"], work["result"]
    owned = (
        bool(validation["receipts"])
        and {r["name"] for r in validation["receipts"]} == methods.CHECKS
        and all(
            r["passed"] and r["exit_code"] == 0 and not r["timed_out"]
            for r in validation["receipts"]
        )
        and verify_measured(validation["coverage"], a.OWNED)
        and not validation.get("terminal_rejected")
    )
    agreement = (
        bool(result) and not result["label_disagreements"] and result["completed_count"] == 50
    )
    ready = int(owned and agreement and not evidence["failures"] and not fixture)
    cls = (
        "disqualified"
        if not fixture and not owned
        else "blocked"
        if evidence["failures"] or not agreement
        else "null"
        if fixture
        else "circular_positive"
    )
    value = dict(work["template"])
    value.update(
        experiment_id=8932,
        task_id=a.TASK,
        milestone="2026.11.764",
        run_date="20261108",
        honest_verdict="complete_"
        + (
            "disqualified_owned_validation"
            if cls == "disqualified"
            else "blocked_pilot_provenance"
            if cls == "blocked"
            else "null_fixture_audit"
            if fixture
            else "pilot_labels_authenticated"
        ),
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if cls == "blocked" else "success",
        inference_substrate="blocked_no_run"
        if evidence["failures"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="blocked_no_run" if evidence["failures"] else "no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="constructed_graph_final_answer_development_and_rate_planning_only",
        independent_generalization_score=0,
        random_seed=764032,
        duration_s=work["ended"] - work["started"],
        phase_spans=work["phase_spans"],
        preconditions_checked=dict(
            authority_mode=evidence["authority_mode"],
            authenticated_inputs=not evidence["failures"],
            arc_suspended=True,
            generator_weights_frozen=True,
        ),
        source_artifact_hashes=work["input_references"] + evidence["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            methods.reference(Path(work["raw"]) / name) for name in ["work.json", "validation.json"]
        ],
        cited_upstream_artifacts=[
            dict(r, scope="opaque_original_producer_no_current_compute")
            for r in evidence["references"]
            if Path(r["path"]).name == Path(methods.PILOT).name
        ],
        gate_check_summary=evidence["failures"]
        + (
            [
                dict(
                    check="exact_label_agreement",
                    upstream=pilot.TASK,
                    artifact_field="label_disagreements_and_completed_count",
                    expected="50 completed and no disagreements",
                    observed="rejected_producer",
                    path=str(Path(work["raw"]) / "work.json"),
                    hash=methods.reference(Path(work["raw"]) / "work.json")["sha256"],
                )
            ]
            if result and not agreement
            else []
        ),
        acceptance_gates=dict(
            owned_checks=owned,
            authenticated_inputs=not evidence["failures"],
            exact_label_agreement=agreement,
        ),
        required_checks_passed=owned,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        pilot_labels_ready_score=ready,
        fixture_only=fixture,
        flagged_adversarial=bool(validation.get("terminal_rejected")),
        source_receipts=dict(
            original_counts={
                key: evidence["pilot"].get(key)
                for key in [
                    "intended_count",
                    "completed_count",
                    "failed_count",
                    "censored_count",
                    "excluded_count",
                    "independent_count",
                ]
            },
            custody_scope="authenticated original producer only; no current model invocation",
            fover_prefix_labels=dict(
                audit_note=a.NOTE,
                reference=methods.reference(Path(work["raw"]) / "inputs" / a.NOTE),
                independent_errors=0,
            ),
        ),
        calibration_evaluation_exclusion=dict(
            milestone="2026.11.764",
            all_pilot_graphs_excluded=True,
            unit_ids=[r["unit_id"] for r in result.get("rows", [])],
            graph_seeds=[r["seed"] for r in result.get("rows", [])],
            graph_hashes=[
                canonical_hash(
                    dict(nodes=r["original_graph"]["nodes"], edges=r["original_graph"]["edges"])
                )
                for r in result.get("rows", [])
            ],
        ),
        repository_health=work["repository_health"],
        **result,
    )
    if not result:
        value.update(
            rows=[],
            error_type_rows=[],
            label_disagreements=[],
            semantic_error_rate=a.rate(0, 0),
            intended_count=50,
            completed_count=0,
            failed_count=0,
            censored_count=50,
            excluded_count=0,
            independent_count=0,
        )
    value["field_principles"] = {
        key: "Retrospective graph-label audit only; no detector benefit, localization or natural-language generalization."
        for key in value
    }
    value["field_principles"].update(
        pilot_labels_ready_score="Fifty authenticated completed answers, exact independent label agreement and passing owned controls only.",
        semantic_error_rate="Wrong-distance answers divided by valid in-graph answers; formatting and absent-node counts stay visible separately.",
        model_invocation_counts="Zero current model calls; all historical compute stays in opaque producer references.",
        calibration_evaluation_exclusion="Every pilot graph is development-only and excluded from all V764 calibration and evaluation.",
        source_receipts="FoVer prefix labels are invalid evidence; note citation adds zero independent samples.",
    )
    value["reproducibility_checksum"] = pilot.checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute every claim from sealed inputs so a changed, rehashed headline still fails."""
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if methods.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_text())
        evidence = a.authenticate(Path(work["raw"]) / "inputs")
        if evidence != work["evidence"]:
            raise ValueError("authentication_drift")
        result = (
            a.reduce(evidence["pilot"], evidence["runtime"]) if not evidence["failures"] else {}
        )
        if result != work["result"]:
            raise ValueError("reduction_drift")
        validation = json.loads((Path(work["raw"]) / "validation.json").read_text())
        passed = value == build(work, validation, value["fixture_only"])
        return dict(
            passed=passed,
            replay_passed=passed,
            reason="replay_passed" if passed else "semantic_drift",
        )
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False, reason="missing_or_changed_evidence")


def main(argv: list[str] | None = None) -> int:
    """Private CLI cases skip nested qualification and never become research samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--input-root", type=Path, default=a.ROOT)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261108":
        parser.error("this producer requires --date 20261108")
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or a.ROOT / "results" / (a.NAME + ".json")
    raw = output.parent / "raw" / a.NAME / "invocations" / str(time.time_ns())
    work = freeze(args.input_root, raw)
    validation: Json = dict(receipts=[], coverage={})
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot-8932-validation-"))
        a.progress("before_owned_checks")
        validation["receipts"] = run_commands(
            a.ROOT, validation_plan(private), log_dir=raw / "validation", heartbeat_s=15
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], a.OWNED, raw / "measured")
        a.progress("after_owned_checks", len(validation["receipts"]), 0)
        a.progress("before_repository_health")
        work["repository_health"] = dict(
            receipts=run_commands(
                a.ROOT,
                [
                    CommandSpec(
                        "full_python_suite",
                        (str(a.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
                        "repository_health",
                        300,
                    )
                ],
                log_dir=raw / "repository_health",
                heartbeat_s=15,
            ),
            affects_owned_readiness=False,
            repeated=False,
        )
        a.progress("after_repository_health")
    work["ended"] = time.monotonic()
    work["phase_spans"].append(
        dict(
            phase="validation_and_health",
            duration_s=work["ended"] - work["started"] - work["phase_spans"][0]["duration_s"],
        )
    )
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    value = build(work, validation, fixture)
    terminal: list[Json] = []

    def validate(candidate: Path) -> Json:
        report = replay(candidate)
        if not fixture:
            py = str(a.ROOT / ".venv/bin/python")
            commands = [
                CommandSpec(
                    "cold_replay_cli",
                    (
                        py,
                        "-u",
                        "-c",
                        "import os,subprocess,sys,tempfile; env=dict(os.environ); env.pop('PYTHONPATH',None); raise SystemExit(subprocess.run(sys.argv[1:],cwd=tempfile.gettempdir(),env=env,timeout=60).returncode)",
                        py,
                        "-u",
                        str(a.ROOT / a.CLI),
                        "--cold-replay",
                        str(candidate),
                    ),
                    "terminal",
                    75,
                ),
                CommandSpec(
                    "adversarial",
                    (py, "scripts/adversarial_verify.py", "--json", str(candidate)),
                    "terminal",
                    60,
                ),
                CommandSpec(
                    "strict_rows",
                    (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
                    "terminal",
                    60,
                ),
            ]
            terminal.extend(
                run_commands(a.ROOT, commands, log_dir=raw / "terminal_validation", heartbeat_s=15)
            )
            report.update(
                passed=report["passed"] and pilot.terminal_passed(terminal), receipts=terminal
            )
        return report

    a.progress("before_terminal_validation")
    try:
        published = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        validation["terminal_rejected"] = True
        validation["receipts"].extend(terminal)
        atomic_json(raw / "validation.json", validation)
        value = build(work, validation, fixture)
        published = publish_primary(output, value, replay)
    a.progress("terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                **published,
                pilot_labels_ready_score=value["pilot_labels_ready_score"],
                honest_verdict=value["honest_verdict"],
            )
        ),
        flush=True,
    )
    return 0

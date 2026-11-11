"""REQ-REPORT-8931: publish CPU collection methods without claiming live answers."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import v764_branch_contract as branch
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured, verify_measured
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
ROOT = pilot.ROOT
NAME = "experiment_8931_v764_graph_source_methods"
TASK = "exp8931-graph-source-methods"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/graph_source_methods_8931.py",
    "python/carnot/reporting/graph_source_8931.py",
    CLI,
]
TEST = "tests/python/test_graph_source_methods_8931.py"
PILOT = "results/experiment_8918_v763_bfs_error_pilot.json"
INPUT_PINS = {
    PILOT: "sha256:b2d62a2d78535ad3bf5ca9aa83c330f2490ed7d40941a7a42b18a489aafd800e",
    pilot.OWNED[0]: "sha256:8f50020b09f44e1c7b895b251456e58bc7424a19ec8feabe90ca26b64d27bd9d",
    pilot.OWNED[1]: "sha256:94c595196cce338bd3e1122b92872789784e1a74c8fe167e563238e968c13d3f",
    pilot.TEST: "sha256:da79f22262424751d4f92b450446bc533bf6158276261522fac6f8bce9f80bc0",
}
CHECKS = {
    "owned_unit_child",
    "coverage_combine",
    "coverage_json",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "e2e018",
    "e2e021",
}
terminal_passed = pilot.terminal_passed


def reference(path: Path) -> Json:
    """Absent external bytes need a named receipt, not an invented digest."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def authenticate(root: Path) -> Json:
    """Check original producer identity and task authority before constructing methods."""
    failures, references = [], []
    for name, expected in INPUT_PINS.items():
        path = root / name
        ref = reference(path)
        references.append(ref)
        observed = (
            "missing_file"
            if not path.is_file()
            else "rejected_producer"
            if ref["sha256"] != expected
            else None
        )
        if observed:
            failures.append(
                dict(
                    check="original_bytes",
                    upstream="exp8918-bfs-error-pilot",
                    artifact_field="sha256",
                    expected=expected,
                    observed=observed,
                    path=ref["path"],
                    hash=ref["sha256"],
                )
            )
    path = root / PILOT
    if path.is_file():
        try:
            value = json.loads(path.read_text())
            required = dict(
                experiment_id=8918,
                task_id=pilot.TASK,
                run_date="20261108",
                pilot_ready_score=1,
                pilot_error_gate_score=1,
                qwen_runtime_ready_score=1,
                n_errors=43,
                completed_count=50,
            )
            for field, expected_value in required.items():
                actual = value.get(field)
                if actual != expected_value or type(actual) is not type(expected_value):
                    observed = (
                        "missing_field"
                        if field not in value
                        else "observed_zero"
                        if type(actual) is int and actual == 0
                        else "rejected_producer"
                    )
                    failures.append(
                        dict(
                            check="pilot_identity_and_gate",
                            upstream=pilot.TASK,
                            artifact_field=field,
                            expected=expected_value,
                            observed=observed,
                            path=str(path),
                            hash=sha256_file(path),
                        )
                    )
        except (ValueError, AttributeError):
            failures.append(
                dict(
                    check="pilot_json",
                    upstream=pilot.TASK,
                    artifact_field="object",
                    expected="original producer JSON",
                    observed="rejected_producer",
                    path=str(path),
                    hash=sha256_file(path),
                )
            )
    contract = branch.authority(root)
    failures.extend(dict(row, hash=row.get("sha256")) for row in contract["failures"])
    return dict(
        failures=failures,
        references=references,
        authority_ready=contract["ready"],
        authority_mode=contract["comparison_mode"],
    )


def freeze(root: Path, raw: Path) -> Json:
    """Retain exact source bytes and all graph identities before validation or future inference."""
    source.progress("before_input_authentication")
    started = time.monotonic_ns()
    authenticate(root)
    raw.mkdir(parents=True, exist_ok=True)
    inputs = raw / "inputs"
    for name in [*INPUT_PINS, branch.DESIGN, branch.ACTIVE, branch.STAGED]:
        path = root / name
        if path.is_file():
            target = inputs / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    authentication = authenticate(inputs)
    source.progress("after_input_authentication", len(INPUT_PINS), len(authentication["failures"]))
    sources = []
    for name in [
        *OWNED,
        TEST,
        "AGENTS.md",
        "CLAUDE.md",
        "CODEX.md",
        "research-program.md",
        "openspec/capabilities/research-reporting/spec.md",
        branch.DESIGN,
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "python/carnot/inference/qwen_sufficiency_7920.py",
        "python/carnot/inference/sota_models.py",
        "python/carnot/reporting/v764_branch_contract.py",
        "python/carnot/reporting/v756_branch_qualification.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "ops/e2e-test-plan.md",
        "docs/research-notes/bfs-labeled-error-source-plan-2026-11-07.md",
    ]:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        sources.append(
            dict(reference(target), source_path=str(ROOT / name), executed_source=name in OWNED)
        )
    source.progress("before_protocol_freeze", 0, 370)
    protocol = source.graph_protocol(pilot.graph_manifest())
    atomic_json(raw / "protocol.json", protocol)
    source.progress("after_protocol_freeze", 370, 0)
    work = dict(
        raw=str(raw.absolute()),
        started_ns=started,
        ended_ns=time.monotonic_ns(),
        authentication=authentication,
        sources=sources,
        protocol_reference=reference(raw / "protocol.json"),
        input_references=[reference(p) for p in sorted(inputs.rglob("*")) if p.is_file()],
        template=ExperimentTemplate(
            8931, "Frozen graph source methods", f"results/{NAME}.json"
        ).build_result({}, status="success"),
        phase_spans=[
            dict(phase="authenticate_and_freeze", duration_s=(time.monotonic_ns() - started) / 1e9)
        ],
        repository_health={},
    )
    atomic_json(raw / "work.json", work)
    return work


def validation_plan(private: Path) -> list[CommandSpec]:
    """Keep the pilot's bounded checks and change only their explicitly owned paths."""
    commands = pilot.validation_plan(private)
    substitutions = dict(zip(pilot.OWNED + [pilot.TEST], OWNED + [TEST]))
    config = private / "coverage.ini"
    text = config.read_text()
    for before, after in substitutions.items():
        text = text.replace(before, after)
    config.write_text(text)
    return [
        replace(
            c,
            argv=tuple(substitutions.get(a, a) for a in c.argv)
            + ((pilot.TEST,) if c.name == "owned_unit_child" else ()),
        )
        for c in commands
    ]


def qualified(validation: Json) -> bool:
    """Passing fixture booleans cannot substitute for measured unit and child statements."""
    receipts = validation["receipts"]
    return (
        bool(receipts)
        and {r["name"] for r in receipts} == CHECKS
        and all(
            r["passed"] and r.get("exit_code") == 0 and not r.get("timed_out") for r in receipts
        )
        and verify_measured(validation["coverage"], OWNED)
    )


def build(work: Json, validation: Json, *, fixture: bool = False) -> Json:
    """Method readiness is administrative; no fixture or planned graph is a research sample."""
    protocol = json.loads(Path(work["protocol_reference"]["path"]).read_text())
    terminal_rejected = validation.get("terminal_rejected", False)
    owned = qualified(validation) and not terminal_rejected
    failures = work["authentication"]["failures"]
    cls = "disqualified" if not fixture and not owned else "blocked" if failures else "null"
    ready = int(owned and not failures and not fixture)
    raw = Path(work["raw"])
    value = dict(work["template"])
    value.update(
        experiment_id=8931,
        task_id=TASK,
        milestone="2026.11.764",
        run_date="20261108",
        honest_verdict="complete_disqualified_terminal_validation"
        if terminal_rejected
        else "complete_disqualified_owned_validation"
        if cls == "disqualified"
        else "complete_blocked_external_inputs"
        if cls == "blocked"
        else "complete_null_fixture_methods"
        if fixture
        else "complete_null_source_methods_ready",
        verdict_class=cls,
        status="error" if cls == "disqualified" else "blocked" if cls == "blocked" else "success",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="constructed_graph_final_answers_only_methods",
        independent_generalization_score=0,
        duration_s=(work["ended_ns"] - work["started_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        preconditions_checked=dict(
            work["authentication"],
            arc_suspended=True,
            generator_weights_frozen=True,
            gpu_tasks_authorized_here=False,
        ),
        source_artifact_hashes=work["input_references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[reference(raw / "work.json"), work["protocol_reference"]],
        cited_upstream_artifacts=[
            dict(r, scope="opaque_original_producer_reference_no_current_compute")
            for r in work["authentication"]["references"]
            if Path(r["path"]).name == Path(PILOT).name
        ],
        rows=[],
        intended_count=0,
        completed_count=0,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        scheduled_future_graph_count=370,
        scheduled_future_answer_count=420,
        gate_check_summary=failures,
        acceptance_gates=dict(
            owned_checks=owned, authenticated_inputs=not failures, protocol_frozen=True
        ),
        required_checks_passed=owned,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        source_methods_ready_score=ready,
        fixture_only=fixture,
        flagged_adversarial=terminal_rejected,
        protocol_hash=canonical_hash(protocol),
        **protocol,
        label_controls=dict(
            parser="historical strict single uppercase two-letter label with optional terminal . or !",
            categories=["malformed", "out_of_graph", "valid_wrong_distance", "correct"],
            oracle="BFS shortest distance; independent adjacency-matrix small-graph test",
            runtime_failures="censored; no reasoning-error label",
            step_localization=False,
        ),
        resume_controls=dict(
            completed_unit_reuse=True,
            first_response_only=True,
            binding=["model", "settings", "graph", "query", "role", "shard", "feature_schema"],
            raw_before_labels=True,
            interrupted_intent="censored_without_retry",
            outcome_selection=False,
        ),
        development_controls=dict(fixture_tests=TEST, counted_as_research=False),
        future_recipes=dict(
            control=dict(task_id="exp8934-depth-control", shard="control", required_answers=100),
            collection=[
                dict(task_id=f"exp893{5 + i}-graph-corpus-{shard.lower()}", shard=shard, graphs=160)
                for i, shard in enumerate(("A", "B"))
            ],
            runner="carnot.reporting.graph_source_8931.collect",
            model=source.SETTINGS["model"],
            quantization="Q4_K_M",
            generator_weights_frozen=True,
        ),
        repository_health=work["repository_health"],
    )
    validation_path = raw / "validation.json"
    if validation_path.is_file():
        value["raw_shard_hashes"].append(reference(validation_path))
    value["field_principles"] = {
        key: "This field describes a frozen CPU collection contract, not live source evidence or learned benefit."
        for key in value
    }
    value["field_principles"].update(
        rows="No current model answers were scheduled; graph manifests and checks are administrative receipts outside comparison rows.",
        source_methods_ready_score="Executable tested methods only; future paired depth acceptance and at least 200 independent errors remain unmeasured.",
        graph_manifest="370 fixed fresh graph identities; paired queries on one graph never increase independent sample size.",
        feature_schema="Only two finite response-derived confidence features; oracle truth and nuisance controls remain outside the learned view.",
        model_invocation_counts="Zero current model activity; pinned pilot compute remains an opaque producer reference.",
        repository_health="One bounded broad-suite health run is independent of owned method qualification; its failures remain visible.",
        owned_statement_coverage="Measured unit and real CLI child statements, with no exclusions; percentages alone cannot qualify readiness.",
    )
    value["schema"] = sorted(set(value) | {"reproducibility_checksum"})
    value["reproducibility_checksum"] = pilot.checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild all claims from sealed operands; rehashing a changed aggregate cannot repair it."""
    try:
        value = json.loads(path.read_text())
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_text())
        if work["authentication"] != authenticate(Path(work["raw"]) / "inputs"):
            raise ValueError("input_authentication_drift")
        protocol = json.loads(Path(work["protocol_reference"]["path"]).read_text())
        if protocol != source.graph_protocol(pilot.graph_manifest()):
            raise ValueError("protocol_drift")
        validation_path = Path(work["raw"]) / "validation.json"
        validation = (
            json.loads(validation_path.read_text())
            if validation_path.is_file()
            else dict(
                receipts=value["validation_receipts"], coverage=value["owned_statement_coverage"]
            )
        )
        expected = build(work, validation, fixture=value["fixture_only"])
        passed = expected == value and value["reproducibility_checksum"] == pilot.checksum(value)
        return dict(
            passed=passed,
            replay_passed=passed,
            reason="replay_passed" if passed else "semantic_drift",
        )
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False, reason="missing_or_changed_evidence")


def main(argv: list[str] | None = None) -> int:
    """Publish only checked methods; the private CLI never launches a model or nested qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture-e2e", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay is not None:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261108":
        parser.error("this producer requires --date 20261108")
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or ROOT / "results" / (NAME + ".json")
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    source.progress("invocation_start")
    work = freeze(ROOT, raw)
    validation: Json = dict(receipts=[], coverage={})
    if not fixture:
        private = Path(tempfile.mkdtemp(prefix="carnot-8931-validation-"))
        source.progress("before_owned_checks")
        validation["receipts"] = run_commands(
            ROOT, validation_plan(private), log_dir=raw / "validation", heartbeat_s=15
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], OWNED, raw / "measured")
        source.progress("after_owned_checks")
        source.progress("before_repository_health")
        health = run_commands(
            ROOT,
            [
                CommandSpec(
                    "full_python_suite",
                    (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
                    "repository_health",
                    300,
                )
            ],
            log_dir=raw / "repository_health",
            heartbeat_s=15,
        )
        work["repository_health"] = dict(
            receipts=health, affects_owned_readiness=False, repeated=False
        )
        source.progress("after_repository_health")
    work["ended_ns"] = time.monotonic_ns()
    work["phase_spans"].append(
        dict(
            phase="validation_and_health",
            duration_s=(work["ended_ns"] - work["started_ns"]) / 1e9
            - work["phase_spans"][0]["duration_s"],
        )
    )
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    value = build(work, validation, fixture=fixture)
    terminal_receipts: list[Json] = []

    def validate(candidate: Path) -> Json:
        report = replay(candidate)
        if not fixture:
            py = str(ROOT / ".venv/bin/python")
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
                        str(ROOT / CLI),
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
            terminal_receipts.extend(
                run_commands(ROOT, commands, log_dir=raw / "terminal_validation", heartbeat_s=15)
            )
            report.update(
                passed=report["passed"] and terminal_passed(terminal_receipts),
                receipts=terminal_receipts,
            )
        return report

    source.progress("before_terminal_validation")
    try:
        published = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        validation["receipts"].extend(terminal_receipts)
        validation["terminal_rejected"] = True
        atomic_json(raw / "validation.json", validation)
        value = build(work, validation, fixture=fixture)
        value.update(
            required_checks_passed=False,
            source_methods_ready_score=0,
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_terminal_validation",
            flagged_adversarial=True,
            status="error",
        )
        value["acceptance_gates"]["owned_checks"] = False
        value["reproducibility_checksum"] = pilot.checksum(value)
        published = publish_primary(output, value, replay)
    source.progress("terminal_published", 1, 0)
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                source_methods_ready_score=value["source_methods_ready_score"],
            )
        ),
        flush=True,
    )
    return 0

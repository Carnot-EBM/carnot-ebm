"""Freeze executable calibration methods; fixture success is never live benefit.

REQ-REPORT-8937; SCENARIO-REPORT-8937-CONTRACT.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from carnot.autoresearch import graph_calibration_benchmark as b
from carnot.reporting import bfs_error_pilot_8918 as checks
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
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8937_v764_calibration_methods", "exp8937-calibration-methods"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/autoresearch/graph_calibration_benchmark.py",
    "python/carnot/reporting/calibration_methods_8937.py",
    CLI,
]
TEST = "tests/python/test_calibration_methods_8937.py"
INPUT = "results/experiment_8931_v764_graph_source_methods.json"
PIN = "sha256:810b976a4d7883a3a600c9501982eb882d1feb658b0a54cb86afea00320f5b57"
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


def reference(path: Path) -> Json:
    """An absent input has no digest; its path still identifies the blocked gate."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def authenticate(root: Path) -> Json:
    """Authenticate exact source methods and branch authority before numerical work."""
    path, failures, refs = root / INPUT, [], []

    def fail(check: str, field: str, expected: Any, observed: Any) -> None:
        failures.append(
            dict(
                check=check,
                upstream="exp8931-graph-source-methods",
                artifact_field=field,
                expected=expected,
                observed=observed,
                path=str(path),
                hash=reference(path)["sha256"],
            )
        )

    if not path.is_file():
        fail("missing_file", "producer_bytes", PIN, None)
    else:
        refs.append(reference(path))
        if sha256_file(path) != PIN:
            fail("rejected_producer", "producer_bytes", PIN, sha256_file(path))
        try:
            value = json.loads(path.read_text())
            for field, expected in dict(
                experiment_id=8931,
                task_id="exp8931-graph-source-methods",
                milestone="2026.11.764",
                run_date="20261108",
                source_methods_ready_score=1,
                required_checks_passed=True,
            ).items():
                actual = value.get(field)
                if actual != expected or type(actual) is not type(expected):
                    fail(
                        "missing_field"
                        if field not in value
                        else "observed_zero"
                        if type(actual) is int and actual == 0
                        else "rejected_producer",
                        field,
                        expected,
                        actual,
                    )
            for ref in (
                value["source_artifact_hashes"]
                + value["code_config_hashes"]
                + value["raw_shard_hashes"]
            ):
                operand = Path(ref["path"])
                observed = reference(operand)["sha256"]
                if observed != ref["sha256"]:
                    failures.append(
                        dict(
                            check="missing_file" if observed is None else "rejected_producer",
                            upstream="exp8931-original-operand",
                            artifact_field="sha256",
                            expected=ref["sha256"],
                            observed=observed,
                            path=str(operand),
                            hash=observed,
                        )
                    )
                refs.append(reference(operand))
        except (ValueError, AttributeError, KeyError, TypeError):
            fail("rejected_producer", "json_object", "original source methods", "malformed")
    authority = branch.authority(root)
    failures.extend(dict(r, hash=r.get("sha256")) for r in authority["failures"])
    return dict(
        ready=not failures,
        failures=failures,
        references=refs,
        authority_mode=authority.get("comparison_mode"),
    )


def manifest() -> Json:
    """All choices are frozen before any live train/calibration/test corpus is opened."""
    return dict(
        feature_allowlist=list(b.FEATURES),
        training_equations=dict(
            probability="p(error|x)=sigmoid(E(x)/T)",
            loss="mean_i[softplus(E_i)-y_i*E_i] = (n0/n)*mean_0 softplus(E) + (n1/n)*mean_1 softplus(-E)",
            update="theta[t+1]=theta[t]-0.01*grad(population_binary_log_loss)",
            logistic="sigmoid(w dot standardized_x + bias); train only",
            scaling="train mean and population standard deviation; constant scale=1",
        ),
        hyperparameters=dict(
            input_dim=2,
            hidden_dims=[4],
            activation="silu",
            seed=b.SEED,
            steps=b.STEPS,
            learning_rate=b.RATE,
            full_batch=True,
            class_balancing=False,
            energy_initialization="native Gibbs Xavier hidden, zero output",
            logistic_initialization=[0, 0, 0],
            temperature_grid=list(b.GRID),
            temperature_selection="calibration Brier, ascending temperature ties",
        ),
        decision_rules=dict(
            accept="p(error)<=0.1",
            reject="p(error)>=0.9",
            escalate="0.1<p(error)<0.9",
            certified_risk_guarantee=False,
            diagnostic_coverage=0.5,
            confidence="1-p(error); accept highest confidence correct",
            tie_breaker="SHA256('8937/coverage/'+unit_id), then occurrence order",
            bootstrap=dict(
                draws=10000,
                seed=b.SEED,
                alpha=0.05 / 3,
                unit="graph",
                paired=True,
                rerank_each_draw=True,
            ),
            acceptance=dict(
                brier_vs_prevalence="lower>0",
                brier_vs_logistic="lower>0",
                accepted_error_vs_logistic="upper<=0",
            ),
        ),
        nuisance_view=dict(
            features=["token_count", "format_valid"],
            format_valid="strict single two-uppercase-letter answer with optional terminal . or !; no graph membership check",
            comparator="train-only standardized logistic, zero initialization, same SGD",
        ),
        model_state_schema=dict(
            energy=[[4, 2], [4], [4], []],
            logistic=[3],
            nuisance=[3],
            scaling=["mean[2]", "scale[2]"],
            temperature="one grid scalar",
            prevalence="train mean(label)",
            initial_states="all three heads retained",
        ),
        scoring_imports="Gibbs and numerical libraries only; no graph, oracle, PCIB or FoVer loader",
        future_roles=dict(train=160, calibration=80, test=80, test_fitting=False),
    )


def development() -> Json:
    """Planted response confidence tests the estimator, not real detector quality."""
    rng = np.random.default_rng(b.SEED)
    fixture: Json = {}
    for role, n in [("train", 80), ("calibration", 40), ("diagnostic", 40)]:
        y = [int(i % 4 != 0) for i in range(n)]
        fixture[role] = dict(
            labels=y,
            features=[
                dict(
                    mean_token_nll=float(0.2 + 1.8 * v + rng.uniform(0, 0.02)),
                    first_answer_top2_margin=float(2 - 1.8 * v + rng.uniform(0, 0.02)),
                )
                for v in y
            ],
            nuisance=[dict(token_count=1 + i % 3, format_valid=i % 5 != 0) for i in range(n)],
        )
    train, cal, diagnostic = (fixture[k] for k in ("train", "calibration", "diagnostic"))
    fit = b.fit(
        train["features"], train["labels"], train["nuisance"], cal["features"], cal["labels"]
    )
    p = b.score(fit, diagnostic["features"], diagnostic["nuisance"])
    ids = [f"fixture-{i:02d}" for i in range(40)]
    rows = b.comparison_rows(p, diagnostic["labels"], ids, ids)
    shuffled_labels = {k: rng.permutation(v["labels"]).tolist() for k, v in fixture.items()}
    shuffled = b.fit(
        train["features"],
        shuffled_labels["train"],
        train["nuisance"],
        cal["features"],
        shuffled_labels["calibration"],
    )
    sp = b.score(shuffled, diagnostic["features"], diagnostic["nuisance"])
    truth = np.asarray(diagnostic["labels"])
    base = float(np.mean((fit["prevalence"] - truth) ** 2))
    shuffled_loss = float(np.mean((np.asarray(sp["energy"]) - shuffled_labels["diagnostic"]) ** 2))
    constant_state = dict(fit, energy=fit["initial_states"]["energy"])
    constant = np.asarray(
        b.score(constant_state, diagnostic["features"], diagnostic["nuisance"])["energy"]
    )
    constant_brier = float(np.mean((constant - truth) ** 2))
    controls = [
        dict(
            control="planted_signal",
            passed=all(
                np.mean((np.asarray(p[a]) - truth) ** 2) < base for a in ("energy", "logistic")
            ),
        ),
        dict(control="shuffled_labels", passed=shuffled_loss >= base - 0.03, brier=shuffled_loss),
        dict(
            control="constant_score",
            passed=bool(np.ptp(constant) == 0 and constant_brier >= base),
            score_range=float(np.ptp(constant)),
            brier=constant_brier,
            prevalence_brier=base,
        ),
    ]
    operations: list[tuple[str, Callable[[], Any]]] = [
        ("missing_feature", lambda: b.features([{}])),
        (
            "degenerate_class",
            lambda: b.fit(
                train["features"], [1] * 80, train["nuisance"], cal["features"], cal["labels"]
            ),
        ),
    ]
    for name, operation in operations:
        try:
            operation()
            passed = False
        except ValueError:
            passed = True
        controls.append(dict(control=name, passed=passed))
    rejected = []
    for key in ("label", "distance", "path"):
        leaked = dict(train["features"][0], **{key: 1})
        envelope = dict(features=leaked, hash=canonical_hash(leaked))
        try:
            b.features([envelope["features"]])
            rejected.append(False)
        except ValueError:
            rejected.append(True)
    controls.append(dict(control="rehashed_leakage", passed=all(rejected)))
    return dict(
        fixture=fixture,
        fit=fit,
        shuffled_fit=shuffled,
        shuffled_labels=shuffled_labels,
        rows=rows,
        control_rows=controls,
        fixture_diagnostics=b.summarize(rows),
        numerical_training_calls=1200,
    )


def freeze(root: Path, raw: Path) -> Json:
    """Retain original producer and executed bytes before any expensive work begins."""
    started = time.monotonic_ns()
    b.progress("before_authentication")
    authentication = authenticate(root)
    raw.mkdir(parents=True, exist_ok=True)
    refs = []
    names = [
        *OWNED,
        TEST,
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        "openspec/capabilities/research-reporting/spec.md",
        branch.DESIGN,
        branch.ACTIVE,
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/models/gibbs/__init__.py",
        "python/carnot/training/nce.py",
        "python/carnot/autoresearch/calibrated_decision_benchmark.py",
        "ops/e2e-test-plan.md",
        "docs/research-notes/jepa-two-model-consensus-plan-2026-11-05.md",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
    ]
    for name in names:
        target = raw / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        refs.append(
            dict(reference(target), source_path=str(ROOT / name), executed_source=name in OWNED)
        )
    retained = []
    for i, ref in enumerate(authentication["references"]):
        path = Path(ref["path"])
        if path.is_file():
            target = raw / "inputs" / str(i) / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            retained.append(dict(reference(target), original_path=str(path)))
    b.progress("after_authentication", len(retained), len(authentication["failures"]))
    work = dict(
        raw=str(raw.absolute()),
        started_ns=started,
        ended_ns=time.monotonic_ns(),
        authentication=authentication,
        sources=refs,
        retained_inputs=retained,
        repository_health={},
        phase_spans=[],
        template=ExperimentTemplate(
            8937, "Frozen response calibration methods", f"results/{NAME}.json", seed=b.SEED
        ).build_result({}, status="success"),
    )
    atomic_json(raw / "work.json", work)
    return work


def validation_plan(private: Path) -> list[CommandSpec]:
    """Reuse bounded unit/child and private CLI checks without changing their assertions."""
    commands = checks.validation_plan(private)
    substitutions = dict(zip(checks.OWNED + [checks.TEST], OWNED + [TEST]))
    config = private / "coverage.ini"
    text = config.read_text()
    for before, after in substitutions.items():
        text = text.replace(before, after)
    config.write_text(text)
    return [replace(c, argv=tuple(substitutions.get(a, a) for a in c.argv)) for c in commands]


def qualified(validation: Json) -> bool:
    """A percentage claim cannot replace actual SQLite statement membership."""
    receipts = validation["receipts"]
    return (
        bool(receipts)
        and {r["name"] for r in receipts} == CHECKS
        and all(r["passed"] and r["exit_code"] == 0 and not r["timed_out"] for r in receipts)
        and verify_measured(validation["coverage"], OWNED)
    )


def checksum(value: Json) -> str:
    """Bind all claims while keeping the digest itself outside its own input."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def build(work: Json, result: Json, validation: Json) -> Json:
    """Method readiness requires owned checks, not a scientific win on fixtures."""
    result = deepcopy(result)
    failures = work["authentication"]["failures"]
    valid = qualified(validation)
    controls = bool(result) and all(r["passed"] for r in result["control_rows"])
    ready = int(valid and controls and not failures)
    cls = "blocked" if failures else "null" if ready else "disqualified"
    p, raw = manifest(), Path(work["raw"])
    rows = result.get("rows", [])
    value: Json = dict(work["template"])
    value.update(
        experiment_id=8937,
        task_id=TASK,
        milestone="2026.11.764",
        run_date="20261108",
        honest_verdict=f"complete_{cls}_calibration_methods",
        verdict_class=cls,
        status="blocked" if cls == "blocked" else "error" if cls == "disqualified" else "success",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load" if result else "blocked_no_run",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=result.get("numerical_training_calls", 0),
        verifier_is_oracle=False,
        exposure_scope="synthetic calibration development only; no live graph train/calibration/test rows opened",
        independent_generalization_score=0,
        duration_s=(work["ended_ns"] - work["started_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        preconditions_checked=work["authentication"],
        random_seed=b.SEED,
        source_artifact_hashes=work["retained_inputs"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[reference(raw / "work.json")]
        + [
            reference(raw / n)
            for n in ("development.json", "validation.json")
            if (raw / n).is_file()
        ],
        cited_upstream_artifacts=[
            dict(
                producer="exp8931-graph-source-methods",
                sha256=PIN,
                scope="opaque original producer; no inherited current compute",
            )
        ],
        rows=rows,
        intended_count=len(rows),
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        gate_check_summary=failures,
        acceptance_gates=dict(
            owned_checks=valid,
            owned_controls=controls,
            authenticated_inputs=not failures,
            real_quality_evaluated=False,
            future_test_benefit=p["decision_rules"]["acceptance"],
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        calibration_methods_ready_score=ready,
        method_manifest=p,
        **p,
        control_rows=result.get("control_rows", []),
        development_evidence=result,
        fixture_diagnostics=result.get("fixture_diagnostics"),
        fixture_only=True,
        model_state=result.get("fit"),
        flagged_adversarial=False,
        repository_health=work["repository_health"],
        methodology="Frozen Gibbs 2x4 energy and same-feature logistic with train-prevalence and syntax-only nuisance controls; population-weighted full-batch classification.",
        methodology_note="Fixtures verify implementation only; graph labels remain outside the response-only scorer. No natural-language generalization or verifier moat is measured.",
        generator_weights_frozen=True,
        arc_suspended=True,
    )
    value["field_principles"] = {
        k: "Frozen executable methods or current administrative receipt; synthetic fixtures cannot support live quality or independent generalization."
        for k in [*value, "reproducibility_checksum", "field_principles"]
    }
    value["field_principles"].update(
        calibration_methods_ready_score="Correct executable estimator and frozen methods, independent of live data availability or test benefit.",
        numerical_training_calls="1200 direct fixture SGD updates (two fits, three heads, 200 each); validation/replay repetitions are separate validation work, never upstream compute.",
        rows="Per fixture unit and arm Brier operands only; all synthetic, zero independent research units.",
        control_rows="Administrative acceptance controls; planted and shuffled fixtures are not scientific findings.",
        owned_statement_coverage="Measured owned unit/child statements from SQLite, with zero exclusions.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute the methods and fixture outcomes, rejecting rehashed semantic drift."""
    try:
        value = json.loads(path.read_text())
        if value["reproducibility_checksum"] != checksum(value):
            raise ValueError("checksum_drift")
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_text())
        raw = Path(work["raw"])
        validation = (
            json.loads((raw / "validation.json").read_text())
            if (raw / "validation.json").is_file()
            else dict(
                receipts=value["validation_receipts"], coverage=value["owned_statement_coverage"]
            )
        )
        result = development() if not work["authentication"]["failures"] else {}
        expected = build(work, result, validation)
        passed = expected == value
        return dict(
            passed=passed,
            replay_passed=passed,
            reason="replay_passed" if passed else "semantic_drift",
        )
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False, reason="missing_or_changed_evidence")


def terminal_commands(candidate: Path) -> list[CommandSpec]:
    """Run existing scanners unchanged and a real cold CLI outside this checkout."""
    py = str(ROOT / ".venv/bin/python")
    return [
        CommandSpec(
            "cold_replay_cli",
            (
                py,
                "-u",
                "-c",
                "import os,subprocess,sys,tempfile; env=dict(os.environ); env.pop('PYTHONPATH',None); raise SystemExit(subprocess.run(sys.argv[1:],cwd=tempfile.gettempdir(),env=env,timeout=120).returncode)",
                py,
                "-u",
                str(ROOT / CLI),
                "--cold-replay",
                str(candidate),
            ),
            "terminal",
            150,
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


def main(argv: list[str] | None = None) -> int:
    """Publish only after bounded owned checks; private children cannot self-qualify."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261108"])
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--private-output", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261108":
        parser.error("this producer requires --date 20261108")
    if args.private_output and args.private_output.resolve().is_relative_to(ROOT):
        parser.error("private outputs must remain outside checkout")
    output = args.private_output or ROOT / "results" / (NAME + ".json")
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    work = freeze(ROOT, raw)
    b.progress("before_fixture_benchmark", 0, 6)
    result = development() if not work["authentication"]["failures"] else {}
    b.progress("after_fixture_benchmark", len(result.get("control_rows", [])), 0)
    atomic_json(raw / "development.json", result)
    measured_end = time.monotonic_ns()
    validation: Json = dict(receipts=[], coverage={})
    if not args.private_output:
        private = Path(tempfile.mkdtemp(prefix="carnot8937-validation-"))
        b.progress("before_owned_checks")
        validation["receipts"] = run_commands(
            ROOT, validation_plan(private), log_dir=raw / "validation", heartbeat_s=15
        )
        if (private / ".coverage").is_file():
            validation["coverage"] = measured([private / ".coverage"], OWNED, raw / "measured")
        b.progress("after_owned_checks")
        b.progress("before_full_python_suite")
        work["repository_health"] = dict(
            receipts=run_commands(
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
            ),
            affects_owned_readiness=False,
            repeated=False,
        )
        b.progress("after_full_python_suite")
    work["ended_ns"] = time.monotonic_ns()
    work["phase_spans"] = [
        dict(
            phase="authentication_and_fixtures",
            start_s=0,
            end_s=(measured_end - work["started_ns"]) / 1e9,
        ),
        dict(
            phase="validation",
            start_s=(measured_end - work["started_ns"]) / 1e9,
            end_s=(work["ended_ns"] - work["started_ns"]) / 1e9,
        ),
    ]
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    value = build(work, result, validation)
    receipts: list[Json] = []

    def validate(candidate: Path) -> Json:
        report = replay(candidate)
        if not args.private_output:
            receipts.extend(
                run_commands(
                    ROOT,
                    terminal_commands(candidate),
                    log_dir=raw / "terminal_validation",
                    heartbeat_s=15,
                )
            )
            report.update(
                passed=report["passed"] and all(r["passed"] for r in receipts), receipts=receipts
            )
        return report

    b.progress("before_terminal_validation")
    try:
        published = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        validation["receipts"].extend(receipts)
        atomic_json(raw / "validation.json", validation)
        value = build(work, result, validation)
        published = publish_primary(output, value, replay)
    b.progress("terminal_published", 1, 0)
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                calibration_methods_ready_score=value["calibration_methods_ready_score"],
            )
        ),
        flush=True,
    )
    return 0

"""REQ-REPORT-8818: bind private raw science to checked atomic publication."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

from coverage import Coverage
from coverage.exceptions import CoverageException

from carnot.reporting import evidence_decisive_8818 as m
from carnot.reporting import evidence_evaluation_runner_8779 as shared
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.v746_retained_evidence import reference, load, aggregate
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]


def validate(raw: Path, private: Path) -> Json:
    """Reuse exact test collection and actual unit/child coverage rather than claiming coverage."""
    validator = namespace(shared, "decisive8818_validation")
    validator.e = SimpleNamespace(OWNED=m.OWNED, TESTS=m.TESTS, ROOT=m.ROOT)
    validator.original = SimpleNamespace(
        CONSUMERS=[
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]
    )
    validator.child = stream
    return dict(validator.validate(raw, private))


def prepare(raw: Path) -> Json:
    """Freeze methods before calibration, then commit evaluation decisions before label release."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    m.progress("before_authentication")
    authority = m.authenticate(m.UPSTREAM)
    m.meter.freeze()
    method = dict(
        arms=list(m.ARMS),
        primitive_limit=m.BUDGET,
        seed=m.SEED,
        operations=list(m.meter.KINDS),
        frozen_before_evaluation=True,
        operation_definition="One charge per original comparison expression, loop/comprehension iteration or dict get/setdefault. Fact comparisons include terms and signs; assignment and binding loops count separately. No source-line or machine-instruction equivalence.",
        protocol_source=reference(m.UPSTREAM),
        confidence=0.9875,
        multiplicity="four primary families; Bonferroni alpha 0.05/4",
        thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        deterministic_confidence="decisive decisions score one, insufficient zero",
        exact_reference="unrestricted unchanged joint-symbolic, separately costed",
        implementation_references=[
            reference(m.ROOT / p)
            for p in m.OWNED
            + [
                "python/carnot/reporting/evidence_set_methods_8768.py",
                "python/carnot/reporting/evidence_intervention_methods_8791.py",
                "python/carnot/reporting/evidence_qualification_8777.py",
                "python/carnot/reporting/v749_verifier_protocol.py",
                "python/carnot/reporting/verifier_family_surface_8755.py",
            ]
        ],
    )
    atomic_json(raw / "method.json", method)
    m.progress("after_authentication_before_calibration")
    calibration = m.cohort(60, "calibration")
    calibration_decisions = m.measure(calibration["inputs"])
    calibration_rows = m.score(calibration, calibration_decisions)
    selection = m.select(calibration_rows)
    atomic_json(
        raw / "calibration.json",
        dict(
            cohort=calibration,
            decisions=calibration_decisions,
            rows=calibration_rows,
            selection=selection,
        ),
    )
    frozen_at = time.monotonic_ns()
    m.progress("before_evaluation_freeze", 0, 240)
    cohort = m.cohort(240, "evaluation")
    collisions = sorted(
        {w["structure"] for w in calibration["worlds"]} & {w["structure"] for w in cohort["worlds"]}
    )
    disjoint = not collisions
    atomic_json(raw / "inputs.json", cohort)
    if not disjoint:
        authority["diagnostics"].append(
            dict(
                check="internal_failure",
                artifact_field="graph_structures_disjoint",
                expected=True,
                observed=False,
                overlapping_structures=collisions,
                calibration_reference=reference(raw / "calibration.json"),
                **reference(raw / "inputs.json"),
            )
        )
    decisions = m.measure(cohort["inputs"])
    atomic_json(raw / "decisions.json", dict(rows=decisions, commitment=canonical_hash(decisions)))
    measured_at = time.monotonic_ns()
    rows = m.score(cohort, decisions)
    atomic_json(
        raw / "private_labels.json",
        dict(
            rows=[
                {k: r[k] for k in ["unit_id", "truth", "condition", "valid_alternative"]}
                for r in rows
                if r["arm"] == "evidence_set"
            ]
        ),
    )
    checks = m.controls()
    end = time.monotonic_ns()
    spans = [
        dict(phase=p, start_s=(a - start) / 1e9, end_s=(b - start) / 1e9)
        for p, a, b in [
            ("authentication_calibration", start, frozen_at),
            ("input_only_evaluation", frozen_at, measured_at),
            ("independent_reduction", measured_at, end),
        ]
    ]
    receipt = build_current_work_receipt(
        events=[],
        run_id=raw.name,
        owner_pid=os.getpid(),
        inference_substrate="deterministic_local",
        inference_substrate_details=dict(scope="signed encoded graphs; exact CPU work; no models"),
        inference_substrate_class="cpu_exact_solver_or_simulator",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=spans,
    )
    pre = dict(
        passed=authority["passed"] and disjoint and selection["comparator"] is not None,
        authenticated_protocol=authority["passed"],
        graph_structures_disjoint=disjoint,
        private_output_writable=True,
        optional_branches_required=False,
        comparator_selected_before_evaluation=selection["comparator"] is not None,
    )
    debt = m.ROOT / "results/raw" / m.NAME / "startup/full_python_suite.receipt.json"
    return dict(
        global_test_spec_debt=dict(
            receipt=reference(debt) if debt.is_file() else None,
            full_suite_pass=False,
            scope="bounded global diagnostic; separate from owned acceptance",
        ),
        rows=rows,
        selection=selection,
        controls=checks,
        preconditions_checked=pre,
        source_artifact_hashes=authority["references"],
        gate_check_summary=authority["diagnostics"],
        code_config_hashes=method["implementation_references"],
        current_work_receipt=receipt,
        raw_shard_hashes=[
            reference(raw / p)
            for p in [
                "method.json",
                "calibration.json",
                "inputs.json",
                "decisions.json",
                "private_labels.json",
            ]
        ],
    )


def build(work: Json, validation: Json, date: str) -> Json:
    """Readiness qualifies measured evidence; lack of utility is a completed null."""
    qualified = bool(
        validation["union"]["passed"]
        and validation["coverage"]["complete"]
        and all(r["passed"] for r in validation["receipts"])
    )
    checks = all(r["passed"] for r in work["controls"])
    ready = int(qualified and checks and work["preconditions_checked"]["passed"])
    summary = m.reduce(work["rows"], work["selection"]["comparator"] or "joint_symbolic")
    utility = int(ready and all(summary["acceptance_gates"].values()))
    verdict = (
        "disqualified"
        if not qualified
        or not checks
        or work["preconditions_checked"].get("graph_structures_disjoint") is False
        or work["selection"]["comparator"] is None
        else "blocked"
        if not work["preconditions_checked"]["passed"]
        else "circular_positive"
        if utility
        else "null"
    )
    receipt = work["current_work_receipt"]
    rows = [r for r in work["rows"] if r["arm"] in m.ARMS]
    value = dict(
        summary,
        experiment_id=8818,
        task_id=m.TASK,
        milestone="2026.11.755",
        run_date=date,
        honest_verdict=f"complete_{verdict}_evidence_decisive_coverage",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        duration_s=receipt["duration_s"],
        random_seed=m.SEED,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"] + validation["raw_references"],
        phase_spans=receipt["phase_spans"],
        preconditions_checked=work["preconditions_checked"],
        rows=rows,
        intended_count=4320,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=0,
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=240,
        gate_check_summary=work["gate_check_summary"],
        required_checks_passed=qualified,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        exposure_scope="constructed eight-vertex graphs with two or three shared variables; input-only evaluation sealed before independent label release",
        independent_generalization_score=0,
        cited_upstream_artifacts=work["source_artifact_hashes"],
        evidence_rows_ready_score=ready,
        evidence_utility_score=utility,
        current_work_receipt=receipt,
        comparator_selection=work["selection"],
        mutation_controls=work["controls"],
        historical_nulls_preserved=True,
        flagged_adversarial=False,
        methodology_note="Exact encoded reasoning remains circular. No learned-semantic, natural-text, LLM or hidden-game improvement claim. Readiness means valid evidence, not benefit.",
        global_test_spec_debt=work["global_test_spec_debt"],
    )
    return value


def sealed_value(
    work: Json, validation: Json, date: str, work_ref: Json, validation_ref: Json
) -> Json:
    """Bind the full reduction and its actual check receipts to immutable private bytes."""
    value = build(work, validation, date)
    value.update(work_reference=work_ref, validation_reference=validation_ref)
    value["field_principles"] = {
        k: "Useful decisions and errors retain explicit denominators; exact semantics remain circular; hashes bind evidence."
        for k in value
    }
    value["field_principles"].update(
        field_principles="Every field states its evidence scope",
        reproducibility_checksum="Bind the complete reduction",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def audit(cohort: Json, decisions: list[Json]) -> bool:
    """Re-execute each operation ledger without treating a transport hash as semantic proof."""
    measured = m.measure(cohort["inputs"])
    timing = {"wall_time_s", "cpu_time_s", "started_monotonic_ns", "ended_monotonic_ns"}
    return len(measured) == len(decisions) and all(
        {k: v for k, v in a.items() if k not in timing}
        == {k: v for k, v in b.items() if k not in timing}
        and b["wall_time_s"] == (b["ended_monotonic_ns"] - b["started_monotonic_ns"]) / 1e9
        and b["wall_time_s"] >= 0
        and b["cpu_time_s"] >= 0
        for a, b in zip(measured, decisions, strict=True)
    )


def validation_evidence(validation: Json) -> bool:
    """Recompute actual child outcomes and SQLite coverage instead of trusting readiness flags."""
    try:
        union = validation["union"]
        if aggregate(union["retained_child_rows"], union["node_ids"]) != union:
            return False
        ref = validation["coverage"]["combined_reference"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            return False
        cov = Coverage(data_file=ref["path"], config_file=False)
        cov.load()
        files: list[Json] = []
        for name in m.OWNED:
            _, statements, excluded, missing, _ = cov.analysis2(str(m.ROOT / name))
            files.append(
                dict(path=name, statements=len(statements), excluded=excluded, missing=missing)
            )
        return bool(
            files == validation["coverage"]["files"]
            and all(f["statements"] > 0 and not f["excluded"] and not f["missing"] for f in files)
        )
    except (OSError, ValueError, KeyError, TypeError, CoverageException):
        return False


def replay(path: Path) -> bool:
    """Cold replay rejects changed decisions, independently reduced labels and aggregates."""
    try:
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict) or value.get("reproducibility_checksum") != canonical_hash(
            {k: v for k, v in value.items() if k != "reproducibility_checksum"}
        ):
            return False
        work, validation = load(value["work_reference"]), load(value["validation_reference"])
        refs = (
            work["source_artifact_hashes"]
            + work["code_config_hashes"]
            + work["raw_shard_hashes"]
            + validation["raw_references"]
        )
        if any(sha256_file(Path(ref["path"])) != ref["sha256"] for ref in refs):
            return False
        if value["required_checks_passed"] and not validation_evidence(validation):
            return False
        raw = Path(value["work_reference"]["path"]).parent
        cohort = json.loads((raw / "inputs.json").read_bytes())
        decisions = json.loads((raw / "decisions.json").read_bytes())
        calibration = json.loads((raw / "calibration.json").read_bytes())
        if cohort != m.cohort(240, "evaluation") or calibration["cohort"] != m.cohort(
            60, "calibration"
        ):
            return False
        if decisions["commitment"] != canonical_hash(decisions["rows"]):
            return False
        if not audit(cohort, decisions["rows"]) or not audit(
            calibration["cohort"], calibration["decisions"]
        ):
            return False
        if m.score(cohort, decisions["rows"]) != work["rows"]:
            return False
        calibration_rows = m.score(calibration["cohort"], calibration["decisions"])
        if (
            calibration_rows != calibration["rows"]
            or m.select(calibration_rows) != work["selection"]
        ):
            return False
        if work["controls"] != m.controls() or validate_current_work_receipt(
            work["current_work_receipt"]
        ):
            return False
        return bool(
            value
            == sealed_value(
                work,
                validation,
                value["run_date"],
                value["work_reference"],
                value["validation_reference"],
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(candidate: Path, raw: Path) -> Json:
    """Run both independent checkers and real outside-checkout replay before publication."""
    checks = []
    for name, script, extra in [
        ("adversarial", "scripts/adversarial_verify.py", []),
        ("rows_strict", "scripts/verdict_row_consistency_lint.py", ["--strict"]),
    ]:
        checks.append(
            stream(
                name,
                [sys.executable, "-u", script, str(candidate), *extra],
                raw / "terminal",
                deadline=180,
            )
        )
    with TemporaryDirectory(prefix="exp8818-cold-") as directory:
        checks.append(
            stream(
                "cold_replay",
                [
                    "env",
                    "-u",
                    "PYTHONPATH",
                    sys.executable,
                    "-u",
                    "-c",
                    "import os,runpy,sys; os.chdir(sys.argv.pop(1)); runpy.run_path(sys.argv.pop(1),run_name='__main__')",
                    directory,
                    str(m.ROOT / m.CLI),
                    "--cold-replay",
                    str(candidate),
                ],
                raw / "terminal",
                deadline=300,
            )
        )
    return dict(passed=all(c["passed"] for c in checks), receipts=checks)


def main(argv: list[str] | None = None) -> int:
    """Private science uses the same cohort but never grants itself owned qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261105"], default="20261105")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    m.progress("start_no_model_load")
    work = prepare(raw)
    atomic_json(raw / "work.json", work)
    with TemporaryDirectory(prefix="exp8818-checks-") as directory:
        m.progress("before_owned_validation")
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
        m.progress("after_owned_validation")
    atomic_json(raw / "validation.json", validation)
    value = sealed_value(
        work,
        validation,
        args.date,
        reference(raw / "work.json"),
        reference(raw / "validation.json"),
    )
    m.progress("before_terminal_publication")
    publication = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    atomic_json(raw / "publication_receipt.json", publication)
    m.progress("complete_" + value["verdict_class"], len(value["rows"]), 0)
    return 0

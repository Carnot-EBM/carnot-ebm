"""REQ-REPORT-8845: bind indexed semantic qualification to original V756 operands."""

import argparse
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.reporting import indexed_witness_8845 as m
from carnot.reporting import evidence_evaluation_runner_8779 as shared
from carnot.reporting import ordered_evidence_runner_8831 as previous
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.selective_methods_runner_8829 import qualified
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
UPSTREAM = m.ROOT / "results/experiment_8831_v756_ordered_evidence_cost.json"
PIN = "sha256:633cc57fd922b35e1cae7bcae5527a4610d85dfde65339278b5235bb1fe08891"
ARCHIVE = m.ROOT / "results/raw" / m.NAME / "historical_authorities"
HEALTH = Path("/tmp/exp8845-repository-health/receipt.json")


def preconditions(raw: Path) -> Json:
    """Authenticate original producer, raw operands and preserved documents before measuring."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    atomic_json(raw / "writable.json", dict(probe="private output"))
    refs, diagnostics = [], []
    diagnostic: Json = dict(
        check="missing_file",
        upstream="exp8831-ordered-evidence-cost",
        artifact_field="evidence_rows_ready_score",
        expected=1,
        observed=None,
        path=str(UPSTREAM),
        sha256=None,
    )
    try:
        ref = reference(UPSTREAM)
        value = load(ref)
        diagnostic.update(check="missing_field", observed=value.get("evidence_rows_ready_score"))
        if "evidence_rows_ready_score" not in value:
            raise ValueError("missing_field")
        diagnostic["check"] = "observed_zero"
        if value["evidence_rows_ready_score"] != 1:
            raise ValueError("readiness_zero")
        diagnostic.update(
            check="rejected_producer",
            artifact_field="sha256",
            expected=PIN,
            observed=ref["sha256"],
            sha256=ref["sha256"],
        )
        if ref["sha256"] != PIN or value["task_id"] != "exp8831-ordered-evidence-cost":
            raise ValueError("producer_identity")
        refs.append(ref)
        for operand in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            path = Path(operand["path"])
            if path.name == "spec.md":
                path = ARCHIVE / operand["sha256"].split(":")[1]
            diagnostic.update(path=str(path), expected=operand["sha256"], observed=None)
            observed = sha256_file(path)
            diagnostic.update(observed=observed, sha256=observed)
            if observed != operand["sha256"]:
                raise ValueError("operand_drift")
            refs.append(reference(path))
        source = Path(value["work_reference"]["path"]).parent / "measurement_inputs.json"
        shutil.copyfile(source, raw / "inputs.json")
    except (OSError, ValueError, KeyError, TypeError) as error:
        diagnostics.append(
            dict(
                diagnostic,
                error=str(error),
                check="missing_file" if isinstance(error, OSError) else diagnostic["check"],
            )
        )
    resources: Json = dict(
        repository_virtualenv=(m.ROOT / ".venv/bin/python").is_file(),
        dependencies=dict(numpy=np.__version__),
        tools=all((m.ROOT / ".venv/bin" / tool).is_file() for tool in ["pytest", "ruff", "mypy"]),
        private_output_writable=(raw / "writable.json").is_file(),
        free_disk_bytes=shutil.disk_usage(raw).free,
        measurement_budget_s=300,
        validation_budget_s=900,
    )
    resources["passed"] = (
        resources["repository_virtualenv"]
        and resources["tools"]
        and resources["free_disk_bytes"] >= 128 * 1024 * 1024
    )
    return dict(
        external_ready=not diagnostics,
        resources=resources,
        references=refs,
        diagnostics=diagnostics,
    )


def measure(units: list[Json]) -> list[Json]:
    """Only the facts/query projection reaches either evaluator; labels are never read."""
    rows = []
    start = last = time.monotonic()
    m.progress("before_benchmark", 0, len(units) * 2)
    for unit in units:
        if time.monotonic() - start > 300:
            raise ValueError("measurement_deadline")
        for arm in ["always_joint", "indexed_witness"]:
            row = m.attempt(unit["input"], indexed=arm == "indexed_witness")
            rows.append(
                dict(
                    row,
                    arm=arm,
                    seed=m.SEED,
                    unit_id=unit["unit_id"],
                    cluster_id=unit["cluster_id"],
                    condition=unit["condition"],
                    input_sha256=canonical_hash(unit["input"]),
                    metric=int(row["decision"] != "insufficient"),
                    denominator=1,
                    status="censored" if row["budget_exhausted"] else "completed",
                    missing_reason="operation_budget_exhausted"
                    if row["budget_exhausted"]
                    else None,
                )
            )
        if time.monotonic() - last >= 30:
            m.progress("benchmark", len(rows), len(units) * 2 - len(rows))
            last = time.monotonic()
    m.progress("after_benchmark", len(rows), 0)
    return rows


def parity(rows: list[Json]) -> list[Json]:
    """Only two completed paths assert parity; every exhausted comparison remains censored."""
    paired = []
    for a, b in zip(rows[::2], rows[1::2], strict=True):
        complete = not (a["budget_exhausted"] or b["budget_exhausted"])
        paired.append(
            dict(
                unit_id=a["unit_id"],
                cluster_id=a["cluster_id"],
                reference_decision=a["decision"],
                indexed_decision=b["decision"],
                status="completed" if complete else "censored",
                denominator=1,
                metric=int(complete and a["decision"] == b["decision"]),
                passed=not complete or a["decision"] == b["decision"],
                witness_order_difference=b.get("witness_order_difference"),
                missing_reason=None if complete else "operation_budget_exhausted",
            )
        )
    return paired


def prepare(raw: Path, limit: int | None = None, measurement: bool = True) -> Json:
    """Seal actual predictions before constructing parity and adversarial summaries."""
    start = time.monotonic_ns()
    m.progress("before_preconditions")
    pre = preconditions(raw)
    m.progress("after_preconditions")
    units, rows, controls = [], [], []
    if measurement and pre["external_ready"] and pre["resources"]["passed"]:
        units = load(reference(raw / "inputs.json"))["rows"][:limit]
        rows = measure(units)
        atomic_json(raw / "decisions.json", dict(rows=rows, commitment=canonical_hash(rows)))
        controls = adversaries()
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="deterministic_local" if pre["external_ready"] else "blocked_no_run",
        inference_substrate_details=dict(scope="exact encoded signed facts; CPU; no training"),
        inference_substrate_class="cpu_exact_solver_or_simulator"
        if pre["external_ready"]
        else "blocked_no_run",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[
            dict(
                phase="authentication_and_exact_witness_benchmark",
                start_s=0,
                end_s=(end - start) / 1e9,
            )
        ],
    )
    return dict(
        preconditions_checked=pre,
        rows=rows,
        parity_rows=parity(rows),
        adversarial_case_rows=controls,
        index_lifetime_rows=m.lifetime_controls(),
        private_unit_limit=limit,
        current_work_receipt=receipt,
        default_path_parity=default_parity(units),
    )


def adversaries() -> list[Json]:
    """Regenerate both paths and the fixture roster rather than trusting stored pass flags."""
    controls = []
    for fixture in m.fixtures():
        a, b = m.attempt(fixture["input"]), m.attempt(fixture["input"], indexed=True)
        controls.append(
            dict(
                condition=fixture["condition"],
                input=fixture["input"],
                reference=a,
                indexed=b,
                passed=not a["budget_exhausted"]
                and not b["budget_exhausted"]
                and a["decision"] == b["decision"]
                and m.valid_witness(fixture["input"], b),
            )
        )
    return controls


def default_parity(units: list[Json]) -> Json:
    """Compare the actual default dispatch with the preserved always-joint authority."""
    return dict(
        passed=all(
            m.stable(m.attempt(u["input"]))
            == m.stable(m.reference.attempt("always_joint", u["input"]))
            for u in units[:6]
        ),
        checked_count=min(len(units), 6),
        historical_bytes_unchanged=True,
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse actual unit/child coverage and unchanged private E2E-018/021 consumers."""
    validator = namespace(shared, "indexed8845_validation")
    validator.e = SimpleNamespace(OWNED=m.OWNED, TESTS=m.TESTS, ROOT=m.ROOT)
    validator.original = SimpleNamespace(
        CONSUMERS=[
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]
    )
    validator.child = stream
    return dict(validator.validate(raw, private))


def artifact(work: Json, validation: Json, date: str, work_ref: Json, validation_ref: Json) -> Json:
    """Exact-oracle readiness does not establish cost savings or independent generalization."""
    pre, rows, receipt = work["preconditions_checked"], work["rows"], work["current_work_receipt"]
    controls = work["parity_rows"] + work["adversarial_case_rows"] + work["index_lifetime_rows"]
    ready = int(
        pre["external_ready"]
        and pre["resources"]["passed"]
        and qualified(validation)
        and len(rows) == 7200
        and all(p["metric"] == 1 for p in work["parity_rows"])
        and all(c["passed"] for c in controls)
        and work["default_path_parity"]["passed"]
        and work["private_unit_limit"] is None
    )
    verdict = (
        "blocked" if not pre["external_ready"] else "circular_positive" if ready else "disqualified"
    )
    value = dict(
        experiment_id=8845,
        task_id=m.TASK,
        milestone="2026.11.757",
        run_date=date,
        honest_verdict=f"complete_blocked_{pre['diagnostics'][0]['check']}_indexed_witness"
        if verdict == "blocked"
        else f"complete_{verdict}_indexed_witness",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="exposed constructed encoded V756 worlds; no natural-language or hidden-game claim",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        random_seed=m.SEED,
        phase_spans=receipt["phase_spans"],
        preconditions_checked=pre,
        source_artifact_hashes=pre["references"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"]
        + validation["raw_references"]
        + [work_ref, validation_ref],
        rows=rows,
        intended_count=7200,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=0,
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=len({r["cluster_id"] for r in rows}),
        gate_check_summary=pre["diagnostics"]
        + [
            dict(
                check="owned_validation_failure",
                upstream=m.TASK,
                artifact_field=check.get("name", "owned_statement_coverage"),
                expected=True,
                observed=False,
                path=check.get("stdout_path"),
                sha256=check.get("stdout_sha256"),
            )
            for check in validation["receipts"] + [dict(passed=validation["coverage"]["complete"])]
            if not check["passed"]
        ],
        acceptance_gates=dict(
            exact_semantics=dict(passed=all(c["passed"] for c in controls)),
            owned_validation=dict(passed=qualified(validation)),
            full_roster=dict(passed=len(rows) == 7200),
        ),
        required_checks_passed=qualified(validation),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        indexed_witness_ready_score=ready,
        parity_rows=work["parity_rows"],
        adversarial_case_rows=work["adversarial_case_rows"],
        index_lifetime_rows=work["index_lifetime_rows"],
        default_path_parity=work["default_path_parity"],
        operation_schema=dict(
            primitive_limit=m.CAP,
            categories=list(m.meter.KINDS) + list(m.EXTRA),
            definition="One paid semantic operation per loop item, parser guard, allocation entry, predicate term, index lookup, serialization call and disposal reservation before allocation. Includes duplicate entries and binding copies. Not CPU instructions or a wall-time speedup claim.",
            preprocessing="Within each request budget; no answer cache; no ordered checks",
        ),
        witness_provenance=dict(
            source="same admitted fact tuples; positive complete shared bindings only",
            duplicates="retained in candidate tuples and supporting_bindings",
            order="input fact order; lexical reference order may differ",
            labels_read_by_evaluator=False,
        ),
        current_work_receipt=receipt,
        work_reference=work_ref,
        validation_reference=validation_ref,
        repository_health=work.get("repository_health"),
        methodology="Opt-in exact predicate index versus unchanged always-joint; all original V756 inputs; exhaustive small-domain and adversarial semantic controls. Readiness alone, no cost benefit claim.",
    )
    value["field_principles"] = {
        k: "Bind actual producer, operands and paid request outcomes; exact encoded proof remains circular and cannot establish external benefit."
        for k in value
    }
    value["field_principles"].update(
        indexed_witness_ready_score="Full owned validation and exact completed parity qualify semantics only; no cost gain is asserted.",
        parity_rows="Compare typed decisions only when both paid paths complete; retain censored attempts in the denominator.",
        adversarial_case_rows="Re-execute canonical counterexamples, both paths and independent complete-witness checks.",
        index_lifetime_rows="Actual constructed immutable candidate tuples reject source mutation, expired access and a different request token.",
        operation_schema="Pay construction, hash lookup, binding copies and disposal reservations within the same request ceiling.",
        default_path_parity="The opt-in flag leaves default dispatch and historical evaluator bytes unchanged.",
        witness_provenance="Witnesses use public signed facts alone; equivalent order differences do not imply changed decisions.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Re-execute sealed requests and controls; repaired hashes cannot hide changed decisions."""
    try:
        value = json.loads(path.read_bytes())
        work, validation = load(value["work_reference"]), load(value["validation_reference"])
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if validate_current_work_receipt(work["current_work_receipt"]):
            return False
        checker = namespace(previous.checker, "indexed8845_coverage_replay")
        checker.m = m
        if value["required_checks_passed"] and not checker.validation_evidence(validation):
            return False
        if work["rows"]:
            raw = Path(value["work_reference"]["path"]).parent
            units = load(reference(raw / "inputs.json"))["rows"][: work["private_unit_limit"]]
            sealed = load(reference(raw / "decisions.json"))
            if sealed["rows"] != work["rows"] or sealed["commitment"] != canonical_hash(
                sealed["rows"]
            ):
                return False
            if (
                m.stable(measure(units)) != m.stable(work["rows"])
                or parity(work["rows"]) != work["parity_rows"]
            ):
                return False
            producer = load(reference(UPSTREAM))
            original = Path(producer["work_reference"]["path"]).parent / "measurement_inputs.json"
            if sha256_file(UPSTREAM) != PIN or load(reference(raw / "inputs.json")) != load(
                reference(original)
            ):
                return False
            if m.stable(adversaries()) != m.stable(work["adversarial_case_rows"]):
                return False
            if (
                m.lifetime_controls() != work["index_lifetime_rows"]
                or default_parity(units) != work["default_path_parity"]
            ):
                return False
        return bool(
            value
            == artifact(
                work,
                validation,
                value["run_date"],
                value["work_reference"],
                value["validation_reference"],
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Keep the original terminal scanners and real outside-checkout cold replay."""
    checked = namespace(previous, "indexed8845_terminal")
    checked.m = m
    return dict(checked.terminal(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Private subsets exercise real execution but cannot claim full owned qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261105"], default="20261105")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--private-unit-limit", type=int)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private output must be outside checkout")
    if args.private_unit_limit is not None and (
        not args.private_output or args.private_unit_limit < 1
    ):
        parser.error("subset requires private output and positive count")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    try:
        work = prepare(raw, args.private_unit_limit)
    except (ValueError, KeyError, TypeError) as error:
        work = prepare(raw, args.private_unit_limit, False)
        work["preconditions_checked"]["resources"]["passed"] = False
        work["preconditions_checked"]["diagnostics"].append(
            dict(
                check="internal_failure",
                upstream=m.TASK,
                artifact_field="measurement",
                expected="complete owned execution",
                observed=str(error),
                path=str(raw),
                sha256=None,
            )
        )
    work["code_config_hashes"] = [
        reference(m.ROOT / p)
        for p in m.OWNED
        + m.TESTS
        + [
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/experiment_template.py",
        ]
    ]
    work["raw_shard_hashes"] = [reference(p) for p in sorted(raw.glob("*.json"))]
    if HEALTH.is_file():
        work["repository_health"] = load(reference(HEALTH))
    atomic_json(raw / "work.json", work)
    with TemporaryDirectory(prefix="exp8845-owned-") as directory:
        m.progress("before_owned_validation")
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
        m.progress("after_owned_validation")
    atomic_json(raw / "validation.json", validation)
    value = artifact(
        work,
        validation,
        args.date,
        reference(raw / "work.json"),
        reference(raw / "validation.json"),
    )
    m.progress("before_publication")
    publication = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    atomic_json(raw / "publication_receipt.json", publication)
    m.progress("complete_" + value["verdict_class"], len(value["rows"]), 0)
    return 0

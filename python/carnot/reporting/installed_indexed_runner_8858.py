"""REQ-REPORT-8858: qualify the installed exact evaluator against immutable authorities."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

from carnot.reporting import evidence_evaluation_runner_8779 as shared
from carnot.reporting import indexed_cost_runner_8847 as previous
from carnot.reporting import indexed_cost_8847 as cost
from carnot.reporting import indexed_witness_8845 as witness
from carnot.reporting import ordered_evidence_runner_8831 as ordered
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_methods_runner_8829 import qualified
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8858_v758_installed_indexed_verifier", "exp8858-installed-indexed-verifier"
CLI = f"scripts/experiments/{NAME}.py"
CLOSURE = [
    f"python/carnot/verify/{name}.py"
    for name in ["indexed_witness", "_indexed_witness_core", "_indexed_witness_meter", "__init__"]
]
OWNED = CLOSURE + [
    "scripts/experiments/installed_indexed_probe_8858.py",
    "python/carnot/reporting/installed_indexed_runner_8858.py",
    CLI,
]
TESTS = ["tests/python/test_installed_indexed_verifier_8858.py"]
FIXTURES = ROOT / "results/raw" / NAME / "authorities/semantic_fixtures.json"
FIXTURE_PIN = "sha256:525c73f8e13eb870e7e6e637f4858f04a19f43cba24b181f67509855e3187235"
COST_PIN = "sha256:47548df6e4e4437f12f353298461b28be99129ee7759a55da779fb93a55f9a3e"
SEED = 7588858


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose real phase boundaries without extending the fixed execution budget."""
    print(f"[exp8858] phase={phase} completed={completed} pending={pending}", flush=True)


def preconditions(raw: Path) -> Json:
    """Authenticate original sources before accepting any readiness or structured input."""
    pre = previous.preconditions(raw)
    diagnostic: Json = dict(
        check="missing_file",
        upstream="exp8847-indexed-cost",
        artifact_field="indexed_cost_ready_score",
        expected=1,
        observed=None,
        path=str(cost.ROOT / "results" / (cost.NAME + ".json")),
        sha256=None,
    )
    try:
        path = Path(diagnostic["path"])
        value = load(reference(path))
        diagnostic.update(check="missing_field", observed=value.get("indexed_cost_ready_score"))
        if "indexed_cost_ready_score" not in value:
            raise ValueError("missing_field")
        diagnostic["check"] = "observed_zero"
        if value["indexed_cost_ready_score"] != 1:
            raise ValueError("readiness_zero")
        diagnostic.update(
            check="rejected_producer",
            expected=COST_PIN,
            observed=sha256_file(path),
            sha256=sha256_file(path),
        )
        if (
            sha256_file(path) != COST_PIN
            or value["task_id"] != "exp8847-indexed-cost"
            or not value["required_checks_passed"]
        ):
            raise ValueError("producer_identity")
        for ref in (
            [reference(path), value["work_reference"], value["validation_reference"]]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
            + [dict(path=str(FIXTURES), sha256=FIXTURE_PIN)]
        ):
            diagnostic.update(path=ref["path"], expected=ref["sha256"], observed=None)
            observed = sha256_file(Path(ref["path"]))
            diagnostic.update(observed=observed, sha256=observed)
            if observed != ref["sha256"]:
                raise ValueError("operand_drift")
            pre["references"].append(ref)
    except (OSError, ValueError, KeyError, TypeError) as error:
        pre["diagnostics"].append(
            dict(
                diagnostic,
                error=str(error),
                check="missing_file" if isinstance(error, OSError) else diagnostic["check"],
            )
        )
    pre["external_ready"] = not pre["diagnostics"]
    pre["resources"].update(build_tool=shutil.which("uv") is not None, total_deadline_s=4200)
    pre["resources"]["passed"] &= pre["resources"]["build_tool"]
    return dict(pre)


def installed(raw: Path, units: list[Json]) -> Json:
    """Build a wheel and execute its public entry from a dependency-free private environment."""
    atomic_json(raw / "requests.json", dict(rows=units))
    receipts = []
    with TemporaryDirectory(prefix="exp8858-install-") as directory:
        private = Path(directory)
        source = private / "source"
        source.mkdir()
        for name in ["pyproject.toml", "README.md", "LICENSE", "NOTICE"]:
            shutil.copy2(ROOT / name, source / name)
        shutil.copytree(
            ROOT / "python",
            source / "python",
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
        )
        wheelhouse = raw / "wheel"
        for name, argv in [
            (
                "build",
                [
                    "uv",
                    "build",
                    "--wheel",
                    "--no-build-isolation",
                    "--project",
                    str(source),
                    "--out-dir",
                    str(wheelhouse),
                    "--python",
                    sys.executable,
                ],
            ),
            ("venv", ["uv", "venv", str(private / "venv"), "--python", sys.executable]),
        ]:
            progress("before_" + name)
            receipts.append(stream(name, argv, raw / "install_logs", deadline=120))
            progress("after_" + name)
        wheels = list(wheelhouse.glob("*.whl"))
        if len(wheels) != 1 or not all(r["passed"] for r in receipts):
            raise ValueError("package_build")
        python = private / "venv/bin/python"
        for name, argv in [
            (
                "install",
                ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheels[0])],
            ),
            (
                "installed_probe",
                [
                    "env",
                    "-u",
                    "PYTHONPATH",
                    str(python),
                    "-I",
                    "-u",
                    str(ROOT / "scripts/experiments/installed_indexed_probe_8858.py"),
                    str(raw / "requests.json"),
                    str(raw / "installed.json"),
                ],
            ),
        ]:
            progress("before_" + name)
            receipts.append(stream(name, argv, raw / "install_logs", deadline=300))
            progress("after_" + name)
        if not all(r["passed"] for r in receipts):
            raise ValueError("installed_execution")
    value = load(reference(raw / "installed.json"))
    return dict(value, package_hash=reference(wheels[0]), receipts=receipts)


def prepare(raw: Path, limit: int | None = None, measurement: bool = True) -> Json:
    """Retain the complete schedule even when owned installation or controls fail."""
    started = time.monotonic_ns()
    progress("before_preconditions")
    pre = preconditions(raw)
    progress("after_preconditions")
    units = load(reference(FIXTURES))["rows"][:limit] if FIXTURES.is_file() else []
    proof: Json = dict(rows=[], installed_import_rows=[], receipts=[], package_hash=None)
    errors = []
    if pre["external_ready"] and pre["resources"]["passed"] and measurement:
        try:
            proof = installed(raw, units)
        except (ValueError, OSError, KeyError, TypeError) as error:
            errors.append(str(error))
    rows, semantic, operations = [], [], []
    for i, unit in enumerate(units):
        actual = proof["rows"][i] if i < len(proof["rows"]) else None
        expected = unit["expected"]
        semantics = actual is not None and all(
            actual.get(k) == expected.get(k)
            for k in [
                "decision",
                "witness",
                "supporting_bindings",
                "witness_order_difference",
                "budget_exhausted",
            ]
        )
        paid = actual is not None and all(
            actual.get(k) == expected.get(k)
            for k in ["primitive_limit", "primitive_operations", "operation_counts"]
        )
        identity = dict(unit_id=unit["unit_id"], cluster_id=unit["cluster_id"], denominator=1)
        semantic.append(dict(identity, metric=int(semantics), passed=semantics))
        operations.append(dict(identity, metric=int(paid), passed=paid))
        rows.append(
            dict(
                identity,
                **(actual or {}),
                arm="installed_indexed_witness",
                seed=SEED,
                metric=int(semantics and paid),
                status="completed" if actual is not None else "pending",
                missing_reason=None if actual is not None else "installation_unfinished",
            )
        )
    from carnot.verify import _indexed_witness_core as core
    from carnot.verify import _indexed_witness_meter as installed_meter

    control = namespace(witness, "installed8858_ownership")
    control.REQUEST, control.meter, control.build_index = (
        core.REQUEST,
        installed_meter,
        core.build_index,
    )
    ownership = control.lifetime_controls() if proof["rows"] else []
    ended = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="no_model_load" if proof["rows"] else "blocked_no_run",
        inference_substrate_details=dict(
            compute="exact structured-record CPU evaluation", model_invoked=False
        ),
        inference_substrate_class="no_model_load" if proof["rows"] else "blocked_no_run",
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=[
            dict(phase="installed_qualification", start_s=0, end_s=(ended - started) / 1e9)
        ],
    )
    atomic_json(raw / "decisions.json", dict(rows=rows))
    return dict(
        preconditions_checked=pre,
        rows=rows,
        semantic_parity_rows=semantic,
        operation_parity_rows=operations,
        ownership_rows=ownership,
        installed_import_rows=proof["installed_import_rows"],
        package_hash=proof["package_hash"],
        installation_receipts=proof["receipts"],
        current_work_receipt=receipt,
        private_unit_limit=limit,
        internal_errors=errors,
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse real unit/child coverage and unchanged private historical consumers."""
    validator = namespace(shared, "installed8858_validation")
    validator.e = SimpleNamespace(OWNED=OWNED, TESTS=TESTS, ROOT=ROOT)
    validator.original = SimpleNamespace(
        CONSUMERS=[
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]
    )
    validator.child = stream
    result = dict(validator.validate(raw, private))
    return result


def artifact(work: Json, validation: Json, date: str, work_ref: Json, validation_ref: Json) -> Json:
    """Installed readiness requires actual parity and checks; exact oracle success stays circular."""
    pre, rows, receipt = work["preconditions_checked"], work["rows"], work["current_work_receipt"]
    controls = (
        work["semantic_parity_rows"]
        + work["operation_parity_rows"]
        + work["ownership_rows"]
        + work["installed_import_rows"]
    )
    gates = dict(
        authorities=pre["external_ready"],
        resources=pre["resources"]["passed"],
        exact_installed_parity=(
            bool(controls)
            and all(c["passed"] for c in controls)
            and [c["check"] for c in work["ownership_rows"]]
            == [
                "source_mutation_isolated",
                "immutable_table",
                "stale_reuse_rejected",
                "cross_query_rejected",
            ]
            and [c["check"] for c in work["installed_import_rows"]]
            == [
                "installed_outside_checkout_no_pythonpath",
                "no_reporting_training_or_artifact_loaders",
            ]
        ),
        full_roster=(
            work["private_unit_limit"] is None
            and FIXTURES.is_file()
            and len(rows) == 3614
            and [r["unit_id"] for r in rows]
            == [u["unit_id"] for u in load(reference(FIXTURES))["rows"]]
            and [r["unit_id"] for r in rows] == [r["unit_id"] for r in work["semantic_parity_rows"]]
            and [r["unit_id"] for r in rows]
            == [r["unit_id"] for r in work["operation_parity_rows"]]
        ),
        owned_validation=qualified(validation),
        installation=bool(work["package_hash"])
        and all(c["passed"] for c in work["installation_receipts"]),
        owned_execution=not work["internal_errors"],
    )
    ready = int(all(gates.values()))
    verdict = (
        "blocked" if not pre["external_ready"] else "circular_positive" if ready else "disqualified"
    )
    value = dict(
        experiment_id=8858,
        task_id=TASK,
        milestone="2026.11.758",
        run_date=date,
        honest_verdict="complete_blocked_" + pre["diagnostics"][0]["check"]
        if verdict == "blocked"
        else "complete_" + verdict + "_installed_indexed_verifier",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="explicit constructed signed records; no natural-language truth or service timing claim",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=receipt["phase_spans"],
        preconditions_checked=pre,
        random_seed=SEED,
        source_artifact_hashes=pre["references"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"]
        + validation["raw_references"]
        + [work_ref, validation_ref],
        rows=rows,
        intended_count=3614,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=0,
        censored_count=sum(r["status"] != "completed" for r in rows),
        excluded_count=0,
        independent_count=len({r["cluster_id"] for r in rows}),
        gate_check_summary=pre["diagnostics"]
        + [
            dict(
                check="owned_gate_failure",
                upstream=TASK,
                artifact_field=k,
                expected=True,
                observed=v,
                path=work_ref.get("path"),
                sha256=work_ref.get("sha256"),
            )
            for k, v in gates.items()
            if not v
        ],
        acceptance_gates=gates,
        required_checks_passed=qualified(validation),
        validation_receipts=validation["receipts"] + work["installation_receipts"],
        owned_statement_coverage=validation["coverage"],
        installed_index_ready_score=ready,
        caller_inventory=work["caller_inventory"],
        dependency_closure=CLOSURE,
        semantic_parity_rows=work["semantic_parity_rows"],
        operation_parity_rows=work["operation_parity_rows"],
        installed_import_rows=work["installed_import_rows"],
        ownership_rows=work["ownership_rows"],
        package_hash=work["package_hash"],
        current_work_receipt=receipt,
        work_reference=work_ref,
        validation_reference=validation_ref,
        repository_health=work.get("repository_health"),
        methodology="Build a private wheel; compare its outside-checkout public calls against all 3600 original requests and 14 pinned adversaries. No model, training, or timing benefit measurement.",
    )
    value["field_principles"] = {
        k: "Bind actual executed sources, installed outcomes and validation; constructed exact-oracle evidence establishes installed readiness only."
        for k in value
    }
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Re-execute sealed requests through an installed wheel; repaired hashes cannot forge parity."""
    try:
        value = load(reference(path))
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
        checker = namespace(ordered.checker, "installed8858_coverage_replay")
        checker.m = SimpleNamespace(ROOT=ROOT, OWNED=OWNED)
        if value["required_checks_passed"] and not checker.validation_evidence(validation):
            return False
        if work["package_hash"]:
            units = load(reference(FIXTURES))["rows"][: work["private_unit_limit"]]
            with TemporaryDirectory(prefix="exp8858-replay-") as directory:
                observed = installed(Path(directory), units)
            for unit, actual, saved, semantic, paid in zip(
                units,
                observed["rows"],
                work["rows"],
                work["semantic_parity_rows"],
                work["operation_parity_rows"],
                strict=True,
            ):
                for key, expected in unit["expected"].items():
                    if actual.get(key) != expected or saved.get(key) != expected:
                        return False
                identity = dict(
                    unit_id=unit["unit_id"],
                    cluster_id=unit["cluster_id"],
                    denominator=1,
                    metric=1,
                    passed=True,
                )
                if semantic != identity or paid != identity:
                    return False
            if observed["installed_import_rows"] != work["installed_import_rows"]:
                return False
            from carnot.verify import _indexed_witness_core as core
            from carnot.verify import _indexed_witness_meter as installed_meter

            control = namespace(witness, "installed8858_replay_ownership")
            control.REQUEST, control.meter, control.build_index = (
                core.REQUEST,
                installed_meter,
                core.build_index,
            )
            if control.lifetime_controls() != work["ownership_rows"]:
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
    """Retain the original prospective scanners and real outside-checkout CLI replay."""
    checker = namespace(ordered, "installed8858_terminal")
    checker.m = SimpleNamespace(ROOT=ROOT, CLI=CLI, progress=progress)
    return dict(checker.terminal(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Only a complete owned qualification can grant installed readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261106"], default="20261106")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--private-unit-limit", type=int)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_output and output.resolve().is_relative_to(ROOT):
        parser.error("private output must be outside checkout")
    if args.private_unit_limit is not None and (
        not args.private_output or args.private_unit_limit < 1
    ):
        parser.error("subset requires private output and a positive count")
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    work = prepare(raw, args.private_unit_limit)
    inventory = [
        p
        for base in ["python", "scripts/experiments", "tests/python"]
        for p in (ROOT / base).rglob("*.py")
        if "indexed_witness_8845" in p.read_text() or "indexed_cost_8847" in p.read_text()
    ]
    work["caller_inventory"] = [reference(p) for p in sorted(inventory)]
    work["code_config_hashes"] = [
        reference(ROOT / p)
        for p in OWNED
        + TESTS
        + [
            "python/carnot/__init__.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/experiment_template.py",
        ]
    ]
    work["raw_shard_hashes"] = [reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    health = ROOT / "results/raw" / NAME / "repository_health/receipt.json"
    work["repository_health"] = (
        load(reference(health))
        if health.is_file()
        else dict(status="not_run", scope="global_health_only")
    )
    atomic_json(raw / "work.json", work)
    progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8858-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    progress("after_owned_validation")
    atomic_json(raw / "validation.json", validation)
    value = artifact(
        work,
        validation,
        args.date,
        reference(raw / "work.json"),
        reference(raw / "validation.json"),
    )
    progress("before_publication")
    publication = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    atomic_json(raw / "publication_receipt.json", publication)
    for operand in raw.rglob("*"):
        if operand.is_file():
            operand.chmod(0o444)
    progress("complete_" + value["verdict_class"], len(value["rows"]), 0)
    return 0

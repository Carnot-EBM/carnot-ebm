"""REQ-REPORT-8871: qualify installed size dispatch with immutable exact evidence."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

from carnot.reporting import evidence_evaluation_runner_8779 as shared
from carnot.reporting import installed_indexed_runner_8858 as historical
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
NAME, TASK = "experiment_8871_v759_size_aware_verifier", "exp8871-size-aware-verifier"
CLI = f"scripts/experiments/{NAME}.py"
PROBE = "scripts/experiments/size_aware_probe_8871.py"
OWNED = [
    "python/carnot/verify/size_aware_witness.py",
    "python/carnot/reporting/size_aware_verifier_8871.py",
    PROBE,
    CLI,
]
TESTS = [
    "tests/python/test_size_aware_verifier_8871.py",
    "tests/python/test_size_aware_research_8871.py",
]
SIZES, SEED, CAP = (8, 16, 17, 64, 256, 1024), 7598871, 200000
HISTORY = [
    (
        ROOT / "results/experiment_8858_v758_installed_indexed_verifier.json",
        "sha256:28ef6d677e2684ad69d8916f4d4e2ec24dcb453b9349bf80de5ada2e70af65e6",
        "installed_index_ready_score",
    ),
    (
        ROOT / "results/experiment_8861_v758_installed_index_cost.json",
        "sha256:7d3e94e552dac90e0a6c54ee0ab0fcbf8135b7337b1ed5a0a7a911ceac511d31",
        "installed_cost_ready_score",
    ),
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries so a bounded child never appears silently stalled."""
    print(f"[exp8871] phase={phase} completed={completed} pending={pending}", flush=True)


def preconditions(raw: Path) -> Json:
    """Reject missing external inputs before any wheel build or owned evaluation."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    refs, diagnostics = [], []
    for path, pin, field in HISTORY:
        diagnostic = dict(
            check="missing_file",
            upstream=str(path),
            artifact_field=field,
            expected=1,
            observed=None,
            path=str(path),
            sha256=None,
        )
        try:
            value = load(reference(path))
            diagnostic.update(check="missing_field", observed=value.get(field))
            if field not in value:
                raise ValueError("missing_field")
            diagnostic["check"] = "observed_zero"
            if value[field] != 1:
                raise ValueError("readiness_zero")
            diagnostic.update(
                check="rejected_producer",
                expected=pin,
                observed=sha256_file(path),
                sha256=sha256_file(path),
            )
            if sha256_file(path) != pin or not value["required_checks_passed"]:
                raise ValueError("producer_identity")
            for ref in (
                [reference(path)]
                + value["source_artifact_hashes"]
                + value["code_config_hashes"]
                + value["raw_shard_hashes"]
            ):
                diagnostic.update(path=ref["path"], expected=ref["sha256"], observed=None)
                observed = sha256_file(Path(ref["path"]))
                diagnostic.update(observed=observed, sha256=observed)
                if observed != ref["sha256"]:
                    raise ValueError("operand_drift")
                refs.append(ref)
            shutil.copy2(path, raw / path.name)
        except (OSError, ValueError, KeyError, TypeError) as error:
            diagnostics.append(
                dict(
                    diagnostic,
                    check="missing_file" if isinstance(error, OSError) else diagnostic["check"],
                    error=str(error),
                )
            )
    probe = raw / "writable.json"
    atomic_json(probe, dict(private=True))
    resources = dict(
        private_writable=load(reference(probe))["private"],
        free_disk_bytes=shutil.disk_usage(raw).free,
        tools=all((ROOT / ".venv/bin" / p).is_file() for p in ("python", "pytest", "ruff", "mypy")),
        uv=shutil.which("uv") is not None,
        total_deadline_s=4200,
        hard_cap_s=4800,
        validation_reserve_s=1800,
    )
    resources["passed"] = all(
        (
            resources["private_writable"],
            resources["tools"],
            resources["uv"],
            resources["free_disk_bytes"] > 100_000_000,
        )
    )
    return dict(
        external_ready=not diagnostics,
        resources=resources,
        diagnostics=diagnostics,
        references=list({r["path"]: r for r in refs}.values()),
    )


def freeze(*, pilot: bool = False) -> list[Json]:
    """Seal independent worlds and fixed interventions without consulting any arm or clock."""
    rng, worlds = random.Random(SEED + int(pilot)), []
    for size, count, population in [(s, 1 if pilot else 30, "ordinary") for s in SIZES] + [
        (64, 1 if pilot else 40, "stress")
    ]:
        for i in range(count):
            cluster = f"{'pilot' if pilot else 'later'}/{population}/{size}/{i:03}"
            prefix = f"{rng.getrandbits(64):016x}"
            query = [["?x", "r", prefix + "b"], [prefix + "g", "gate", prefix + "h"]]
            facts = [[prefix + "a", "r", prefix + "b", True], [*query[-1], True]]
            if population == "stress":
                query = [[f"?v{k}", f"r{k}", prefix + "b"] for k in range(4)] + [query[-1]]
                facts = [
                    [prefix + str(j), f"r{k}", prefix + "b", True]
                    for k in range(4)
                    for j in range(6)
                ] + [facts[-1]]
            requests = []
            for condition in ("support", "refute", "missing", "conflict"):
                records = deepcopy(facts)
                if condition == "refute":
                    records[-1][3] = False
                elif condition == "missing":
                    records.pop()
                elif condition == "conflict":
                    records.append([*records[0][:3], False])
                while len(records) < size:
                    records.append([prefix + str(len(records)), "noise", "n", False])
                payload = dict(facts=records, query=query)
                requests.append(
                    dict(
                        unit_id=cluster + "/" + condition,
                        input=payload,
                        input_sha256=canonical_hash(payload),
                        truth="support"
                        if condition == "support"
                        else "refute"
                        if condition == "refute"
                        else "insufficient",
                    )
                )
            arms = ["indexed", "size_aware"]
            rng.shuffle(arms)
            worlds.append(
                dict(
                    cluster_id=cluster,
                    size=size,
                    population=population,
                    arm_order=arms,
                    requests=requests,
                )
            )
    return worlds


def qualification() -> list[Json]:
    """Freeze boundary and join adversaries separately from pilot and cost worlds."""
    units = []
    for size in (0, 1, 15, 16, 17, 64):
        payload = dict(
            facts=(
                [["z", "r", "z", True], ["a", "r", "a", True]] * 2
                + [["n", "noise", "m", False]] * size
            )[:size],
            query=[["?x", "r", "?x"]],
        )
        units.append(dict(unit_id=f"boundary/{size}", input=payload))
    for i, unit in enumerate(load(reference(historical.FIXTURES))["rows"][-14:]):
        units.append(dict(unit_id=f"adversary/{i}", input=unit["input"]))
    return units


def installed(raw: Path, worlds: list[Json]) -> Json:
    """Build once and run pilots and sealed requests in a dependency-free private wheel environment."""
    started = time.monotonic()
    receipts = []
    with TemporaryDirectory(prefix="exp8871-installed-") as directory:
        private = Path(directory)
        source = private / "source"
        source.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE"):
            shutil.copy2(ROOT / name, source / name)
        shutil.copytree(
            ROOT / "python",
            source / "python",
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
        )
        python = private / "venv/bin/python"
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
                    str(raw / "wheel"),
                    "--python",
                    sys.executable,
                ],
            ),
            ("venv", ["uv", "venv", str(private / "venv"), "--python", sys.executable]),
        ]:
            progress("before_" + name)
            receipts.append(stream(name, argv, raw / "install_logs", deadline=120))
            progress("after_" + name)
        wheels = list((raw / "wheel").glob("*.whl"))
        if len(wheels) != 1 or not all(r["passed"] for r in receipts):
            raise ValueError("package_build")
        receipts.append(
            stream(
                "install",
                ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheels[0])],
                raw / "install_logs",
                deadline=120,
            )
        )
        for phase, roster in [("pilot", freeze(pilot=True)), ("measurement", worlds)]:
            atomic_json(
                raw / (phase + "_requests.json"),
                dict(worlds=roster, qualification=qualification(), cap=CAP, seed=SEED),
            )
            progress("before_" + phase, 0, len(roster))
            command = [
                "env",
                "-u",
                "PYTHONPATH",
                str(python),
                "-I",
                "-u",
                "-c",
                "import os,runpy,sys; os.chdir(sys.argv.pop(1)); sys.argv[0]=sys.argv.pop(1); runpy.run_path(sys.argv[0],run_name='__main__')",
                str(private),
                str(ROOT / PROBE),
                str(raw / (phase + "_requests.json")),
                str(raw / (phase + ".json")),
            ]
            receipts.append(stream(phase, command, raw / "install_logs", deadline=900))
            progress("after_" + phase, len(roster), 0)
            if not all(r["passed"] for r in receipts):
                raise ValueError("installed_execution")
            pilot = load(reference(raw / (phase + ".json")))
            if not pilot["size_dispatch_correct"] or not all(
                c["passed"] for c in pilot["installed_import_rows"]
            ):
                raise ValueError("installed_correctness")
            if (
                phase == "pilot"
                and time.monotonic()
                - started
                + pilot["measurement_s"] * len(worlds) / len(roster)
                + 1800
                >= 4200
            ):
                raise ValueError("pilot_deadline_projection")
        return dict(pilot, package_hash=reference(wheels[0]), receipts=receipts)


def prepare(raw: Path, limit: int | None = None) -> Json:
    """Retain every scheduled arm when external inputs block or owned execution fails."""
    started = time.monotonic_ns()
    progress("before_preconditions")
    pre = preconditions(raw)
    progress("after_preconditions")
    worlds = freeze()[:limit]
    atomic_json(raw / "cost_roster.json", dict(worlds=worlds, threshold=16, seed=SEED))
    errors: list[str] = []
    proof: Json = dict(
        rows=[],
        controls=[],
        semantic_parity_rows=[],
        boundary_size_rows=[],
        ownership_rows=[],
        installed_import_rows=[],
        package_hash=None,
        receipts=[],
    )
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            proof = installed(raw, worlds)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
    observed = {(r["unit_id"], r["arm"]): r for r in proof["rows"]}
    rows = []
    for world in worlds:
        for arm in world["arm_order"]:
            for unit in world["requests"]:
                row = observed.get((unit["unit_id"], arm))
                rows.append(
                    row
                    if row is not None
                    else dict(
                        unit_id=unit["unit_id"],
                        cluster_id=world["cluster_id"],
                        arm=arm,
                        seed=SEED,
                        metric=0,
                        denominator=1,
                        size=world["size"],
                        population=world["population"],
                        status="failed" if errors else "censored",
                        missing_reason=errors[0] if errors else "external_or_resource_gate",
                        decision="insufficient",
                        witness=None,
                        primitive_operations=0,
                        operation_counts={},
                    )
                )
    ended = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="no_model_load" if proof["rows"] else "blocked_no_run",
        inference_substrate_class="no_model_load" if proof["rows"] else "blocked_no_run",
        inference_substrate_details=dict(
            compute="installed exact CPU record joins", model_invoked=False
        ),
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=[
            dict(
                phase="installed_size_qualification_and_roster",
                start_s=0,
                end_s=(ended - started) / 1e9,
            )
        ],
    )
    sources = (
        OWNED
        + TESTS
        + historical.CLOSURE
        + [
            "pyproject.toml",
            "python/carnot/__init__.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/experiment_template.py",
        ]
    )
    inventory = [
        "python/carnot/verify/indexed_witness.py",
        "python/carnot/verify/_indexed_witness_core.py",
        "python/carnot/verify/_indexed_witness_meter.py",
        "python/carnot/verify/__init__.py",
        "python/carnot/reporting/installed_indexed_runner_8858.py",
        "python/carnot/reporting/installed_cost_8861.py",
        "python/carnot/reporting/installed_cost_runner_8861.py",
        "scripts/experiments/installed_indexed_probe_8858.py",
        "scripts/experiments/installed_cost_worker_8861.py",
        "tests/python/test_installed_indexed_verifier_8858.py",
        "tests/python/test_installed_index_cost_8861.py",
        "pyproject.toml",
    ]
    return dict(
        proof,
        rows=rows,
        preconditions_checked=pre,
        internal_errors=errors,
        private_unit_limit=limit,
        cost_roster=reference(raw / "cost_roster.json"),
        current_work_receipt=receipt,
        code_config_hashes=[reference(ROOT / p) for p in dict.fromkeys(sources)],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
        caller_inventory=[reference(ROOT / p) for p in dict.fromkeys(inventory + OWNED + TESTS)],
    )


def validate(raw: Path, private: Path) -> Json:
    """Use actual unit/child coverage and the unchanged private authority/audit consumers."""
    validator = namespace(shared, "size8871_validation")
    validator.e = SimpleNamespace(OWNED=OWNED, TESTS=TESTS, ROOT=ROOT)
    validator.original = SimpleNamespace(
        CONSUMERS=[
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
            "tests/python/test_installed_indexed_verifier_8858.py",
            "tests/python/test_installed_index_cost_8861.py",
        ]
    )
    validator.child = stream
    return dict(validator.validate(raw, private))


def artifact(work: Json, validation: Json, date: str, work_ref: Json, validation_ref: Json) -> Json:
    """Release correctness only after all frozen installed controls and owned checks pass."""
    pre, rows, receipt = work["preconditions_checked"], work["rows"], work["current_work_receipt"]
    controls = work["controls"] + work["installed_import_rows"]
    gates = dict(
        authorities=pre["external_ready"],
        resources=pre["resources"]["passed"],
        exact_installed_parity=bool(controls) and all(c["passed"] for c in controls),
        complete_roster=work["private_unit_limit"] is None
        and len(rows) == 1760
        and len({(r["unit_id"], r["arm"]) for r in rows}) == 1760,
        installation=bool(work["package_hash"])
        and bool(work["receipts"])
        and all(c["passed"] for c in work["receipts"]),
        owned_validation=qualified(validation),
        owned_execution=not work["internal_errors"],
    )
    ready = int(all(gates.values()))
    verdict = (
        "blocked" if not pre["external_ready"] else "circular_positive" if ready else "disqualified"
    )
    value = dict(
        experiment_id=8871,
        task_id=TASK,
        milestone="2026.11.759",
        run_date=date,
        honest_verdict="complete_" + verdict + "_size_aware_verifier",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="constructed signed-record exact oracle; no natural language, speed or Rust claim",
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
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
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
        validation_receipts=validation["receipts"] + work["receipts"],
        owned_statement_coverage=validation["coverage"],
        size_dispatch_ready_score=ready,
        caller_inventory=work["caller_inventory"],
        dispatch_rule=dict(
            threshold=16,
            at_most="direct_immutable_facts",
            above="existing_index",
            frozen_before_timing=True,
            default_unchanged=True,
        ),
        semantic_parity_rows=work["semantic_parity_rows"],
        operation_accounting_rows=[
            dict(
                unit_id=r["unit_id"],
                arm=r["arm"],
                primitive_operations=r["primitive_operations"],
                operation_counts=r["operation_counts"],
                boundary_operation_counts=r.get("boundary_operation_counts"),
                complete_primitive_operations=r.get("complete_primitive_operations"),
            )
            for r in rows
        ],
        boundary_size_rows=work["boundary_size_rows"],
        ownership_rows=work["ownership_rows"],
        installed_import_rows=work["installed_import_rows"],
        package_hash=work["package_hash"],
        cost_roster=work["cost_roster"],
        current_work_receipt=receipt,
        work_reference=work_ref,
        validation_reference=validation_ref,
        repository_health=work.get("repository_health"),
        methodology="Freeze threshold16 before separate pilot worlds; authenticate Exp8858/8861; install private wheel without dependencies; compare full exact witness multisets with indexed and independent scan references. Retain all180 ordinary and40 wide-join clusters and finite-budget censors. Times are descriptive, with no benefit gate or speed claim.",
    )
    value["field_principles"] = {
        k: "Bind current installed exact-oracle correctness, complete scheduled denominators and actual paid work; administrative hashes do not establish benefit or generalization."
        for k in value
    }
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Authenticate sealed bytes and re-execute installed decisions to reject rehashed semantic changes."""
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
        checker = namespace(ordered.checker, "size8871_coverage_replay")
        checker.m = SimpleNamespace(ROOT=ROOT, OWNED=OWNED)
        if value["required_checks_passed"] and not checker.validation_evidence(validation):
            return False
        if value != artifact(
            work,
            validation,
            value["run_date"],
            value["work_reference"],
            value["validation_reference"],
        ):
            return False
        if work["package_hash"]:
            worlds = load(work["cost_roster"])["worlds"]
            if worlds != freeze()[: work["private_unit_limit"]]:
                return False
            with TemporaryDirectory(prefix="exp8871-replay-") as directory:
                observed = installed(Path(directory), worlds)
            keys = (
                "unit_id",
                "cluster_id",
                "arm",
                "decision",
                "witness",
                "supporting_bindings",
                "budget_exhausted",
                "primitive_operations",
                "operation_counts",
                "status",
                "dispatch_branch",
            )
            if len(observed["rows"]) != len(work["rows"]) or any(
                any(a.get(k) != b.get(k) for k in keys)
                for a, b in zip(observed["rows"], work["rows"], strict=True)
            ):
                return False
            if any(
                observed[k] != work[k]
                for k in [
                    "controls",
                    "semantic_parity_rows",
                    "boundary_size_rows",
                    "ownership_rows",
                ]
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Reuse unchanged prospective scanners and real outside-checkout CLI replay."""
    checker = namespace(ordered, "size8871_terminal")
    checker.m = SimpleNamespace(ROOT=ROOT, CLI=CLI, progress=progress)
    return dict(checker.terminal(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Keep checkpoints private and publish only a terminal artifact with bound checks."""
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
    health = Path("/tmp/exp8871-health")
    if (health / "exit_code").is_file():
        shutil.copytree(health, raw / "repository_health")
        work["repository_health"] = dict(
            exit_code=int((health / "exit_code").read_text()),
            scope="unrelated_global_health",
            logs=[reference(p) for p in sorted((raw / "repository_health").iterdir())],
        )
    atomic_json(raw / "work.json", work)
    progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8871-validation-") as directory:
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

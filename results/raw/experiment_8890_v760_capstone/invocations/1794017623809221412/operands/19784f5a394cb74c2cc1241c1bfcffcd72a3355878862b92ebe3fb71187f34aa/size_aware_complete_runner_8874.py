"""REQ-REPORT-8874: preserve installed parent clocks and publish checked evidence."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any, Callable

from carnot.reporting import size_aware_complete_cost_8874 as m
from carnot.reporting import size_aware_verifier_8871 as qualification
from carnot.reporting import installed_cost_runner_8861 as original
from carnot.reporting import evidence_evaluation_runner_8779 as shared
from carnot.reporting import ordered_evidence_runner_8831 as ordered
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_methods_runner_8829 import qualified
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
HISTORY = qualification.HISTORY + [
    (
        m.ROOT / "results/experiment_8871_v759_size_aware_verifier.json",
        "sha256:079a01a6b8a926a65ffa2e43db3b76c32de507adf8ab7113e1a3931fc3c5b176",
        "size_dispatch_ready_score",
    )
]


def preconditions(raw: Path) -> Json:
    """Require original authority bytes and wheel custody before any measurement."""
    checker = namespace(qualification, "cost8874_preconditions")
    checker.HISTORY = HISTORY
    pre = dict(checker.preconditions(raw))
    pre["package_hash"] = None
    if pre["external_ready"]:
        package = load(reference(HISTORY[-1][0]))["package_hash"]
        if sha256_file(Path(package["path"])) != package["sha256"]:
            pre["external_ready"] = False
            pre["diagnostics"].append(
                dict(
                    check="rejected_producer",
                    upstream="exp8871",
                    artifact_field="package_hash",
                    expected=package["sha256"],
                    observed=sha256_file(Path(package["path"])),
                    path=package["path"],
                    sha256=package["sha256"],
                )
            )
        else:
            pre["package_hash"] = package
            pre["references"].append(package)
    return pre


def process(
    python: Path,
    baseline: Path,
    worlds: list[Json],
    repeats: list[int],
    boundary: str,
    private: Path,
    raw: Path,
    deadline_at: float,
) -> Json:
    """A cold parent pays encoding through child exit; resident rows pay each full request."""
    name = f"{boundary}-{worlds[0]['requests'][0]['unit_id'].replace('/', '_')}-{worlds[0]['arm_order'][0]}-{repeats[0]}"
    m.progress("before_" + name)
    start = time.monotonic_ns()
    inputs = [
        dict(w, requests=[{k: v for k, v in q.items() if k != "truth"} for q in w["requests"]])
        for w in worlds
    ]
    encoded = json.dumps(
        dict(worlds=inputs, repeats=repeats, cap=m.CAP, seed=m.SEED), sort_keys=True
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [str(python), "-I", "-u", str(m.ROOT / m.WORKER), str(baseline)]
    error = None
    observed: Json = dict(rows=[], installed_imports=[])
    stderr = ""
    try:
        deadline = min(deadline_at, time.monotonic() + (30 if boundary == "cold" else 600))
        if deadline <= time.monotonic():
            raise subprocess.TimeoutExpired(command, 0)
        child = subprocess.Popen(
            command,
            cwd=private,
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pending: str | None = encoded
        forwarded = 0
        while True:
            try:
                stdout, stderr = child.communicate(
                    pending, timeout=max(0.001, min(30, deadline - time.monotonic()))
                )
                break
            except subprocess.TimeoutExpired as waiting:
                pending = None
                partial = (waiting.stderr or b"").decode()
                print(partial[forwarded:], end="", flush=True)
                forwarded = len(partial)
                m.progress("waiting_" + name)
                if time.monotonic() >= deadline:
                    child.kill()
                    child.communicate()
                    raise
        print(stderr[forwarded:], end="", flush=True)
        if child.returncode:
            raise ValueError("child_exit:" + str(child.returncode))
        observed = json.loads(stdout)
        expected = {
            (q["unit_id"], a, r)
            for w in worlds
            for q in w["requests"]
            for a in w["arm_order"]
            for r in repeats
        }
        if (
            len(observed["rows"]) != len(expected)
            or {(q["unit_id"], q["arm"], q["repeat"]) for q in observed["rows"]} != expected
        ):
            raise ValueError("child_roster")
        del child, stdout, pending
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as failure:
        error = str(failure)
        observed = dict(
            rows=[
                dict(
                    unit_id=q["unit_id"],
                    cluster_id=w["cluster_id"],
                    arm=a,
                    repeat=r,
                    size=w["size"],
                    population=w["population"],
                    seed=m.SEED,
                    denominator=1,
                    metric=0,
                    decision="insufficient",
                    witness=None,
                    status="failed",
                    missing_reason=error,
                    primitive_operations=0,
                    operation_counts={},
                    complete_wall_time_s=0,
                )
                for w in worlds
                for q in w["requests"]
                for a in w["arm_order"]
                for r in repeats
            ],
            installed_imports=[],
        )
    del encoded, inputs
    end = time.monotonic_ns()
    labels = {q["unit_id"]: q["truth"] for w in worlds for q in w["requests"]}
    rows = [dict(r, boundary=boundary, truth=labels[r["unit_id"]]) for r in observed["rows"]]
    if boundary == "cold":
        for row in rows:
            row["complete_wall_time_s"] = (end - start) / 1e9
    overhead = 0.0
    if boundary == "resident":
        overhead = max(
            0.0, (end - start) / 1e9 - sum(row["complete_wall_time_s"] for row in rows)
        ) / len(rows)
        for row in rows:
            row["resident_process_overhead_s"] = overhead
            row["complete_wall_time_s"] += overhead
    receipt = dict(
        name=name,
        boundary=boundary,
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        wall_time_s=(end - start) / 1e9,
        passed=error is None,
        missing_reason=error,
        cwd=str(private),
        python=str(python),
        pythonpath_present=False,
        installed_imports=observed["installed_imports"],
        stderr=stderr,
        command=command,
        request_count=len(rows),
        resident_overhead_per_observation_s=overhead,
    )
    result = dict(rows=rows, receipt=receipt)
    atomic_json(raw / (name + ".json"), result)
    m.progress("after_" + name, len(rows), 0)
    return result


def measure(raw: Path, worlds: list[Json], package: Json, *, pilot: bool = True) -> Json:
    """Install once, bound the pilot, then retain every scheduled cold and resident unit."""
    raw.mkdir(parents=True, exist_ok=True)
    wheel = raw / Path(package["path"]).name
    shutil.copy2(package["path"], wheel)
    baseline = raw / "always_joint.py"
    baseline.write_text(m.baseline_source())
    start = time.monotonic()
    receipts, rows, cold, resident = [], [], [], []
    pilot_receipt: Json = {}
    with TemporaryDirectory(prefix="exp8874-installed-") as directory:
        private = Path(directory)
        python = private / "venv/bin/python"
        for name, argv in [
            ("venv", ["uv", "venv", str(private / "venv"), "--python", sys.executable]),
            ("install", ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)]),
        ]:
            m.progress("before_" + name)
            receipts.append(stream(name, argv, raw / "installation", deadline=120))
            m.progress("after_" + name)
            if not receipts[-1]["passed"]:
                raise ValueError("installation_failed")
        installation_s = time.monotonic() - start
        deadline_at = start + 2000
        if pilot:
            m.progress("before_pilot")
            pstart = time.monotonic()
            prows = []
            for world in m.freeze(pilot=True):
                for arm in world["arm_order"]:
                    for req in world["requests"]:
                        selected = dict(world, requests=[req], arm_order=[arm])
                        prows.extend(
                            process(
                                python,
                                baseline,
                                [selected],
                                [0],
                                "cold",
                                private,
                                raw / "pilot",
                                deadline_at,
                            )["rows"]
                        )
            elapsed = time.monotonic() - pstart
            projected = elapsed * len(worlds) / 7 * 3 * 2 + 1200
            pilot_receipt = dict(
                duration_s=elapsed,
                projected_total_s=projected,
                limit_s=4200,
                passed=projected < 4200 and all(r["status"] != "failed" for r in prows),
            )
            atomic_json(raw / "pilot.json", pilot_receipt)
            m.progress("after_pilot", len(prows), 0)
            if not pilot_receipt["passed"]:
                raise ValueError("pilot_deadline_or_execution")
        for i, world in enumerate(worlds):
            for repeat in range(3):
                for arm in world["arm_order"]:
                    for req in world["requests"]:
                        selected = dict(world, requests=[req], arm_order=[arm])
                        result = process(
                            python,
                            baseline,
                            [selected],
                            [repeat],
                            "cold",
                            private,
                            raw / "processes",
                            deadline_at,
                        )
                        rows.extend(result["rows"])
                        cold.append(result["receipt"])
            m.progress("cold_worlds", i + 1, len(worlds) - i - 1)
        result = process(
            python,
            baseline,
            worlds,
            list(range(3)),
            "resident",
            private,
            raw / "processes",
            deadline_at,
        )
        rows.extend(result["rows"])
        resident.append(result["receipt"])
        disposal = time.monotonic()
    installation_s += time.monotonic() - disposal
    pins = {
        Path(ref["path"]).name: ref["sha256"]
        for ref in load(reference(HISTORY[-1][0]))["code_config_hashes"]
        if "/verify/" in ref["path"]
    }
    imports_valid = all(
        c["passed"]
        and len(c["installed_imports"]) == 4
        and all(
            "/site-packages/carnot/verify/" in p["path"]
            and pins.get(Path(p["path"]).name) == p["sha256"]
            and not Path(p["path"]).is_relative_to(m.ROOT)
            for p in c["installed_imports"]
        )
        for c in cold + resident
    )
    return dict(
        rows=rows,
        clocks=cold + resident,
        cold_process_rows=cold,
        resident_rows=resident,
        installation_receipts=receipts,
        installation_s=installation_s,
        package_hashes=[reference(wheel)],
        installed_imports_valid=imports_valid,
        pilot_receipt=pilot_receipt,
    )


def prepare(raw: Path, limit: int | None = None) -> Json:
    """Seal all operands before measurement and keep missing scheduled units explicit."""
    start = time.monotonic_ns()
    m.progress("before_preconditions")
    pre = preconditions(raw / "authorities")
    m.progress("after_preconditions")
    worlds = m.freeze()[:limit]
    atomic_json(
        raw / "frozen_worlds.json",
        dict(
            worlds=worlds,
            seed=m.SEED,
            cap=m.CAP,
            mixture_weights={str(s): 1 / 6 for s in m.SIZES},
            repeats=3,
            index_reuse=False,
        ),
    )
    (raw / "frozen_worlds.json").chmod(0o444)
    measured: Json = dict(
        rows=[],
        clocks=[],
        cold_process_rows=[],
        resident_rows=[],
        installation_receipts=[],
        installation_s=0.0,
        package_hashes=[],
        installed_imports_valid=False,
        pilot_receipt={},
    )
    errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            measured = measure(raw, worlds, pre["package_hash"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
    if not measured["rows"]:
        # Preserve already paid children if installation or the pilot fails late.
        retained = [load(reference(p)) for p in sorted((raw / "processes").glob("*.json"))]
        measured["rows"] = [r for c in retained for r in c["rows"]]
        measured["clocks"] = [c["receipt"] for c in retained]
    observed = {(r["boundary"], r["unit_id"], r["arm"], r["repeat"]) for r in measured["rows"]}
    for world in worlds:
        for boundary in m.BOUNDARIES:
            for repeat in range(3):
                for arm in m.ARMS:
                    for req in world["requests"]:
                        if (boundary, req["unit_id"], arm, repeat) not in observed:
                            measured["rows"].append(
                                dict(
                                    unit_id=req["unit_id"],
                                    cluster_id=world["cluster_id"],
                                    arm=arm,
                                    repeat=repeat,
                                    boundary=boundary,
                                    size=world["size"],
                                    population=world["population"],
                                    seed=m.SEED,
                                    denominator=1,
                                    metric=0,
                                    decision="insufficient",
                                    witness=None,
                                    truth=req["truth"],
                                    status="failed" if errors else "censored",
                                    missing_reason=errors[0]
                                    if errors
                                    else "external_or_resource_gate",
                                    primitive_operations=0,
                                    operation_counts={},
                                    complete_wall_time_s=0.0,
                                )
                            )
    end = time.monotonic_ns()
    substrate = "no_model_load" if measured["clocks"] else "blocked_no_run"
    receipt = build_current_work_receipt(
        run_id=m.TASK + "/" + str(start),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=substrate,
        inference_substrate_class=substrate,
        inference_substrate_details=dict(
            compute="installed Python signed-record verification", model_invoked=False
        ),
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[
            dict(phase="authenticated_complete_measurement", start_s=0, end_s=(end - start) / 1e9)
        ],
    )
    sources = (
        m.OWNED
        + m.TESTS
        + [
            "python/carnot/reporting/size_aware_verifier_8871.py",
            "python/carnot/reporting/installed_cost_8861.py",
            "python/carnot/reporting/installed_cost_runner_8861.py",
            "scripts/experiments/installed_cost_worker_8861.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/experiment_template.py",
        ]
    )
    return dict(
        measured,
        worlds=worlds,
        private_world_limit=limit,
        preconditions_checked=pre,
        internal_errors=errors,
        current_work_receipt=receipt,
        code_config_hashes=[reference(m.ROOT / p) for p in sources],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
    )


def validate(raw: Path, private: Path) -> Json:
    """Retain real unit/child SQLite and the unchanged original private E2E checks."""
    validator = namespace(shared, "cost8874_validation")
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
    """A complete cost study may be ready while either small-request benefit guard fails."""
    rows, pre, receipt = work["rows"], work["preconditions_checked"], work["current_work_receipt"]
    reduction = m.reduce(rows, work["installation_s"]) if work["clocks"] else m.reduce([], 0)
    gates = dict(
        authorities=pre["external_ready"],
        resources=pre["resources"]["passed"],
        full_roster=work["private_world_limit"] is None
        and len(work["worlds"]) == 220
        and len(rows) == 15840,
        all_measurements_present=len(work["clocks"]) == len(work["worlds"]) * 36 + 1
        and all(r["status"] in ("completed", "censored") for r in rows),
        installed_imports=work["installed_imports_valid"],
        owned_execution=not work["internal_errors"],
        exact_completed_parity=reduction["quality_gates"].get("exact_completed_parity", False),
        owned_validation=qualified(validation),
    )
    ready = int(all(gates.values()))
    benefit = int(ready and reduction["size_cost_benefit_score"])
    verdict = (
        "blocked"
        if not pre["external_ready"]
        else "circular_positive"
        if benefit
        else "null"
        if ready
        else "disqualified"
    )
    value = dict(
        experiment_id=8874,
        task_id=m.TASK,
        milestone="2026.11.759",
        run_date=date,
        honest_verdict="complete_" + verdict + "_size_aware_complete_cost",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="Constructed signed records and exact labels; no language, independent generalization or Rust10x evidence.",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=receipt["phase_spans"],
        preconditions_checked=pre,
        random_seed=m.SEED,
        source_artifact_hashes=pre["references"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"]
        + validation["raw_references"]
        + [work_ref, validation_ref],
        rows=rows,
        intended_count=len(work["worlds"]) * 72,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=len(work["worlds"]),
        gate_check_summary=pre["diagnostics"]
        + [
            dict(
                check="owned_gate_failure",
                upstream=m.TASK,
                artifact_field=k,
                expected=True,
                observed=v,
                path=work_ref["path"],
                sha256=work_ref["sha256"],
            )
            for k, v in gates.items()
            if not v
        ],
        acceptance_gates=gates,
        required_checks_passed=qualified(validation),
        validation_receipts=validation["receipts"] + work["installation_receipts"],
        owned_statement_coverage=validation["coverage"],
        size_cost_ready_score=ready,
        size_cost_benefit_score=benefit,
        complete_request_rows=rows,
        operation_rows=[
            {
                k: r[k]
                for k in (
                    "unit_id",
                    "cluster_id",
                    "boundary",
                    "repeat",
                    "arm",
                    "primitive_operations",
                    "operation_counts",
                    "denominator",
                    "status",
                    "missing_reason",
                )
            }
            for r in rows
        ],
        censoring_rows=[r for r in rows if r["status"] != "completed"],
        per_world_results=reduction["per_world_results"],
        paired_intervals=reduction["paired_intervals"],
        quality_gates=reduction["quality_gates"],
        benefit_gates=reduction["benefit_gates"],
        cold_speed_claim=bool(ready and reduction["cold_benefit_score"]),
        cold_process_rows=work["cold_process_rows"],
        resident_rows=work["resident_rows"],
        package_hashes=work["package_hashes"],
        installation_accounting=dict(
            installation_s=work["installation_s"],
            full_cost_per_arm=True,
            deployment_worlds=180,
            allocation="equal_per_world_added_to_each_boundary",
            mixture_weights={str(s): 1 / 6 for s in m.SIZES},
            stress_separate=True,
            index_reuse=False,
            repeats=3,
        ),
        pilot_receipt=work["pilot_receipt"],
        dispatch_rule=dict(threshold=16, frozen=True, opt_in=True),
        size17_role="fixed_dispatch_boundary_stress_no_tuning",
        prd_rust10x_score=0,
        current_work_receipt=receipt,
        work_reference=work_ref,
        validation_reference=validation_ref,
        repository_health=work.get(
            "repository_health", dict(status="not_run", scope="unrelated_global_health")
        ),
        methodology="Three observations averaged within each request; four requests per independently paired world. Cold parent clocks pay serialization, spawn/import, validation, solving, response and process disposal. Resident clocks pay request serialization through response disposal plus an equal allocation of measured parent startup/import, transport and exit overhead. Raw inner and parent clocks remain separate. Full installation and environment disposal are charged equally per arm across180 ordinary worlds. The six ordinary sizes have weights1/6;40 wide-join stress worlds remain separate. Whole-world stratified bootstrap uses160 fixed endpoints,10000 draws and Bonferroni family error.01. Failed units and finite-cap censors remain scheduled. Threshold16 is fixed; Python cannot establish Rust10x.",
    )
    value["field_principles"] = {
        k: "Retain paid work and immutable inputs with complete independent denominators; administrative hashes are not scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        size_cost_ready_score="Complete authenticated costs and passing owned checks establish readiness independently of benefit.",
        size_cost_benefit_score="Resident complete mixture lower>1 and both small-size regression upper<=.05 plus exact quality gates are mandatory.",
        cold_speed_claim="Cold benefit requires its own simultaneous guard checks.",
        installation_accounting="Measured deployment setup and disposal are separately charged in full to each arm.",
        size17_role="Dispatch-boundary stress is reported without threshold selection.",
        prd_rust10x_score="Python-only entry points cannot meet the native Rust PRD requirement.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path, *, execute: bool = True) -> bool:
    """Rebuild the terminal claim and re-execute installed semantics without reusing clocks."""
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
        frozen = [
            ref
            for ref in work["raw_shard_hashes"]
            if Path(ref["path"]).name == "frozen_worlds.json"
        ]
        if (
            len(frozen) != 1
            or work["worlds"] != load(frozen[0])["worlds"]
            or work["worlds"] != m.freeze()[: work["private_world_limit"]]
        ):
            return False
        operands = [
            load(ref)
            for ref in work["raw_shard_hashes"]
            if Path(ref["path"]).parent.name == "processes"
        ]
        if work["clocks"]:
            order = lambda row: (row["boundary"], row["unit_id"], row["arm"], row["repeat"])
            if sorted(work["rows"], key=order) != sorted(
                [r for p in operands for r in p["rows"]], key=order
            ):
                return False
            if sorted(work["clocks"], key=lambda c: c["name"]) != sorted(
                [p["receipt"] for p in operands], key=lambda c: c["name"]
            ):
                return False
            for operand in operands:
                c = operand["receipt"]
                if c["wall_time_s"] != (c["ended_monotonic_ns"] - c["started_monotonic_ns"]) / 1e9:
                    return False
                for row in operand["rows"]:
                    if row["complete_wall_time_s"] != (
                        c["wall_time_s"]
                        if row["boundary"] == "cold"
                        else (row["ended_monotonic_ns"] - row["started_monotonic_ns"]) / 1e9
                        + row["resident_process_overhead_s"]
                    ):
                        return False
        if value["required_checks_passed"]:
            checker = namespace(ordered.checker, "cost8874_coverage_replay")
            checker.m = SimpleNamespace(ROOT=m.ROOT, OWNED=m.OWNED)
            if not checker.validation_evidence(validation):
                return False
        if value != artifact(
            work,
            validation,
            value["run_date"],
            value["work_reference"],
            value["validation_reference"],
        ):
            return False
        if execute and work["package_hashes"]:
            with TemporaryDirectory(prefix="exp8874-replay-") as directory:
                raw = Path(directory)
                wheel = work["package_hashes"][0]
                python = raw / "venv/bin/python"
                for name, argv in [
                    ("venv", ["uv", "venv", str(raw / "venv"), "--python", sys.executable]),
                    (
                        "install",
                        [
                            "uv",
                            "pip",
                            "install",
                            "--python",
                            str(python),
                            "--no-deps",
                            wheel["path"],
                        ],
                    ),
                ]:
                    if not stream(name, argv, raw / "logs", deadline=120)["passed"]:
                        return False
                baseline = raw / "scan.py"
                baseline.write_text(m.baseline_source())
                repeated = process(
                    python,
                    baseline,
                    work["worlds"],
                    list(range(3)),
                    "resident",
                    raw,
                    raw / "replay",
                    time.monotonic() + 300,
                )
            keys = (
                "unit_id",
                "cluster_id",
                "arm",
                "repeat",
                "decision",
                "witness",
                "status",
                "primitive_operations",
                "operation_counts",
            )
            stable: Callable[[Json], Json] = lambda r: {k: r[k] for k in keys}
            if [stable(r) for r in repeated["rows"]] != [
                stable(r) for r in work["rows"] if r["boundary"] == "resident"
            ]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Use unchanged prospective scanners and a real outside-checkout replay process."""
    checker = namespace(ordered, "cost8874_terminal")
    checker.m = SimpleNamespace(ROOT=m.ROOT, CLI=m.CLI, progress=m.progress)
    return dict(checker.terminal(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Keep subsets private and publish one immutable checked terminal invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261106"], default="20261106")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--private-world-limit", type=int)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private output must be outside checkout")
    if args.private_world_limit is not None and (
        not args.private_output or args.private_world_limit < 1
    ):
        parser.error("subset requires private output and a positive count")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    work = prepare(raw, args.private_world_limit)
    health = Path("/tmp/exp8874-health")
    if (health / "receipt.json").is_file():
        shutil.copytree(health, raw / "repository_health")
        work["repository_health"] = dict(
            load(reference(raw / "repository_health/receipt.json")),
            logs=[reference(p) for p in sorted((raw / "repository_health").iterdir())],
        )
    atomic_json(raw / "work.json", work)
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8874-validation-") as directory:
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
    for operand in raw.rglob("*"):
        if operand.is_file():
            operand.chmod(0o444)
    m.progress("complete_" + value["verdict_class"], value["completed_count"], 0)
    return 0

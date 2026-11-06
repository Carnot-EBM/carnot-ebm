"""REQ-REPORT-8861: authenticate installed bytes, execute complete boundaries and publish evidence."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

from carnot.reporting import installed_cost_8861 as m
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
UPSTREAM = m.ROOT / "results/experiment_8858_v758_installed_indexed_verifier.json"
PIN = "sha256:28ef6d677e2684ad69d8916f4d4e2ec24dcb453b9349bf80de5ada2e70af65e6"
COST = m.ROOT / "results/experiment_8847_v757_indexed_cost.json"
COST_PIN = "sha256:47548df6e4e4437f12f353298461b28be99129ee7759a55da779fb93a55f9a3e"


def preconditions(raw: Path) -> Json:
    """Authenticate original producer and operand bytes before treating readiness as evidence."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    refs, diagnostics, package = [], [], None
    for path, pin, field in [
        (UPSTREAM, PIN, "installed_index_ready_score"),
        (COST, COST_PIN, "indexed_cost_ready_score"),
    ]:
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
                artifact_field="sha256",
                expected=pin,
                observed=sha256_file(path),
                sha256=sha256_file(path),
            )
            if sha256_file(path) != pin or not value["required_checks_passed"]:
                raise ValueError("producer_identity")
            operands = (
                [reference(path)]
                + value["source_artifact_hashes"]
                + value["code_config_hashes"]
                + value["raw_shard_hashes"]
            )
            if field == "installed_index_ready_score":
                package = value["package_hash"]
                operands.append(package)
            for ref in operands:
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
                    error=str(error),
                    check="missing_file" if isinstance(error, OSError) else diagnostic["check"],
                )
            )
    with TemporaryDirectory(prefix="exp8861-preflight-") as directory:
        probe = Path(directory) / "writable"
        probe.write_text("private writable")
        resources = dict(
            private_writable=probe.read_text() == "private writable",
            free_disk_bytes=shutil.disk_usage(raw).free,
            tools=all(
                (m.ROOT / ".venv/bin" / tool).is_file()
                for tool in ("python", "pytest", "ruff", "mypy")
            ),
            uv=shutil.which("uv") is not None,
            total_deadline_s=4200,
            measurement_deadline_s=900,
            reserved_validation_s=1800,
        )
    resources["passed"] = bool(
        resources["private_writable"]
        and resources["tools"]
        and resources["uv"]
        and resources["free_disk_bytes"] > 100_000_000
    )
    return dict(
        external_ready=not diagnostics,
        diagnostics=diagnostics,
        references=list({r["path"]: r for r in refs}.values()),
        resources=resources,
        package_hash=package,
    )


def process(
    python: Path,
    baseline: Path,
    worlds: list[Json],
    arm: str,
    boundary: str,
    private: Path,
    raw: Path,
    deadline_at: float = float("inf"),
) -> Json:
    """The parent pays serialization through child exit; inner clocks label the resident boundary."""
    name = boundary + "-" + worlds[0]["cluster_id"].replace("/", "_") + "-" + arm
    selected_arms = m.ARMS if arm == "paired" else (arm,)
    m.progress("before_" + name, 0, len(worlds) * 4 * len(selected_arms))
    start = time.monotonic_ns()
    inputs = [
        dict(
            w,
            requests=[
                {k: v for k, v in q.items() if k not in ("truth", "condition")}
                for q in w["requests"]
            ],
        )
        for w in worlds
    ]
    encoded = json.dumps(dict(worlds=inputs, cap=m.CAP, seed=m.SEED), sort_keys=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    error, stderr = None, ""
    observed: Json = dict(rows=[], clocks=[], installed_imports=[])
    try:
        deadline = min(time.monotonic() + (30 if boundary == "cold" else 300), deadline_at)
        if deadline <= time.monotonic():
            raise subprocess.TimeoutExpired("measurement_deadline", 0)
        child = subprocess.Popen(
            [str(python), "-I", "-u", str(m.ROOT / m.WORKER), str(baseline), arm],
            cwd=private,
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pending_input: str | None = encoded
        forwarded = 0
        while True:
            try:
                stdout, stderr = child.communicate(
                    pending_input, timeout=max(0.001, min(30, deadline - time.monotonic()))
                )
                break
            except subprocess.TimeoutExpired as waiting:
                pending_input = None
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
            raise subprocess.CalledProcessError(child.returncode, child.args)
        candidate = json.loads(stdout)
        if (
            not isinstance(candidate, dict)
            or {(q["unit_id"], q["arm"]) for q in candidate.get("rows", [])}
            != {
                (q["unit_id"], selected)
                for w in worlds
                for q in w["requests"]
                for selected in selected_arms
            }
            or len(candidate.get("rows", [])) != len(worlds) * 4 * len(selected_arms)
            or len(candidate.get("clocks", [])) != len(worlds) * len(selected_arms)
        ):
            raise ValueError("child_roster")
        observed = candidate
        del stdout, pending_input
        del child
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as failure:
        error = str(failure)
    del encoded, inputs
    end = time.monotonic_ns()
    for world in worlds:
        for selected in selected_arms:
            for q in world["requests"]:
                if error:
                    observed["rows"].append(
                        dict(
                            unit_id=q["unit_id"],
                            cluster_id=world["cluster_id"],
                            arm=selected,
                            size=world["size"],
                            population=world["population"],
                            denominator=1,
                            metric=0,
                            seed=m.SEED,
                            decision="insufficient",
                            witness=None,
                            status="failed",
                            missing_reason=error,
                            primitive_operations=0,
                            operation_counts={},
                        )
                    )
    labels = {q["unit_id"]: q["truth"] for w in worlds for q in w["requests"]}
    row_boundary = "resident_inner" if boundary == "resident" else boundary
    rows = [dict(r, boundary=row_boundary, truth=labels[r["unit_id"]]) for r in observed["rows"]]
    receipt = dict(
        name=name,
        boundary="resident_full_corpus" if boundary == "resident" else boundary,
        arm=arm,
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        wall_time_s=(end - start) / 1e9,
        duration_s=(end - start) / 1e9,
        passed=error is None,
        missing_reason=error,
        world_count=len(worlds),
        request_count=len(worlds) * 4 * len(selected_arms),
        cwd=str(private),
        python=str(python),
        pythonpath_present=False,
        stderr=stderr,
        installed_imports=observed["installed_imports"],
    )
    clocks = [dict(c, boundary=row_boundary) for c in observed["clocks"]]
    if boundary == "cold":
        clocks = [
            dict(
                receipt,
                cluster_id=worlds[0]["cluster_id"],
                size=worlds[0]["size"],
                population=worlds[0]["population"],
            )
        ]
    elif error:
        clocks = [
            dict(
                cluster_id=w["cluster_id"],
                size=w["size"],
                population=w["population"],
                boundary=row_boundary,
                arm=selected,
                wall_time_s=(end - start) / 1e9 / len(worlds) / len(selected_arms),
            )
            for w in worlds
            for selected in selected_arms
        ]
    world_clocks = {(c["cluster_id"], c["arm"]): c["wall_time_s"] for c in clocks}
    rows = [
        dict(row, world_wall_time_s=world_clocks[(row["cluster_id"], row["arm"])]) for row in rows
    ]
    result = dict(rows=rows, clocks=clocks, receipt=receipt)
    atomic_json(raw / (name + ".json"), result)
    m.progress("after_" + name, len(rows), 0)
    return result


def measure(raw: Path, worlds: list[Json], package: Json) -> Json:
    """Install the preserved wheel once and declare its full cost for each arm's deployment."""
    installation_start = time.monotonic_ns()
    deadline_at = time.monotonic() + 900
    receipts, rows, clocks, cold, resident = [], [], [], [], []
    wheel = raw / Path(package["path"]).name
    shutil.copy2(package["path"], wheel)
    baseline = raw / "always_joint.py"
    baseline.write_text(m.baseline_source())
    with TemporaryDirectory(prefix="exp8861-installed-") as directory:
        private = Path(directory)
        python = private / "venv/bin/python"
        for name, argv in [
            ("venv", ["uv", "venv", str(private / "venv"), "--python", sys.executable]),
            ("install", ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)]),
        ]:
            m.progress("before_" + name)
            receipt = stream(name, argv, raw / "installation", deadline=120)
            receipts.append(receipt)
            m.progress("after_" + name)
            if not receipt["passed"]:
                raise ValueError("installation_failed")
        setup_s = (time.monotonic_ns() - installation_start) / 1e9
        for i, world in enumerate(worlds):
            for arm in world["arm_order"]:
                result = process(
                    python, baseline, [world], arm, "cold", private, raw / "processes", deadline_at
                )
                rows.extend(result["rows"])
                clocks.extend(result["clocks"])
                cold.append(result["receipt"])
            m.progress("cold_worlds", i + 1, len(worlds) - i - 1)
        result = process(
            python, baseline, worlds, "paired", "resident", private, raw / "processes", deadline_at
        )
        rows.extend(result["rows"])
        clocks.extend(result["clocks"])
        resident.append(result["receipt"])
        disposal_start = time.monotonic_ns()
    installation_s = setup_s + (time.monotonic_ns() - disposal_start) / 1e9
    installed_pins = {
        Path(r["path"]).name: r["sha256"]
        for r in load(reference(UPSTREAM))["code_config_hashes"]
        if "/verify/" in r["path"]
    }
    imports_valid = all(
        r["passed"]
        and len(r["installed_imports"]) == 3
        and all(
            "/site-packages/carnot/verify/" in p["path"]
            and not Path(p["path"]).is_relative_to(m.ROOT)
            and installed_pins[Path(p["path"]).name] == p["sha256"]
            for p in r["installed_imports"]
        )
        for r in cold + resident
    )
    return dict(
        rows=rows,
        clocks=clocks,
        cold_process_rows=cold,
        resident_rows=resident,
        installation_receipts=receipts,
        installation_s=installation_s,
        package_hashes=[reference(wheel)],
        installed_imports_valid=imports_valid,
    )


def prepare(raw: Path, limit: int | None = None) -> Json:
    """Seal every intended world and label before starting any measurement process."""
    start = time.monotonic_ns()
    m.progress("before_preconditions")
    pre = preconditions(raw / "authorities")
    m.progress("after_preconditions")
    worlds = m.freeze()
    atomic_json(
        raw / "frozen_worlds.json",
        dict(
            worlds=worlds,
            seed=m.SEED,
            cap=m.CAP,
            deployment_weights={str(s): 0.25 for s in m.SIZES},
            warmup_requests=0,
            index_reuse=False,
        ),
    )
    (raw / "frozen_worlds.json").chmod(0o444)
    worlds = worlds[:limit]
    measured: Json = dict(
        rows=[],
        clocks=[],
        cold_process_rows=[],
        resident_rows=[],
        installation_receipts=[],
        installation_s=0.0,
        package_hashes=[],
        installed_imports_valid=False,
    )
    errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            measured = measure(raw, worlds, pre["package_hash"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
    if not measured["rows"]:
        measured["rows"] = [
            dict(
                unit_id=q["unit_id"],
                cluster_id=w["cluster_id"],
                size=w["size"],
                population=w["population"],
                arm=arm,
                boundary=boundary,
                seed=m.SEED,
                decision="insufficient",
                witness=None,
                truth=q["truth"],
                metric=0,
                denominator=1,
                status="pending",
                missing_reason="external_precondition"
                if not pre["external_ready"]
                else "owned_measurement_unavailable",
                primitive_operations=0,
                operation_counts={},
            )
            for w in worlds
            for q in w["requests"]
            for boundary in ("cold", "resident")
            for arm in m.ARMS
        ]
    end = time.monotonic_ns()
    substrate = "no_model_load" if measured["clocks"] else "blocked_no_run"
    receipt = build_current_work_receipt(
        run_id=m.TASK + "/" + str(start),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=substrate,
        inference_substrate_details=dict(
            compute="Python structured-record CPU verification", model_invoked=False
        ),
        inference_substrate_class=substrate,
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[
            dict(phase="authenticated_installed_measurement", start_s=0, end_s=(end - start) / 1e9)
        ],
    )
    return dict(
        measured,
        preconditions_checked=pre,
        worlds=worlds,
        private_world_limit=limit,
        internal_errors=errors,
        current_work_receipt=receipt,
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse actual unit/child coverage and the preserved private authority and audit consumers."""
    validator = namespace(shared, "installed8861_validation")
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
    """Complete measurements grant readiness independently of the registered speed benefit."""
    pre, rows, receipt = work["preconditions_checked"], work["rows"], work["current_work_receipt"]
    reduction = (
        m.reduce(rows, work["clocks"], work["installation_s"])
        if work["clocks"]
        else m.reduce([], [], 0)
    )
    gates = dict(
        authorities=pre["external_ready"],
        resources=pre["resources"]["passed"],
        full_roster=work["private_world_limit"] is None
        and len(rows) == 2560
        and len(work["worlds"]) == 160,
        all_measurements_present=len(work["clocks"]) == len(work["worlds"]) * 4
        and all(r["status"] in ("completed", "censored") for r in rows),
        installed_imports=work["installed_imports_valid"],
        owned_execution=not work["internal_errors"],
        exact_completed_parity=reduction["quality_gates"].get("exact_completed_parity", False),
        owned_validation=qualified(validation),
    )
    ready = int(all(gates.values()))
    benefit = int(ready and reduction["installed_cost_benefit_score"])
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
        experiment_id=8861,
        task_id=m.TASK,
        milestone="2026.11.758",
        run_date=date,
        honest_verdict="complete_blocked_" + pre["diagnostics"][0]["check"]
        if verdict == "blocked"
        else "complete_" + verdict + "_installed_index_cost",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="Constructed signed records with exact construction labels; no natural-language or native speed claim.",
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
        intended_count=len(work["worlds"]) * 16,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        pending_count=sum(r["status"] == "pending" for r in rows),
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
        installed_cost_ready_score=ready,
        installed_cost_benefit_score=benefit,
        complete_request_rows=[r for r in rows if r["status"] == "completed"],
        operation_rows=[
            {
                k: r[k]
                for k in (
                    "unit_id",
                    "cluster_id",
                    "boundary",
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
        cold_process_rows=work["cold_process_rows"],
        resident_rows=work["resident_rows"],
        package_hashes=work["package_hashes"],
        installation_s=work["installation_s"],
        installation_accounting=dict(
            full_cost_per_arm=True,
            deployment_worlds=120,
            allocation="equal_per_world",
            warmup_requests=0,
            index_reuse=False,
        ),
        current_work_receipt=receipt,
        work_reference=work_ref,
        validation_reference=validation_ref,
        repository_health=work.get(
            "repository_health", dict(status="not_run", scope="global_health_only")
        ),
        methodology="Paired random arm order within each independently constructed world. Cold parent clocks include transport, startup, validation, request construction, all four proof responses, disposal and exit. Resident parent clocks include the complete corpus; inner world clocks have a separate label. Installation is paid in full per arm for the frozen 120-world deployment. All censors remain in denominators. Simultaneous whole-world intervals cover both boundaries, four sizes and the equal-size mixture. Adversarial worlds are a separate declared population. Both arms are Python; the native PRD target is untested.",
    )
    value["field_principles"] = {
        k: "Retain measured work, immutable operands and complete denominators; constructed oracle evidence cannot establish language generalization."
        for k in value
    }
    value["field_principles"].update(
        dict(
            pending_count="Unfinished owned requests remain in the intended roster.",
            quality_gates="Completed proof parity, decisive coverage and false accepts use every request, including censors.",
            benefit_gates="A simultaneous lower mixture ratio above one and size16 upper latency regression at most5 percent are separate from readiness.",
            installation_s="Parent-measured setup and disposal cost is paid in full for each arm of the declared deployment.",
            installation_accounting="The fixed120-world deployment pays equal installation allocation per world; no warmup or reused index is hidden.",
            current_work_receipt="Owned process clocks and an empty model-event ledger describe only this invocation.",
            work_reference="Exact sidecar bytes bind all measurement operands.",
            validation_reference="Exact sidecar bytes bind executed tests, retained logs and actual statement coverage.",
            repository_health="The full Python suite is global health; it does not replace scoped owned validation.",
            methodology="The declared cold and resident boundaries limit the interpretation of all cost ratios.",
        )
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path, *, execute: bool = True) -> bool:
    """Recompute retained clocks and rerun installed decisions; changing a hash cannot repair a claim."""
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
            or work["worlds"] != load(frozen[0])["worlds"][: work["private_world_limit"]]
        ):
            return False
        operands = [
            load(ref)
            for ref in work["raw_shard_hashes"]
            if Path(ref["path"]).parent.name == "processes"
        ]
        if work["clocks"]:
            if len(operands) != len(work["worlds"]) * 2 + 1:
                return False
            for field in ("rows", "clocks"):
                order = lambda row: (
                    row["boundary"],
                    row["cluster_id"],
                    row["arm"],
                    row.get("unit_id", ""),
                )
                if sorted(work[field], key=order) != sorted(
                    [row for operand in operands for row in operand[field]], key=order
                ):
                    return False
            if sorted(
                work["cold_process_rows"] + work["resident_rows"], key=lambda row: row["name"]
            ) != sorted([operand["receipt"] for operand in operands], key=lambda row: row["name"]):
                return False
        if value["required_checks_passed"]:
            checker = namespace(ordered.checker, "installed8861_coverage_replay")
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
            with TemporaryDirectory(prefix="exp8861-replay-") as directory:
                repeated = measure(Path(directory), work["worlds"], work["package_hashes"][0])
            keys = (
                "unit_id",
                "cluster_id",
                "boundary",
                "arm",
                "decision",
                "witness",
                "status",
                "primitive_operations",
                "operation_counts",
            )

            def stable(row: Json) -> Json:
                return {key: row[key] for key in keys}

            if [stable(r) for r in repeated["rows"]] != [stable(r) for r in work["rows"]]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Run the unchanged prospective scanners and outside-checkout CLI replay on candidate bytes."""
    checker = namespace(ordered, "installed8861_terminal")
    checker.m = SimpleNamespace(ROOT=m.ROOT, CLI=m.CLI, progress=m.progress)
    return dict(checker.terminal(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Keep subsets private; publish only one checked terminal identity."""
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
    work["raw_shard_hashes"] = [reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    health = m.ROOT / "results/raw" / m.NAME / "repository_health/receipt.json"
    work["repository_health"] = (
        load(reference(health))
        if health.is_file()
        else dict(status="not_run", scope="global_health_only")
    )
    atomic_json(raw / "work.json", work)
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8861-validation-") as directory:
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
    publication = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    atomic_json(raw / "publication_receipt.json", publication)
    for operand in raw.rglob("*"):
        if operand.is_file():
            operand.chmod(0o444)
    m.progress(
        "complete_" + value["verdict_class"], value["completed_count"], value["pending_count"]
    )
    return 0

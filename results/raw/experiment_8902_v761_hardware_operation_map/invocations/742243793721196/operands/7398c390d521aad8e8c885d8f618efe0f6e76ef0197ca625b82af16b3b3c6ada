"""REQ-REPORT-8882: authenticate, diagnose and publish one immutable request-cost study."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from zipfile import ZipFile
from types import SimpleNamespace
from typing import Any, Callable

from carnot.reporting import request_cost_attribution_8882 as m
from carnot.reporting import size_aware_complete_runner_8874 as previous
from carnot.reporting import size_aware_complete_cost_8874 as design
from carnot.reporting import size_aware_verifier_8871 as authority
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
HISTORY = previous.HISTORY + [
    (
        m.ROOT / "results/experiment_8874_v759_size_aware_complete_cost.json",
        "sha256:2aaaa2de3a2f2d4f4873c51623d9031f4bcdbd63a7eb83b136102170393b5f8c",
        "size_cost_ready_score",
    )
]
START = time.monotonic()


def preconditions(raw: Path) -> Json:
    """Reject changed original evidence before installing or measuring the retained wheel."""
    checker = namespace(authority, "attribution8882_authority")
    checker.HISTORY = HISTORY
    pre = dict(checker.preconditions(raw))
    pre["package_hash"] = (
        load(reference(previous.HISTORY[-1][0]))["package_hash"] if pre["external_ready"] else None
    )
    return pre


def process(
    python: Path,
    baseline: Path,
    worlds: list[Json],
    repeats: list[int],
    boundary: str,
    private: Path,
    raw: Path,
    deadline: float,
) -> Json:
    """Reuse the qualified parent supervisor and retain its unmodified complete clocks."""
    runner = namespace(previous, "attribution8882_parent")
    runner.m = m
    result = dict(
        runner.process(python, baseline, worlds, repeats, boundary, private, raw, deadline)
    )
    for row in result["rows"]:
        row["instrumented"] = worlds[0]["instrumented"]
    atomic_json(raw / (result["receipt"]["name"] + ".json"), result)
    return result


def measure(raw: Path, worlds: list[Json], package: Json) -> Json:
    """Install once, pilot the fixed schedule and retain every failed or censored child."""
    wheel = raw / Path(package["path"]).name
    shutil.copy2(package["path"], wheel)
    baseline = raw / "always_joint.py"
    baseline.write_text(m.baseline_source())
    rows, clocks, installations = [], [], []
    with TemporaryDirectory(prefix="exp8882-installed-") as directory:
        private = Path(directory)
        python = private / "venv/bin/python"
        started = time.monotonic()
        for name, argv in [
            ("venv", ["uv", "venv", str(private / "venv"), "--python", sys.executable]),
            ("install", ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)]),
        ]:
            m.progress("before_" + name)
            installations.append(stream(name, argv, raw / "installation", deadline=120))
            m.progress("after_" + name)
            if not installations[-1]["passed"]:
                raise ValueError("installation_failed")
        installation_s = time.monotonic() - started
        deadline = min(START + 2400, time.monotonic() + 1200)
        m.progress("before_pilot")
        pilot_start = time.monotonic()
        pilot_world = deepcopy(design.freeze(pilot=True)[0])
        for observer in pilot_world.get("observer_order", [False, True]):
            selected = dict(
                pilot_world,
                instrumented=observer,
                arm_order=["size_aware"],
                requests=pilot_world["requests"][:1],
            )
            pilot = process(
                python,
                baseline,
                [selected],
                [0],
                "cold",
                private,
                raw / "pilot" / str(observer),
                deadline,
            )
            if not pilot["receipt"]["passed"]:
                raise ValueError("pilot_failed")
        projected = (time.monotonic() - pilot_start) * len(worlds) * 36 + 1800
        pilot_receipt = dict(
            duration_s=time.monotonic() - pilot_start,
            projected_total_s=projected,
            limit_s=4200,
            passed=projected < 4200,
        )
        atomic_json(raw / "pilot_receipt.json", pilot_receipt)
        m.progress("after_pilot")
        if not pilot_receipt["passed"]:
            raise ValueError("pilot_budget")
        total = len(worlds) * 144
        for world in worlds:
            for observer in world["observer_order"]:
                selected = dict(world, instrumented=observer)
                folder = raw / "processes" / str(observer)
                for boundary in m.BOUNDARIES:
                    selections = (
                        [
                            (dict(selected, arm_order=[a], requests=[q]), [repeat])
                            for repeat in range(3)
                            for a in world["arm_order"]
                            for q in world["requests"]
                        ]
                        if boundary == "cold"
                        else [(selected, list(range(3)))]
                    )
                    for chosen, repeats in selections:
                        child = process(
                            python, baseline, [chosen], repeats, boundary, private, folder, deadline
                        )
                        rows.extend(child["rows"])
                        clocks.append(dict(child["receipt"], instrumented=observer))
                m.progress("paired_schedule", len(rows), total - len(rows))
        m.progress("before_environment_disposal")
        disposal_start = time.monotonic()
    m.progress("after_environment_disposal")
    disposal_s = time.monotonic() - disposal_start
    with ZipFile(wheel) as archive:
        expected_imports = {
            Path(p).name: "sha256:" + hashlib.sha256(archive.read(p)).hexdigest()
            for p in archive.namelist()
            if p.startswith("carnot/verify/") and p.endswith(".py")
        }
    imports_valid = all(
        c["installed_imports"]
        and all(
            p["sha256"] == expected_imports.get(Path(p["path"]).name)
            and str(private / "venv") in p["path"]
            for p in c["installed_imports"]
        )
        for c in clocks
    )
    return dict(
        rows=rows,
        clocks=clocks,
        installation_receipts=installations,
        installation_s=installation_s,
        disposal_s=disposal_s,
        installed_imports_valid=imports_valid,
        package_hashes=[reference(wheel)],
        pilot_receipt=pilot_receipt,
    )


def prepare(raw: Path, limit: int | None = None) -> Json:
    """Seal the roster before clocks and preserve scheduled absence without fabricating time."""
    start = time.monotonic_ns()
    m.progress("before_preconditions")
    pre = preconditions(raw / "authorities")
    m.progress("after_preconditions")
    worlds = m.freeze()[:limit]
    atomic_json(
        raw / "frozen_worlds.json",
        dict(worlds=worlds, seed=m.SEED, repeats=3, diagnostic_only=True),
    )
    measured: Json = dict(
        rows=[],
        clocks=[],
        installation_receipts=[],
        installation_s=0,
        disposal_s=0,
        installed_imports_valid=False,
        package_hashes=[],
        pilot_receipt={},
    )
    errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            measured = measure(raw, worlds, pre["package_hash"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
    if not measured["rows"]:
        retained = [load(reference(p)) for p in sorted((raw / "processes").rglob("*.json"))]
        measured["rows"] = [r for p in retained for r in p["rows"]]
        measured["clocks"] = [
            dict(p["receipt"], instrumented=p["rows"][0]["instrumented"]) for p in retained
        ]
    seen = {
        (r["boundary"], r["unit_id"], r["arm"], r["repeat"], r["instrumented"])
        for r in measured["rows"]
    }
    for world in worlds:
        for boundary in m.BOUNDARIES:
            for arm in m.ARMS:
                for repeat in range(3):
                    for q in world["requests"]:
                        for observer in (False, True):
                            if (boundary, q["unit_id"], arm, repeat, observer) not in seen:
                                measured["rows"].append(
                                    dict(
                                        unit_id=q["unit_id"],
                                        cluster_id=world["cluster_id"],
                                        size=world["size"],
                                        boundary=boundary,
                                        arm=arm,
                                        repeat=repeat,
                                        instrumented=observer,
                                        seed=m.SEED,
                                        denominator=1,
                                        metric=0,
                                        status="failed" if errors else "censored",
                                        missing_reason=errors[0]
                                        if errors
                                        else "external_or_resource_gate",
                                        decision="insufficient",
                                        witness=None,
                                        primitive_operations=0,
                                        operation_counts={},
                                        complete_wall_time_s=0,
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
            compute="retained installed signed-record consumer; exclusive research clocks",
            model_invoked=False,
        ),
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[
            dict(phase="authenticated_request_cost_diagnosis", start_s=0, end_s=(end - start) / 1e9)
        ],
    )
    sources = (
        m.OWNED
        + m.TESTS
        + design.OWNED
        + [
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
    )
    return dict(
        measured,
        worlds=worlds,
        private_world_limit=limit,
        preconditions_checked=pre,
        internal_errors=errors,
        current_work_receipt=receipt,
        producer_invocation=dict(argv=sys.argv, cwd=str(Path.cwd()), python=sys.executable),
        code_config_hashes=[reference(m.ROOT / p) for p in sources],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
    )


def validate(raw: Path, private: Path) -> Json:
    """Retain unit and real child coverage plus unchanged private E2E-018/021 checks."""
    runner = namespace(previous, "attribution8882_validation")
    runner.m = m
    return dict(runner.validate(raw, private))


def artifact(work: Json, validation: Json, date: str, work_ref: Json, validation_ref: Json) -> Json:
    """Keep administrative completion separate from observer-qualified cost attribution."""
    rows, pre, receipt = work["rows"], work["preconditions_checked"], work["current_work_receipt"]
    reduced = m.reduce(rows)
    gates = dict(
        authorities=pre["external_ready"],
        resources=pre["resources"]["passed"],
        installed_imports=work["installed_imports_valid"],
        full_roster=work["private_world_limit"] is None
        and len(rows) == 8640
        and len(work["worlds"]) == 60,
        owned_execution=not work["internal_errors"]
        and reduced["quality_gates"]["no_owned_failures"],
        owned_validation=qualified(validation),
    )
    ready = int(all(gates.values()) and reduced["cost_attribution_ready_score"])
    verdict = (
        "blocked"
        if not pre["external_ready"]
        else "disqualified"
        if not all(gates.values())
        else "null"
    )
    value: Json = dict(
        reduced,
        experiment_id=8882,
        task_id=m.TASK,
        milestone="2026.11.760",
        run_date=date,
        honest_verdict="complete_" + verdict + "_request_cost_attribution",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="Exposed retained constructed signed-record requests; cost diagnosis only, no superiority test.",
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
        cited_upstream_artifacts=[reference(p) for p, _, _ in HISTORY if p.is_file()],
        rows=rows,
        intended_count=len(work["worlds"]) * 144 if work["worlds"] else 8640,
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
        acceptance_gates=dict(gates, **reduced["attribution_gates"]),
        required_checks_passed=qualified(validation),
        validation_receipts=validation["receipts"] + work["installation_receipts"],
        owned_statement_coverage=validation["coverage"],
        cost_attribution_ready_score=ready,
        complete_parent_costs=work["clocks"],
        installation_accounting=dict(
            installation_s=work["installation_s"],
            environment_disposal_s=work["disposal_s"],
            amortization_worlds=60,
            per_arm_world_s=work["installation_s"] / 60,
            included_in_component_ratios=False,
            warmup=work["pilot_receipt"],
            timeout_costs_s=sum(c["wall_time_s"] for c in work["clocks"] if not c["passed"]),
            startup_policy="Import startup is retained per process and allocated equally in resident parent costs; request_elapsed_ns separately exposes steady service. Unaccounted residual is never treated as startup.",
        ),
        package_hashes=work["package_hashes"],
        producer_invocation=work["producer_invocation"],
        current_work_receipt=receipt,
        work_reference=work_ref,
        validation_reference=validation_ref,
        dispatch_rule=dict(threshold=16, frozen=True, index_reuse=False),
        prd_rust10x_score=0,
        repository_health=work.get(
            "repository_health", dict(status="not_run", scope="unrelated_global_health")
        ),
        methodology="Ten original exposed ordinary worlds at each of six fixed sizes, all four requests, three arms, cold/resident and three repeats. Frozen seeded on/off order pairs whole worlds. Exclusive nested intervals use unchanged installed code and caps. Parent clocks include spawn, transport, disposal and observer setup. No-op clock samples are retained, never subtracted. Repeats average within world; paired bootstrap99-percent intervals use10000 draws. Every stratum must have observer overhead upper<=.05 and measured component closure residual between0 and5 percent. Residuals remain explicit; negative or missing clocks block attribution. Censors and failures retain their slots. This is a bottleneck diagnosis, not optimization or hardware evidence.",
    )
    value["field_principles"] = {
        k: "Bind exact producer bytes and scheduled independent worlds; completion and observed operation counts do not establish elapsed fractions or scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        cost_attribution_ready_score="Only non-overlapping clocks, bounded residuals and paired observer overhead at most5 percent permit attribution.",
        component_rows="Exclusive measured intervals remain exploratory unless observer and closure gates pass.",
        uninstrumented_rows="Retain the full complete-clock comparator; no estimated observer subtraction.",
        dominant_cost="No unqualified component may become a removable bottleneck claim.",
        next_mechanism="Do not select an optimization from unqualified observer timings.",
        whole_world_intervals="Average repeats first; bootstrap whole independent worlds within each fixed size/mode/arm stratum.",
        installation_accounting="Deployment setup, warmup and timeouts remain visible separately from steady request service.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path, *, execute: bool = True) -> bool:
    """Rebuild reductions from raw operands and rerun installed semantics after rehashing."""
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
        if (
            validate_current_work_receipt(work["current_work_receipt"])
            or work["worlds"] != m.freeze()[: work["private_world_limit"]]
        ):
            return False
        frozen = next(
            ref
            for ref in work["raw_shard_hashes"]
            if Path(ref["path"]).name == "frozen_worlds.json"
        )
        if load(frozen)["worlds"] != work["worlds"]:
            return False
        operands = [
            load(ref)
            for ref in work["raw_shard_hashes"]
            if Path(ref["path"]).parent.parent.name == "processes"
        ]
        order = lambda row: (
            row["boundary"],
            row["unit_id"],
            row["arm"],
            row["repeat"],
            row["instrumented"],
        )
        if work["clocks"]:
            clock_order = lambda c: (c["name"], c["instrumented"])
            if sorted(work["clocks"], key=clock_order) != sorted(
                [dict(p["receipt"], instrumented=p["rows"][0]["instrumented"]) for p in operands],
                key=clock_order,
            ):
                return False
            if sorted(work["rows"], key=order) != sorted(
                [r for p in operands for r in p["rows"]], key=order
            ):
                return False
            for operand in operands:
                clock = operand["receipt"]
                if (
                    clock["wall_time_s"]
                    != (clock["ended_monotonic_ns"] - clock["started_monotonic_ns"]) / 1e9
                ):
                    return False
                for row in operand["rows"]:
                    totals: dict[str, int] = {}
                    last = row["started_monotonic_ns"]
                    for span in row.get("component_intervals", []):
                        if (
                            span["start_ns"] < last
                            or span["end_ns"] < span["start_ns"]
                            or span["end_ns"] > row["ended_monotonic_ns"]
                        ):
                            return False
                        totals[span["component"]] = (
                            totals.get(span["component"], 0) + span["end_ns"] - span["start_ns"]
                        )
                        last = span["end_ns"]
                    if totals != row.get("components_ns", {}):
                        return False
                    expected = (
                        clock["wall_time_s"]
                        if row["boundary"] == "cold"
                        else (row["ended_monotonic_ns"] - row["started_monotonic_ns"]) / 1e9
                        + row["resident_process_overhead_s"]
                    )
                    if row["complete_wall_time_s"] != expected:
                        return False
        if value["required_checks_passed"]:
            checker = namespace(ordered.checker, "attribution8882_coverage_replay")
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
            with TemporaryDirectory(prefix="exp8882-replay-") as directory:
                scratch = Path(directory)
                python = scratch / "venv/bin/python"
                for name, argv in [
                    ("venv", ["uv", "venv", str(scratch / "venv"), "--python", sys.executable]),
                    (
                        "install",
                        [
                            "uv",
                            "pip",
                            "install",
                            "--python",
                            str(python),
                            "--no-deps",
                            work["package_hashes"][0]["path"],
                        ],
                    ),
                ]:
                    m.progress("before_replay_" + name)
                    installed = stream(name, argv, scratch / "logs", deadline=120)["passed"]
                    m.progress("after_replay_" + name)
                    if not installed:
                        return False
                baseline = scratch / "scan.py"
                baseline.write_text(m.baseline_source())
                for observer in (False, True):
                    repeated = process(
                        python,
                        baseline,
                        [dict(w, instrumented=observer) for w in work["worlds"]],
                        list(range(3)),
                        "resident",
                        scratch,
                        scratch / str(observer),
                        time.monotonic() + 300,
                    )
                    stable: Callable[[Json], Json] = lambda r: {
                        k: r[k] for k in ("unit_id", "arm", "repeat", *m.SEMANTICS)
                    }
                    if [stable(r) for r in repeated["rows"]] != [
                        stable(r)
                        for r in work["rows"]
                        if r["boundary"] == "resident" and r["instrumented"] == observer
                    ]:
                        return False
        return True
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Run the unchanged prospective scanners and actual outside-checkout CLI replay."""
    runner = namespace(previous, "attribution8882_terminal")
    runner.m = m
    return dict(runner.terminal(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Keep partial checkpoints private and expose only terminal validated bytes."""
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
    health = Path("/tmp/exp8882-health/receipt.json")
    if health.is_file():
        shutil.copytree(health.parent, raw / "repository_health")
        work["repository_health"] = dict(
            load(reference(raw / "repository_health/receipt.json")),
            logs=[reference(p) for p in sorted((raw / "repository_health").iterdir())],
        )
    atomic_json(raw / "work.json", work)
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8882-validation-") as directory:
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

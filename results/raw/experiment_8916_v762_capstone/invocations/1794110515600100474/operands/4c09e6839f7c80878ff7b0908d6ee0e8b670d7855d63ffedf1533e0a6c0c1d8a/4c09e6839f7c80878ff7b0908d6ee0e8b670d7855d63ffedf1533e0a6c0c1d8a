"""REQ-REPORT-8894: authenticate inputs and qualify a private buffered observer."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable
from zipfile import ZipFile

from carnot.reporting import buffered_clock_methods_8894 as m
from carnot.reporting import request_cost_runner_8882 as old
from carnot.reporting import request_cost_attribution_8882 as original
from carnot.reporting import size_aware_complete_cost_8874 as design
from carnot.reporting import size_aware_complete_runner_8874 as parent
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
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
HISTORY = old.HISTORY + [
    (
        m.ROOT / "results/experiment_8882_v760_request_cost_attribution.json",
        "sha256:cbb9f79adfe6fbe6c974169366efb5603005f4abb7ea0e50b7d6415753612a93",
        "required_checks_passed",
    )
]
START = time.monotonic()


def preconditions(raw: Path) -> Json:
    """Accept the authenticated old null, while requiring its original checks and bytes."""
    checker = namespace(old, "buffered8894_authority")
    checker.HISTORY = HISTORY
    return dict(checker.preconditions(raw))


def summarize(result: Json, world: Json, arm: str, boundary: str, block: int) -> Json:
    """Keep parent and child clocks separate and expose all child remainder explicitly."""
    rows, receipt = result["rows"], result["receipt"]
    observed = world["block_order"][block]
    valid = receipt["passed"] and bool(rows)
    partitions = [
        m.partition(
            r["endpoints"], r["start_ns"], r["end_ns"], endpoint_count=r.get("endpoint_count", -1)
        )
        for r in rows
        if observed and "endpoints" in r
    ]
    child_ns = rows[0].get("block_elapsed_ns", 0)
    request_ns = sum(r.get("end_ns", 0) - r.get("start_ns", 0) for r in rows)
    remainder = child_ns - request_ns
    valid &= len(partitions) == len(rows) if observed else True
    valid &= (
        all(p["valid"] for p in partitions) and remainder >= 0 and 0 < child_ns < 60_000_000_000
    )
    valid &= all(
        r.get("block_elapsed_ns") == child_ns
        and child_ns == r["block_end_ns"] - r["block_start_ns"]
        for r in rows
    )
    status = (
        "failed"
        if not receipt["passed"] or any(r["status"] == "failed" for r in rows)
        else "censored"
        if any(r["status"] == "censored" for r in rows)
        else "completed"
    )
    return dict(
        unit_id=f"{world['cluster_id']}/{arm}/{boundary}/{block}",
        cluster_id=world["cluster_id"],
        size=world["size"],
        arm=arm,
        boundary=boundary,
        block=block,
        instrumented=observed,
        seed=m.SEED,
        metric=int(valid),
        denominator=1,
        status=status,
        missing_reason=receipt["missing_reason"]
        or next((r["missing_reason"] for r in rows if r["missing_reason"]), None),
        elapsed_ns=round(receipt["wall_time_s"] * 1e9) if boundary == "cold" else child_ns,
        parent_elapsed_ns=receipt["ended_monotonic_ns"] - receipt["started_monotonic_ns"],
        child_elapsed_ns=child_ns,
        request_elapsed_ns=request_ns,
        explicit_unclassified_ns=remainder,
        closure_valid=bool(valid),
        partitions=partitions,
        semantics=[{k: r[k] for k in ("unit_id", "repeat", *m.SEMANTICS)} for r in rows],
    )


def process(
    python: Path,
    baseline: Path,
    world: Json,
    arm: str,
    boundary: str,
    block: int,
    count: int,
    private: Path,
    raw: Path,
    deadline: float,
) -> Json:
    """Reuse the original bounded supervisor without its resident overhead allocation."""
    runner = namespace(parent, "buffered8894_parent")
    runner.m = m
    selected = dict(
        world,
        arm_order=[arm],
        instrumented=world["block_order"][block],
        warmup=boundary == "resident",
    )
    # The cold supervisor retains the complete parent; resident cost comes from the child block.
    result = runner.process(
        python, baseline, [selected], list(range(count)), "cold", private, raw, deadline
    )
    return dict(
        row=summarize(result, world, arm, boundary, block),
        raw_reference=reference(raw / (result["receipt"]["name"] + ".json")),
    )


def measure(
    raw: Path, worlds: list[Json], package: Json, *, frozen_count: int | None = None
) -> Json:
    """Install once, freeze resolution-only repetitions, then execute development ABBA blocks."""
    raw.mkdir(parents=True, exist_ok=True)
    baseline = raw / "scan.py"
    baseline.write_text(m.baseline_source())
    wheel = raw / Path(package["path"]).name
    shutil.copy2(package["path"], wheel)
    rows: list[Json] = []
    operands, receipts, calibration = [], [], []
    deadline = min(START + 2400, time.monotonic() + 1800)
    with TemporaryDirectory(prefix="exp8894-installed-") as directory:
        private = Path(directory)
        python = private / "venv/bin/python"
        started = time.monotonic()
        for name, argv in [
            ("venv", ["uv", "venv", str(private / "venv"), "--python", sys.executable]),
            ("install", ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)]),
        ]:
            m.progress("before_" + name)
            receipts.append(stream(name, argv, raw / "installation", deadline=120))
            m.progress("after_" + name)
            if not receipts[-1]["passed"]:
                raise ValueError("installation_failed")
        installation_s = time.monotonic() - started
        ticks = [time.monotonic_ns() for _ in range(101)]
        resolution = max(1, math.ceil(time.get_clock_info("monotonic").resolution * 1e9))
        effective = max(resolution, min(b - a for a, b in zip(ticks, ticks[1:]) if b > a))
        for size in sorted({w["size"] for w in worlds}):
            world = next(w for w in worlds if w["size"] == size)
            off = world["block_order"].index(False)
            for arm in m.ARMS:
                result = process(
                    python,
                    baseline,
                    world,
                    arm,
                    "resident",
                    off,
                    1,
                    private,
                    raw / "calibration" / str(size) / arm,
                    deadline,
                )
                calibration.append(result)
        count = m.repetitions(effective, [x["row"]["child_elapsed_ns"] for x in calibration])
        if frozen_count is not None:
            count = frozen_count
        precision = dict(
            resolution_ns=resolution,
            effective_resolution_ns=effective,
            raw_clock_samples_ns=ticks,
            count=count,
            max_count=1024,
            formula="min(1024,max(1,ceil(1000*effective_resolution_ns/min_plain_block_ns)))",
            projected_max_block_s=max(x["row"]["child_elapsed_ns"] for x in calibration)
            * count
            / 1e9,
            sufficient_ticks=count * min(x["row"]["child_elapsed_ns"] for x in calibration)
            >= 1000 * effective,
            projected_total_s=max(x["row"]["parent_elapsed_ns"] for x in calibration)
            * len(worlds)
            * 24
            * 4
            / 1e9
            + 1800,
        )
        if (
            precision["projected_max_block_s"] >= 60
            or precision["projected_total_s"] >= 4200
            or not precision["sufficient_ticks"]
            or not all(x["row"]["closure_valid"] for x in calibration)
        ):
            raise ValueError("pilot_resolution_or_runtime")
        atomic_json(raw / "precision.json", precision)
        for world in worlds:
            for boundary in m.BOUNDARIES:
                for arm in world["arm_order"]:
                    for block in range(4):
                        m.progress(
                            "before_development_block", len(rows), len(worlds) * 24 - len(rows)
                        )
                        result = process(
                            python,
                            baseline,
                            world,
                            arm,
                            boundary,
                            block,
                            count,
                            private,
                            raw / "blocks" / world["cluster_id"] / boundary / arm / str(block),
                            deadline,
                        )
                        rows.append(result["row"])
                        operands.append(result["raw_reference"])
                        m.progress(
                            "after_development_block", len(rows), len(worlds) * 24 - len(rows)
                        )
        with ZipFile(wheel) as archive:
            pins = {
                Path(p).name: "sha256:" + hashlib.sha256(archive.read(p)).hexdigest()
                for p in archive.namelist()
                if p.startswith("carnot/verify/") and p.endswith(".py")
            }
        imports_valid = all(
            all(
                "/site-packages/carnot/verify/" in p["path"]
                and not Path(p["path"]).is_relative_to(m.ROOT)
                and pins.get(Path(p["path"]).name) == p["sha256"]
                for p in load(ref)["receipt"]["installed_imports"]
            )
            and len(load(ref)["receipt"]["installed_imports"]) == 4
            for ref in operands
        )
        m.progress("before_disposal")
    m.progress("after_disposal")
    return dict(
        rows=rows,
        process_references=operands,
        calibration=calibration,
        precision=precision,
        repetition_count=count,
        installation_s=installation_s,
        installation_receipts=receipts,
        package_hashes=[reference(wheel)],
        installed_imports_valid=imports_valid,
    )


def prepare(raw: Path, limit: int | None = None) -> Json:
    """Seal future worlds before development clocks and preserve all scheduled absences."""
    start = time.monotonic_ns()
    m.progress("before_preconditions")
    pre = preconditions(raw / "authorities")
    m.progress("after_preconditions")
    worlds, future = m.freeze(pilot=True)[:limit], m.freeze()
    atomic_json(raw / "frozen_worlds.json", dict(development=worlds, future=future))
    measured: Json = dict(
        rows=[],
        process_references=[],
        calibration=[],
        precision={},
        repetition_count=1,
        installation_s=0,
        installation_receipts=[],
        package_hashes=[],
        installed_imports_valid=False,
    )
    errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            measured = measure(raw / "measured", worlds, pre["package_hash"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
            retained = sorted((raw / "measured/blocks").rglob("cold-*.json"))
            for operand in retained:
                result = load(reference(operand))
                child = result["rows"][0]
                world = next(w for w in worlds if w["cluster_id"] == child["cluster_id"])
                block_label, arm, boundary = (
                    operand.parent.name,
                    operand.parent.parent.name,
                    operand.parent.parent.parent.name,
                )
                measured["rows"].append(summarize(result, world, arm, boundary, int(block_label)))
                measured["process_references"].append(reference(operand))
    seen = {row["unit_id"] for row in measured["rows"]}
    for world in worlds:
        for boundary in m.BOUNDARIES:
            for arm in world["arm_order"]:
                for block, observed in enumerate(world["block_order"]):
                    unit = f"{world['cluster_id']}/{arm}/{boundary}/{block}"
                    if unit not in seen:
                        measured["rows"].append(
                            dict(
                                unit_id=unit,
                                cluster_id=world["cluster_id"],
                                size=world["size"],
                                arm=arm,
                                boundary=boundary,
                                block=block,
                                instrumented=observed,
                                seed=m.SEED,
                                denominator=1,
                                metric=0,
                                status="failed" if errors else "censored",
                                missing_reason=errors[0] if errors else "external_or_resource_gate",
                                elapsed_ns=0,
                                semantics=[],
                                closure_valid=False,
                            )
                        )
    end = time.monotonic_ns()
    substrate = "no_model_load" if measured["process_references"] else "blocked_no_run"
    receipt = build_current_work_receipt(
        run_id=m.TASK + "/" + str(start),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=substrate,
        inference_substrate_class=substrate,
        execution_venue="host",
        inference_substrate_details=dict(
            compute="installed Python buffered clock development controls", model_invoked=False
        ),
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[
            dict(
                phase="preconditions_and_development_clock_qualification",
                start_s=0,
                end_s=(end - start) / 1e9,
            )
        ],
    )
    sources = (
        m.OWNED
        + m.TESTS
        + original.OWNED
        + design.OWNED
        + [
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "openspec/capabilities/research-reporting/spec.md",
        ]
    )
    return dict(
        measured,
        worlds=worlds,
        future=future,
        private_world_limit=limit,
        internal_errors=errors,
        preconditions_checked=pre,
        current_work_receipt=receipt,
        producer_invocation=dict(argv=sys.argv, cwd=str(Path.cwd()), python=sys.executable),
        code_config_hashes=[reference(m.ROOT / p) for p in dict.fromkeys(sources)],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse measured unit/child coverage and private E2E-018/021 checks unchanged."""

    def child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        # Resolving the1.29 million tracked result paths exceeds the old collection startup budget.
        if name == "collect_once":
            kwargs["deadline"] = 180
        return stream(name, argv, logs, **kwargs)

    runner = namespace(parent, "buffered8894_validation")
    runner.m = m
    runner.stream = child
    return dict(runner.validate(raw, private))


def artifact(work: Json, validation: Json, date: str, work_ref: Json, validation_ref: Json) -> Json:
    """Separate observer protocol readiness from the future elapsed acceptance decision."""
    pre, receipt, rows = work["preconditions_checked"], work["current_work_receipt"], work["rows"]
    reduced, controls = m.reduce(rows), m.controls()
    gates = dict(
        authorities=pre["external_ready"],
        resources=pre["resources"]["passed"],
        installed_imports=work["installed_imports_valid"],
        owned_execution=not work["internal_errors"],
        complete_development_roster=work["private_world_limit"] is None and len(rows) == 576,
        clock_controls=controls["passed"],
        precision=bool(work["precision"]) and work["precision"]["projected_max_block_s"] < 60,
        owned_validation=qualified(validation),
        **{k: reduced[k] for k in ("semantic_identity", "finite_nonnegative_closure")},
    )
    ready = int(all(gates.values()))
    verdict = (
        "blocked"
        if not pre["external_ready"]
        else "disqualified"
        if not all(gates.values())
        else "circular_positive"
    )
    value: Json = dict(
        reduced,
        experiment_id=8894,
        task_id=m.TASK,
        milestone="2026.11.761",
        run_date=date,
        honest_verdict="complete_" + verdict + "_buffered_clock_methods",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="Constructed development signed-record worlds; future roster unexecuted.",
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
        intended_count=len(work["worlds"]) * 24,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=len(work["worlds"]),
        acceptance_gates=gates,
        required_checks_passed=qualified(validation),
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
        validation_receipts=validation["receipts"] + work["installation_receipts"],
        owned_statement_coverage=validation["coverage"],
        observer_ready_score=ready,
        observer_schema=dict(
            version=1,
            clock="time.monotonic_ns",
            buffer="preallocated array(q) triples: timestamp,phase,serialization_site",
            nested_intervals=False,
            hot_path_logging=False,
            endpoint_count="Buffer cursor count retained separately from exported endpoints; any deletion fails closure.",
            serialization_sites=dict(request=1, attempt=2, core=3, response=4),
        ),
        phase_partition=list(m.PHASES),
        clock_controls=controls,
        repetition_count=work["repetition_count"],
        clock_resolution=work["precision"],
        future_world_manifest=dict(
            worlds=work["future"],
            independent_count=144,
            clusters_per_size=24,
            executed_count=0,
            arms=list(m.ARMS),
            boundaries=list(m.BOUNDARIES),
            acceptance_owner="exp8899",
            semantic_identity_required=True,
            finite_nonnegative_closure_required=True,
            familywise_confidence=0.99,
            simultaneous_cells=36,
            overhead_upper_at_most=0.05,
            resampling_unit="whole_world",
            draws=m.DRAWS,
            repetition_count=work["repetition_count"],
        ),
        closure_tolerance=dict(
            material_negative_ns=0,
            policy="retain signed residual; reject any negative or missing endpoint",
        ),
        installation_accounting=dict(
            installation_s=work["installation_s"],
            included_in_resident_phase_sums=False,
            parent_policy="Full supervisor clocks include startup, compilation, warmup, transport and exit; installation is separate paid work.",
        ),
        current_work_receipt=receipt,
        producer_invocation=work["producer_invocation"],
        work_reference=work_ref,
        validation_reference=validation_ref,
        repository_health=work.get(
            "repository_health", dict(status="not_run", scope="unrelated_full_suite")
        ),
        methodology="Development worlds only. Random ABBA/BAAB blocks pair unchanged installed requests and opt-in private AST copies. Flat buffered endpoints retain all request serialization and unclassified remainder. Real parent and child clocks remain separate. Repetition count uses timing resolution only. Bootstrap whole development clusters with Bonferroni simultaneous99-percent diagnostic bounds over36 cells. Future24 clusters per size remain unexecuted; Exp8899 owns acceptance. No optimization, board test, natural-language generalization or installed behavior change.",
    )
    value["field_principles"] = {
        k: "Bind actual producer bytes, real elapsed operands and independent whole worlds; administrative completion does not establish scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        observer_ready_score="Development clock controls and frozen protocol only; Exp8899 owns measurement acceptance.",
        pilot_overhead_rows="Real development diagnostics; never a future-world acceptance result.",
        independent_count="Development world clusters only; repeats and four requests do not increase n.",
        future_world_manifest="Frozen unexecuted independent clusters and simultaneous acceptance rule.",
        phase_partition="Adjacent endpoints partition one child clock; parent and installation clocks are never subtracted from phases.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path, *, execute: bool = True) -> bool:
    """Rebuild every summary from retained operands before running a fresh installed control."""
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
            or work["worlds"] != m.freeze(pilot=True)[: work["private_world_limit"]]
            or work["future"] != m.freeze()
        ):
            return False
        frozen = next(
            ref
            for ref in work["raw_shard_hashes"]
            if Path(ref["path"]).name == "frozen_worlds.json"
        )
        if load(frozen) != dict(development=work["worlds"], future=work["future"]):
            return False
        if work["precision"]:
            precision_ref = next(
                ref
                for ref in work["raw_shard_hashes"]
                if Path(ref["path"]).name == "precision.json"
            )
            precision = load(precision_ref)
            if work["precision"] != precision or work["repetition_count"] != precision["count"]:
                return False
            ticks = precision["raw_clock_samples_ns"]
            effective = max(
                precision["resolution_ns"], min(b - a for a, b in zip(ticks, ticks[1:]) if b > a)
            )
            if effective != precision["effective_resolution_ns"]:
                return False
            samples = work["calibration"]
            if (
                m.repetitions(effective, [c["row"]["child_elapsed_ns"] for c in samples])
                != work["repetition_count"]
            ):
                return False
            for sample in samples:
                result = load(sample["raw_reference"])
                row = sample["row"]
                world = next(w for w in work["worlds"] if w["cluster_id"] == row["cluster_id"])
                if row != summarize(
                    result, world, row["arm"], "resident", world["block_order"].index(False)
                ):
                    return False
        actual = {row["unit_id"]: row for row in work["rows"]}
        for ref in work["process_references"]:
            result, source = load(ref), Path(ref["path"])
            receipt = result["receipt"]
            if (
                receipt["wall_time_s"]
                != (receipt["ended_monotonic_ns"] - receipt["started_monotonic_ns"]) / 1e9
            ):
                return False
            world = next(
                w for w in work["worlds"] if w["cluster_id"] == result["rows"][0]["cluster_id"]
            )
            row = summarize(
                result,
                world,
                source.parent.parent.name,
                source.parent.parent.parent.name,
                int(source.parent.name),
            )
            if actual[row["unit_id"]] != row:
                return False
        if value["required_checks_passed"]:
            checker = namespace(ordered.checker, "buffered8894_coverage")
            checker.m = m
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
            with TemporaryDirectory(prefix="exp8894-replay-") as directory:
                scratch = Path(directory)
                repeated = measure(
                    scratch,
                    work["worlds"][:1],
                    work["package_hashes"][0],
                    frozen_count=work["repetition_count"],
                )
                stable: Callable[[Json], tuple[Any, ...]] = lambda row: (
                    row["unit_id"],
                    row["semantics"],
                )
                original = [
                    row
                    for row in work["rows"]
                    if row["cluster_id"] == work["worlds"][0]["cluster_id"]
                ]
                if repeated["repetition_count"] != work["repetition_count"]:
                    return False
                if [stable(row) for row in repeated["rows"]] != [stable(row) for row in original]:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Run unchanged prospective scanners and the actual outside-checkout CLI replay."""
    runner = namespace(parent, "buffered8894_terminal")
    runner.m = m
    return dict(runner.terminal(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Keep checkpoints private until all terminal scans inspect prospective bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261107"], default="20261107")
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
    template = ExperimentTemplate(
        8894, "Buffered clock methods", "results/" + m.NAME + ".json", seed=m.SEED
    )
    with template.phase("development_clock_qualification"):
        work = prepare(raw, args.private_world_limit)
    health = Path("/tmp/exp8894-health/receipt.json")
    if health.is_file():
        shutil.copytree(health.parent, raw / "repository_health")
        work["repository_health"] = dict(
            load(reference(raw / "repository_health/receipt.json")),
            logs=[
                reference(p) for p in sorted((raw / "repository_health").iterdir()) if p.is_file()
            ],
        )
    atomic_json(raw / "work.json", work)
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8894-validation-") as directory:
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
    m.progress("before_terminal_publication")
    publication = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    atomic_json(raw / "publication_receipt.json", publication)
    for operand in raw.rglob("*"):
        if operand.is_file():
            operand.chmod(0o444)
    m.progress("complete_" + value["verdict_class"], value["completed_count"], 0)
    return 0

"""REQ-REPORT-8921: freeze development cells and reduce honest observer qualification."""

from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import numpy as np

from carnot.reporting import buffered_clock_methods_8894 as prior
from carnot.reporting import observer_custody_8907 as custody
from carnot.reporting import size_aware_complete_cost_8874 as design
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT, SEED = prior.ROOT, 7638921
NAME, TASK = "experiment_8921_v763_inline_clock_methods", "exp8921-inline-clock-methods"
CLI = f"scripts/experiments/{NAME}.py"
WORKER = "scripts/experiments/inline_clock_worker_8921.py"
OWNED = [
    "python/carnot/reporting/inline_clock_methods_8921.py",
    "python/carnot/reporting/inline_clock_runner_8921.py",
    WORKER,
    CLI,
]
TESTS = ["tests/python/test_inline_clock_methods_8921.py"]
MODES = ("off", "buffered", "inline")
__all__ = ["prior", "custody"]
UPSTREAM = ROOT / "results/experiment_8912_v762_complete_request_cost.json"
PIN = "sha256:0a468088fa1dd6d95fd88199fc1276ff83bb6fd1f0a43572e1bf58310d53a57f"
HEALTH = ROOT / "results/raw" / NAME / "repository_health"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose phase progress without printing inside a timed request."""
    print(f"[exp8921] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, pin: str = PIN) -> Json:
    """Accept valid complete clocks even though the prior observer failed overhead."""
    malformed = False
    try:
        value = load(reference(path)) if path.is_file() else {}
    except (ValueError, TypeError):
        value, malformed = {}, True
    field = "complete_cost_rows_ready_score"
    digest = sha256_file(path) if path.is_file() else None
    check = (
        "missing_file"
        if digest is None
        else "rejected_producer"
        if malformed
        else "missing_field"
        if field not in value
        else "observed_zero"
        if type(value[field]) is not int or value[field] != 1
        else "rejected_producer"
        if digest != pin
        or value.get("experiment_id") != 8912
        or value.get("task_id") != "exp8912-complete-request-cost"
        or value.get("required_checks_passed") is not True
        else "admitted"
    )
    return dict(
        check=check,
        upstream="exp8912-complete-request-cost",
        artifact_field=field,
        expected=1,
        observed=value.get(field),
        path=str(path),
        hash=digest,
    )


def freeze() -> list[Json]:
    """Use new development seeds without reading or evaluating the future roster."""
    producer = namespace(design, "inline8921_worlds")
    producer.SEED = SEED
    worlds = producer.freeze()
    rng = random.Random(SEED)
    selected = []
    for size in prior.SIZES:
        for world in [w for w in worlds if w["size"] == size and w["population"] == "ordinary"][:4]:
            world["cluster_id"] = "development8921/" + world["cluster_id"]
            for unit in world["requests"]:
                unit["unit_id"] = "development8921/" + unit["unit_id"]
            order = list(MODES)
            rng.shuffle(order)
            world["mode_order"] = order + order[::-1]
            selected.append(world)
    return selected


def reduce(rows: list[Json]) -> Json:
    """Average repeated requests within worlds before simultaneous resampling."""
    cells: Any = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        if row["unit_id"] in seen:
            raise ValueError("duplicate_unit")
        seen.add(row["unit_id"])
        cells[(row["size"], row["boundary"], row["arm"])][row["cluster_id"]].append(row)
    complete = set(cells) == {
        (s, b, a) for s in prior.SIZES for b in prior.BOUNDARIES for a in prior.ARMS
    }
    semantic, closure = bool(rows), bool(rows)
    overheads = []
    for (size, boundary, arm), worlds in sorted(cells.items()):
        data = []
        for sample in worlds.values():
            by_mode = {mode: [r for r in sample if r["mode"] == mode] for mode in MODES}
            counts = [len(s) for s in by_mode.values()]
            complete &= len(worlds) == 4 and min(counts) > 0 and len(set(counts)) == 1
            semantics: dict[str, Any] = {}
            for row in sample:
                request = row.get("request_id", "fixture")
                semantic &= row["semantics"] == semantics.setdefault(request, row["semantics"])
            closure &= all(
                r["closure_valid"]
                and r["status"] != "failed"
                and np.isfinite(r["elapsed_ns"])
                and r["elapsed_ns"] > 0
                for r in sample
            )
            data.append(
                [
                    sum(r["elapsed_ns"] for r in by_mode[mode]) / max(1, len(by_mode[mode]))
                    for mode in MODES
                ]
            )
        values = np.array(data)
        valid = bool(np.all(np.isfinite(values)) and np.all(values > 0))
        indices = np.random.default_rng(SEED + size).integers(0, len(values), (10000, len(values)))
        draws = values[indices].mean(axis=1)
        for column, mode in enumerate(MODES[1:], 1):
            upper = (
                float(np.quantile(draws[:, column] / draws[:, 0] - 1, 1 - 0.01 / 72))
                if valid
                else None
            )
            overheads.append(
                dict(
                    size=size,
                    boundary=boundary,
                    arm=arm,
                    mode=mode,
                    estimate=float(values[:, column].mean() / values[:, 0].mean() - 1)
                    if valid
                    else None,
                    upper=upper,
                    accepted=upper is not None and upper <= 0.05,
                    independent_count=len(worlds),
                    familywise_confidence=0.99,
                    simultaneous_cell_count=72,
                    draws=10000,
                    scope="development_only_whole_world",
                )
            )
    ready = (
        complete
        and semantic
        and closure
        and all(r["accepted"] for r in overheads if r["mode"] == "inline")
    )
    return dict(
        complete=bool(complete),
        semantic_identity=bool(semantic),
        finite_nonnegative_closure=bool(closure),
        development_overhead_rows=overheads,
        inline_clock_ready_score=int(ready),
    )


def schedule(worlds: list[Json]) -> list[Json]:
    """Register every request before execution so failures cannot shrink its denominator."""
    return [
        dict(
            unit_id=f"{unit['unit_id']}/{boundary}/{arm}/{block}",
            request_id=unit["unit_id"],
            cluster_id=world["cluster_id"],
            size=world["size"],
            boundary=boundary,
            arm=arm,
            block=block,
            mode=mode,
            seed=SEED,
            metric=0,
            denominator=1,
        )
        for world in worlds
        for boundary in prior.BOUNDARIES
        for arm in world["arm_order"]
        for unit in world["requests"]
        for block, mode in enumerate(world["mode_order"])
    ]


def replay(path: Path) -> bool:
    """Authenticate paid clocks and rebuild every public aggregate from retained rows."""
    try:
        value = load(reference(path))
        work, validation = load(value["work_reference"]), load(value["validation_reference"])
        if (
            validate_current_work_receipt(work["current_work_receipt"])
            or work["worlds"] != freeze()
            or not all(authenticate_row(r) for r in work["rows"])
        ):
            return False
        for ref in (
            value["code_config_hashes"]
            + value["source_artifact_hashes"]
            + value["raw_shard_hashes"]
        ):
            load(ref) if ref["path"].endswith(".json") else sha256_file(Path(ref["path"]))
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        from carnot.reporting.inline_clock_runner_8921 import artifact

        rebuilt = artifact(work, validation, value["work_reference"], value["validation_reference"])
        return canonical_hash(value) == canonical_hash(rebuilt)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def authenticate_row(row: Json) -> bool:
    """Recompute clocks and semantic bytes from their authenticated original parent."""
    if not row["raw_reference"]:
        return bool(
            row["status"] in ("failed", "censored")
            and not row["closure_valid"]
            and row["elapsed_ns"] == 0
        )
    raw = load(row["raw_reference"])
    if len(raw["rows"]) != 1:
        return False
    child, parent = raw["rows"][0], raw["receipt"]
    if (
        row["request_id"] != child["unit_id"]
        or row["cluster_id"] != child["cluster_id"]
        or row["arm"] != child["arm"]
        or row["size"] != child["size"]
        or child["instrumented"] != (False if row["mode"] == "off" else row["mode"])
    ):
        return False
    start, end = parent["started_monotonic_ns"], parent["ended_monotonic_ns"]
    begin, stop = child.get("block_start_ns", start), child.get("block_end_ns", start)
    a, b = child.get("start_ns", start), child.get("end_ns", start)
    partition = (
        prior.partition(child["endpoints"], a, b, endpoint_count=child["endpoint_count"])
        if row["mode"] != "off" and "endpoints" in child
        else {}
    )
    valid = (
        parent["passed"]
        and child["status"] != "failed"
        and start <= begin <= a < b <= stop <= end
        and (row["mode"] == "off" or partition.get("valid", False))
    )
    return bool(
        row["partition"] == partition
        and row["closure_valid"] == valid
        and row["elapsed_ns"] == (end - start if row["boundary"] == "cold" else stop - begin)
        and row["parent_elapsed_ns"] == end - start
        and row["startup_ns"] == begin - start
        and row["exit_ns"] == end - stop
        and row["child_remainder_ns"] == stop - begin - (b - a)
        and row["semantics"] == [{k: child[k] for k in prior.SEMANTICS}]
    )

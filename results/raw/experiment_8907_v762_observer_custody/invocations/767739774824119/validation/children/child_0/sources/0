"""REQ-REPORT-8907: resolve original observer operands without changing their seals."""

from pathlib import Path
import sys
from types import ModuleType
from typing import Any
from unittest.mock import patch

from carnot.reporting import buffered_clock_runner_8894 as original
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, SEED = "experiment_8907_v762_observer_custody", "exp8907-observer-custody", 7628907
CLI = f"scripts/experiments/{NAME}.py"
OBSERVER = "python/carnot/reporting/buffered_clock_methods_8894.py"
CURRENT_SPEC = ROOT / "openspec/capabilities/research-reporting/spec.md"
OWNED = [
    "python/carnot/reporting/observer_custody_8907.py",
    "python/carnot/reporting/observer_custody_runner_8907.py",
    CLI,
]
TESTS = ["tests/python/test_observer_custody_8907.py"]
BUNDLE = dict(
    path=str(
        ROOT / "results/raw/experiment_8903_v761_capstone/invocations/terminal-v761/work.json"
    ),
    sha256="sha256:7a7e72a2cfe8e423ee8a67b1b9fc63777bb4b4653efb43f52bb7861fa79b498e",
)
PIN = "sha256:3b18a631f41e40bb0cb672bb1a47be306436f0221995ae19f3001781a200f181"
CASES = (
    "current_spec_append",
    "retained_spec",
    "retained_code",
    "rehashed_clock",
    "missing_operand",
    "invalid_producer",
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose completed work without extending any fixed subprocess deadline."""
    print(f"[exp8907] phase={phase} completed={completed} pending={pending}", flush=True)


def require(condition: Any, reason: str) -> None:
    """A fresh checksum cannot authorize a changed historical assertion."""
    if not condition:
        raise ValueError(reason)


def retained_path(operand: Json, index: dict[str, Json]) -> Path:
    """Content identity survives path deduplication in the sealed execution bundle."""
    return Path(index[operand["sha256"]]["path"])


class CustodyError(ValueError):
    """Keep the exact failed external operand available to the terminal report."""

    def __init__(self, diagnostic: Json):
        self.diagnostic = diagnostic
        super().__init__(diagnostic["check"] + ":" + diagnostic["path"])


def checked_path(operand: Json, index: dict[str, Json]) -> Path:
    """Report missing bytes and mismatched bytes without substituting checkout files."""
    source = retained_path(operand, index)
    observed = sha256_file(source) if source.is_file() else None
    if observed != operand["sha256"]:
        raise CustodyError(
            dict(
                check="missing_file" if observed is None else "rejected_producer",
                upstream="exp8894-buffered-clock-methods",
                artifact_field="retained_operand.sha256",
                expected=operand["sha256"],
                observed=observed,
                path=str(source),
                hash=observed,
                original_path=operand.get("path"),
            )
        )
    return source


def producer_gate(value: Json, pin: str) -> None:
    """Keep missing and zero readiness distinct from an invalid producer."""
    require("observer_ready_score" in value, "missing_field:observer_ready_score")
    require(
        type(value["observer_ready_score"]) is int and value["observer_ready_score"] == 1,
        "observed_zero:observer_ready_score",
    )
    require(
        pin == PIN
        and value["experiment_id"] == 8894
        and value["task_id"] == "exp8894-buffered-clock-methods"
        and value["required_checks_passed"] is True
        and value["verdict_class"] == "circular_positive",
        "rejected_producer:original_identity",
    )


def open_observer(
    bundle: Json = BUNDLE, *, control_raw: Path | None = None
) -> tuple[Any, Path, Json]:
    """Bind only this consumer's namespace to already retained, authenticated bytes."""
    require(bundle["sha256"] == BUNDLE["sha256"], "rejected_producer:bundle_anchor")
    execution = load(bundle)
    require(execution["contract"]["ready"], "rejected_producer:execution_snapshot")
    slot = next(s for s in execution["slots"] if s["experiment_id"] == 8894)
    index = {r["sha256"]: r for r in slot["snapshots"]}
    primary = checked_path(
        dict(
            path=str(ROOT / "results/experiment_8894_v761_buffered_clock_methods.json"), sha256=PIN
        ),
        index,
    )
    value = load(dict(path=str(primary), sha256=PIN))
    producer_gate(value, sha256_file(primary))
    operands = value["source_artifact_hashes"] + value["code_config_hashes"]
    operands += value["raw_shard_hashes"] + [value["work_reference"], value["validation_reference"]]
    manifest, paths = [], {}
    for i, operand in enumerate(operands):
        source = checked_path(operand, index)
        paths[operand["path"]] = source
        manifest.append(
            dict(original_path=operand["path"], path=str(source), sha256=operand["sha256"])
        )
        if i % 1000 == 0:
            progress("authenticate_retained", i, len(operands) - i)
    equivalence = []
    implementation = {
        r["path"]: r for r in operands if r["path"].endswith(".py") and "/results/" not in r["path"]
    }
    for operand in implementation.values():
        if Path(operand["path"]).suffix == ".py":
            current = sha256_file(Path(operand["path"]))
            require(current == operand["sha256"], "owned_equivalence:" + operand["path"])
            equivalence.append(
                dict(
                    unit_id=operand["path"],
                    retained_sha256=operand["sha256"],
                    current_sha256=current,
                    passed=True,
                    criterion="exact_source_bytes",
                )
            )

    def resolved_load(ref: Json) -> Json:
        return load(dict(path=str(paths.get(ref["path"], Path(ref["path"]))), sha256=ref["sha256"]))

    source_module = ModuleType("retained_observer")
    source_module.__file__ = str(paths[str(ROOT / OBSERVER)])
    with patch.object(sys, "dont_write_bytecode", True):
        methods = namespace(source_module, "observer8907_methods")
    runner_path = str(ROOT / "python/carnot/reporting/buffered_clock_runner_8894.py")
    source_module.__file__ = str(paths[runner_path])
    with patch.object(sys, "dont_write_bytecode", True):
        runner = namespace(source_module, "observer8907_runner")
    methods.ROOT = ROOT
    methods.WORKER = str(paths[str(ROOT / methods.WORKER)])
    runner.m = methods
    runner.load = resolved_load
    runner.sha256_file = lambda path: sha256_file(paths.get(str(path), Path(path)))
    work = resolved_load(value["work_reference"])
    # Fresh controls install the original wheel and execute the retained worker.
    work_package = work["package_hashes"][0]
    package = dict(work_package, path=str(paths[work_package["path"]]))
    base_measure = runner.measure

    def measured(raw: Path, worlds: list[Json], ignored: Json, **kwargs: Any) -> Json:
        raw = control_raw if control_raw is not None else raw
        workers = raw / "retained_workers"
        workers.mkdir(parents=True)
        for name, source in paths.items():
            if Path(name).parent == ROOT / "scripts/experiments" and name.endswith(".py"):
                (workers / Path(name).name).write_bytes(source.read_bytes())
        methods.WORKER = str(workers / "buffered_clock_worker_8894.py")
        return dict(base_measure(raw, worlds, package, **kwargs))

    runner.measure = measured
    return (
        runner,
        primary,
        dict(
            retained_operand_manifest=manifest,
            observer_equivalence_rows=equivalence,
            original_receipt_hashes=[value["work_reference"], value["validation_reference"]],
            execution_snapshot=bundle,
            original_primary=dict(path=str(primary), sha256=PIN),
        ),
    )


def qualify(
    bundle: Json = BUNDLE, *, execute: bool = True, control_raw: Path | None = None
) -> Json:
    """Original replay reduces every retained clock before one fresh installed world."""
    progress("before_retained_replay")
    runner, primary, evidence = open_observer(bundle, control_raw=control_raw)
    require(runner.replay(primary, execute=execute), "owned_replay:original_assertions")
    value = load(dict(path=str(primary), sha256=PIN))
    require(
        value["repetition_count"] == 1 and len(value["future_world_manifest"]["worlds"]) == 144,
        "owned_replay:frozen_protocol",
    )
    progress("after_retained_replay", 1, 0)
    return dict(
        evidence,
        passed=True,
        repetition_count=1,
        future_world_count=144,
        future_manifest_checksum=canonical_hash(value["future_world_manifest"]),
        fresh_installed_control_executed=execute,
    )


def replay(path: Path) -> Json:
    """A rehashed headline still has to reduce from qualified original evidence."""
    from carnot.reporting.observer_custody_runner_8907 import verify

    try:
        return verify(path)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        return dict(passed=False, error=str(error))


def controls(raw: Path) -> list[Json]:
    """Attack private copies; original bytes and receipts remain untouched."""
    from carnot.reporting.current_work_receipt import atomic_json

    execution = load(BUNDLE)
    slot = next(s for s in execution["slots"] if s["experiment_id"] == 8894)
    original_resolve = retained_path
    rows: list[Json] = []
    for kind in CASES:
        progress("before_mutation_" + kind, len(rows), 6 - len(rows))
        source = next(
            r
            for r in slot["snapshots"]
            if (
                r["sha256"] == PIN
                if kind == "invalid_producer"
                else r["path"].endswith("/spec.md")
                if "spec" in kind
                else r["path"].endswith("/buffered_clock_methods_8894.py")
                if kind == "retained_code"
                else "/cold-" in r["path"]
            )
        )
        target = raw / kind / Path(source["path"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind in {"rehashed_clock", "invalid_producer"}:
            value = load(source)
            if kind == "invalid_producer":
                value["experiment_id"] = 8899
            else:
                value["rows"][0]["end_ns"] += 1
            value["reproducibility_checksum"] = canonical_hash(value)
            atomic_json(target, value)
        elif kind != "missing_operand":
            target.write_bytes(Path(source["path"]).read_bytes() + b"\n# unrelated later append\n")

        def resolve(operand: Json, index: dict[str, Json]) -> Path:
            return (
                target
                if operand["sha256"] == source["sha256"]
                else original_resolve(operand, index)
            )

        rejected, error = False, None
        with (
            patch.object(sys.modules[__name__], "CURRENT_SPEC", target)
            if kind == "current_spec_append"
            else patch.object(sys.modules[__name__], "retained_path", resolve)
        ):
            try:
                qualify(execute=False)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rejected, error = True, str(exc)
        rows.append(
            dict(
                unit_id=kind,
                cluster_id="custody_controls",
                arm="mutation",
                seed=SEED,
                metric=int(rejected == (kind != "current_spec_append")),
                denominator=1,
                status="completed",
                missing_reason=None,
                passed=rejected == (kind != "current_spec_append"),
                rejected=rejected,
                expected_rejected=kind != "current_spec_append",
                error=error,
                original_operand=source,
                mutated_operand=reference(target)
                if target.exists()
                else dict(path=str(target), sha256=None),
            )
        )
        progress("after_mutation_" + kind, len(rows), 6 - len(rows))
    return rows

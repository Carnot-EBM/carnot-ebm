"""REQ-REPORT-8404: immutable consumer custody qualifies process recovery only."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import direct_state_qualification_8390 as old
from carnot.reporting import historical_consumer_roots_8402 as authority
from carnot.reporting import v717_contract_runner as commands
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
base, s = old.base, old.s
ROOT = old.ROOT
NAME, TASK = "experiment_8404_v724_direct_state_qualification", "exp8404-direct-state-qualification"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_direct_state_qualification_8404.py"
OWNED = [
    "python/carnot/reporting/direct_state_qualification_8404.py",
    "python/carnot/reporting/direct_state_runner_8404.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8402_v724_historical_consumer_roots.json"
UPSTREAM_PIN = "sha256:94d4baf2872bcf2bc58a7dd7a96f431c33ccad339c272927f1fc9673306b944d"
ORIGINAL = "results/experiment_8390_v723_direct_state_qualification.json"
BARRIERS = old.BARRIERS
reference, progress = old.reference, old.progress


class QualifiedStore:
    """Observe shipped pointer publication and directory sync without changing writes."""

    def __init__(self, path: Path, frozen: Json) -> None:
        self._store = old.DirectStore(path, frozen)
        self._commit = self._store.commit
        self._store.commit = self.commit

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def commit(self, state: Json, hook: Any = lambda phase: None) -> None:
        """The post-pointer sync must be observed in a child, before acknowledgment."""
        publish, sync = s.atomic_json, s.sync_directory
        pointer_published = False

        def atomic(path: Path, value: Json) -> None:
            nonlocal pointer_published
            publish(path, value)
            pointer_published = True
            hook("pointer_publication")

        def directory(path: Path) -> None:
            sync(path)
            if pointer_published:
                hook("directory_fsync")

        with patch.object(s, "atomic_json", atomic), patch.object(s, "sync_directory", directory):
            self._commit(state, hook)


@contextmanager
def bindings() -> Iterator[None]:
    """Adapters apply only during this invocation; historical protocols stay untouched."""
    with (
        old.bindings(),
        patch.multiple(base, NAME=NAME, TASK=TASK, CLI=CLI, TEST=TEST, OWNED=OWNED),
        patch.object(s, "Store", QualifiedStore),
        patch.multiple(old, CLI=CLI, NAME=NAME, TASK=TASK),
    ):
        yield


def dependencies(root: Path, raw: Path) -> Json:
    """A pinned administrative source and independently authenticated family precede crashes."""
    result: Json = dict(failures=[], refs=[], historical_consumer_receipt={}, original_logs=[])
    try:
        primary = authority.checked(root / UPSTREAM, UPSTREAM_PIN)
        value = json.loads(primary.read_bytes())
        if value.get("historical_direct_consumers_ready_score") != 1:
            raise ValueError("historical_direct_consumers_ready_score")
        classification = authority.authenticate_family(primary, "direct")
        ref = value["historical_family_receipts"]["direct"]
        result["historical_consumer_receipt"] = dict(
            primary=reference(primary),
            family_reference=ref,
            classification=classification,
            historical_fixture_hashes=value["historical_fixture_hashes"],
        )
        result["refs"].extend([reference(primary), ref])
        protocol = json.loads(
            authority.checked(root / authority.PROTOCOL, authority.PROTOCOL_PIN).read_bytes()
        )
        authority.checked(root / authority.METHODS, authority.METHODS_PIN)
        for ref in protocol["preserved_protocols"]:
            authority.checked(Path(ref["path"]), ref["sha256"])
        result["refs"].extend(
            [
                reference(root / authority.PROTOCOL),
                reference(root / authority.METHODS),
                *protocol["preserved_protocols"],
            ]
        )
        original = authority.checked(root / ORIGINAL, authority.PRIMARY_PINS[8390])
        original_value = json.loads(original.read_bytes())
        result["refs"].append(reference(original))
        for receipt in original_value["validation_receipts"]:
            if not receipt["passed"]:
                log = authority.checked(Path(receipt["stdout_path"]), receipt["stdout_sha256"])
                target = raw / "original_logs" / (receipt["name"] + ".stdout")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(log.read_bytes())
                result["original_logs"].append(dict(reference(target), original_receipt=receipt))
                result["refs"].append(reference(target))
        result["historical_model_provenance"] = value["historical_model_provenance"]
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result["failures"].append(
            authority.failure(
                root / UPSTREAM,
                "authenticated_direct_family_and_inputs",
                True,
                None if not (root / UPSTREAM).exists() else str(error),
            )
        )
    return result


def unstarted(frozen: Json, raw: Path) -> list[Json]:
    """Preserve every intended arm when an external gate closes before observation."""
    return [
        dict(
            seed=frozen["seed"],
            arm=a,
            barrier=b,
            status="unstarted",
            passed=None,
            state_hash=None,
            actual_exit=None,
            latency_s=None,
            receipts=[],
            kill_reference=None,
            missing_reason="authenticated_historical_direct_family_or_current_authority_absent",
            absolute_metric=None,
            raw_numerator=None,
            raw_denominator=1,
        )
        for a, b in [
            ("uninterrupted", "none"),
            *[(a, b) for b in BARRIERS for a in ("restart", "two_readers")],
        ]
    ]


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Check independent consumer custody and current authority before fresh process deaths."""
    progress("preconditions_current_and_historical_before")
    deps = dependencies(root, raw)
    checked = authority.authority(root, raw / "current_authority")
    gates = [*deps["failures"], *checked["gate_check_summary"]]
    task = [t for t in checked["tasks"] if t["id"] == TASK]
    if len(task) != 1 or task[0]["MODEL_SPECS"] != []:
        gates.append(authority.failure(root / authority.DESIGN, "exact_current_task", TASK, task))
    with (
        patch.object(base.authority, "authority", lambda *args: checked),
        patch.object(base, "measure_trace", unstarted if gates else base.measure_trace),
    ):
        work = dict(base.measure(root, raw, private))
    work["gate_check_summary"] = [*gates, *work["gate_check_summary"]]
    work.update(
        root=str(root),
        current_authority=checked,
        input_ready=work["input_ready"] and not gates,
        historical_consumer_receipt=deps["historical_consumer_receipt"],
        original_failure_logs=deps["original_logs"],
        historical_model_provenance=deps.get("historical_model_provenance", []),
    )
    work["source_artifact_hashes"].extend(deps["refs"])
    atomic_json(Path(work["work_path"]), work)
    os.environ["CARNOT8404_WORK"] = work["work_path"]
    return work


def build(work: Json, receipts: list[Json]) -> Json:
    """Every barrier and authentic owned coverage are needed for deployment readiness."""
    value = dict(old.BASE_BUILD(work, receipts))
    complete = value["completed_count"] == 33 and all(r["passed"] for r in work["rows"])
    hooks = [r for r in work["rows"] if r["barrier"] == "directory_fsync"]
    observed = bool(hooks) and all(
        r["status"] == "completed"
        and r["actual_exit"] == -9
        and json.loads(Path(r["kill_reference"]["path"]).read_bytes())["directory_fsync"] is True
        for r in hooks
    )
    coverage = work.get("owned_coverage", {})
    covered = coverage_ready(coverage)
    ready = (
        value["required_checks_passed"]
        and not value["gate_check_summary"]
        and complete
        and observed
        and covered
        and work["input_ready"]
        and work["current_authority"]["activated"]
    )
    value.update(
        experiment_id=8404,
        task_id=TASK,
        milestone="2026.10.724",
        run_date="20261011",
        honest_verdict="complete_" + value["verdict_class"] + "_direct_state_qualification",
        intended_count=33,
        censored_count=33 - value["completed_count"],
        direct_state_ready_score=int(ready),
        historical_consumer_receipt=work["historical_consumer_receipt"],
        original_failure_logs=work["original_failure_logs"],
        current_authority=work["current_authority"],
        historical_model_provenance=work["historical_model_provenance"],
        directory_fsync_hook_executed=observed,
        owned_coverage=coverage,
        closed_exposed_utility=dict(H1=-0.00390625, H2=0, reopened=False),
    )
    value["acceptance_gates"].update(
        exact_recovery=complete,
        direct_ready=bool(ready),
        current_contract=work["current_authority"]["activated"],
        owned_coverage=covered,
        directory_fsync_hook=observed,
    )
    value["sample_size_budget"].update(barriers=list(BARRIERS), intended=33)
    value["raw_shard_hashes"].extend(work.get("coverage_shards", []))
    value["methodology_note"] = (
        "Shipped direct state and optimizer; five genuine process-death barriers, three constructed seeds and two pinned concurrent readers. Independent immutable Exp8402 family custody and genuine current owned coverage qualify deployment only. H1=-0.00390625 and H2=0 stay closed. No power-loss certification, generator updates or semantic benefit."
    )
    value["field_principles"].update(
        {
            k: "Bind actual invocation and sealed full-state recovery; engineering qualification grants no scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def coverage_ready(coverage: Json) -> bool:
    """Only the real hashed report can authorize coverage, including every owned file."""
    if not coverage:
        return False
    try:
        ref = coverage["reference"]
        report = json.loads(authority.checked(Path(ref["path"]), ref["sha256"]).read_bytes())
        return bool(
            dict(report["totals"], reference=ref) == coverage
            and report["totals"]["missing_lines"] == 0
            and report["totals"]["percent_covered"] == 100
            and all(
                report["files"][p]["summary"]["num_statements"] > 0
                and report["files"][p]["summary"]["missing_lines"] == 0
                for p in OWNED
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def replay_value(value: Json, private: Path) -> bool:
    """Recompute current authority and family custody before inherited primitive replay."""
    if value["experiment_id"] != 8404:
        return False
    digest = value.pop("reproducibility_checksum")
    if digest != canonical_hash(value):
        return False
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    sealed = [
        dict(r, exists=True, snapshot_path=r["path"]) for r in value["source_artifact_hashes"]
    ]
    with frozen_inputs(sealed, private / "inputs"):
        checked = authority.authority(Path(work["root"]), private / "authority")
    if any(
        checked[k] != work["current_authority"][k]
        for k in ("tasks", "activated", "canonical_tasks_sha256")
    ):
        return False
    family = work["historical_consumer_receipt"]
    if (
        family
        and authority.authenticate_family(Path(family["primary"]["path"]), "direct")
        != family["classification"]
    ):
        return False
    value["experiment_id"] = 8376
    value["reproducibility_checksum"] = canonical_hash(value)
    legacy = private / "legacy.json"
    atomic_json(legacy, value)

    def rebuilt(w: Json, receipts: list[Json]) -> Json:
        result = build(w, receipts)
        result["experiment_id"] = 8376
        result.pop("reproducibility_checksum")
        result["reproducibility_checksum"] = canonical_hash(result)
        return result

    with bindings(), patch.object(base, "build", rebuilt):
        return bool(old.BASE_REPLAY(legacy))


def replay(path: Path) -> bool:
    """Absent or changed operands fail in a fresh bounded interpreter."""
    try:
        with TemporaryDirectory(dir="/var/tmp", prefix="exp8404-cold-") as directory:
            return replay_value(json.loads(path.read_bytes()), Path(directory))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def controls(candidate: Path, raw: Path) -> list[Json]:
    """Keep positive, missing, deliberate error, rehashed tamper and corruption child evidence."""
    with patch.object(old.runner, "e", sys.modules[__name__]):
        return list(old.runner.controls(candidate, raw))


def state_controls(work: Json, raw: Path) -> list[Json]:
    """Fresh invalid-state children use this invocation's CLI and preserve old probes."""
    with bindings():
        return list(old.state_controls(work, raw))


def plan(private: Path) -> list[Json]:
    """Freeze genuine unit and child shards with immutable historical consumer roots."""
    with patch.object(commands, "m", sys.modules[__name__]):
        frozen = commands.manifest(private)
    os.environ["COVERAGE_RCFILE"] = str(private / "coverage.ini")
    os.environ["COVERAGE_FILE"] = str(private / ".coverage")
    frozen[0]["deadline"] = 900
    fixture = json.loads((ROOT / UPSTREAM).read_bytes())["historical_fixture_hashes"]
    manifest = Path(fixture["720"]["root"]).parent / "manifest.json"
    direct = authority.authenticate_family(ROOT / UPSTREAM, "direct")
    args = [
        str(ROOT / ".venv/bin/pytest"),
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "-p",
        "carnot.testing.historical_roots_8402",
        "--historical-fixture-manifest=" + str(manifest),
        "--historical-current-root="
        + str(Path(fixture["720"]["root"]).parents[1] / "current/pinned"),
        *direct["consumer_node_ids"],
        "--basetemp=" + str(private / "e2e020"),
    ]
    frozen.insert(
        2,
        dict(
            name="private_E2E020_direct_consumers",
            argv=args,
            deadline=1200,
            expected=0,
            scope="owned",
        ),
    )
    next(c for c in frozen if c["name"] == "coverage_combine")["argv"].append("--keep")
    return [dict(c, deadline_s=c["deadline"]) for c in frozen]


def main(argv: list[str] | None = None) -> int:
    """The thin CLI keeps historical dates fixed while accepting this invocation's date."""
    from carnot.reporting.direct_state_runner_8404 import main as run

    return int(run(argv))


worker = base.worker

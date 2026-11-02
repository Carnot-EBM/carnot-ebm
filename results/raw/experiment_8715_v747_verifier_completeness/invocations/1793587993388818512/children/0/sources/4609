"""REQ-REPORT-8390: current authority qualifies the shipped direct state.

Small adapters preserve the original state machine and frozen numerical policy.
Genuine process death checks storage recovery, never scientific benefit.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import direct_atomic_state_8376 as base
from carnot.reporting import direct_atomic_runner_8376 as runner
from carnot.reporting import v723_contract_methods as authority
from carnot.reporting import v717_contract_runner as commands
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import frozen_inputs
from carnot.verify import direct_atomic_state_8376 as s
from carnot.verify.direct_atomic_state_8376 import Store as DirectStore

Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK = "experiment_8390_v723_direct_state_qualification", "exp8390-direct-state-qualification"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_direct_state_qualification_8390.py"
OWNED = ["python/carnot/reporting/direct_state_qualification_8390.py", CLI]
MODEL_SPECS: list[Json] = []
BARRIERS = (
    "temporary_write",
    "file_fsync",
    "pointer_publication",
    "directory_fsync",
    "acknowledgment",
)
PROBES = (
    "valid",
    "stale",
    "duplicate",
    "coefficients",
    "missing",
    "pending",
    "issued",
    "applied",
    "cursor",
)
ORIGINAL = "results/experiment_8376_v722_direct_atomic_state.json"
ORIGINAL_PIN = "sha256:73a13119920df03c523053f516b06143a6f5cff18b7f7a6b1b5af1e67f1b5897"
reference, progress = base.reference, base.progress
BASE_BUILD, BASE_REPLAY = base.build, base.replay


class QualifiedStore:
    """Expose both publication barriers without changing any state or write order."""

    def __init__(self, path: Path, frozen: Json) -> None:
        self._store = DirectStore(path, frozen)
        self._commit = self._store.commit
        self._store.commit = self.commit

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def commit(self, state: Json, hook: Any = lambda phase: None) -> None:
        """Observe the existing pointer rename and its subsequent directory fsync."""
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
    """Bind only invocation-local adapters; old artifacts and protocols stay frozen."""

    def seal(path: Path, value: Json) -> None:
        if path.name == "kill.json":
            value["directory_fsync"] = value["barrier"] in ("directory_fsync", "acknowledgment")
        atomic_json(path, value)

    with (
        patch.multiple(base, NAME=NAME, TASK=TASK, CLI=CLI, TEST=TEST, OWNED=OWNED),
        patch.object(base.authority, "authority", authority.authority),
        patch.multiple(s, Store=QualifiedStore, BARRIERS=BARRIERS),
        patch.object(base, "atomic_json", seal),
    ):
        yield


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate current full objects separately from original blocked controls."""
    work = dict(base.measure(root, raw, private))
    checked = authority.authority(root, raw / "current_authority")
    work["current_authority"] = checked
    work["root"] = str(root)
    work["original_controls"] = {}
    for name, pin in [
        (ORIGINAL, ORIGINAL_PIN),
        (authority.PROTOCOL, authority.PROTOCOL_PIN),
        (authority.METHODS, authority.METHODS_PIN),
        (authority.ACTIVE, None),
        (authority.DESIGN, None),
        ("python/carnot/verify/direct_atomic_state_8376.py", None),
        ("python/carnot/reporting/direct_atomic_state_8376.py", None),
        ("python/carnot/reporting/direct_atomic_runner_8376.py", None),
        ("tests/python/test_direct_atomic_state_8376.py", None),
    ]:
        source = root / name
        if not source.is_file():
            work["gate_check_summary"].append(
                authority.failure(source, "input_available", True, None)
            )
            continue
        ref = reference(source)
        if pin and ref["sha256"] != pin:
            work["gate_check_summary"].append(
                authority.failure(source, "input_hash", pin, ref["sha256"])
            )
        saved = raw / "current_inputs" / (str(len(work["source_artifact_hashes"])) + ".bin")
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(source.read_bytes())
        work["source_artifact_hashes"].append(
            dict(
                reference(saved),
                source_path=str(source),
                imported_fields="current authority or original direct controls",
            )
        )
        if name == ORIGINAL:
            work["original_controls"] = json.loads(saved.read_bytes())
            gates = work["original_controls"]["acceptance_gates"]
            if not (gates["exact_recovery"] and gates["owned_validation"]):
                work["gate_check_summary"].append(
                    authority.failure(source, "original_state_controls", True, gates)
                )
    atomic_json(Path(work["work_path"]), work)
    os.environ["CARNOT8390_WORK"] = work["work_path"]
    return work


def build(work: Json, receipts: list[Json]) -> Json:
    """Require current authority and every measured arm before granting readiness."""
    value = dict(BASE_BUILD(work, receipts))
    completed = value["completed_count"]
    ready = (
        value["required_checks_passed"]
        and not value["gate_check_summary"]
        and completed == 33
        and work["input_ready"]
    )
    value.update(
        experiment_id=8390,
        task_id=TASK,
        milestone="2026.10.723",
        honest_verdict="complete_" + value["verdict_class"] + "_direct_state_qualification",
        intended_count=33,
        censored_count=33 - completed,
        direct_state_ready_score=int(ready),
        current_authority=work["current_authority"],
        original_state_controls=dict(
            exact_recovery=work["original_controls"]
            .get("acceptance_gates", {})
            .get("exact_recovery"),
            owned_validation=work["original_controls"]
            .get("acceptance_gates", {})
            .get("owned_validation"),
        ),
        historical_replay_required=False,
    )
    value["acceptance_gates"].update(
        exact_recovery=completed == 33 and all(r["passed"] for r in work["rows"]),
        direct_ready=bool(ready),
    )
    value["sample_size_budget"].update(barriers=list(BARRIERS), intended=33)
    value["methodology_note"] = (
        "Unchanged direct spline state, one writer and two pinned readers. Genuine SIGKILL at five barriers; three constructed seeds, eleven arms per seed. Complete V723 authority is independent of historical Exp8374 replay. H1=-0.00390625 and H2=0 remain closed exposed findings. No power-loss certification, table, refit, generator update or generalization claim."
    )
    value["field_principles"].update(
        {
            k: "Bind current authority and original state controls independently; process recovery grants no semantic benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reuse primitive replay after independently checking sealed current authority."""
    try:
        value = json.loads(path.read_bytes())
        if value["experiment_id"] != 8390:
            return False
        digest = value.pop("reproducibility_checksum")
        if digest != canonical_hash(value):
            return False
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        with TemporaryDirectory(dir="/var/tmp", prefix="exp8390-cold-") as directory:
            private = Path(directory)
            sealed = [
                dict(r, exists=True, snapshot_path=r["path"])
                for r in value["source_artifact_hashes"]
            ]
            with frozen_inputs(sealed, private / "inputs"):
                checked = authority.authority(Path(work["root"]), private / "authority")
            for field in ("tasks", "activated", "canonical_tasks_sha256"):
                if checked[field] != work["current_authority"][field]:
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
                return bool(BASE_REPLAY(legacy))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def probe(frozen: Path, path: Path, mode: str) -> int:
    """Reject invalid state in a new interpreter, even after an attacker repairs hashes."""
    try:
        trace = json.loads(frozen.read_bytes())
        store = QualifiedStore(path, trace)
        store.initialize()
        for event in trace["events"][:15]:
            store.apply(event)
        feedback = next(e for e in trace["events"] if e["kind"] == "feedback")
        if mode in ("stale", "duplicate"):
            event = (
                dict(feedback, id="stale-feedback")
                if mode == "stale"
                else dict(feedback, y=1 - feedback["y"])
            )
            store.apply(event)
        elif mode == "missing":
            (path / "current.json").unlink()
        elif mode != "valid":
            pointer = json.loads((path / "current.json").read_bytes())
            snapshot = path / pointer["file"]
            envelope = json.loads(snapshot.read_bytes())
            state = envelope["state"]
            if mode == "coefficients":
                state["head"]["coefficients"][0] = 1e300
            elif mode == "cursor":
                state["release_cursor"] += 1
            elif mode == "pending":
                state["pending"] = []
            else:
                ledger = state[mode]
                ledger[next(iter(ledger))]["version"] += 1
            envelope["sha256"] = canonical_hash(state)
            pointer["sha256"] = envelope["sha256"]
            atomic_json(snapshot, envelope)
            atomic_json(path / "current.json", pointer)
        store.read()
        return 0
    except (OSError, ValueError, KeyError, TypeError):
        return 1


def state_controls(work: Json, raw: Path) -> list[Json]:
    """Retain one bounded fresh-process receipt for every valid or invalid operand."""
    return (
        [
            child(
                "state_" + mode,
                [
                    sys.executable,
                    "-u",
                    str(ROOT / CLI),
                    "--probe",
                    work["traces"][0]["path"],
                    str(raw / mode),
                    mode,
                ],
                raw / "logs",
                deadline=60,
                expected=int(mode != "valid"),
            )
            for mode in PROBES
        ]
        if work["traces"]
        else []
    )


def plan(private: Path) -> list[Json]:
    """Use shipped subprocess coverage so only this invocation's new statements count."""
    with patch.object(commands, "m", sys.modules[__name__]):
        frozen = commands.manifest(private)
    os.environ["COVERAGE_RCFILE"] = str(private / "coverage.ini")
    os.environ["COVERAGE_FILE"] = str(private / ".coverage")
    frozen[0]["deadline"] = 600
    extra = dict(
        name="private_E2E020_direct_consumers",
        argv=[
            str(ROOT / ".venv/bin/pytest"),
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            "tests/python/test_hard_exit_learning_qualification_8206.py",
            "tests/python/test_local_consumer_qualification_8347.py",
            "tests/python/test_threshold_guard_8362.py",
            "tests/python/test_direct_atomic_state_8376.py",
            "--basetemp=" + str(private / "e2e020"),
        ],
        deadline=1200,
        expected=0,
        scope="owned",
    )
    frozen.insert(2, extra)
    return [dict(c, deadline_s=c["deadline"]) for c in frozen]


def main(argv: list[str] | None = None) -> int:
    """Reuse checked atomic publication with no model loads and a4800-second task cap."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--probe" in args:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--probe", nargs=3, required=True)
        frozen, state, mode = parser.parse_args(args).probe
        progress("fresh_state_probe_" + mode)
        return probe(Path(frozen), Path(state), mode)
    original_controls = runner.controls

    def controls(candidate: Path, raw: Path) -> list[Json]:
        value = json.loads(candidate.read_bytes())
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        return list(original_controls(candidate, raw)) + state_controls(
            work, raw / "state_controls"
        )

    with (
        bindings(),
        patch.multiple(
            runner,
            e=sys.modules[__name__],
            plan=plan,
            controls=controls,
            SCRATCH=Path.home() / ".cache/carnot-exp8390-private",
        ),
    ):
        return int(runner.main(args))


worker = base.worker

"""REQ-REPORT-9269-V789: qualify eight private replay shards with compact authenticated tables."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v788_contract as previous
from carnot.reporting import v789_contract_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
legacy = previous.legacy
_measure, _build, _replay = previous.measure, previous.build, previous.replay
_main, _binding = previous._main, previous.binding
SHARDS = [
    *previous.SHARDS,
    (
        "direct_consumers_v788",
        [
            "tests/python/test_v788_contract_9255.py",
            "tests/python/test_v788_capstone_9268.py",
            "tests/python/test_victris_energy_bot_9258.py",
        ],
    ),
]
CHECKS = [
    "owned_tests",
    *[name for name, _ in SHARDS],
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
]
MAX_PRIMARY_BYTES = 10_485_760


@contextmanager
def binding() -> Iterator[None]:
    """Restore all predecessor bindings so interleaved historical replay keeps its identity."""
    with (
        patch.multiple(
            previous,
            e=e,
            SHARDS=SHARDS,
            CHECKS=CHECKS,
            MODIFIED_MODULES=e.AFFECTED,
            measure=measure,
            build=build,
            replay=replay,
            qualify=qualify,
            receipts_pass=receipts_pass,
        ),
        _binding(),
    ):
        yield


def compact_guard(value: Json) -> None:
    """Reject oversized primary bytes before checkers or consumers allocate them."""
    if len((json.dumps(value, indent=2, sort_keys=True) + "\n").encode()) >= MAX_PRIMARY_BYTES:
        raise ValueError("primary_size")


def receipts_pass(work: Json, validation: Json) -> bool:
    """Statement coverage must name exactly the frozen owned list in the same order."""
    owned = work["scope"].get("owned_changed_paths", [])
    coverage = validation.get("owned_statement_coverage", {})
    if owned != e.OWNED or [r.get("path") for r in coverage.get("files", [])] != owned:
        return False
    with binding():
        return bool(legacy.receipts_pass(work, validation))


def measure(root: Path, raw: Path, date: str) -> Json:
    """Reuse private shard construction with V789 identity and immutable V788 inputs."""
    with binding():
        work = dict(_measure(root, raw, date))
    work["scope"]["validation_command_deadlines_s"]["direct_consumers_v788"] = 1200
    work["scope"]["deferred_validation_obligations"] = [
        dict(
            node_file=p,
            producer_reference=e.ref(e.ROOT / p),
            required_by=task,
            current_qualification_credit=0,
        )
        for p, task in [
            ("tests/python/test_causal_validation_9181.py", "exp9271-forward-qualification"),
            ("tests/python/test_v788_corpus_prefilter_9256.py", "exp9270-corpus-prefilter"),
        ]
    ]
    atomic_json(raw / "frozen_scope.json", work["scope"])
    atomic_json(raw / "work.json", work)
    return work


def qualify(work: Json) -> Json:
    """Supervise each private shard once; measure exactly the owned implementation files."""
    raw = Path(work["raw"])

    def run(command: Json) -> Json:
        return dict(
            legacy.lifecycle.child(
                command["name"],
                command["argv"],
                raw / "validation",
                deadline=work["scope"]["validation_command_deadlines_s"].get(command["name"], 600),
                heartbeat=30,
            )
        )

    commands = work["scope"]["commands"]
    with binding(), ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(run, commands[:9]))
        checks.extend(run(command) for command in commands[9:])
        for receipt in checks:
            receipt["scope_sha256"] = canonical_hash(work["scope"])
        coverage = legacy.lifecycle.measured(
            sorted(raw.glob(".coverage.*")),
            work["scope"]["owned_changed_paths"],
            raw / "combined_coverage",
        )
        health = legacy.lifecycle.child(
            "global_python_suite",
            [str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
            raw / "repository_health",
            deadline=600,
            heartbeat=30,
            scope="repository_health",
        )
    return dict(checks=checks, owned_statement_coverage=coverage, repository_health=health)


def build(work: Json, validation: Json) -> Json:
    """Only current affected checks qualify readiness; historical zeros remain historical."""
    with binding():
        value = dict(_build(work, validation))
    value.update(
        schema="carnot.v789_contract.v1",
        experiment_id=9269,
        task_id=e.TASK,
        milestone="2026.11.789",
        honest_verdict="complete_" + value["verdict_class"] + "_v789_contract",
        preserved_v788=work["inspection"]["preserved_v788"],
    )
    value["validation_scope_manifest"] = e.compact_table(
        Path(work["raw"]), "validation_scope_manifest", work["scope"]
    )
    value["methodology_note"] = (
        "Authenticate original human source and four immutable protocol owners; isolate V782-V788 roots; execute eight consumer shards and exact owned coverage; bind fourteen complete ordered V789 tasks. Bulky tables retain hashed sidecars. No model load, generation, queue activation or independent generalization."
    )
    value["acceptance_gates"]["replay_authority"]["condition"] = (
        "All eight private consumer shards, owned coverage at 100 percent with zero exclusions, scoped checks, E2E-018/021 and cold replay pass."
    )
    value["acceptance_gates"]["replay_authority"]["principle"] = (
        "Eight consumer shards, isolated V782-V788 roots and exact owned coverage must pass."
    )
    value["field_principles"].update(
        historical_input_roots="Hashed sidecars preserve every isolated V782-V788 suite without inflating primary JSON.",
        consumer_shard_rows="Eight complete shards retain every affected consumer test.",
        replay_authority_ready_score="Eight current shards and cold replay qualify immutable source authority.",
        validation_scope_manifest="Complete frozen code, tests and original obligations remain authenticated in sidecars.",
    )
    value["acceptance_gates"]["compactness"] = dict(
        condition="Primary JSON is strictly below 10 MB; all three compact sidecars authenticate",
        principle="Compact storage cannot omit validation obligations or weaken replay authority",
    )
    value["reproducibility_checksum"] = legacy.checksum(value)
    compact_guard(value)
    return value


def replay(path: Path) -> Json:
    """Reauthenticate compact tables and all four owners before the original cold reduction."""
    if path.stat().st_size >= MAX_PRIMARY_BYTES:
        raise ValueError("primary_size")
    value = json.loads(path.read_bytes())
    for name in e.TABLES:
        reference = value[name]
        sidecar = Path(reference["path"])
        if (
            e.ref(sidecar)["sha256"] != reference["sha256"]
            or len(json.loads(sidecar.read_bytes())) != reference["count"]
        ):
            raise ValueError("compact_sidecar_drift:" + name)
    with binding():
        return dict(_replay(path))


def main(argv: list[str] | None = None) -> int:
    """One monotonic budget reserves 1200 seconds for verification and 300 for publication."""
    args = sys.argv[1:] if argv is None else argv
    started = time.monotonic()
    deadline = min(float(os.environ.get("CARNOT_V789_DEADLINE", started + 4800)), started + 4800)
    supervisor = legacy.lifecycle.child

    def bounded(name: str, *args: Any, **kwargs: Any) -> Json:
        reserve = 300 if name in {"cold_replay", "adversarial", "strict_rows"} else 1500
        kwargs["deadline"] = min(
            kwargs.get("deadline", 600), max(0, deadline - reserve - time.monotonic())
        )
        return dict(supervisor(name, *args, **kwargs))

    mapped = ["20261117" if a == "20261121" else "invalid" if a == "20261117" else a for a in args]
    original_measure = measure
    with (
        patch(__name__ + ".measure", lambda root, raw, _: original_measure(root, raw, "20261121")),
        binding(),
        patch.object(legacy.lifecycle, "child", bounded),
        patch.dict(
            os.environ, CARNOT_V783_DEADLINE=str(deadline), CARNOT_V789_DEADLINE=str(deadline)
        ),
    ):
        return int(_main(mapped))

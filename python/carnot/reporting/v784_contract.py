"""REQ-REPORT-9208-V784: qualify corrected replay without changing historical outcomes."""

from collections.abc import Iterator
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v783_contract as previous
from carnot.reporting import v784_contract_evidence as e
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.protocol_owners import OWNERS, validate

Json = dict[str, Any]
legacy = previous.legacy
_measure, _build, _main = previous.measure, previous.build, previous.main
_base_replay = previous.prior._replay
AFFECTED = [
    "tests/python/test_v783_contract_9194.py",
    "tests/python/test_v783_capstone_9207.py",
    "tests/python/test_v782_contract_9180.py",
    "tests/python/test_v782_capstone_9193.py",
    "tests/python/test_causal_validation_9181.py",
]


@contextmanager
def binding() -> Iterator[None]:
    """Reuse checked publication while keeping each protocol's owner independent."""
    with (
        patch.multiple(previous, e=e, measure=measure, build=build, replay=replay, qualify=qualify),
        previous.binding(),
    ):
        yield


def protocol_rows(work: Json) -> list[Json]:
    """Even unavailable original protocol owners leave explicit terminal blocked operands."""
    rows = []
    for protocol, owner in OWNERS.items():
        try:
            rows.append(validate(work, protocol))
        except (OSError, ValueError, KeyError) as error:
            path = Path(owner["path"])
            observed = e.ref(path)["sha256"] or "missing_file"
            if observed == owner["sha256"]:
                raise ValueError("protocol_drift:" + protocol) from error
            rows.append(
                dict(
                    protocol=protocol,
                    owning_configuration=owner,
                    original_hash=owner["sha256"],
                    sidecar=work["protocol_sidecars"].get(protocol + "_reference"),
                    cold_replay_outcome="blocked",
                    gate=e.check(path, "sha256", owner["sha256"], observed, "protocol_owner"),
                    error=str(error),
                )
            )
    return rows


def measure(root: Path, raw: Path, date: str) -> Json:
    """Freeze the complete affected commands and original private roots before validation."""
    with binding(), patch.object(e, "HISTORICAL_BASE", raw / "historical", create=True):
        work = dict(_measure(root, raw, date))
    work["protocol_owner_rows"] = protocol_rows(work)
    command = next(c for c in work["scope"]["commands"] if c["name"] == "direct_consumers")
    command["argv"] = list(dict.fromkeys([*command["argv"], *AFFECTED]))
    modified = [
        *AFFECTED,
        "python/carnot/reporting/v782_contract.py",
        "python/carnot/reporting/v783_contract.py",
        "python/carnot/reporting/v779_contract_evidence.py",
    ]
    for command in work["scope"]["commands"]:
        if command["name"] in ["ruff_check", "ruff_format"]:
            command["argv"].extend(modified)
        elif command["name"] == "strict_mypy":
            command["argv"].extend(modified[-3:])
        elif command["name"] == "spec_coverage":
            command["argv"].extend(AFFECTED)
    # Current commands qualify changed code; retained manifests still identify every old obligation.
    work["scope"]["code_test_hashes"].extend(e.ref(e.ROOT / p) for p in modified)
    work["scope"]["owned_changed_paths"] = modified[-3:]
    work["scope"]["validation_command_deadlines_s"] = dict(
        owned_tests=1200, direct_consumers=1200, global_python_suite=600
    )
    cfg = Path(work["scope"]["coverage_config_reference"]["path"])
    direct = next(c for c in work["scope"]["commands"] if c["name"] == "direct_consumers")
    direct["argv"] = [
        "env",
        "COVERAGE_FILE=" + str(raw / ".coverage"),
        "COVERAGE_RCFILE=" + str(cfg),
        direct["argv"][0],
        "-m",
        "coverage",
        "run",
        "--rcfile=" + str(cfg),
        "-m",
        *direct["argv"][2:],
    ]
    cfg.write_text(
        cfg.read_text().replace(
            "[report]", "".join("    " + str(e.ROOT / p) + "\n" for p in modified[-3:]) + "[report]"
        )
    )
    work["scope"]["coverage_config_reference"] = e.ref(cfg)
    work["scope"]["validation_obligations"] = work["inspection"]["validation_obligations"]
    atomic_json(raw / "frozen_scope.json", work["scope"])
    atomic_json(raw / "work.json", work)
    return work


def qualify(work: Json) -> Json:
    """Execute owned coverage and all affected suites; record broad health exactly once."""
    raw = Path(work["raw"])

    def run(command: Json) -> Json:
        """Each complete suite keeps supervised logs and the shared parent deadline."""
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
    with binding(), ThreadPoolExecutor(max_workers=2) as pool:
        checks = list(pool.map(run, commands[:2]))
        checks.extend(run(command) for command in commands[2:])
        for receipt in checks:
            receipt["scope_sha256"] = canonical_hash(work["scope"])
        coverage = legacy.lifecycle.measured(
            sorted(raw.glob(".coverage.*")), e.OWNED, raw / "combined_coverage"
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
    """Only complete corrected executions qualify readiness; old zero gates stay zero."""
    with binding():
        value = dict(_build(work, validation))
    value.update(
        schema="carnot.v784_contract.v1",
        experiment_id=9208,
        task_id=e.TASK,
        milestone="2026.11.784",
        honest_verdict="complete_" + value["verdict_class"] + "_v784_contract",
        replay_authority_ready_score=int(
            value["required_checks_passed"] and value["verdict_class"] == "null"
        ),
        protocol_owner_rows=work["protocol_owner_rows"],
        historical_input_roots=work["inspection"]["historical_input_roots"],
        preserved_v783=work["inspection"]["preserved_v783"],
        target_identifiability_rubric=e.FROZEN["target_identifiability_rubric"],
        full_task_objects=work["inspection"]["contract"]["tasks"],
    )
    if not work["inspection"]["contract"]["rows"]:
        for index, row in enumerate(value["rows"]):
            row.update(unit_id=f"exp{9208 + index}", group_id=f"exp{9208 + index}")
    value["acceptance_gates"].update(
        replay_authority=dict(
            condition="All original V783 owned, V782 affected and new rejecting controls pass in fresh private processes with100 percent owned statements and zero exclusions",
            principle="Task agreement and historical receipts alone grant no replay readiness",
        ),
        target_identifiability=dict(
            condition="Freeze present/absent/ambiguous original evidence before prediction work",
            principle="Missing human taxonomy evidence cannot be replaced by guessed categories",
        ),
    )
    value["canonical_task_hash"] = canonical_hash(value["full_task_objects"])
    value["protocol_owner_rows"] = [
        dict(
            row,
            cold_replay_outcome=(
                "passed_current_private_cli"
                if value["replay_authority_ready_score"]
                else "not_qualified"
            ),
        )
        for row in work["protocol_owner_rows"]
    ]
    value["methodology_note"] = (
        "Authenticate original source and immutable protocol owners; compare fourteen complete ordered current task objects; execute all affected original assertions in isolated historical roots. Preserve V783 failures and zero gates. No model load, generation, learning, activation or independent generalization."
    )
    value["reproducibility_checksum"] = legacy.checksum(value)
    return value


def replay(path: Path) -> Json:
    """A fresh interpreter must recover original authority and reject rehashed false protocols."""
    value = json.loads(path.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    if protocol_rows(work) != work["protocol_owner_rows"]:
        raise ValueError("protocol_owner_drift")
    with (
        binding(),
        patch.object(e, "HISTORICAL_BASE", Path(work["raw"]) / "historical", create=True),
    ):
        return dict(_base_replay(path))


def main(argv: list[str] | None = None) -> int:
    """Keep real checker publication, private-output refusal and supervised deadlines."""
    with binding():
        return int(_main(argv))

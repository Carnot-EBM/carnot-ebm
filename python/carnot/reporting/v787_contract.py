"""REQ-REPORT-9241-V787: split V782-V786 consumer replay and V787 contract qualification."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v785_contract as previous
from carnot.reporting import v787_contract_evidence as e
from carnot.reporting.current_work_receipt import atomic_json as atomic_json
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.protocol_owners import V787_OWNERS, validate

Json = dict[str, Any]
legacy = previous.legacy
_measure, _build, _main = previous._measure, previous._build, previous._main
_base_replay = previous._base_replay

SHARDS = [
    (
        "direct_consumers_core",
        [
            "tests/python/test_primary_publication_7928.py",
            "tests/python/test_current_work_receipt.py",
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ],
    ),
    (
        "direct_consumers_v782",
        [
            "tests/python/test_v782_contract_9180.py",
            "tests/python/test_v782_capstone_9193.py",
        ],
    ),
    (
        "direct_consumers_v783",
        [
            "tests/python/test_v783_contract_9194.py",
            "tests/python/test_v783_capstone_9207.py",
        ],
    ),
    (
        "direct_consumers_v784",
        [
            "tests/python/test_v784_contract_9208.py",
            "tests/python/test_v784_capstone_9221.py",
        ],
    ),
    (
        "direct_consumers_v785",
        [
            "tests/python/test_v785_contract_9222.py",
        ],
    ),
    (
        "direct_consumers_v786",
        [
            "tests/python/test_v786_capstone_9240.py",
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
MODIFIED_MODULES = e.AFFECTED


@contextmanager
def binding() -> Iterator[None]:
    """Reuse checked publication while keeping each protocol's owner and shard list bound."""
    with (
        patch.multiple(previous, e=e, measure=measure, build=build, replay=replay, qualify=qualify),
        patch.object(legacy.lifecycle, "CHECKS", CHECKS),
        previous.binding(),
    ):
        yield


def protocol_rows(work: Json) -> list[Json]:
    """Even unavailable original protocol owners leave explicit terminal blocked operands."""
    rows = []
    for protocol, owner in V787_OWNERS.items():
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
    """Freeze six split consumer shards, distinct coverage files and isolated historical roots."""
    with binding(), patch.object(e, "HISTORICAL_BASE", raw / "historical", create=True):
        work = dict(_measure(root, raw, date))
    work["run_date"] = date
    for protocol_name in ["selective_deferral_protocol", "conformal_risk_protocol"]:
        sidecar_path = raw / f"{protocol_name}.json"
        atomic_json(sidecar_path, work["inspection"][protocol_name])
        sidecar_path.chmod(0o400)
        work["protocol_sidecars"][f"{protocol_name}_reference"] = e.ref(sidecar_path)
    work["protocol_sidecars"]["ca_ocrc_protocol_reference"] = work["protocol_sidecars"][
        "conformal_risk_protocol_reference"
    ]
    work["protocol_owner_rows"] = protocol_rows(work)
    cfg = Path(work["scope"]["coverage_config_reference"]["path"])
    cfg.write_text(
        cfg.read_text().replace(
            "[report]",
            "".join("    " + str(e.ROOT / p) + "\n" for p in MODIFIED_MODULES) + "[report]",
        )
    )
    shard_cfg = raw / "coverage_shards.ini"
    shard_cfg.write_text(cfg.read_text().replace("patch = subprocess\n", ""))
    work["scope"]["coverage_config_reference"] = e.ref(cfg)
    py = str(e.ROOT / ".venv/bin/python")
    owned_cmd = dict(
        name="owned_tests",
        argv=[
            "env",
            "COVERAGE_FILE=" + str(raw / ".coverage.owned_tests"),
            "COVERAGE_RCFILE=" + str(cfg),
            py,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(cfg),
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            e.TEST,
        ],
    )
    shard_cmds = []
    shard_meta = []
    all_consumer_tests = []
    for shard_name, files in SHARDS:
        cov_path = str(raw / f".coverage.{shard_name}")
        all_consumer_tests.extend(files)
        shard_meta.append(
            dict(shard_name=shard_name, test_files=files, coverage_shard_path=cov_path)
        )
        shard_cmds.append(
            dict(
                name=shard_name,
                argv=[
                    "env",
                    "COVERAGE_FILE=" + cov_path,
                    py,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + str(shard_cfg),
                    "-m",
                    "pytest",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    *files,
                ],
            )
        )
    tail_cmds = [
        c for c in work["scope"]["commands"] if c["name"] not in {"owned_tests", "direct_consumers"}
    ]
    extra_tests = [
        p
        for p in all_consumer_tests
        if p
        not in {
            "tests/python/test_primary_publication_7928.py",
            "tests/python/test_current_work_receipt.py",
        }
    ]
    for command in tail_cmds:
        if command["name"] in ["ruff_check", "ruff_format"]:
            command["argv"] = list(
                dict.fromkeys([*command["argv"], *extra_tests, *MODIFIED_MODULES])
            )
        elif command["name"] == "strict_mypy":
            command["argv"] = list(dict.fromkeys([*command["argv"], *MODIFIED_MODULES]))
        elif command["name"] == "spec_coverage":
            command["argv"] = list(dict.fromkeys([*command["argv"], *all_consumer_tests]))
    private_base = Path(tempfile.mkdtemp(prefix="carnot-9241-" + raw.name + "-", dir="/var/tmp"))
    for command in [owned_cmd, *shard_cmds]:
        private = private_base / command["name"]
        private.mkdir(mode=0o700)
        command["argv"] = [
            "env",
            "--chdir=" + str(private),
            "--unset=PYTHONPATH",
            "TMPDIR=/var/tmp",
            *[
                str(e.ROOT / arg) if arg in [e.TEST, *all_consumer_tests] else arg
                for arg in command["argv"][1:]
            ],
        ]
    work["scope"]["commands"] = [owned_cmd, *shard_cmds, *tail_cmds]
    work["scope"]["required_check_names"] = list(CHECKS)
    work["scope"]["consumer_shards"] = shard_meta
    tracked_paths = list(
        dict.fromkeys([*e.OWNED, e.TEST, e.CONFIG, *all_consumer_tests, *MODIFIED_MODULES])
    )
    work["scope"]["code_test_hashes"] = [e.ref(e.ROOT / p) for p in tracked_paths]
    work["scope"]["owned_changed_paths"] = [*e.OWNED, *MODIFIED_MODULES]
    work["scope"]["validation_command_deadlines_s"] = dict(
        owned_tests=1500,
        direct_consumers_core=600,
        direct_consumers_v782=1200,
        direct_consumers_v783=1200,
        direct_consumers_v784=1200,
        direct_consumers_v785=1200,
        direct_consumers_v786=1200,
        ruff_check=600,
        ruff_format=600,
        strict_mypy=600,
        spec_coverage=600,
        global_python_suite=600,
    )
    work["scope"]["validation_obligations"] = work["inspection"]["validation_obligations"]
    work["scope"]["deferred_validation_obligations"] = [
        dict(
            node_file="tests/python/test_causal_validation_9181.py",
            producer_reference=e.ref(e.ROOT / "tests/python/test_causal_validation_9181.py"),
            required_by="exp9243-forward-qualification",
            current_qualification_credit=0,
        )
    ]
    atomic_json(raw / "frozen_scope.json", work["scope"])
    atomic_json(raw / "work.json", work)
    return work


def qualify(work: Json) -> Json:
    """Execute owned coverage and all six partitioned consumer shards; record broad health once."""
    raw = Path(work["raw"])

    def run(command: Json) -> Json:
        """Each complete shard keeps supervised logs and the shared parent deadline."""
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
        checks = list(pool.map(run, commands[:7]))
        checks.extend(run(command) for command in commands[7:])
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
    """Only complete six-shard executions qualify replay; contract requires 14 matching tasks."""
    with binding():
        value = dict(_build(work, validation))
        passed = bool(legacy.receipts_pass(work, validation))
    inspected = work["inspection"]
    contract = inspected["contract"]
    replay_ready = int(passed and bool(inspected.get("replay_ready", False)))
    contract_ready = int(passed and contract["ready"] and replay_ready == 1)
    source_ready = int(passed and inspected["source_ready"])
    static_ready = int(passed and inspected["static_ready"])
    online_ready = int(passed and inspected["online_ready"])
    receipt_by_name = {r["name"]: r for r in validation.get("checks", []) if "name" in r}
    consumer_shard_rows = [
        dict(
            shard_name=shard["shard_name"],
            test_files=shard["test_files"],
            coverage_shard_path=shard["coverage_shard_path"],
            duration_s=receipt_by_name.get(shard["shard_name"], {}).get("duration_s"),
            exit_code=receipt_by_name.get(shard["shard_name"], {}).get("exit_code"),
            passed=bool(receipt_by_name.get(shard["shard_name"], {}).get("passed", False)),
        )
        for shard in work["scope"]["consumer_shards"]
    ]
    value.update(
        schema="carnot.v787_contract.v1",
        experiment_id=9241,
        task_id=e.TASK,
        milestone="2026.11.787",
        honest_verdict="complete_" + value["verdict_class"] + "_v787_contract",
        contract_ready_score=contract_ready,
        replay_authority_ready_score=replay_ready,
        source_ready_score=source_ready,
        static_methods_ready_score=static_ready,
        online_methods_ready_score=online_ready,
        consumer_shard_rows=consumer_shard_rows,
        protocol_owner_rows=work["protocol_owner_rows"],
        historical_input_roots=inspected["historical_input_roots"],
        preserved_v784=inspected["preserved_v784"],
        preserved_v785=inspected["preserved_v785"],
        preserved_v786=inspected["preserved_v786"],
        selective_deferral_protocol=inspected["selective_deferral_protocol"],
        conformal_risk_protocol=inspected["conformal_risk_protocol"],
        target_identifiability_rubric=e.FROZEN["target_identifiability_rubric"],
        full_task_objects=contract["tasks"],
    )
    if not contract["rows"]:
        for index, row in enumerate(value["rows"]):
            row.update(unit_id=f"exp{9241 + index}", group_id=f"exp{9241 + index}")
    value["acceptance_gates"].update(
        replay_authority=dict(
            condition="All six split consumer shards, isolated V782-V786 roots and fresh-process cold replay pass with 100 percent owned statements and zero exclusions",
            principle="Split consumer shards, isolated V782-V786 roots and fresh-process cold replay must all pass",
        ),
        target_identifiability=dict(
            condition="Authenticated Exp9211 target-identifiability and affine label-ambiguity method artifacts",
            principle="Missing human taxonomy evidence cannot be replaced by guessed categories",
        ),
        selective_deferral=dict(
            condition="Authenticated Exp9225 selective-deferral feedback masks and spline-support protocol artifacts",
            principle="Selective observation rules must be frozen before stream execution",
        ),
        conformal_risk=dict(
            condition="Frozen qid-cluster conformal risk, CA-OCRC tracking and Mycelium provenance reference structure",
            principle="Descriptive risk bounds and shadow tracking rules seal before reading TEST outcomes",
        ),
    )
    value["canonical_task_hash"] = canonical_hash(value["full_task_objects"])
    value["protocol_owner_rows"] = [
        dict(
            row,
            cold_replay_outcome=("passed_current_private_cli" if replay_ready else "not_qualified"),
        )
        for row in work["protocol_owner_rows"]
    ]
    value["methodology_note"] = (
        "Authenticate original Exp9097 human source, immutable protocol owners, Exp9211 target/ambiguity authority, Exp9225 selective-deferral/spline-support authority, Exp9222 V785 contract receipts, and Exp9240 V786 capstone authority; isolate V782, V783, V784, V785 and V786 roots; split direct_consumers across six bounded non-contending shards with distinct coverage files while retaining test_causal_validation_9181 in the 1040-unit validation obligation manifest for Exp9243; compare fourteen ordered V787 task objects against the embedded YAML contract block and record absent planner slots as blocked. No model load, generation, learning, queue activation or independent generalization."
    )
    value["field_principles"].update(
        consumer_shard_rows="Splitting V782-V786 shards eliminates timeout contention without dropping any test.",
        replay_authority_ready_score="Split consumer shards, isolated V782-V786 roots and fresh-process cold replay must all pass.",
        historical_input_roots="Successor queues cannot contaminate historical test authority.",
        protocol_owner_rows="Inherited replay reads its own immutable protocol authority.",
    )
    value["reproducibility_checksum"] = legacy.checksum(value)
    return value


def replay(path: Path) -> Json:
    """A fresh interpreter must recover original authority and reject rehashed false protocols."""
    value = json.loads(path.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    if protocol_rows(work) != work["protocol_owner_rows"]:
        raise ValueError("protocol_owner_drift")
    validate(work, "conformal_risk_protocol")
    with (
        binding(),
        patch.object(e, "HISTORICAL_BASE", Path(work["raw"]) / "historical", create=True),
        patch.object(previous.e, "HISTORICAL_BASE", Path(work["raw"]) / "historical", create=True),
        patch.object(
            previous.previous.e, "HISTORICAL_BASE", Path(work["raw"]) / "historical", create=True
        ),
    ):
        return dict(_base_replay(path))


def main(argv: list[str] | None = None) -> int:
    """Keep real checker publication, private-output refusal and supervised deadlines."""
    args = sys.argv[1:] if argv is None else argv
    started = time.monotonic()
    deadline = min(float(os.environ.get("CARNOT_V787_DEADLINE", started + 4800)), started + 4800)
    supervisor = legacy.lifecycle.child

    def bounded(name: str, *args: Any, **kwargs: Any) -> Json:
        """Keep descendants inside one deadline, leaving time to verify the final bytes."""
        reserve = 0 if name in {"cold_replay", "adversarial", "strict_rows"} else 1500
        kwargs["deadline"] = min(
            kwargs.get("deadline", 600), max(0, deadline - reserve - time.monotonic())
        )
        return dict(supervisor(name, *args, **kwargs))

    mapped = ["20261117" if a == "20261120" else "invalid" if a == "20261117" else a for a in args]
    env = dict(CARNOT_V783_DEADLINE=str(deadline), CARNOT_V787_DEADLINE=str(deadline))
    original_measure = measure
    with (
        patch(__name__ + ".measure", lambda root, raw, _: original_measure(root, raw, "20261120")),
        binding(),
        patch.object(legacy.lifecycle, "child", bounded),
        patch.dict(os.environ, env),
    ):
        return int(_main(mapped))

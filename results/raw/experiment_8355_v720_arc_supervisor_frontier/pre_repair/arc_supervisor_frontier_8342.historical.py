"""REQ-REPORT-8342: preserve qualified ARC mechanics while binding current authority.

An empty supervisor ledger can be a valid observation. Execution qualification
and new outcome support therefore remain separate quantities.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil
from types import FunctionType, SimpleNamespace
from typing import Any, cast

from carnot.reporting import arc_supervisor_artifact_8328 as prior_artifact
from carnot.reporting import arc_supervisor_execution_8328 as prior_execution
from carnot.reporting import arc_supervisor_frontier_8328 as prior_reader
from carnot.reporting import v719_contract_replay as authority
from carnot.reporting.current_work_receipt import sha256_file

Json = dict[str, Any]
ROOT = prior_reader.ROOT
NAME = "experiment_8342_v719_arc_supervisor_frontier"
CLI = "scripts/experiments/" + NAME + ".py"
OUTPUT = ROOT / "results" / (NAME + ".json")
TEST = "tests/python/test_arc_supervisor_frontier_8342.py"
OWNED = ["python/carnot/reporting/arc_supervisor_frontier_8342.py", CLI, *prior_artifact.OWNED[:3]]
MODEL_SPECS: list[Json] = []
BUDGET = prior_artifact.BUDGET
json_document = prior_reader.json_document


def progress(phase: str) -> None:
    """Flush each boundary so the parent can observe work before children finish."""
    print(f"[exp8342] phase={phase}", flush=True)


def bind(original: Callable[..., Any], **overrides: Any) -> Callable[..., Any]:
    """Keep the original statements and validators while changing invocation globals."""
    function = cast(FunctionType, original)
    invocation = {
        key: globals()[key]
        for key in [
            "NAME",
            "CLI",
            "OUTPUT",
            "TEST",
            "OWNED",
            "BUDGET",
            "build",
            "replay",
            "plan",
            "controls",
            "publish",
        ]
    }
    return cast(
        Callable[..., Any],
        FunctionType(
            function.__code__,
            dict(function.__globals__) | invocation | overrides,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        ),
    )


reader = SimpleNamespace(**(vars(prior_reader) | dict(authority=authority, progress=progress)))


def measure(raw: Path, private: Path, precondition_failures: list[Json] | None = None) -> Json:
    """Authenticate current tasks directly, keeping the historical frontier untouched."""
    probe = private / "exp8342-private-probe"
    failures = list(precondition_failures or [])
    try:
        probe.write_bytes(b"private scratch")
        if probe.read_bytes() != b"private scratch":
            raise ValueError("private_scratch_rw")
    except (OSError, ValueError) as error:
        failures.append(
            dict(
                prior_reader.native.authority.operand(
                    private,
                    "private_scratch_rw",
                    True,
                    str(error),
                    None,
                ),
                passed=False,
            )
        )
    work = cast(
        Json,
        bind(prior_reader.measure, authority=authority, progress=progress)(raw, private, failures),
    )
    protocol = ROOT / authority.PROTOCOL
    digest = sha256_file(protocol) if protocol.is_file() else None
    if digest != authority.base.PIN:
        work["failures"].append(
            dict(
                prior_reader.native.authority.operand(
                    protocol,
                    "protocol_sha256",
                    authority.base.PIN,
                    digest,
                    digest,
                ),
                passed=False,
            )
        )
    labels = [
        *OWNED,
        TEST,
        "python/carnot/reporting/v719_contract_replay.py",
        "python/carnot/reporting/v717_contract_methods.py",
        "python/carnot/reporting/v709_execution.py",
        "openspec/change-proposals/research-roadmap-v718-preserved-20261009.md",
    ]
    manifest = raw / "command_manifest.json"
    paths = [ROOT / label for label in labels] + ([manifest] if manifest.is_file() else [])
    for index, source in enumerate(paths):
        digest = sha256_file(source)
        saved = raw / "inputs" / (f"v719-{index}-" + digest[7:] + ".bin")
        saved.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, saved)
        work["source_artifact_hashes"][str(source)] = digest
        work["snapshots"][str(saved)] = digest
    return work


reader.measure = measure
artifact = SimpleNamespace(
    **(
        vars(prior_artifact)
        | dict(
            m=reader,
            NAME=NAME,
            CLI=CLI,
            OUTPUT=OUTPUT,
            TEST=TEST,
            OWNED=OWNED,
        )
    )
)


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Current identity does not upgrade historical failures or manufacture outcomes."""
    value = cast(Json, bind(prior_artifact.build, m=reader)(work, receipts, raw, output))
    value.update(
        experiment_id=8342, task_id="exp8342-arc-supervisor-frontier", milestone="2026.10.719"
    )
    return value


def replay(value: Json) -> bool:
    """Fresh replay uses the unchanged environment joins and all primitive bindings."""
    return bool(bind(prior_artifact.replay, m=reader)(value))


def plan(private: Path) -> list[Json]:
    """Reuse the existing executor; its manifest measures real CLI children as well."""
    return cast(list[Json], bind(prior_execution.plan, a=artifact, m=reader)(private))


def controls(value: Json, raw: Path) -> list[Json]:
    """Real cold children distinguish valid bytes from altered and rehashed claims."""
    return cast(list[Json], bind(prior_execution.controls, m=reader)(value, raw))


def publish(value: Json, work: Json, output: Path, raw: Path) -> None:
    """The existing publisher retains findings and validates failure recovery atomically."""
    bind(prior_execution.publish, m=reader)(value, work, output, raw)


def run(output: Path, private: Path, *, fixture: bool = False) -> int:
    """Qualification runs before the one current inspection, with no game or model calls."""
    return int(bind(prior_execution.run, a=artifact, m=reader)(output, private, fixture=fixture))

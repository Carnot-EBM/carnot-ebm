"""REQ-REPORT-8482: current authority and independent panel seals govern live collection.

The existing policy and recorder supply all observations. Invocation aliases
change identity only; public history cannot grant hidden-game or solve credit.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import arc_generalization_panel_8468 as old
from carnot.reporting import v729_contract_methods as authority
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK = "experiment_8482_v729_arc_panel_qualification", "exp8482-arc-panel-qualification"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_arc_panel_qualification_8482.py"
OUTPUT = ROOT / "results" / (NAME + ".json")
OWNED = [
    "python/carnot/reporting/arc_panel_qualification_8482.py",
    "python/carnot/reporting/arc_panel_validation_8482.py",
    CLI,
]
TASK_PIN = "sha256:1b5071c1cb762daaeb3936b268b451ccd6205b2f86f62608713a24d8190548b2"
PRIOR = "results/experiment_8468_v728_arc_generalization_panel.json"
PRIOR_PIN = "sha256:4a23ce380ecb60a4f584f0cc87904999605049034a24c5f20e9e7b3bd4d2e195"
INPUTS = list(
    dict.fromkeys(
        [
            *old.INPUTS[:-3],
            PRIOR,
            "results/experiment_8439_v726_arc_supervisor_support.json",
            "results/experiment_8453_v727_arc_seal_and_support.json",
            authority.PROTOCOL,
            authority.METHODS,
            *old.OWNED,
            old.TEST,
            *OWNED,
            TEST,
        ]
    )
)
MODEL_SPECS: list[Json] = []
runtime, sdk, snapshot = old.runtime, old.sdk, old.snapshot
_BUILD, _REPLAY, _RUN, _PRECONDITIONS = old.build, old.replay, old.run, old.preconditions


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual boundaries so an operator can distinguish a bounded child from a stall."""
    print(f"[exp8482] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def bindings() -> Iterator[None]:
    """Restore every historical producer after invocation-only reuse of its checked mechanics."""
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        TASK_PIN=TASK_PIN,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        INPUTS=INPUTS,
        authority=authority,
        build=build,
        progress=progress,
    ):
        yield


def preconditions(private: Path, root: Path) -> tuple[list[Json], Json, Json]:
    """Missing complete authority closes collection; optional hardware is outside this CPU task."""
    with bindings():
        checks, hashes, resources = _PRECONDITIONS(private, root)
    observed = hashes.get(str(root / PRIOR))
    if observed != PRIOR_PIN:
        checks.append(old.gate("preserved_primary", root / PRIOR, "sha256", PRIOR_PIN, observed))
        resources["task_sha256"] = None
    resources.update(
        invocation_argv=list(sys.argv),
        episode_limit_budget_s=1920,
        collection_overhead_budget_s=480,
    )
    return checks, hashes, resources


def build(work: Json, receipts: list[Json], raw: Path, output: Path, *, seal: bool = True) -> Json:
    """Qualification can repair rejection coverage without creating absent live observations."""
    with bindings():
        value = dict(_BUILD(work, receipts, raw, output, seal=seal))
    value.update(
        experiment_id=8482,
        task_id=TASK,
        milestone="2026.10.729",
        honest_verdict=f"complete_{value['verdict_class']}_v729_arc_panel_qualification",
        methodology_note="Qualify independent panel rejection before collection. Complete V729 authority is required. Unstarted rows are missing observations. Known public levels are diagnostic, not new solves; cached numeric heads are not current LLM generation.",
    )
    value["field_principles"].update(
        milestone="Bind the actual invocation to complete independent V729 authority.",
        methodology_note="Separate rejection qualification, live execution and resolved support.",
    )
    return value


def replay(value: Json) -> bool:
    """Independent registry and authority operands reject even self-rehashed blocked work."""
    try:
        with bindings():
            if not _REPLAY(value):
                return False
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        with TemporaryDirectory(prefix="exp8482-replay-", dir="/var/tmp") as directory:
            with frozen_inputs(work.get("source_input_receipts", []), Path(directory) / "writes"):
                contract = authority.authority(ROOT, Path(directory) / "authority")
                for gate in contract["gate_check_summary"]:
                    if not any(
                        f["field"] == gate["artifact_field"]
                        and f["expected"] == gate["expected"]
                        and f["observed"] == gate["observed"]
                        for f in work["failures"]
                    ):
                        return False
                registry = yaml.safe_load((ROOT / "ops/arc_solve_registry.yaml").read_bytes())[
                    "games"
                ]
                if (
                    work.get("panel")
                    and old.freeze(work["panel"]["installed_roster"], registry) != work["panel"]
                ):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def plan(private: Path) -> list[Json]:
    """Freeze exact test IDs and genuine unit/child coverage before any episode starts."""
    from carnot.reporting.arc_panel_validation_8482 import manifest

    return list(manifest(private))


def run(output: Path, private: Path, *, private_e2e: bool = False) -> int:
    """Reuse bounded SDK collection and unchanged terminal consumers under this identity."""
    with bindings(), patch.multiple(old, preconditions=preconditions, plan=plan, replay=replay):
        return int(_RUN(output, private, private_e2e=private_e2e))

"""REQ-REPORT-8496: qualify original custody before observing the frozen live panel.

Invocation bindings preserve the shipped policy and every historical assertion.
Public game progress remains diagnostic and cannot create hidden-game evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import arc_panel_qualification_8482 as shipped
from carnot.reporting import arc_generalization_panel_8468 as old
from carnot.reporting import v730_contract_methods as authority
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK = "experiment_8496_v730_arc_panel_qualification", "exp8496-arc-panel-qualification"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_arc_panel_qualification_8496.py"
OUTPUT = ROOT / "results" / (NAME + ".json")
OWNED = [
    "python/carnot/reporting/arc_panel_qualification_8496.py",
    "python/carnot/reporting/arc_panel_validation_8496.py",
    CLI,
]
TASK_PIN = "sha256:03e907385286c4d0087d64856f532e0aba75bfcaf42b8f87839c555d7977b57d"
PRIOR, PRIOR_PIN = shipped.PRIOR, shipped.PRIOR_PIN
INPUTS = list(
    dict.fromkeys(
        [
            *old.INPUTS,
            PRIOR,
            "results/experiment_8482_v729_arc_panel_qualification.json",
            "results/experiment_8439_v726_arc_supervisor_support.json",
            "results/experiment_8453_v727_arc_seal_and_support.json",
            authority.PROTOCOL,
            authority.METHODS,
            *shipped.OWNED,
            shipped.TEST,
            *OWNED,
            TEST,
        ]
    )
)
MODEL_SPECS: list[Json] = []
runtime, sdk, snapshot = old.runtime, old.sdk, old.snapshot
_BUILD, _REPLAY, _RUN, _PRECONDITIONS = old.build, shipped.replay, old.run, old.preconditions


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so bounded children cannot look like an abandoned task."""
    print(f"[exp8496] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def bindings() -> Iterator[None]:
    """Restore historical identities after reusing the original live collector and reducers."""
    with (
        patch.multiple(
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
        ),
        patch.multiple(shipped, bindings=bindings_old, authority=authority, TASK=TASK),
    ):
        yield


@contextmanager
def bindings_old() -> Iterator[None]:
    """Shipped replay already runs inside our invocation binding; keep its assertions intact."""
    yield


def preconditions(private: Path, root: Path) -> tuple[list[Json], Json, Json]:
    """Authenticate current full authority while keeping sealed V728 fixture authority separate."""
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
        optional_cuda_required=False,
        optional_board_required=False,
    )
    return checks, hashes, resources


def build(work: Json, receipts: list[Json], raw: Path, output: Path, *, seal: bool = True) -> Json:
    """A new publication binds actual transitions without turning prior unstarted rows into data."""
    with bindings():
        value = dict(_BUILD(work, receipts, raw, output, seal=seal))
    value.update(
        experiment_id=8496,
        task_id=TASK,
        milestone="2026.10.730",
        honest_verdict=f"complete_{value['verdict_class']}_v730_arc_panel_qualification",
        methodology_note="Successful sealed V728 replay precedes panel-only rejection. Exact original nodes and392 baseline statements qualify before16 current model-free live episodes. Historical receipts are separate from current activated V730 authority; public progress and right-censored redirects cannot authorize hidden-game transfer or new solve credit.",
    )
    value["field_principles"].update(
        milestone="Bind this invocation to independently matched complete V730 authority.",
        methodology_note="Separate original rejection coverage, execution and resolved support.",
    )
    return value


def replay(value: Json) -> bool:
    """Shipped custody checks reconstruct primitive rows and independent current panel metadata."""
    with bindings():
        return bool(_REPLAY(value))


def plan(private: Path) -> list[Json]:
    """Freeze complete original node unions and real Coverage.py child shards before episodes."""
    from carnot.reporting.arc_panel_validation_8496 import manifest

    return list(manifest(private))


def run(output: Path, private: Path, *, private_e2e: bool = False) -> int:
    """Reuse bounded production collection and unchanged terminal publication consumers."""
    with bindings(), patch.multiple(old, preconditions=preconditions, plan=plan, replay=replay):
        return int(_RUN(output, private, private_e2e=private_e2e))

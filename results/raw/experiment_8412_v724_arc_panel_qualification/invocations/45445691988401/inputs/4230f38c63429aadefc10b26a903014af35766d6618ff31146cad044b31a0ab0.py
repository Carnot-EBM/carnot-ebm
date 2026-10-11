"""REQ-REPORT-8412: private scratch repair qualifies exposed engineering transfer.

The shipped collector and runtime retain their budgets and assertions. Current
qualification cannot reopen the closed semantic-benefit or hidden-game claims.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import arc_generalization_panel_8398 as old
from carnot.reporting import historical_consumer_roots_8402 as current

Json = dict[str, Any]
ROOT, base = old.ROOT, old.base
NAME, TASK = "experiment_8412_v724_arc_panel_qualification", "exp8412-arc-panel-qualification"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_arc_panel_qualification_8412.py"
OUTPUT = ROOT / "results" / (NAME + ".json")
OWNED = ["python/carnot/reporting/arc_panel_qualification_8412.py", CLI]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:778901886f2379fd797585783ff6336f9799fdba8ddd8c3ec89bd3500c7842f9"
PRIOR = "results/experiment_8398_v723_arc_generalization_panel.json"
PRIOR_PIN = "sha256:060a0279d41c0a113b9efa7212f0783b9d67fdb5382342b08ef53b08d953527a"
BEFORE = "results/raw/experiment_8412_v724_arc_panel_qualification/historical_before/"
INPUTS = list(
    dict.fromkeys(
        [
            *old.INPUTS,
            PRIOR,
            current.PROTOCOL,
            current.METHODS,
            *OWNED,
            TEST,
            BEFORE + "historical_before.stdout",
            BEFORE + "historical_before.stderr",
            BEFORE + "historical_before.receipt.json",
        ]
    )
)
OLD_BUILD, OLD_REPLAY, OLD_PRECONDITIONS, OLD_RUN = (
    old.build,
    old.replay,
    old.preconditions,
    old.run,
)
OLD_PLAN, OLD_CONTROLS, OLD_PUBLISH = old.plan, old.controls, old.publish
HISTORICAL_TEST = base.TEST
sdk, runtime = old.sdk, old.runtime


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual boundaries so the operator can distinguish work from a stall."""
    print(f"[exp8412] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def bindings() -> Iterator[None]:
    """Invocation-only aliases preserve every historical file and submitted default."""
    values: Json = dict(
        NAME=NAME,
        TASK=TASK,
        TASK_PIN=TASK_PIN,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        INPUTS=INPUTS,
        build=build,
        progress=progress,
    )
    with (
        old.bindings(),
        patch.multiple(old, **values),
        patch.multiple(base, **values),
        patch.object(old, "authority", current),
        patch.object(base.authority_module, "authority", current.authority),
    ):
        yield


def preconditions(private: Path, root: Path) -> tuple[list[Json], Json, Json]:
    """Missing or substituted external inputs close collection before any action."""
    with bindings():
        checks, hashes, resources = OLD_PRECONDITIONS(private, root)
    if hashes.get(str(root / PRIOR)) != PRIOR_PIN:
        checks.append(
            old.gate(
                "historical_receipt_hash",
                root / PRIOR,
                "sha256",
                PRIOR_PIN,
                hashes.get(str(root / PRIOR)),
            )
        )
    if checks:
        resources["task_sha256"] = None
    return checks, hashes, resources


def historical_checks(private: Path) -> int:
    """Provision the owned parent before unchanged pytest binds its historical closure."""
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    (private / "writes").mkdir(parents=True, exist_ok=True, mode=0o700)
    with patch.object(base, "TEST", HISTORICAL_TEST):
        return int(old.historical_checks(private))


def plan(private: Path) -> list[Json]:
    """Freeze owned commands and genuine child coverage before observing the panel."""
    with bindings():
        specs = OLD_PLAN(private)
    historical = next(s for s in specs if s["name"] == "historical_wrapper_assertions")
    historical["argv"][3] = historical["argv"][3].replace(
        "arc_generalization_panel_8398", "arc_panel_qualification_8412"
    )
    return list(specs)


def build(work: Json, receipts: list[Json], raw: Path, output: Path, *, seal: bool = True) -> Json:
    """Engineering receipts never grant new solve credit or arm-refinement authority."""
    work["run_date"] = "20261011"
    value = OLD_BUILD(work, receipts, raw, output, seal=seal)
    cells: Json = {}
    for event in value["supervisor_outcome_rows"]:
        key = event["episode_id"].split(":")[0] + ":" + str(event.get("arm", "unknown"))
        cells[key] = cells.get(key, 0) + 1
    value.update(
        experiment_id=8412,
        task_id=TASK,
        milestone="2026.10.724",
        honest_verdict=f"complete_{value['verdict_class']}_v724_arc_panel_qualification",
        arm_refinement_authorized=False,
        overlapping_game_count=len({key.split(":")[0] for key in cells}),
        firings_per_arm_cell=cells,
    )
    value["field_principles"].update(
        run_date="Bind the new invocation to its authorized date, 20261011.",
        milestone="Current engineering qualification leaves historical benefit nulls closed.",
        arm_refinement_authorized="Two exposed games cannot authorize any supervisor arm change.",
        overlapping_game_count="Count games with actual supervisor firings; zero is a measured null.",
        firings_per_arm_cell="Count actual redirect-arm firings per game, never seeds as independent games.",
    )
    return dict(value)


def replay(value: Json) -> bool:
    """Fresh reductions use sealed authority and primitive operands, not today's roadmap."""
    with bindings():
        return bool(OLD_REPLAY(value))


def controls(value: Json, raw: Path) -> list[Json]:
    """Shipped fresh-process controls reject absent and self-rehashed observations."""
    with bindings():
        return list(OLD_CONTROLS(value, raw))


def publish(value: Json, work: Json, output: Path, raw: Path) -> None:
    """Unchanged validators retain findings before checked terminal bytes become visible."""
    with bindings():
        OLD_PUBLISH(value, work, output, raw)


def run(output: Path, private: Path, *, private_e2e: bool = False) -> int:
    """Reuse actual collection while keeping genuine Coverage.py child shards durable."""
    execution = old.execute

    def execute(specs: list[Json], logs: Path) -> list[Json]:
        receipts = execution(specs, logs)
        for path in private.iterdir():
            if path.name.startswith((".coverage", "coverage")) and path.is_file():
                target = logs.parent / "coverage" / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        return list(receipts)

    with (
        bindings(),
        patch.multiple(
            old,
            plan=plan,
            preconditions=preconditions,
            controls=controls,
            publish=publish,
            execute=execute,
        ),
    ):
        return int(OLD_RUN(output, private, private_e2e=private_e2e))

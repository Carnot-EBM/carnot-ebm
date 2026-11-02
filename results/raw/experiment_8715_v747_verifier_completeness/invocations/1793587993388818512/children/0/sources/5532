"""REQ-VERIFY-8527: qualify bounded source readers without recursive historical validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_criterion_decision_runner as shipped
from carnot.reporting import v729_contract_runner as collected
from carnot.reporting import v732_contract_methods as historical
from carnot.reporting import v733_source_consumer_qualification as e
from carnot.reporting.primary_publication import reader_receipt

Json = dict[str, Any]
ORIGINAL = "tests/python/test_v732_strongest_baseline_8513.py"


def original_workspace(private: Path) -> Path:
    """Copy code into disk scratch so original CLI children see original authority.

    Symlinking Python would resolve its root back to the live checkout. Real
    source copies keep subprocesses isolated while absolute historical source
    references continue to authenticate the original sealed bytes.
    """
    root = private / "original-v732"
    root.mkdir(parents=True, exist_ok=True)
    for folder in ["python", "scripts"]:
        for source in (e.ROOT / folder).rglob("*.py"):
            if "__pycache__" not in source.parts:
                target = root / source.relative_to(e.ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    for name in [".venv", "results", "docs"]:
        (root / name).symlink_to(e.ROOT / name, target_is_directory=True)
    for name in ["pyproject.toml", ORIGINAL, "tests/python/conftest.py"]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(e.ROOT / name, target)
    proposals = root / "openspec/change-proposals"
    proposals.mkdir(parents=True)
    for source in (e.ROOT / "openspec/change-proposals").iterdir():
        (proposals / source.name).symlink_to(source)
    design = root / historical.DESIGN
    design.unlink()
    protocol = historical.protocol()
    design.write_bytes(Path(protocol["v732_design_reference"]["path"]).read_bytes())
    (root / historical.ACTIVE).write_bytes(
        Path(protocol["v732_activation_reference"]["path"]).read_bytes()
    )
    return root


def manifest(private: Path) -> list[Json]:
    """Collect the exact original union before disjoint shards with unchanged assertions."""
    with patch.object(collected, "e", e):
        plan = list(collected.manifest(private))
        nodes, receipt = collected.collect(private, ORIGINAL)
    plan.insert(
        0,
        dict(
            name="collect_original_source",
            argv=receipt["argv"],
            expected=0,
            deadline=60,
            scope="owned",
            collected_node_ids=nodes,
        ),
    )
    root = original_workspace(private)
    bootstrap = "import os,sys; root=sys.argv[1]; os.chdir(root); sys.path[:0]=[root+'/python',root]; import pytest; raise SystemExit(pytest.main(sys.argv[2:]))"
    for i in range(0, len(nodes), 4):
        group = nodes[i : i + 4]
        plan.insert(
            i // 4,
            dict(
                name=f"original_source_{i // 4}",
                argv=[
                    sys.executable,
                    "-u",
                    "-c",
                    bootstrap,
                    str(root),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    *group,
                    "--basetemp=" + str(private / f"original-{i}"),
                ],
                expected=0,
                deadline=300,
                scope="owned",
                collected_node_ids=group,
            ),
        )
    return plan


def main(argv: list[str] | None = None) -> int:
    """The dated invocation owns finite deadlines and private control publications."""
    e.progress("start_no_model_load_LLM_loads_0_generations_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261015"], default="20261015")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private control outputs must be outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8527-owned-", dir="/var/tmp") as directory:
            with (
                patch.object(shipped, "e", e),
                patch.object(shipped, "manifest", manifest),
                patch.object(
                    shipped,
                    "reader_receipt",
                    lambda task, results, **kwargs: reader_receipt(
                        task, results, **dict(kwargs, field="comparator_inputs_ready_score")
                    ),
                ),
            ):
                return int(
                    shipped.run(args.root, output, Path(directory), control=args.private_control)
                )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

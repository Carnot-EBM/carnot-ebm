"""REQ-VERIFY-8528: reuse bounded children and exact-byte publication for numeric heads."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_criterion_decision_runner as shipped
from carnot.reporting import v729_contract_runner as collected
from carnot.reporting import v733_source_consumer_runner as source_runner
from carnot.reporting import v733_contract_methods as methods
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.primary_publication import reader_receipt
from carnot.verify import v733_identity_brier_fit as e

Json = dict[str, Any]


def original_workspace(private: Path) -> Path:
    """Original tests and CLI children need the original authority, never current milestone dates."""
    root = Path(source_runner.original_workspace(private))
    for source in (e.ROOT / "tests/python").rglob("*.py"):
        target = root / source.relative_to(e.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    upstream = e.read(dict(path=str(e.ROOT / e.UPSTREAM), sha256=e.UPSTREAM_PIN))
    ref = next(
        r
        for r in upstream["source_artifact_hashes"]
        if Path(r["path"]).name == "experiment_8487_v730_source_control_fit.json"
    )
    work = e.read(e.read(ref)["primitive_reference"])
    for role, name in [("design", methods.DESIGN), ("active", methods.ACTIVE)]:
        snap = work["authority"]["authority_snapshots"][role]
        source = Path(snap["snapshot_path"])
        if sha256_file(source) != snap["sha256"]:
            raise ValueError("original_fit_authority")
        target = root / name
        target.unlink()
        target.write_bytes(source.read_bytes())
    return root


def manifest(private: Path) -> list[Json]:
    """Collect original assertions and preserve their union in bounded serial shards."""
    with patch.object(collected, "e", e):
        plan = list(collected.manifest(private))
        nodes, receipt = collected.collect(
            private, "tests/python/test_v730_source_control_fit_8487.py"
        )
    plan.insert(
        0,
        dict(
            name=receipt["name"],
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
            i // 4 + 1,
            dict(
                name=f"historical_original_fit_{i // 4}",
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
    """A dated invocation owns deadlines; private controls cannot overwrite repository evidence."""
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
        with TemporaryDirectory(prefix="exp8528-owned-", dir="/var/tmp") as directory:
            with (
                patch.object(shipped, "e", e),
                patch.object(shipped, "manifest", manifest),
                patch.object(
                    shipped,
                    "reader_receipt",
                    lambda task, results, **kwargs: reader_receipt(
                        task, results, **dict(kwargs, field="identity_heads_ready_score")
                    ),
                ),
            ):
                return int(
                    shipped.run(args.root, output, Path(directory), control=args.private_control)
                )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

"""REQ-VERIFY-8484: bounded serial validation governs one atomic terminal primary."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone as old
from carnot.reporting import v729_capstone_evidence as e
from carnot.reporting import v729_contract_runner as collection
from carnot.reporting.v729_capstone_report import build
from carnot.reporting.gatemate_continuity_8400 import budget

Json = dict[str, Any]
_MANIFEST, _RUN = old.manifest, old.run
_CHILD = old.child
SUITES = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def manifest(private: Path) -> list[Json]:
    """Exact collected unions retain slow assertions while genuine subprocess coverage owns new code."""
    with patch.object(old, "e", e):
        plan = list(_MANIFEST(private))
    plan = [p for p in plan if p["name"] not in {"private_E2E018_consumers", "private_E2E021"}]
    collections, shards = [], []
    for suite in [e.TEST, *SUITES]:
        nodes, receipt = collection.collect(private, suite)
        collections.append(
            dict(
                name=receipt["name"],
                argv=receipt["argv"],
                expected=0,
                deadline=60,
                scope="owned",
                suite=suite,
                original_node_ids=nodes,
            )
        )
        if suite == e.TEST:
            plan[0].update(collected_node_ids=nodes)
        else:
            size = 1 if suite == SUITES[-1] else 8
            groups = [nodes[i : i + size] for i in range(0, len(nodes), size)]
            assert [n for group in groups for n in group] == nodes
            for index, group in enumerate(groups):
                name = f"private_{Path(suite).stem}_{index}"
                shards.append(
                    dict(
                        name=name,
                        suite=suite,
                        shard_node_ids=group,
                        argv=[
                            str(e.ROOT / ".venv/bin/pytest"),
                            "-n",
                            "0",
                            "-o",
                            "addopts=",
                            "--no-cov",
                            "-q",
                            *group,
                            "--basetemp=" + str(private / name),
                        ],
                        expected=0,
                        deadline=300,
                        scope="owned",
                    )
                )
    return [*collections, plan[0], *shards, *plan[1:]]


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """The shipped pipeline distinguishes external absence from owned validation failure."""
    selected_manifest = _MANIFEST if control else manifest
    with patch.multiple(
        old,
        e=e,
        build=build,
        manifest=selected_manifest,
        child=complete_child,
        terminal_child=complete_child,
    ):
        return int(_RUN(root, output, private, control=control, failed=failed))


def complete_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """A complete twenty-branch replay needs a bounded budget without dropping slower branches."""
    if name in {"cold_valid", "cold_rehashed", "terminal_cold"}:
        kwargs["deadline"] = 600
    return dict(_CHILD(name, argv, logs, **kwargs))


def main(argv: list[str] | None = None) -> int:
    """Private disk scratch and a dated task cap keep CPU reconciliation independent of devices."""
    e.progress("start_no_model_load_current_LLM_invocations_0", 0, 14)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261013"], default="20261013")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_request:
        if not args.worker_output:
            parser.error("worker output required")
        return e.worker_process(args.worker_request, args.worker_output)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private controls must publish outside the repository")
    with budget(4800), TemporaryDirectory(prefix="exp8484-owned-", dir="/var/tmp") as directory:
        return run(
            args.root,
            output,
            Path(directory),
            control=args.private_control,
            failed=args.failed_child,
        )

"""REQ-VERIFY-8723: validate all nodes before allocating invocation-owned scratch."""

from collections.abc import Callable
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from carnot.reporting import v746_retained_evidence as e

Json = dict[str, Any]


def collect(
    private: Path,
    raw: Path,
    suites: list[str],
    owned: list[str],
    child: Callable[..., Json],
) -> tuple[list[Json], Json]:
    """A rejected inventory cannot leave child state or inherit old coverage files."""
    token = uuid4().hex
    receipt = child(
        "collect_once",
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--collect-only",
            "-q",
            *suites,
        ],
        raw / ("collection-" + token),
        deadline=60,
    )
    nodes = [line for line in Path(receipt["stdout_path"]).read_text().splitlines() if "::" in line]
    if not receipt["passed"] or not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("single_exact_collection_failed")
    if (
        not suites
        or len(suites) != len(set(suites))
        or any(
            n.split("::", 1)[0] not in suites
            or any(not part.strip() for part in n.split("::"))
            or n != n.strip()
            for n in nodes
        )
    ):
        raise ValueError("malformed_collection_inventory")
    groups = [[n for n in nodes if n.startswith(suite + "::")] for suite in suites]
    for suite, group in zip(suites, groups):
        if not group:
            raise ValueError("missing_collection_nodes:" + suite)
    owner = private / ("collection-" + token)
    configs = [raw / f"coverage-{token}-{i}.ini" for i in range(len(suites))]
    inventory = raw / f"collection-{token}-inventory.json"
    if owner.exists() or any(p.exists() for p in [*configs, inventory]):
        raise ValueError("collection_scratch_collision")
    owner.mkdir()
    plan = []
    for i, (suite, group, config) in enumerate(zip(suites, groups, configs)):
        scratch = owner / str(i)
        scratch.mkdir()
        with config.open("x") as stream:
            stream.write(
                f"[run]\nparallel=true\npatch=subprocess\ndata_file={scratch}/.coverage\n[report]\nexclude_lines=\n"
            )
        plan.append(
            dict(
                name=f"child_{i}",
                role="owned" if suite in owned else "consumer",
                nodes=group,
                scratch=str(scratch),
                config=e.reference(config),
                deadline=600,
                argv=[
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + str(config),
                    "-m",
                    "pytest",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    *group,
                    "--basetemp=" + str(scratch / "tmp"),
                    "--junitxml=" + str(scratch / "junit.xml"),
                ],
            )
        )
    e.atomic_json(inventory, dict(node_ids=nodes, receipt=receipt, scratch_owner=str(owner)))
    return plan, receipt

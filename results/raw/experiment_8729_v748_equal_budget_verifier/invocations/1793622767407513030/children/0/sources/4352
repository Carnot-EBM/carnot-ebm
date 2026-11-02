"""REQ-VERIFY-8413: reuse bounded children and exact terminal-byte publication."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8413 as e
from carnot.reporting import board_operation_runner_8399 as prior
from carnot.reporting import hardware_operation_runner_8371 as shipped
from carnot.reporting.v709_execution import execute

Json = dict[str, Any]


def bound_popen(original: Any) -> Any:
    """Original CLI children need the same immutable inputs as their parent tests."""

    def start(argv: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(argv, (list, tuple)) and str(e.ROOT / e.prior.CLI) in argv:
            index = argv.index(str(e.ROOT / e.prior.CLI))
            argv = [
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / e.CLI),
                "--historical-argv",
                *argv[index:],
            ]
        return original(argv, *args, **kwargs)

    return start


def historical_main(argv: list[str]) -> int:
    """Run the unchanged board CLI with transitive, hash-verified historical aliases."""
    if argv[0] != str(e.ROOT / e.prior.CLI):
        raise ValueError("historical_cli_identity")
    primary = e.authority.checked(e.ROOT / e.READER, e.READER_PIN)
    refs = json.loads(primary.read_bytes())["source_artifact_hashes"]
    sys.argv = argv
    with (
        e.prior.prior.source_aliases(refs),
        patch.object(subprocess, "Popen", bound_popen(subprocess.Popen)),
    ):
        try:
            runpy.run_path(argv[0], run_name="__main__")
        except SystemExit as exit_code:
            return int(exit_code.code or 0)
    raise RuntimeError("historical_cli_did_not_exit")


def manifest(private: Path) -> list[Json]:
    """Keep unchanged source-drift controls and private E2E-018 with scoped coverage."""
    with patch.object(prior, "e", e):
        plan: list[Json] = prior.manifest(private)
    plan[2]["argv"].append("--keep")
    original = (
        "import json,sys,pytest,subprocess; from unittest.mock import patch; "
        "from carnot.reporting import board_operation_runner_8413 as r; "
        "from carnot.reporting import board_operation_boundary_8399 as e; "
        "p=e.ROOT/'results/experiment_8399_v723_board_operation_boundary.json'; "
        f"assert e.base.sha256_file(p)=={e.READER_PIN!r}; "
        "refs=json.loads(p.read_bytes())['source_artifact_hashes']; "
        "\nwith e.prior.source_aliases(refs), patch.object(subprocess, 'Popen', r.bound_popen(subprocess.Popen)):\n raise SystemExit(pytest.main(sys.argv[1:]))\n"
    )
    plan.append(
        dict(
            name="original_source_drift_and_tamper_controls",
            argv=[
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                "-c",
                original,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_board_operation_boundary_8399.py",
                "--basetemp=" + str(private / "original"),
            ],
            expected=0,
            deadline=600,
            scope="owned",
        )
    )
    return plan


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Actual child exits determine qualification independently of external absence."""

    def retained(plan: list[Json], logs: Path) -> list[Json]:
        """Preserve genuine coverage databases before temporary scratch is removed."""
        receipts: list[Json] = execute(plan, logs)
        for source in private.glob(".coverage*"):
            if source.is_file():
                destination = logs.parent / "coverage_shards" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
        return receipts

    with (
        patch.object(shipped, "e", e),
        patch.object(shipped, "manifest", manifest),
        patch.object(shipped, "execute", retained),
    ):
        return int(shipped.run(root, output, private, control=control))

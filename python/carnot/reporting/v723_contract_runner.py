"""REQ-VERIFY-8388: shipped children and validators own the execution boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v722_contract_runner as base
from carnot.reporting import v723_contract_methods as e

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Freeze owned coverage and real consumer checks; report global health separately."""
    with patch.object(base.base, "m", e):
        plan = base.base.manifest(private)
    config = private / "coverage.ini"
    adapter = "python/carnot/reporting/v722_contract_methods.py"
    config.write_text(
        config.read_text().replace("[report]", "    " + str(e.ROOT / adapter) + "\n[report]")
    )
    report = next(p for p in plan if p["name"] == "coverage_report")
    report["argv"].append("--include=" + ",".join(str(e.ROOT / p) for p in e.OWNED))
    plan.append(
        dict(
            name="repaired_statement_coverage",
            argv=[
                str(e.ROOT / ".venv/bin/python"),
                "-c",
                "import json,pathlib; p=pathlib.Path(" + repr(str(e.ROOT / adapter)) + "); "
                "line=next(i+1 for i,s in enumerate(p.read_text().splitlines()) if 'with TemporaryDirectory(prefix=\"exp8374-cold-\"' in s); "
                "files=json.loads(pathlib.Path("
                + repr(str(private / "coverage.json"))
                + ").read_bytes())['files']; "
                "data=next(v for k,v in files.items() if k.endswith('v722_contract_methods.py')); "
                "assert line in data['executed_lines']; print('new historical adapter statement: 100 percent')",
            ],
            expected=0,
            deadline=30,
            scope="owned",
        )
    )
    plan[0]["deadline"] = 900
    pytests = [
        e.TEST + "::test_private_e2e018",
        "tests/python/test_restricted_decision_audit_8210.py",
    ]
    for index, test in enumerate(pytests):
        plan.append(
            dict(
                name="private_E2E" + ("018" if index == 0 else "021"),
                argv=[
                    str(e.ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    test,
                    "--basetemp=" + str(private / f"e2e-{index}"),
                ],
                expected=0,
                deadline=240,
                scope="owned",
            )
        )
    plan.append(
        dict(
            name="repository_health_once",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(private / "global"),
            ],
            expected=0,
            deadline=900,
            scope="global",
        )
    )
    return list(plan)


def main(argv: list[str] | None = None) -> int:
    """Reuse the existing task cap and disk scratch while binding the current adapter."""
    with patch.object(base, "e", e), patch.object(base, "manifest", manifest):
        return int(base.main(argv))

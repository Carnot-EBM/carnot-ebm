"""REQ-VERIFY-8335: small adapters reuse bounded qualification and publication."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import reserved_prediction_seal_8335 as e
from carnot.reporting import sentence_spline_execution_8334 as qualified
from carnot.reporting.current_work_receipt import atomic_json

Json = dict[str, Any]
check = qualified.check
BASE_MANIFEST = qualified.manifest
BASE_QUALIFY = qualified.qualify_findings


def manifest(private: Path, candidate: Path) -> Json:
    """Reuse the frozen test plan with only this task's statement scope."""
    with patch.object(qualified, "e", e):
        plan = BASE_MANIFEST(private, candidate)
    with (private / "coverage.ini").open("a") as stream:
        stream.write("patch = subprocess\n")
    return plan


def qualify_findings(raw: Path) -> Json:
    """Unchanged constructed controls qualify the consumer before natural use."""
    with patch.object(qualified, "e", qualified.e if qualified.e is not e else e.upstream):
        return BASE_QUALIFY(raw)


def publish(value: Json, output: Path, raw: Path) -> None:
    """Reuse checked atomic publication and retained failure recovery."""
    with patch.object(qualified, "e", e):
        qualified.publish(value, output, raw)


def main(argv: list[str] | None = None) -> int:
    """A scoring child sees one allowlisted bundle and never evaluator paths."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--score-child":
        e.progress("start_predictor_only_child")
        try:
            _, operand, output, issued = args
            bundle = json.loads(Path(operand).read_bytes())
            atomic_json(Path(output), dict(rows=e.score(bundle, issued)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            e.progress("prediction_rejected_" + str(error))
            return 1
        e.progress("finished_predictor_only_child", 128, 0)
        return 0
    with (
        patch.object(qualified, "e", e),
        patch.object(qualified, "manifest", manifest),
        patch.object(qualified, "qualify_findings", qualify_findings),
    ):
        return int(qualified.main(args))

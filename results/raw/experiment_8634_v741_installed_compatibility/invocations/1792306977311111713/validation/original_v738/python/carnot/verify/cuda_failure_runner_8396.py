"""REQ-VERIFY-8396: reuse shipped deadlines, private coverage and publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.verify import cuda_failure_cause_8396 as q
from carnot.verify import runtime_evidence_execution_8382 as base

Json = dict[str, Any]
PY = base.PY
BASE_MANIFEST = base.manifest
SCRATCH = Path.home() / ".cache/carnot-exp8396-private"


def manifest(private: Path) -> list[Json]:
    """Freeze owned checks once and keep inherited repository health separate."""
    with patch.object(base, "q", q):
        plan = [s for s in BASE_MANIFEST(private) if s["scope"] != "global"]
    consumers = next(s for s in plan if s["name"] == "typed_runtime_consumers")
    consumers["argv"].insert(-1, "tests/python/test_v723_contract_methods_8388.py")
    return plan


def main(argv: list[str] | None = None) -> int:
    """Use one entry point for exact initialization, real probes and sealed replay."""
    args = list(sys.argv[1:] if argv is None else argv)
    q.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    if "--driver-init" in args:
        parser = argparse.ArgumentParser()
        parser.add_argument("--driver-init", action="store_true")
        parser.add_argument("--library", default="libcuda.so.1")
        parsed = parser.parse_args(args)
        q.progress("driver_initialization_before")
        result = q.initialize(parsed.library)
        q.progress("driver_initialization_after", 1, 0)
        print(json.dumps(result), flush=True)
        return 0
    with (
        patch.object(base, "q", q),
        patch.object(base, "SCRATCH", SCRATCH),
        patch.object(base, "manifest", manifest),
    ):
        return int(base.main(args))

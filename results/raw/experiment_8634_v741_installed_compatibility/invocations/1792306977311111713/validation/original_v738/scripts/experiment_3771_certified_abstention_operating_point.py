#!/usr/bin/env python3
"""Run Exp 3771 Certified abstention operating point."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

sys.path.insert(0, str(ROOT / "python"))


def _load_exp_module():
    return importlib.import_module("carnot.pipeline.certified_abstention_operating_point_3771")


exp = _load_exp_module()


def main() -> int:
    output = exp.write_artifact(
        ROOT,
        tests_run=[
            ".venv/bin/pytest tests/python/test_experiment_3771_certified_abstention_operating_point.py -q",
        ],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    
    # Run adversarial verify
    verifier_path = ROOT / "scripts" / "adversarial_verify.py"
    if verifier_path.exists():
        spec = importlib.util.spec_from_file_location("carnot_adversarial_verify", verifier_path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            report = dict(module.verify_artifact(output))
            flags = [dict(flag) for flag in report.get("flags", []) if isinstance(flag, dict)]
            
            # Print warnings and critical flags
            has_critical = False
            for flag in flags:
                severity = str(flag.get("severity", "")).lower()
                print(f"[{severity.upper()}] {flag.get('kind', 'UNKNOWN')}: {flag.get('detail', '')}")
                if severity == "critical":
                    has_critical = True
                    
            if has_critical:
                print("FAILED ADVERSARIAL VERIFY (CRITICAL FLAGS PRESENT)")
                return 1
    
    print(f"Generated artifact at: {output}")
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the REQ-REPORT-7110 V624 evidence-ingress quarantine replay."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.experiment_7110_v624_evidence_ingress_quarantine import main


if __name__ == "__main__":
    raise SystemExit(main())

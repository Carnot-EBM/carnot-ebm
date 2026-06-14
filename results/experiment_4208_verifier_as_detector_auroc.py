#!/usr/bin/env python3
"""Run Exp 4208 via the maintained detector probe."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exp_verifier_detector_auroc import main  # noqa: E402


if __name__ == "__main__":
    main()

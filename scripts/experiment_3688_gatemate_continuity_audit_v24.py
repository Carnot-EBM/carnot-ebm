#!/usr/bin/env python3
"""Entrypoint for Exp 3688 GateMate continuity audit v24."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../python")))

from carnot.experiment_3688_gatemate_continuity_audit_v24 import main


if __name__ == "__main__":  # pragma: no cover
    main()

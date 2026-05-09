#!/usr/bin/env python3
"""Exp 1627 Nabla Reasoner Debug Script.

Spec: REQ-VERIFY-1627, SCENARIO-VERIFY-1627.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from carnot.verify.nabla_debug import run_sweep


if __name__ == "__main__":
    artifact = run_sweep()
    print(f"Converges: {artifact['optimizer_converges']}")
    print(f"Optimal LR: {artifact['optimal_learning_rate']}")
    print(f"Optimal Momentum: {artifact['optimal_momentum']}")

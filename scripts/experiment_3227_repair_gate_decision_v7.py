#!/usr/bin/env python3
"""Write the Exp 3227 repair-gate decision v7 artifact."""

from carnot.verify.repair_gate_decision_v7 import write_artifact


if __name__ == "__main__":
    print(write_artifact().as_posix())

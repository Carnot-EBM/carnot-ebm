#!/usr/bin/env python3
"""Write the Exp 3195 adaptive verification granularity policy artifact."""

from carnot.verify.adaptive_verification_granularity_policy_v1 import write_artifact


if __name__ == "__main__":
    print(write_artifact().as_posix())

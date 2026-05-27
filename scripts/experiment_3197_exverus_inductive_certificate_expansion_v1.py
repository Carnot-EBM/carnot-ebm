#!/usr/bin/env python3
"""Write the Exp 3197 ExVerus inductive certificate expansion artifact."""

from carnot.verify.exverus_inductive_certificate_expansion_v1 import write_artifact


if __name__ == "__main__":
    print(write_artifact().as_posix())

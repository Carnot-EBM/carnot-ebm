#!/usr/bin/env python3
"""Write the Exp 3196 GenCP domain preview repair compiler artifact."""

from carnot.verify.gencp_domain_preview_repair_compiler_v1 import write_artifact


if __name__ == "__main__":
    print(write_artifact().as_posix())

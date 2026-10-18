#!/usr/bin/env python3
"""Thin entrypoint for REQ-ARC-WMTE-7581's bounded transport canary."""

from carnot.experiment_7581_v662_arc_bounded_canary import main


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Thin entrypoint for REQ-VERIFY-7491 lossless response-window protocol."""

from carnot.experiment_7491_v656_window_protocol import main


if __name__ == "__main__":
    raise SystemExit(main())

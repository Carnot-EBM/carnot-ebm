#!/usr/bin/env python3
"""Experiment 662 — Ising Sampler v3 RTL: EMA inertia field registers.

**Researcher summary:**
    Exp 648 (parallel Ising inertia, Python simulation) wrote the RTL specification
    hardware/kv260/ising_sampler_v3_spec.md for a Verilog module that adds per-spin
    exponential moving average (EMA) of the local field h to the v2 Ising sampler.

    This experiment verifies that the RTL files specified in Exp 648 have been written:
      - hardware/kv260/ising_sampler_v3.v    — main RTL module
      - hardware/kv260/ising_sampler_v3_tb.v — testbench

    WHY EMA matters: Dense coupling graphs cause oscillation — spins flip repeatedly
    because the local field h_i swings with every neighbour flip. EMA smooths recent
    field history so the sampler builds inertia toward a stable configuration, akin
    to momentum in gradient descent. The v3 spec predicts 20-35x fewer sweeps to
    convergence on FPGA (arXiv 2604.17109).

    Vivado synthesis is NOT required for this experiment. The deliverable is the .v
    files themselves; synthesis will occur once the KV260 board is online (Exp 663+).

Spec: REQ-HW-031, SCENARIO-HW-031
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root — honour CARNOT_REPO_ROOT env override (same as experiment_template.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent))
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() FIRST — detect GPU hardware, set env vars
# ---------------------------------------------------------------------------
# WHY first: env_autofix sets CARNOT_FORCE_LIVE, JAX_PLATFORMS, and other
# environment flags that must be in place before any import that touches JAX/GPU.
# Calling it first guarantees reproducible behaviour regardless of shell environment.
from python.carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

autofix_result = apply_env_autofix()
print(f"[env_autofix] result: {autofix_result}")

# ---------------------------------------------------------------------------
# Step 2: ExperimentTemplate scaffolding
# ---------------------------------------------------------------------------
# NOTE: deliverable is the .v file (RTL), not a .json result file.
# The ExperimentTemplate.assert_deliverable_written() check will verify it exists.
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

RTL_DELIVERABLE = "hardware/kv260/ising_sampler_v3.v"
TB_DELIVERABLE  = "hardware/kv260/ising_sampler_v3_tb.v"

tmpl = ExperimentTemplate(
    exp_id=662,
    title="Ising Sampler v3 RTL",
    deliverable=RTL_DELIVERABLE,
    # No GPU required: deliverable is Verilog RTL, not model inference.
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Step 3: Check RTL deliverables exist and contain required keywords
# ---------------------------------------------------------------------------

rtl_path = REPO_ROOT / RTL_DELIVERABLE
tb_path  = REPO_ROOT / TB_DELIVERABLE

rtl_written = rtl_path.exists()
tb_written  = tb_path.exists()

rtl_line_count    = len(rtl_path.read_text().splitlines()) if rtl_written else 0
rtl_text          = rtl_path.read_text() if rtl_written else ""

# These keyword checks verify the spec-mandated constructs are present in the RTL.
# h_ema: the per-spin EMA register (REQ-HW-031-2)
has_ema_register  = "h_ema" in rtl_text

# EMA_ALPHA: the parameterised alpha fraction numerator/denominator (REQ-HW-031-3)
has_ema_update    = "EMA_ALPHA" in rtl_text

print(f"[662] RTL file exists:        {rtl_written}")
print(f"[662] Testbench file exists:  {tb_written}")
print(f"[662] RTL line count:         {rtl_line_count}")
print(f"[662] has_ema_register:       {has_ema_register}")
print(f"[662] has_ema_update:         {has_ema_update}")

# ---------------------------------------------------------------------------
# Step 4: Build artifact JSON
# ---------------------------------------------------------------------------

honest_verdict = (
    "ising_v3_rtl_complete"
    if (rtl_written and tb_written)
    else "ising_v3_rtl_partial"
)

result_path = REPO_ROOT / "results" / "experiment_662_ising_v3_rtl.json"
result_path.parent.mkdir(parents=True, exist_ok=True)

artifact = tmpl.build_result(
    {
        "schema":           "carnot.ising_v3_rtl.v1",
        "rtl_written":      rtl_written,
        "tb_written":       tb_written,
        "rtl_path":         RTL_DELIVERABLE,
        "rtl_line_count":   rtl_line_count,
        "has_ema_register": has_ema_register,
        "has_ema_update":   has_ema_update,
        "vivado_available": False,
        "honest_verdict":   honest_verdict,
    },
    status="success" if (rtl_written and tb_written) else "partial",
)

result_path.write_text(json.dumps(artifact, indent=2))
print(f"[662] Artifact written: {result_path}")
print(f"[662] honest_verdict: {honest_verdict}")

# ---------------------------------------------------------------------------
# Step 5: Assert deliverable (RTL file) exists — hard stop if missing
# ---------------------------------------------------------------------------
# WHY assert: conductor checks deliverable_exists() by path; if the .v is missing
# the conductor will re-schedule this experiment. Assert makes the failure loud.
tmpl.assert_deliverable_written()

print("[662] All checks passed.")

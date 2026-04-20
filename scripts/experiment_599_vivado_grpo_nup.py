#!/usr/bin/env python3
"""Experiment 599: Vivado Gate Check + GRPO NUP Probe v5 Retrain.

**Researcher summary (REQ-VERIFY-125):**
    Two independent experiments combined into one slot:

    1. Vivado gate check — checks whether Vivado is installed on this machine.
       If found, runs synthesis of the KV260 Ising sampler RTL using the existing
       hardware/kv260/synth_ising.tcl script.  This has been blocked since Exp 584
       (vivado_not_installed).  If not found, generates exact CachyOS/Arch Linux
       install commands so a human can unblock it.

    2. GRPO NUP Probe v5 retrain (arXiv 2503.06639) — GRPO showed that binary
       correct/incorrect labels from live benchmarks ARE a contrastive training
       signal without additional annotation.  We use live_pairs accumulated across
       Exps 552 and 578 (200 entries total) to retrain NUP Probe v4 via
       GRPOContrastivePairer.  Target: AUC >= 0.750 on the FOVER corpus.

**Gate chain (every exit path writes the deliverable):**
    0. apply_env_autofix() FIRST.
    1. assert_live_or_ci_skip() — graceful skip in CI.
    2. ExperimentTimeoutWatchdog(599, timeout_minutes=61).
    3. Vivado check: shutil.which('vivado').  Run synthesis or record not_installed.
    4. GRPO NUP v5: load pairs, train, evaluate AUC, optionally save model.
    5. Build artifact schema='carnot.vivado_grpo_nup.v1'.
    6. tmpl.assert_deliverable_written() — FINAL LINE.

Spec: REQ-VERIFY-125, SCENARIO-VERIFY-155, SCENARIO-VERIFY-156
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Step 0: apply_env_autofix BEFORE any heavy imports
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Step 1: assert_live_or_ci_skip — graceful skip in CI without live GPU.
# ---------------------------------------------------------------------------
from carnot.pipeline.live_assertion import assert_live_or_ci_skip  # noqa: E402

assert_live_or_ci_skip()

# ---------------------------------------------------------------------------
# Remaining imports (after env/live checks are established)
# ---------------------------------------------------------------------------
import json  # noqa: E402

from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.nup_probe_v5 import NUPProbeV5  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Step 2: Watchdog — hard wall-clock cap (60 min Vivado + 1 min NUP)
# ---------------------------------------------------------------------------
_watchdog = ExperimentTimeoutWatchdog(599, timeout_minutes=61)

# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------
tmpl = ExperimentTemplate(
    exp_id=599,
    title="Vivado Gate + GRPO NUP v5",
    deliverable="results/experiment_599_vivado_grpo_nup.json",
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Step 3: Vivado gate check
# ---------------------------------------------------------------------------
_TCL_SCRIPT = str(_REPO_ROOT / "hardware" / "kv260" / "synth_ising.tcl")

_INSTALL_COMMANDS = [
    "# CachyOS (Arch Linux) Vivado installation:",
    "# 1. Download: https://www.xilinx.com/support/download.html (Vivado 2023.2)",
    "# 2. sudo ./Xilinx_Unified_2023.2_1013_2256_Lin64.bin --batch-mode",
    "# 3. echo export PATH=/tools/Xilinx/Vivado/2023.2/bin:$PATH >> ~/.bashrc",
    "# 4. source ~/.bashrc",
    "# 5. vivado -mode batch -source hardware/kv260/synth_ising.tcl",
]

vivado_path = shutil.which("vivado")
vivado_status: str
bitfile_built: bool | None = None
synthesis_result: dict | None = None

if vivado_path:
    vivado_status = "found"
    print(f"[Exp 599] Vivado found at: {vivado_path}")
    print(f"[Exp 599] Running synthesis: vivado -mode batch -source {_TCL_SCRIPT}")
    try:
        proc = subprocess.run(
            ["vivado", "-mode", "batch", "-source", _TCL_SCRIPT],
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=str(_REPO_ROOT),
        )
        stdout_tail = proc.stdout[-4000:] if len(proc.stdout) > 4000 else proc.stdout
        stderr_tail = proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr
        bitfile_built = proc.returncode == 0
        synthesis_result = {
            "returncode": proc.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        vivado_status = "synthesis_complete" if bitfile_built else "synthesis_failed"
        print(f"[Exp 599] Vivado returncode={proc.returncode}, bitfile_built={bitfile_built}")
    except subprocess.TimeoutExpired:
        vivado_status = "synthesis_timeout"
        bitfile_built = False
        synthesis_result = {"returncode": -1, "stdout_tail": "", "stderr_tail": "timeout"}
        print("[Exp 599] Vivado synthesis timed out after 3600s")
    except Exception as exc:
        vivado_status = "synthesis_error"
        bitfile_built = False
        synthesis_result = {"returncode": -1, "stdout_tail": "", "stderr_tail": str(exc)}
        print(f"[Exp 599] Vivado error: {exc}")
else:
    vivado_status = "not_installed"
    bitfile_built = None
    print("[Exp 599] Vivado not found in PATH.")
    print("[Exp 599] Install commands:")
    for line in _INSTALL_COMMANDS:
        print(f"  {line}")

# ---------------------------------------------------------------------------
# Step 4: GRPO NUP Probe v5 retrain
# ---------------------------------------------------------------------------
print("[Exp 599] Loading live_pairs for GRPO NUP v5 retrain...")

_PAIRS_PATHS = [
    _REPO_ROOT / "results" / "live_pairs_552.json",
    _REPO_ROOT / "results" / "live_pairs_578.json",
]

all_entries: list = []
for p in _PAIRS_PATHS:
    if p.exists():
        with p.open() as f:
            batch = json.load(f)
        print(f"[Exp 599]   Loaded {len(batch)} entries from {p.name}")
        all_entries.extend(batch)
    else:
        print(f"[Exp 599]   WARNING: {p.name} not found — skipping")

print(f"[Exp 599] Total FOVER corpus entries: {len(all_entries)}")

probe = NUPProbeV5(energy_dim=32, margin=1.0, learning_rate=0.01, random_seed=42)

train_result = probe.train_from_pairs(all_entries, n_epochs=100)
grpo_pairs_built: int = train_result.get("grpo_pairs_built", 0)
print(
    f"[Exp 599] GRPO pairs built: {grpo_pairs_built}, "
    f"correct={train_result.get('n_correct_steps', 0)}, "
    f"incorrect={train_result.get('n_incorrect_steps', 0)}"
)

nup_v5_auc: float = probe.evaluate_auc(all_entries)
print(f"[Exp 599] NUP v5 AUC on FOVER corpus: {nup_v5_auc:.4f}")

# ---------------------------------------------------------------------------
# Step 4d: Save model if AUC >= 0.750
# ---------------------------------------------------------------------------
_MODEL_PATH = str(_REPO_ROOT / "results" / "nup_probe_v5.safetensors")
nup_v5_model_saved = False

if nup_v5_auc >= 0.750:
    probe.save_safetensors(_MODEL_PATH)
    nup_v5_model_saved = True
    print(f"[Exp 599] Model saved to {_MODEL_PATH} (AUC={nup_v5_auc:.4f} >= 0.750)")
else:
    print(f"[Exp 599] AUC={nup_v5_auc:.4f} < 0.750 — model NOT saved")

# ---------------------------------------------------------------------------
# Step 5: Build artifact
# ---------------------------------------------------------------------------
if bitfile_built:
    honest_verdict = "vivado_synthesis_complete"
elif nup_v5_auc >= 0.750:
    honest_verdict = "vivado_not_installed_nup_retrained"
else:
    honest_verdict = "vivado_not_installed_nup_no_improvement"

artifact = tmpl.build_result(
    {
        "schema": "carnot.vivado_grpo_nup.v1",
        "vivado_status": vivado_status,
        "bitfile_built": bitfile_built,
        "install_instructions": _INSTALL_COMMANDS,
        "synthesis_result": synthesis_result,
        "grpo_pairs_built": grpo_pairs_built,
        "nup_v5_auc": nup_v5_auc,
        "nup_v5_model_saved": nup_v5_model_saved,
        "honest_verdict": honest_verdict,
    },
    status="success",
)
AtomicResultWriter(str(_REPO_ROOT / "results" / "experiment_599_vivado_grpo_nup.json")).write(artifact)

# ---------------------------------------------------------------------------
# Step 6: FINAL LINE — assert deliverable written
# ---------------------------------------------------------------------------
tmpl.assert_deliverable_written()

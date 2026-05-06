"""Build the Exp 1422 Discrete SB KV260 RTL specification packet.

This module turns Exp 1399's CPU-only feasibility evidence into a concrete RTL
handoff document and a small JSON artifact.  It is intentionally not a Vivado
runner: the point of Exp 1422 is to make the next RTL/synthesis step explicit
while keeping the claim boundary honest.

Spec refs: REQ-ISING-023, SCENARIO-ISING-033.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP1399_PATH = PROJECT_ROOT / "results" / "experiment_1399_discrete_sb_kv260_cpu_simulation.json"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "results" / "experiment_1422_discrete_sb_kv260_rtl_spec.json"
DEFAULT_RTL_SPEC_PATH = PROJECT_ROOT / "hardware" / "kv260" / "discrete_sb_rtl_spec.md"

EXPERIMENT_ID = 1422
DEFAULT_RUN_DATE = "20260506"
CPU_ONLY_HONEST_VERDICT = "rtl_spec_complete_budget_fits_no_synthesis_or_board_execution"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "rtl_spec_complete",
    "rtl_spec_path",
    "estimated_lut",
    "estimated_bram",
    "kv260_budget_fits",
    "synthesis_command_documented",
    "hardware_execution_performed",
    "hardware_claim_allowed",
    "honest_verdict",
}


def load_exp1399(path: str | Path = EXP1399_PATH) -> dict[str, Any]:
    """Load the Exp 1399 CPU-only dSB feasibility artifact.

    Exp 1422 should not recompute convergence.  It needs the prior artifact as
    provenance so the RTL handoff can preserve exactly what was measured and
    exactly what was only estimated.
    """

    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_resource_estimate(exp1399: Mapping[str, Any]) -> dict[str, Any]:
    """Return the KV260 resource estimate that the RTL spec must explain.

    The dense matrix size is the central hardware fact: 256 variables with int8
    couplings consume 256 * 256 bytes.  The 2000-LUT update-unit estimate is
    carried forward from Exp 1399 and remains an arithmetic estimate until a
    synthesis report replaces it.
    """

    resource = dict(exp1399.get("kv260_resource_estimate", {}))
    n_variables = int(resource.get("n_variables", 256))
    bits_per_coupling = int(resource.get("bits_per_coupling", 8))
    estimated_bram_kb = float(
        resource.get("bram_estimate_kb_for_256var", exp1399["bram_estimate_kb_for_256var"])
    )
    kv260_bram_budget_kb = int(
        resource.get("kv260_bram_budget_kb", exp1399["kv260_bram_budget_kb"])
    )
    estimated_lut = int(
        resource.get("lut_estimate_per_update_unit", exp1399["lut_estimate_per_update_unit"])
    )
    kv260_lut_budget = int(resource.get("kv260_lut_budget", 117_000))
    bram_fits = bool(resource.get("bram_budget_feasible", exp1399["bram_budget_feasible"]))
    lut_fits = bool(resource.get("kv260_lut_budget_fits", exp1399["kv260_lut_budget_fits"]))
    matrix_bytes = int(n_variables * n_variables * bits_per_coupling / 8)
    return {
        "n_variables": n_variables,
        "bits_per_coupling": bits_per_coupling,
        "matrix_bytes": matrix_bytes,
        "estimated_bram_kb": estimated_bram_kb,
        "kv260_bram_budget_kb": kv260_bram_budget_kb,
        "bram_budget_feasible": bram_fits,
        "estimated_lut": estimated_lut,
        "kv260_lut_budget": kv260_lut_budget,
        "kv260_lut_budget_fits": lut_fits,
        "kv260_budget_fits": bram_fits and lut_fits,
        "convergence_speedup_discrete_sb": exp1399.get("convergence_speedup_discrete_sb"),
    }


def build_rtl_spec_markdown(
    *,
    estimate: Mapping[str, Any],
    exp1399: Mapping[str, Any],
    run_date: str = DEFAULT_RUN_DATE,
) -> str:
    """Build the human-readable RTL handoff spec.

    The markdown is deliberately concrete about registers, buffers, and update
    timing, but it avoids pretending that a Verilog module, synthesis report, or
    board run already exists.
    """

    n_variables = int(estimate["n_variables"])
    matrix_bytes = int(estimate["matrix_bytes"])
    estimated_bram_kb = float(estimate["estimated_bram_kb"])
    speedup = estimate.get("convergence_speedup_discrete_sb")
    speedup_text = "not reported" if speedup is None else f"{float(speedup):.6g}x"
    matrix_formula = (
        f"{n_variables} x {n_variables} x {int(estimate['bits_per_coupling'])} bits = "
        f"{matrix_bytes} bytes = {estimated_bram_kb:.1f} KB"
    )
    return f"""# Discrete SB KV260 RTL Specification

Spec refs: REQ-ISING-023, SCENARIO-ISING-033.

Run date: {run_date}
Status: specification and synthesis plan only.

## Source Evidence

Exp 1399 measured a CPU-only Discrete Simulated Bifurcation convergence speedup
of {speedup_text} against the Gibbs baseline.  Its KV260 evidence was arithmetic
only: the dense int8 coupling matrix and one update unit fit estimated BRAM/LUT
budgets, while Vivado synthesis, bitfile generation, and KV260 board execution
were not performed.

## Datapath

The RTL target is a single-update-unit dSB datapath for N={n_variables}.  The
unit preserves the Exp 1399 update rule:

```text
x_i(t+1) = sign(x_i(t) + eta * sum_j J_ij * x_j(t) - pressure(t))
```

The proposed pipeline is:

1. Snapshot the current spin/position vector into `spin_cur`.
2. For each row `i`, stream `J[i, 0..N-1]` from dense BRAM and accumulate
   `field_i = sum_j J_ij * spin_cur[j]` in a signed fixed-point accumulator.
3. Compute `candidate_i = spin_cur[i] + eta * field_i - pressure`.
4. Write `spin_next[i] = +1` when `candidate_i >= 0`, otherwise `-1`.
5. Commit `spin_next` to `spin_cur` only after all N rows have been evaluated.

The row-serial schedule keeps the LUT estimate close to Exp 1399's one-update
unit estimate.  Parallel row units are a later optimization and must be
synthesized separately because they multiply the accumulator and memory-port
pressure.

## Memory Layout

The dense coupling matrix is row-major signed int8:

```text
J_MATRIX[row][col] at byte offset row * 256 + col
```

Resource arithmetic:

- {matrix_formula}.
- KV260 budget basis: 144 BRAM_36 blocks = {int(estimate["kv260_bram_budget_kb"])} KB.
- Estimated update logic: {int(estimate["estimated_lut"])} LUTs.
- KV260 LUT comparison budget: {int(estimate["kv260_lut_budget"])} LUTs.
- Budget fit from arithmetic estimate: {bool(estimate["kv260_budget_fits"])}.

Additional small memories are expected for `spin_cur`, `spin_next`, optional
host-provided initial spins, best-state readback, and scalar control registers.
Those buffers are tiny compared with the 64 KB dense J matrix, but final BRAM
count must come from Vivado.

## Random/Noise Source Assumptions

Exp 1399 used small Gaussian CPU noise only to choose the initial signs.  The
RTL plan does not add stochastic noise during the dSB update.  Hardware should
support two initialization modes:

- Host-loaded packed initial spins for deterministic replay.
- A seedable 32-bit LFSR/xorshift initializer that fills `spin_cur` before
  `START`, with the seed exposed through the host register map.

Any future thermal or dither noise source must be benchmarked as a new
algorithmic variant, not silently mixed into this Exp 1399-equivalent RTL.

## Update Schedule

`pressure(t)` is a fixed-point linear ramp from 0 to 1 across `MAX_STEPS`, using
the same schedule family as Exp 1399.  The RTL stores `PRESSURE_START`,
`PRESSURE_DELTA`, and `MAX_STEPS`; each completed sweep increments the pressure
accumulator.  A sweep means all N row updates have written `spin_next` and the
snapshot commit has occurred.

The default synthesis configuration should use:

- `N_VARIABLES = 256`
- `COUPLING_BITS = 8`
- `ETA_Q1_15` matching Exp 1399's eta = 0.09
- `MAX_STEPS = 128`
- one row accumulator / update unit

## Host Interface

Use AXI-Lite control plus a memory-mapped BRAM upload window with 32-bit byte
addresses.  The dense matrix alone occupies 64 KB, so the address map must not
reuse the older 16-bit-only v1 prototype assumption.

| Address | Name | Access | Description |
| --- | --- | --- | --- |
| `0x00000` | `CONTROL` | R/W | bit0 START, bit1 RESET, bit2 LOAD_PRNG_INIT |
| `0x00004` | `STATUS` | R | bit0 READY, bit1 BUSY, bit2 DONE, bit3 ERROR |
| `0x00008` | `N_VARIABLES` | R | default 256 |
| `0x0000C` | `MAX_STEPS` | R/W | default 128 |
| `0x00010` | `ETA_Q1_15` | R/W | fixed-point eta |
| `0x00014` | `PRESSURE_START_Q1_15` | R/W | default 0 |
| `0x00018` | `PRESSURE_DELTA_Q1_15` | R/W | ramp increment |
| `0x0001C` | `RNG_SEED` | R/W | seed for optional initializer |
| `0x01000..0x10FFF` | `J_MATRIX` | W | 65536 byte row-major int8 matrix |
| `0x12000..0x1201F` | `SPIN_INIT` | W | packed initial spins, 1 bit per spin |
| `0x12100..0x1211F` | `SPIN_OUT` | R | packed final spins, 1 bit per spin |
| `0x12200` | `BEST_ENERGY` | R | optional signed fixed-point best energy |

## Synthesis Plan

The intended non-board synthesis command for a Vivado-equipped host is:

```bash
vivado -mode batch -source hardware/kv260/synth_discrete_sb.tcl
```

The planned TCL should target `xck26-sfvc784-2LV-c`, add the future
`hardware/kv260/discrete_sb_256.v` source, set the top module, run synthesis,
write utilization/timing reports, and stop before any board programming step.
Board validation remains a separate action after a bitfile exists and
`CARNOT_KV260_BITFILE` is configured.

## Claim Boundary

No hardware execution was performed for Exp 1422.  No Vivado synthesis report,
bitfile, timing closure, or KV260 board readback exists for this dSB RTL plan.
The only allowed claim is that the RTL specification is complete and the
resource estimate fits the KV260 arithmetic budget inherited from Exp 1399.

Hardware execution and hardware correctness claims are not allowed until a later
run produces synthesis reports and/or KV260 board validation artifacts.
"""


def build_artifact(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    exp1399: Mapping[str, Any],
    rtl_spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the JSON artifact that records the Exp 1422 handoff status."""

    root = Path(project_root)
    spec_path = Path(rtl_spec_path) if rtl_spec_path is not None else root / DEFAULT_RTL_SPEC_PATH.relative_to(PROJECT_ROOT)
    estimate = build_resource_estimate(exp1399)
    artifact = {
        "status": "complete",
        "run_date": run_date,
        "rtl_spec_complete": True,
        "rtl_spec_path": spec_path.relative_to(root).as_posix(),
        "estimated_lut": int(estimate["estimated_lut"]),
        "estimated_bram": float(estimate["estimated_bram_kb"]),
        "estimated_bram_unit": "KB",
        "kv260_budget_fits": bool(estimate["kv260_budget_fits"]),
        "synthesis_command_documented": True,
        "hardware_execution_performed": False,
        "hardware_claim_allowed": False,
        "honest_verdict": CPU_ONLY_HONEST_VERDICT,
        "resource_estimate": estimate,
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "source_experiment_id": 1399,
            "source_hardware_execution_performed": bool(
                dict(exp1399.get("metadata", {})).get("hardware_execution_performed", False)
            ),
            "source_synthesis_performed": bool(
                dict(exp1399.get("metadata", {})).get("synthesis_performed", False)
            ),
            "synthesis_performed": False,
            "board_validation_performed": False,
            "claim_boundary": "specification_and_resource_estimate_only",
        },
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 1422 artifact before it is written to disk."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    if artifact["rtl_spec_complete"] is not True:
        raise ValueError("rtl_spec_complete must be true")
    if artifact["synthesis_command_documented"] is not True:
        raise ValueError("synthesis_command_documented must be true")
    if artifact["hardware_execution_performed"] is not False:
        raise ValueError("hardware_execution_performed must remain false for Exp 1422")
    if artifact["hardware_claim_allowed"] is not False:
        raise ValueError("hardware_claim_allowed must remain false without synthesis/board validation")
    if artifact["honest_verdict"] != CPU_ONLY_HONEST_VERDICT:
        raise ValueError("honest_verdict must preserve the no-hardware-execution claim boundary")


def write_outputs(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    exp1399_path: str | Path = EXP1399_PATH,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    rtl_spec_path: str | Path = DEFAULT_RTL_SPEC_PATH,
) -> dict[str, Any]:
    """Write the RTL markdown spec and validated Exp 1422 JSON artifact."""

    root = Path(project_root)
    exp1399 = load_exp1399(exp1399_path)
    estimate = build_resource_estimate(exp1399)
    spec_path = Path(rtl_spec_path)
    markdown = build_rtl_spec_markdown(estimate=estimate, exp1399=exp1399, run_date=run_date)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(markdown, encoding="utf-8")

    artifact = build_artifact(
        project_root=root,
        run_date=run_date,
        exp1399=exp1399,
        rtl_spec_path=spec_path,
    )
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact

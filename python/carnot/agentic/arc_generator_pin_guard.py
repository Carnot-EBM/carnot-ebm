"""Runtime refusal for harnesses frozen on a retired generator pin.

REQ-ARC-WMTE-6621 (Conversion 4 of
docs/research-notes/cumulative-coherence-rule-to-check-2026-08-21.md).

The failure this closes: a frozen A/B harness pins the generator it was
written against (deliberately, per never-prune — its recorded results
were taken on that model). When the live pin later moves, the harness
banner says "MEASURES THE RETIRED GENERATOR" — but a banner is prose,
and prose gets read only after a run is launched and discarded. That
happened on 2026-08-20: a supervisor A/B came up on Qwen3.5-9B and was
only caught via /proc/<pid>/cmdline.

The conversion: prose warning in a file header becomes a runtime
refusal. A harness whose frozen pin differs from the canonical live pin
(`arc_executable_world_model.ARC_LIVE_GENERATOR_REPO_SUBSTR` — one
constant, imported, never duplicated) refuses to run unless the caller
passes an explicit override flag. A deliberate archaeology run states
its intent with the flag; an accidental run is refused with the
explanation. False-positive rate is zero by construction.
"""

from __future__ import annotations

import sys


class RetiredPinError(SystemExit):
    """Raised (exits non-zero) when a frozen pin no longer matches the live pin."""


def check_frozen_pin(
    frozen_pin: str,
    *,
    allow_retired: bool = False,
    harness_name: str = "this harness",
    live_pin: str | None = None,
    model_path_override: str | None = None,
) -> str:
    """Refuse to run when `frozen_pin` differs from the live generator pin.

    Call from a harness main() BEFORE any model load. Returns the live
    pin on success so callers can log it. `allow_retired=True` (the
    --allow-retired-pin flag) converts the refusal into a loud stderr
    warning — for deliberate archaeology against the retired model,
    where the results are knowingly not live-path evidence.

    `model_path_override`: pass the harness's explicit weights-path
    override (CARNOT_ARC_GGUF_PATH) when set. An explicit model_path
    WINS over the frozen repo pin at load time, and since commit
    512eca0e6b the recorded model label derives from the weights
    actually loaded — so the frozen pin is cosmetic in that
    configuration, and refusing would force --allow-retired-pin onto
    runs that already measure the LIVE generator. The guard's failure
    mode is the DEFAULT resolution silently picking retired weights;
    an explicit path is not that failure.
    """
    if model_path_override and str(model_path_override).strip():
        print(
            f"[pin-guard] {harness_name}: explicit model_path override "
            f"{str(model_path_override).strip()!r} supersedes the frozen repo "
            f"pin {frozen_pin!r}; the recorded label derives from the loaded "
            "weights. Pin guard not applicable to this run.",
            file=sys.stderr,
        )
        return live_pin if live_pin is not None else frozen_pin
    if live_pin is None:
        from carnot.agentic.arc_executable_world_model import (
            ARC_LIVE_GENERATOR_REPO_SUBSTR,
        )

        live_pin = ARC_LIVE_GENERATOR_REPO_SUBSTR
    if frozen_pin == live_pin:
        return live_pin
    if allow_retired:
        print(
            f"WARNING: {harness_name} runs its FROZEN pin {frozen_pin!r}, not the "
            f"live pin {live_pin!r} (--allow-retired-pin). Results are NOT citable "
            "as live-path evidence.",
            file=sys.stderr,
        )
        return live_pin
    raise RetiredPinError(
        f"REFUSING TO RUN: {harness_name} pins the RETIRED generator "
        f"{frozen_pin!r}; the live pin is {live_pin!r} "
        "(arc_executable_world_model.ARC_LIVE_GENERATOR_REPO_SUBSTR). "
        "A run here measures the retired model while the harness prose claims "
        "the live path — the exact silent mismatch this guard exists to stop. "
        "Pass --allow-retired-pin for a deliberate archaeology run, or use the "
        "live-pin harnesses."
    )

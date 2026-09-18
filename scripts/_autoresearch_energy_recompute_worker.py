#!/usr/bin/env python3
"""Fresh-interpreter worker for the REQ-AUTO-025 CRITICAL-2 fix (2026-09-16
adversarial review).

**Why this file exists as a standalone script, not a function call.** A
sandboxed autoresearch hypothesis runs IN-PROCESS, in the same interpreter
as `scripts/autoresearch_conductor_round.py` (`sandbox.py`'s own docstring
says its isolation is "not a security boundary"). The adversarial review
proved that matters: a hypothesis can do
`import carnot.autoresearch.verifier_auroc_benchmark as vab; vab.
_binary_auroc = lambda l, s: 1.0` (or the equivalent monkeypatch against
`toy_benchmarks.BENCHMARK_ENERGY_FUNCTIONS`, or mutate the `lru_cache`d
held-out row dicts in place) and have that patch STILL be in effect when
the "independent" recompute runs immediately afterward in the same process
-- fabricating a real, git-committed "discovery." Spawning a brand-new
Python process for the recompute closes this completely: this worker
re-imports every dependency from scratch, so nothing a hypothesis mutated
in the CALLER's `sys.modules` or `lru_cache`s can reach it.

**The contract.** Reads one JSON object from stdin:
`{"benchmark_name": str, "final_state": Any}`. Writes exactly one JSON
object to stdout: `{"energy": float | None, "brier": float | None}`.
`brier` is only ever non-None for the `calibrated_decision` benchmark
(REQ-AUTO-018) -- every other benchmark's caller reads `energy` only and
ignores the extra key, so this is an additive extension, not a breaking
change to the original `{"energy": ...}` shape. Never raises past this
boundary and never writes anything else to stdout -- any failure (bad
JSON, an unknown benchmark name, a malformed final_state, an internal
exception) becomes `{"energy": None, "brier": None}`, the same
"unscoreable" contract every recompute function in this project already
has. `final_state` is JSON, never a live Python object, so nothing a
hypothesis constructed (a class instance, a closure) can cross this
boundary either.

Spec: REQ-AUTO-025, REQ-AUTO-018
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from carnot.autoresearch.calibrated_decision_benchmark import (  # noqa: E402
    CALIBRATED_DECISION_BENCHMARK_NAME,
    recompute_calibrated_decision_metrics,
)
from carnot.autoresearch.toy_benchmarks import (  # noqa: E402
    BENCHMARK_ENERGY_FUNCTIONS,
    recompute_final_energy,
)
from carnot.autoresearch.verifier_auroc_benchmark import (  # noqa: E402
    VERIFIER_AUROC_BENCHMARK_NAME,
    recompute_verifier_auroc_energy,
)


def _recompute(benchmark_name: str, final_state: object) -> tuple[float | None, float | None]:
    if benchmark_name in BENCHMARK_ENERGY_FUNCTIONS:
        return recompute_final_energy(benchmark_name, final_state), None
    if benchmark_name == VERIFIER_AUROC_BENCHMARK_NAME:
        return recompute_verifier_auroc_energy(final_state), None
    if benchmark_name == CALIBRATED_DECISION_BENCHMARK_NAME:
        result = recompute_calibrated_decision_metrics(final_state)
        if result is None:
            return None, None
        return result["final_energy"], result["brier"]
    return None, None


def main() -> int:
    energy: float | None = None
    brier: float | None = None
    try:
        payload = json.loads(sys.stdin.read())
        benchmark_name = payload["benchmark_name"]
        final_state = payload.get("final_state")
        if isinstance(benchmark_name, str):
            energy, brier = _recompute(benchmark_name, final_state)
    except Exception:  # noqa: BLE001 -- this boundary must never crash, ever
        energy, brier = None, None
    print(json.dumps({"energy": energy, "brier": brier}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

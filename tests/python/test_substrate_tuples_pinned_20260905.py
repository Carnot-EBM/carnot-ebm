"""The substrate allowlists are FROZEN: one pinned length per tuple, plus the class enum.

Spec: REQ-SUBSTRATE-FREEZE-1 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-SUBSTRATE-FREEZE-1 (each tuple's length is pinned, so a widening
fails this test), SCENARIO-SUBSTRATE-FREEZE-2 (the one known duplicate is named, not
hidden), SCENARIO-SUBSTRATE-FREEZE-3 (the class enum is pinned),
REQ-SUBSTRATE-CPU-EXACT-1 and SCENARIO-SUBSTRATE-CPU-EXACT-1.

WHY A TEST AND NOT A HOOK. From the alias lint's own commit (a76b5f03f8), 30 commits
widened these tuples and 29 were [conductor] commits, which skip every hook. A hook is
inert on that population. This test runs before every conductor step, so a widening
fails LOUD: it lands, and the next step stalls on a red suite. The operator chose loud
over silent on 2026-09-05.

IF YOU ARE THE AGENT READING THIS FAILURE: do not add a name to a tuple. Declare
`inference_substrate_class` on the artifact instead (REQ-SUBSTRATE-CLASS-1); the
substrate name stays as prose. Changing a pin is an operator decision.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import scripts.adversarial_verify as av  # noqa: E402
from carnot.agentic.arc_solve_artifact_discipline import SUBSTRATE_DURATION_FLOORS  # noqa: E402

# Measured 2026-09-05 at the freeze. Elements, not distinct values: see the duplicate test.
PINS = {
    "AGGREGATION_SUBSTRATE_ALIASES": 14,
    "NO_LLM_SUBSTRATE_ALIASES": 77,
    "LIVE_MODEL_SUBSTRATE_ALIASES": 40,
    "DETERMINISTIC_VERIFIER_SUBSTRATES": 24,
}


def test_each_alias_tuple_length_is_pinned() -> None:
    # SCENARIO-SUBSTRATE-FREEZE-1
    for name, expected in PINS.items():
        members = getattr(av, name)
        assert isinstance(members, tuple), name
        assert len(members) == expected, (
            f"{name} has {len(members)} members, pinned at {expected}. The allowlists are "
            "frozen: declare inference_substrate_class on the artifact instead of adding a "
            "name. A pin change is an operator decision."
        )


def test_the_no_llm_tuple_holds_exactly_one_duplicate_and_this_test_names_it() -> None:
    # SCENARIO-SUBSTRATE-FREEZE-2. Found by the 2026-09-05 census review: entries were
    # appended without reading the list, and the alias lint resolves the tuple to a set,
    # so it cannot see a duplicate. Removing it changes the element pin above and is
    # left to the operator, in the open.
    members = av.NO_LLM_SUBSTRATE_ALIASES
    assert len(members) == 77
    assert len(set(members)) == 76
    duplicates = sorted(value for value, count in Counter(members).items() if count > 1)
    assert duplicates == ["cached_sota_event_energy_calibration"]
    assert "cached_sota_event_energy_calibration" in av.DETERMINISTIC_VERIFIER_SUBSTRATES


def test_the_arc_lint_floor_table_is_pinned() -> None:
    # SCENARIO-SUBSTRATE-FREEZE-1, the ARC lint's own table.
    assert len(SUBSTRATE_DURATION_FLOORS) == 13, (
        f"SUBSTRATE_DURATION_FLOORS has {len(SUBSTRATE_DURATION_FLOORS)} entries, pinned at 13."
    )


def test_the_class_enum_is_pinned_at_seven() -> None:
    # SCENARIO-SUBSTRATE-FREEZE-3 and REQ-SUBSTRATE-CPU-EXACT-1.
    assert sorted(av.SUBSTRATE_CLASSES) == [
        "aggregation",
        "blocked_no_run",
        "cpu_exact_solver_or_simulator",
        "model_bounded_generation",
        "model_full_generation",
        "model_load_no_generation",
        "no_model_load",
    ]
    assert set(av.SUBSTRATE_CLASS_FLOORS) == av.SUBSTRATE_CLASSES
    assert "hardware_board" not in av.SUBSTRATE_CLASSES
    assert sorted(av.EXECUTION_VENUES) == ["gatemate", "host", "kv260", "polarfire"]

#!/usr/bin/env python3
"""Experiment 345: SessionMemory — multi-session persistence of learned pipeline state.

**Researcher summary:**
    CaseMemory (Exp 135) and ConstraintTemplateLibrary (Exp 343) accumulate
    learned error patterns in-memory but reset on every process restart.
    This experiment validates that SessionMemory correctly persists all three
    learning components — CaseMemory, ConstraintTemplateLibrary, and
    PerModelFPTracker — to disk and restores them with zero information loss.

**What this experiment does:**
    1. Creates a SessionMemory backed by .carnot_sessions/ for model Gemma4-E4B-it.
    2. Populates a fresh CaseMemory with 10 synthetic arithmetic violation patterns.
    3. Calls save() and verifies the state file exists at the expected path.
    4. Calls load() and verifies all 10 pattern entries are present in the
       restored CaseMemory.
    5. Checks ConstraintTemplateLibrary observations and PerModelFPTracker stats
       round-trip correctly.

**Success criteria:**
    - save_restore_verified = True
    - n_patterns_saved == n_patterns_restored == 10
    - storage_path points to the actual file on disk

Spec: REQ-LEARN-020, REQ-LEARN-021,
      SCENARIO-LEARN-035, SCENARIO-LEARN-036, SCENARIO-LEARN-037
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

# Ensure the script finds the local package even when run from the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from experiment_template import ExperimentTemplate

from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
from carnot.pipeline.case_memory import CaseMemory, CaseRecord
from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary
from carnot.pipeline.session_memory import SessionMemory

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

EXP_ID = 345
TITLE = "SessionMemory — multi-session persistence of learned pipeline state"
DELIVERABLE = "results/experiment_345_session_memory.json"
STORAGE_DIR = ".carnot_sessions"
MODEL_ID = "Gemma4-E4B-it"
N_PATTERNS = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_case_memory(n: int) -> CaseMemory:
    """Populate a CaseMemory with *n* synthetic arithmetic violation traces.

    **Why synthetic traces?**
        The session persistence test does not need real LLM outputs — it only
        needs to verify that whatever is stored can be faithfully round-tripped.
        Synthetic records are deterministic, fast, and do not require a GPU.
    """
    cm = CaseMemory()
    for i in range(n):
        record = CaseRecord.normalize(
            benchmark="synthetic",
            benchmark_slice="arithmetic",
            model_name=MODEL_ID,
            case_id=f"case_{i}",
            violation_types=(f"carry_error_{i % 3}",),
            description_texts=(f"Synthetic arithmetic question {i}",),
            prompt_text=f"What is {i * 7} times {i + 1}?",
        )
        cm.record(record)
    return cm


def _make_library_with_observations(model_id: str) -> ConstraintTemplateLibrary:
    """Return a ConstraintTemplateLibrary with a few carry_check observations."""
    lib = ConstraintTemplateLibrary()
    lib.register_builtin_templates()
    # Simulate that carry errors have been seen 5 times for this model.
    lib.observe_pattern("carry_check", model_id, 5)
    return lib


def _make_tracker_with_stats(model_id: str) -> PerModelFPTracker:
    """Return a PerModelFPTracker with a few recorded observations."""
    tracker = PerModelFPTracker(min_observations=10)
    for _ in range(3):
        tracker.update(model_id, "range_check", was_fp=True, was_tp=False)
    for _ in range(2):
        tracker.update(model_id, "range_check", was_fp=False, was_tp=True)
    return tracker


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Step 1: Create SessionMemory and fresh components
    # ------------------------------------------------------------------
    sm = SessionMemory(storage_dir=STORAGE_DIR, model_id=MODEL_ID)

    # Clean slate: remove any leftover state from a previous run.
    sm.clear()
    assert not sm.exists(), "clear() should have removed any prior state"

    case_memory = _make_synthetic_case_memory(N_PATTERNS)
    template_library = _make_library_with_observations(MODEL_ID)
    fp_tracker = _make_tracker_with_stats(MODEL_ID)

    n_patterns_saved = len(list(case_memory.entries()))
    assert n_patterns_saved == N_PATTERNS, (
        f"Expected {N_PATTERNS} entries in CaseMemory before save, got {n_patterns_saved}"
    )

    # ------------------------------------------------------------------
    # Step 2: Save and verify file existence
    # ------------------------------------------------------------------
    sm.save(case_memory, template_library, fp_tracker)
    assert sm.exists(), "SessionMemory.exists() must be True after save()"

    storage_path = str(sm._state_path())
    assert pathlib.Path(storage_path).exists(), f"State file not found at {storage_path}"

    # ------------------------------------------------------------------
    # Step 3: Load and verify round-trip fidelity
    # ------------------------------------------------------------------
    loaded = sm.load()
    assert loaded is not None, "load() returned None — save/load failed"

    loaded_cm, loaded_lib, loaded_tracker = loaded
    n_patterns_restored = len(list(loaded_cm.entries()))

    save_restore_verified = n_patterns_restored == n_patterns_saved

    # Verify template library round-trip
    lib_dict_before = template_library.to_dict()
    lib_dict_after = loaded_lib.to_dict()
    lib_observations_match = lib_dict_before == lib_dict_after

    # Verify FP tracker round-trip
    tracker_dict_before = fp_tracker.to_dict()
    tracker_dict_after = loaded_tracker.to_dict()
    tracker_stats_match = tracker_dict_before == tracker_dict_after

    # ------------------------------------------------------------------
    # Step 4: Verify list_sessions finds this model
    # ------------------------------------------------------------------
    sessions = SessionMemory.list_sessions(STORAGE_DIR)
    from carnot.pipeline.session_memory import _escape_model_id
    safe_id = _escape_model_id(MODEL_ID)
    sessions_verified = safe_id in sessions

    # ------------------------------------------------------------------
    # Step 5: Build result artifact
    # ------------------------------------------------------------------
    result = {
        "schema": "carnot.session_memory.v1",
        "save_restore_verified": save_restore_verified,
        "n_patterns_saved": n_patterns_saved,
        "n_patterns_restored": n_patterns_restored,
        "storage_path": storage_path,
        "lib_observations_match": lib_observations_match,
        "tracker_stats_match": tracker_stats_match,
        "sessions_verified": sessions_verified,
        "sessions_found": sessions,
    }

    overall_status = (
        "success"
        if (save_restore_verified and lib_observations_match and tracker_stats_match)
        else "failure"
    )

    artifact = tmpl.build_result(result, status=overall_status)

    output_path = pathlib.Path(DELIVERABLE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))

    print(f"[Exp {EXP_ID}] Status: {overall_status}")
    print(f"  save_restore_verified : {save_restore_verified}")
    print(f"  n_patterns_saved      : {n_patterns_saved}")
    print(f"  n_patterns_restored   : {n_patterns_restored}")
    print(f"  lib_observations_match: {lib_observations_match}")
    print(f"  tracker_stats_match   : {tracker_stats_match}")
    print(f"  storage_path          : {storage_path}")
    print(f"  Result written to     : {DELIVERABLE}")


if __name__ == "__main__":
    main()

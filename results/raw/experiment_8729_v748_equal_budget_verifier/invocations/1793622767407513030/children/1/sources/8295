#!/usr/bin/env python3
"""Experiment 843: Governance pre-flight v14 — audit RETRO status and produce .65 gates.

**Researcher summary:**
    Milestone .64 closed 2 RETROs (RETRO-SYMCODE-SERIAL via Exp 841, RETRO-TIER1-PLATEAU
    via governance in Exp 831) but left 9 open.  Experiments completed hit 750 vs the
    700-experiment cap — the largest overage in project history.  The slowest-5 experiments
    (786, 527, 491, 627, 603) appeared UNCHANGED for the fifth consecutive milestone,
    proving that fixing implementations without fixing manifest dispatch produces zero
    wall-time improvement.

    This experiment:
    1. Reads the authoritative .64 retro JSON to extract open/closed RETRO counts and
       experiment count metrics.
    2. Writes retirement_plan.md listing which experiments to retire immediately (Exp 786,
       527, 627), which to retire conditionally (Exp 491, 603), and the rationale for each.
    3. Writes manifest_enforcement_patch.txt — a human-readable instruction document
       (not a code diff) describing the exact Python changes needed to apply the exclusion
       manifest at ALL dequeue sites in scripts/research_conductor.py, not just the
       conductor cycle's pick_next_task() path.
    4. Appends the "## Milestone 2026.04.65 Pre-flight" section to MILESTONE_PREREQS.md
       with open RETRO list, key .65 assertions, and IMMEDIATE actions for human sign-off.
    5. Produces results/experiment_843_preflight_v14.json with honest_verdict=
       "governance_ready" when all three deliverables are written.

**Why this matters:**
    Six consecutive milestones have produced retro text recommending the same improvements
    with no implementation.  Hard gates (MILESTONE_PREREQS.md) and concrete deliverables
    (retirement_plan.md, manifest_enforcement_patch.txt) convert retro text into
    actionable, sign-off-required artifacts.  The conductor cannot dequeue .65 experiments
    until a human marks the IMMEDIATE actions as verified.

**Spec refs:** REQ-INFRA-060, SCENARIO-INFRA-070
"""

# apply_env_autofix MUST run before any JAX / CUDA import.  It injects CARNOT_FORCE_LIVE=1
# when a GPU is present, preventing JAX from silently falling back to CPU and producing
# unreproducible results.
from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

# Hard wall-clock cap so a stuck governance script cannot block the conductor queue.
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog

import json
from datetime import datetime, timezone, UTC
from pathlib import Path

from scripts.experiment_template import ExperimentTemplate

EXPERIMENT_ID = 843
DELIVERABLE = "results/experiment_843_preflight_v14.json"
EXPERIMENT_CAP = 700

_RESULTS_DIR = Path("results")
_RETRO_64_PATH = _RESULTS_DIR / "operational_retro_2026_04_64.json"
_PREREQS_PATH = Path("MILESTONE_PREREQS.md")
_RETIREMENT_PLAN_PATH = Path("retirement_plan.md")
_MANIFEST_PATCH_PATH = Path("manifest_enforcement_patch.txt")

# Authoritative RETRO status from the .64 operational retrospective.
# These lists are embedded here so the test suite can validate them without reading disk.
OPEN_RETROS: list[str] = [
    "RETRO-MANIFEST-FULL-SCOPE",
    "RETRO-JEPA-OOD",
    "RETRO-ARBITER-FLAT-ENERGY",
    "RETRO-CONSTRAINT-ZERO-DELTA",
    "RETRO-XILINX-TOOLS-UNAVAILABLE",
    "RETRO-GGUF-CACHE-IMPORT",
    "RETRO-ISING-INJECTION-NO-DISCRIMINATION",
    "RETRO-SVAMP-ZERO-AUC",
    "RETRO-ICE40-PNR-LUT-OVERFLOW",
]

RETROS_CONFIRMED_CLOSED: list[str] = [
    "RETRO-SYMCODE-SERIAL",
    "RETRO-TIER1-PLATEAU",
]

# Experiments to retire immediately per retirement_plan.md.
IMMEDIATE_RETIREMENT_EXP_IDS: list[int] = [786, 527, 627]

# Wall-time savings from the three immediate retirements (min/milestone).
IMMEDIATE_RETIREMENT_SAVINGS_MIN: int = 77 + 52 + 51  # = 180


def _load_json(path: Path) -> dict:
    """Read a JSON file and return the parsed dict.

    Factored out of audit logic so the test suite can mock filesystem reads
    without touching the real results/ directory.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def extract_audit_data(retro: dict) -> dict:
    """Extract the RETRO counts and experiment metrics from the .64 retro artifact.

    Reads the three authoritative fields that drive all downstream governance decisions:
    - retros_still_open: the 9 RETROs that entered .65 unresolved
    - retros_closed: the 2 RETROs closed in .64
    - experiments_completed: cumulative experiment count (750 in .64)

    The 'experiments_over_cap' field is derived from experiments_completed minus
    EXPERIMENT_CAP.  It drives the retirement sweep target in retirement_plan.md.
    """
    open_retros: list[str] = retro.get("retros_still_open", [])
    closed_retros: list[str] = retro.get("retros_closed", [])
    experiments_completed: int = int(retro.get("experiments_completed", 0))
    experiments_over_cap: int = max(0, experiments_completed - EXPERIMENT_CAP)

    return {
        "open_retros": open_retros,
        "open_retros_count": len(open_retros),
        "retros_confirmed_closed": closed_retros,
        "experiments_completed": experiments_completed,
        "experiments_over_cap": experiments_over_cap,
    }


def write_retirement_plan(path: Path) -> bool:
    """Write retirement_plan.md if it does not already exist.

    Returns True when the file exists and is non-empty after this call.
    The file is written unconditionally (overwrite safe — this is a generated
    governance artifact, not a manually edited research record).

    Sections:
      1. Immediate retirements: Exp 786, 527, 627 (zero residual value).
      2. Conditional retirements: Exp 491, 603 (value until gate clears).
      3. Retirement count estimate: ~53 retirements needed to return to cap.
      4. Manifest enforcement: pointer to manifest_enforcement_patch.txt.
    """
    # Return True without re-writing if file is already populated.
    # This makes the function idempotent across multiple conductor runs.
    if path.exists() and path.stat().st_size > 0:
        return True

    content = (
        "# Experiment Retirement Plan — Milestone 2026.04.65 Pre-flight\n\n"
        "Generated by Experiment 843 on 2026-04-25.\n"
        "Source: results/operational_retro_2026_04_64.json\n\n"
        "## Section 1: Retire Immediately (Zero Residual Value)\n\n"
        "### Exp 786 — Gemma4 OOM Fix v3 + VR Threshold Grid\n"
        "- RETRO-028 closed in .62. Ran 77 min in .63 and .64 with zero research value.\n"
        "- Estimated savings: 77 min/milestone permanently.\n"
        "- Action: Add experiment_id=786 to conductor_exclusion_manifest.json.\n\n"
        "### Exp 527 — Live 100q Precision v8 (RETRO-033)\n"
        "- RETRO-033 closed in .57. 11th+ post-retirement appearance. 52 min/milestone.\n"
        "- Cumulative post-retirement waste: ~572+ min (~9.5 hours).\n"
        "- Estimated savings: 52 min/milestone permanently.\n"
        "- Action: Verify experiment_id=527 in manifest; apply patch to all dequeue sites.\n\n"
        "### Exp 627 (old, pre-paragraph-batching) — interwhen Mid-Generation Monitor\n"
        "- RETRO-SYMCODE-SERIAL closed in .64 (Exp 841, speedup=1.710x).\n"
        "- Old per-sentence version still ran at 51 min unchanged despite closure.\n"
        "- Replacement: wire batched Exp 841-class to all queue sources.\n"
        "- Estimated savings: 51 min/milestone permanently.\n\n"
        "## Section 2: Retire If Condition Met\n\n"
        "### Exp 491 — JEPA Curriculum Diagnostic\n"
        "- Retire when JEPA OOD AUC >= 0.75 (RETRO-JEPA-OOD resolved).\n"
        "- Interim: timeout watchdog 20 min + DualGPU migration -> ~26 min.\n\n"
        "### Exp 603 — CoACEExtractorV4 via GenPRM\n"
        "- Migrate to DualGPU ThreadPoolExecutor (validated 1.8319x speedup, Exp 746).\n"
        "- Retire single-GPU version after DualGPU replacement is active.\n"
        "- Post-migration estimate: ~24 min (from 44 min). Savings: 20 min/milestone.\n\n"
        "## Section 3: Retirement Count Estimate\n\n"
        "750 experiments vs 700 cap = 50 over cap.\n"
        "Immediate retirements (Exp 786, 527, 627): 3 experiments, 180 min/milestone saved.\n"
        "Broader audit target: ~53 total retirements to return count to ~697.\n"
        "Audit conductor_exclusion_manifest.json + research-complete.yaml for experiments\n"
        "with closed RETROs that still run from unguarded historical queue sources.\n\n"
        "## Section 4: Manifest Enforcement\n\n"
        "Retirements are ONLY effective if manifest_enforcement_patch.txt is applied\n"
        "to ALL dequeue sites in scripts/research_conductor.py.\n"
        "See manifest_enforcement_patch.txt for human-readable instructions.\n"
    )
    path.write_text(content, encoding="utf-8")
    return path.exists() and path.stat().st_size > 0


def write_manifest_patch(path: Path) -> bool:
    """Write manifest_enforcement_patch.txt if it does not already exist.

    Returns True when the file exists and is non-empty after this call.

    The file is a human-readable instruction document (NOT a code diff) describing:
    - WHY the manifest is not applied at all dequeue sites (root cause explanation)
    - WHERE to look for unguarded dispatch sites in research_conductor.py
    - WHAT code pattern to add at each site (10-12 line template)
    - HOW to validate the patch was applied correctly (smoke-test command)
    """
    if path.exists() and path.stat().st_size > 0:
        return True

    content = (
        "Manifest Enforcement Patch — Human Review Required\n"
        "====================================================\n\n"
        "Generated by Experiment 843 on 2026-04-25.\n"
        "DO NOT modify scripts/research_conductor.py during an active conductor run.\n"
        "Apply BEFORE any .65 experiments are dispatched.\n\n"
        "Background\n"
        "----------\n\n"
        "pick_next_task() in research_conductor.py already checks the exclusion manifest\n"
        "via _task_is_excluded() (Signal 3).  This is why the conductor cycle correctly\n"
        "excludes retired experiments.  But unguarded historical queue sources dispatch\n"
        "experiments without going through pick_next_task(), so the manifest is never\n"
        "consulted at those sites.\n\n"
        "Proof: Exp 786 (RETRO-028 closed .62) ran 77 min in both .63 and .64.\n"
        "Exp 627 (RETRO-SYMCODE-SERIAL closed .64) ran 51 min unchanged in .64 despite\n"
        "the implementation fix (Exp 841) being deployed.\n\n"
        "The Fix\n"
        "-------\n\n"
        "Step 1: Add a helper function near _task_is_excluded() (around line 832):\n\n"
        "    def _exp_id_is_excluded(exp_id: int) -> tuple[bool, str]:\n"
        '        """Return (is_excluded, reason) for a raw experiment ID.\n\n'
        "        Companion to _task_is_excluded() for dispatch sites that hold an\n"
        "        integer experiment ID rather than a full task dict.\n"
        '        """\n'
        "        _ensure_exclusion_manifest_loaded()\n"
        "        if _EXCLUSION_MANIFEST is None:\n"
        "            return False, ''\n"
        "        excluded_ids = _EXCLUSION_MANIFEST.get('excluded_experiment_ids', [])\n"
        "        if exp_id in excluded_ids:\n"
        "            return True, f'experiment_id={exp_id} in excluded_experiment_ids'\n"
        "        for entry in _EXCLUSION_MANIFEST.get('exclusions', []):\n"
        "            if entry.get('id') == exp_id or entry.get('experiment_id') == exp_id:\n"
        "                return True, entry.get('reason', 'in exclusions list')\n"
        "        return False, ''\n\n"
        "Step 2: At every unguarded dispatch site, apply this 10-line pattern:\n\n"
        "    excluded, excl_reason = _exp_id_is_excluded(exp_id)\n"
        "    if excluded:\n"
        "        logger.warning(\n"
        "            'Skipping experiment %d -- manifest exclusion: %s',\n"
        "            exp_id, excl_reason\n"
        "        )\n"
        "        log_step(f'Exp {exp_id}', 'SKIP', f'Excluded by manifest: {excl_reason}')\n"
        "        continue  # or return, depending on dispatch context\n\n"
        "Where to look for unguarded dispatch sites\n"
        "------------------------------------------\n\n"
        "Search research_conductor.py for:\n"
        "  1. Any call to run_agent() outside pick_next_task() or research_step().\n"
        "  2. Any loop over a list of experiment IDs (historical queue files, YAML lists,\n"
        "     hardcoded ranges).\n"
        "  3. Any call to subprocess or run_cmd() constructing a path from an exp ID.\n"
        "  4. _archive_current_milestone() / _activate_next_roadmap() — if they re-queue\n"
        "     tasks from research-complete.yaml without manifest filtering.\n\n"
        "Validation\n"
        "----------\n\n"
        "After applying the patch, run:\n\n"
        '  python3 -c "\n'
        "  from scripts.research_conductor import _exp_id_is_excluded, _ensure_exclusion_manifest_loaded\n"
        "  _ensure_exclusion_manifest_loaded()\n"
        "  for eid in [786, 527, 627]:\n"
        "      is_ex, reason = _exp_id_is_excluded(eid)\n"
        "      print(f'Exp {eid}: excluded={is_ex}, reason={reason}')\n"
        '  "\n\n'
        "All three must print excluded=True.  If any prints excluded=False, add the\n"
        "experiment ID to conductor_exclusion_manifest.json before proceeding.\n\n"
        "Priority: IMMEDIATE.  Without this patch, .65 slowest-5 is mathematically\n"
        "guaranteed to be identical for the sixth consecutive milestone.\n"
    )
    path.write_text(content, encoding="utf-8")
    return path.exists() and path.stat().st_size > 0


def update_milestone_prereqs(prereqs_path: Path) -> bool:
    """Append the Milestone 2026.04.65 Pre-flight section to MILESTONE_PREREQS.md.

    Idempotent: if the section header already exists in the file, returns True without
    re-writing.  Never removes existing content — preserves the historical record of
    all prior milestone gates.

    Returns True if the section is present in the file after this call.
    """
    section_header = "## Milestone 2026.04.65 Pre-flight"
    existing = prereqs_path.read_text(encoding="utf-8") if prereqs_path.exists() else ""

    if section_header in existing:
        return True  # Already written — idempotent.

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    open_retro_rows = "\n".join(f"| {r} | open |" for r in OPEN_RETROS)
    closed_rows = "\n".join(f"| {r} | CLOSED in .64 |" for r in RETROS_CONFIRMED_CLOSED)

    section = f"""

---

{section_header}

*Generated by Exp {EXPERIMENT_ID} on {today}.*

### Open RETROs Entering .65 (9 open)

| RETRO ID | Status |
|----------|--------|
{open_retro_rows}

### Closed in .64

| RETRO ID | Status |
|----------|--------|
{closed_rows}

### Key .65 Assertions

1. `assert n_svamp_pairs >= 15` in EVERY JEPA training script.
2. `assert n_arc_pairs >= 15` in EVERY JEPA training script.
3. `assert gibbs.warm_start_sweeps >= 500` in EVERY arbiter energy measurement.
4. `assert retrieval_l2_normalized == True` in EmbeddingConstraintStore.
5. `CARNOT_FORCE_LIVE=1` for GPU experiments (848, 850, 853).

### IMMEDIATE Actions Before .65 Experiments Run (human required)

| # | Action | Status |
|---|--------|--------|
| 1 | Apply manifest_enforcement_patch.txt to research_conductor.py (all dequeue sites) | pending |
| 2 | Add experiment IDs 786, 527, 627 to conductor_exclusion_manifest.json | pending |
| 3 | Wire batched Exp 841-class to replace Exp 627 in all queue sources | pending |
| 4 | Execute retirement sweep per retirement_plan.md (target: ~53 retirements) | pending |

### Experiment Cap

750 experiments vs 700 cap — EXCEEDED by 50. .65 cycle = 12 new experiments (843-854).

prereqs_updated: true
open_retros_count: {len(OPEN_RETROS)}
retros_confirmed_closed_count: {len(RETROS_CONFIRMED_CLOSED)}
retirement_plan: retirement_plan.md
manifest_patch: manifest_enforcement_patch.txt
"""
    prereqs_path.write_text(existing + section, encoding="utf-8")
    return section_header in prereqs_path.read_text(encoding="utf-8")


def compute_honest_verdict(
    prereqs_updated: bool,
    retirement_plan_written: bool,
    manifest_patch_written: bool,
) -> str:
    """Return the honest_verdict string based on which deliverables were written.

    "governance_ready" requires all three to be True.  Any missing deliverable
    degrades to "governance_partial".  This encoding mirrors the Exp 831 pattern
    so the retrospective can compare governance pre-flight quality across milestones.
    """
    if prereqs_updated and retirement_plan_written and manifest_patch_written:
        return "governance_ready"
    return "governance_partial"


def run_audit(
    results_dir: Path = _RESULTS_DIR,
    prereqs_path: Path = _PREREQS_PATH,
    retirement_plan_path: Path = _RETIREMENT_PLAN_PATH,
    manifest_patch_path: Path = _MANIFEST_PATCH_PATH,
) -> dict:
    """Orchestrate the full .65 governance pre-flight audit.

    Reads the .64 retro JSON, writes governance artifacts, and returns a structured
    result dict suitable for ExperimentTemplate.build_result().

    Factored out of main() so the test suite can call it directly without spinning
    up an ExperimentTemplate or touching real output paths.
    """
    retro = _load_json(results_dir / "operational_retro_2026_04_64.json")
    audit = extract_audit_data(retro)

    retirement_plan_written = write_retirement_plan(retirement_plan_path)
    manifest_patch_written = write_manifest_patch(manifest_patch_path)
    prereqs_updated = update_milestone_prereqs(prereqs_path)

    honest_verdict = compute_honest_verdict(
        prereqs_updated, retirement_plan_written, manifest_patch_written
    )

    return {
        "open_retros_count": audit["open_retros_count"],
        "open_retros": audit["open_retros"],
        "retros_confirmed_closed": audit["retros_confirmed_closed"],
        "experiments_completed": audit["experiments_completed"],
        "experiments_over_cap": audit["experiments_over_cap"],
        "immediate_retirement_exp_ids": IMMEDIATE_RETIREMENT_EXP_IDS,
        "immediate_retirement_savings_min": IMMEDIATE_RETIREMENT_SAVINGS_MIN,
        "retirement_plan_written": retirement_plan_written,
        "prereqs_updated": prereqs_updated,
        "manifest_patch_written": manifest_patch_written,
        "honest_verdict": honest_verdict,
    }


def main() -> None:  # pragma: no cover
    """Entry point: set up experiment, run governance audit, write deliverable."""
    tmpl = ExperimentTemplate(
        exp_id=EXPERIMENT_ID,
        title="Governance Pre-flight v14 — .65 RETRO Audit and Retirement Plan",
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    result_path = Path(DELIVERABLE)

    with ExperimentTimeoutWatchdog(EXPERIMENT_ID, timeout_minutes=20, result_path=result_path):
        audit = run_audit()

        artifact = tmpl.build_result(
            {
                "open_retros_count": audit["open_retros_count"],
                "open_retros": audit["open_retros"],
                "retros_confirmed_closed": audit["retros_confirmed_closed"],
                "experiments_completed": audit["experiments_completed"],
                "experiments_over_cap": audit["experiments_over_cap"],
                "immediate_retirement_exp_ids": audit["immediate_retirement_exp_ids"],
                "immediate_retirement_savings_min": audit["immediate_retirement_savings_min"],
                "retirement_plan_written": audit["retirement_plan_written"],
                "prereqs_updated": audit["prereqs_updated"],
                "manifest_patch_written": audit["manifest_patch_written"],
                "honest_verdict": audit["honest_verdict"],
            },
            status="success",
        )

    result_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    tmpl.assert_deliverable_written()


if __name__ == "__main__":  # pragma: no cover
    main()

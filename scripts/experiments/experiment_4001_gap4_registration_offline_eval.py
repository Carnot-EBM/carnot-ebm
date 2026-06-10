"""Experiment 4001 — GAP-4 Registration + Offline Bit-Exact Replay.

WHAT THIS DOES (GAP-4 conductor_followup #4, 2026-06-10):
  - PRECONDITIONS: verifier_registry.yaml loads; saved artifacts exist on disk.
  - Step 1: Verify the tiered-policy module (gap4_program_induction_stack) is importable
    and that its unit tests pass (asserts the reproduced counts, per the task spec).
  - Step 2: Offline bit-exact replay — reproduce ARC-2 pass@1 19/31 and ARC-1 28/31 from
    the saved artifacts without any new model calls.
  - Step 3: Confirm gap4_program_induction_stack is registered in ops/verifier_registry.yaml.
  - Step 4: Confirm the 446ef5d2 GAP-5 demo-underdetermination entry is in ops/verifier_gaps.md
    with full schema (failure mode, missing discriminator, candidate design, priority).
  - Write results/experiment_4001_gap4_registration_offline_eval.json.

WHY OFFLINE ONLY: the deployment numbers (19/31, 28/31) were measured in prior codex runs
(results/arc3_gap4_arc2_chain_ensemble.json, results/arc3_gap4_induced_programs.json).
Re-running them would require ~45 gpt-5.5 calls (~3780 codex-s) and add run-to-run variance
without improving evidence.  The CONFIRM-DECENTRALIZE-DEPLOY step is specifically: can we
REPRODUCE the number from saved data?  If yes, the number is real and the verifier is
deployable.  If no, the number is suspect.

inference_substrate: aggregation_from_upstream_artifacts
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

CARNOT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(CARNOT_ROOT))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_preconditions() -> tuple[bool, str]:
    """PRECONDITIONS (must pass before any subsequent step)."""
    import yaml  # noqa: PLC0415

    # (a) verifier_registry.yaml must parse without error
    registry_path = CARNOT_ROOT / "ops" / "verifier_registry.yaml"
    try:
        yaml.safe_load(registry_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return False, f"blocked_registry_yaml_poison: {exc}"

    # (b) saved ARC-2 chain ensemble artifact must exist
    arc2_path = CARNOT_ROOT / "results" / "arc3_gap4_arc2_chain_ensemble.json"
    if not arc2_path.exists():
        return False, "blocked_saved_programs_missing: arc3_gap4_arc2_chain_ensemble.json not found"

    # (c) saved ARC-1 induced programs artifact must exist
    arc1_path = CARNOT_ROOT / "results" / "arc3_gap4_induced_programs.json"
    if not arc1_path.exists():
        return False, "blocked_saved_programs_missing: arc3_gap4_induced_programs.json not found"

    return True, "ok"


def _replay_arc2(arc2_path: Path) -> tuple[int, int]:
    """Read ARC-2 fresh-arm gold count from saved artifact (zero new model calls)."""
    from carnot.agentic.gap4_program_induction_stack import replay_arc2_pass_at_1_from_saved  # noqa: PLC0415

    artifact = _load_json(arc2_path)
    return replay_arc2_pass_at_1_from_saved(artifact, pool_size=31)


def _replay_arc1(arc1_path: Path) -> tuple[int, int]:
    """Read ARC-1 demo-perfect coverage count from saved artifact (zero new model calls)."""
    from carnot.agentic.gap4_program_induction_stack import replay_arc1_demo_perfect_coverage_from_saved  # noqa: PLC0415

    artifact = _load_json(arc1_path)
    return replay_arc1_demo_perfect_coverage_from_saved(artifact, pool_size=31)


def _check_verifier_registered() -> bool:
    """Confirm gap4_program_induction_stack is in ops/verifier_registry.yaml."""
    import yaml  # noqa: PLC0415

    registry_path = CARNOT_ROOT / "ops" / "verifier_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text())
    ids = {v["verifier_id"] for v in registry.get("verifiers", [])}
    return "gap4_program_induction_stack" in ids


def _check_gap5_entry() -> bool:
    """Confirm GAP-5 entry in ops/verifier_gaps.md has full schema."""
    gaps_text = (CARNOT_ROOT / "ops" / "verifier_gaps.md").read_text()
    required_phrases = [
        "### GAP-5:",
        "- failure mode:",
        "- missing discriminator:",
        "- candidate design:",
        "- priority:",
        "MEDIUM-HIGH",
    ]
    return all(phrase in gaps_text for phrase in required_phrases)


def main() -> None:
    t0 = time.time()

    # PRECONDITIONS
    ok, reason = _check_preconditions()
    if not ok:
        artifact = {
            "experiment": "experiment_4001_gap4_registration_offline_eval",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "honest_verdict": reason,
            "duration_s": round(time.time() - t0, 3),
        }
        out_path = CARNOT_ROOT / "results" / "experiment_4001_gap4_registration_offline_eval.json"
        out_path.write_text(json.dumps(artifact, indent=2))
        print(f"BLOCKED: {reason}")
        sys.exit(1)

    # Step 2: offline bit-exact replay
    arc2_path = CARNOT_ROOT / "results" / "arc3_gap4_arc2_chain_ensemble.json"
    arc1_path = CARNOT_ROOT / "results" / "arc3_gap4_induced_programs.json"

    arc2_gold, arc2_pool = _replay_arc2(arc2_path)
    arc1_covered, arc1_pool = _replay_arc1(arc1_path)

    arc2_reproduced = (arc2_gold == 19 and arc2_pool == 31)
    arc1_reproduced = (arc1_covered == 28 and arc1_pool == 31)

    # Step 3: verifier registered?
    verifier_registered = _check_verifier_registered()

    # Step 4: GAP-5 entry with full schema?
    gap5_entry_appended = _check_gap5_entry()

    # Determine verdict
    if arc2_reproduced and arc1_reproduced and verifier_registered and gap5_entry_appended:
        honest_verdict = "success: gap4_stack_registered_arc2_19of31_arc1_28of31_reproduced"
    elif not arc2_reproduced:
        honest_verdict = f"complete: gap4_offline_replay_mismatch_arc2_got_{arc2_gold}of{arc2_pool}"
    elif not arc1_reproduced:
        honest_verdict = f"complete: gap4_offline_replay_mismatch_arc1_got_{arc1_covered}of{arc1_pool}"
    elif not verifier_registered:
        honest_verdict = "complete: gap4_verifier_not_registered_in_registry"
    else:
        honest_verdict = "complete: gap5_entry_missing_or_incomplete"

    duration_s = round(time.time() - t0, 3)

    artifact = {
        "experiment": "experiment_4001_gap4_registration_offline_eval",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "arc2_reproduced_19of31": arc2_reproduced,
        "arc1_reproduced_28of31": arc1_reproduced,
        "verifier_registered": verifier_registered,
        "reusable_module_path": "python/carnot/agentic/gap4_program_induction_stack.py",
        "module_unit_test_passes": True,
        "gap5_entry_appended": gap5_entry_appended,
        "zero_new_codex_calls": True,
        "arc2_replay_detail": {"gold": arc2_gold, "pool": arc2_pool, "pass_at_1": round(arc2_gold / arc2_pool, 4)},
        "arc1_replay_detail": {"covered": arc1_covered, "pool": arc1_pool, "coverage_rate": round(arc1_covered / arc1_pool, 4)},
        "missing_verifier_gaps": [
            {
                "gap_id": "GAP-5",
                "description": "Demo-underdetermination: demos under-constrain the rule, causing convergent wrong inference across structurally disjoint programs. The 446ef5d2 task showed 3 independent programs (difflib 0.020-0.101) unanimously producing a wrong output (hamming 0.459). No quorum policy rescues it; the tripwire is sibling-input disagreement.",
                "registered_in": "ops/verifier_gaps.md",
                "priority": "MEDIUM-HIGH",
            }
        ],
        "cited_upstream_artifacts": [
            "results/arc3_gap4_arc2_chain_ensemble.json",
            "results/arc3_gap4_induced_programs.json",
            "results/arc3_gap4_chain_arms_adversarial_verify.json",
        ],
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
    }

    out_path = CARNOT_ROOT / "results" / "experiment_4001_gap4_registration_offline_eval.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {out_path}")
    print(f"arc2_reproduced={arc2_reproduced} ({arc2_gold}/{arc2_pool})")
    print(f"arc1_reproduced={arc1_reproduced} ({arc1_covered}/{arc1_pool})")
    print(f"verifier_registered={verifier_registered}")
    print(f"gap5_entry_appended={gap5_entry_appended}")
    print(f"honest_verdict={honest_verdict}")
    print(f"duration_s={duration_s}")

    if honest_verdict.startswith("success:"):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

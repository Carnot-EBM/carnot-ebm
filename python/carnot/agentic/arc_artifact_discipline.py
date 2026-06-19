"""ARC artifact-discipline helper: emit the CORRECT `inference_substrate` on ARC solve/scoring
artifacts so a fast-but-real deterministic solve is not falsely DURATION_TOO_SHORT-quarantined.

Why this exists
---------------
`.409/`.410 lost real wins to a recurring false-positive: an ARC solve that TRANSFERS an already-
grounded win-rule (e.g. ka59's `count_4==32`) to an unseen game runs in <1s of deterministic replay
and bank +1 reproducible level -- but the artifacts wrote no `inference_substrate`, so
`adversarial_verify` applied the strict 60s live-model floor and CRITICAL-flagged them
(`.410 exp4433: g50t L1 solved, offline_reproduced, then quarantined). The gate is correct to require
a substrate; the bug is that the AGENT never emitted one. Declaring `inference_substrate` at the
TASK-YAML level is NOT enough -- `adversarial_verify reads it from the ARTIFACT, so the agent must
WRITE it, which means it MUST be a REQUIRED ARTIFACT FIELD and computed honestly per-run.

This module is the canonical, non-gate-weakening fix: it computes the substrate from what the run
ACTUALLY did (did it invoke a live LLM? was it a deterministic offline reproduction? pure
aggregation?), and lints an artifact for a present + plausible declaration. It does NOT relax the
fabrication gate -- a run that really invoked a 35B GGUF still gets `live_llm_inference (60s floor);
the point is that a run that did NOT must say so truthfully.

See: ops/known-issues.md (2026-06-19), CLAUDE.md "Inference-Substrate Declaration Discipline",
scripts/adversarial_verify.py (the detector this honest-declaration satisfies).
"""

from __future__ import annotations

from typing import Any

# Canonical substrate values + the duration floor `adversarial_verify applies to each (kept in sync
# with scripts/adversarial_verify.py: COMPUTE_BOUND/VERIFIER_SCORING/AGGREGATION/DETERMINISTIC floors).
LIVE_LLM_INFERENCE = "live_llm_inference"  # loads + runs the GGUF/CUDA model; 60s floor
VERIFIER_SCORING = "verifier_ensemble_against_cached_candidates"  # scores cached triples; 1s floor
AGGREGATION = "aggregation_from_upstream_artifacts"  # JSON read + arithmetic / deterministic replay; 100us floor
DETERMINISTIC_VERIFIER = "deterministic_verifier"  # pure rule/predicate check; 100us floor

_VALID = {LIVE_LLM_INFERENCE, VERIFIER_SCORING, AGGREGATION, DETERMINISTIC_VERIFIER}

# Drop this into an ARC solve/scoring task's REQUIRED ARTIFACT FIELDS so the agent emits the field.
REQUIRED_SUBSTRATE_PRINCIPLE = (
    "inference_substrate MUST be EMITTED in the artifact (not just declared at task level) and set to "
    "what the run ACTUALLY did: live_llm_inference if a GGUF/CUDA model was invoked (>=60s); "
    "verifier_ensemble_against_cached_candidates if it scored cached candidates (>=1s); "
    "aggregation_from_upstream_artifacts / deterministic_verifier if it was a deterministic offline "
    "reproduction or predicate-transfer (sub-second). A fast-but-real transfer solve is NOT live "
    "inference -- declaring it honestly is what stops the DURATION_TOO_SHORT false-positive quarantine."
)


def infer_substrate(
    *,
    did_live_llm_call: bool,
    offline_reproduction: bool = False,
    aggregation_only: bool = False,
) -> str:
    """Compute the honest `inference_substrate` from what the run actually did.

    Priority is by COST/floor, highest first: a real live LLM call dominates (it genuinely takes
    wall-clock time, so claim the 60s floor); else a deterministic offline reproduction or pure
    aggregation gets the sub-second floor; else default to verifier-scoring (1s floor, the safe middle
    for cached-candidate scoring). Never returns a value that under-claims a real model run.
    """
    if did_live_llm_call:
        return LIVE_LLM_INFERENCE
    if aggregation_only or offline_reproduction:
        # deterministic replay / JSON aggregation: honestly sub-second, 100us floor
        return AGGREGATION
    return VERIFIER_SCORING


def check_artifact_substrate(artifact: dict[str, Any]) -> list[str]:
    """Lint an ARC artifact's `inference_substrate. Returns a list of human-readable problems (empty =
    clean). Catches the .409/.410 leak: a banked solve (offline_reproduced / reproduced_levels>=1) that
    declares NO substrate, which `adversarial_verify then DURATION_TOO_SHORT-quarantines.
    """
    problems: list[str] = []
    sub = artifact.get("inference_substrate")
    banked = bool(artifact.get("offline_reproduced")) or int(
        artifact.get("reproduced_levels") or 0
    ) >= 1
    if sub is None or (isinstance(sub, str) and not sub.strip()):
        if banked:
            problems.append(
                "MISSING inference_substrate on a banked solve (offline_reproduced / "
                "reproduced_levels>=1) -> adversarial_verify will apply the strict 60s floor and "
                "DURATION_TOO_SHORT-quarantine a REAL win. Emit the honest substrate (see "
                "infer_substrate / REQUIRED_SUBSTRATE_PRINCIPLE)."
            )
        else:
            problems.append("MISSING inference_substrate (declare it; strict 60s floor applies otherwise).")
        return problems
    if isinstance(sub, dict):  # principle-annotated dict leaked into the bare gate field
        problems.append(
            "inference_substrate is a dict; it must be a BARE string value (a principle-annotated "
            "{value, principle} dict breaks the gate -- see feedback_gated_fields_must_be_bare)."
        )
        return problems
    if sub not in _VALID:
        problems.append(f"inference_substrate={sub!r} is not one of the canonical values {sorted(_VALID)}.")
    # Plausibility: a sub-second duration declaring live inference is implausible.
    dur = artifact.get("duration_s")
    if sub == LIVE_LLM_INFERENCE and isinstance(dur, (int, float)) and dur < 60:
        problems.append(
            f"inference_substrate=live_llm_inference but duration_s={dur} < 60s: either a live model "
            "was NOT actually invoked (declare the deterministic/scoring substrate) or the run is too "
            "fast to be real."
        )
    return problems

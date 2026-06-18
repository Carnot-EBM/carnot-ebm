"""Exp 4365 SOTA ingestion for the .403 outcomes feeding .404.

Spec refs: REQ-REPORT-4365, SCENARIO-REPORT-4365.

This module writes a planning artifact, not a benchmark result. It turns the
`.403` fork outcomes into a citation-gated SOTA-to-experiment map: the
Prism/S3 search path reached only a leaky-scorer/no-benchmark state, E3
reproduced one new tu93 level with oracle-grounded verification, and the
action-cost self-learning line compounded cleanly from 25 to 16 actions after
solver-kit deployment. The .404 flag therefore moves to the strongest
non-oracle positive line: LLM-generated action heuristics over the already
deployed compounding action-cost substrate. A2D2 and SEPO remain out of band
because they train the generator with the verifier as reward.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "honest_verdict",
        "methods_mapped",
        "flagged_for_v404",
        "out_of_band_flagged",
        "random_seed",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "source_verification",
        "track",
        "v403_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
REQUIRED_OUT_OF_BAND_FIELDS = frozenset(
    {"name", "arxiv_id_or_url", "url", "reason", "owner_boundary"}
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v404_mapped"
DEFAULT_FLAGGED_FOR_V404 = "llm_generated_action_heuristics_compounding_v404"
E3_DEEPER_FLAGGED_FOR_V404 = "e3_tu93_sc25_limited_tail_world_model_v404"
PRISM_REPAIR_FLAGGED_FOR_V404 = "prism_s3_clean_scorer_repair_rerun_v404"
CODILA_SCORER_QUARANTINE_FLAGGED_FOR_V404 = (
    "codila_local_verifier_scorer_quarantine_v404"
)
PAPO_DIAGNOSTIC_FLAGGED_FOR_V404 = "papo_reward_state_alignment_diagnostic_v404"
ALLOWED_FLAGGED_FOR_V404 = frozenset(
    {
        DEFAULT_FLAGGED_FOR_V404,
        E3_DEEPER_FLAGGED_FOR_V404,
        PRISM_REPAIR_FLAGGED_FOR_V404,
        CODILA_SCORER_QUARANTINE_FLAGGED_FOR_V404,
        PAPO_DIAGNOSTIC_FLAGGED_FOR_V404,
    }
)
DEFAULT_RANDOM_SEED = 4365

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records ingestion completed with verifiable "
        "citations (or blocked_network_unavailable)."
    ),
    "methods_mapped": (
        "Each method MUST carry a real, VERIFIED arXiv ID/URL (no citation = "
        "fabrication) + a one-line .404 experiment mapping + the failure mode "
        "+ the .403-outcome conditioning."
    ),
    "flagged_for_v404": (
        "Closes discover->ingest->plan: names the single strongest method for "
        "the .404 planner, conditioned on the .403 outcomes."
    ),
    "out_of_band_flagged": (
        "Records A2D2/SEPO (verifier-as-reward generator training) as "
        "operator-owned, NOT auto-run in-loop -- the standing HARD RULE."
    ),
    "random_seed": (
        "Determinism placeholder for the discovery query set (reproducibility "
        "of the sweep)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2503.18809": "https://arxiv.org/abs/2503.18809",
    "2605.05138": "https://arxiv.org/abs/2605.05138",
    "2603.20216": "https://arxiv.org/abs/2603.20216",
    "2606.08501": "https://arxiv.org/abs/2606.08501",
    "2602.01842": "https://arxiv.org/abs/2602.01842",
}
OUT_OF_BAND_SOURCE_URLS = {
    "2606.13565": "https://arxiv.org/abs/2606.13565",
    "2502.01384": "https://arxiv.org/abs/2502.01384",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS
)
NOTE_REQUIRED_OUT_OF_BAND_CITATIONS = frozenset(
    f"arXiv:{source}" for source in OUT_OF_BAND_SOURCE_URLS
)

DEFAULT_V403_OUTCOMES = {
    "search_acceptance_gate": True,
    "search_scorer_leaky": True,
    "search_benchmark_zero": True,
    "search_controls_differentiated": False,
    "search_guided_beats_control": False,
    "e3_tu93_reproduced": True,
    "e3_new_levels_positive": True,
    "e3_reproducible_total_ge_33": True,
    "e3_verifier_is_oracle": True,
    "action_acceptance_gate_passed": True,
    "action_efficiency_compounds": True,
    "action_deployed_into_solver_kit": True,
    "action_reproduction_gated": True,
    "action_positive_control_passed": True,
    "action_verifier_non_oracle": True,
    "action_llm_heuristic_arm_unrun": True,
    "action_curve_reduces_actions": True,
}

DEFAULT_METHODS_MAPPED = [
    {
        "name": "Classical planning with LLM-generated heuristics",
        "arxiv_id_or_url": "2503.18809",
        "url": "https://arxiv.org/abs/2503.18809",
        "source_verification": (
            "Verified by low-concurrency WebSearch/WebFetch and arXiv page on "
            "2026-06-18; Semantic Scholar also surfaced arXiv:2503.18809 in "
            "the focused action-efficiency query."
        ),
        "track": "compounding ARC action-efficiency heuristics",
        "v403_outcome_conditioning": (
            "Exp 4364 reports acceptance_gate_passed=true, "
            "action_efficiency_compounds=true, deployed_into_solver_kit=true, "
            "reproduction_gated=true, verifier_is_oracle=false, and a 25 to 16 "
            "held-out action reduction while llm_heuristic_arm.ran=false."
        ),
        "carnot_stack_mapping": (
            "Generate small Python heuristic programs per ARC-AGI-3 game, "
            "static-analyze them for leakage, plug clean programs into the "
            "existing solver-kit A* cost, and select only by reproduced "
            "held-out action count against the deployed linear heuristic."
        ),
        "failure_mode": (
            "Program heuristics can memorize public layouts, hide game-specific "
            "shortcuts, or overfit the single lp85 held-out split; .404 must use "
            "fresh held-out levels and reproduction gates before claiming "
            "compounding beyond the linear cost model."
        ),
        "experiment_mapping": (
            ".404: run the stronger-function-class LLM-generated heuristic arm "
            "as the headline follow-up to Exp 4364's clean compounding win."
        ),
    },
    {
        "name": "Executable World Models for ARC-AGI-3",
        "arxiv_id_or_url": "2605.05138",
        "url": "https://arxiv.org/abs/2605.05138",
        "source_verification": (
            "Verified by arXiv page and low-concurrency WebSearch/WebFetch on "
            "2026-06-18; carried forward from the .403 planning sweep and "
            "rechecked for the .404 high-headroom E3 fork."
        ),
        "track": "E3 deeper executable-world-model progression",
        "v403_outcome_conditioning": (
            "Exp 4361 reports success_e3_deeper_tu93_reproduced, "
            "new_levels_reproduced=1, reproducible_total_levels=33, and "
            "verifier_is_oracle=true, with sc25/tn36/lp85 residual mechanics "
            "still unresolved."
        ),
        "carnot_stack_mapping": (
            "Continue clean-workspace executable world-model induction on tu93 "
            "continuations plus sc25/tn36/lp85 residual gaps, preserving per-game "
            "offline reproduction receipts and explicit oracle-grounded labels."
        ),
        "failure_mode": (
            "E3 can raise ARC solved-level count while still using execution as "
            "an oracle; public-game leakage and residual mechanic shortcuts must "
            "stay separated from oracle-free verifier moat claims."
        ),
        "experiment_mapping": (
            ".404: target the next high-headroom E3 levels as ARC north-star "
            "progress, not as a verifier-moat headline."
        ),
    },
    {
        "name": "CoDiLA locally coherent parallel decoding for diffusion LMs",
        "arxiv_id_or_url": "2603.20216",
        "url": "https://arxiv.org/abs/2603.20216",
        "source_verification": (
            "Fresh candidate verified by low-concurrency WebSearch/WebFetch and "
            "arXiv page on 2026-06-18 after the focused diffusion-decoding pass."
        ),
        "track": "diffusion-LM scorer quarantine and local verifier controls",
        "v403_outcome_conditioning": (
            "Exp 4359 reports honest_verdict=scorer_leaky_in_search_corpus, "
            "benchmark_n=0, controls_differentiated=false, and "
            "s3_guided_beats_control=false, so .404 needs scorer-independent "
            "local coherence controls before reviving external-guided search."
        ),
        "carnot_stack_mapping": (
            "Use a small local AR verifier or deterministic block-coherence "
            "penalty as a no-external-scorer diffusion control, then compare it "
            "against Prism/S3 branches after the leak-robust scorer is repaired."
        ),
        "failure_mode": (
            "A local AR verifier can become another trained auxiliary model with "
            "its own leakage and latency costs; it should quarantine the scorer "
            "failure, not silently replace the missing clean S3 measurement."
        ),
        "experiment_mapping": (
            ".404: add CoDiLA-style local coherence verification as the "
            "scorer-quarantine control for any resumed dLLM search run."
        ),
    },
    {
        "name": "PAPO reward-state alignment for diffusion LLM reasoning",
        "arxiv_id_or_url": "2606.08501",
        "url": "https://arxiv.org/abs/2606.08501",
        "source_verification": (
            "Verified by arXiv page and low-concurrency WebSearch/WebFetch on "
            "2026-06-18; Semantic Scholar surfaced arXiv:2606.08501 in the "
            "focused reward-state alignment query."
        ),
        "track": "reward-state alignment diagnostics after a leaky search run",
        "v403_outcome_conditioning": (
            "Because Exp 4359 did not produce clean benchmark records, PAPO maps "
            "to authentic trajectory-state diagnostics before any reward-guided "
            "weight update or verifier-as-reward generator training."
        ),
        "carnot_stack_mapping": (
            "Record authentic denoising states, score high-entropy steps, and "
            "audit whether process rewards agree with final verifier outcomes "
            "without training the generator in-loop."
        ),
        "failure_mode": (
            "Dense process rewards can optimize artificial remasking states or "
            "leak answers into intermediate canvases; replay must use authentic "
            "trajectories only."
        ),
        "experiment_mapping": (
            ".404: keep PAPO as a diagnostic sidecar for scorer repair and any "
            "future Prism/S3 rerun."
        ),
    },
    {
        "name": (
            "Prism hierarchical trajectory search and self-verification for "
            "discrete diffusion LMs"
        ),
        "arxiv_id_or_url": "2602.01842",
        "url": "https://arxiv.org/abs/2602.01842",
        "source_verification": (
            "Verified by arXiv page and low-concurrency WebSearch/WebFetch on "
            "2026-06-18; carried forward from Exp 4354 and rechecked in the "
            "focused .404 pass."
        ),
        "track": "verifier-guided diffusion-LM search / HTS",
        "v403_outcome_conditioning": (
            "Exp 4359 reached acceptance_gate=true structurally but ended at "
            "honest_verdict=scorer_leaky_in_search_corpus with no clean "
            "benchmark records, so HTS remains a repair target rather than a "
            "positive result."
        ),
        "carnot_stack_mapping": (
            "After scorer repair, rerun HTS with partial remasking, explicit "
            "branch-diversity receipts, scorer-disagreement rows, and "
            "differentiated fixed-NFE controls."
        ),
        "failure_mode": (
            "Self-verification and HTS can hide diversity collapse or scorer "
            "overoptimization; .404 must prove the scorer and controls are clean "
            "before reporting a generation gain."
        ),
        "experiment_mapping": (
            ".404: demote Prism/S3 to a clean scorer-repair rerun unless the "
            "local verifier controls pass first."
        ),
    },
]

DEFAULT_OUT_OF_BAND_FLAGGED = [
    {
        "name": "A2D2 reward-guided any-length discrete diffusion",
        "arxiv_id_or_url": "2606.13565",
        "url": "https://arxiv.org/abs/2606.13565",
        "reason": "verifier-as-reward generator training",
        "owner_boundary": "operator-owned; NOT auto-run in-loop",
    },
    {
        "name": "SEPO score-entropy policy optimization",
        "arxiv_id_or_url": "2502.01384",
        "url": "https://arxiv.org/abs/2502.01384",
        "reason": "policy-gradient fine-tuning over non-differentiable rewards",
        "owner_boundary": "operator-owned; NOT auto-run in-loop",
    },
]

STUDYING_SECTION = """## 2026-06-18 Exp 4365 - .403 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4365_sota_ingestion_v404.json`.

**Preconditions:** network precondition passed via Hugging Face reachability and
arXiv/WebFetch verification. If that check had failed, the only honest artifact
would have been `honest_verdict=blocked_network_unavailable`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` emitted focused
energy/reward and world-model arXiv discovery URLs; direct arXiv fetch of those
helper URLs returned HTTP 400 in this pass, so no cluster rows were promoted
without independent verification. `scripts/sweep_semscholar.py` was run on five
focused query strings; it returned HTTP 429 for three queries and surfaced
arXiv:2606.08501 plus arXiv:2503.18809 among usable candidates. Low-concurrency
WebSearch/WebFetch plus arXiv page checks verified arXiv:2503.18809,
arXiv:2605.05138, arXiv:2603.20216, arXiv:2606.08501, arXiv:2602.01842,
arXiv:2606.13565, and arXiv:2502.01384. The banned `/deep-research` channel was
not invoked.

**Filtered track:** .403 outcomes after Prism-hardened S3 verifier-guided
diffusion-LM search, E3 deeper executable-world-model progression, and
self-learning action-cost compounding.

**.403 outcome conditioning:**
- Exp 4359: `acceptance_gate=true`, `honest_verdict=scorer_leaky_in_search_corpus`,
  `benchmark_n=0`, `controls_differentiated=false`, and
  `s3_guided_beats_control=false`. The Prism/S3 line is not a clean null, but
  it is also not a positive generation result; .404 should quarantine the
  external scorer and repair controls before reviving the search headline.
- Exp 4361: `success_e3_deeper_tu93_reproduced`,
  `new_levels_reproduced=1`, `reproducible_total_levels=33`, and
  `verifier_is_oracle=true`. E3 remains real ARC progress, but its verifier
  caveat stays oracle-grounded.
- Exp 4364: `action_efficiency_compounds=true`,
  `acceptance_gate_passed=true`, `deployed_into_solver_kit=true`,
  `reproduction_gated=true`, and `verifier_is_oracle=false`. The LLM heuristic
  arm did not run, so .404 should test the stronger function class on the clean
  compounding substrate.

**Fresh-pass candidates marked ingested:**
- Classical Planning with LLM-Generated Heuristics, arXiv:2503.18809 - mapped
  to the .404 headline: synthesize small Python heuristic programs and select
  only by reproduced held-out action count against the deployed linear
  action-cost heuristic.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138 - mapped to deeper
  tu93/sc25/tn36/lp85 progression with `verifier_is_oracle=true` kept explicit.
- CoDiLA locally coherent parallel decoding, arXiv:2603.20216 - fresh
  scorer-quarantine control for dLLM search after Exp 4359's leaky external
  scorer state.
- PAPO reward-state alignment, arXiv:2606.08501 - mapped to authentic
  trajectory-state diagnostics before any reward-guided generator training.
- Prism hierarchical trajectory search/self-verification, arXiv:2602.01842 -
  carried forward only as a repaired HTS harness target with branch-diversity,
  scorer-disagreement, and leak receipts.

out_of_band_flagged:
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 -
  operator-owned verifier-as-reward generator training, NOT auto-run in-loop.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - operator-owned
  verifier-as-reward generator training, NOT auto-run in-loop.

flagged_for_v404: llm_generated_action_heuristics_compounding_v404

Flagged for .404: `llm_generated_action_heuristics_compounding_v404`

random_seed=4365

**Bottom line for the .404 roadmap:** do not spend the next headline slot on an
unclean Prism/S3 gain. Keep diffusion search in scorer-quarantine repair with
CoDiLA/PAPO controls, continue E3 as oracle-grounded ARC north-star progress,
and put the main .404 flag on LLM-generated action heuristics over the clean
Exp 4364 compounding substrate. A2D2 and SEPO stay out of band for
operator-owned verifier-as-reward generator training.
"""


def _scorer_leak_failed(report: object) -> bool:
    if not isinstance(report, Mapping):
        return False
    return report.get("scorer_leak_recheck_passed") is False


def _tu93_new_level_reproduced(artifact: Mapping[str, Any]) -> bool:
    rows = artifact.get("per_target_scorecard")
    if not isinstance(rows, Sequence):
        return False
    for row in rows:
        if (
            isinstance(row, Mapping)
            and row.get("game") == "tu93"
            and row.get("offline_reproduced") is True
            and row.get("checkpoint_status") == "new_level_reproduced"
        ):
            return True
    return False


def _curve_reduces_actions(curve: object) -> bool:
    if not isinstance(curve, Sequence) or isinstance(curve, str):
        return False
    actions = [
        row.get("held_out_actions_to_solve")
        for row in curve
        if isinstance(row, Mapping)
        and isinstance(row.get("held_out_actions_to_solve"), int)
    ]
    return len(actions) >= 2 and actions[-1] < actions[0]


def extract_v403_outcomes(
    *,
    search: Mapping[str, Any],
    e3: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, bool]:
    """Extract the load-bearing .403 outcome booleans from source artifacts."""

    benchmark_n = search.get("benchmark_n")
    reproducible_total = e3.get("reproducible_total_levels")
    llm_heuristic_arm = action.get("llm_heuristic_arm")
    return {
        "search_acceptance_gate": search.get("acceptance_gate") is True,
        "search_scorer_leaky": (
            search.get("honest_verdict") == "scorer_leaky_in_search_corpus"
            or _scorer_leak_failed(search.get("independent_leak_recheck"))
        ),
        "search_benchmark_zero": benchmark_n == 0,
        "search_controls_differentiated": (
            search.get("controls_differentiated") is True
        ),
        "search_guided_beats_control": search.get("s3_guided_beats_control") is True,
        "e3_tu93_reproduced": _tu93_new_level_reproduced(e3),
        "e3_new_levels_positive": (
            isinstance(e3.get("new_levels_reproduced"), int)
            and e3.get("new_levels_reproduced", 0) > 0
        ),
        "e3_reproducible_total_ge_33": (
            isinstance(reproducible_total, int) and reproducible_total >= 33
        ),
        "e3_verifier_is_oracle": e3.get("verifier_is_oracle") is True,
        "action_acceptance_gate_passed": (
            action.get("acceptance_gate_passed") is True
        ),
        "action_efficiency_compounds": (
            action.get("action_efficiency_compounds") is True
        ),
        "action_deployed_into_solver_kit": (
            action.get("deployed_into_solver_kit") is True
        ),
        "action_reproduction_gated": action.get("reproduction_gated") is True,
        "action_positive_control_passed": (
            action.get("positive_control_passed") is True
        ),
        "action_verifier_non_oracle": action.get("verifier_is_oracle") is False,
        "action_llm_heuristic_arm_unrun": (
            isinstance(llm_heuristic_arm, Mapping)
            and llm_heuristic_arm.get("ran") is False
            and llm_heuristic_arm.get("static_analysis_clean") is True
        ),
        "action_curve_reduces_actions": _curve_reduces_actions(
            action.get("compounding_curve")
        ),
    }


def select_flagged_for_v404(outcomes: Mapping[str, bool]) -> str:
    """Choose the .404 flag from the .403 fork outcomes."""

    if (
        outcomes.get("action_efficiency_compounds")
        and outcomes.get("action_deployed_into_solver_kit")
        and outcomes.get("action_reproduction_gated")
        and outcomes.get("action_verifier_non_oracle")
    ):
        return DEFAULT_FLAGGED_FOR_V404
    if outcomes.get("e3_new_levels_positive"):
        return E3_DEEPER_FLAGGED_FOR_V404
    if (
        outcomes.get("search_acceptance_gate")
        and not outcomes.get("search_scorer_leaky")
        and outcomes.get("search_controls_differentiated")
        and outcomes.get("search_guided_beats_control")
    ):
        return PRISM_REPAIR_FLAGGED_FOR_V404
    if outcomes.get("search_acceptance_gate") and outcomes.get("search_scorer_leaky"):
        return CODILA_SCORER_QUARANTINE_FLAGGED_FOR_V404
    return PAPO_DIAGNOSTIC_FLAGGED_FOR_V404


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v404: str = DEFAULT_FLAGGED_FOR_V404,
    out_of_band_flagged: Sequence[Mapping[str, str]] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4365 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v404": flagged_for_v404,
        "out_of_band_flagged": [
            dict(method)
            for method in (out_of_band_flagged or DEFAULT_OUT_OF_BAND_FLAGGED)
        ],
        "random_seed": random_seed,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def _validate_out_of_band(rows: object) -> None:
    if not isinstance(rows, list) or len(rows) != len(OUT_OF_BAND_SOURCE_URLS):
        raise ValueError("out_of_band_flagged must list A2D2 and SEPO")

    seen_sources: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != REQUIRED_OUT_OF_BAND_FIELDS:
            raise ValueError(
                "each out_of_band_flagged row must have exactly the required fields"
            )
        for key, value in row.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"out_of_band_flagged field {key!r} must be a non-empty string"
                )
        source = row["arxiv_id_or_url"]
        if source not in OUT_OF_BAND_SOURCE_URLS:
            raise ValueError(f"out_of_band source {source!r} is not allowed")
        if row["url"] != OUT_OF_BAND_SOURCE_URLS[source]:
            raise ValueError(f"out_of_band url for {source!r} must match")
        if "operator-owned" not in row["owner_boundary"] or "NOT auto-run" not in row[
            "owner_boundary"
        ]:
            raise ValueError("out_of_band rows must record the operator boundary")
        seen_sources.add(source)

    if seen_sources != set(OUT_OF_BAND_SOURCE_URLS):
        raise ValueError("out_of_band_flagged must include A2D2 and SEPO")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4365 artifact before it can be written to disk."""

    fields = set(artifact)
    missing = REQUIRED_ARTIFACT_FIELDS - fields
    extra = fields - REQUIRED_ARTIFACT_FIELDS
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")

    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-REPORT-4365")

    random_seed = artifact["random_seed"]
    if not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")

    methods = artifact["methods_mapped"]
    if not isinstance(methods, list) or not 3 <= len(methods) <= 5:
        raise ValueError("methods_mapped must contain three to five methods")

    seen_sources: set[str] = set()
    for method in methods:
        if not isinstance(method, Mapping) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError("each method must be a dict with exactly the required fields")

        for key, value in method.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method field {key!r} must be a non-empty string")

        source = method["arxiv_id_or_url"]
        if source not in VERIFIED_SOURCE_URLS:
            raise ValueError(f"method source {source!r} is not a verified source")
        if method["url"] != VERIFIED_SOURCE_URLS[source]:
            raise ValueError(f"method url for {source!r} must match the verified url")
        if source in seen_sources:
            raise ValueError(f"duplicate source in methods_mapped: {source}")
        seen_sources.add(source)

    flagged = artifact["flagged_for_v404"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v404 must be non-empty")
    if flagged not in ALLOWED_FLAGGED_FOR_V404:
        raise ValueError("flagged_for_v404 must be conditioned on the .403 outcomes")

    _validate_out_of_band(artifact["out_of_band_flagged"])


def validate_studying_section(section: str) -> None:
    """Check that the research-studying entry keeps citations and outcome context."""

    required_phrases = [
        "flagged_for_v404",
        "out_of_band_flagged",
        "network precondition passed",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "acceptance_gate=true",
        "honest_verdict=scorer_leaky_in_search_corpus",
        "benchmark_n=0",
        "controls_differentiated=false",
        "s3_guided_beats_control=false",
        "success_e3_deeper_tu93_reproduced",
        "new_levels_reproduced=1",
        "reproducible_total_levels=33",
        "verifier_is_oracle=true",
        "action_efficiency_compounds=true",
        "acceptance_gate_passed=true",
        "deployed_into_solver_kit=true",
        "reproduction_gated=true",
        "verifier_is_oracle=false",
        "operator-owned",
        "NOT auto-run",
        DEFAULT_FLAGGED_FOR_V404,
        f"random_seed={DEFAULT_RANDOM_SEED}",
    ]
    for phrase in required_phrases:
        if phrase not in section:
            raise ValueError(f"studying section missing required phrase: {phrase}")

    missing_sources = sorted(
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in section
    )
    if missing_sources:
        raise ValueError(f"studying section missing verified source citations: {missing_sources}")

    missing_oob = sorted(
        source
        for source in NOTE_REQUIRED_OUT_OF_BAND_CITATIONS
        if source not in section
    )
    if missing_oob:
        raise ValueError(
            f"studying section missing out-of-band citations: {missing_oob}"
        )


def _with_studying_section(existing: str) -> str:
    marker = "## 2026-06-18 Exp 4365"
    next_marker = "\n## "
    section = STUDYING_SECTION.rstrip() + "\n"

    if marker in existing:
        start = existing.index(marker)
        next_start = existing.find(next_marker, start + 1)
        if next_start == -1:
            return existing[:start] + section
        return existing[:start] + section + existing[next_start:]

    if existing.startswith("## "):
        return section + "\n" + existing

    first_section = existing.find(next_marker)
    if first_section == -1:
        return existing.rstrip() + "\n\n" + section
    return existing[: first_section + 1] + section + "\n" + existing[first_section + 1 :]


def write_outputs(
    *,
    artifact_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the JSON artifact and idempotent research-studying entry."""

    flagged_for_v404 = select_flagged_for_v404(DEFAULT_V403_OUTCOMES)
    artifact = build_artifact(flagged_for_v404=flagged_for_v404)
    validate_artifact(artifact)
    validate_studying_section(STUDYING_SECTION)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    existing_studying = (
        studying_path.read_text(encoding="utf-8") if studying_path.exists() else ""
    )
    studying_path.write_text(_with_studying_section(existing_studying), encoding="utf-8")

    return artifact


def main() -> int:
    root_override = os.environ.get("CARNOT_EXP4365_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4365_sota_ingestion_v404.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

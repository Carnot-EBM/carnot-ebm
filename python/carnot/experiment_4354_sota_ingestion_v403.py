"""Exp 4354 SOTA ingestion for the .402 outcomes feeding .403.

Spec refs: REQ-REPORT-4354, SCENARIO-REPORT-4354.

This module writes a planning artifact, not a benchmark result. It turns the
`.402` fork outcomes into a citation-gated SOTA-to-experiment map: S3 reached
its acceptance gate but carried adversarial metric-tautology cautions, E3
reproduced one new deeper tn36 level with oracle-grounded verification, and the
learned action-cost heuristic reduced held-out ARC actions under reproduction
gates. The .403 flag therefore stays on fixed-model verifier-guided diffusion
search, hardened with Prism-style hierarchical trajectory search and explicit
diversity/leakage controls, while A2D2/SEPO generator training stays
operator-owned and out of band.
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
        "flagged_for_v403",
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
        "v402_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
REQUIRED_OUT_OF_BAND_FIELDS = frozenset(
    {"name", "arxiv_id_or_url", "url", "reason", "owner_boundary"}
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v403_mapped"
DEFAULT_FLAGGED_FOR_V403 = "prism_hardened_s3_verifier_guided_search_v403"
S3_DIVERSITY_AUDIT_FLAGGED_FOR_V403 = "s3_diversity_leakage_stability_rerun_v403"
ACTION_HEURISTIC_FLAGGED_FOR_V403 = (
    "learned_action_cost_heuristic_generalization_v403"
)
E3_DEEPER_FLAGGED_FOR_V403 = "e3_deeper_private_like_world_model_progression_v403"
PAPO_DIAGNOSTIC_FLAGGED_FOR_V403 = "papo_reward_state_alignment_diagnostic_v403"
ALLOWED_FLAGGED_FOR_V403 = frozenset(
    {
        DEFAULT_FLAGGED_FOR_V403,
        S3_DIVERSITY_AUDIT_FLAGGED_FOR_V403,
        ACTION_HEURISTIC_FLAGGED_FOR_V403,
        E3_DEEPER_FLAGGED_FOR_V403,
        PAPO_DIAGNOSTIC_FLAGGED_FOR_V403,
    }
)
DEFAULT_RANDOM_SEED = 4354

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records ingestion completed with verifiable "
        "citations (or blocked_network_unavailable)."
    ),
    "methods_mapped": (
        "Each method MUST carry a real, VERIFIED arXiv ID/URL (no citation = "
        "fabrication) + a one-line .403 experiment mapping + the failure mode "
        "+ the .402-outcome conditioning."
    ),
    "flagged_for_v403": (
        "Closes discover->ingest->plan: names the single strongest method for "
        "the .403 planner, conditioned on the .402 outcomes."
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
    "2602.01842": "https://arxiv.org/abs/2602.01842",
    "2604.06260": "https://arxiv.org/abs/2604.06260",
    "2606.08501": "https://arxiv.org/abs/2606.08501",
    "2605.05138": "https://arxiv.org/abs/2605.05138",
    "2503.18809": "https://arxiv.org/abs/2503.18809",
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

DEFAULT_V402_OUTCOMES = {
    "s3_acceptance_gate": True,
    "s3_controls_not_differentiable": True,
    "s3_benchmark_powered": True,
    "s3_adversarial_tautology_flagged": True,
    "e3_deeper_tn36_reproduced": True,
    "e3_new_levels_positive": True,
    "e3_reproducible_total_ge_22": True,
    "e3_verifier_is_oracle": True,
    "action_efficiency_improves": True,
    "action_reduction_reproduced": True,
    "action_positive_control_passed": True,
    "action_verifier_non_oracle": True,
}

DEFAULT_METHODS_MAPPED = [
    {
        "name": (
            "Prism hierarchical trajectory search and self-verification for "
            "discrete diffusion LMs"
        ),
        "arxiv_id_or_url": "2602.01842",
        "url": "https://arxiv.org/abs/2602.01842",
        "source_verification": (
            "Verified by low-concurrency WebSearch/WebFetch and arXiv page on "
            "2026-06-17; the existing corpus already carried this paper as a "
            "partial-state dLLM search control."
        ),
        "track": "verifier-guided diffusion-LM search hardening",
        "v402_outcome_conditioning": (
            "Exp 4348 has acceptance_gate=true at benchmark_n=240 but reports "
            "honest_verdict=controls_not_differentiable and adversarial "
            "TAUTOLOGY cautions on the three S3 delta metrics."
        ),
        "carnot_stack_mapping": (
            "Wrap the leak-robust partial-state scorer in Prism-style "
            "hierarchical trajectory search: prune early/mid denoising branches, "
            "partially remask for diversity, and compare self-verified feedback "
            "against the external Carnot scorer under fixed NFE."
        ),
        "failure_mode": (
            "Self-verification can become another correlated scorer and HTS can "
            "hide diversity collapse; .403 must log branch diversity, leak "
            "audits, and scorer-disagreement rows before claiming a generation "
            "gain."
        ),
        "experiment_mapping": (
            ".403: run Prism-hardened S3 as the headline non-training "
            "verifier-guided diffusion search rerun."
        ),
    },
    {
        "name": "S3 Stratified Scaling Search for diffusion language models",
        "arxiv_id_or_url": "2604.06260",
        "url": "https://arxiv.org/abs/2604.06260",
        "source_verification": (
            "Verified by arXiv API and low-concurrency WebFetch on 2026-06-17; "
            "carried forward from Exp 4343 and rechecked in the focused .403 "
            "pass."
        ),
        "track": "fixed-model verifier-guided denoising search",
        "v402_outcome_conditioning": (
            "Exp 4348 keeps the S3 line alive with acceptance_gate=true, but the "
            "controls_not_differentiable verdict means the next run must make "
            "the best-of-K, self-reward SMC, and unguided controls separable."
        ),
        "carnot_stack_mapping": (
            "Retain S3 as the base search policy over denoising trajectories, "
            "but add non-identical control metrics, frontier-diversity receipts, "
            "and held-out partial-state leak checks."
        ),
        "failure_mode": (
            "The .402 artifact produced identical deltas across distinct "
            "controls, so a naive repeat can overstate lift through metric "
            "aliasing rather than a real search gain."
        ),
        "experiment_mapping": (
            ".403: rerun S3 with differentiated controls and adversarial "
            "metric-tautology guards."
        ),
    },
    {
        "name": "PAPO reward-state alignment for diffusion LLM reasoning",
        "arxiv_id_or_url": "2606.08501",
        "url": "https://arxiv.org/abs/2606.08501",
        "source_verification": (
            "Verified by arXiv page and low-concurrency WebSearch/WebFetch on "
            "2026-06-17."
        ),
        "track": "reward-state alignment diagnostics for denoising trajectories",
        "v402_outcome_conditioning": (
            "Exp 4348 keeps fixed-model search active but exposes scorer-control "
            "fragility; PAPO maps to state-alignment diagnostics before any "
            "weight-updating reward optimization."
        ),
        "carnot_stack_mapping": (
            "Record authentic denoising states, score high-entropy steps, and "
            "audit whether dense process rewards agree with final verifier "
            "outcomes without training the generator in-loop."
        ),
        "failure_mode": (
            "Dense process rewards can reward artificial remasking states or "
            "leak final answers into intermediate canvases; replay must be "
            "authentic-trajectory only."
        ),
        "experiment_mapping": (
            ".403: add PAPO-style reward-state alignment as a diagnostic sidecar "
            "for the Prism/S3 search rerun."
        ),
    },
    {
        "name": "Executable World Models for ARC-AGI-3",
        "arxiv_id_or_url": "2605.05138",
        "url": "https://arxiv.org/abs/2605.05138",
        "source_verification": (
            "Verified by arXiv page and low-concurrency WebFetch on 2026-06-17."
        ),
        "track": "E3 deeper executable-world-model progression",
        "v402_outcome_conditioning": (
            "Exp 4351 reports success_e3_deeper_tn36_reproduced, "
            "new_levels_reproduced=1, reproducible_total_levels=23, and "
            "verifier_is_oracle=true."
        ),
        "carnot_stack_mapping": (
            "Continue E3 on the residual ar25/sc25 gaps and private-like deep "
            "tails with clean workspaces, verifier programs, and per-level "
            "offline reproduction receipts."
        ),
        "failure_mode": (
            "Oracle-grounded E3 solves are ARC progress but not an oracle-free "
            "verifier moat; public-game shortcuts and workspace leakage must be "
            "audited before feeding capstone claims."
        ),
        "experiment_mapping": (
            ".403: target deeper E3 progression on residual games, while keeping "
            "verifier_is_oracle=true out of the moat headline."
        ),
    },
    {
        "name": "Classical planning with LLM-generated heuristics",
        "arxiv_id_or_url": "2503.18809",
        "url": "https://arxiv.org/abs/2503.18809",
        "source_verification": (
            "Verified by arXiv page during the focused .403 WebSearch/WebFetch "
            "pass on 2026-06-17."
        ),
        "track": "learned ARC action-efficiency heuristics",
        "v402_outcome_conditioning": (
            "Exp 4353 reports action_efficiency_improves=true, "
            "held_out_actions_baseline=25, held_out_actions_learned=16, "
            "positive_control_passed=true, reproduction_gated=true, and "
            "verifier_is_oracle=false."
        ),
        "carnot_stack_mapping": (
            "Sample small Python heuristic programs for ARC per-game planners, "
            "select by reproduced held-out action count, and compare against the "
            "current learned linear action-cost heuristic."
        ),
        "failure_mode": (
            "LLM-generated heuristics can encode game-specific leakage or "
            "overfit tiny held-out splits; .403 needs fresh held-out levels, "
            "static-analysis receipts, and reproduction gates."
        ),
        "experiment_mapping": (
            ".403: generalize the learned action-cost win into a bounded "
            "LLM-generated heuristic search over reproduced ARC levels."
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

STUDYING_SECTION = """## 2026-06-17 Exp 4354 - .402 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4354_sota_ingestion_v403.json`.

**Preconditions:** network precondition passed via Hugging Face reachability and
arXiv/WebFetch verification. If that check had failed, the only honest artifact
would have been `honest_verdict=blocked_network_unavailable`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` emitted the focused
energy/reward arXiv discovery URL. `scripts/sweep_semscholar.py` was run on the
five focused query strings and returned HTTP 429 for each query, so it produced
no usable arXiv IDs in this pass. Low-concurrency WebSearch/WebFetch plus arXiv
page checks verified arXiv:2602.01842, arXiv:2604.06260, arXiv:2606.08501,
arXiv:2605.05138, arXiv:2503.18809, arXiv:2512.24156, arXiv:2605.25931,
arXiv:2606.13565, and arXiv:2502.01384. The banned `/deep-research` channel was
not invoked.

**Filtered track:** .402 outcomes after S3 verifier-guided diffusion-LM search,
E3 deeper executable-world-model progression, and learned action-cost heuristic
action-efficiency.

**.402 outcome conditioning:**
- Exp 4348: `acceptance_gate=true`, `honest_verdict=controls_not_differentiable`,
  `benchmark_n=240`, and adversarial verification reports `TAUTOLOGY` across
  the S3-vs-control deltas. The S3 line remains alive, but .403 must harden the
  search and controls before claiming a clean generation gain.
- Exp 4351: `success_e3_deeper_tn36_reproduced`,
  `new_levels_reproduced=1`, `reproducible_total_levels=23`, and
  `verifier_is_oracle=true`. E3 has real ARC progress, but those solves remain
  oracle-grounded and should not be promoted as an oracle-free verifier moat.
- Exp 4353: `action_efficiency_improves=true`,
  `held_out_actions_baseline=25`, `held_out_actions_learned=16`,
  `positive_control_passed=true`, `reproduction_gated=true`, and
  `verifier_is_oracle=false`. The next self-learning step should generalize
  action-efficiency heuristics under reproduction gates.

**Fresh-pass candidates marked ingested:**
- Prism hierarchical trajectory search/self-verification, arXiv:2602.01842 -
  mapped to the .403 headline: harden S3 with hierarchical pruning, partial
  remasking, self-verified feedback, and explicit diversity/leakage receipts.
- S3 Stratified Scaling Search, arXiv:2604.06260 - carried forward as the base
  fixed-model verifier-guided denoising search, but only with differentiated
  controls after the .402 metric-tautology caution.
- PAPO reward-state alignment, arXiv:2606.08501 - mapped to authentic
  trajectory-state alignment diagnostics, not in-loop generator training.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138 - mapped to deeper
  private-like E3 progression with verifier_is_oracle=true kept explicit.
- Classical Planning with LLM-Generated Heuristics, arXiv:2503.18809 - mapped
  to a bounded program-heuristic generalization of the reproduced action-count
  win from Exp 4353.

**Screened but not mapped as strongest rows:** Graph-Based Exploration for
ARC-AGI-3 (arXiv:2512.24156) and AERA speed-depth trade-off
(arXiv:2605.25931) were verified and read as ARC exploration context. They
support the E3/action-efficiency direction, but the .402 outcomes point more
directly to executable-world-model continuation and learned/LLM-generated action
heuristics.

out_of_band_flagged:
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 -
  operator-owned verifier-as-reward generator training, NOT auto-run in-loop.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - operator-owned
  verifier-as-reward generator training, NOT auto-run in-loop.

flagged_for_v403: prism_hardened_s3_verifier_guided_search_v403

Flagged for .403: `prism_hardened_s3_verifier_guided_search_v403`

random_seed=4354

**Bottom line for the .403 roadmap:** keep the headline on non-training
verifier-guided diffusion-LM search, but do not repeat the .402 artifact shape.
Use Prism-style hierarchical trajectory search and partial-remasking controls to
make S3's lift auditable, keep PAPO as a state-alignment diagnostic, continue
E3 deeper progression as oracle-grounded ARC progress, generalize the learned
action-cost heuristic with reproduced action-count gates, and keep A2D2/SEPO
out of band for operator-owned generator training.
"""


def _adversarial_tautology_flagged(report: object) -> bool:
    if not isinstance(report, Mapping):
        return False
    text_values = [
        value for value in report.values() if isinstance(value, str)
    ]
    return any("TAUTOLOGY" in value for value in text_values)


def _tn36_new_level_reproduced(artifact: Mapping[str, Any]) -> bool:
    rows = artifact.get("per_target_scorecard")
    if not isinstance(rows, Sequence):
        return False
    for row in rows:
        if (
            isinstance(row, Mapping)
            and row.get("game") == "tn36"
            and row.get("offline_reproduced") is True
            and row.get("checkpoint_status") == "new_level_reproduced"
        ):
            return True
    return False


def extract_v402_outcomes(
    *,
    s3: Mapping[str, Any],
    e3: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, bool]:
    """Extract the load-bearing .402 outcome booleans from source artifacts."""

    benchmark_n = s3.get("benchmark_n")
    baseline_actions = action.get("held_out_actions_baseline")
    learned_actions = action.get("held_out_actions_learned")
    reproducible_total = e3.get("reproducible_total_levels")
    return {
        "s3_acceptance_gate": s3.get("acceptance_gate") is True,
        "s3_controls_not_differentiable": (
            s3.get("honest_verdict") == "controls_not_differentiable"
        ),
        "s3_benchmark_powered": isinstance(benchmark_n, int) and benchmark_n >= 200,
        "s3_adversarial_tautology_flagged": _adversarial_tautology_flagged(
            s3.get("adversarial_verify")
        ),
        "e3_deeper_tn36_reproduced": _tn36_new_level_reproduced(e3),
        "e3_new_levels_positive": (
            isinstance(e3.get("new_levels_reproduced"), int)
            and e3.get("new_levels_reproduced", 0) > 0
        ),
        "e3_reproducible_total_ge_22": (
            isinstance(reproducible_total, int) and reproducible_total >= 22
        ),
        "e3_verifier_is_oracle": e3.get("verifier_is_oracle") is True,
        "action_efficiency_improves": (
            action.get("action_efficiency_improves") is True
        ),
        "action_reduction_reproduced": (
            isinstance(baseline_actions, int)
            and isinstance(learned_actions, int)
            and learned_actions < baseline_actions
            and action.get("reproduction_gated") is True
        ),
        "action_positive_control_passed": (
            action.get("positive_control_passed") is True
        ),
        "action_verifier_non_oracle": action.get("verifier_is_oracle") is False,
    }


def select_flagged_for_v403(outcomes: Mapping[str, bool]) -> str:
    """Choose the .403 flag from the .402 fork outcomes."""

    if outcomes.get("s3_acceptance_gate") and outcomes.get(
        "s3_adversarial_tautology_flagged"
    ):
        return DEFAULT_FLAGGED_FOR_V403
    if outcomes.get("s3_acceptance_gate"):
        return S3_DIVERSITY_AUDIT_FLAGGED_FOR_V403
    if (
        outcomes.get("action_efficiency_improves")
        and outcomes.get("action_reduction_reproduced")
        and outcomes.get("action_verifier_non_oracle")
    ):
        return ACTION_HEURISTIC_FLAGGED_FOR_V403
    if outcomes.get("e3_new_levels_positive"):
        return E3_DEEPER_FLAGGED_FOR_V403
    return PAPO_DIAGNOSTIC_FLAGGED_FOR_V403


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v403: str = DEFAULT_FLAGGED_FOR_V403,
    out_of_band_flagged: Sequence[Mapping[str, str]] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4354 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v403": flagged_for_v403,
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
    """Validate the Exp 4354 artifact before it can be written to disk."""

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
        raise ValueError("field_principles must match REQ-REPORT-4354")

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

    flagged = artifact["flagged_for_v403"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v403 must be non-empty")
    if flagged not in ALLOWED_FLAGGED_FOR_V403:
        raise ValueError("flagged_for_v403 must be conditioned on the .402 outcomes")

    _validate_out_of_band(artifact["out_of_band_flagged"])


def validate_studying_section(section: str) -> None:
    """Check that the research-studying entry keeps citations and outcome context."""

    required_phrases = [
        "flagged_for_v403",
        "out_of_band_flagged",
        "network precondition passed",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "acceptance_gate=true",
        "honest_verdict=controls_not_differentiable",
        "benchmark_n=240",
        "TAUTOLOGY",
        "success_e3_deeper_tn36_reproduced",
        "new_levels_reproduced=1",
        "reproducible_total_levels=23",
        "verifier_is_oracle=true",
        "action_efficiency_improves=true",
        "held_out_actions_baseline=25",
        "held_out_actions_learned=16",
        "positive_control_passed=true",
        "reproduction_gated=true",
        "verifier_is_oracle=false",
        "operator-owned",
        "NOT auto-run",
        DEFAULT_FLAGGED_FOR_V403,
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
    marker = "## 2026-06-17 Exp 4354"
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

    flagged_for_v403 = select_flagged_for_v403(DEFAULT_V402_OUTCOMES)
    artifact = build_artifact(flagged_for_v403=flagged_for_v403)
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
    root_override = os.environ.get("CARNOT_EXP4354_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4354_sota_ingestion_v403.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Exp 4332 SOTA ingestion for the .400 fork outcomes feeding .401.

Spec refs: REQ-REPORT-4332, SCENARIO-REPORT-4332.

This module writes a planning artifact, not a benchmark result. It turns the
`.400` fork outcomes into a citation-gated SOTA-to-experiment map: the
second-corpus guided-generation replication failed its independent leak check,
adaptive guided generation was a bounded null, E3 produced a high-accuracy
partial world model but no reproduced solve, and the learned frame encoder did
not improve cross-game search. The .401 flag therefore moves to a leak-robust
diffusion-native partial-state reward model before any larger generation claim.
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
        "flagged_for_v401",
        "random_seed",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "track",
        "source_read",
        "v400_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v401_mapped"
DEFAULT_FLAGGED_FOR_V401 = "leak_robust_diffusion_native_partial_state_reward_v401"
SCALED_ARC_GRID_GENERATION_FLAGGED_FOR_V401 = "scaled_arc_grid_guided_generation_v401"
MULTI_GAME_E3_SWEEP_FLAGGED_FOR_V401 = "multi_game_e3_world_model_sweep_v401"
RICHER_ENCODER_MORE_GAMES_FLAGGED_FOR_V401 = "richer_encoder_more_games_value_transfer_v401"
REWARD_SCORE_MATCHING_ABLATION_FLAGGED_FOR_V401 = "reward_score_matching_guidance_ablation_v401"
AGENT2WORLD_E3_REPAIR_FLAGGED_FOR_V401 = "agent2world_adaptive_e3_repair_v401"
ALLOWED_FLAGGED_FOR_V401 = frozenset(
    {
        DEFAULT_FLAGGED_FOR_V401,
        SCALED_ARC_GRID_GENERATION_FLAGGED_FOR_V401,
        MULTI_GAME_E3_SWEEP_FLAGGED_FOR_V401,
        RICHER_ENCODER_MORE_GAMES_FLAGGED_FOR_V401,
        REWARD_SCORE_MATCHING_ABLATION_FLAGGED_FOR_V401,
        AGENT2WORLD_E3_REPAIR_FLAGGED_FOR_V401,
    }
)
DEFAULT_RANDOM_SEED = 4332

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real, VERIFIED arXiv ID/URL (no citation = "
        "fabrication per adversarial_verify discipline) + a one-line .401 "
        "experiment mapping."
    ),
    "flagged_for_v401": (
        "Closes discover->ingest->plan: names the strongest method for the .401 "
        "planner, conditioned on the .400 outcomes."
    ),
    "random_seed": (
        "Determinism placeholder for the discovery query set (recorded for "
        "reproducibility of the sweep)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2602.11146": "https://arxiv.org/abs/2602.11146",
    "2502.01384": "https://arxiv.org/abs/2502.01384",
    "2512.22336": "https://arxiv.org/abs/2512.22336",
    "2605.25931": "https://arxiv.org/abs/2605.25931",
    "2605.15256": "https://arxiv.org/abs/2605.15256",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS
)

DEFAULT_V400_OUTCOMES = {
    "in_generation_moat_replicates": False,
    "moat_benchmark_empty": True,
    "second_corpus_controls_differentiated": False,
    "scorer_leak_recheck_passed": False,
    "adaptive_guidance_beats_control": False,
    "adaptive_controls_differentiated": True,
    "adaptive_ci_excludes_zero": False,
    "offline_reproduced": False,
    "plan_executed": False,
    "reproduced_levels_positive": False,
    "e3_verifier_partial_model": True,
    "learned_encoder_transfer_helps": False,
    "baseline_solves_held_out": True,
    "cross_game_ci_lower_exceeds_one": False,
}

DEFAULT_METHODS_MAPPED = [
    {
        "name": "DiNa-LRM diffusion-native latent reward modeling",
        "arxiv_id_or_url": "2602.11146",
        "url": "https://arxiv.org/abs/2602.11146",
        "track": "leak-robust noisy-state reward modeling for diffusion generation",
        "source_read": (
            "DiNa-LRM trains a timestep-conditioned reward head directly on noisy "
            "diffusion states, with noise-calibrated uncertainty and inference-time "
            "noise ensembling for robust reward evaluation."
        ),
        "v400_outcome_conditioning": (
            "Exp 4325 returned honest_verdict=scorer_leaky_on_second_corpus with "
            "in_generation_moat_replicates=false and scorer_leak_recheck_passed=false."
        ),
        "carnot_stack_mapping": (
            "Replace the leaky Exp 4292 partial-state scorer with a reward head "
            "trained on masked/noisy intermediate states and answer-cell-masked "
            "validation before reusing it inside DiffusionGemma guidance."
        ),
        "failure_mode": (
            "The paper is image-latent diffusion, not ARC grid-token diffusion. "
            "The .401 port must prove that noisy-state rewards stay oracle-distinct "
            "under answer masking and do not recover final cells from position cues."
        ),
        "experiment_mapping": (
            ".401: build a DiNa-LRM-style leak-robust partial-state reward scorer "
            "and rerun the second-corpus moat gate before any scaled generation claim."
        ),
    },
    {
        "name": "SEPO score-entropy policy optimization",
        "arxiv_id_or_url": "2502.01384",
        "url": "https://arxiv.org/abs/2502.01384",
        "track": "policy-gradient fine-tuning of discrete diffusion with external rewards",
        "source_read": (
            "SEPO fine-tunes discrete diffusion models over non-differentiable "
            "rewards with a policy-gradient objective designed for score-entropy "
            "parameterizations."
        ),
        "v400_outcome_conditioning": (
            "Exp 4326 found adaptive_guidance_beats_control=false with CI95 "
            "[-0.075, 0.35], so .401 needs a real reward-optimization step rather "
            "than another schedule-only adaptive sampler."
        ),
        "carnot_stack_mapping": (
            "Use exact ARC or leak-robust verifier rewards as black-box terminal "
            "signals for a tiny DiffusionGemma adapter or replay head, with "
            "unguided and reward-score-matching controls."
        ),
        "failure_mode": (
            "Policy-gradient updates can optimize verifier artifacts and damage "
            "grid syntax. Keep KL-to-base constraints, syntax-validity gates, and "
            "held-out corpus leak checks load-bearing."
        ),
        "experiment_mapping": (
            ".401: run a bounded SEPO adapter over discrete denoising trajectories "
            "only after the new partial-state reward scorer passes leak audit."
        ),
    },
    {
        "name": "Agent2World adaptive world-model testing",
        "arxiv_id_or_url": "2512.22336",
        "url": "https://arxiv.org/abs/2512.22336",
        "track": "executable world-model induction and behavior-aware verification",
        "source_read": (
            "Agent2World generates symbolic world models and validates them with "
            "adaptive unit tests plus simulation-based feedback, producing "
            "multi-turn trajectories for later supervised improvement."
        ),
        "v400_outcome_conditioning": (
            "Exp 4327 reached verifier_best_accuracy=0.8875 but offline_reproduced=false "
            "and reproduced_levels=0, so the gap is behavior-level model coverage."
        ),
        "carnot_stack_mapping": (
            "Turn the E3 verifier into an adaptive test-team loop that searches "
            "for hidden transition mismatches before planning, then records the "
            "failed tests as training examples for future world-model induction."
        ),
        "failure_mode": (
            "Agent2World includes a web-research sub-agent in its original design; "
            "Carnot's in-loop port must keep discovery offline and only import the "
            "adaptive testing and simulation-validation pattern."
        ),
        "experiment_mapping": (
            ".401: repair the ar25 hidden-undo-stack rule gap with adaptive world-model "
            "tests before attempting a multi-game E3 sweep."
        ),
    },
    {
        "name": "AERA explore-verify-plan ARC-AGI-3 agent",
        "arxiv_id_or_url": "2605.25931",
        "url": "https://arxiv.org/abs/2605.25931",
        "track": "information-gain budgeting for interactive ARC-AGI-3 reasoning",
        "source_read": (
            "AERA separates EXPLORE, VERIFY, and PLAN phases and frames ARC-AGI-3 "
            "performance as a speed-depth trade-off between action efficiency and "
            "information gain."
        ),
        "v400_outcome_conditioning": (
            "Exp 4327 had plan_executed=false and no reproduced level, indicating "
            "that the current E3 loop planned before it had enough verified mechanics."
        ),
        "carnot_stack_mapping": (
            "Add an explicit exploration budget that collects verifier-targeted "
            "transition lemmas, then allow planning only after the world model "
            "passes those mechanics checks."
        ),
        "failure_mode": (
            "The paper also documents public-set shortcuts, so .401 must reject "
            "shortcut action schemas and report private/offline-env verifier outcomes "
            "rather than leaderboard-style public RHAE alone."
        ),
        "experiment_mapping": (
            ".401: make E3 explore-before-plan on ar25/ka59, with verifier-gated "
            "mechanic lemmas as the success precondition."
        ),
    },
    {
        "name": "ReactiveGWM game-agnostic interaction representation",
        "arxiv_id_or_url": "2605.15256",
        "url": "https://arxiv.org/abs/2605.15256",
        "track": "cross-game interaction logic representation for search values",
        "source_read": (
            "ReactiveGWM decouples player controls from NPC behavior and learns "
            "game-agnostic interaction modules that transfer zero-shot across "
            "game world models."
        ),
        "v400_outcome_conditioning": (
            "Exp 4331 reported learned_encoder_transfer_helps=false with "
            "cross_game_state_reduction_ci95 lower bound exactly 1.0 despite "
            "baseline_solves_held_out=true."
        ),
        "carnot_stack_mapping": (
            "Replace raw-frame-only value features with disentangled action-role "
            "and object-interaction embeddings, then train held-out search values "
            "on role transitions rather than per-game pixels."
        ),
        "failure_mode": (
            "ReactiveGWM targets video-game NPC world models, not symbolic ARC grids. "
            "The .401 representation must show state-count reduction under leave-one-game-out "
            "splits, not just prettier embeddings."
        ),
        "experiment_mapping": (
            ".401: run a richer cross-game value encoder over action-role interaction "
            "features after the tiny frame encoder's flat result."
        ),
    },
]

STUDYING_SECTION = """## 2026-06-17 Exp 4332 - .400 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4332_sota_ingestion_v401.json`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` and
`scripts/sweep_semscholar.py` imported successfully; `sweep_clusters.py`
emitted focused arXiv discovery URLs for verifier/reward and world-model
clusters. Semantic Scholar was reachable through the helper but returned HTTP
429 for the three focused keyword probes in this loop. Low-concurrency
WebSearch/WebFetch verified arXiv:2602.11146, arXiv:2502.01384,
arXiv:2512.22336, arXiv:2605.25931, arXiv:2605.15256, arXiv:2604.17415,
arXiv:2605.18548, arXiv:2606.00291, arXiv:2605.26491, and arXiv:2510.23691.
The banned `/deep-research` channel was not invoked.

**Filtered track:** .400 outcomes after second-corpus guided-generation
replication, adaptive guided-generation scale-up, E3 executable-world-model
induction on ar25, and learned frame-encoder cross-game value transfer.

**.400 outcome conditioning:**
- Exp 4325: `honest_verdict=scorer_leaky_on_second_corpus`,
  `in_generation_moat_replicates=false`, `controls_differentiated=false`,
  `scorer_leak_recheck_passed=false`, `benchmark_n=0`,
  `carnot_minus_best_control_delta=0.0`, and `replication_ci95=[0.0, 0.0]`;
  the first in-generation moat did not replicate because the second-corpus
  scorer failed the independent leak recheck.
- Exp 4326: `adaptive_guidance_beats_control=false`,
  `adaptive_ci95=[-0.075, 0.35]`, `adaptive_controls_differentiated=true`, and
  `adaptive_benchmark_n=40`; adaptive guidance differentiated controls but did
  not beat the engaged control.
- Exp 4327: `offline_reproduced=false`, `plan_executed=false`,
  `reproduced_levels=0`, `verifier_best_accuracy=0.8875`, and
  `residual_mismatch_class=missing_world_model_rule_gap_hidden_undo_stack_action7`;
  E3 made a useful partial world model but no reproduced solve.
- Exp 4331: `learned_encoder_transfer_helps=false`,
  `cross_game_state_reduction=1.0084925690021231`,
  `cross_game_state_reduction_ci95=[1.0, 1.0303068758652514]`, and
  `baseline_solves_held_out=true`; the positive-control solver worked, but the
  learned frame encoder still did not reduce held-out search states.

**Fresh-pass candidates marked ingested:**
- DiNa-LRM diffusion-native latent reward modeling, arXiv:2602.11146 - mapped to
  a leak-robust partial-state reward scorer before any scaled generation claim.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - mapped to bounded
  discrete-diffusion reward optimization after the adaptive schedule-only null.
- Agent2World adaptive world-model testing, arXiv:2512.22336 - mapped to
  behavior-aware E3 verifier tests for hidden transition gaps.
- AERA explore-verify-plan ARC-AGI-3 agent, arXiv:2605.25931 - mapped to an
  explicit information-gain budget before E3 planning.
- ReactiveGWM game-agnostic interaction representation, arXiv:2605.15256 -
  mapped to richer cross-game value features after the tiny frame encoder stayed flat.

**Screened but not mapped as strongest rows:** Reward Score Matching
(arXiv:2604.17415), STT-Arena (arXiv:2605.18548), Representation-Rationalizability
(arXiv:2606.00291), Diffusion LAIR (arXiv:2605.26491), and Game-TARS
(arXiv:2510.23691) were read as relevant context. They were not selected as
strongest rows because the observed .400 failures point more directly to
leak-robust noisy-state rewards, adaptive world-model tests, explore-before-plan
discipline, and game-invariant interaction features.

Already-covered context not re-ingested as fresh method rows: A2D2
(arXiv:2606.13565), TR2-D2 (arXiv:2509.25171), Reward-State Alignment
(arXiv:2606.08501), diffusion step selection (arXiv:2603.12554), Executable
World Models for ARC-AGI-3 (arXiv:2605.05138), and Graph-Based Exploration for
ARC-AGI-3 (arXiv:2512.24156).

flagged_for_v401:
`leak_robust_diffusion_native_partial_state_reward_v401`.

Flagged for .401: `leak_robust_diffusion_native_partial_state_reward_v401`.

random_seed=4332

**Bottom line for the .401 roadmap:** do not scale the Exp 4315 guided-generation
claim yet. The second-corpus leak check failed and the adaptive run was a bounded
null, so the strongest .401 entry is leak-robust diffusion-native partial-state reward
scoring. Keep E3 on adaptive testing plus explore-before-plan repair, and
retry cross-game value transfer only with richer game-invariant interaction
features.
"""


def _ci_excludes_zero(values: object) -> bool:
    if isinstance(values, Sequence) and len(values) == 2:
        lower, upper = values
        if isinstance(lower, int | float) and isinstance(upper, int | float):
            return lower > 0 or upper < 0
    return False


def _ci_lower_exceeds_one(values: object) -> bool:
    if isinstance(values, Sequence) and len(values) == 2:
        lower = values[0]
        return isinstance(lower, int | float) and lower > 1.0
    return False


def extract_v400_outcomes(
    *,
    moat: Mapping[str, Any],
    adaptive: Mapping[str, Any],
    e3: Mapping[str, Any],
    transfer: Mapping[str, Any],
) -> dict[str, bool]:
    """Extract the load-bearing .400 outcome booleans from source artifacts."""

    reproduced_levels = e3.get("reproduced_levels")
    verifier_best_accuracy = e3.get("verifier_best_accuracy")
    return {
        "in_generation_moat_replicates": (
            moat.get("in_generation_moat_replicates") is True
        ),
        "moat_benchmark_empty": moat.get("benchmark_n") == 0,
        "second_corpus_controls_differentiated": (
            moat.get("controls_differentiated") is True
        ),
        "scorer_leak_recheck_passed": moat.get("scorer_leak_recheck_passed") is True,
        "adaptive_guidance_beats_control": (
            adaptive.get("adaptive_guidance_beats_control") is True
        ),
        "adaptive_controls_differentiated": (
            adaptive.get("controls_differentiated") is True
        ),
        "adaptive_ci_excludes_zero": _ci_excludes_zero(adaptive.get("adaptive_ci95")),
        "offline_reproduced": e3.get("offline_reproduced") is True,
        "plan_executed": e3.get("plan_executed") is True,
        "reproduced_levels_positive": (
            isinstance(reproduced_levels, int) and reproduced_levels > 0
        ),
        "e3_verifier_partial_model": (
            e3.get("offline_reproduced") is False
            and isinstance(verifier_best_accuracy, int | float)
            and verifier_best_accuracy >= 0.8
        ),
        "learned_encoder_transfer_helps": (
            transfer.get("learned_encoder_transfer_helps") is True
        ),
        "baseline_solves_held_out": transfer.get("baseline_solves_held_out") is True,
        "cross_game_ci_lower_exceeds_one": _ci_lower_exceeds_one(
            transfer.get("cross_game_state_reduction_ci95")
        ),
    }


def select_flagged_for_v401(outcomes: Mapping[str, bool]) -> str:
    """Choose the .401 flag from the .400 fork outcomes."""

    if outcomes.get("in_generation_moat_replicates") and outcomes.get(
        "adaptive_guidance_beats_control"
    ):
        return SCALED_ARC_GRID_GENERATION_FLAGGED_FOR_V401
    if outcomes.get("offline_reproduced") and outcomes.get("reproduced_levels_positive"):
        return MULTI_GAME_E3_SWEEP_FLAGGED_FOR_V401
    if outcomes.get("learned_encoder_transfer_helps") and outcomes.get(
        "baseline_solves_held_out"
    ):
        return RICHER_ENCODER_MORE_GAMES_FLAGGED_FOR_V401
    if outcomes.get("scorer_leak_recheck_passed") is False:
        return DEFAULT_FLAGGED_FOR_V401
    if outcomes.get("adaptive_guidance_beats_control"):
        return REWARD_SCORE_MATCHING_ABLATION_FLAGGED_FOR_V401
    return AGENT2WORLD_E3_REPAIR_FLAGGED_FOR_V401


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v401: str = DEFAULT_FLAGGED_FOR_V401,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4332 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v401": flagged_for_v401,
        "random_seed": random_seed,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4332 artifact before it can be written to disk."""

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
        raise ValueError("field_principles must match REQ-REPORT-4332")

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

    flagged = artifact["flagged_for_v401"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v401 must be non-empty")
    if flagged not in ALLOWED_FLAGGED_FOR_V401:
        raise ValueError("flagged_for_v401 must be conditioned on the .400 outcomes")


def validate_studying_section(section: str) -> None:
    """Check that the research-studying entry keeps citations and outcome context."""

    required_phrases = [
        "flagged_for_v401",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "in_generation_moat_replicates=false",
        "controls_differentiated=false",
        "scorer_leak_recheck_passed=false",
        "benchmark_n=0",
        "carnot_minus_best_control_delta=0.0",
        "replication_ci95=[0.0, 0.0]",
        "adaptive_guidance_beats_control=false",
        "adaptive_ci95=[-0.075, 0.35]",
        "adaptive_controls_differentiated=true",
        "adaptive_benchmark_n=40",
        "offline_reproduced=false",
        "plan_executed=false",
        "reproduced_levels=0",
        "verifier_best_accuracy=0.8875",
        "residual_mismatch_class=missing_world_model_rule_gap_hidden_undo_stack_action7",
        "learned_encoder_transfer_helps=false",
        "cross_game_state_reduction=1.0084925690021231",
        "cross_game_state_reduction_ci95=[1.0, 1.0303068758652514]",
        "baseline_solves_held_out=true",
        DEFAULT_FLAGGED_FOR_V401,
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


def _with_studying_section(existing: str) -> str:
    marker = "## 2026-06-17 Exp 4332"
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

    flagged_for_v401 = select_flagged_for_v401(DEFAULT_V400_OUTCOMES)
    artifact = build_artifact(flagged_for_v401=flagged_for_v401)
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
    root_override = os.environ.get("CARNOT_EXP4332_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4332_sota_ingestion_v401.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

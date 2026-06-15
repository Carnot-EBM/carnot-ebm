"""Exp 4265 SOTA ingestion for the .394 fork outcomes feeding .395.

Spec refs: REQ-REPORT-4265, SCENARIO-REPORT-4265.

This module writes a planning artifact, not a benchmark result. It closes the
`.394` fork outcomes into a concrete SOTA-to-experiment mapping: the ARC
selector win survived provenance and multi-seed hardening, but synthesis failed
to beat selector-only, cross-game transfer was blocked by missing game IDs, and
DiffusionGemma remained loader-blocked. That combination points .395 toward a
controlled ARC task-family and cross-game generalization setup before another
expensive synthesis or diffusion full run.
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
        "flagged_for_v395",
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
        "v394_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v395_mapped"
DEFAULT_FLAGGED_FOR_V395 = "arc_tgi_family_generator_cross_game_generalization_v395"
DEFAULT_RANDOM_SEED = 4265

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL (no citation = fabrication "
        "per adversarial_verify discipline) + a one-line .395 experiment mapping."
    ),
    "flagged_for_v395": (
        "Closes discover->ingest->plan: names the strongest method for the .395 "
        "planner, conditioned on the .394 outcomes."
    ),
    "random_seed": (
        "Determinism placeholder for the discovery query set (recorded for "
        "reproducibility of the sweep)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2603.05099": "https://arxiv.org/abs/2603.05099",
    "2606.03305": "https://arxiv.org/abs/2606.03305",
    "2604.24357": "https://arxiv.org/abs/2604.24357",
    "2603.12554": "https://arxiv.org/abs/2603.12554",
    "2606.12847": "https://arxiv.org/abs/2606.12847",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "ARC-TGI task-family generators for cross-game generalization",
        "arxiv_id_or_url": "2603.05099",
        "url": "https://arxiv.org/abs/2603.05099",
        "track": "cross-game ARC transfer",
        "source_read": (
            "ARC-TGI releases human-validated task-family generators with "
            "reasoning-chain templates and task-level constraints for controlled "
            "ARC sampling."
        ),
        "v394_outcome_conditioning": (
            "Exp 4258 blocked because game IDs were unrecoverable, while Exp "
            "4256 and Exp 4257 hardened the within-pool selector win. The next "
            "question is therefore controlled game-disjoint generalization, not "
            "another within-pool rerank."
        ),
        "carnot_stack_mapping": (
            "Materialize generator/family IDs beside each ARC candidate pool row, "
            "then build train-family/test-family splits with exact target hashes "
            "and the existing Set-Encoder scoring path."
        ),
        "failure_mode": (
            "Generator-generated data can become its own distribution. The .395 "
            "gate must keep original ARC tasks as a held-out sanity read and "
            "report family-disjoint and original-task metrics separately."
        ),
        "experiment_mapping": (
            ".395: recover or attach task-family IDs, build an ARC-TGI-style "
            "family-disjoint candidate pool, and rerun Set-Encoder versus vote "
            "on held-out families."
        ),
    },
    {
        "name": "Reliability Gap benchmark-auditing provenance discipline",
        "arxiv_id_or_url": "2606.03305",
        "url": "https://arxiv.org/abs/2606.03305",
        "track": "learned-verifier leak and contamination auditing",
        "source_read": (
            "The paper finds contamination detectors fail under distribution "
            "shift and small benchmark scale, and cannot replace transparent "
            "data provenance for benchmark-level auditing."
        ),
        "v394_outcome_conditioning": (
            "Exp 4256 survived a provenance-blind feature audit, but the origin "
            "probe was high. .395 should preserve explicit source manifests "
            "instead of relying on statistical leak detectors after the fact."
        ),
        "carnot_stack_mapping": (
            "Add source-kind, generator-family, fold, and target-hash manifests "
            "as first-class pool columns before training, then require every "
            "reported split to cite those manifests."
        ),
        "failure_mode": (
            "A detector-only clean read can be underpowered or shifted. Treat "
            "detector scores as diagnostics and make transparent provenance the "
            "acceptance gate."
        ),
        "experiment_mapping": (
            ".395: pre-register a provenance manifest and reject any selector, "
            "synthesis, or cross-game claim whose rows cannot be traced to a "
            "source-kind and family split."
        ),
    },
    {
        "name": "DPRM token-ordering guidance for diffusion language models",
        "arxiv_id_or_url": "2604.24357",
        "url": "https://arxiv.org/abs/2604.24357",
        "track": "discrete-diffusion verifier and process-reward guidance",
        "source_read": (
            "DPRM is a plug-in Doob h-transform process-reward module that keeps "
            "the host diffusion LM fixed and changes token ordering through "
            "reward-tilted reveal decisions."
        ),
        "v394_outcome_conditioning": (
            "Exp 4260 blocked at the DiffusionGemma GGUF loader, so DPRM is not "
            "a full-run recommendation yet. It is the right method to queue after "
            "loader repair because it uses verifier/process rewards without "
            "changing the base denoiser."
        ),
        "carnot_stack_mapping": (
            "Map the Set-Encoder or process verifier score into a token/cell "
            "ordering policy for a loader-fixed masked diffusion run, with a "
            "tiny smoke before any full ARC budget."
        ),
        "failure_mode": (
            "Ordering guidance can optimize partial reveal confidence rather "
            "than exact ARC grid correctness. Exact-match validation and a "
            "selector-only baseline remain mandatory."
        ),
        "experiment_mapping": (
            ".395 only after loader repair: tiny DPRM-style guided reveal smoke "
            "using verifier rewards, then GO/NO-GO for a full DiffusionGemma run."
        ),
    },
    {
        "name": "Entropy-guided step selection and stepwise advantages for diffusion LLM RL",
        "arxiv_id_or_url": "2603.12554",
        "url": "https://arxiv.org/abs/2603.12554",
        "track": "discrete-diffusion training and guidance",
        "source_read": (
            "This method formulates diffusion sequence generation as a denoising "
            "MDP with exact stepwise policy gradients and entropy-guided step "
            "selection for efficient updates."
        ),
        "v394_outcome_conditioning": (
            "Because Exp 4260 never loaded DiffusionGemma and Exp 4259 showed "
            "deterministic synthesis underperformed, this belongs behind a "
            "loader-repair and smoke gate, not ahead of the cross-game fix."
        ),
        "carnot_stack_mapping": (
            "If a diffusion LM loads, use verifier rewards as intermediate "
            "advantages over denoising steps instead of final-output-only reward "
            "or monolithic live LoRA training."
        ),
        "failure_mode": (
            "The method is a post-training recipe. It will not help while the "
            "DiffusionGemma loader is blocked, and sparse ARC exact-match rewards "
            "may still be too coarse without per-cell evidence."
        ),
        "experiment_mapping": (
            ".395 deferred path: after DiffusionGemma loader repair, run a "
            "bounded entropy-step smoke with exact-grid rewards and compare to "
            "unguided diffusion."
        ),
    },
    {
        "name": "L-VARC language-guided abstraction with inference-time visual backbone",
        "arxiv_id_or_url": "2606.12847",
        "url": "https://arxiv.org/abs/2606.12847",
        "track": "cross-task ARC abstraction transfer",
        "source_read": (
            "L-VARC trains a lightweight visual ARC model with a language-guided "
            "privileged-information branch, then discards that branch at "
            "inference."
        ),
        "v394_outcome_conditioning": (
            "Exp 4259 says naive grid synthesis did not beat selection. L-VARC "
            "suggests using language-derived abstractions as training-only "
            "structure for generalization, while preserving non-language "
            "inference-time claims."
        ),
        "carnot_stack_mapping": (
            "Attach task-family or reasoning-template embeddings during training "
            "of an ARC selector/visual model, then evaluate with those privileged "
            "signals removed on held-out families."
        ),
        "failure_mode": (
            "Language descriptions can leak hidden rules. The branch must be "
            "training-only, family-disjoint, and audited against target leakage "
            "before it can support a generalization claim."
        ),
        "experiment_mapping": (
            ".395 ablation: train with and without language/template privileged "
            "features on ARC-TGI families, discard them at inference, and score "
            "held-out family exact matches."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-15: .394 forks map for .395

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_v395_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `ARC-TGI task-family generators for cross-game generalization`, arxiv_id_or_url: `2603.05099`, url: `https://arxiv.org/abs/2603.05099`}
  - {name: `Reliability Gap benchmark-auditing provenance discipline`, arxiv_id_or_url: `2606.03305`, url: `https://arxiv.org/abs/2606.03305`}
  - {name: `DPRM token-ordering guidance for diffusion language models`, arxiv_id_or_url: `2604.24357`, url: `https://arxiv.org/abs/2604.24357`}
  - {name: `Entropy-guided step selection and stepwise advantages for diffusion LLM RL`, arxiv_id_or_url: `2603.12554`, url: `https://arxiv.org/abs/2603.12554`}
  - {name: `L-VARC language-guided abstraction with inference-time visual backbone`, arxiv_id_or_url: `2606.12847`, url: `https://arxiv.org/abs/2606.12847`}
  - principle: Each method MUST carry a real arXiv ID/URL (no citation = fabrication per adversarial_verify discipline) + a one-line .395 experiment mapping.
- flagged_for_v395: `arc_tgi_family_generator_cross_game_generalization_v395`
  - principle: Closes discover->ingest->plan: names the strongest method for the .395 planner, conditioned on the .394 outcomes.
- random_seed: `4265`
  - principle: Determinism placeholder for the discovery query set (recorded for reproducibility of the sweep).

## Fresh-pass provenance

Read `CLAUDE.md` SOTA-Ingestion Cycle Discipline, `research-studying.md`,
`research-references.md`, `results/experiment_4251_sota_ingestion_set_encoder_offline_rft.json`,
the .394 fork artifacts Exp 4256 through Exp 4264, `scripts/sweep_clusters.py`,
and `scripts/sweep_semscholar.py`.

Reliable-channel helper pass, not `/deep-research`:
- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_semscholar.py "Compute as Teacher generative synthesis selection robustness best of n aggregation" --limit 8`
- `python3 scripts/sweep_semscholar.py "discrete diffusion classifier guidance verifier energy guided language model EDLM" --limit 8`
- `python3 scripts/sweep_semscholar.py "benchmark contamination detection data leakage auditing learned verifier membership inference" --limit 8`
- `python3 scripts/sweep_semscholar.py "ARC abstraction reasoning cross task transfer cross game generalization" --limit 8`

The sweep helpers imported successfully. The cluster helper emitted the
broadened arXiv API URLs for the reliable channel. Semantic Scholar returned
HTTP 429 for the four focused queries, so no Semantic-Scholar-only promotion is
claimed. WebSearch/WebFetch was reachable and verified arXiv:2603.05099,
arXiv:2606.03305, arXiv:2604.24357, arXiv:2603.12554, and arXiv:2606.12847.
The banned `/deep-research` channel was not invoked.

## Prior-covered methods not re-ingested

The .392/.393/.394 sweeps already covered Compute-as-Teacher (arXiv:2509.14234),
GSA / LLMs Can Generate a Better Answer by Aggregating Their Own Responses
(arXiv:2503.04104), GenSelect-BoN (arXiv:2602.02143), Reward-Guided Stitching
(arXiv:2602.22871), S3 (arXiv:2604.06260), EDLM (arXiv:2410.21357),
Unlocking Guidance for Discrete State-Space Diffusion and Flow Models
(arXiv:2406.01572), CoDeC (arXiv:2510.27055), ARC of Progress
(arXiv:2603.13372), ARCTraj (arXiv:2511.11079), and Compositional
Neuro-Symbolic Reasoning (arXiv:2604.02434). They remain context, but they are
not counted as fresh `methods_mapped` rows here.

## .394 fork outcome read

Exp 4256: `arc_provenance_blind_win_survives`,
`provenance_blind_delta=0.3846153846`, `win_survives_provenance_blind=true`.
The leak audit hardened the selector win, but the high origin probe means .395
must keep transparent provenance rather than trusting detector-only audits.

Exp 4257: `arc_oracle_distinct_win_replicates_multiseed`,
`mean_delta=0.4576923077`, `cross_seed_ci95=[0.4377176136, 0.4776670017]`,
and `oracle_distinct_win_replicates=true`. The within-pool selector win is now
robust enough to test transfer.

Exp 4258: `blocked_arc_game_ids_unrecoverable`, so cross-game transfer was not
measured. This is a data/partition blocker, not evidence of generalization or
collapse.

Exp 4259: `arc_synthesis_underperforms_selection`,
`synthesis_breaks_oracle_ceiling=false`, `synthesis_minus_oracle_delta=-0.2826086957`,
`synthesis_beats_selection=false`, and `exact_match_validated=true`. The
selection win should not be escalated into a generative synthesis headline.

Exp 4260: `blocked_diffusiongemma_gguf_loader_failed` and `preflight_go=false`.
DiffusionGemma remains a loader-repair path, not a .395 full-run bet.

Exp 4264: `code_oracle_distinct_replication_corpus_specific`,
`code_replication_beats_vote=false`, and `code_predictor_minus_vote_delta=-0.00625`.
The code read does not replicate the ARC moat; it supports keeping .395 focused
on ARC transfer and provenance.

## SOTA -> experiment mapping

## ARC-TGI: recover the blocked cross-game axis

**Method/source:** ARC-TGI, arXiv:2603.05099
(https://arxiv.org/abs/2603.05099), provides human-validated ARC task-family
generators with reasoning-chain templates and task-level constraints.

**Carnot stack mapping:** Persist generator or task-family IDs next to each ARC
candidate row, use them to create family-disjoint train/test splits, and score
the existing Set-Encoder against vote on held-out families.

**.394 conditioning:** Exp 4256 and Exp 4257 hardened the within-pool win, but
Exp 4258 blocked the real OOD test because game IDs were unrecoverable. ARC-TGI
directly fixes that missing split variable.

**Failure mode:** Generator data can overfit to generator artifacts. Keep
original ARC held-out tasks as a sanity read and separate generated-family
metrics from original-task metrics.

**Experiment mapping:** .395 builds an ARC-TGI-style family-disjoint candidate
pool and reruns Set-Encoder versus vote on held-out families.

## Reliability Gap: provenance before detector-only leak audits

**Method/source:** Reliability Gap in Benchmark Auditing, arXiv:2606.03305
(https://arxiv.org/abs/2606.03305), shows contamination detectors can fail
under distribution shift and small benchmark scale.

**Carnot stack mapping:** Make source-kind, generator-family, fold, and target
hashes first-class manifest columns before training or evaluation.

**.394 conditioning:** Exp 4256 survived provenance-blind scoring, but the
origin probe was high. The next milestone should rely on transparent row
provenance and family splits, not post-hoc detector confidence.

**Failure mode:** Statistical leak detectors can be underpowered. Treat them as
diagnostics, not acceptance gates.

**Experiment mapping:** .395 rejects any selector, synthesis, or transfer claim
whose rows cannot be traced to a source-kind and family manifest.

## DPRM: queue diffusion guidance behind loader repair

**Method/source:** DPRM, arXiv:2604.24357
(https://arxiv.org/abs/2604.24357), uses a Doob h-transform process-reward
module to guide token ordering in diffusion language models without changing the
host denoiser.

**Carnot stack mapping:** Once DiffusionGemma loads, map verifier rewards into
token or cell reveal ordering for a tiny guided denoising smoke.

**.394 conditioning:** Exp 4260 blocked before guidance could run, so DPRM is
not a full-run recommendation yet.

**Failure mode:** Ordering guidance may optimize confidence rather than exact
grid correctness. Exact ARC match and selector-only controls remain required.

**Experiment mapping:** .395 only after loader repair: run a tiny DPRM-style
guided reveal smoke, then decide whether a full DiffusionGemma run is warranted.

## Entropy-guided diffusion RL: stepwise rewards, not final-only guesses

**Method/source:** Reinforcement Learning for Diffusion LLMs with
Entropy-Guided Step Selection and Stepwise Advantages, arXiv:2603.12554
(https://arxiv.org/abs/2603.12554), derives stepwise policy-gradient updates
over denoising trajectories.

**Carnot stack mapping:** If DiffusionGemma loads, use verifier rewards as
intermediate denoising advantages rather than final-output-only reward.

**.394 conditioning:** Loader failure and synthesis underperformance put this
behind the cross-game/provenance work.

**Failure mode:** Sparse ARC exact-match reward may still be too coarse without
per-cell evidence.

**Experiment mapping:** .395 deferred path: bounded entropy-step smoke after
DiffusionGemma loader repair, compared to unguided diffusion.

## L-VARC: training-only semantic abstraction

**Method/source:** L-VARC, arXiv:2606.12847
(https://arxiv.org/abs/2606.12847), trains a lightweight ARC visual model with
a language-guided privileged-information branch that is discarded at inference.

**Carnot stack mapping:** Attach reasoning-template or family embeddings during
training, remove them at inference, and evaluate exact matches on held-out
families.

**.394 conditioning:** Exp 4259 says naive synthesis did not beat selection, so
semantic abstraction should be a transfer scaffold rather than a generation
headline.

**Failure mode:** Language descriptions can leak hidden rules. The branch must
be training-only and family-disjoint.

**Experiment mapping:** .395 ablates template-privileged training on ARC-TGI
families with privileged features removed at test time.

## Flagged for .395

`arc_tgi_family_generator_cross_game_generalization_v395` is the strongest next
method. The reason is conditional on the .394 outcomes: the ARC selector win
survived leak and multi-seed hardening, but synthesis did not break the
selection ceiling, cross-game transfer was blocked by missing game IDs,
DiffusionGemma preflight did not load, and code replication was corpus-specific.
Therefore .395 should repair the missing transfer substrate first: build a
transparent provenance manifest plus ARC-TGI-style family-disjoint candidate
pool, then rerun Set-Encoder versus vote on held-out task families. Keep
DiffusionGemma as loader repair, not a full-run .395 bet.

random_seed=4265
"""

STUDYING_SECTION = """## 2026-06-15 Exp 4265 - .394 fork SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-v395-2026-06-15.md`.

**Filtered track:** .394 ARC oracle-distinct forks after the selector win
survived provenance-blind and multi-seed hardening, while cross-game transfer
was blocked, synthesis underperformed selection, DiffusionGemma preflight was
loader-blocked, and code replication read corpus-specific.

**Fresh-pass candidates marked ingested:**
- ARC-TGI, arXiv:2603.05099 - mapped as the strongest .395 method: recover the
  missing task-family/game-disjoint transfer substrate.
- Reliability Gap in Benchmark Auditing, arXiv:2606.03305 - mapped to
  provenance-first leak discipline after the high-origin-probe but surviving
  provenance-blind audit.
- DPRM, arXiv:2604.24357 - mapped to verifier/process-reward token ordering
  only after DiffusionGemma loader repair.
- Entropy-guided step selection for diffusion LLM RL, arXiv:2603.12554 - mapped
  to a deferred denoising-step reward smoke after loader repair.
- L-VARC, arXiv:2606.12847 - mapped to training-only semantic abstraction over
  ARC-TGI families, with privileged features removed at inference.

Already-covered context not re-ingested as fresh method rows: Compute-as-Teacher,
GSA, GenSelect-BoN, Reward-Guided Stitching, S3, EDLM, arXiv:2406.01572
discrete guidance, CoDeC, ARC of Progress, ARCTraj, and Compositional
Neuro-Symbolic Reasoning.

.394 status mapped honestly: Exp 4256 `arc_provenance_blind_win_survives` with
`provenance_blind_delta=0.3846153846`; Exp 4257
`arc_oracle_distinct_win_replicates_multiseed` with `mean_delta=0.4576923077`;
Exp 4258 `blocked_arc_game_ids_unrecoverable`; Exp 4259
`arc_synthesis_underperforms_selection` with `synthesis_breaks_oracle_ceiling=false`
and `synthesis_minus_oracle_delta=-0.2826086957`; Exp 4260
`blocked_diffusiongemma_gguf_loader_failed` with `preflight_go=false`; Exp 4264
`code_oracle_distinct_replication_corpus_specific` with
`code_replication_beats_vote=false`.

flagged_for_v395:
`arc_tgi_family_generator_cross_game_generalization_v395`.

Flagged for .395: `arc_tgi_family_generator_cross_game_generalization_v395`.

**Bottom line for the .395 roadmap:** do not spend .395 on full DiffusionGemma
or another synthesis headline yet. First repair the transfer substrate with a
provenance manifest and ARC-TGI-style family-disjoint candidate pool, then test
whether the hardened Set-Encoder win survives held-out task families. Keep DiffusionGemma as loader repair, not a full-run .395 bet.
"""


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v395: str = DEFAULT_FLAGGED_FOR_V395,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4265 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v395": flagged_for_v395,
        "random_seed": random_seed,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4265 artifact before it can be written to disk."""

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
        raise ValueError("field_principles must match REQ-REPORT-4265")

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

    flagged = artifact["flagged_for_v395"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v395 must be non-empty")
    if "arc_tgi" not in flagged.lower():
        raise ValueError("flagged_for_v395 must name the ARC-TGI cross-game method")


def validate_markdown_note(note: str) -> None:
    """Check that the human-readable note keeps citations and outcome context."""

    required_phrases = [
        "Flagged for .395",
        "Prior-covered methods not re-ingested",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "arc_provenance_blind_win_survives",
        "provenance_blind_delta=0.3846153846",
        "arc_oracle_distinct_win_replicates_multiseed",
        "mean_delta=0.4576923077",
        "blocked_arc_game_ids_unrecoverable",
        "arc_synthesis_underperforms_selection",
        "synthesis_breaks_oracle_ceiling=false",
        "synthesis_minus_oracle_delta=-0.2826086957",
        "blocked_diffusiongemma_gguf_loader_failed",
        "preflight_go=false",
        "code_oracle_distinct_replication_corpus_specific",
        "code_replication_beats_vote=false",
        DEFAULT_FLAGGED_FOR_V395,
        f"random_seed={DEFAULT_RANDOM_SEED}",
    ]
    for phrase in required_phrases:
        if phrase not in note:
            raise ValueError(f"markdown note missing required phrase: {phrase}")

    missing_sources = sorted(
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in note
    )
    if missing_sources:
        raise ValueError(f"markdown note missing verified source citations: {missing_sources}")


def _with_studying_section(existing: str) -> str:
    marker = "## 2026-06-15 Exp 4265"
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
    note_path: Path,
    artifact_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the note, JSON artifact, and idempotent research-studying entry."""

    artifact = build_artifact()
    validate_artifact(artifact)
    validate_markdown_note(NOTE_MARKDOWN)

    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(NOTE_MARKDOWN, encoding="utf-8")

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    existing_studying = studying_path.read_text(encoding="utf-8") if studying_path.exists() else ""
    studying_path.write_text(_with_studying_section(existing_studying), encoding="utf-8")

    return artifact


def main() -> int:
    root_override = os.environ.get("CARNOT_EXP4265_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        note_path=repo_root / "docs/research-notes/sota-ingestion-v395-2026-06-15.md",
        artifact_path=repo_root / "results/experiment_4265_sota_ingestion_v395.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

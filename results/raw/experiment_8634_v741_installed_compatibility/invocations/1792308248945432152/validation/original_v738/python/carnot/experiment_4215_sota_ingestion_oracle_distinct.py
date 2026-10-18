"""Exp 4215 SOTA ingestion for oracle-distinct verifier planning.

Spec refs: REQ-REPORT-4215, SCENARIO-REPORT-4215.

This module writes a planning artifact, not a benchmark result. It ingests the
`.390 planning sweep` and maps the oracle-distinct learned-verifier literature
onto the `.391` experiment plan while preserving that Exp 4210 was blocked
upstream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "honest_verdict",
        "methods_mapped",
        "flagged_for_v391",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "carnot_stack_mapping",
        "implication",
        "failure_mode",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_oracle_distinct_mapped_v391"
DEFAULT_FLAGGED_FOR_V391 = "arbiter_conservative_override_arc_wrong_majority_v391"

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL; an ingestion note without "
        "verifiable citations is treated as fabrication (adversarial_verify "
        "discipline)."
    ),
    "flagged_for_v391": (
        "Closes discover->ingest->plan: names the strongest method for the next "
        "planner (e.g. ARBITER conservative-override, or a learned-ARC-energy "
        "distill)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2605.26172": "https://arxiv.org/abs/2605.26172",
    "2512.15146": "https://arxiv.org/abs/2512.15146",
    "2504.16828": "https://arxiv.org/abs/2504.16828",
    "2510.08049": "https://arxiv.org/abs/2510.08049",
    "2402.06457": "https://arxiv.org/abs/2402.06457",
    "2509.19681": "https://arxiv.org/abs/2509.19681",
    "2603.11226": "https://arxiv.org/abs/2603.11226",
    "2604.00442": "https://arxiv.org/abs/2604.00442",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "ARBITER conservative override for wrong-majority recovery",
        "arxiv_id_or_url": "2605.26172",
        "url": "https://arxiv.org/abs/2605.26172",
        "carnot_stack_mapping": (
            "Maps directly to the A2/A3 oracle-distinct ARC verifier: majority "
            "vote is treated as a prior over basins, and the learned verifier "
            "overrides only when its margin is high."
        ),
        "implication": (
            "The next A3-style measurement should stratify wrong-majority cases "
            "where oracle@K exceeds vote@1 and ask whether a conservative "
            "learned-margin override recovers the outvoted correct answer."
        ),
        "failure_mode": (
            "ARBITER uses same-pool evidence and does not by itself train an ARC "
            "energy verifier; without an A2 selector and a held-out ARC split, "
            "it is a design pattern rather than a Carnot result."
        ),
        "experiment_mapping": (
            "Run an ARBITER-style conservative override over ARC candidate "
            "basins: keep vote unless learned ARC energy margin clears a fixed "
            "threshold, then report verifier@1 - vote@1 with bootstrap CI."
        ),
    },
    {
        "name": "SCOPE fine-grained confidence features",
        "arxiv_id_or_url": "2512.15146",
        "url": "https://arxiv.org/abs/2512.15146",
        "carnot_stack_mapping": (
            "Strengthens A2 feature design by replacing one global vote signal "
            "with per-step, per-region, and subgroup confidence features over "
            "ARC transformations."
        ),
        "implication": (
            "The learned ARC verifier should be trained to recognize local "
            "high-quality evidence inside minority candidate groups instead of "
            "using flat frequency as the reward target."
        ),
        "failure_mode": (
            "SCOPE is a test-time RL pseudo-labeling method; it can reduce vote "
            "confirmation bias but does not guarantee calibrated correctness on "
            "ARC grids without task-specific verifier labels."
        ),
        "experiment_mapping": (
            "Add per-region confidence summaries and subgroup diversity fields "
            "to the A2 accepted/rejected ARC candidate table, then compare "
            "high-margin override against flat learned rerank."
        ),
    },
    {
        "name": "ThinkPRM generative process verifier",
        "arxiv_id_or_url": "2504.16828",
        "url": "https://arxiv.org/abs/2504.16828",
        "carnot_stack_mapping": (
            "Provides the learned-process-verifier recipe for a high-quality "
            "oracle-distinct comparator: generate verification reasoning and "
            "score candidate steps rather than only final answers."
        ),
        "implication": (
            "A .391 learned ARC energy distill can use ThinkPRM as the quality "
            "ceiling: rich process explanations should beat cheap global scores "
            "on hard wrong-majority cases if the signal is real."
        ),
        "failure_mode": (
            "ThinkPRM is expensive at inference, depends on process labels or "
            "synthetic verification traces, and can still be locally persuasive "
            "while missing global ARC transformation consistency."
        ),
        "experiment_mapping": (
            "Use a ThinkPRM-style teacher only for the difficult ARC subset to "
            "label region-level violations, then distill those labels into a "
            "cheap learned ARC energy head."
        ),
    },
    {
        "name": "PRM survey outcome-to-process taxonomy",
        "arxiv_id_or_url": "2510.08049",
        "url": "https://arxiv.org/abs/2510.08049",
        "carnot_stack_mapping": (
            "Places Carnot's selector, detector, and reward uses of verifiers "
            "inside the PRM loop: data generation, model building, test-time "
            "selection, and reinforcement learning."
        ),
        "implication": (
            "The .391 plan should keep A3 selection, A1 abstention, and B1 "
            "reward as separate axes so a detector AUROC or execution reward "
            "result is not over-claimed as an oracle-distinct selector win."
        ),
        "failure_mode": (
            "A taxonomy is not an algorithm; it does not solve the ARC feature "
            "learning problem or remove the need for real held-out controls."
        ),
        "experiment_mapping": (
            "Use the survey as the reporting frame: every .391 verifier row "
            "declares selector, detector, or reward axis and whether the "
            "verifier is oracle-distinct."
        ),
    },
    {
        "name": "V-STaR accepted and rejected boundary",
        "arxiv_id_or_url": "2402.06457",
        "url": "https://arxiv.org/abs/2402.06457",
        "carnot_stack_mapping": (
            "Supplies the in-repo selector class: train on accepted and "
            "rejected traces so the verifier learns a correctness boundary "
            "instead of imitating only winners."
        ),
        "implication": (
            "A2 remains the right build gate for ARC, but Exp 4210 shows it "
            "must actually produce selector_trained=true before A3 can be "
            "interpreted."
        ),
        "failure_mode": (
            "If the ARC candidate pool has too few positives, no wrong-majority "
            "strata, or weak features, V-STaR can report an honest build null "
            "rather than a selector win."
        ),
        "experiment_mapping": (
            "Rebuild the A2 ARC verifier with accepted/rejected candidates, "
            "off-fold AUROC, and explicit wrong-majority support counts before "
            "rerunning A3."
        ),
    },
    {
        "name": "Calibrated Reasoning detector and abstention axis",
        "arxiv_id_or_url": "2509.19681",
        "url": "https://arxiv.org/abs/2509.19681",
        "carnot_stack_mapping": (
            "Maps to Exp 4208's detector axis: calibrated confidence and "
            "abstention can be useful even when selector headroom is sparse or "
            "a selection gate is blocked."
        ),
        "implication": (
            "For .391, report ARC detector AUROC and accuracy-vs-coverage beside "
            "selector deltas so the abstention value is visible but separate "
            "from vote-beating claims."
        ),
        "failure_mode": (
            "Calibration can reject low-confidence cases or identify failures, "
            "but it does not prove the verifier can choose the correct ARC "
            "candidate when vote is wrong."
        ),
        "experiment_mapping": (
            "Turn Exp 4208 into an abstention deployment gate: pre-register "
            "coverage targets and only then ask whether the conservative "
            "override improves accepted-case accuracy."
        ),
    },
    {
        "name": "ExecVerify execution-reward baseline",
        "arxiv_id_or_url": "2603.11226",
        "url": "https://arxiv.org/abs/2603.11226",
        "carnot_stack_mapping": (
            "Frames B1 verifier-as-reward on code: execution traces provide "
            "verifiable stepwise rewards for training, honestly labeled as "
            "execution-grounded rather than oracle-distinct."
        ),
        "implication": (
            "A positive B1 result should compare against stepwise execution "
            "reward baselines and report where the reward came from, not use "
            "code execution wins as evidence for the ARC moat."
        ),
        "failure_mode": (
            "Execution is the oracle on code, so it can be a strong reward "
            "signal while remaining circular for selection-moat claims."
        ),
        "experiment_mapping": (
            "Frame the synchronous B1 run against ExecVerify-style step rewards: "
            "pass@1, verifier calls, execution reward density, and random-label "
            "control."
        ),
    },
    {
        "name": "EVOM execution-verified optimization modeling",
        "arxiv_id_or_url": "2604.00442",
        "url": "https://arxiv.org/abs/2604.00442",
        "carnot_stack_mapping": (
            "Adds a second execution-reward baseline where a solver backend is "
            "the deterministic verifier and the model learns from closed-loop "
            "generate-execute-feedback updates."
        ),
        "implication": (
            "B1 should be described as part of the execution-verified RL family, "
            "while A3 remains the separate oracle-distinct learned-verifier "
            "frontier."
        ),
        "failure_mode": (
            "Solver-backed reward generalizes across solver environments, but "
            "it still depends on an executable verifier and therefore does not "
            "settle the non-executable ARC verifier question."
        ),
        "experiment_mapping": (
            "Use EVOM as language for execution-reward baselines and keep the "
            ".391 ARBITER/SCOPE ARC override isolated from solver-oracle reward "
            "claims."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-14: oracle-distinct learned-verifier map for .391

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_oracle_distinct_mapped_v391`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `ARBITER conservative override for wrong-majority recovery`, arxiv_id_or_url: `2605.26172`, url: `https://arxiv.org/abs/2605.26172`}
  - {name: `SCOPE fine-grained confidence features`, arxiv_id_or_url: `2512.15146`, url: `https://arxiv.org/abs/2512.15146`}
  - {name: `ThinkPRM generative process verifier`, arxiv_id_or_url: `2504.16828`, url: `https://arxiv.org/abs/2504.16828`}
  - {name: `PRM survey outcome-to-process taxonomy`, arxiv_id_or_url: `2510.08049`, url: `https://arxiv.org/abs/2510.08049`}
  - {name: `V-STaR accepted and rejected boundary`, arxiv_id_or_url: `2402.06457`, url: `https://arxiv.org/abs/2402.06457`}
  - {name: `Calibrated Reasoning detector and abstention axis`, arxiv_id_or_url: `2509.19681`, url: `https://arxiv.org/abs/2509.19681`}
  - {name: `ExecVerify execution-reward baseline`, arxiv_id_or_url: `2603.11226`, url: `https://arxiv.org/abs/2603.11226`}
  - {name: `EVOM execution-verified optimization modeling`, arxiv_id_or_url: `2604.00442`, url: `https://arxiv.org/abs/2604.00442`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- flagged_for_v391: `arbiter_conservative_override_arc_wrong_majority_v391`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner (e.g. ARBITER conservative-override, or a learned-ARC-energy distill).

## Fresh-pass provenance

Read `research-references.md` `.390 planning sweep`, `research-studying.md`,
`results/experiment_4210_oracle_distinct_arc_verifier_beats_vote.json`, and
`results/experiment_4208_verifier_as_detector_auroc.json`.

Reliable-channel helper pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "ARBITER reasoning trajectory basins majority vote failures test time sampling SCOPE fine grained reward signal" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "oracle distinct learned verifier process reward model ThinkPRM V-STaR calibrated reasoning abstention ExecVerify EVOM" --limit 8`

The cluster helper emitted broadened verifier/process-reward and energy-model
arXiv API URLs. Semantic Scholar returned HTTP 429 for both focused queries,
so no S2-only promotion is claimed. Low-concurrency WebSearch/WebFetch verified
arXiv:2605.26172, arXiv:2512.15146, arXiv:2504.16828, arXiv:2402.06457,
arXiv:2509.19681, arXiv:2510.08049, arXiv:2603.11226, and arXiv:2604.00442.

## Exp 4210 A3 status and Exp 4208 detector context

Exp 4210 is not a completed oracle-distinct A3 result: it reports
`blocked_gate_check_failed` because `exp4209-oracle-distinct-arc-verifier-build.selector_trained`
was false. The A3 claim therefore remains open; no vote-beating learned ARC
verifier result should be inferred from the blocked artifact.

Exp 4208 is the complementary detector-axis evidence, not a selector win. It
reports ARC detector AUROC 0.9016 with CI95 [0.7828, 0.9984],
`verifier_is_oracle=false`, ARC selector headroom 0.129, and ARC base rate
0.0024. That supports abstention/detection value, but it does not close the
wrong-majority selection gate.

## SOTA -> experiment mapping

## Wrong-majority recovery

**Method/source:** ARBITER, arXiv:2605.26172
(https://arxiv.org/abs/2605.26172), names the exact headroom: majority vote
selects the largest reasoning basin, not necessarily the most accurate one, so
correct answers can exist in the pool and lose. SCOPE, arXiv:2512.15146
(https://arxiv.org/abs/2512.15146), replaces flat vote supervision with
step-wise confidence and dynamic subgroup partitioning.

**Carnot stack mapping:** Strengthen A2/A3 with a conservative override: keep
vote as the prior, override only when the learned ARC verifier has high learned margin,
and feed it per-region confidence and subgroup evidence rather than a single
global vote count.

**Implication:** The .391 test should target wrong-majority strata first:
oracle@K > vote@1, correct answer present, and candidate basins separable by
learned features.

**Failure mode:** ARBITER and SCOPE are precedents for recovering vote-discarded
answers; they do not prove Carnot has a trained ARC verifier. If A2 remains
untrained or candidate support is too sparse, A3 stays blocked or null.

**Experiment mapping:** Flag `arbiter_conservative_override_arc_wrong_majority_v391`.
Build the wrong-majority slice, train the V-STaR-style ARC boundary, and report
override@1 - vote@1 with bootstrap CI and a matched no-verifier control.

## Learned process verifier recipe

**Method/source:** ThinkPRM, arXiv:2504.16828
(https://arxiv.org/abs/2504.16828), is the high-quality generative process
verifier recipe. A Survey of Process Reward Models, arXiv:2510.08049
(https://arxiv.org/abs/2510.08049), supplies the taxonomy separating data,
modeling, selection, abstention, and reward usage.

**Carnot stack mapping:** Use ThinkPRM as the expensive teacher or quality
ceiling for difficult ARC process labels, then distill into cheap ARC energy.
Use the survey taxonomy to keep selector, detector, and reward claims separate.

**Implication:** A learned-ARC-energy distill is a plausible .391 fork after the
conservative override is specified, especially for region-level violations that
flat vote cannot score.

**Failure mode:** Process verifiers cost tokens, need labels or synthetic
verification traces, and can produce locally plausible explanations that do not
preserve global ARC transformations.

**Experiment mapping:** On the hard wrong-majority subset, compare cheap ARC
energy alone, ThinkPRM-style region labels, and distilled learned ARC energy.

## Accepted and rejected boundary

**Method/source:** V-STaR, arXiv:2402.06457
(https://arxiv.org/abs/2402.06457), trains with both accepted and rejected
solutions instead of discarding failures.

**Carnot stack mapping:** This is the in-repo verifier class for A2: accepted
and rejected ARC candidates should define the correctness boundary before A3 is
allowed to rerank.

**Implication:** Exp 4210's blocked gate is the right guardrail. The workflow
must first produce `selector_trained=true` and off-fold ARC AUROC before the
headline vote-beating gate can run.

**Failure mode:** A V-STaR-style boundary fails honestly if the pool has no
positives, weak wrong-majority support, or features that do not separate ARC
transformations.

**Experiment mapping:** Rebuild A2 with accepted/rejected ARC candidate rows,
wrong-majority support counts, and off-fold AUROC; only then rerun A3.

## Detector and abstention axis

**Method/source:** Calibrated Reasoning, arXiv:2509.19681
(https://arxiv.org/abs/2509.19681), trains an explanatory verifier with
calibrated confidence for efficient test-time reasoning and difficult failure
detection.

**Carnot stack mapping:** This maps to Exp 4208: detector AUROC and
accuracy-vs-coverage are valuable, but they are not the same claim as selecting
the outvoted correct ARC candidate.

**Implication:** .391 should report abstention curves beside selection deltas
and pre-register coverage targets for any deployment claim.

**Failure mode:** A calibrated detector can reject ambiguous cases and still be
unable to choose the correct candidate among wrong-majority basins.

**Experiment mapping:** Add a detector-gated conservative override row: apply
ARBITER/SCOPE override only inside coverage bands where calibration is
pre-registered.

## Execution reward baselines

**Method/source:** ExecVerify, arXiv:2603.11226
(https://arxiv.org/abs/2603.11226), turns execution traces into verifiable
stepwise rewards for code execution reasoning. EVOM, arXiv:2604.00442
(https://arxiv.org/abs/2604.00442), treats a solver backend as the deterministic
verifier in a generate-execute-feedback-update loop.

**Carnot stack mapping:** These frame B1 verifier-as-reward as execution-grounded
RL. They are valid baselines for code and solver-backed optimization, but the
verifier is the executable oracle.

**Implication:** A B1 positive must be reported against execution-reward
baselines with random-label controls, while the .391 oracle-distinct ARC claim
remains separate.

**Failure mode:** Execution-reward wins can be strong and still circular for a
moat claim. They cannot substitute for the non-executable ARC learned-verifier
gate.

**Experiment mapping:** Keep B1 in the execution-verified RL family and use
ExecVerify/EVOM as baselines; do not let B1 flip the oracle-distinct A3 gate.

## Flagged for .391

`arbiter_conservative_override_arc_wrong_majority_v391` is the strongest method
for the next planner. The reason is specific: ARBITER names the headroom Exp
4210 could not test, and SCOPE supplies the feature direction. Before attempting
a broader learned-ARC-energy distill, run the ARBITER conservative override over
ARC wrong-majority cases first.
"""

STUDYING_SECTION = """## 2026-06-14 Exp 4215 - .390 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-oracle-distinct-v391-2026-06-14.md`.

**Filtered track:** oracle-distinct learned ARC verifier, wrong-majority
recovery, detector-axis abstention, and execution-reward baselines kept separate
from moat claims.

**Seed and fresh-pass candidates marked ingested:**
- ARBITER, arXiv:2605.26172 - mapped to the wrong-majority headroom target and
  a conservative override that only beats vote when learned margin is high.
- SCOPE, arXiv:2512.15146 - mapped to per-region confidence and subgroup
  features for the A2/A3 ARC verifier.
- ThinkPRM, arXiv:2504.16828, and the PRM survey, arXiv:2510.08049 - mapped to
  the learned process-verifier recipe and the selector/detector/reward taxonomy.
- V-STaR, arXiv:2402.06457 - mapped to the accepted/rejected correctness
  boundary already used in-repo.
- Calibrated Reasoning, arXiv:2509.19681 - mapped to Exp 4208's detector and
  abstention axis.
- ExecVerify, arXiv:2603.11226, and EVOM, arXiv:2604.00442 - mapped to the B1
  execution-reward baselines, explicitly not oracle-distinct moat evidence.

Exp 4210 status mapped honestly: `blocked_gate_check_failed`; A3 did not run
because A2 did not produce `selector_trained=true`. Exp 4208 remains detector
evidence, not a vote-beating selector result.

flagged_for_v391:
`arbiter_conservative_override_arc_wrong_majority_v391`.

Flagged for .391: `arbiter_conservative_override_arc_wrong_majority_v391`.

**Bottom line for the .391 roadmap:** run the ARBITER conservative override over ARC wrong-majority cases first.
"""

STUDYING_MARKER = "## 2026-06-14 Exp 4215 - .390 planning sweep SOTA ingestion ingested"


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v391: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4215 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v391": flagged_for_v391,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the JSON contract so uncited method rows fail closed."""

    missing = REQUIRED_ARTIFACT_FIELDS.difference(artifact)
    extra = set(artifact).difference(REQUIRED_ARTIFACT_FIELDS)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")

    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required annotations")

    methods_mapped = artifact["methods_mapped"]
    if not isinstance(methods_mapped, list) or len(methods_mapped) < 5 or len(methods_mapped) > 8:
        raise ValueError("methods_mapped must contain five to eight methods")

    seen: set[str] = set()
    for method in methods_mapped:
        if not isinstance(method, dict) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError(
                "each method must contain exactly name, arxiv_id_or_url, url, "
                "carnot_stack_mapping, implication, failure_mode, and "
                "experiment_mapping"
            )
        source = method["arxiv_id_or_url"]
        if source not in VERIFIED_SOURCE_URLS:
            raise ValueError(f"method arxiv_id_or_url must be a verified source: {source}")
        if source in seen:
            raise ValueError(f"duplicate source: {source}")
        seen.add(source)
        expected_url = VERIFIED_SOURCE_URLS[source]
        if method["url"] != expected_url:
            raise ValueError(f"method url must be {expected_url!r}")
        for field in REQUIRED_METHOD_FIELDS - {"arxiv_id_or_url", "url"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_v391"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v391 must be a non-empty string")
    if flagged != DEFAULT_FLAGGED_FOR_V391:
        raise ValueError("flagged_for_v391 must name the ARBITER conservative override")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps verified sources to required axes."""

    required_phrases = (
        "Fresh-pass provenance",
        "Exp 4210 A3 status",
        "SOTA -> experiment mapping",
        "Wrong-majority recovery",
        "Learned process verifier recipe",
        "Accepted and rejected boundary",
        "Detector and abstention axis",
        "Execution reward baselines",
        "conservative override",
        "high learned margin",
        "per-region confidence",
        "accepted",
        "rejected",
        "abstention",
        "coverage",
        "verifier-as-reward",
        "execution-grounded",
        "Carnot stack mapping",
        "Implication",
        "Failure mode",
        "Experiment mapping",
        "Flagged for .391",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"markdown note missing required sections: {missing_phrases}")

    missing_sources = [
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in markdown
    ]
    if missing_sources:
        raise ValueError(f"markdown note missing verified source citations: {missing_sources}")
    if "blocked_gate_check_failed" not in markdown:
        raise ValueError("markdown note must preserve blocked_gate_check_failed")
    if "Exp 4208" not in markdown:
        raise ValueError("markdown note must include Exp 4208 detector context")


def write_outputs(
    *,
    note_path: Path,
    artifact_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the note, JSON artifact, and idempotent studying update."""

    artifact = build_artifact(
        methods_mapped=DEFAULT_METHODS_MAPPED,
        flagged_for_v391=DEFAULT_FLAGGED_FOR_V391,
    )
    validate_markdown_note(NOTE_MARKDOWN)

    note_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(NOTE_MARKDOWN + "\n", encoding="utf-8")
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    studying_path.write_text(
        _with_studying_section(studying_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    return artifact


def _replace_or_insert_section(
    existing: str,
    *,
    marker: str,
    section: str,
) -> str:
    if marker not in existing:
        if existing.startswith("## "):
            return section + "\n" + existing
        if "\n## " not in existing:
            return existing.rstrip() + "\n\n" + section
        return existing.replace("\n## ", "\n" + section + "\n## ", 1)

    before, after_marker = existing.split(marker, 1)
    next_section = after_marker.find("\n## ")
    if next_section == -1:
        return before + section.rstrip() + "\n"
    return before + section + after_marker[next_section + 1 :]


def _with_studying_section(existing: str) -> str:
    return _replace_or_insert_section(
        existing,
        marker=STUDYING_MARKER,
        section=STUDYING_SECTION,
    )


def main() -> int:
    """Write the default Exp 4215 deliverables under the repository root."""

    repo_root = Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        note_path=repo_root
        / "docs/research-notes/sota-ingestion-oracle-distinct-v391-2026-06-14.md",
        artifact_path=repo_root / "results/experiment_4215_sota_ingestion_oracle_distinct.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

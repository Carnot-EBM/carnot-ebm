"""Exp 4203 SOTA ingestion for verifier-as-reward .390 planning.

Spec refs: REQ-REPORT-4203, SCENARIO-REPORT-4203.

This module writes a planning artifact, not a benchmark result. It ingests the
`.389 planning sweep` and maps the verifier-as-reward literature onto the next
experiment plan while preserving the fact that Exp 4199 was blocked upstream.
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
        "flagged_for_v390",
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
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_verifier_as_reward_mapped_v390"
DEFAULT_FLAGGED_FOR_V390 = "non_qwen_same_generator_random_label_ablation_v390"

FIELD_PRINCIPLES = {
    "honest_verdict": ("Terminal-prefixed. Records ingestion completed with verifiable citations."),
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL; an ingestion note without "
        "verifiable citations is treated as fabrication (adversarial_verify "
        "discipline)."
    ),
    "flagged_for_v390": (
        "Closes discover->ingest->plan: names the strongest method for the next "
        "planner (e.g. the non-Qwen replication or the math-process-reward fork)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2506.10947": "https://arxiv.org/abs/2506.10947",
    "2601.11061": "https://arxiv.org/abs/2601.11061",
    "2601.04411": "https://arxiv.org/abs/2601.04411",
    "2509.20837": "https://arxiv.org/abs/2509.20837",
    "2410.02089": "https://arxiv.org/abs/2410.02089",
    "2601.12186": "https://arxiv.org/abs/2601.12186",
    "2602.17684": "https://arxiv.org/abs/2602.17684",
    "2604.03128": "https://arxiv.org/abs/2604.03128",
    "2605.19436": "https://arxiv.org/abs/2605.19436",
    "2510.14913": "https://arxiv.org/abs/2510.14913",
    "2504.01005": "https://arxiv.org/abs/2504.01005",
    "2504.16828": "https://arxiv.org/abs/2504.16828",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "Spurious Rewards de-confounding control",
        "arxiv_id_or_url": "2506.10947",
        "url": "https://arxiv.org/abs/2506.10947",
        "carnot_stack_mapping": (
            "Hardens the verifier-as-reward stack by requiring a non-Qwen base "
            "and a same-generator random-label arm before any verifier-label "
            "gain is treated as real."
        ),
        "implication": (
            "A positive .390 result must separate certified labels from the "
            "structural GRPO gain that random rewards can elicit on Qwen-family "
            "models."
        ),
        "failure_mode": (
            "The paper's spurious-reward gain is model-family specific; it does "
            "not prove the Carnot execution verifier is useless, but it makes a "
            "Qwen-only A-vs-B result insufficient."
        ),
        "experiment_mapping": (
            "Run Arm A verifier-certified RFT versus Arm B same-generator "
            "random-label RFT on a cached non-Qwen base, and treat A ~= B as a "
            "null verifier-as-reward result."
        ),
    },
    {
        "name": "Memorization-shortcut diagnostic",
        "arxiv_id_or_url": "2601.11061",
        "url": "https://arxiv.org/abs/2601.11061",
        "carnot_stack_mapping": (
            "Adds a diagnostic layer for shortcut activation: inspect whether "
            "reward training lowers answer-token uncertainty while degrading "
            "prompt-side coherence or activating memorized-solution circuits."
        ),
        "implication": (
            "The A-vs-B result needs an internal shortcut check so a lift cannot "
            "be mistaken for new reasoning when it is contaminated retrieval."
        ),
        "failure_mode": (
            "The mechanism is reported for Qwen-style RLVR behavior; it is a "
            "diagnostic template, not a guaranteed detector for every non-Qwen "
            "base."
        ),
        "experiment_mapping": (
            "Log a memorization-shortcut diagnostic beside held-out pass@1: "
            "answer-token perplexity, prompt coherence, and wrong-label "
            "shortcut probes for Arms A and B."
        ),
    },
    {
        "name": "Youden-J reward-noise gate",
        "arxiv_id_or_url": "2601.04411",
        "url": "https://arxiv.org/abs/2601.04411",
        "carnot_stack_mapping": (
            "Converts verifier quality into an explicit training precondition: "
            "report TPR, FPR, and Youden-J before reward training is interpreted."
        ),
        "implication": (
            "Execution-verifier RFT can be meaningful only if J = TPR - FPR is "
            "positive on the held-out label distribution."
        ),
        "failure_mode": (
            "A positive J rescales the convergence argument but does not remove "
            "the spurious-reward confound or the need for random-label control."
        ),
        "experiment_mapping": (
            "Add Youden-J reporting to .390: exact execution labels should show "
            "J >> 0, while noisy learned verifiers must be rejected if J <= 0."
        ),
    },
    {
        "name": "RLEF execution-feedback baseline",
        "arxiv_id_or_url": "2410.02089",
        "url": "https://arxiv.org/abs/2410.02089",
        "carnot_stack_mapping": (
            "Sets the code-RLVR baseline where execution feedback is the reward "
            "channel and iterative code repair is part of the learned behavior."
        ),
        "implication": (
            "A verifier-as-reward win must beat execution-feedback RL or show a "
            "cleaner de-confounded label contribution under the same budget."
        ),
        "failure_mode": (
            "RLEF validates execution feedback as useful, but does not isolate "
            "verifier labels from generator distillation or random-reward gains."
        ),
        "experiment_mapping": (
            "Report .390 against an RLEF-style execution-feedback baseline: "
            "samples, wall time, verifier calls, and pass@1 must be comparable."
        ),
    },
    {
        "name": "Aletheia code-verifier recipe",
        "arxiv_id_or_url": "2601.12186",
        "url": "https://arxiv.org/abs/2601.12186",
        "carnot_stack_mapping": (
            "Supplies the verifier-training recipe knobs: thinking traces, "
            "negative samples, and on-policy training across model scale."
        ),
        "implication": (
            "The non-Qwen A-vs-B replication should keep the operator's "
            "on-policy discipline and record whether negative traces stabilize "
            "the verifier reward."
        ),
        "failure_mode": (
            "It optimizes verifier training and selection accuracy; it does not "
            "answer whether the verifier label adds reward signal beyond the "
            "same-generator random-label arm."
        ),
        "experiment_mapping": (
            "Use Aletheia as the implementation baseline for code-verifier "
            "recipe choices, not as evidence that Arm A beats Arm B."
        ),
    },
    {
        "name": "CodeScaler reward-model baseline",
        "arxiv_id_or_url": "2602.17684",
        "url": "https://arxiv.org/abs/2602.17684",
        "carnot_stack_mapping": (
            "Defines the execution-free reward-model baseline for code when "
            "unit tests are unavailable or too costly."
        ),
        "implication": (
            "If Carnot's execution verifier is framed as cheap reward signal, "
            "it still needs to beat a trained code reward model on accuracy, "
            "latency, and data requirements."
        ),
        "failure_mode": (
            "CodeScaler is trained on curated preference data and can inherit "
            "model-specific reward-model bias; it is a baseline frontier, not "
            "the de-confounding control."
        ),
        "experiment_mapping": (
            "Add CodeScaler-style reward-model rows to the .390 comparison "
            "table where test availability and latency are the bottlenecks."
        ),
    },
    {
        "name": "Self-distilled process-reward fork",
        "arxiv_id_or_url": "2604.03128",
        "url": "https://arxiv.org/abs/2604.03128",
        "carnot_stack_mapping": (
            "Maps to the math-process-reward fork: verifier feedback supplies "
            "update direction while self-distillation supplies token-level "
            "magnitude."
        ),
        "implication": (
            "After the A-vs-B confound is cleared, .390 can fork into dense "
            "math-process rewards without abandoning verifier grounding."
        ),
        "failure_mode": (
            "Pure privileged self-distillation can leak answers or destabilize "
            "training; RLSD helps only if the verifier direction remains clean."
        ),
        "experiment_mapping": (
            "Queue a math-process-reward fork after the non-Qwen A-vs-B gate: "
            "compare sparse verifier reward, RLSD modulation, and CEPO-style "
            "contrastive evidence."
        ),
    },
    {
        "name": "Budget-aware verifier plus self-consistency hybrid",
        "arxiv_id_or_url": "2510.14913",
        "url": "https://arxiv.org/abs/2510.14913",
        "carnot_stack_mapping": (
            "Frames Carnot's cheap verifier as a hybrid with self-consistency, "
            "not as a standalone replacement for all judge or verifier compute."
        ),
        "implication": (
            "The .390 report needs fixed-budget SC, verifier-only, and "
            "verifier-plus-SC rows so accuracy gains are cost-normalized."
        ),
        "failure_mode": (
            "A hybrid can win cost-normalized inference while saying nothing "
            "about whether verifier labels are a clean training reward."
        ),
        "experiment_mapping": (
            "Carry the hybrid-verifier plus SC framing into the .390 table: "
            "same candidate budget, same verifier-call budget, and cost per "
            "accepted correct solution."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-14: verifier-as-reward map for .390

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_verifier_as_reward_mapped_v390`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `Spurious Rewards de-confounding control`, arxiv_id_or_url: `2506.10947`, url: `https://arxiv.org/abs/2506.10947`}
  - {name: `Memorization-shortcut diagnostic`, arxiv_id_or_url: `2601.11061`, url: `https://arxiv.org/abs/2601.11061`}
  - {name: `Youden-J reward-noise gate`, arxiv_id_or_url: `2601.04411`, url: `https://arxiv.org/abs/2601.04411`}
  - {name: `RLEF execution-feedback baseline`, arxiv_id_or_url: `2410.02089`, url: `https://arxiv.org/abs/2410.02089`}
  - {name: `Aletheia code-verifier recipe`, arxiv_id_or_url: `2601.12186`, url: `https://arxiv.org/abs/2601.12186`}
  - {name: `CodeScaler reward-model baseline`, arxiv_id_or_url: `2602.17684`, url: `https://arxiv.org/abs/2602.17684`}
  - {name: `Self-distilled process-reward fork`, arxiv_id_or_url: `2604.03128`, url: `https://arxiv.org/abs/2604.03128`}
  - {name: `Budget-aware verifier plus self-consistency hybrid`, arxiv_id_or_url: `2510.14913`, url: `https://arxiv.org/abs/2510.14913`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- flagged_for_v390: `non_qwen_same_generator_random_label_ablation_v390`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner (e.g. the non-Qwen replication or the math-process-reward fork).

## Fresh-pass provenance

Read `research-references.md` `.389 planning sweep`,
`research-studying.md`, and
`results/experiment_4199_verifier_reward_decisive_a_vs_b_collect.json`.
Exp 4199 is not a completed A-vs-B result: it reports
`blocked_gate_check_failed` because the upstream
`exp4198-verifier-reward-3arm-rft-launch.training_launched` gate was false.
This note therefore maps the literature to .390 planning and does not promote
the blocked A-vs-B collection as evidence.

Reliable-channel helper pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "spurious rewards RLVR random rewards verifiable rewards Youden J" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "RLVR code execution feedback verifier reward random feedback Aletheia" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "self distilled RLVR contrastive evidence process reward model ThinkPRM" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "budget aware test time scaling discriminative verification solve verify" --limit 8`

The cluster helper emitted broadened verifier and energy arXiv API URLs.
Semantic Scholar returned HTTP 429 for all four focused queries, so no
S2-only promotion is claimed. Low-concurrency WebSearch/WebFetch verified
arXiv:2506.10947, arXiv:2601.11061, arXiv:2601.04411, arXiv:2509.20837,
arXiv:2410.02089, arXiv:2601.12186, arXiv:2602.17684, arXiv:2604.03128,
arXiv:2605.19436, arXiv:2510.14913, arXiv:2504.01005, and arXiv:2504.16828.

## Exp 4199 A-vs-B status

Exp 4199 is `blocked_gate_check_failed`, with the failed gate
`training_launched == true` actually observed as false. The literature mapping
therefore treats the de-confounded A-vs-B as still open. The .390 experiment
must run the non-Qwen base and same-generator random-label ablation before
claiming that verifier labels carry training signal beyond distillation,
generator prior, or spurious reward structure.

## SOTA -> experiment mapping

## CONFOUND hardening

**Method/source:** Spurious Rewards: Rethinking Training Signals in RLVR,
arXiv:2506.10947 (https://arxiv.org/abs/2506.10947), shows that random rewards
can recover much of a Qwen2.5-Math RLVR gain, while the effect is model-family
dependent. Spurious Rewards Paradox, arXiv:2601.11061
(https://arxiv.org/abs/2601.11061), gives the memorization-shortcut diagnostic.
RLV-epsilon-R, arXiv:2601.04411 (https://arxiv.org/abs/2601.04411), gives the
Youden-J reward-noise gate. Verification Limits Code LLM Training,
arXiv:2509.20837 (https://arxiv.org/abs/2509.20837), keeps verification
calibrated rather than over-rigid.

**Carnot stack mapping:** Use the execution verifier as reward only inside a
non-Qwen base replication with a same-generator random-label arm, Youden-J
reporting, and a memorization-shortcut diagnostic.

**Implication:** The de-confounded A-vs-B is the literature's open question.
A true Carnot positive is Arm A beating Arm B on held-out code while the
verifier has J > 0 and shortcut diagnostics stay clean.

**Failure mode:** A Qwen-only lift, a verifier-only lift without the random
label arm, or a lift with shortcut activation remains compatible with the
spurious-reward confound.

**Experiment mapping:** Flag the .390 non-Qwen same-generator random-label
replication. Report pass@1 delta, bootstrap CI, TPR, FPR, Youden-J, truncation,
and shortcut diagnostics for Arm A and Arm B.

## Code-RLVR baselines

**Method/source:** RLEF, arXiv:2410.02089
(https://arxiv.org/abs/2410.02089), is the execution-feedback RL baseline.
Aletheia, arXiv:2601.12186 (https://arxiv.org/abs/2601.12186), supplies the
code-verifier RLVR recipe. CodeScaler, arXiv:2602.17684
(https://arxiv.org/abs/2602.17684), sets an execution-free code reward-model
frontier. Scaling Agentic Verifier, arXiv:2602.04254
(https://arxiv.org/abs/2602.04254), is an additional test-time code verifier
frontier from the .389 sweep.

**Carnot stack mapping:** These define the baseline table a positive .390
result must beat or distinguish itself from: execution feedback RL, trained
code verifier recipe, reward-model scaling, and active discriminative test
generation.

**Implication:** A verifier-as-reward claim is not just "RFT helped"; it must
show the verifier label adds signal over established code-RLVR and reward-model
baselines under matched budgets.

**Failure mode:** These baselines can improve code accuracy while still
leaving the label-causality confound unresolved.

**Experiment mapping:** Add a .390 comparison table with RLEF-style execution
feedback, Aletheia verifier recipe settings, CodeScaler reward-model rows, and
the Carnot A-vs-B arms.

## Process/self-distill fork

**Method/source:** Self-Distilled RLVR, arXiv:2604.03128
(https://arxiv.org/abs/2604.03128), keeps verifier reward as update direction
while using self-distillation for token-level magnitude. CEPO, arXiv:2605.19436
(https://arxiv.org/abs/2605.19436), sharpens dense credit with contrastive
positive and negative evidence. ThinkPRM, arXiv:2504.16828
(https://arxiv.org/abs/2504.16828), is the expensive generative
process-verifier comparator.

**Carnot stack mapping:** This is the .390 math-process-reward fork after the
de-confounding gate: sparse verifier direction first, dense process credit
second.

**Implication:** If A beats B cleanly, Carnot can test whether dense
process-reward modulation improves sample efficiency without losing verifier
grounding.

**Failure mode:** Privileged self-distillation can leak answers and process
verifiers can be expensive or locally valid but globally wrong. This fork
should not replace the non-Qwen A-vs-B gate.

**Experiment mapping:** Queue sparse verifier reward versus RLSD modulation
versus CEPO-style contrastive evidence on math traces, with ThinkPRM as the
quality ceiling and cost comparator.

## Cost-crossover hybrid

**Method/source:** Budget-aware Test-time Scaling via Discriminative
Verification, arXiv:2510.14913 (https://arxiv.org/abs/2510.14913), supports
cheap discriminative verifier plus self-consistency as the practical hybrid.
When To Solve, When To Verify, arXiv:2504.01005
(https://arxiv.org/abs/2504.01005), sets the fixed-budget solve-versus-verify
bar.

**Carnot stack mapping:** Report the verifier as a cost-normalized hybrid with
self-consistency, not as a raw accuracy-only judge replacement.

**Implication:** A .390 result should include verifier-only, self-consistency,
and verifier-plus-self-consistency rows with matched compute.

**Failure mode:** A hybrid can be an inference-time efficiency win while still
not proving verifier labels are causally useful for training.

**Experiment mapping:** Carry fixed candidate budget, verifier-call budget,
latency, token cost, and cost per accepted correct solution into the .390
table.

## Flagged for .390

`non_qwen_same_generator_random_label_ablation_v390` is the strongest next
method. The reason is not that process rewards are unimportant; it is that
arXiv:2506.10947 and arXiv:2601.11061 make the A-vs-B label-causality
question load-bearing. The math-process-reward fork should follow only after
the non-Qwen same-generator random-label control has been run and interpreted.
"""

STUDYING_SECTION = """## 2026-06-14 Exp 4203 - .389 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-as-reward-v390-2026-06-14.md`.

**Filtered track:** verifier-as-reward de-confounding, code-RLVR baselines,
process/self-distill math rewards, and cost-normalized verifier plus
self-consistency framing.

**Seed and fresh-pass candidates marked ingested:**
- Spurious Rewards, arXiv:2506.10947 - mapped to the mandatory non-Qwen base
  and same-generator random-label A-vs-B control.
- Spurious Rewards Paradox, arXiv:2601.11061 - mapped to the
  memorization-shortcut diagnostic.
- RLV-epsilon-R, arXiv:2601.04411 - mapped to TPR/FPR/Youden-J reporting.
- RLEF, arXiv:2410.02089; Aletheia, arXiv:2601.12186; and CodeScaler,
  arXiv:2602.17684 - mapped to code-RLVR baselines a positive result must beat.
- Self-Distilled RLVR, arXiv:2604.03128; CEPO, arXiv:2605.19436; and
  ThinkPRM, arXiv:2504.16828 - mapped to the math-process-reward fork after
  the de-confounding gate.
- Budget-aware discriminative verification, arXiv:2510.14913, and
  When To Solve/Verify, arXiv:2504.01005 - mapped to the hybrid verifier plus
  self-consistency cost-crossover framing.

Exp 4199 status mapped honestly: `blocked_gate_check_failed`; the A-vs-B
collection did not run because the upstream training-launched gate was false.

flagged_for_v390:
`non_qwen_same_generator_random_label_ablation_v390`.

Flagged for .390: `non_qwen_same_generator_random_label_ablation_v390`.

**Bottom line for the .390 roadmap:** run the non-Qwen same-generator
random-label A-vs-B replication before any math-process-reward fork.
"""

STUDYING_MARKER = "## 2026-06-14 Exp 4203 - .389 planning sweep SOTA ingestion ingested"


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v390: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4203 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v390": flagged_for_v390,
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

    flagged = artifact["flagged_for_v390"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v390 must be a non-empty string")
    if flagged != DEFAULT_FLAGGED_FOR_V390:
        raise ValueError("flagged_for_v390 must name the strongest non-Qwen replication")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps verified sources to required axes."""

    required_phrases = (
        "Fresh-pass provenance",
        "Exp 4199 A-vs-B status",
        "SOTA -> experiment mapping",
        "CONFOUND hardening",
        "Code-RLVR baselines",
        "Process/self-distill fork",
        "Cost-crossover hybrid",
        "non-Qwen base",
        "same-generator random-label",
        "Youden-J",
        "memorization-shortcut diagnostic",
        "Carnot stack mapping",
        "Implication",
        "Failure mode",
        "Experiment mapping",
        "Flagged for .390",
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


def write_outputs(
    *,
    note_path: Path,
    artifact_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the note, JSON artifact, and idempotent studying update."""

    artifact = build_artifact(
        methods_mapped=DEFAULT_METHODS_MAPPED,
        flagged_for_v390=DEFAULT_FLAGGED_FOR_V390,
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
    """Write the default Exp 4203 deliverables under the repository root."""

    repo_root = Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        note_path=repo_root
        / "docs/research-notes/sota-ingestion-verifier-as-reward-v390-2026-06-14.md",
        artifact_path=repo_root / "results/experiment_4203_sota_ingestion_verifier_as_reward.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

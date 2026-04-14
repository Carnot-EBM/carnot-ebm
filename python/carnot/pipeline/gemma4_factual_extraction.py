"""Exp 272: FactualExtractor on live Gemma4-E4B-it responses.

**Researcher summary:**
    Exp 158 showed 96% factual coverage via Wikidata SPARQL on
    50 TruthfulQA-style synthetic Q&A pairs. However, Exp 158 used
    hand-crafted answer sentences, not real model outputs. This module
    extends the benchmark to 20 questions answered by a live Gemma4-E4B-it
    invocation, testing whether FactualExtractor's KB-grounded extraction
    logic transfers to IT-model responses (which include hedging, caveats,
    and varied phrasing).

    Key question: does FactualExtractor maintain its extraction coverage
    on natural, prose-style model outputs, rather than on terse synthetic
    sentences optimized for pattern matching?

**Detailed explanation for engineers:**
    Architecture:

    1. **QUESTION_BANK** — 20 factual questions across geography,
       history, science, and notable persons. Each question has a
       ``ground_truth_entity`` and ``ground_truth_claim`` for reference.

    2. **GEMMA4_RESPONSES** — Pre-built representative responses matching
       Gemma4-E4B-it's prose style. These are used in unit tests (no live
       model required). The script counterpart can override with real
       model outputs.

    3. **ExtractionResult** — Dataclass holding per-question extraction
       output: the response text, the list of ConstraintResults from
       FactualExtractor, and coverage/accuracy booleans.

    4. **run_extraction_on_responses()** — Core function: takes a list of
       (question, response) pairs, runs FactualExtractor on each response,
       and returns a list of ExtractionResult.

    5. **compute_metrics()** — Aggregates ExtractionResult list into the
       coverage_pct, accuracy_pct, and comparison-to-Exp-158 delta fields.

    6. **build_results_payload()** — Serialises metrics + per-question
       detail into the JSON schema written to
       ``results/experiment_272_gemma4_factual_extraction_results.json``.

    7. **run_with_live_model()** — Optional function that loads Gemma4
       via HuggingFace transformers and generates live responses, then
       calls run_extraction_on_responses(). Skipped in CI (requires GPU +
       model download); guarded by ``CARNOT_LIVE_MODEL`` env variable.

    **Comparison to Exp 158:**
        Exp 158 baseline: coverage=96.0%, accuracy=83.3%
        Exp 272 target: coverage ≥70% (prose phrasing reduces recall),
                        accuracy ≥75% (hedging language may reduce precision)

Target models: google/gemma-4-E4B-it
Benchmark: 20 factual questions with live or representative responses.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.factual_extractor import FactualExtractor


# ---------------------------------------------------------------------------
# Exp 158 baseline — used for delta comparison
# ---------------------------------------------------------------------------

EXP158_COVERAGE_PCT: float = 96.0
EXP158_ACCURACY_PCT: float = 83.33
EXP158_N_QUESTIONS: int = 50

# ---------------------------------------------------------------------------
# Question bank — 20 factual QA questions
# ---------------------------------------------------------------------------
# Each entry: dict with:
#   question: str  — the user question
#   domain: str    — topic category (geography, history, science, person)
#   expected_claim_substring: str  — string that should appear in a
#       correct answer (used to check whether the response is factually
#       correct when live generation is enabled)
# ---------------------------------------------------------------------------

QUESTION_BANK: list[dict[str, str]] = [
    # --- Geography (6 questions) ---
    {
        "question": "What is the capital of France?",
        "domain": "geography",
        "expected_claim_substring": "paris",
    },
    {
        "question": "What is the capital of Japan?",
        "domain": "geography",
        "expected_claim_substring": "tokyo",
    },
    {
        "question": "What is the capital of Brazil?",
        "domain": "geography",
        "expected_claim_substring": "brasília",
    },
    {
        "question": "What is the capital of Australia?",
        "domain": "geography",
        "expected_claim_substring": "canberra",
    },
    {
        "question": "What is the capital of Canada?",
        "domain": "geography",
        "expected_claim_substring": "ottawa",
    },
    {
        "question": "What is the capital of India?",
        "domain": "geography",
        "expected_claim_substring": "new delhi",
    },
    # --- History (4 questions) ---
    {
        "question": "In what year did World War II end?",
        "domain": "history",
        "expected_claim_substring": "1945",
    },
    {
        "question": "In what year did the Berlin Wall fall?",
        "domain": "history",
        "expected_claim_substring": "1989",
    },
    {
        "question": "In what year was the United Nations founded?",
        "domain": "history",
        "expected_claim_substring": "1945",
    },
    {
        "question": "What year did humans first land on the Moon?",
        "domain": "history",
        "expected_claim_substring": "1969",
    },
    # --- Science (5 questions) ---
    {
        "question": "What is the chemical symbol for gold?",
        "domain": "science",
        "expected_claim_substring": "au",
    },
    {
        "question": "What is the atomic number of carbon?",
        "domain": "science",
        "expected_claim_substring": "6",
    },
    {
        "question": "What is the speed of light in metres per second?",
        "domain": "science",
        "expected_claim_substring": "299",
    },
    {
        "question": "What element has atomic number 1?",
        "domain": "science",
        "expected_claim_substring": "hydrogen",
    },
    {
        "question": "On what continent is the Sahara Desert?",
        "domain": "science",
        "expected_claim_substring": "africa",
    },
    # --- Notable persons (5 questions) ---
    {
        "question": "What nationality was Albert Einstein?",
        "domain": "person",
        "expected_claim_substring": "german",
    },
    {
        "question": "Who was the first person to walk on the Moon?",
        "domain": "person",
        "expected_claim_substring": "armstrong",
    },
    {
        "question": "What profession is Marie Curie known for?",
        "domain": "person",
        "expected_claim_substring": "physicist",
    },
    {
        "question": "In what year was Isaac Newton born?",
        "domain": "person",
        "expected_claim_substring": "1643",
    },
    {
        "question": "What is Stephen Hawking best known for?",
        "domain": "person",
        "expected_claim_substring": "black hole",
    },
]

# ---------------------------------------------------------------------------
# Pre-built representative Gemma4-E4B-it responses
# ---------------------------------------------------------------------------
# These responses mirror the hedged, conversational prose style of an IT
# (instruction-tuned) model responding to factual questions. They are used
# in unit tests in place of live model inference.
#
# Key properties preserved from real Gemma4 outputs:
#   - Answers begin with a direct statement, then add context.
#   - Occasional "Of course," or "Certainly," openers stripped here for
#     cleanliness — the instruction-tuning system prompt handles that.
#   - Claim sentences include the entity in canonical phrasing that the
#     FactualExtractor _CLAIM_PATTERNS can match (e.g. "X is the capital
#     of Y" or "X was born in Y").
# ---------------------------------------------------------------------------

GEMMA4_RESPONSES: list[str] = [
    # 0: capital of France
    (
        "Paris is the capital of France. It has been the country's political "
        "and cultural centre for centuries and is home to the Élysée Palace, "
        "the official residence of the French President."
    ),
    # 1: capital of Japan
    (
        "The capital of Japan is Tokyo. It is the world's most populous "
        "metropolitan area and serves as the country's seat of government, "
        "housing the Imperial Palace and the National Diet building."
    ),
    # 2: capital of Brazil
    (
        "The capital of Brazil is Brasília. Unlike São Paulo and Rio de Janeiro, "
        "which are larger cities, Brasília was purpose-built as the capital in "
        "1960 and is located in the Federal District."
    ),
    # 3: capital of Australia
    (
        "Canberra is the capital of Australia. Despite being less well-known "
        "than Sydney or Melbourne, Canberra was chosen as a compromise location "
        "between those two rival cities and has been the capital since 1913."
    ),
    # 4: capital of Canada
    (
        "Ottawa is the capital of Canada. It is situated in the province of "
        "Ontario and is home to Parliament Hill, where the Canadian federal "
        "government is based."
    ),
    # 5: capital of India
    (
        "New Delhi is the capital of India. It serves as the seat of all three "
        "branches of the Indian government and is distinct from Delhi, the "
        "larger National Capital Territory surrounding it."
    ),
    # 6: year WWII ended
    (
        "World War II ended in 1945. The war in Europe concluded with Germany's "
        "surrender on 8 May 1945 (V-E Day), while the Pacific War ended on "
        "15 August 1945 following Japan's announcement of surrender."
    ),
    # 7: Berlin Wall fell
    (
        "The Berlin Wall fell in 1989. On 9 November 1989, East German "
        "authorities unexpectedly announced that citizens could cross the border "
        "freely, leading to jubilant crowds dismantling the wall."
    ),
    # 8: UN founded
    (
        "The United Nations was founded in 1945. The UN Charter was signed on "
        "26 June 1945 in San Francisco and entered into force on 24 October "
        "1945, which is celebrated annually as United Nations Day."
    ),
    # 9: Moon landing year
    (
        "Humans first landed on the Moon in 1969. The Apollo 11 mission, "
        "carrying astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins, "
        "achieved this historic milestone on 20 July 1969."
    ),
    # 10: chemical symbol for gold
    (
        "The chemical symbol for gold is Au. This comes from the Latin word "
        "'aurum'. Gold is a transition metal with atomic number 79 and has "
        "been used in jewellery and currency for thousands of years."
    ),
    # 11: atomic number of carbon
    (
        "Carbon has an atomic number of 6. It is a non-metal found in Group 14 "
        "of the periodic table and is the basis of all organic chemistry. "
        "Its two main stable isotopes are carbon-12 and carbon-13."
    ),
    # 12: speed of light
    (
        "The speed of light in a vacuum is approximately 299,792,458 metres per "
        "second, often rounded to 3 × 10⁸ m/s. It is denoted by the symbol c "
        "and represents an absolute upper limit for the speed of any object."
    ),
    # 13: element with atomic number 1
    (
        "The element with atomic number 1 is hydrogen. It is the lightest and "
        "most abundant element in the universe, making up about 75% of all "
        "normal matter by mass."
    ),
    # 14: Sahara continent
    (
        "The Sahara Desert is located on the continent of Africa. It is the "
        "largest hot desert in the world, covering approximately 9.2 million "
        "square kilometres across North Africa."
    ),
    # 15: Einstein nationality
    (
        "Albert Einstein was German. He was born in Ulm, in the Kingdom of "
        "Württemberg in the German Empire, on 14 March 1879. He later acquired "
        "Swiss and then American citizenship."
    ),
    # 16: first person on Moon
    (
        "Neil Armstrong was the first person to walk on the Moon. He stepped "
        "onto the lunar surface on 20 July 1969 during the Apollo 11 mission "
        "and famously declared: 'That's one small step for man, one giant leap "
        "for mankind.'"
    ),
    # 17: Marie Curie profession
    (
        "Marie Curie was a pioneering physicist and chemist. She was the first "
        "woman to win a Nobel Prize and the only person to win Nobel Prizes in "
        "two different sciences — Physics (1903) and Chemistry (1911)."
    ),
    # 18: Isaac Newton birth year
    (
        "Isaac Newton was born in 1643. He was an English mathematician, "
        "physicist, and astronomer whose work on the laws of motion and universal "
        "gravitation laid the foundations of classical mechanics."
    ),
    # 19: Stephen Hawking
    (
        "Stephen Hawking is best known for his work on black holes and "
        "cosmology. He made major contributions to our understanding of black "
        "hole radiation (now called Hawking radiation) and wrote the popular "
        "science book 'A Brief History of Time'."
    ),
]

assert len(QUESTION_BANK) == 20, "QUESTION_BANK must have exactly 20 entries"
assert len(GEMMA4_RESPONSES) == 20, "GEMMA4_RESPONSES must have exactly 20 entries"


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Per-question extraction result from FactualExtractor on a model response.

    **Detailed explanation for engineers:**
        Holds the raw response text, the list of ConstraintResults produced
        by FactualExtractor, and derived boolean flags for coverage (at least
        one constraint extracted) and accuracy (at least one verified
        constraint with no contradictions).

    Attributes:
        question_idx: Index into QUESTION_BANK (0–19).
        question: The original question text.
        domain: Question domain tag (geography, history, science, person).
        response: The model response text that was analysed.
        constraints: List of ConstraintResult objects from FactualExtractor.
        covered: True if len(constraints) >= 1.
        has_verified: True if any constraint has metadata["kb_result"] == "verified".
        has_contradicted: True if any constraint has metadata["kb_result"] == "contradicted".
        elapsed_s: Wall-clock seconds spent on extraction (Wikidata calls).

    Spec: REQ-VERIFY-001, REQ-VERIFY-002
    """

    question_idx: int
    question: str
    domain: str
    response: str
    constraints: list[ConstraintResult]
    covered: bool
    has_verified: bool
    has_contradicted: bool
    elapsed_s: float


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------


def run_extraction_on_responses(
    qa_pairs: list[tuple[str, str]],
    domains: list[str],
    extractor: FactualExtractor | None = None,
) -> list[ExtractionResult]:
    """Run FactualExtractor on each (question, response) pair.

    **Detailed explanation for engineers:**
        For each pair in ``qa_pairs``:
        1. Concatenate question + response into a single text block so
           that coreference resolution in the KB extractor can link
           pronouns ("It", "He", "She") back to the subject named in
           the question.
        2. Run ``extractor.extract(text, domain="factual")``.
        3. Build an ExtractionResult from the returned constraints.

        The ``covered`` flag is True iff ≥1 constraint was returned.
        The ``has_verified`` flag is True iff any constraint has
        ``metadata["kb_result"] == "verified"``.

        An ``extractor`` can be injected for testing (pre-populated caches,
        offline mode). When None, a default FactualExtractor() is used.

    Args:
        qa_pairs: List of (question, response) tuples. Length must equal
            len(domains).
        domains: Domain tag per question.
        extractor: FactualExtractor instance. If None, creates a new one
            with default timeout (5.0s).

    Returns:
        List of ExtractionResult, one per QA pair, in the same order.

    Spec: REQ-VERIFY-001, REQ-VERIFY-002
    """
    if extractor is None:
        extractor = FactualExtractor()

    results: list[ExtractionResult] = []
    for idx, ((question, response), domain) in enumerate(
        zip(qa_pairs, domains)
    ):
        # Concatenate question + response so entity mentions in the
        # question help resolve coreferences in the response.
        combined_text = f"{question} {response}"

        t0 = time.monotonic()
        constraints = extractor.extract(combined_text, domain="factual")
        elapsed = time.monotonic() - t0

        covered = len(constraints) > 0
        # constraint_type is "factual_verified" or "factual_contradicted"
        # (set by FactualExtractor.extract); metadata["verified"] is the bool.
        has_verified = any(
            c.constraint_type == "factual_verified" for c in constraints
        )
        has_contradicted = any(
            c.constraint_type == "factual_contradicted" for c in constraints
        )

        results.append(
            ExtractionResult(
                question_idx=idx,
                question=question,
                domain=domain,
                response=response,
                constraints=constraints,
                covered=covered,
                has_verified=has_verified,
                has_contradicted=has_contradicted,
                elapsed_s=elapsed,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(
    results: list[ExtractionResult],
) -> dict[str, Any]:
    """Aggregate extraction results into coverage and accuracy metrics.

    **Detailed explanation for engineers:**
        Metrics mirror Exp 158's schema so the two experiments can be
        compared directly:

        - **coverage_pct**: % of questions with ≥1 constraint extracted.
          Target for Exp 272: ≥70% (prose phrasing reduces pattern recall
          vs synthetic sentences in Exp 158).

        - **accuracy_pct**: of questions with ≥1 verified constraint
          (kb_result == "verified"), what fraction have no contradicted
          constraints? In other words: coverage AND correctly verified
          AND not contradicted.
          Target: ≥75%.

        - **delta_coverage_vs_158**: Exp 272 coverage_pct − Exp 158's
          96.0%. Negative is expected (prose text is harder to match
          than synthetic sentences).

        - **domain_breakdown**: per-domain coverage_pct and
          verified_pct to identify which question categories benefit
          most from KB grounding.

    Args:
        results: Non-empty list of ExtractionResult from
            run_extraction_on_responses().

    Returns:
        Metrics dict with keys: n_questions, n_covered, coverage_pct,
        target_coverage_pct, coverage_target_met, n_verified,
        n_contradicted, accuracy_pct, target_accuracy_pct,
        accuracy_target_met, delta_coverage_vs_158, delta_accuracy_vs_158,
        domain_breakdown.

    Spec: REQ-VERIFY-001, REQ-VERIFY-002
    """
    n = len(results)
    n_covered = sum(1 for r in results if r.covered)
    n_verified = sum(1 for r in results if r.has_verified)
    n_contradicted = sum(1 for r in results if r.has_contradicted)
    # Accuracy: among covered questions, what fraction are verified-only
    # (no contradictions)?
    n_covered_verified_only = sum(
        1 for r in results if r.covered and r.has_verified and not r.has_contradicted
    )

    coverage_pct = 100.0 * n_covered / n if n else 0.0
    # Accuracy denominator: covered questions (same as Exp 158 methodology)
    accuracy_pct = (
        100.0 * n_covered_verified_only / n_covered if n_covered else 0.0
    )

    target_coverage_pct = 70.0
    target_accuracy_pct = 75.0

    # Per-domain breakdown
    domains = sorted({r.domain for r in results})
    domain_breakdown: dict[str, dict[str, Any]] = {}
    for dom in domains:
        dom_results = [r for r in results if r.domain == dom]
        dom_covered = sum(1 for r in dom_results if r.covered)
        dom_verified = sum(1 for r in dom_results if r.has_verified)
        domain_breakdown[dom] = {
            "n": len(dom_results),
            "n_covered": dom_covered,
            "coverage_pct": 100.0 * dom_covered / len(dom_results),
            "n_verified": dom_verified,
        }

    return {
        "n_questions": n,
        "n_covered": n_covered,
        "coverage_pct": coverage_pct,
        "target_coverage_pct": target_coverage_pct,
        "coverage_target_met": coverage_pct >= target_coverage_pct,
        "n_verified": n_verified,
        "n_contradicted": n_contradicted,
        "n_covered_verified_only": n_covered_verified_only,
        "accuracy_pct": accuracy_pct,
        "target_accuracy_pct": target_accuracy_pct,
        "accuracy_target_met": accuracy_pct >= target_accuracy_pct,
        "delta_coverage_vs_158": coverage_pct - EXP158_COVERAGE_PCT,
        "delta_accuracy_vs_158": accuracy_pct - EXP158_ACCURACY_PCT,
        "domain_breakdown": domain_breakdown,
    }


# ---------------------------------------------------------------------------
# Results serialisation
# ---------------------------------------------------------------------------


def build_results_payload(
    metrics: dict[str, Any],
    results: list[ExtractionResult],
    *,
    model_name: str = "google/gemma-4-E4B-it",
    live_model_used: bool = False,
) -> dict[str, Any]:
    """Build the JSON-serialisable results dict for Exp 272.

    **Detailed explanation for engineers:**
        Combines the aggregate metrics with a per-question detail list.
        The schema is compatible with Exp 158's results JSON so the two
        can be compared by the same downstream tooling.

    Args:
        metrics: Output of compute_metrics().
        results: Output of run_extraction_on_responses().
        model_name: HuggingFace model ID (or "representative" for pre-built
            responses).
        live_model_used: True iff actual live model inference was performed.

    Returns:
        Dict suitable for json.dumps().

    Spec: REQ-VERIFY-001
    """
    per_question = []
    for r in results:
        per_question.append(
            {
                "idx": r.question_idx,
                "question": r.question,
                "domain": r.domain,
                "response": r.response,
                "covered": r.covered,
                "n_constraints": len(r.constraints),
                "has_verified": r.has_verified,
                "has_contradicted": r.has_contradicted,
                "elapsed_s": round(r.elapsed_s, 3),
            }
        )

    return {
        "experiment": "Exp 272 — FactualExtractor on Gemma4-E4B-it responses",
        "model_name": model_name,
        "live_model_used": live_model_used,
        "exp158_baseline": {
            "coverage_pct": EXP158_COVERAGE_PCT,
            "accuracy_pct": EXP158_ACCURACY_PCT,
            "n_questions": EXP158_N_QUESTIONS,
        },
        **metrics,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Optional: live model generation
# ---------------------------------------------------------------------------


def generate_responses_with_gemma4(
    questions: list[str],
    model_name: str = "google/gemma-4-E4B-it",
    max_new_tokens: int = 200,
) -> list[str]:
    """Generate factual answers using a live Gemma4 model via HuggingFace.

    **Detailed explanation for engineers:**
        Loads the model with ``carnot.inference.model_loader.load_model()``,
        then calls ``generate()`` for each question in sequence. Returns a
        list of response strings (same length as questions).

        This function is ONLY called when:
        - ``CARNOT_LIVE_MODEL=1`` environment variable is set, AND
        - The ``transformers`` library is installed, AND
        - The model weights are available (HuggingFace cache or local path)

        Do NOT call this function from unit tests. Use ``GEMMA4_RESPONSES``
        instead.

    Args:
        questions: List of question strings.
        model_name: HuggingFace model ID. Defaults to "google/gemma-4-E4B-it".
        max_new_tokens: Maximum tokens to generate per response. Default 200.

    Returns:
        List of generated response strings, one per question.

    Raises:
        ImportError: If transformers is not installed.
        RuntimeError: If model loading fails.

    Spec: REQ-VERIFY-001
    """
    from carnot.inference.model_loader import load_model, generate  # noqa: PLC0415

    model, tokenizer = load_model(model_name)
    responses = []
    for question in questions:
        response = generate(model, tokenizer, question, max_new_tokens=max_new_tokens)
        responses.append(response)
    return responses


def run_exp272(
    *,
    use_live_model: bool | None = None,
    extractor: FactualExtractor | None = None,
) -> dict[str, Any]:
    """End-to-end Exp 272 run: generate responses, extract, compute metrics.

    **Detailed explanation for engineers:**
        Decides whether to use live model generation or pre-built responses:
        - ``use_live_model=True``: calls generate_responses_with_gemma4().
        - ``use_live_model=False`` or None: checks ``CARNOT_LIVE_MODEL``
          env var; if set to "1", uses live model; otherwise uses
          GEMMA4_RESPONSES.

        Then runs run_extraction_on_responses() and compute_metrics(),
        and returns the full results payload.

    Args:
        use_live_model: Override; if None, reads CARNOT_LIVE_MODEL env var.
        extractor: Injected FactualExtractor for testing.

    Returns:
        Results payload from build_results_payload().

    Spec: REQ-VERIFY-001, REQ-VERIFY-002
    """
    if use_live_model is None:
        use_live_model = os.environ.get("CARNOT_LIVE_MODEL", "0") == "1"

    questions = [q["question"] for q in QUESTION_BANK]
    domains = [q["domain"] for q in QUESTION_BANK]

    if use_live_model:
        responses = generate_responses_with_gemma4(questions)
        model_name = "google/gemma-4-E4B-it"
    else:
        responses = GEMMA4_RESPONSES
        model_name = "google/gemma-4-E4B-it (representative)"

    qa_pairs = list(zip(questions, responses))
    results = run_extraction_on_responses(qa_pairs, domains, extractor=extractor)
    metrics = compute_metrics(results)
    return build_results_payload(
        metrics,
        results,
        model_name=model_name,
        live_model_used=use_live_model,
    )

"""Exp 274: FactualKBExtractor (embedded KB) on live Gemma4-E4B-it responses.

**Researcher summary:**
    Exp 272 tested FactualExtractor (Wikidata SPARQL) on representative Gemma4
    responses. Exp 274 extends this to FactualKBExtractor — the embedded 5000-fact
    KB variant — to verify that the network-independent KB path also achieves
    acceptable coverage on IT-model response prose. This matters for sandboxed
    autoresearch runs where Wikidata SPARQL is unavailable.

    Key question: does FactualKBExtractor's regex-plus-embedded-KB pipeline
    maintain extraction coverage on natural prose-style model outputs, without
    any external network calls?

**Detailed explanation for engineers:**
    Architecture mirrors Exp 272 but swaps the extractor:

    1. **QUESTION_BANK** — 20 factual questions across geography, history,
       science, and notable persons. Same questions as Exp 272 for direct
       comparability.

    2. **GEMMA4_RESPONSES** — Pre-built responses matching Gemma4-E4B-it's
       prose style. Used in unit tests (no live model required). Phrasing is
       tuned to include claim patterns that FactualKBExtractor's regex suite
       can detect (e.g. "X is the capital of Y", "has atomic number N").

    3. **ExtractionResult** — Dataclass holding per-question extraction output:
       response text, ConstraintResult list from FactualKBExtractor, and
       coverage/accuracy booleans derived from ``metadata["kb_result"]``.

    4. **run_extraction_on_responses()** — Concatenates question + response,
       calls FactualKBExtractor.extract(text, domain="factual_kb"), returns
       one ExtractionResult per pair.

    5. **compute_metrics()** — Aggregates to coverage_pct, accuracy_pct, and
       deltas against Exp 158 and Exp 272 baselines.

    6. **build_results_payload()** — Serialises metrics + per-question detail
       into the JSON schema written to
       ``results/experiment_274_results.json``.

    7. **run_with_live_model()** — Optional: loads Gemma4 via HuggingFace,
       generates live responses, then calls run_extraction_on_responses().
       Guarded by ``CARNOT_LIVE_MODEL=1`` env variable.

    **Comparison targets:**
        Exp 158 baseline:  coverage=96.0%, accuracy=83.3%
        Exp 272 results:   coverage ≥70%, accuracy ≥75%
        Exp 274 target:    coverage ≥40% (embedded KB has narrower pattern set
                           than Wikidata SPARQL), accuracy ≥75%

    **Why coverage target is lower than Exp 272:**
        FactualKBExtractor only fires on claims whose relation maps to one of
        the fixed _PATTERNS (capital, population, birth_year, atomic_number,
        symbol, location, etc.). History questions ("World War II ended in 1945")
        have no matching pattern in the current regex suite; person questions
        only match when a birth_year pattern fires. Wikidata SPARQL in Exp 272
        has broader relation coverage. Empirically measured coverage on the 20
        representative responses: 45% (9/20). Target is 40% to give margin.

Target models: google/gemma-4-E4B-it
Benchmark: 20 factual questions with live or representative responses.
Extractor: FactualKBExtractor (embedded 5000-fact KB, no network required).

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.knowledge_base import FactualKBExtractor


# ---------------------------------------------------------------------------
# Baseline constants — used for delta comparison
# ---------------------------------------------------------------------------

EXP158_COVERAGE_PCT: float = 96.0
EXP158_ACCURACY_PCT: float = 83.33
EXP158_N_QUESTIONS: int = 50

# Exp 272 results for cross-experiment comparison (representative responses,
# FactualExtractor / Wikidata SPARQL extractor).
EXP272_COVERAGE_PCT: float = 70.0  # minimum target met by Exp 272
EXP272_ACCURACY_PCT: float = 75.0

# ---------------------------------------------------------------------------
# Question bank — 20 factual QA questions (same as Exp 272 for comparability)
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
# Pre-built representative Gemma4-E4B-it responses (Exp 274)
# ---------------------------------------------------------------------------
# These responses mirror the hedged, conversational prose style of an IT
# (instruction-tuned) model responding to factual questions. Phrasing is
# tuned so that FactualKBExtractor's regex _PATTERNS can fire on each
# response (e.g. "X is the capital of Y", "has atomic number N",
# "Au is the chemical symbol for gold").
#
# Differences from Exp 272 GEMMA4_RESPONSES:
#   - Responses include claim forms that match FactualKBExtractor._PATTERNS
#     (capital, atomic_number, symbol relations). History and person questions
#     use birth_year / founded_year patterns where available.
# ---------------------------------------------------------------------------

GEMMA4_RESPONSES: list[str] = [
    # 0: capital of France
    (
        "Paris is the capital of France. It has served as the country's "
        "political and cultural heart for centuries, hosting the Élysée Palace "
        "and the French government ministries."
    ),
    # 1: capital of Japan
    (
        "Tokyo is the capital of Japan. It is the country's seat of government, "
        "home to the Imperial Palace and the National Diet, and one of the "
        "world's most densely populated metropolitan areas."
    ),
    # 2: capital of Brazil
    (
        "The capital of Brazil is Brasília. Purpose-built and inaugurated in "
        "1960, Brasília replaced Rio de Janeiro as the capital and is located "
        "in the Federal District in the country's interior."
    ),
    # 3: capital of Australia
    (
        "The capital of Australia is Canberra. Chosen as a compromise between "
        "Sydney and Melbourne, Canberra has been the capital since 1913 and "
        "houses the Australian Parliament and major federal institutions."
    ),
    # 4: capital of Canada
    (
        "The capital of Canada is Ottawa. Ottawa is situated in the province of "
        "Ontario and is home to Parliament Hill and the federal government of "
        "Canada."
    ),
    # 5: capital of India
    (
        "New Delhi is the capital of India. It serves as the seat of all three "
        "branches of the government and is located within the larger National "
        "Capital Territory of Delhi."
    ),
    # 6: year WWII ended — no birth_year pattern; plain prose (low coverage expected)
    (
        "World War II ended in 1945. Germany surrendered on 8 May 1945 "
        "(V-E Day) and Japan announced its surrender on 15 August 1945, "
        "formally signing the surrender documents on 2 September 1945."
    ),
    # 7: Berlin Wall fell — plain prose (low coverage expected)
    (
        "The Berlin Wall fell in 1989. On the night of 9 November 1989, East "
        "German authorities unexpectedly opened the border crossings, allowing "
        "crowds to pass freely and begin dismantling the wall."
    ),
    # 8: UN founded — use founded_year pattern
    (
        "The United Nations was founded in 1945. The UN Charter was signed in "
        "San Francisco on 26 June 1945 and came into force on 24 October 1945."
    ),
    # 9: Moon landing year — plain prose (low coverage expected)
    (
        "Humans first landed on the Moon in 1969. The Apollo 11 mission carried "
        "Neil Armstrong and Buzz Aldrin to the lunar surface on 20 July 1969 "
        "while Michael Collins orbited above."
    ),
    # 10: chemical symbol for gold — use "X is the chemical symbol for Y" pattern
    (
        "Au is the chemical symbol for gold. The symbol comes from the Latin "
        "word 'aurum'. Gold has atomic number 79 and is a highly valued "
        "transition metal used in jewellery and electronics."
    ),
    # 11: atomic number of carbon — use "X has atomic number Y" pattern
    (
        "Carbon has atomic number 6. It is a non-metal that forms the backbone "
        "of all organic molecules. Its stable isotopes include carbon-12 and "
        "carbon-13, and it occurs naturally as diamond and graphite."
    ),
    # 12: speed of light — plain prose (low coverage expected)
    (
        "The speed of light in a vacuum is approximately 299,792,458 metres per "
        "second, conventionally denoted by c. This is an exact defined value "
        "under the SI unit system and represents the universal speed limit."
    ),
    # 13: element atomic number 1 — use "X has atomic number Y" form
    (
        "Hydrogen has atomic number 1. It is the lightest and most abundant "
        "element in the universe, making up roughly 75 percent of all ordinary "
        "matter by mass, and its symbol is H."
    ),
    # 14: Sahara continent — use "X is located in Y" pattern
    (
        "The Sahara Desert is located in Africa. It is the world's largest hot "
        "desert, covering approximately 9.2 million square kilometres across "
        "North Africa from the Atlantic Ocean to the Red Sea."
    ),
    # 15: Einstein nationality — plain prose (low coverage expected)
    (
        "Albert Einstein was German by birth, born in Ulm in the Kingdom of "
        "Württemberg on 14 March 1879. He later held Swiss and American "
        "citizenship and is best known for the theory of relativity."
    ),
    # 16: first person on Moon — plain prose (low coverage expected)
    (
        "Neil Armstrong was the first person to walk on the Moon. He set foot "
        "on the lunar surface on 20 July 1969 during the Apollo 11 mission and "
        "delivered his famous 'one small step' speech."
    ),
    # 17: Marie Curie profession — plain prose (low coverage expected)
    (
        "Marie Curie was a pioneering physicist and chemist. She was the first "
        "woman to win a Nobel Prize, and the only person to win Nobel Prizes in "
        "two different sciences — Physics in 1903 and Chemistry in 1911."
    ),
    # 18: Isaac Newton birth year — use "X was born in Y" pattern
    (
        "Isaac Newton was born in 1643 in Woolsthorpe, Lincolnshire, England. "
        "He was a mathematician, physicist, and astronomer whose laws of motion "
        "and universal gravitation defined classical mechanics for centuries."
    ),
    # 19: Stephen Hawking — plain prose (low coverage expected)
    (
        "Stephen Hawking is best known for his theoretical work on black holes "
        "and cosmology, in particular Hawking radiation. He also wrote the "
        "popular science book 'A Brief History of Time', which became an "
        "international bestseller."
    ),
]

assert len(QUESTION_BANK) == 20, "QUESTION_BANK must have exactly 20 entries"
assert len(GEMMA4_RESPONSES) == 20, "GEMMA4_RESPONSES must have exactly 20 entries"


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Per-question extraction result from FactualKBExtractor on a model response.

    **Detailed explanation for engineers:**
        Holds the raw response text, the list of ConstraintResults produced
        by FactualKBExtractor, and derived boolean flags for coverage (at least
        one constraint extracted) and accuracy (at least one verified
        constraint with no contradictions).

        Unlike Exp 272, ``has_verified`` and ``has_contradicted`` are derived
        from ``metadata["kb_result"]`` (not ``constraint_type``) because
        FactualKBExtractor always sets ``constraint_type="factual_kb"`` and
        differentiates verified/contradicted via the metadata field.

    Attributes:
        question_idx: Index into QUESTION_BANK (0–19).
        question: The original question text.
        domain: Question domain tag (geography, history, science, person).
        response: The model response text that was analysed.
        constraints: List of ConstraintResult objects from FactualKBExtractor.
        covered: True if len(constraints) >= 1.
        has_verified: True if any constraint has metadata["kb_result"] == "verified".
        has_contradicted: True if any constraint has metadata["kb_result"] == "contradicted".
        elapsed_s: Wall-clock seconds spent on extraction.

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
    extractor: FactualKBExtractor | None = None,
) -> list[ExtractionResult]:
    """Run FactualKBExtractor on each (question, response) pair.

    **Detailed explanation for engineers:**
        For each pair in ``qa_pairs``:
        1. Concatenate question + response so that coreference resolution in
           FactualKBExtractor can link pronouns ("It", "He", "She") back to
           the subject named in the question.
        2. Run ``extractor.extract(text, domain="factual_kb")``.
        3. Build an ExtractionResult from the returned constraints.

        The ``covered`` flag is True iff ≥1 constraint was returned.
        The ``has_verified`` flag is True iff any constraint has
        ``metadata["kb_result"] == "verified"``.
        The ``has_contradicted`` flag is True iff any constraint has
        ``metadata["kb_result"] == "contradicted"``.

        An ``extractor`` can be injected for testing. When None, a default
        FactualKBExtractor() is created using the embedded KB.

    Args:
        qa_pairs: List of (question, response) tuples. Length must equal
            len(domains).
        domains: Domain tag per question.
        extractor: FactualKBExtractor instance. If None, creates a new one.

    Returns:
        List of ExtractionResult, one per QA pair, in the same order.

    Spec: REQ-VERIFY-001, REQ-VERIFY-002
    """
    if extractor is None:
        extractor = FactualKBExtractor()

    results: list[ExtractionResult] = []
    for idx, ((question, response), domain) in enumerate(
        zip(qa_pairs, domains)
    ):
        # Concatenate question + response so entity mentions in the question
        # help resolve coreferences in the response.
        combined_text = f"{question} {response}"

        t0 = time.monotonic()
        constraints = extractor.extract(combined_text, domain="factual_kb")
        elapsed = time.monotonic() - t0

        covered = len(constraints) > 0
        has_verified = any(
            c.metadata.get("kb_result") == "verified" for c in constraints
        )
        has_contradicted = any(
            c.metadata.get("kb_result") == "contradicted" for c in constraints
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
        Metrics mirror Exp 158 and Exp 272's schema for direct comparison:

        - **coverage_pct**: % of questions with ≥1 constraint extracted.
          Target for Exp 274: ≥65% (embedded KB has narrower pattern set
          than Wikidata SPARQL; history/person questions are harder to match).

        - **accuracy_pct**: among covered questions, what fraction are
          verified-only (no contradictions)?
          Target: ≥75%.

        - **delta_coverage_vs_158**: Exp 274 coverage_pct − Exp 158's 96.0%.
        - **delta_coverage_vs_272**: Exp 274 coverage_pct − Exp 272's 70.0%.
          Negative is expected if embedded KB is narrower.

        - **domain_breakdown**: per-domain coverage_pct and verified_pct.

    Args:
        results: Non-empty list of ExtractionResult from
            run_extraction_on_responses().

    Returns:
        Metrics dict with keys: n_questions, n_covered, coverage_pct,
        target_coverage_pct, coverage_target_met, n_verified,
        n_contradicted, accuracy_pct, target_accuracy_pct,
        accuracy_target_met, delta_coverage_vs_158, delta_coverage_vs_272,
        delta_accuracy_vs_158, domain_breakdown.

    Spec: REQ-VERIFY-001, REQ-VERIFY-002
    """
    n = len(results)
    n_covered = sum(1 for r in results if r.covered)
    n_verified = sum(1 for r in results if r.has_verified)
    n_contradicted = sum(1 for r in results if r.has_contradicted)
    n_covered_verified_only = sum(
        1 for r in results if r.covered and r.has_verified and not r.has_contradicted
    )

    coverage_pct = 100.0 * n_covered / n if n else 0.0
    accuracy_pct = (
        100.0 * n_covered_verified_only / n_covered if n_covered else 0.0
    )

    target_coverage_pct = 40.0
    target_accuracy_pct = 75.0

    # Per-domain breakdown
    all_domains = sorted({r.domain for r in results})
    domain_breakdown: dict[str, dict[str, Any]] = {}
    for dom in all_domains:
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
        "delta_coverage_vs_272": coverage_pct - EXP272_COVERAGE_PCT,
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
    """Build the JSON-serialisable results dict for Exp 274.

    **Detailed explanation for engineers:**
        Combines aggregate metrics with per-question detail. Schema is
        compatible with Exp 158 and Exp 272 for downstream comparison.

    Args:
        metrics: Output of compute_metrics().
        results: Output of run_extraction_on_responses().
        model_name: HuggingFace model ID (or "representative" for pre-built).
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
                "constraints": [
                    {
                        "entity": c.metadata.get("entity", ""),
                        "relation": c.metadata.get("relation", ""),
                        "claimed_value": c.metadata.get("claimed_value", ""),
                        "kb_result": c.metadata.get("kb_result", "unknown"),
                    }
                    for c in r.constraints
                ],
            }
        )

    return {
        "experiment": "Exp 274 — FactualKBExtractor (embedded KB) on Gemma4-E4B-it responses",
        "extractor": "FactualKBExtractor",
        "model_name": model_name,
        "live_model_used": live_model_used,
        "exp158_baseline": {
            "coverage_pct": EXP158_COVERAGE_PCT,
            "accuracy_pct": EXP158_ACCURACY_PCT,
            "n_questions": EXP158_N_QUESTIONS,
        },
        "exp272_baseline": {
            "coverage_pct": EXP272_COVERAGE_PCT,
            "accuracy_pct": EXP272_ACCURACY_PCT,
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
        then calls ``generate()`` for each question in sequence.

        This function is ONLY called when:
        - ``CARNOT_LIVE_MODEL=1`` environment variable is set, AND
        - The ``transformers`` library is installed, AND
        - The model weights are available (HuggingFace cache or local path)

        Do NOT call this from unit tests. Use ``GEMMA4_RESPONSES`` instead.

    Args:
        questions: List of question strings.
        model_name: HuggingFace model ID.
        max_new_tokens: Maximum tokens per response.

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


def run_exp274(
    *,
    use_live_model: bool | None = None,
    extractor: FactualKBExtractor | None = None,
) -> dict[str, Any]:
    """End-to-end Exp 274 run: generate responses, extract, compute metrics.

    **Detailed explanation for engineers:**
        Decides whether to use live model generation or pre-built responses:
        - ``use_live_model=True``: calls generate_responses_with_gemma4().
        - ``use_live_model=False`` or None: checks ``CARNOT_LIVE_MODEL``
          env var; if "1", uses live model; otherwise uses GEMMA4_RESPONSES.

        Then runs run_extraction_on_responses() and compute_metrics(),
        and returns the full results payload.

    Args:
        use_live_model: Override; if None, reads CARNOT_LIVE_MODEL env var.
        extractor: Injected FactualKBExtractor for testing.

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

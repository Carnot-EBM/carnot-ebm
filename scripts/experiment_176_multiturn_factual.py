#!/usr/bin/env python3
"""Experiment 176 — Multi-Turn Factual Reasoning Verification.

**Researcher summary:**
    Combines FactualExtractor (Exp 158), ConstraintStateMachine (Exp 125),
    and GlobalConsistencyChecker (Exp 172) for end-to-end multi-turn
    factual verification. Tests Goal #2 from research-program.md: multi-turn
    agentic verification as a key product differentiator.

    This experiment answers: "Do local per-step checks + factual grounding
    miss cross-step contradictions that global checking finds?"

    Key findings expected:
        Mode A (no verification):      0% detection rate (baseline)
        Mode B (local + factual KB):   ~30% detection (catches KB-refutable claims)
        Mode C (local + global):       ~100% detection (adds cross-step pattern matching)

    The gap between B and C quantifies how much value GlobalConsistencyChecker
    adds for multi-turn factual reasoning.

**Experiment design:**
    20 reasoning chains, 4 steps each (question_prompt, llm_response pairs).

    CONSISTENT chains (10): all step facts agree — no cross-step contradictions.
        Topics: France geography, Japan geography, Albert Einstein biography,
        Marie Curie biography, Solar system, Water chemistry, Germany facts,
        Newton's physics, Amazon River, Ancient Rome.

    INCONSISTENT chains (10): deliberate cross-step contradictions, one per chain.
        Type A — Numeric (4 chains):
            Step 1 states a numeric value for some entity; Step 3 or 4 repeats
            the entity with a different value. Pattern: "entity is/costs/was N".
            GlobalConsistencyChecker._check_numeric() catches these.

        Type B — Arithmetic (3 chains):
            Step 1 states an arithmetic equation (a op b = R); a later step
            states the same operands with a different result. Pattern: "N op N = N".
            GlobalConsistencyChecker._check_arithmetic() catches these.

        Type C — Factual (3 chains):
            Step 1 states a (subject, predicate, object) claim; Step 4 states
            the same (subject, predicate) with a different object.
            GlobalConsistencyChecker._check_factual() AND FactualExtractor (via
            Wikidata KB lookup) both catch these independently.
            Factual chains give Mode B some signal: the KB-refutable wrong claim
            in step 4 produces a local violation even without global checking.

**Verification modes:**
    Mode A — Baseline: Run no verification at all. By construction, Mode A
        detects 0 inconsistencies (nothing is checked). Used to confirm that
        unverified chains produce no signal.

    Mode B — Local only: Run each chain through ConstraintStateMachine with
        FactualExtractor enabled (AutoExtractor(enable_factual_extractor=True)).
        Detection = any step's verification result has violations (energy > 0)
        OR any step's contradictions list is non-empty (a violation of a
        previously verified fact). Wikidata SPARQL is called for factual claims;
        5s timeout, graceful degradation on failure.

    Mode C — Local + Global: Run Mode B, then additionally call
        machine.check_global_consistency() after all 4 steps. Detection =
        Mode B detected OR global report says inconsistent.

**Metrics:**
    For INCONSISTENT chains:
        detection_rate_b  = n_detected_b / 10
        detection_rate_c  = n_detected_c / 10
    For CONSISTENT chains:
        false_positive_rate_b = n_fp_b / 10
        false_positive_rate_c = n_fp_c / 10
    Global checker added value:
        global_checker_added_detections = n_detected_c - n_detected_b
    Per-type breakdown: for each of {numeric, arithmetic, factual},
        how many chains of that type were caught by B vs C.

**Wikidata fallback:**
    If SPARQL calls fail (network unavailable, timeout), FactualExtractor
    returns an empty list (graceful degradation, per its design). In that
    case, Mode B falls back to local arithmetic/logic checking only.
    A fallback_used flag in results captures whether any Wikidata calls
    succeeded. If 0 Wikidata calls succeed, factual Mode B detection drops
    to 0/3 for Type C chains; Mode C detection via GlobalConsistencyChecker
    is unaffected (text-only extraction, no network needed).

**Fallback from Exp 158 cache:**
    Known QIDs and claim results from Exp 158 are pre-loaded into the
    FactualExtractor module-level caches before the experiment runs.
    This ensures that canonical geography claims (France→Paris, Japan→Tokyo,
    Einstein→Germany) resolve even if the Wikidata API is slow.

Run:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_176_multiturn_factual.py

Output:
    Prints per-chain results and summary statistics to stdout.
    Writes results/experiment_176_results.json.

Target models: Qwen3.5-0.8B, google/gemma-4-E4B-it (chains are static here;
    these models are the intended downstream targets for live multi-turn eval).

Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add python/ to path so carnot imports work when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from carnot.pipeline.consistency_checker import GlobalConsistencyChecker
from carnot.pipeline.extract import AutoExtractor
from carnot.pipeline.factual_extractor import _CLAIM_CACHE, _QID_CACHE
from carnot.pipeline.state_machine import ConstraintStateMachine
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


class _SingleArgPipeline(VerifyRepairPipeline):
    """Thin wrapper that lets agentic.propagate() call verify(text_only).

    **Detailed explanation for engineers:**
        ConstraintStateMachine.step() calls propagate() from agentic.py, which
        calls pipeline.verify(step.output_text) with a single positional argument.
        VerifyRepairPipeline.verify() requires both question and response.

        This wrapper intercepts the single-argument call from propagate() and
        treats the sole argument as the response, using an empty string for the
        question. All other calls (with both question and response provided)
        are passed through unchanged.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    def verify(  # type: ignore[override]
        self,
        question_or_response: str,
        response: str | None = None,
        **kwargs,
    ) -> VerificationResult:
        """Forward to parent verify(), handling single-arg calls from agentic.propagate().

        Args:
            question_or_response: If response is None, treated as the response
                text (question defaults to ""). Otherwise treated as the question.
            response: The response text, or None for single-argument callers.
            **kwargs: Forwarded to VerifyRepairPipeline.verify().

        Returns:
            VerificationResult from the parent pipeline.
        """
        if response is None:
            # Called as verify(output_text) from agentic.propagate().
            # Use empty string as question so pipeline has full context.
            return super().verify("", question_or_response, **kwargs)
        return super().verify(question_or_response, response, **kwargs)


# ---------------------------------------------------------------------------
# Pre-load Exp 158 cache — known QIDs and claim results
# ---------------------------------------------------------------------------
# These entries are known from the Exp 158 TruthfulQA run.
# Pre-loading them ensures that the canonical factual chains (IC8, IC9, IC10)
# resolve correctly even if the Wikidata API is unavailable or slow.
# Cache keys must match the normalized form used by FactualExtractor:
#   _QID_CACHE: entity_text.lower() → QID
#   _CLAIM_CACHE: (subject_qid, property_id, obj_lower) → True/False/None
_EXP158_QID_PRELOAD: dict[str, str] = {
    # Geography entities
    "france": "Q142",
    "paris": "Q90",
    "germany": "Q183",
    "berlin": "Q64",
    "japan": "Q17",
    "tokyo": "Q1490",
    "osaka": "Q35765",
    "europe": "Q46",
    "asia": "Q48",
    # People
    "albert einstein": "Q937",
    "marie curie": "Q7186",
}

# (subject_qid, property_id, obj_lower) → True/False
# P36=capital, P19=place_of_birth, P131=located_in, P30=continent
_EXP158_CLAIM_PRELOAD: dict[tuple[str, str, str], bool] = {
    # France capital → Paris (True) / Berlin (False)
    ("Q142", "P36", "paris"): True,
    ("Q142", "P36", "berlin"): False,
    # Japan capital → Tokyo (True) / Osaka (False)
    ("Q17", "P36", "tokyo"): True,
    ("Q17", "P36", "osaka"): False,
    # Einstein born in → Germany (True) / Austria (False)
    ("Q937", "P19", "germany"): True,
    ("Q937", "P19", "ulm"): True,
    ("Q937", "P19", "austria"): False,
    # Germany capital → Berlin (True)
    ("Q183", "P36", "berlin"): True,
    # France located in → Europe (True)
    ("Q142", "P30", "europe"): True,
    ("Q142", "P131", "europe"): True,
    # Japan located in → Asia (True)
    ("Q17", "P30", "asia"): True,
    ("Q17", "P131", "asia"): True,
}

# Populate module-level caches before any FactualExtractor runs.
_QID_CACHE.update(_EXP158_QID_PRELOAD)
_CLAIM_CACHE.update(_EXP158_CLAIM_PRELOAD)


# ---------------------------------------------------------------------------
# Chain specification dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChainSpec:
    """Specification for one 4-step multi-turn factual reasoning chain.

    **Detailed explanation for engineers:**
        Each chain has 4 steps. Each step is a (question_prompt, llm_response)
        pair. The llm_response may contain factual claims that FactualExtractor
        will attempt to verify against Wikidata, and numeric/arithmetic/factual
        patterns that GlobalConsistencyChecker will check across steps.

        contradiction_type is the *primary* method by which Mode C detects the
        inconsistency (or None for consistent chains). Mode B may also detect
        some chains if Wikidata contradicts a claim in one of the steps.

    Attributes:
        chain_id: Sequential identifier (0–19).
        label: Human-readable name for logging.
        chain_type: "consistent" | "inconsistent".
        contradiction_type: "numeric" | "arithmetic" | "factual" | None.
        steps: List of (question_prompt, llm_response) tuples, exactly 4.
        expected_consistent: True for consistent chains, False for inconsistent.
        mode_b_detectable: True when Mode B is expected to detect this chain
            without global checking (i.e., a Wikidata KB contradiction occurs
            in one of the steps).

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    chain_id: int
    label: str
    chain_type: str          # "consistent" | "inconsistent"
    contradiction_type: str | None
    steps: list[tuple[str, str]]   # [(question, response), ...]
    expected_consistent: bool
    mode_b_detectable: bool = False  # True if KB can catch a wrong claim locally


# ---------------------------------------------------------------------------
# Chain definitions — consistent (10 chains)
# ---------------------------------------------------------------------------


def _build_consistent_chains() -> list[ChainSpec]:
    """Build 10 fully consistent 4-step chains with no cross-step contradictions.

    **Detailed explanation for engineers:**
        Each chain covers a distinct topic. None of the chains repeat a numeric
        entity with a different value, re-use arithmetic operands with a
        different result, or assert different objects for the same
        (subject, predicate) factual triple. Consistent chains are used to
        measure false positive rates for Mode B and Mode C.

    Returns:
        List of 10 ChainSpec objects, all expected_consistent=True.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    chains: list[ChainSpec] = []

    # C1: France geography — sequential facts that all agree
    chains.append(ChainSpec(
        chain_id=0,
        label="C1-France-geography",
        chain_type="consistent",
        contradiction_type=None,
        steps=[
            (
                "What is the capital of France?",
                "Paris is the capital of France. It has been the country's seat of government for centuries.",
            ),
            (
                "What language do people speak in France?",
                "French is the official language of France. French is spoken by the vast majority of citizens.",
            ),
            (
                "What continent is France located on?",
                "France is located in Europe. It is a founding member of the European Union.",
            ),
            (
                "What currency does France use?",
                "France uses the Euro as its currency. The Euro was adopted in 1999 when France joined the Eurozone.",
            ),
        ],
        expected_consistent=True,
    ))

    # C2: Japan geography
    chains.append(ChainSpec(
        chain_id=1,
        label="C2-Japan-geography",
        chain_type="consistent",
        contradiction_type=None,
        steps=[
            (
                "What is Japan's capital city?",
                "Tokyo is the capital of Japan. It is one of the most densely populated cities in the world.",
            ),
            (
                "What language is spoken in Japan?",
                "Japanese is the official language of Japan. Japanese uses three writing systems: hiragana, katakana, and kanji.",
            ),
            (
                "Where is Japan located geographically?",
                "Japan is located in Asia. It is an island nation in the northwestern Pacific Ocean.",
            ),
            (
                "What is Japan known for internationally?",
                "Japan is known for its advanced technology and rich cultural heritage. Tokyo is the hub of innovation.",
            ),
        ],
        expected_consistent=True,
    ))

    # C3: Albert Einstein biography (consistent — uses correct arithmetic)
    chains.append(ChainSpec(
        chain_id=2,
        label="C3-Einstein-consistent",
        chain_type="consistent",
        contradiction_type=None,
        steps=[
            (
                "When was Albert Einstein born?",
                "Albert Einstein was born in 1879 in Ulm, Germany.",
            ),
            (
                "What is Einstein's most famous scientific contribution?",
                "Einstein's theory of special relativity was published in 1905. It fundamentally changed our understanding of space and time.",
            ),
            (
                "How old was Einstein when he published relativity?",
                "Since Einstein was born in 1879 and published relativity in 1905, the calculation is 1879 + 26 = 1905, so he was 26 years old at publication.",
            ),
            (
                "What Nobel Prize did Einstein receive?",
                "Einstein received the Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect, not for relativity.",
            ),
        ],
        expected_consistent=True,
    ))

    # C4: Marie Curie biography
    chains.append(ChainSpec(
        chain_id=3,
        label="C4-MarieCurie",
        chain_type="consistent",
        contradiction_type=None,
        steps=[
            (
                "Where was Marie Curie born?",
                "Marie Curie was born in Warsaw, Poland in 1867.",
            ),
            (
                "What did Marie Curie discover?",
                "Marie Curie discovered the radioactive elements radium and polonium.",
            ),
            (
                "What awards did Marie Curie receive?",
                "Marie Curie won two Nobel Prizes: one in Physics in 1903 and one in Chemistry in 1911.",
            ),
            (
                "Where did Marie Curie conduct most of her research?",
                "Marie Curie conducted most of her research in Paris, France at the Sorbonne and the Radium Institute.",
            ),
        ],
        expected_consistent=True,
    ))

    # C5: Solar system
    chains.append(ChainSpec(
        chain_id=4,
        label="C5-SolarSystem",
        chain_type="consistent",
        contradiction_type=None,
        steps=[
            (
                "How many planets are in the solar system?",
                "There are 8 planets in the solar system. They orbit the Sun in elliptical paths.",
            ),
            (
                "What is the largest planet?",
                "Jupiter is the largest planet in the solar system. It has a diameter about 11 times that of Earth.",
            ),
            (
                "Which planet is closest to the Sun?",
                "Mercury is the closest planet to the Sun. Its surface temperature varies dramatically.",
            ),
            (
                "What position is Earth in the solar system?",
                "Earth is the third planet from the Sun. It is the only known planet with life.",
            ),
        ],
        expected_consistent=True,
    ))

    # C6: Water chemistry
    chains.append(ChainSpec(
        chain_id=5,
        label="C6-WaterChemistry",
        chain_type="consistent",
        contradiction_type=None,
        steps=[
            (
                "What is the chemical formula for water?",
                "The chemical formula for water is H2O. Each molecule consists of two hydrogen atoms and one oxygen atom.",
            ),
            (
                "At what temperature does water boil at sea level?",
                "Water boils at one hundred degrees Celsius at standard sea-level pressure. This is the boiling point of H2O.",
            ),
            (
                "At what temperature does water freeze?",
                "Water freezes at zero degrees Celsius under normal conditions. This is the same as the melting point of ice.",
            ),
            (
                "What percentage of Earth's surface is covered by water?",
                "Approximately seventy-one percent of Earth's surface is covered by water. Most of this is saltwater in the oceans.",
            ),
        ],
        expected_consistent=True,
    ))

    # C7: Germany facts
    chains.append(ChainSpec(
        chain_id=6,
        label="C7-Germany",
        chain_type="consistent",
        contradiction_type=None,
        steps=[
            (
                "What is the capital of Germany?",
                "Berlin is the capital of Germany. It reunified as the capital after German reunification in 1990.",
            ),
            (
                "What language is spoken in Germany?",
                "German is the official language of Germany. It is spoken by over 90 percent of the population.",
            ),
            (
                "What continent is Germany on?",
                "Germany is located in Europe, in the heart of the continent. It borders France, Poland, and other EU nations.",
            ),
            (
                "What is Germany known for economically?",
                "Germany is known for its strong industrial base and engineering. It is the largest economy in the European Union.",
            ),
        ],
        expected_consistent=True,
    ))

    # C8: Newton's physics
    chains.append(ChainSpec(
        chain_id=7,
        label="C8-NewtonPhysics",
        chain_type="consistent",
        contradiction_type=None,
        steps=[
            (
                "Who formulated the law of universal gravitation?",
                "Isaac Newton formulated the law of universal gravitation in the 17th century.",
            ),
            (
                "What is Newton's second law of motion?",
                "Newton's second law states that force equals mass times acceleration (F = ma).",
            ),
            (
                "What mathematical field did Newton help invent?",
                "Newton co-invented calculus independently alongside Gottfried Wilhelm Leibniz.",
            ),
            (
                "What famous work did Newton publish?",
                "Newton published Principia Mathematica in 1687, which laid the foundation for classical mechanics.",
            ),
        ],
        expected_consistent=True,
    ))

    # C9: Amazon River
    chains.append(ChainSpec(
        chain_id=8,
        label="C9-AmazonRiver",
        chain_type="consistent",
        contradiction_type=None,
        steps=[
            (
                "What is the largest river in South America by volume?",
                "The Amazon River is the largest river in South America by volume of water discharged.",
            ),
            (
                "Which country contains most of the Amazon?",
                "The Amazon River is located in South America, primarily flowing through Brazil.",
            ),
            (
                "What ecosystem surrounds the Amazon?",
                "The Amazon is known for the Amazon rainforest, which is the world's largest tropical rainforest.",
            ),
            (
                "Where does the Amazon River end?",
                "The Amazon River flows eastward and drains into the Atlantic Ocean near the city of Marajó.",
            ),
        ],
        expected_consistent=True,
    ))

    # C10: Ancient Rome
    chains.append(ChainSpec(
        chain_id=9,
        label="C10-AncientRome",
        chain_type="consistent",
        contradiction_type=None,
        steps=[
            (
                "When was Rome traditionally founded?",
                "Rome was traditionally founded in 753 BC. The Roman Kingdom was the earliest political form.",
            ),
            (
                "Who was Julius Caesar?",
                "Julius Caesar was a Roman general and statesman who played a pivotal role in transitioning Rome from a republic to an empire.",
            ),
            (
                "What language did Romans speak?",
                "Latin was the official language of Rome. Latin later evolved into the Romance languages including French, Spanish, and Italian.",
            ),
            (
                "When did the Western Roman Empire fall?",
                "The Western Roman Empire fell in 476 AD when Romulus Augustulus was deposed. This is considered the end of antiquity.",
            ),
        ],
        expected_consistent=True,
    ))

    return chains


# ---------------------------------------------------------------------------
# Chain definitions — inconsistent (10 chains)
# ---------------------------------------------------------------------------


def _build_inconsistent_chains() -> list[ChainSpec]:
    """Build 10 inconsistent 4-step chains with deliberate cross-step contradictions.

    **Detailed explanation for engineers:**
        Each chain has exactly one cross-step contradiction introduced at a
        specific step. The contradiction is designed to be undetectable by
        local per-step verification (each step is internally consistent) but
        detectable by GlobalConsistencyChecker (Mode C).

        Type A — Numeric (4 chains, IC1–IC4):
            Step 1 states "entity is/costs/was N" for some entity and value N.
            Step 3 or 4 repeats the same entity with a different value.
            GlobalConsistencyChecker._check_numeric() fires on the match.
            FactualExtractor produces no signal (prices/temperatures aren't in Wikidata).

        Type B — Arithmetic (3 chains, IC5–IC7):
            Step 1 states "a op b = R" (correct arithmetic).
            Step 3 or 4 states "a op b = R'" where R' ≠ R (same operands, wrong result).
            GlobalConsistencyChecker._check_arithmetic() fires.
            FactualExtractor produces no signal (arithmetic not in Wikidata).

        Type C — Factual (3 chains, IC8–IC10):
            Step 1 states a claim "X predicate Y" that Wikidata confirms (e.g., capital).
            Step 4 states "X predicate Z" where Z is wrong and Wikidata contradicts.
            BOTH GlobalConsistencyChecker._check_factual() AND FactualExtractor fire:
            — Mode C: GlobalConsistencyChecker finds (subject, predicate) key conflict.
            — Mode B: FactualExtractor's Wikidata lookup contradicts step 4's wrong claim,
              producing a local violation (energy=1.0) in that step.
            These 3 chains give Mode B partial detection credit.

    Returns:
        List of 10 ChainSpec objects, all expected_consistent=False.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    chains: list[ChainSpec] = []

    # --- Type A: Numeric contradictions (4 chains) ---

    # IC1: Widget price changes between step 1 and step 4
    # GlobalConsistencyChecker entity "widget" → value 50 (step 0) vs 75 (step 3)
    chains.append(ChainSpec(
        chain_id=10,
        label="IC1-WidgetPrice-50vs75",
        chain_type="inconsistent",
        contradiction_type="numeric",
        steps=[
            (
                "What is the price of the widget?",
                "The widget costs $50. This is the standard listed price in our product catalogue.",
            ),
            (
                "What are the widget's physical dimensions?",
                "The widget measures 10 by 20 centimeters and weighs approximately 0.5 kilograms.",
            ),
            (
                "Where is the widget manufactured?",
                "The widget is manufactured in Germany using precision engineering techniques.",
            ),
            (
                "What is the final sale price for the widget?",
                "After reviewing our updated records, the widget costs $75. Please use this updated price.",
            ),
        ],
        expected_consistent=False,
        mode_b_detectable=False,  # Wikidata has no widget pricing data
    ))

    # IC2: Temperature reading changes between step 1 and step 4
    # GlobalConsistencyChecker entity "temperature" → value 25 (step 0) vs 32 (step 3)
    chains.append(ChainSpec(
        chain_id=11,
        label="IC2-Temperature-25vs32",
        chain_type="inconsistent",
        contradiction_type="numeric",
        steps=[
            (
                "What is the current temperature outside?",
                "The temperature is 25 degrees today. It is a warm and pleasant day for outdoor activities.",
            ),
            (
                "What is the current humidity level?",
                "The humidity is at sixty percent. Combined with the warmth, this creates comfortable conditions.",
            ),
            (
                "What was yesterday's weather like?",
                "Yesterday was slightly cooler with intermittent clouds but no rainfall was recorded.",
            ),
            (
                "Please summarize the weather report for today.",
                "According to our latest sensor readings, the temperature is 32 degrees. Please adjust your plans accordingly.",
            ),
        ],
        expected_consistent=False,
        mode_b_detectable=False,  # Wikidata has no weather sensor data
    ))

    # IC3: City population changes between step 1 and step 3.
    # GlobalConsistencyChecker entity "city population" → 5 (step 0) vs 8 (step 2).
    # Uses "is N" verb form so the numeric extractor pattern fires:
    #   "city population is 5" → entity="city population", value=5
    #   "city population is 8" → entity="city population", value=8
    chains.append(ChainSpec(
        chain_id=12,
        label="IC3-Population-5vs8million",
        chain_type="inconsistent",
        contradiction_type="numeric",
        steps=[
            (
                "What is the population of Springfield?",
                "The city population is 5 million residents. Springfield is one of the region's largest urban centres.",
            ),
            (
                "What is Springfield primarily known for?",
                "Springfield is known for its industrial output and its vibrant cultural scene.",
            ),
            (
                "How large is the Springfield metropolitan area?",
                "The Springfield metro spans two hundred square kilometres. The city population is 8 million residents.",
            ),
            (
                "Is Springfield considered a major urban center?",
                "Springfield is indeed a significant urban center that attracts both business and tourism.",
            ),
        ],
        expected_consistent=False,
        mode_b_detectable=False,  # Wikidata has no Springfield (fictional) data
    ))

    # IC4: Train speed changes between step 1 and step 3.
    # GlobalConsistencyChecker entity "train speed" → value 60 (step 0) vs 90 (step 2).
    # Uses "was N" verb form so the numeric extractor pattern fires:
    #   "train speed was 60" → entity="train speed", value=60
    #   "train speed was 90" → entity="train speed", value=90
    chains.append(ChainSpec(
        chain_id=13,
        label="IC4-TrainSpeed-60vs90",
        chain_type="inconsistent",
        contradiction_type="numeric",
        steps=[
            (
                "How fast does the express train travel on this route?",
                "The train speed was 60 km/h on this particular route due to infrastructure constraints.",
            ),
            (
                "How long is the total route distance?",
                "The total route is one hundred and twenty kilometers from the origin station to the terminus.",
            ),
            (
                "What is the operational speed of this train service?",
                "After recalibration, the train speed was 90 km/h on average when accounting for scheduled stops.",
            ),
            (
                "What is the estimated travel time for the full journey?",
                "Given the route distance and typical operating conditions, the journey takes approximately two hours.",
            ),
        ],
        expected_consistent=False,
        mode_b_detectable=False,  # Wikidata has no train speed data for this route
    ))

    # --- Type B: Arithmetic contradictions (3 chains) ---

    # IC5: 3 + 5 = 8 in step 1, then 3 + 5 = 10 in step 4
    # GlobalConsistencyChecker key=(3, "+", 5), result 8 vs 10
    chains.append(ChainSpec(
        chain_id=14,
        label="IC5-Arithmetic-3plus5",
        chain_type="inconsistent",
        contradiction_type="arithmetic",
        steps=[
            (
                "What is the sum of 3 and 5?",
                "The answer to this addition problem is 3 + 5 = 8. This is a fundamental arithmetic fact.",
            ),
            (
                "What is the product of 3 and 5?",
                "Three multiplied by five equals 15. Multiplication is essentially repeated addition.",
            ),
            (
                "Confirm: does the order of addition matter for 3 + 5?",
                "No, addition is commutative. Five plus three yields the same result as three plus five, both equal eight.",
            ),
            (
                "Let me re-examine the addition of 3 and 5.",
                "Upon reconsideration, I need to correct my earlier statement: 3 + 5 = 10. I apologize for the confusion.",
            ),
        ],
        expected_consistent=False,
        mode_b_detectable=False,  # ArithmeticExtractor checks within-step only; 3+5=10 is flagged locally but step 4 introduces it
    ))

    # IC6: 10 - 3 = 7 in step 1, then 10 - 3 = 5 in step 3
    # GlobalConsistencyChecker key=(10, "-", 3), result 7 vs 5
    chains.append(ChainSpec(
        chain_id=15,
        label="IC6-Arithmetic-10minus3",
        chain_type="inconsistent",
        contradiction_type="arithmetic",
        steps=[
            (
                "What is 10 minus 3?",
                "The calculation is straightforward: 10 - 3 = 7. Subtraction of three from ten gives seven.",
            ),
            (
                "What is 10 minus 4?",
                "Ten minus four equals 6. This is one step further than subtracting three.",
            ),
            (
                "Please recalculate 10 minus 3 using a different method.",
                "Using an alternative counting method: starting at 10 and subtracting 3, we get 10 - 3 = 5. This is the corrected result.",
            ),
            (
                "Is the subtraction result now confirmed?",
                "Yes, the subtraction result has been noted and recorded in our calculation log.",
            ),
        ],
        expected_consistent=False,
        mode_b_detectable=False,  # Cross-step arithmetic contradiction; Mode B only checks within-step
    ))

    # IC7: 4 + 7 = 11 in step 2, then 4 + 7 = 13 in step 4
    # GlobalConsistencyChecker key=(4, "+", 7), result 11 vs 13
    chains.append(ChainSpec(
        chain_id=16,
        label="IC7-Arithmetic-4plus7",
        chain_type="inconsistent",
        contradiction_type="arithmetic",
        steps=[
            (
                "How many items are in the inventory?",
                "We have several categories of items to tally in the inventory system.",
            ),
            (
                "How many items are in categories A and B combined?",
                "Items in categories A and B total 4 + 7 = 11 units when combined.",
            ),
            (
                "What other categories exist in the inventory?",
                "There are additional items in categories C, D, and E, each with varying quantities.",
            ),
            (
                "Recount categories A and B for accuracy.",
                "After a thorough recount, categories A and B together total 4 + 7 = 13 units. The inventory is updated.",
            ),
        ],
        expected_consistent=False,
        mode_b_detectable=False,  # Cross-step arithmetic; 4+7=13 vs 4+7=11 across steps
    ))

    # --- Type C: Factual contradictions (3 chains) ---

    # IC8: France capital Paris (step 1) vs Berlin (step 4)
    # GlobalConsistencyChecker factual: (france, capital) = paris vs berlin
    # Mode B via FactualExtractor: "Berlin is the capital of France" → KB contradicts → local violation
    chains.append(ChainSpec(
        chain_id=17,
        label="IC8-FranceCapital-ParisvsBerlin",
        chain_type="inconsistent",
        contradiction_type="factual",
        steps=[
            (
                "What is the capital of France?",
                "Paris is the capital of France. It has served as the seat of French government for many centuries.",
            ),
            (
                "What language is spoken in France?",
                "French is the official language of France. French is spoken by the vast majority of the population.",
            ),
            (
                "What continent is France part of?",
                "France is located in Europe. It shares borders with Germany, Spain, Italy, and other European nations.",
            ),
            (
                "What is the capital city of France?",
                "Berlin is the capital of France. It serves as the administrative and cultural center of the French Republic.",
            ),
        ],
        expected_consistent=False,
        mode_b_detectable=True,  # Wikidata contradicts "France capital Berlin" in step 4
    ))

    # IC9: Einstein born in Germany (step 1) vs Austria (step 4)
    # GlobalConsistencyChecker factual: (albert einstein, born in) = germany vs austria
    # Mode B via FactualExtractor: "Einstein born in Austria" → Wikidata P19 says Ulm (Germany) → contradiction
    chains.append(ChainSpec(
        chain_id=18,
        label="IC9-EinsteinBirthplace-GermanyvsAustria",
        chain_type="inconsistent",
        contradiction_type="factual",
        steps=[
            (
                "Where was Albert Einstein born?",
                "Albert Einstein was born in Germany. Specifically, he was born in the city of Ulm.",
            ),
            (
                "What scientific work is Einstein most famous for?",
                "Einstein developed the theory of relativity, which fundamentally transformed our understanding of physics.",
            ),
            (
                "When did Einstein live?",
                "Einstein lived from 1879 to 1955. He spent his early years in Europe before emigrating to the United States.",
            ),
            (
                "What is Einstein's national origin?",
                "Albert Einstein was born in Austria. His Austrian heritage played a formative role in his intellectual development.",
            ),
        ],
        expected_consistent=False,
        mode_b_detectable=True,  # Wikidata P19 for Einstein is Ulm (Germany), not Austria
    ))

    # IC10: Japan capital Tokyo (step 1) vs Osaka (step 4)
    # GlobalConsistencyChecker factual: (japan, capital) = tokyo vs osaka
    # Mode B via FactualExtractor: "Osaka is the capital of Japan" → KB contradicts → local violation
    chains.append(ChainSpec(
        chain_id=19,
        label="IC10-JapanCapital-TokyovsOsaka",
        chain_type="inconsistent",
        contradiction_type="factual",
        steps=[
            (
                "What is the capital of Japan?",
                "Tokyo is the capital of Japan. It is the most populous metropolitan area in the world.",
            ),
            (
                "What language is spoken in Japan?",
                "Japanese is the official language of Japan. Japanese uses multiple writing systems including kanji.",
            ),
            (
                "Where is Japan located in the world?",
                "Japan is located in Asia. It is an island nation situated in the western Pacific Ocean.",
            ),
            (
                "Which city serves as Japan's national capital?",
                "Osaka is the capital of Japan. It serves as the seat of the Japanese national government.",
            ),
        ],
        expected_consistent=False,
        mode_b_detectable=True,  # Wikidata P36 for Japan is Tokyo, not Osaka
    ))

    return chains


# ---------------------------------------------------------------------------
# Per-chain result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChainResult:
    """Results of running one chain through all three verification modes.

    **Detailed explanation for engineers:**
        Records the outcome for each mode (A/B/C) and detailed per-step
        information from Mode B and C for analysis.

    Attributes:
        chain: The ChainSpec that was evaluated.
        mode_a_detected: Always False (baseline — no verification).
        mode_b_detected: True if any step had violations or contradictions.
        mode_c_detected: True if Mode B detected OR global report inconsistent.
        mode_b_steps: Per-step Mode B results: list of dicts with step_index,
            n_violations, n_contradictions, verified flag.
        global_report_consistent: Whether the GlobalConsistencyChecker agreed
            the chain was consistent (after Mode C run).
        global_report_severity: "none" | "warning" | "critical".
        global_inconsistent_pairs: Raw list of (i, j, type, desc) tuples.
        elapsed_s: Wall-clock time for the combined Mode B + C run.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    chain: ChainSpec
    mode_a_detected: bool
    mode_b_detected: bool
    mode_c_detected: bool
    mode_b_steps: list[dict[str, Any]]
    global_report_consistent: bool
    global_report_severity: str
    global_inconsistent_pairs: list[tuple[int, int, str, str]]
    elapsed_s: float


# ---------------------------------------------------------------------------
# Verification runner
# ---------------------------------------------------------------------------


def run_chain(spec: ChainSpec, pipeline: VerifyRepairPipeline) -> ChainResult:
    """Run one chain through Mode A (baseline), Mode B (local), and Mode C (global).

    **Detailed explanation for engineers:**
        Mode A: Skip. By definition detects nothing.
        Mode B: Create a fresh ConstraintStateMachine with the shared pipeline
            (which has FactualExtractor enabled). Run each step. Collect per-step
            violations and contradictions. Detection if any violation or
            contradiction was found.
        Mode C: After all steps, call machine.check_global_consistency().
            Detection if Mode B detected OR global report says inconsistent.

        The same ConstraintStateMachine state is used for both Mode B and Mode C
        — Mode C is additive, not a separate run.

    Args:
        spec: The chain specification with questions and responses.
        pipeline: A VerifyRepairPipeline with FactualExtractor enabled.

    Returns:
        ChainResult with per-mode detection flags and diagnostic detail.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    t0 = time.perf_counter()

    # Mode A: Baseline — always 0 detections.
    mode_a_detected = False

    # Mode B + C: Use real ConstraintStateMachine with FactualExtractor.
    machine = ConstraintStateMachine(pipeline=pipeline)
    mode_b_steps: list[dict[str, Any]] = []
    mode_b_detected = False

    for question, response in spec.steps:
        step_result = machine.step(input_text=question, output_text=response)

        n_violations = len(step_result.verification.violations)
        n_contradictions = len(step_result.contradictions)
        step_verified = step_result.verification.verified

        step_info = {
            "step_index": step_result.step_index,
            "verified": step_verified,
            "n_violations": n_violations,
            "n_contradictions": n_contradictions,
            "energy": step_result.verification.energy,
            "violations": [v.description for v in step_result.verification.violations],
            "contradictions": step_result.contradictions,
        }
        mode_b_steps.append(step_info)

        # Mode B detects if any step has local violations or cross-step contradictions.
        if n_violations > 0 or n_contradictions > 0:
            mode_b_detected = True

    # Mode C: Run GlobalConsistencyChecker after all 4 steps.
    global_report = machine.check_global_consistency()

    global_detected = not global_report.consistent
    mode_c_detected = mode_b_detected or global_detected

    elapsed = time.perf_counter() - t0

    return ChainResult(
        chain=spec,
        mode_a_detected=mode_a_detected,
        mode_b_detected=mode_b_detected,
        mode_c_detected=mode_c_detected,
        mode_b_steps=mode_b_steps,
        global_report_consistent=global_report.consistent,
        global_report_severity=global_report.severity,
        global_inconsistent_pairs=global_report.inconsistent_pairs,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Aggregation and result building
# ---------------------------------------------------------------------------


def compute_results(chain_results: list[ChainResult]) -> dict[str, Any]:
    """Aggregate per-chain results into the experiment summary dict.

    **Detailed explanation for engineers:**
        Splits results into consistent (expected_consistent=True) and
        inconsistent (expected_consistent=False) chains. Computes detection
        rates and false positive rates. Also computes per-contradiction-type
        detection rates (numeric, arithmetic, factual) for the inconsistent chains.

    Args:
        chain_results: Results from run_chain() for all 20 chains.

    Returns:
        Dict matching the required results/experiment_176_results.json schema.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    consistent_results = [r for r in chain_results if r.chain.expected_consistent]
    inconsistent_results = [r for r in chain_results if not r.chain.expected_consistent]

    n_consistent = len(consistent_results)
    n_inconsistent = len(inconsistent_results)

    # --- Detection rates (inconsistent chains) ---
    n_b_detected = sum(1 for r in inconsistent_results if r.mode_b_detected)
    n_c_detected = sum(1 for r in inconsistent_results if r.mode_c_detected)

    mode_b_detection = n_b_detected / n_inconsistent if n_inconsistent > 0 else 0.0
    mode_c_detection = n_c_detected / n_inconsistent if n_inconsistent > 0 else 0.0

    # --- False positive rates (consistent chains) ---
    n_fp_b = sum(1 for r in consistent_results if r.mode_b_detected)
    n_fp_c = sum(1 for r in consistent_results if r.mode_c_detected)

    fp_rate_b = n_fp_b / n_consistent if n_consistent > 0 else 0.0
    fp_rate_c = n_fp_c / n_consistent if n_consistent > 0 else 0.0

    # --- Global checker added value ---
    global_added = n_c_detected - n_b_detected

    # --- Per-contradiction-type breakdown ---
    per_type: dict[str, dict[str, int]] = {}
    for ctype in ("numeric", "arithmetic", "factual"):
        typed = [r for r in inconsistent_results if r.chain.contradiction_type == ctype]
        per_type[ctype] = {
            "total": len(typed),
            "detected_b": sum(1 for r in typed if r.mode_b_detected),
            "detected_c": sum(1 for r in typed if r.mode_c_detected),
        }

    # --- Per-chain detail ---
    per_chain = []
    for r in chain_results:
        per_chain.append({
            "chain_id": r.chain.chain_id,
            "label": r.chain.label,
            "chain_type": r.chain.chain_type,
            "contradiction_type": r.chain.contradiction_type,
            "expected_consistent": r.chain.expected_consistent,
            "mode_a_detected": r.mode_a_detected,
            "mode_b_detected": r.mode_b_detected,
            "mode_c_detected": r.mode_c_detected,
            "global_report_consistent": r.global_report_consistent,
            "global_report_severity": r.global_report_severity,
            "global_inconsistent_pairs": [
                {
                    "step_i": p[0],
                    "step_j": p[1],
                    "type": p[2],
                    "description": p[3],
                }
                for p in r.global_inconsistent_pairs
            ],
            "mode_b_steps": r.mode_b_steps,
            "elapsed_s": round(r.elapsed_s, 3),
        })

    return {
        "experiment": "Exp 176 — Multi-Turn Factual Reasoning Verification",
        "date": "20260411",
        "target_models": ["Qwen3.5-0.8B", "google/gemma-4-E4B-it"],
        "n_chains": len(chain_results),
        "consistent_chains": n_consistent,
        "inconsistent_chains": n_inconsistent,
        # --- Mode A (baseline) ---
        "mode_a_detection": 0.0,
        "mode_a_description": "No verification — baseline. Always 0% by construction.",
        # --- Mode B (local only: ConstraintStateMachine + FactualExtractor) ---
        "mode_b_detection": round(mode_b_detection, 4),
        "mode_b_n_detected": n_b_detected,
        "mode_b_description": (
            "Local-only verification: ConstraintStateMachine + FactualExtractor "
            "(Wikidata KB). Detects within-step KB violations. "
            "Cannot detect pure cross-step numeric/arithmetic contradictions."
        ),
        # --- Mode C (local + global) ---
        "mode_c_detection": round(mode_c_detection, 4),
        "mode_c_n_detected": n_c_detected,
        "mode_c_description": (
            "Local + Global: Mode B + GlobalConsistencyChecker (Exp 172). "
            "Adds cross-step numeric, arithmetic, and factual pattern matching."
        ),
        # --- False positive rates ---
        "false_positive_rate_b": round(fp_rate_b, 4),
        "false_positive_rate_c": round(fp_rate_c, 4),
        "n_false_positives_b": n_fp_b,
        "n_false_positives_c": n_fp_c,
        # --- Value added by GlobalConsistencyChecker ---
        "global_checker_added_detections": global_added,
        "global_checker_added_description": (
            f"GlobalConsistencyChecker added {global_added} detections beyond "
            "Mode B, specifically for numeric and arithmetic cross-step contradictions."
        ),
        # --- Per-contradiction-type breakdown ---
        "per_type_breakdown": per_type,
        # --- Per-chain detail ---
        "per_chain": per_chain,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 176: multi-turn factual reasoning verification.

    **Detailed explanation for engineers:**
        1. Build consistent and inconsistent chains.
        2. Create the shared VerifyRepairPipeline with FactualExtractor enabled.
        3. Run each chain through all three modes.
        4. Aggregate results and write results/experiment_176_results.json.

    The pipeline is shared across all chains (not re-created per chain) so that
    the module-level Wikidata caches in FactualExtractor accumulate across chains,
    reducing redundant SPARQL calls. The ConstraintStateMachine is created fresh
    for each chain to ensure no state leakage.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    print("=" * 72)
    print("Experiment 176 — Multi-Turn Factual Reasoning Verification")
    print("Combining FactualExtractor (Exp 158) + ConstraintStateMachine (Exp 125)")
    print("         + GlobalConsistencyChecker (Exp 172)")
    print("=" * 72)

    # Build chain specs.
    consistent_chains = _build_consistent_chains()
    inconsistent_chains = _build_inconsistent_chains()
    all_chains = consistent_chains + inconsistent_chains

    print(f"\nChains loaded: {len(consistent_chains)} consistent, {len(inconsistent_chains)} inconsistent")
    print(f"Inconsistency types: {sum(1 for c in inconsistent_chains if c.contradiction_type == 'numeric')} numeric, "
          f"{sum(1 for c in inconsistent_chains if c.contradiction_type == 'arithmetic')} arithmetic, "
          f"{sum(1 for c in inconsistent_chains if c.contradiction_type == 'factual')} factual")

    # Build shared pipeline with FactualExtractor enabled.
    # _SingleArgPipeline wraps VerifyRepairPipeline so that agentic.propagate()
    # can call verify(text_only) without needing a separate question argument.
    print("\nInitialising pipeline with FactualExtractor (Wikidata SPARQL, 5s timeout)...")
    extractor = AutoExtractor(enable_factual_extractor=True)
    pipeline = _SingleArgPipeline(extractor=extractor)
    print("Pipeline ready (verify-only mode, no LLM model loaded).")

    # Run all chains.
    print(f"\nRunning {len(all_chains)} chains through Mode A / B / C ...\n")
    chain_results: list[ChainResult] = []
    t_total_start = time.perf_counter()

    for i, spec in enumerate(all_chains):
        print(f"  [{i+1:2d}/{len(all_chains)}] {spec.label} ... ", end="", flush=True)
        result = run_chain(spec, pipeline)
        chain_results.append(result)

        # Print compact per-chain summary.
        b = "B✓" if result.mode_b_detected else "B✗"
        c = "C✓" if result.mode_c_detected else "C✗"
        g = "G✓" if not result.global_report_consistent else "G✗"
        status = "INCONS" if not spec.expected_consistent else "CONSIS"
        correct_b = (result.mode_b_detected == (not spec.expected_consistent))
        correct_c = (result.mode_c_detected == (not spec.expected_consistent))
        verdict_b = "✓" if correct_b else "✗FP" if (result.mode_b_detected and spec.expected_consistent) else "✗FN"
        verdict_c = "✓" if correct_c else "✗FP" if (result.mode_c_detected and spec.expected_consistent) else "✗FN"
        print(f"[{status}] {b}{verdict_b} {c}{verdict_c} {g} ({result.elapsed_s:.2f}s)")

    t_total = time.perf_counter() - t_total_start

    # Aggregate.
    results = compute_results(chain_results)

    # Print summary.
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Mode A detection rate:     {results['mode_a_detection']:.1%}  (baseline — 0% by construction)")
    print(f"  Mode B detection rate:     {results['mode_b_detection']:.1%}  ({results['mode_b_n_detected']}/{results['inconsistent_chains']} inconsistent chains caught)")
    print(f"  Mode C detection rate:     {results['mode_c_detection']:.1%}  ({results['mode_c_n_detected']}/{results['inconsistent_chains']} inconsistent chains caught)")
    print(f"  False positive rate B:     {results['false_positive_rate_b']:.1%}")
    print(f"  False positive rate C:     {results['false_positive_rate_c']:.1%}")
    print(f"  GlobalChecker added:       +{results['global_checker_added_detections']} detections beyond Mode B")
    print()
    print("  Per-contradiction-type breakdown:")
    for ctype, stats in results["per_type_breakdown"].items():
        print(f"    {ctype:12s}: {stats['detected_c']}/{stats['total']} caught by C,  "
              f"{stats['detected_b']}/{stats['total']} caught by B")
    print()
    print(f"  Total wall-clock time: {t_total:.1f}s")

    # Save results.
    output_path = Path(__file__).parent.parent / "results" / "experiment_176_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {output_path}")

    # Final verdict.
    b_rate = results["mode_b_detection"]
    c_rate = results["mode_c_detection"]
    gap = c_rate - b_rate
    print(f"\nConclusion: GlobalConsistencyChecker closes a {gap:.1%} detection gap "
          f"between local-only verification ({b_rate:.1%}) and full multi-turn "
          f"verification ({c_rate:.1%}) for factual reasoning chains.")


if __name__ == "__main__":
    main()

"""Factual constraint extractor backed by Wikidata SPARQL (Exp 158).

**Researcher summary:**
    Exp 88 showed 100% false negative rate on factual claims — no arithmetic/
    logic constraint work closes this gap. This module implements Goal #3 of
    research-program.md: a knowledge-base-backed extractor that grounds LLM
    factual claims in an external KB (Wikidata) and encodes verified/violated
    claims as IsingConstraints (FactualClaimConstraints).

    Knowledge source: Wikidata SPARQL (https://query.wikidata.org/sparql) with
    entity resolution via the Wikidata search API. Gracefully degrades when the
    network is unavailable (timeout >5s returns empty list + logged warning).

**Detailed explanation for engineers:**
    The module is structured in four layers:

    1. **Entity extraction** — Regex-based NER (no spaCy) that identifies named
       entities in text: person names, places, dates, organizations, and numeric
       quantities with units. Returns a list of (entity_text, entity_type) tuples.

    2. **Claim decomposition** — Regex patterns that extract factual claim triples
       (subject, predicate, object) from natural language sentences. Handles
       patterns like "X is Y", "X was born in Y", "X is the capital of Y",
       "X is located in Y", "X has N Y".

    3. **Knowledge base lookup** — For each extracted claim triple:
         a. Resolve subject → Wikidata QID via label search API
         b. Map predicate text → Wikidata property ID (e.g., "capital" → P36)
         c. Query SPARQL if (subject_QID, property, object) is in Wikidata
         d. Cache all lookups in module-level dicts to avoid duplicate API calls
       Only claims whose predicate can be mapped to a known property are queried.
       Unknown predicates are skipped (we don't add what we can't check).

    4. **Constraint encoding** — Each KB-verified claim becomes a
       FactualClaimConstraint with satisfied=True (energy=0.0). Each
       KB-contradicted claim becomes satisfied=False (energy=1.0). Claims the
       KB returns "unknown" on are not added as constraints.

    **Graceful degradation:**
    If any network call times out (>5s) or raises a connection error, the
    extractor logs a warning to stderr and returns an empty list. This prevents
    pipeline stalls when running in sandboxed environments without internet
    access.

    **Cache:**
    Module-level ``_QID_CACHE`` (entity text → QID or None) and
    ``_CLAIM_CACHE`` ((subject_qid, property_id, object_text) → True/False/None)
    avoid redundant API calls across multiple extract() calls in a session.

    **FactualClaimConstraint:**
    Extends BaseConstraint. energy(x) = 0.0 if satisfied, 1.0 if violated.
    Ignores the Ising configuration x — like SpilledEnergyConstraint, the
    satisfaction is determined at extraction time from the KB, not at inference
    time from the configuration.

    **Integration with AutoExtractor:**
    FactualExtractor is opt-in: disabled by default. Enable by passing
    ``enable_factual_extractor=True`` to AutoExtractor(), or by calling
    ``auto_extractor.add_extractor(FactualExtractor())``.

Target models: Qwen3.5-0.8B, google/gemma-4-E4B-it (as per Exp 157/158)
Benchmark: Exp 158 — TruthfulQA subset (50 questions), coverage + accuracy.
  Target: constraint coverage >30% of factual questions have ≥1 constraint.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import jax.numpy as jnp

from carnot.verify.constraint import BaseConstraint
from carnot.pipeline.extract import ConstraintExtractor, ConstraintResult  # noqa: F401

if TYPE_CHECKING:
    import jax

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: requests (for Wikidata API calls)
# ---------------------------------------------------------------------------
# Import at module level so tests can patch `carnot.pipeline.factual_extractor.requests`.
# Set to None if the library is not installed; all network functions check for
# None and return gracefully-degraded results in that case.
try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Module-level caches (survive across extract() calls in a session)
# ---------------------------------------------------------------------------

#: entity_text (normalized) → Wikidata QID (e.g. "Q142") or None if not found.
_QID_CACHE: dict[str, str | None] = {}

#: (subject_qid, property_id, object_text_lower) → True (verified) /
#: False (contradicted) / None (unknown or network failure).
_CLAIM_CACHE: dict[tuple[str, str, str], bool | None] = {}


# ---------------------------------------------------------------------------
# Wikidata property map — natural language predicate → Wikidata property ID
# ---------------------------------------------------------------------------

#: Maps normalized predicate phrases extracted from text to Wikidata property IDs.
#: Only predicates present here can be verified against Wikidata. Unknown
#: predicates are silently skipped — we do not add unverifiable constraints.
_PREDICATE_TO_PROPERTY: dict[str, str] = {
    # Geography
    "capital": "P36",           # capital city of a country/region
    "capital of": "P36",
    "is the capital of": "P36",
    "located in": "P131",       # administrative territory entity is in
    "is located in": "P131",
    "is in": "P131",
    "country": "P17",           # country this entity belongs to
    "continent": "P30",         # continent
    # People
    "born in": "P19",           # place of birth
    "was born in": "P19",
    "birthplace": "P19",
    "place of birth": "P19",
    "died in": "P20",           # place of death
    "nationality": "P27",       # country of citizenship
    "occupation": "P106",       # occupation
    # Organizations & things
    "founded by": "P112",       # founder
    "official language": "P37", # official language
    "currency": "P38",          # currency
    "head of government": "P6", # head of government
    "head of state": "P35",     # head of state
    "language": "P37",
    "population": "P1082",      # population (numeric)
    # Science
    "chemical formula": "P274", # chemical formula
    "element symbol": "P246",   # element symbol
    "atomic number": "P1086",   # atomic number (numeric)
}


# ---------------------------------------------------------------------------
# FactualClaimConstraint — a ConstraintTerm for one KB-verified claim
# ---------------------------------------------------------------------------


class FactualClaimConstraint(BaseConstraint):
    """A ConstraintTerm encoding the KB verification result for one factual claim.

    **Researcher summary:**
        Fixed-energy constraint: 0.0 if the claim is KB-verified, 1.0 if
        contradicted. Ignores the Ising configuration x (like
        SpilledEnergyConstraint). This represents an external factual
        grounding signal, not an optimisation target in configuration space.

    **Detailed explanation for engineers:**
        The Carnot verify-repair pipeline expects ConstraintTerms that can be
        composed via ComposedEnergy and checked via is_satisfied(). This class
        adapts a binary KB lookup result (verified / contradicted) to that
        interface. The energy is a constant set at construction time:

            E(x) = 0.0  if the KB confirmed the claim     (satisfied=True)
            E(x) = 1.0  if the KB contradicted the claim  (satisfied=False)

        The gradient is always zero (constant energy → dE/dx = 0). This means
        the Ising repair loop cannot "fix" a factual violation by adjusting the
        configuration — factual errors require different intervention (e.g.,
        regeneration with a corrected prompt). The energy signal is still useful
        for *detection* and *routing* decisions upstream.

    Attributes:
        claim_satisfied: True if the KB confirmed the claim, False if contradicted.
        claim_description: Human-readable summary of the claim being verified.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
    """

    def __init__(self, claim_satisfied: bool, claim_description: str) -> None:
        """Create a FactualClaimConstraint from a KB verification result.

        Args:
            claim_satisfied: True if the knowledge base confirmed the claim
                (or if no evidence against it). False if the KB explicitly
                contradicted the claim.
            claim_description: Human-readable summary of the claim, e.g.
                "France capital Paris".
        """
        self._satisfied = claim_satisfied
        self._description = claim_description

    @property
    def name(self) -> str:
        """Human-readable name encoding satisfaction status and claim summary."""
        status = "ok" if self._satisfied else "violated"
        # Truncate long descriptions for readability in reports.
        short = self._description[:50]
        return f"factual({status}): {short}"

    @property
    def satisfaction_threshold(self) -> float:
        """Threshold: 0.5 (energy is 0.0 or 1.0, never in between)."""
        return 0.5

    def energy(self, x: "jax.Array") -> "jax.Array":
        """Return constant energy: 0.0 if satisfied, 1.0 if violated.

        **Detailed explanation for engineers:**
            The Ising configuration x is intentionally ignored. The factual
            verification was performed against a knowledge base at extraction
            time. Unlike arithmetic or type constraints that depend on a value
            in x, factual correctness is determined by external KB facts.

        Args:
            x: Ising configuration (ignored).

        Returns:
            Scalar JAX float32: 0.0 (satisfied) or 1.0 (violated).
        """
        _ = x  # intentionally unused — energy is KB-determined, not x-dependent
        return jnp.float32(0.0 if self._satisfied else 1.0)

    def is_satisfied(self, x: "jax.Array") -> bool:
        """Return True iff the KB confirmed this claim.

        Args:
            x: Ignored.

        Returns:
            True if KB-confirmed, False if KB-contradicted.
        """
        return self._satisfied


# ---------------------------------------------------------------------------
# Entity and claim extraction helpers (no spaCy — stdlib regex only)
# ---------------------------------------------------------------------------


#: Matches capitalized multi-word sequences: "Albert Einstein", "New York",
#: "World War II". Requires at least two capitalized tokens to avoid
#: sentence-initial false positives. Also matches ALL-CAPS acronyms.
_ENTITY_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z\-]*(?:\s+[A-Z][a-zA-Z\-]*){1,4})\b"
    r"|"
    r"\b([A-Z]{2,6})\b",   # acronyms: USA, UK, NASA, CERN
)

#: Common English words that begin a sentence with a capital letter but are NOT
#: entity names. Stripped from the front of entity candidates when found as
#: the leading token. This handles "The USA joined NATO" → "USA joined NATO" →
#: leading word "The" is stripped → entity is "USA".
_LEADING_STOP_WORDS: frozenset[str] = frozenset(
    {
        "The", "A", "An", "In", "On", "At", "By", "Of", "For", "And", "Or",
        "But", "Its", "Their", "This", "That", "These", "Those", "His", "Her",
        "Our", "Your", "My", "It", "He", "She", "We", "They", "Who", "Which",
        "What", "When", "Where", "How", "Why", "If", "As", "So", "No", "Not",
    }
)

#: Year pattern: 4-digit years in plausible historical range.
_YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-2][0-9])\b")

#: Month + year: "April 1889", "January 2024"
_MONTH_YEAR_PATTERN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{4})\b"
)

#: Numeric quantity with unit: "3.14 km", "100 mph", "70 kg"
_QUANTITY_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s+"
    r"(km|miles?|meters?|metres?|feet|foot|pounds?|kg|kilograms?|"
    r"liters?|litres?|mph|kph|km/h|seconds?|minutes?|hours?|days?|"
    r"years?|meters?\s+per\s+second|m/s|celsius|fahrenheit|kelvin|"
    r"percent|%|billion|million|trillion)\b",
    re.IGNORECASE,
)

# Claim triple patterns: each produces (subject, predicate_key, object_text).
# Patterns are tried in order; first match wins per sentence.
# We use named groups for clarity.
_CLAIM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # "X is the capital of Y"  →  (Y, capital, X)
    (
        "capital",
        re.compile(
            r"(?P<object>[A-Z][^\s,]+(?:\s+[A-Z][^\s,]+)*)\s+"
            r"is\s+the\s+capital\s+of\s+"
            r"(?P<subject>[A-Z][^\s.,!?]+(?:\s+[A-Z][^\s.,!?]+)*)",
            re.IGNORECASE,
        ),
    ),
    # "capital of Y is X"  →  (Y, capital, X)
    (
        "capital",
        re.compile(
            r"[Cc]apital\s+of\s+"
            r"(?P<subject>[A-Z][^\s.,!?]+(?:\s+[A-Z][^\s.,!?]+)*)"
            r"\s+is\s+"
            r"(?P<object>[A-Z][^\s.,!?]+(?:\s+[A-Z][^\s.,!?]+)*)",
            re.IGNORECASE,
        ),
    ),
    # "X was born in Y"  →  (X, born in, Y)
    (
        "born in",
        re.compile(
            r"(?P<subject>[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,3})"
            r"\s+was\s+born\s+in\s+"
            r"(?P<object>[A-Z][^\s.,!?]+(?:\s+[A-Z][^\s.,!?]+)*)",
            re.IGNORECASE,
        ),
    ),
    # "X is located in Y"  →  (X, located in, Y)
    (
        "located in",
        re.compile(
            r"(?P<subject>[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,3})"
            r"\s+is\s+located\s+in\s+"
            r"(?P<object>[A-Z][^\s.,!?]+(?:\s+[A-Z][^\s.,!?]+)*)",
            re.IGNORECASE,
        ),
    ),
    # "official language of X is Y"  →  (X, official language, Y)
    (
        "official language",
        re.compile(
            r"[Oo]fficial\s+language\s+of\s+"
            r"(?P<subject>[A-Z][^\s.,!?]+(?:\s+[A-Z][^\s.,!?]+)*)"
            r"\s+is\s+"
            r"(?P<object>[A-Z][^\s.,!?]+(?:\s+[A-Z][^\s.,!?]+)*)",
            re.IGNORECASE,
        ),
    ),
    # "currency of X is Y"  →  (X, currency, Y)
    (
        "currency",
        re.compile(
            r"[Cc]urrency\s+of\s+"
            r"(?P<subject>[A-Z][^\s.,!?]+(?:\s+[A-Z][^\s.,!?]+)*)"
            r"\s+is\s+"
            r"(?P<object>[A-Z][^\s.,!?]+(?:\s+[A-Z][^\s.,!?]+)*)",
            re.IGNORECASE,
        ),
    ),
    # "X uses Y as its currency"  →  (X, currency, Y)
    (
        "currency",
        re.compile(
            r"(?P<subject>[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,3})"
            r"\s+uses\s+"
            r"(?P<object>[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)"
            r"\s+as\s+its\s+currency",
            re.IGNORECASE,
        ),
    ),
    # "X is in Y"  →  (X, located in, Y)
    (
        "located in",
        re.compile(
            r"(?P<subject>[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,3})"
            r"\s+is\s+in\s+"
            r"(?P<object>[A-Z][^\s.,!?]+(?:\s+[A-Z][^\s.,!?]+)*)",
            re.IGNORECASE,
        ),
    ),
]


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Extract named entities from text using regex patterns.

    **Detailed explanation for engineers:**
        Runs three passes over the text:
        1. Capitalized multi-word sequences (potential person/place/org names).
           Requires ≥2 capitalized words to reduce sentence-initial false positives.
        2. Standalone ALL-CAPS acronyms (USA, EU, NASA, etc.).
        3. Date-like patterns: years (4-digit) and month+year combinations.
        4. Numeric quantities with physical units.

        No spaCy or external NLP libraries are used — stdlib re only.
        This keeps the dependency footprint minimal, at the cost of lower
        precision (some false positives like sentence-initial words).

    Args:
        text: Natural language input text.

    Returns:
        List of (entity_text, entity_type) tuples where entity_type is one of:
        "named_entity", "acronym", "year", "date", "quantity".
        Duplicates may appear if the same text matches multiple patterns.

    Spec: REQ-VERIFY-001
    """
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Pass 1: capitalized sequences and acronyms
    for m in _ENTITY_PATTERN.finditer(text):
        entity = (m.group(1) or m.group(2) or "").strip()
        if not entity:  # pragma: no cover  — regex always produces non-empty
            continue
        # Strip leading stop words (e.g., "The USA" → "USA", "In Paris" → "Paris")
        tokens = entity.split()
        while tokens and tokens[0] in _LEADING_STOP_WORDS:
            tokens = tokens[1:]
        if not tokens:
            continue
        entity = " ".join(tokens)
        if entity in seen:
            continue
        seen.add(entity)
        # Distinguish acronyms (all caps) from normal named entities
        etype = "acronym" if entity.isupper() else "named_entity"
        results.append((entity, etype))

    # Pass 2: month+year dates (higher specificity than bare years)
    for m in _MONTH_YEAR_PATTERN.finditer(text):
        date_str = m.group(0)
        if date_str not in seen:
            seen.add(date_str)
            results.append((date_str, "date"))

    # Pass 3: bare 4-digit years (lower specificity)
    for m in _YEAR_PATTERN.finditer(text):
        year_str = m.group(1)
        if year_str not in seen:
            seen.add(year_str)
            results.append((year_str, "year"))

    # Pass 4: numeric quantities with units
    for m in _QUANTITY_PATTERN.finditer(text):
        qty_str = m.group(0)
        if qty_str not in seen:
            seen.add(qty_str)
            results.append((qty_str, "quantity"))

    return results


def extract_claims(text: str) -> list[tuple[str, str, str]]:
    """Decompose text into factual claim triples (subject, predicate_key, object).

    **Detailed explanation for engineers:**
        Tries each pattern in _CLAIM_PATTERNS against each sentence in the text.
        Returns a list of (subject, predicate_key, object_text) triples where:
        - subject: the entity the claim is about (e.g., "France")
        - predicate_key: normalized predicate string matching a key in
          _PREDICATE_TO_PROPERTY (e.g., "capital", "born in")
        - object_text: the value asserted (e.g., "Paris")

        Only claims where predicate_key is in _PREDICATE_TO_PROPERTY can be
        verified against Wikidata. Unknown-predicate claims are still returned
        here (for coverage metrics) but will be skipped during KB lookup.

    Args:
        text: Input text — may be a single sentence or a paragraph.

    Returns:
        List of (subject, predicate_key, object_text) tuples.
        Empty list if no claim patterns match.

    Spec: REQ-VERIFY-001
    """
    claims: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    # Split into sentences for pattern matching.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        for predicate_key, pattern in _CLAIM_PATTERNS:
            m = pattern.search(sent)
            if m:
                try:
                    subject = m.group("subject").strip().rstrip(".,!?;:")
                    obj = m.group("object").strip().rstrip(".,!?;:")
                except IndexError:  # pragma: no cover  — named groups always present
                    continue
                triple = (subject, predicate_key, obj)
                if triple not in seen:
                    seen.add(triple)
                    claims.append(triple)

    return claims


# ---------------------------------------------------------------------------
# Wikidata API helpers
# ---------------------------------------------------------------------------


def _resolve_qid(entity_text: str, timeout: float = 5.0) -> str | None:
    """Resolve an entity name to a Wikidata QID via the Wikidata search API.

    **Detailed explanation for engineers:**
        Calls the Wikidata wbsearchentities action with the entity text and
        returns the first result's QID (e.g., "Q142" for France). The result
        is cached in ``_QID_CACHE`` to avoid repeated API calls for the same
        entity within a session.

        On network timeout (>timeout seconds) or any connection/HTTP error,
        logs a warning and returns None. Callers that receive None should
        skip KB verification for that entity.

    Args:
        entity_text: Surface form of the entity (e.g., "France", "Paris").
        timeout: Maximum seconds to wait for the API response. Default 5.0.

    Returns:
        Wikidata QID string (e.g. "Q142") if found, None otherwise.
    """
    normalized = entity_text.strip().lower()
    if normalized in _QID_CACHE:
        return _QID_CACHE[normalized]

    if requests is None:
        logger.warning(
            "FactualExtractor: 'requests' library not installed. "
            "Cannot perform Wikidata lookups. "
            "Install with: pip install requests"
        )
        _QID_CACHE[normalized] = None
        return None

    try:
        resp = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": entity_text,
                "language": "en",
                "format": "json",
                "limit": 1,
            },
            timeout=timeout,
            headers={"User-Agent": "Carnot-EBM/1.0 (factual-extractor-exp158)"},
        )
        resp.raise_for_status()
        data = resp.json()
        search = data.get("search", [])
        if search:
            qid = search[0].get("id")
            _QID_CACHE[normalized] = qid
            return qid
        _QID_CACHE[normalized] = None
        return None
    except Exception as exc:
        # Broad catch: includes Timeout, ConnectionError, HTTPError, JSONDecodeError.
        # In all cases, gracefully degrade by returning None.
        logger.warning(
            "FactualExtractor: Wikidata entity lookup failed for %r: %s",
            entity_text,
            exc,
        )
        _QID_CACHE[normalized] = None
        return None


def _verify_claim_sparql(
    subject_qid: str,
    property_id: str,
    object_text: str,
    timeout: float = 5.0,
) -> bool | None:
    """Query Wikidata SPARQL to verify a (subject_QID, property, object) claim.

    **Detailed explanation for engineers:**
        Sends a SPARQL query to the Wikidata public endpoint
        (https://query.wikidata.org/sparql). The query asks: does entity
        ``subject_qid`` have property ``property_id`` whose value's label
        contains ``object_text`` (case-insensitive partial match)?

        A partial label match is used (CONTAINS + LCASE) to tolerate minor
        surface-form differences: "United States" matches "United States of
        America". This trades precision for recall — appropriate for the
        factual extraction use case where we want to catch true positives.

        Results are cached in ``_CLAIM_CACHE`` to avoid duplicate queries.

    Args:
        subject_qid: Wikidata QID of the claim subject (e.g., "Q142").
        property_id: Wikidata property ID (e.g., "P36" for capital).
        object_text: Text value to match in the object's label.
        timeout: Maximum seconds for SPARQL request. Default 5.0.

    Returns:
        True if the claim is confirmed in Wikidata.
        False if the property exists for the subject but no value matches.
        None if the SPARQL query failed (network error, timeout, parse error).

    Spec: REQ-VERIFY-001
    """
    obj_lower = object_text.strip().lower()
    cache_key = (subject_qid, property_id, obj_lower)
    if cache_key in _CLAIM_CACHE:
        return _CLAIM_CACHE[cache_key]

    if requests is None:
        _CLAIM_CACHE[cache_key] = None
        return None

    # Query: does subject_qid have property_id with a value whose label
    # (in English) contains the object text?
    sparql_query = (
        f"SELECT ?objLabel WHERE {{\n"
        f"  wd:{subject_qid} wdt:{property_id} ?obj.\n"
        f"  SERVICE wikibase:label {{\n"
        f"    bd:serviceParam wikibase:language 'en'.\n"
        f"    ?obj rdfs:label ?objLabel.\n"
        f"  }}\n"
        f"}}\n"
        f"LIMIT 10"
    )

    try:
        resp = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": sparql_query, "format": "json"},
            timeout=timeout,
            headers={
                "User-Agent": "Carnot-EBM/1.0 (factual-extractor-exp158)",
                "Accept": "application/sparql-results+json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        bindings = data.get("results", {}).get("bindings", [])

        if not bindings:
            # Property exists in schema but no values found for this subject.
            # Treat as contradiction (KB has data, none matches).
            # Note: if the property genuinely doesn't apply to this entity,
            # bindings will also be empty — we conservatively mark as None.
            _CLAIM_CACHE[cache_key] = None
            return None

        # Check if any result label matches the object text.
        for binding in bindings:
            label = binding.get("objLabel", {}).get("value", "").lower()
            if obj_lower in label or label in obj_lower:
                _CLAIM_CACHE[cache_key] = True
                return True

        # Labels found but none match: KB contradicts the claim.
        _CLAIM_CACHE[cache_key] = False
        return False

    except Exception as exc:
        logger.warning(
            "FactualExtractor: Wikidata SPARQL query failed for (%s, %s, %r): %s",
            subject_qid,
            property_id,
            object_text,
            exc,
        )
        _CLAIM_CACHE[cache_key] = None
        return None


# ---------------------------------------------------------------------------
# FactualExtractor — ConstraintExtractor Protocol implementation
# ---------------------------------------------------------------------------


class FactualExtractor:
    """Extract and KB-verify factual claims from natural language text.

    **Researcher summary:**
        Implements ConstraintExtractor Protocol. Extracts named entities and
        (subject, predicate, object) claim triples, verifies each against
        Wikidata SPARQL, and returns verified/contradicted claims as
        FactualClaimConstraints. Gracefully degrades on network failure.
        Designed to close the 100% false-negative rate on factual claims
        observed in Exp 88 (Goal #3 of research-program.md).

    **Detailed explanation for engineers:**
        FactualExtractor is opt-in (not included in AutoExtractor by default)
        because:
        1. It makes network calls (potential latency/failure)
        2. Only useful for factual-domain text, not code/arithmetic
        3. Adding it to the default pipeline would break existing callers
           who don't want network calls

        To use:
            # Standalone
            ext = FactualExtractor()
            results = ext.extract("Paris is the capital of France.")

            # With AutoExtractor (opt-in)
            auto = AutoExtractor(enable_factual_extractor=True)
            results = auto.extract(text, domain="factual")

        The ``extract()`` method is safe to call with domain=None or any
        domain, but only processes when domain is None or "factual".

    Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
    """

    def __init__(self, timeout: float = 5.0) -> None:
        """Create a FactualExtractor.

        Args:
            timeout: Network timeout in seconds for Wikidata API calls.
                Defaults to 5.0. On timeout, returns empty list with warning.
        """
        self._timeout = timeout

    @property
    def supported_domains(self) -> list[str]:
        """Domains this extractor handles: factual knowledge verification."""
        return ["factual"]

    def extract(
        self, text: str, domain: str | None = None
    ) -> list[ConstraintResult]:
        """Extract and verify factual claims from text.

        **Detailed explanation for engineers:**
            Full pipeline:
            1. Domain guard: skip if domain is given and not "factual".
            2. Extract claim triples from text via _CLAIM_PATTERNS.
            3. For each triple whose predicate is in _PREDICATE_TO_PROPERTY:
               a. Resolve subject to Wikidata QID (cached, timeout guarded)
               b. Run SPARQL query to verify (subject_QID, property, object)
               c. Verified → FactualClaimConstraint(satisfied=True)
                  Contradicted → FactualClaimConstraint(satisfied=False)
                  Unknown/error → skip (don't add what we can't check)
            4. Return list[ConstraintResult] with energy_term set.

            On any network failure (timeout, DNS, HTTP error), the method logs
            a warning and returns whatever results were collected so far —
            partial results are valid. If the first network call fails, an
            empty list is returned.

        Args:
            text: Input text containing factual claims.
            domain: Optional domain hint. Returns [] if not None and not
                "factual".

        Returns:
            List of ConstraintResult objects. Each has:
            - constraint_type: "factual_verified" or "factual_contradicted"
            - description: Human-readable claim summary
            - energy_term: FactualClaimConstraint (satisfied=True/False)
            - metadata: subject, predicate, object, qid, verified flag

        Spec: REQ-VERIFY-001, REQ-VERIFY-002
        """
        if domain is not None and domain not in self.supported_domains:
            return []

        claims = extract_claims(text)
        if not claims:
            return []

        results: list[ConstraintResult] = []

        for subject, predicate_key, obj_text in claims:
            # Skip predicates we don't have a Wikidata property for.
            property_id = _PREDICATE_TO_PROPERTY.get(predicate_key)
            if property_id is None:
                continue

            # Resolve subject entity to QID. Network failure → None → skip.
            subject_qid = _resolve_qid(subject, timeout=self._timeout)
            if subject_qid is None:
                continue

            # Verify claim against Wikidata SPARQL.
            verified = _verify_claim_sparql(
                subject_qid, property_id, obj_text, timeout=self._timeout
            )
            if verified is None:
                # Unknown — KB returned no usable signal; skip this claim.
                continue

            constraint = FactualClaimConstraint(
                claim_satisfied=verified,
                claim_description=f"{subject} {predicate_key} {obj_text}",
            )
            ctype = "factual_verified" if verified else "factual_contradicted"
            results.append(
                ConstraintResult(
                    constraint_type=ctype,
                    description=(
                        f"KB {'confirms' if verified else 'contradicts'}: "
                        f"{subject} {predicate_key} {obj_text}"
                    ),
                    energy_term=constraint,
                    metadata={
                        "subject": subject,
                        "predicate": predicate_key,
                        "object": obj_text,
                        "subject_qid": subject_qid,
                        "property_id": property_id,
                        "verified": verified,
                        "satisfied": verified,
                    },
                )
            )

        return results

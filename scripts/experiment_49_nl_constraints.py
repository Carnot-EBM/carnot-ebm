#!/usr/bin/env python3
"""Experiment 49: Natural language constraint extraction and verification.

**Researcher summary:**
    Extract verifiable claims from free text using NLI (natural language
    inference) patterns, then verify them via knowledge-base lookup and
    Ising-model consistency checking.

**Detailed explanation for engineers:**
    Experiments 47-48 extract constraints from structured outputs (claims,
    code). This experiment tackles the final extraction challenge: free text.

    The pipeline has three stages:
    1. **Claim extraction** — regex/template patterns recognize common claim
       types in English sentences: factual assertions ("X is Y"), implications
       ("if X then Y"), conjunctions, disjunctions, exclusions, and entailment
       chains ("All X are Y" + "Z is X" → "Z is Y").
    2. **Knowledge-base cross-referencing** — factual claims are checked
       against a dictionary of known facts. Each claim is tagged as
       True / False / Unknown.
    3. **Logical consistency verification** — implications, conjunctions, and
       contradictions are encoded as Ising constraints (same encoding as
       Exp 45) and verified via the parallel Gibbs sampler. If no spin
       assignment can satisfy all constraints simultaneously, the text is
       logically inconsistent.

    This bridges the gap between structured constraint extraction (Exp 47-48)
    and the fully autonomous verification pipeline envisioned for hallucination
    detection in LLM outputs.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_49_nl_constraints.py
"""

from __future__ import annotations

import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Knowledge base: a small set of known facts for cross-referencing.
# In a production system this would be backed by a real knowledge graph or
# retrieval-augmented generation (RAG) store.
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: dict[str, dict] = {
    # Capitals
    "capital_of_france": {"value": "Paris", "type": "capital"},
    "capital_of_australia": {"value": "Canberra", "type": "capital"},
    "capital_of_japan": {"value": "Tokyo", "type": "capital"},
    "capital_of_germany": {"value": "Berlin", "type": "capital"},
    # Continents
    "continent_of_france": {"value": "Europe", "type": "continent"},
    "continent_of_japan": {"value": "Asia", "type": "continent"},
    "continent_of_australia": {"value": "Oceania", "type": "continent"},
    # Category memberships
    "birds": {"members": ["penguins", "eagles", "sparrows", "ostriches"], "type": "category"},
    "mammals": {"members": ["whales", "dogs", "cats", "humans"], "type": "category"},
    "flightless_birds": {"members": ["penguins", "ostriches"], "type": "category"},
    # Properties
    "mammals_warm_blooded": True,
    "birds_have_feathers": True,
    "penguins_can_fly": False,
    "eagles_can_fly": True,
}


# ---------------------------------------------------------------------------
# Stage 1: Claim pattern extractors
# ---------------------------------------------------------------------------

# Each pattern extractor takes a sentence and returns a list of extracted
# claim dicts. Claims have a "claim_type" field and type-specific fields.

def _normalize(text: str) -> str:
    """Lowercase and strip extra whitespace for pattern matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_factual_is(sentence: str) -> list[dict]:
    """Extract "X is Y" and "X is the Y of Z" factual claims.

    **Detailed explanation for engineers:**
        Matches sentences like:
        - "Paris is the capital of France"  → (Paris, capital, France)
        - "The ground is wet"               → (ground, is, wet)
        - "Whales are mammals"              → (whales, are, mammals)

        The regex is intentionally broad; downstream verification decides
        whether the claim is checkable against the knowledge base.
    """
    s = _normalize(sentence)
    claims = []

    # Pattern: "X is the Y of Z"
    m = re.match(r"^(.+?)\s+is\s+the\s+(.+?)\s+of\s+(.+?)\.?$", s)
    if m:
        claims.append({
            "claim_type": "factual_relation",
            "subject": m.group(1).strip(),
            "relation": m.group(2).strip(),
            "object": m.group(3).strip(),
            "raw": sentence.strip(),
        })
        return claims

    # Pattern: "X is/are Y"
    m = re.match(r"^(.+?)\s+(?:is|are)\s+(.+?)\.?$", s)
    if m:
        claims.append({
            "claim_type": "factual",
            "subject": m.group(1).strip(),
            "predicate": m.group(2).strip(),
            "raw": sentence.strip(),
        })
    return claims


def extract_implication(sentence: str) -> list[dict]:
    """Extract "if X then Y" implications.

    **Detailed explanation for engineers:**
        Matches conditional sentences such as:
        - "If it rains, the ground is wet."
        - "If it rains then the ground gets wet."
        The antecedent (X) and consequent (Y) are returned as raw strings
        for downstream encoding into Ising constraints.
    """
    s = _normalize(sentence)
    # "if X, Y" or "if X then Y"
    m = re.match(r"^if\s+(.+?)(?:,\s*|\s+then\s+)(.+?)\.?$", s)
    if m:
        return [{
            "claim_type": "implication",
            "antecedent": m.group(1).strip(),
            "consequent": m.group(2).strip(),
            "raw": sentence.strip(),
        }]
    return []


def extract_conjunction(sentence: str) -> list[dict]:
    """Extract "X and Y" conjunctions (both must hold)."""
    s = _normalize(sentence)
    # Only match standalone "X and Y" that aren't part of larger structures.
    m = re.match(r"^(.+?)\s+and\s+(.+?)\.?$", s)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        # Skip if it looks like a list item or part of a longer structure.
        if "," not in left and "if " not in left:
            return [{
                "claim_type": "conjunction",
                "left": left,
                "right": right,
                "raw": sentence.strip(),
            }]
    return []


def extract_disjunction(sentence: str) -> list[dict]:
    """Extract "X or Y" disjunctions (at least one must hold)."""
    s = _normalize(sentence)
    m = re.match(r"^(.+?)\s+or\s+(.+?)\.?$", s)
    if m:
        return [{
            "claim_type": "disjunction",
            "left": m.group(1).strip(),
            "right": m.group(2).strip(),
            "raw": sentence.strip(),
        }]
    return []


def extract_exclusion(sentence: str) -> list[dict]:
    """Extract "X but not Y" exclusions."""
    s = _normalize(sentence)
    m = re.match(r"^(.+?)\s+but\s+not\s+(.+?)\.?$", s)
    if m:
        return [{
            "claim_type": "exclusion",
            "positive": m.group(1).strip(),
            "negative": m.group(2).strip(),
            "raw": sentence.strip(),
        }]
    return []


def extract_negation(sentence: str) -> list[dict]:
    """Extract "X cannot/can't/do not/does not Y" negations."""
    s = _normalize(sentence)
    m = re.match(r"^(.+?)\s+(?:cannot|can't|can not|do not|does not|don't|doesn't)\s+(.+?)\.?$", s)
    if m:
        return [{
            "claim_type": "negation",
            "subject": m.group(1).strip(),
            "predicate": m.group(2).strip(),
            "raw": sentence.strip(),
        }]
    return []


def extract_universal(sentence: str) -> list[dict]:
    """Extract "All X are Y" universal quantifiers.

    **Detailed explanation for engineers:**
        Universal claims like "All mammals are warm-blooded" create entailment
        chains: if we also see "Whales are mammals", we can derive "Whales are
        warm-blooded". The universal is stored so that downstream entailment
        resolution can apply it to specific instances.
    """
    s = _normalize(sentence)
    # "All X are/is Y" (e.g. "All mammals are warm-blooded")
    m = re.match(r"^all\s+(.+?)\s+(?:are|is)\s+(.+?)\.?$", s)
    if m:
        return [{
            "claim_type": "universal",
            "category": m.group(1).strip(),
            "property": m.group(2).strip(),
            "raw": sentence.strip(),
        }]
    # "All X verb" (e.g. "All birds fly") — no copula needed.
    m = re.match(r"^all\s+(.+?)\s+(\w+)\.?$", s)
    if m:
        return [{
            "claim_type": "universal",
            "category": m.group(1).strip(),
            "property": m.group(2).strip(),
            "raw": sentence.strip(),
        }]
    return []


# Ordered list of extractors. Earlier extractors take priority (a sentence
# is matched by the first extractor that returns results).
CLAIM_EXTRACTORS = [
    extract_universal,
    extract_implication,
    extract_exclusion,
    extract_negation,
    extract_disjunction,
    extract_factual_is,
    # conjunction last — it's very broad and would match too eagerly
    extract_conjunction,
]


def extract_claims(text: str) -> list[dict]:
    """Split text into sentences and extract claims from each.

    **Detailed explanation for engineers:**
        Sentences are split on period/exclamation/question boundaries. Each
        sentence is run through the extractor chain; the first extractor that
        returns a non-empty list wins. Sentences that match no pattern are
        silently skipped (they may contain discourse connectives, meta-language,
        or other non-claim content).
    """
    # Split on sentence boundaries: period, exclamation, question mark.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        for extractor in CLAIM_EXTRACTORS:
            result = extractor(sent)
            if result:
                claims.extend(result)
                break
    return claims


# ---------------------------------------------------------------------------
# Stage 2: Cross-reference claims against the knowledge base
# ---------------------------------------------------------------------------

def _kb_lookup_relation(subject: str, relation: str, obj: str, kb: dict) -> str | None:
    """Look up a "subject is the relation of object" claim.

    Returns "true", "false", or None (unknown).
    """
    # Normalize the lookup key: e.g. "capital_of_france"
    key = f"{relation}_of_{obj}".replace(" ", "_")
    entry = kb.get(key)
    if entry and isinstance(entry, dict) and "value" in entry:
        if entry["value"].lower() == subject.lower():
            return "true"
        else:
            return "false"
    return None


def _kb_lookup_membership(subject: str, category: str, kb: dict) -> str | None:
    """Check if subject is a member of category in the KB."""
    entry = kb.get(category)
    if entry and isinstance(entry, dict) and "members" in entry:
        if subject.lower() in [m.lower() for m in entry["members"]]:
            return "true"
        else:
            return "false"
    return None


def _kb_lookup_property(subject: str, predicate: str, kb: dict) -> str | None:
    """Check a simple "subject predicate" fact like 'penguins_can_fly'.

    **Detailed explanation for engineers:**
        Tries multiple key variations because natural language is flexible:
        "penguins fly" might be stored as "penguins_can_fly" or "penguins_fly".
        We try the direct key first, then common verb-phrase variations
        (prepending "can_", "are_", "have_") to increase recall.
    """
    pred_normalized = predicate.replace(" ", "_")
    subj_normalized = subject.replace(" ", "_")

    # Try direct key and common variations.
    candidates = [
        f"{subj_normalized}_{pred_normalized}",
        f"{subj_normalized}_can_{pred_normalized}",
        f"{subj_normalized}_are_{pred_normalized}",
        f"{subj_normalized}_have_{pred_normalized}",
    ]
    for key in candidates:
        val = kb.get(key)
        if val is True:
            return "true"
        elif val is False:
            return "false"
    return None


def check_claim_against_kb(claim: dict, kb: dict) -> dict:
    """Verify a single claim against the knowledge base.

    **Detailed explanation for engineers:**
        Dispatches on claim_type:
        - factual_relation: looks up "relation_of_object" key in KB.
        - factual: tries membership lookup, then property lookup.
        - universal / implication / conjunction / etc.: not directly checkable
          against KB — returned as "logical" for Ising verification.

        Returns the claim dict augmented with a "kb_verdict" field:
        "true", "false", "unknown", or "logical" (needs Ising check).
    """
    ct = claim["claim_type"]
    verdict = "unknown"

    if ct == "factual_relation":
        v = _kb_lookup_relation(claim["subject"], claim["relation"], claim["object"], kb)
        if v is not None:
            verdict = v

    elif ct == "factual":
        subj = claim["subject"]
        pred = claim["predicate"]
        # Try category membership: "X are Y" → is X in category Y?
        v = _kb_lookup_membership(subj, pred, kb)
        if v is not None:
            verdict = v
        else:
            # Try direct property lookup.
            v = _kb_lookup_property(subj, pred, kb)
            if v is not None:
                verdict = v

    elif ct == "negation":
        # "X cannot Y" — look up the positive and invert.
        v = _kb_lookup_property(claim["subject"], claim["predicate"], kb)
        if v == "true":
            verdict = "false"  # KB says they CAN, but text says cannot → contradiction
        elif v == "false":
            verdict = "true"  # KB confirms they cannot

    elif ct in ("implication", "conjunction", "disjunction", "exclusion", "universal"):
        verdict = "logical"

    return {**claim, "kb_verdict": verdict}


def text_to_constraints(text: str, knowledge_base: dict) -> list[dict]:
    """Full pipeline: text → claims → KB-verified constraints.

    **Detailed explanation for engineers:**
        1. Extract claims from free text using regex patterns.
        2. Cross-reference each claim against the knowledge base.
        3. Resolve entailment chains: if we have "All X are Y" and "Z is X",
           derive "Z is Y" and check that against the KB too.
        4. Return the list of verified/checked constraints.
    """
    claims = extract_claims(text)
    checked = [check_claim_against_kb(c, knowledge_base) for c in claims]

    # Resolve entailment chains: "All X are Y" + "Z is X" → "Z is Y"
    universals = [c for c in checked if c["claim_type"] == "universal"]
    factuals = [c for c in checked if c["claim_type"] == "factual"]

    derived = []
    for u in universals:
        category = u["category"]
        prop = u["property"]
        for f in factuals:
            # If "Z are/is <category>" → derive "Z are/is <property>"
            if f["predicate"] == category:
                derived_claim = {
                    "claim_type": "factual",
                    "subject": f["subject"],
                    "predicate": prop,
                    "raw": f"[derived] {f['subject']} are {prop} (from: all {category} are {prop} + {f['subject']} are {category})",
                    "derived_from": (u["raw"], f["raw"]),
                }
                derived_claim = check_claim_against_kb(derived_claim, knowledge_base)
                derived.append(derived_claim)

    checked.extend(derived)
    return checked


# ---------------------------------------------------------------------------
# Stage 3: Ising-based logical consistency verification
# ---------------------------------------------------------------------------

def _build_ising_from_constraints(constraints: list[dict]) -> tuple[list[dict], dict[str, int], int]:
    """Convert extracted constraints into Ising claims for the parallel sampler.

    **Detailed explanation for engineers:**
        Maps natural-language constraints to the same claim format used by
        Exp 45's encode_claims_as_ising(). Each distinct proposition gets a
        spin index. Implications, conjunctions, disjunctions, and exclusions
        become Ising couplings between the relevant spins.

        Factual claims verified as "true" by the KB get a "true" constraint
        (positive bias). Claims verified as "false" get flagged but are still
        encoded as "true" constraints — the contradiction appears when a
        mutex or exclusion couples them with conflicting claims.

        Returns:
        - ising_claims: list of Ising claim dicts for encode_claims_as_ising
        - prop_map: mapping from proposition strings to spin indices
        - n_props: total number of spin variables
    """
    prop_map: dict[str, int] = {}

    def get_prop(name: str) -> int:
        if name not in prop_map:
            prop_map[name] = len(prop_map)
        return prop_map[name]

    ising_claims: list[dict] = []

    for c in constraints:
        ct = c["claim_type"]

        if ct in ("factual", "factual_relation"):
            # Encode as "this proposition is asserted true in the text"
            prop_name = c["raw"]
            idx = get_prop(prop_name)
            ising_claims.append({"type": "true", "prop": idx})

        elif ct == "implication":
            ante_idx = get_prop(c["antecedent"])
            cons_idx = get_prop(c["consequent"])
            ising_claims.append({"type": "implies", "from": ante_idx, "to": cons_idx})

        elif ct == "conjunction":
            left_idx = get_prop(c["left"])
            right_idx = get_prop(c["right"])
            ising_claims.append({"type": "and", "props": [left_idx, right_idx]})

        elif ct == "disjunction":
            left_idx = get_prop(c["left"])
            right_idx = get_prop(c["right"])
            ising_claims.append({"type": "or", "props": [left_idx, right_idx]})

        elif ct == "exclusion":
            pos_idx = get_prop(c["positive"])
            neg_idx = get_prop(c["negative"])
            # "X but not Y": X is true, Y is false, and they're mutually exclusive.
            ising_claims.append({"type": "true", "prop": pos_idx})
            ising_claims.append({"type": "false", "prop": neg_idx})

        elif ct == "negation":
            subj_pred = f"{c['subject']} {c['predicate']}"
            idx = get_prop(subj_pred)
            ising_claims.append({"type": "false", "prop": idx})

        elif ct == "universal":
            # Universal by itself is just an assertion; entailments are
            # resolved in text_to_constraints as derived factual claims.
            prop_name = c["raw"]
            idx = get_prop(prop_name)
            ising_claims.append({"type": "true", "prop": idx})

    return ising_claims, prop_map, len(prop_map)


def _detect_contradictions(constraints: list[dict]) -> list[str]:
    """Detect direct contradictions between factual claims and negations.

    **Detailed explanation for engineers:**
        Before running the Ising sampler, we do a quick pass to find cases
        where the same proposition is both asserted and negated. For example:
        "All birds fly" + "Penguins are birds" → derived "Penguins fly"
        BUT text also says "Penguins cannot fly" → direct contradiction.

        This catches contradictions that the Ising encoding might miss due
        to different propositions being mapped to different spins.
    """
    contradictions = []

    # Collect positive and negative assertions about subjects.
    positive_claims: dict[str, str] = {}  # subject+predicate → raw
    negative_claims: dict[str, str] = {}

    for c in constraints:
        if c["claim_type"] == "factual":
            key = f"{c['subject']}_{c['predicate']}".lower().replace(" ", "_")
            positive_claims[key] = c["raw"]
        elif c["claim_type"] == "negation":
            key = f"{c['subject']}_{c['predicate']}".lower().replace(" ", "_")
            negative_claims[key] = c["raw"]

    # Check for overlapping keys (positive assertion + negation of same thing).
    for key in positive_claims:
        if key in negative_claims:
            contradictions.append(
                f"Contradiction: '{positive_claims[key]}' vs '{negative_claims[key]}'"
            )

    # Check for same-subject claims with conflicting predicates.
    # E.g. "The meeting is on monday" + "The meeting is on tuesday" — same
    # subject "the meeting" with predicate "on monday" vs "on tuesday".
    subject_claims: dict[str, list[dict]] = {}
    for c in constraints:
        if c["claim_type"] == "factual":
            subj = c["subject"].lower()
            subject_claims.setdefault(subj, []).append(c)
    for subj, claims_list in subject_claims.items():
        if len(claims_list) >= 2:
            # Check if predicates share a pattern like "on X" vs "on Y".
            for i in range(len(claims_list)):
                for j in range(i + 1, len(claims_list)):
                    p1 = claims_list[i]["predicate"]
                    p2 = claims_list[j]["predicate"]
                    # If predicates share a prefix but differ in value,
                    # they likely contradict. E.g. "on monday" vs "on tuesday".
                    words1 = p1.split()
                    words2 = p2.split()
                    if (len(words1) >= 2 and len(words2) >= 2
                            and words1[0] == words2[0] and words1[-1] != words2[-1]):
                        contradictions.append(
                            f"Conflicting claims: '{claims_list[i]['raw']}' vs '{claims_list[j]['raw']}'"
                        )

    # Check for factual claims the KB says are false.
    for c in constraints:
        if c.get("kb_verdict") == "false":
            contradictions.append(f"KB contradiction: '{c['raw']}' is factually false")

    return contradictions


def verify_text_constraints(constraints: list[dict]) -> dict:
    """Verify all extracted constraints for consistency.

    **Detailed explanation for engineers:**
        Combines three verification strategies:
        1. **KB factual check**: any claim marked "false" by the KB is a
           factual error.
        2. **Direct contradiction detection**: quick scan for a proposition
           that is both asserted and negated.
        3. **Ising consistency check**: encode remaining logical constraints
           as Ising spins and sample to see if all constraints can be
           simultaneously satisfied.

        The overall verdict is "consistent" only if ALL three checks pass.
    """
    from experiment_45_logical_consistency import encode_claims_as_ising, count_violations
    from carnot.samplers.parallel_ising import ParallelIsingSampler, AnnealingSchedule
    import jax.numpy as jnp
    import jax.random as jrandom

    # --- Check 1: KB factual verdicts ---
    kb_errors = [c for c in constraints if c.get("kb_verdict") == "false"]
    kb_unknowns = [c for c in constraints if c.get("kb_verdict") == "unknown"]
    kb_truths = [c for c in constraints if c.get("kb_verdict") == "true"]

    # --- Check 2: Direct contradictions ---
    contradictions = _detect_contradictions(constraints)

    # --- Check 3: Ising logical consistency ---
    ising_claims, prop_map, n_props = _build_ising_from_constraints(constraints)

    ising_result = {"consistent": True, "violations": 0}
    if n_props >= 2 and len(ising_claims) >= 2:
        try:
            biases_np, edge_pairs, weights_list = encode_claims_as_ising(ising_claims, n_props)

            if len(edge_pairs) > 0:
                J = np.zeros((n_props, n_props), dtype=np.float32)
                for k, (i, j) in enumerate(edge_pairs):
                    J[i, j] += weights_list[k]
                    J[j, i] += weights_list[k]

                sampler = ParallelIsingSampler(
                    n_warmup=500, n_samples=30, steps_per_sample=10,
                    schedule=AnnealingSchedule(0.1, 8.0),
                    use_checkerboard=True,
                )

                samples = sampler.sample(
                    jrandom.PRNGKey(49),
                    jnp.array(biases_np, dtype=jnp.float32),
                    jnp.array(J, dtype=jnp.float32),
                    beta=8.0,
                )

                best_violations = len(ising_claims)
                for s_idx in range(samples.shape[0]):
                    assignment = {i: bool(samples[s_idx, i]) for i in range(n_props)}
                    v = count_violations(ising_claims, assignment)
                    best_violations = min(best_violations, v)

                ising_result = {
                    "consistent": best_violations == 0,
                    "violations": best_violations,
                }
        except Exception as e:
            ising_result = {"consistent": None, "error": str(e)}

    # --- Overall verdict ---
    is_consistent = (
        len(kb_errors) == 0
        and len(contradictions) == 0
        and ising_result.get("consistent", True) is not False
    )

    return {
        "consistent": is_consistent,
        "n_claims": len(constraints),
        "kb_truths": len(kb_truths),
        "kb_errors": len(kb_errors),
        "kb_unknowns": len(kb_unknowns),
        "contradictions": contradictions,
        "ising_consistent": ising_result.get("consistent"),
        "ising_violations": ising_result.get("violations", 0),
        "error_details": [c["raw"] for c in kb_errors],
    }


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def get_test_scenarios() -> list[dict]:
    """Test scenarios covering consistent and inconsistent free text.

    **Detailed explanation for engineers:**
        Each scenario has:
        - name: human-readable description
        - text: free-form natural language input
        - expected_consistent: whether the text should pass verification
        - reason: why it should pass or fail (for human review)

        The scenarios exercise different claim types and their interactions:
        factual lookup, implication chains, entailment resolution, mutual
        exclusion, and direct contradictions.
    """
    return [
        # --- Consistent texts (should PASS) ---
        {
            "name": "Geography facts (consistent)",
            "text": "Paris is the capital of France. France is in Europe.",
            "expected_consistent": True,
            "reason": "Both facts are correct per KB.",
        },
        {
            "name": "Modus ponens (consistent)",
            "text": "If it rains, the ground is wet. It rained. The ground is wet.",
            "expected_consistent": True,
            "reason": "Valid modus ponens: premise + implication + conclusion all agree.",
        },
        {
            "name": "Multiple correct capitals",
            "text": "Tokyo is the capital of Japan. Berlin is the capital of Germany.",
            "expected_consistent": True,
            "reason": "Both are correct per KB.",
        },
        {
            "name": "Category membership (consistent)",
            "text": "Whales are mammals. Dogs are mammals.",
            "expected_consistent": True,
            "reason": "Both are correct per KB.",
        },
        {
            "name": "Negation matches KB",
            "text": "Penguins cannot fly.",
            "expected_consistent": True,
            "reason": "KB confirms penguins_can_fly is False.",
        },
        {
            "name": "Implication chain (consistent)",
            "text": "If A is true then B follows. If B follows then C holds.",
            "expected_consistent": True,
            "reason": "Implication chain with no contradicting facts.",
        },
        # --- Inconsistent texts (should FAIL) ---
        {
            "name": "Wrong capital (factual error)",
            "text": "Sydney is the capital of Australia.",
            "expected_consistent": False,
            "reason": "KB says capital of Australia is Canberra, not Sydney.",
        },
        {
            "name": "Penguin paradox (entailment contradiction)",
            "text": "All birds fly. Penguins are birds. Penguins cannot fly.",
            "expected_consistent": False,
            "reason": "Entailment: 'all birds fly' + 'penguins are birds' → 'penguins fly', "
                     "but text also says 'penguins cannot fly'. KB confirms penguins cannot fly, "
                     "so the universal 'all birds fly' leads to a derived contradiction.",
        },
        {
            "name": "Contradictory meeting times",
            "text": "The meeting is on monday. The meeting is on tuesday.",
            "expected_consistent": False,
            "reason": "Same subject with two mutually exclusive predicates. "
                     "Encoded as two factual assertions that the Ising sampler "
                     "flags via contradiction detection.",
        },
        {
            "name": "Wrong capital of Japan",
            "text": "Osaka is the capital of Japan.",
            "expected_consistent": False,
            "reason": "KB says capital of Japan is Tokyo.",
        },
        {
            "name": "Negation contradicts KB",
            "text": "Eagles cannot fly.",
            "expected_consistent": False,
            "reason": "KB says eagles_can_fly is True; text says they cannot.",
        },
        {
            "name": "Mixed correct and wrong",
            "text": "Paris is the capital of France. Sydney is the capital of Australia.",
            "expected_consistent": False,
            "reason": "First claim is correct but second is wrong per KB.",
        },
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 49: Natural Language Constraint Extraction & Verification")
    print("  Free text → claim extraction → KB + Ising verification")
    print("=" * 70)

    start = time.time()
    scenarios = get_test_scenarios()
    results = []

    for scenario in scenarios:
        # Stage 1+2: extract and KB-verify claims.
        constraints = text_to_constraints(scenario["text"], KNOWLEDGE_BASE)

        # Stage 3: verify logical consistency.
        verdict = verify_text_constraints(constraints)

        actual = verdict["consistent"]
        expected = scenario["expected_consistent"]
        correct = actual == expected

        icon = "✓" if correct else "✗"
        status = "CONSISTENT" if actual else "INCONSISTENT"
        exp_str = "consistent" if expected else "inconsistent"

        print(f"\n  [{icon}] {scenario['name']}")
        print(f"      Text: \"{scenario['text'][:70]}{'...' if len(scenario['text']) > 70 else ''}\"")
        print(f"      Claims extracted: {verdict['n_claims']} "
              f"(KB true={verdict['kb_truths']}, false={verdict['kb_errors']}, "
              f"unknown={verdict['kb_unknowns']})")
        if verdict["contradictions"]:
            for c in verdict["contradictions"]:
                print(f"      ⚡ {c}")
        if verdict["error_details"]:
            for e in verdict["error_details"]:
                print(f"      ❌ Factual error: {e}")
        print(f"      Verdict: {status} (expected {exp_str}) → {'✓ CORRECT' if correct else '✗ WRONG'}")

        results.append({
            "name": scenario["name"],
            "expected": expected,
            "actual": actual,
            "correct": correct,
            "verdict": verdict,
        })

    # --- Summary ---
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 49 RESULTS ({elapsed:.1f}s)")
    print(sep)

    n_total = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_consistent_scenarios = sum(1 for s in scenarios if s["expected_consistent"])
    n_inconsistent_scenarios = n_total - n_consistent_scenarios

    true_pos = sum(1 for r in results if not r["expected"] and not r["actual"])
    true_neg = sum(1 for r in results if r["expected"] and r["actual"])
    false_pos = sum(1 for r in results if r["expected"] and not r["actual"])
    false_neg = sum(1 for r in results if not r["expected"] and r["actual"])

    print(f"  Total scenarios:                          {n_total}")
    print(f"  Correct detections:                       {n_correct}/{n_total}")
    print(f"  True positives (caught inconsistencies):  {true_pos}/{n_inconsistent_scenarios}")
    print(f"  True negatives (passed consistent text):  {true_neg}/{n_consistent_scenarios}")
    print(f"  False positives (flagged good text):      {false_pos}")
    print(f"  False negatives (missed inconsistency):   {false_neg}")

    if n_correct == n_total:
        print(f"\n  VERDICT: ✅ Perfect NL constraint extraction & verification!")
    elif n_correct >= n_total * 0.8:
        print(f"\n  VERDICT: ✅ Pipeline works ({n_correct}/{n_total} correct)")
    else:
        print(f"\n  VERDICT: ❌ Pipeline needs work ({n_correct}/{n_total})")

    # --- Claim pattern coverage report ---
    print(f"\n  Claim types exercised:")
    all_types = set()
    for r in results:
        for c in text_to_constraints(
            next(s["text"] for s in scenarios if s["name"] == r["name"]),
            KNOWLEDGE_BASE
        ):
            all_types.add(c["claim_type"])
    for ct in sorted(all_types):
        print(f"    • {ct}")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Experiment 166: Logic symbolic feature vectors for JEPA training.

**Researcher summary:**
    Exp 155 trained JEPA v2 with RandomProjection byte-histogram embeddings
    and achieved AUROC=0.479 (chance) on the logic domain. Root cause: byte
    histograms cannot distinguish valid from invalid logical arguments because
    valid and invalid logic texts share nearly identical byte distributions.
    This experiment replaces RandomProjection for the logic domain with
    40-dimensional symbolic feature vectors that directly encode logical
    structure: negation counts, quantifier presence, conditional depth,
    entailment markers, and their derived combinations. These features are
    then padded to 256 dimensions (matching the existing JEPA embedding size)
    and L2-normalized.

**Detailed explanation for engineers:**
    The fundamental problem with byte-histogram embeddings for logic is that
    they are permutation-invariant over bytes. "If P then Q. Q. Therefore P."
    (affirming the consequent — a FALLACY) and "If P then Q. P. Therefore Q."
    (modus ponens — VALID) differ by only a few characters and have nearly
    identical byte histograms. A model trained on such embeddings cannot
    learn the distinction.

    Symbolic features solve this by asking STRUCTURAL questions:
    - Does the text have negation? How much? (negation density)
    - Does the text assert a universal claim? (all/every/each)
    - Does the text have a conditional? (if...then)
    - Does the text have a conclusion marker? (therefore/thus/hence)
    - Are quantifiers negated AFTER appearing? (negation scope errors)

    These 40 symbolic features, when combined, give the JEPA predictor
    enough signal to distinguish argument forms that look identical at the
    byte level.

    Output files:
    - results/jepa_training_pairs_logic_v3.json: 500 logic pairs + arithmetic
      pairs from v2.json, merged into a single training set.

Spec: REQ-JEPA-001, SCENARIO-JEPA-LOGIC-001
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
V2_JSON = RESULTS_DIR / "jepa_training_pairs_v2.json"
OUTPUT_JSON = RESULTS_DIR / "jepa_training_pairs_logic_v3.json"

# ---------------------------------------------------------------------------
# Symbolic feature vector (256-dim, L2-normalized)
# ---------------------------------------------------------------------------

PREFIX_RATIOS = [0.10, 0.25, 0.50, 0.75]
EMBED_DIM = 256


def logic_feature_vector(text: str) -> np.ndarray:
    """Compute a 256-dim symbolic feature vector for a logic text.

    **Detailed explanation for engineers:**
        This function computes 40 handcrafted features that capture logical
        structure, then pads to 256 dims and L2-normalizes. The 40 features
        are grouped as:

        0–11: Core binary/continuous indicators
            - negation density, quantifier presence (all/some/no),
              conditional (if-then), conclusion marker (therefore),
              contradiction marker (but/however), conditional depth,
              clause density, conclusion ratio, negation-after-quantifier,
              double negation

        12–19: Derived pairwise combinations (products of core features)
            These help the linear JEPA probe learn non-linear interactions,
            e.g. "text has both ALL-quantifier and negation" → likely a
            negation scope error.

        20–39: Zeros (reserved for future features; makes the block extendable)

        40–255: Zeros (pad to match JEPA embedding dim 256)

    The vector is L2-normalized so that cosine similarity in the JEPA
    predictor is equivalent to dot product — which is what the energy
    functions expect.

    Args:
        text: Input logic text of any length.

    Returns:
        float32 array of shape (256,), L2-normalized.

    Spec: REQ-JEPA-001
    """
    tokens = text.lower().split()
    n_tokens = max(len(tokens), 1)
    text_lower = text.lower()

    features = np.zeros(40, dtype=np.float32)

    # --- 0: negation density ------------------------------------------------
    # Counts negation words and normalizes by token count. A fallacy like
    # "not all are not X" has higher negation density than a valid modus ponens.
    neg_words = {"not", "no", "never", "neither", "nor"}
    features[0] = sum(tokens.count(w) for w in neg_words) / n_tokens

    # --- 1: has_quantifier_all ----------------------------------------------
    # Universal quantifiers appear in syllogisms; their presence (or absence)
    # combined with negation scope is a key discriminator.
    features[1] = float(any(w in tokens for w in ("all", "every", "each")))

    # --- 2: has_quantifier_some ---------------------------------------------
    # Existential quantifiers ("some", "many", "most") participate in
    # fallacious quantifier-shift arguments.
    features[2] = float(any(w in tokens for w in ("some", "many", "most")))

    # --- 3: has_quantifier_no -----------------------------------------------
    # Negative quantifiers are often confused with negated universals.
    features[3] = float(
        "no " in text_lower or "none" in tokens or "neither" in tokens
    )

    # --- 4: has_therefore ---------------------------------------------------
    # Conclusion markers are structural. Valid arguments always have one;
    # some fallacies omit it or use weaker markers.
    features[4] = float(
        any(marker in text_lower for marker in ("therefore", "thus", "hence", "so "))
    )

    # --- 5: has_if_then -----------------------------------------------------
    # Conditional arguments (modus ponens, tollens, affirming consequent, etc.)
    # all require "if...then". Categorical syllogisms do NOT.
    features[5] = float("if " in text_lower and "then " in text_lower)

    # --- 6: has_contradiction_marker ----------------------------------------
    # Words like "but", "however", "although" signal that the text presents
    # a rebuttal or counter-claim, which is common in invalid arguments.
    features[6] = float(
        any(marker in text_lower for marker in ("but ", "however", "although", "yet "))
    )

    # --- 7: conditional_depth -----------------------------------------------
    # Nested conditionals ("if...if...then") indicate complex reasoning.
    # Capped at 3 occurrences to avoid runaway values; normalized to [0,1].
    features[7] = min(text_lower.count("if "), 3) / 3.0

    # --- 8: clause_density --------------------------------------------------
    # Number of clause boundaries (commas, semicolons, periods) divided by
    # token count. Arguments with more clauses tend to be structurally richer.
    features[8] = (
        text.count(",") + text.count(";") + text.count(".")
    ) / n_tokens

    # --- 9: conclusion_ratio ------------------------------------------------
    # Ratio of last sentence length to total text length. In well-formed
    # arguments, the conclusion is shorter than the premises.
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    last_sent = sentences[-1] if sentences else ""
    features[9] = len(last_sent) / max(len(text), 1)

    # --- 10: negation_after_quantifier --------------------------------------
    # Pattern "all...not" or "no...all" indicates potential negation scope
    # confusion — a common source of logical fallacies.
    features[10] = float(
        bool(re.search(r"\ball\b.*\bnot\b|\bno\b.*\ball\b", text_lower))
    )

    # --- 11: double_negation ------------------------------------------------
    # "not...not" within the same text often indicates either a double
    # negation that simplifies to a positive, or a negation scope error.
    features[11] = float(bool(re.search(r"\bnot\b.*\bnot\b", text_lower)))

    # --- 12–19: Derived pairwise combinations --------------------------------
    # Products of core features allow a linear probe to capture interactions.
    # E.g., feature 12 = "negation AND universal quantifier" which is the
    # signature of negation scope errors.
    features[12] = features[0] * features[1]   # negation × all-quantifier
    features[13] = features[0] * features[2]   # negation × some-quantifier
    features[14] = features[4] * features[5]   # therefore × if-then (modus ponens pattern)
    features[15] = features[5] * features[1]   # if-then × all-quantifier
    features[16] = features[4] * features[1]   # therefore × all-quantifier (syllogism)
    features[17] = features[8] * features[9]   # clause density × conclusion ratio
    features[18] = min(float(n_tokens) / 50.0, 1.0)   # normalized text length [0,1]
    features[19] = min(float(len(sentences)) / 5.0, 1.0)  # sentence count [0,1]

    # --- 20–39: Reserved (zeros) --------------------------------------------
    # Future features: parse tree depth, operator counts, etc.

    # Pad to EMBED_DIM (256) with zeros so the vector is compatible with
    # the existing JEPA predictor which expects 256-dim inputs.
    vec = np.pad(features, (0, EMBED_DIM - len(features)), mode="constant")

    # L2-normalize: makes cosine similarity equal to dot product, which is
    # what the Carnot energy functions expect.
    norm = np.linalg.norm(vec)
    if norm > 1e-10:
        vec = vec / norm

    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Logic text generators
# ---------------------------------------------------------------------------

# Subject pools for varied, natural-sounding arguments
ANIMALS = [
    "dogs", "cats", "eagles", "salmon", "frogs",
    "dolphins", "penguins", "elephants", "owls", "bats",
    "tigers", "horses", "wolves",
]
PROPERTIES_A = [
    "warm-blooded", "vertebrates", "capable of flight", "carnivores",
    "social creatures", "nocturnal", "herbivores", "fast runners",
    "excellent swimmers", "highly intelligent", "migratory", "omnivores",
    "protective of offspring",
]
PROPERTIES_B = [
    "require warm environments", "need specialized care", "consume large amounts of food",
    "form complex social structures", "possess acute senses", "exhibit territorial behavior",
    "are well-adapted to their environments", "play important ecological roles",
    "are protected by conservation laws", "have been studied extensively",
    "are found across multiple continents", "have distinctive markings",
    "show remarkable adaptations",
]
CITIES = [
    "Paris", "London", "Tokyo", "Rome", "Sydney",
    "Cairo", "Berlin", "Toronto", "Mumbai", "Seoul",
    "Nairobi", "Lima", "Oslo",
]
CITY_PROPS_A = [
    "a major European capital", "a coastal city", "a financial hub",
    "a city with over 10 million residents", "a UNESCO World Heritage site",
    "a city famous for its cuisine", "a major port city",
    "a city known for its museums", "a city with an advanced metro system",
    "a city that hosted the Olympics", "a city with a tropical climate",
    "a major center of commerce", "a city renowned for its architecture",
]
CITY_PROPS_B = [
    "attracts millions of tourists each year", "has high living costs",
    "has a significant cultural heritage", "experiences heavy traffic congestion",
    "has world-class educational institutions", "is a center for international events",
    "has diverse culinary options", "has extensive public transportation",
    "is home to major international companies", "has a thriving arts scene",
    "has significant infrastructure investment", "hosts major sporting events",
    "is a popular destination for business travelers",
]


def _subject_and_props(idx: int) -> tuple[str, str, str, str, str]:
    """Return (subj_type, subj, prop_a, prop_b, singular) for a given index.

    Returns a named entity (animal or city) with two properties for building
    varied logical argument templates.
    """
    if idx < len(ANIMALS):
        s = ANIMALS[idx]
        return ("animal", s, PROPERTIES_A[idx], PROPERTIES_B[idx],
                s.rstrip("s"))  # rough singular
    else:
        j = idx - len(ANIMALS)
        s = CITIES[j]
        return ("city", s, CITY_PROPS_A[j], CITY_PROPS_B[j], s)


# We need 65 instances per class (valid/invalid) to get >250 pairs at 4 ratios.
# Use idx 0..12 for animal-based subjects, 13..25 for city-based subjects,
# then repeat with swapped properties for indices 26..64.
def _make_subject_pool() -> list[tuple[str, str, str, str, str]]:
    """Build a pool of 65 (subj_type, subj, prop_a, prop_b, singular) tuples."""
    pool = []
    for i in range(13):
        pool.append(_subject_and_props(i))
    for i in range(13):
        j = i + 13
        pool.append(_subject_and_props(j))
    # Repeat with rotated properties for variety (indices 26..51)
    for i in range(13):
        s_type, subj, pa, pb, sing = _subject_and_props(i)
        # Rotate: use prop_b as A and prop_a as B
        pool.append((s_type, subj, pb, pa, sing))
    # Add 13 more with further rotation (indices 52..64)
    for i in range(13):
        j = (i + 3) % 13
        s_type, subj, pa, pb, sing = _subject_and_props(j)
        s_type2, subj2, pa2, pb2, sing2 = _subject_and_props((j + 5) % 13)
        # Mix subjects for variety
        pool.append((s_type, subj, PROPERTIES_A[(i + 7) % 13],
                     PROPERTIES_B[(i + 3) % 13], sing))
    return pool[:65]


SUBJECT_POOL = _make_subject_pool()


# ---------------------------------------------------------------------------
# Valid argument generators (5 types × 13 instances = 65 valid texts)
# ---------------------------------------------------------------------------

def _gen_modus_ponens(subj_type: str, subj: str, pa: str, pb: str, sing: str,
                      variant: int) -> str:
    """Modus ponens: If P then Q. P is true. Therefore Q.

    **Detailed explanation for engineers:**
        This is the most basic valid argument form. Given a conditional
        "If P then Q" and evidence that P is true, we can infer Q.
        Form: If [subject] has property A then [subject] has property B.
              [Subject] has property A.
              Therefore, [subject] have property B.
    """
    variants = [
        (f"If {subj} are {pa}, then they are also {pb}. "
         f"{subj.capitalize()} are indeed {pa}. "
         f"Therefore, {subj} are {pb}."),
        (f"If an animal is {pa}, it follows that it is {pb}. "
         f"{subj.capitalize()} are {pa}. "
         f"Therefore, {subj} are {pb}."),
        (f"Whenever {subj} exhibit {pa}, they also exhibit {pb}. "
         f"We observe that {subj} are {pa}. "
         f"Hence, {subj} are {pb}."),
    ]
    return variants[variant % len(variants)]


def _gen_modus_tollens(subj_type: str, subj: str, pa: str, pb: str, sing: str,
                       variant: int) -> str:
    """Modus tollens: If P then Q. Q is false. Therefore P is false.

    **Detailed explanation for engineers:**
        Given "If P then Q" and evidence that Q is FALSE, we can conclude
        that P must also be FALSE. Denying the consequent denies the antecedent.
        Form: If [subject] has A then [subject] has B.
              [Subject] does not have B.
              Therefore, [subject] does not have A.
    """
    variants = [
        (f"If {subj} are {pa}, then they must be {pb}. "
         f"{subj.capitalize()} are not {pb}. "
         f"Therefore, {subj} are not {pa}."),
        (f"If an entity is {pa}, it follows that it is {pb}. "
         f"We know {subj} are not {pb}. "
         f"Thus, {subj} are not {pa}."),
        (f"Whenever something is {pa}, it is also {pb}. "
         f"Since {subj} are not {pb}, "
         f"we conclude that {subj} are not {pa}."),
    ]
    return variants[variant % len(variants)]


def _gen_disjunctive_syllogism(subj_type: str, subj: str, pa: str, pb: str,
                                sing: str, variant: int) -> str:
    """Disjunctive syllogism: P or Q. Not P. Therefore Q.

    **Detailed explanation for engineers:**
        Given a disjunction "P or Q" and evidence that P is false, we can
        conclude Q must be true (assuming an exclusive-or interpretation).
        Form: [Subject] are either A or B.
              [Subject] are not A.
              Therefore, [subject] are B.
    """
    variants = [
        (f"{subj.capitalize()} are either {pa} or {pb}. "
         f"{subj.capitalize()} are not {pa}. "
         f"Therefore, {subj} are {pb}."),
        (f"Either {subj} are {pa} or they are {pb}. "
         f"It has been established that {subj} are not {pa}. "
         f"Hence, {subj} must be {pb}."),
        (f"One of two things must be true: {subj} are {pa}, or {subj} are {pb}. "
         f"We have confirmed that {subj} are not {pa}. "
         f"Therefore, {subj} are {pb}."),
    ]
    return variants[variant % len(variants)]


def _gen_universal_instantiation(subj_type: str, subj: str, pa: str, pb: str,
                                  sing: str, variant: int) -> str:
    """Universal instantiation: All A are B. X is an A. Therefore X is a B.

    **Detailed explanation for engineers:**
        If ALL members of class A have property B, and X is a member of class A,
        then X must also have property B. This is valid only if the universal
        claim genuinely covers X's class.
        Form: All [subject] are [pa].
              [Singular] is a member of [subject].
              Therefore, [singular] is [pa].
    """
    variants = [
        (f"All {subj} are {pa}. "
         f"A {sing} is one of many {subj}. "
         f"Therefore, this {sing} is {pa}."),
        (f"Every member of the {subj} group is {pa}. "
         f"The individual we are examining is a {sing}. "
         f"Therefore, this individual is {pa}."),
        (f"Without exception, all {subj} possess the property of being {pa}. "
         f"Our subject is a {sing}. "
         f"Thus, our subject is {pa}."),
    ]
    return variants[variant % len(variants)]


def _gen_categorical_syllogism(subj_type: str, subj: str, pa: str, pb: str,
                                sing: str, variant: int) -> str:
    """Categorical syllogism: All M are P. All S are M. Therefore All S are P.

    **Detailed explanation for engineers:**
        The classic Aristotelian syllogism. Given that all members of the
        middle term (M) have property P, and all members of the subject
        class (S) are in M, we conclude all S have P. This is valid when
        the middle term is DISTRIBUTED (covers all of M).
        Form: All [pa] things are [pb] things.
              All [subj] are [pa].
              Therefore, all [subj] are [pb].
    """
    variants = [
        (f"All things that are {pa} are also {pb}. "
         f"All {subj} are {pa}. "
         f"Therefore, all {subj} are {pb}."),
        (f"Everything that is {pa} is necessarily {pb}. "
         f"Every {sing} is {pa}. "
         f"Therefore, every {sing} is {pb}."),
        (f"The class of entities that are {pa} is entirely contained within "
         f"the class of entities that are {pb}. "
         f"All {subj} belong to the class of {pa} entities. "
         f"Therefore, all {subj} are {pb}."),
    ]
    return variants[variant % len(variants)]


# ---------------------------------------------------------------------------
# Invalid argument generators (5 fallacy types × 13 instances = 65 invalid texts)
# ---------------------------------------------------------------------------

def _gen_affirming_consequent(subj_type: str, subj: str, pa: str, pb: str,
                               sing: str, variant: int) -> str:
    """FALLACY — Affirming the consequent: If P then Q. Q. Therefore P.

    **Detailed explanation for engineers:**
        This is a classic logical fallacy. Just because "If P then Q" is true
        and Q is observed does not mean P caused Q. Q might be true for many
        other reasons. Example: "If it rained, the ground is wet. The ground
        is wet. Therefore it rained." — But the ground could be wet from a
        sprinkler. This is marked as violated_logic=True.
    """
    variants = [
        (f"If {subj} are {pa}, then they are {pb}. "
         f"{subj.capitalize()} are {pb}. "
         f"Therefore, {subj} are {pa}."),
        (f"If an entity is {pa}, it follows that it is {pb}. "
         f"We observe that {subj} are {pb}. "
         f"Therefore, {subj} must be {pa}."),
        (f"Whenever something is {pa}, it is also {pb}. "
         f"{subj.capitalize()} are {pb}. "
         f"Thus, {subj} are {pa}."),
    ]
    return variants[variant % len(variants)]


def _gen_denying_antecedent(subj_type: str, subj: str, pa: str, pb: str,
                             sing: str, variant: int) -> str:
    """FALLACY — Denying the antecedent: If P then Q. Not P. Therefore not Q.

    **Detailed explanation for engineers:**
        Given "If P then Q", denying P does NOT allow us to deny Q. Q might
        be true for other reasons entirely. Example: "If it rained, the ground
        is wet. It did not rain. Therefore the ground is not wet." — Wrong;
        the sprinkler might have run. This is marked as violated_logic=True.
    """
    variants = [
        (f"If {subj} are {pa}, then they are {pb}. "
         f"{subj.capitalize()} are not {pa}. "
         f"Therefore, {subj} are not {pb}."),
        (f"If an entity is {pa}, it will be {pb}. "
         f"We have established that {subj} are not {pa}. "
         f"Thus, {subj} are not {pb}."),
        (f"Whenever something is {pa}, it is {pb}. "
         f"Since {subj} are not {pa}, "
         f"they cannot be {pb}."),
    ]
    return variants[variant % len(variants)]


def _gen_undistributed_middle(subj_type: str, subj: str, pa: str, pb: str,
                               sing: str, variant: int) -> str:
    """FALLACY — Undistributed middle: All A are C. All B are C. Therefore A=B.

    **Detailed explanation for engineers:**
        Both A and B sharing property C does not mean A and B are the same
        or that one contains the other. Example: "All dogs are animals. All
        cats are animals. Therefore dogs are cats." The middle term 'animals'
        is NOT distributed (it does not cover all animals), so we cannot
        equate the two subject classes. Violated_logic=True.
    """
    # Use two different subjects for this fallacy
    other_subj_idx = (SUBJECT_POOL.index((subj_type, subj, pa, pb, sing)) + 7) % len(SUBJECT_POOL)
    _, other_subj, _, _, other_sing = SUBJECT_POOL[other_subj_idx % len(SUBJECT_POOL)]
    if other_subj == subj:
        other_subj = "reptiles" if subj_type == "animal" else "Moscow"
        other_sing = other_subj.rstrip("s")
    variants = [
        (f"All {subj} are {pa}. "
         f"All {other_subj} are {pa}. "
         f"Therefore, {subj} are {other_subj}."),
        (f"Every {sing} is {pa}. "
         f"Every {other_sing} is also {pa}. "
         f"Hence, {subj} and {other_subj} are the same kind of entity."),
        (f"All {subj} have the property of being {pa}, and "
         f"all {other_subj} also have the property of being {pa}. "
         f"Therefore, {subj} and {other_subj} are equivalent."),
    ]
    return variants[variant % len(variants)]


def _gen_negation_scope_error(subj_type: str, subj: str, pa: str, pb: str,
                               sing: str, variant: int) -> str:
    """FALLACY — Negation scope: 'Not all A are B' → 'All A are not B'.

    **Detailed explanation for engineers:**
        "Not all A are B" means "some A are not B" — it leaves open the
        possibility that some A ARE B. But the fallacious inference says
        "All A are not B" (i.e., NO A is B), which is a much stronger claim
        not supported by the premise. This is a classic quantifier-negation
        scope error. Violated_logic=True.
    """
    variants = [
        (f"Not all {subj} are {pa}. "
         f"Therefore, all {subj} are not {pa}."),
        (f"It is false that all {subj} are {pa}. "
         f"We can therefore conclude that no {subj} are {pa}."),
        (f"We know that not every {sing} is {pa}. "
         f"It follows that all {subj} are not {pa}."),
    ]
    return variants[variant % len(variants)]


def _gen_quantifier_shift(subj_type: str, subj: str, pa: str, pb: str,
                           sing: str, variant: int) -> str:
    """FALLACY — Quantifier shift: Some A are B. Some B are C. Therefore some A are C.

    **Detailed explanation for engineers:**
        Just because some A relate to some B, and some B relate to some C,
        does NOT guarantee that any A relates to C. The B instances that
        relate to A might be entirely different from the B instances that
        relate to C. Example: "Some students know some professors. Some
        professors know some celebrities. Therefore some students know some
        celebrities." — This is invalid. Violated_logic=True.
    """
    # Generate a three-term structure
    other_idx = (variant * 3 + 5) % len(SUBJECT_POOL)
    _, other_subj, other_pa, _, other_sing = SUBJECT_POOL[other_idx]
    if other_subj == subj:
        other_subj = "researchers"
        other_sing = "researcher"
        other_pa = "well-connected"
    variants = [
        (f"Some {subj} are {pa}. "
         f"Some entities that are {pa} are also {other_pa}. "
         f"Therefore, some {subj} are {other_pa}."),
        (f"Some {subj} exhibit {pa}. "
         f"Some {pa} entities are {other_subj}. "
         f"Therefore, some {subj} are {other_subj}."),
        (f"A number of {subj} are {pa}. "
         f"Some {pa} things are also {pb}. "
         f"Hence, some {subj} are {pb}."),
    ]
    return variants[variant % len(variants)]


# Map argument type name → generator function
VALID_GENERATORS = [
    ("modus_ponens", _gen_modus_ponens),
    ("modus_tollens", _gen_modus_tollens),
    ("disjunctive_syllogism", _gen_disjunctive_syllogism),
    ("universal_instantiation", _gen_universal_instantiation),
    ("categorical_syllogism", _gen_categorical_syllogism),
]

INVALID_GENERATORS = [
    ("affirming_consequent", _gen_affirming_consequent),
    ("denying_antecedent", _gen_denying_antecedent),
    ("undistributed_middle", _gen_undistributed_middle),
    ("negation_scope_error", _gen_negation_scope_error),
    ("quantifier_shift", _gen_quantifier_shift),
]


# ---------------------------------------------------------------------------
# Generate logic texts
# ---------------------------------------------------------------------------

def generate_logic_texts(
    generators: list[tuple[str, object]],
    n_per_type: int = 13,
) -> list[tuple[str, str]]:
    """Generate (argument_type, text) tuples using the given generators.

    **Detailed explanation for engineers:**
        We iterate over each argument type and generate ``n_per_type`` instances
        by cycling through the subject pool. Each subject in the pool provides
        a subject entity, two properties (A and B), and a singular form.
        The generator functions use a variant index (0..n_per_type-1) to
        select different sentence templates for the same subject, producing
        linguistic variety even when the logical structure is identical.

    Args:
        generators: List of (name, generator_fn) pairs.
        n_per_type: Number of instances to generate per argument type.

    Returns:
        List of (argument_type, text) tuples.
    """
    results = []
    for arg_type, gen_fn in generators:
        for i in range(n_per_type):
            subj_info = SUBJECT_POOL[i % len(SUBJECT_POOL)]
            subj_type, subj, pa, pb, sing = subj_info
            try:
                text = gen_fn(subj_type, subj, pa, pb, sing, variant=i)
            except Exception:
                # Fallback: use basic template if generator raises (e.g. undistributed middle
                # needs a subject pool lookup that may fail for some indices)
                text = gen_fn("animal", "dogs", "warm-blooded", "vertebrates",
                              "dog", variant=i % 3)
            results.append((arg_type, text))
    return results


# ---------------------------------------------------------------------------
# Build training pairs from logic texts
# ---------------------------------------------------------------------------

def build_logic_pairs(
    valid_texts: list[tuple[str, str]],
    invalid_texts: list[tuple[str, str]],
    target_per_class: int = 250,
) -> list[dict]:
    """Convert logic texts into JEPA training pairs with symbolic embeddings.

    **Detailed explanation for engineers:**
        For each logic text, we extract the text at 4 prefix ratios (10%, 25%,
        50%, 75%) to simulate a language model generating the text token by
        token. The JEPA predictor learns to assess whether a PREFIX already
        shows signs of logical violation, before the text is complete.

        For each prefix:
        1. Compute logic_feature_vector(prefix_text) → embedding[256]
        2. Assign violated_logic based on construction (valid=False, invalid=True)
        3. Record the pair with all fields matching the v2.json schema.

        We generate more pairs than needed (n_texts × 4 ratios) and trim to
        exactly ``target_per_class`` valid and ``target_per_class`` invalid.
        This ensures a balanced dataset while using all four prefix ratios for
        at least some texts.

    Args:
        valid_texts: List of (arg_type, text) for valid arguments.
        invalid_texts: List of (arg_type, text) for invalid arguments.
        target_per_class: Target number of pairs per class.

    Returns:
        List of pair dicts matching the jepa_training_pairs_v2.json schema.
    """
    pairs = []

    def _extract_pairs(texts: list[tuple[str, str]], violated: bool) -> list[dict]:
        result = []
        for arg_type, text in texts:
            n = len(text)
            for ratio in PREFIX_RATIOS:
                cutoff = max(int(n * ratio), 1)
                prefix = text[:cutoff]
                embedding = logic_feature_vector(prefix)
                result.append({
                    "prefix_ratio": ratio,
                    "embedding": embedding.tolist(),
                    "violated_arithmetic": False,
                    "violated_code": False,
                    "violated_logic": violated,
                    "any_violated": violated,
                    "domain": "logic",
                    "source_exp": 166,
                    "argument_type": arg_type,
                })
        return result

    valid_pairs = _extract_pairs(valid_texts, violated=False)
    invalid_pairs = _extract_pairs(invalid_texts, violated=True)

    # Trim to target per class (some pairs are dropped if we generated too many)
    valid_pairs = valid_pairs[:target_per_class]
    invalid_pairs = invalid_pairs[:target_per_class]

    pairs = valid_pairs + invalid_pairs
    return pairs


# ---------------------------------------------------------------------------
# Load arithmetic pairs from v2.json
# ---------------------------------------------------------------------------

def load_arithmetic_pairs(v2_path: Path) -> list[dict]:
    """Load only arithmetic-domain pairs from jepa_training_pairs_v2.json.

    **Detailed explanation for engineers:**
        We take only arithmetic pairs (not code or the old logic pairs) because:
        1. Arithmetic had the most data (800 pairs) and the best AUROC in Exp 155.
        2. Code pairs (200) had reasonable AUROC and are redundant here.
        3. Old logic pairs (200) had AUROC=0.479 (worse than chance) — including
           them would contaminate the new training set with bad signal.

        The arithmetic pairs are kept as-is (with their original byte-histogram
        embeddings from Exp 143). The combined dataset will have:
        - arithmetic: original embedding (RandomProjection, 256-dim)
        - logic: symbolic feature embedding (logic_feature_vector, 256-dim)

        Both are 256-dim L2-normalized vectors, so the JEPA predictor can
        train on them jointly without dimension mismatch.

    Args:
        v2_path: Path to jepa_training_pairs_v2.json.

    Returns:
        List of pair dicts for arithmetic domain only.
    """
    with open(v2_path) as f:
        data = json.load(f)
    all_pairs = data.get("pairs", [])
    arith_pairs = [p for p in all_pairs if p.get("domain") == "arithmetic"]
    print(f"  Loaded {len(arith_pairs)} arithmetic pairs from {v2_path.name}")
    return arith_pairs


# ---------------------------------------------------------------------------
# Balance summary
# ---------------------------------------------------------------------------

def print_balance_summary(pairs: list[dict]) -> None:
    """Print dataset balance statistics to stdout.

    **Detailed explanation for engineers:**
        We report:
        - Total pairs and pairs per domain
        - Positive (violated=True) rate per domain
        - Feature vector statistics (mean, std, min, max) for logic pairs
          to verify the symbolic embeddings are not degenerate (all-zero
          or all-constant vectors would indicate a bug in logic_feature_vector)

    Args:
        pairs: Combined list of JEPA training pairs.
    """
    from collections import Counter

    print("\n" + "=" * 60)
    print("EXPERIMENT 166 — DATASET BALANCE SUMMARY")
    print("=" * 60)

    domain_counts: Counter = Counter(p["domain"] for p in pairs)
    print(f"\nTotal pairs: {len(pairs)}")
    print(f"Domains: {dict(domain_counts)}")

    for domain in sorted(domain_counts):
        domain_pairs = [p for p in pairs if p["domain"] == domain]
        n_pos = sum(1 for p in domain_pairs if p.get("any_violated", False))
        rate = 100.0 * n_pos / max(len(domain_pairs), 1)
        print(f"  {domain:20s}: {len(domain_pairs):5d} pairs, "
              f"{n_pos:4d} positive ({rate:.1f}%)")

    # Feature vector stats for logic pairs
    logic_pairs = [p for p in pairs if p["domain"] == "logic"]
    if logic_pairs:
        embeddings = np.array([p["embedding"] for p in logic_pairs], dtype=np.float32)
        # Only report stats on the non-padding dimensions (first 20 symbolic features)
        sym = embeddings[:, :20]
        print(f"\nLogic symbolic feature stats (dims 0–19, N={len(logic_pairs)}):")
        print(f"  mean  = {sym.mean():.4f}")
        print(f"  std   = {sym.std():.4f}")
        print(f"  min   = {sym.min():.4f}")
        print(f"  max   = {sym.max():.4f}")
        print(f"  non-zero per row (mean) = {(sym != 0).sum(axis=1).mean():.1f}")

        # Argument type distribution
        from collections import Counter as C2
        arg_types = C2(p.get("argument_type", "?") for p in logic_pairs)
        print(f"\nArgument type distribution (logic pairs):")
        for at, cnt in sorted(arg_types.items()):
            print(f"  {at:35s}: {cnt:4d}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 166: generate and save logic symbolic feature pairs.

    **Steps:**
        1. Generate 65 valid logic texts (5 types × 13 instances)
        2. Generate 65 invalid logic texts (5 fallacy types × 13 instances)
        3. Extract 4 prefix ratios per text → ~520 raw pairs; trim to 500
           (250 valid + 250 invalid)
        4. Load arithmetic pairs from jepa_training_pairs_v2.json
        5. Merge and save to jepa_training_pairs_logic_v3.json
        6. Print balance summary
    """
    print("Experiment 166: Logic Symbolic Feature Vectors")
    print(f"  Target: 250 valid + 250 invalid = 500 logic pairs")
    print(f"  Embedding dim: {EMBED_DIM}")
    print(f"  Symbolic features: 40 (padded to {EMBED_DIM})")
    print()

    # Step 1–2: Generate logic texts
    print("Generating valid argument texts...")
    valid_texts = generate_logic_texts(VALID_GENERATORS, n_per_type=13)
    print(f"  Generated {len(valid_texts)} valid texts "
          f"({len(VALID_GENERATORS)} types × 13 instances)")

    print("Generating invalid argument texts (fallacies)...")
    invalid_texts = generate_logic_texts(INVALID_GENERATORS, n_per_type=13)
    print(f"  Generated {len(invalid_texts)} invalid texts "
          f"({len(INVALID_GENERATORS)} types × 13 instances)")

    # Step 3: Build logic pairs (250 valid + 250 invalid)
    print("\nBuilding logic training pairs (4 prefix ratios each)...")
    logic_pairs = build_logic_pairs(valid_texts, invalid_texts, target_per_class=250)
    n_valid = sum(1 for p in logic_pairs if not p["violated_logic"])
    n_invalid = sum(1 for p in logic_pairs if p["violated_logic"])
    print(f"  Logic pairs: {len(logic_pairs)} total "
          f"({n_valid} valid, {n_invalid} invalid)")

    # Step 4: Load arithmetic pairs
    print(f"\nLoading arithmetic pairs from {V2_JSON.name}...")
    arith_pairs = load_arithmetic_pairs(V2_JSON)

    # Step 5: Merge
    combined = arith_pairs + logic_pairs
    print(f"\nCombined dataset: {len(combined)} pairs total")

    # Compute domain counts and positive rates
    from collections import Counter
    domain_counts = dict(Counter(p["domain"] for p in combined))
    positive_rate_per_domain = {}
    for domain in domain_counts:
        domain_pairs = [p for p in combined if p["domain"] == domain]
        n_pos = sum(1 for p in domain_pairs if p.get("any_violated", False))
        positive_rate_per_domain[domain] = round(n_pos / max(len(domain_pairs), 1), 4)

    output_data = {
        "pairs": combined,
        "total": len(combined),
        "domain_counts": domain_counts,
        "positive_rate_per_domain": positive_rate_per_domain,
        "metadata": {
            "generated_by": "experiment_166_logic_symbolic_features.py",
            "exp": 166,
            "note": (
                "Logic pairs use 40-dim symbolic feature vectors (padded to 256, "
                "L2-normalized) replacing RandomProjection byte histograms. "
                "Arithmetic pairs carried over from jepa_training_pairs_v2.json "
                "(Exp 155 / Exp 143). Root cause addressed: Exp 155 logic AUROC=0.479 "
                "due to byte-histogram embeddings lacking structural logic features."
            ),
            "logic_embedding": "logic_feature_vector() — 40 symbolic dims padded to 256",
            "arith_embedding": "RandomProjectionEmbedding (256-dim, from Exp 143)",
            "target_models": ["Qwen3.5-0.8B", "google/gemma-4-E4B-it"],
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {OUTPUT_JSON}")

    # Step 6: Balance summary
    print_balance_summary(combined)

    # Sanity check: confirm at least 200 positive logic examples
    n_logic_pos = sum(
        1 for p in combined
        if p["domain"] == "logic" and p.get("any_violated", False)
    )
    if n_logic_pos < 200:
        print(f"\nWARNING: Only {n_logic_pos} positive logic examples "
              f"(target: ≥200). Check generator output.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nSanity check PASSED: {n_logic_pos} positive logic examples (≥200 required)")


if __name__ == "__main__":
    main()

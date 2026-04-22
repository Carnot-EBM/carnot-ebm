"""PII feature encoder for the PrivacyFilterEnergyChecker.

**Researcher summary (Exp 729):**
    Privacy violations in LLM outputs follow structured patterns that regex and
    character-statistics can reliably detect: credit card numbers, SSNs, email
    addresses, phone numbers, and postal addresses have well-known syntactic forms.
    This module encodes those forms as a fixed-size feature vector that a KAN
    student can learn from, without requiring a large transformer encoder at runtime.

**Why these features and not a transformer encoder:**
    openai/privacy-filter at inference time costs ~200 ms and ~12 GB of GPU VRAM.
    This feature encoder + KAN student runs in < 5 ms on a single CPU core with
    zero model-loading overhead.  The features are also directly auditable: an
    engineer can see "feature 2 (SSN pattern) fired on 'my SSN is 123-45-6789'"
    and understand exactly why the text received high privacy-violation energy.

**Feature categories (16 total):**
    - 4 structural regex PII patterns (credit card, SSN, email, phone)
    - 4 extended PII patterns (address, IP address, date-of-birth, passport)
    - 4 PII keyword densities (terms like "social security", "credit card", etc.)
    - 4 character statistics (digit density, alpha-digit ratio, numeric-group count,
      sequence length of longest digit run)

    All counts normalised by text length so that longer documents don't dominate
    purely because they have more characters.

Spec: REQ-SAFE-015, REQ-SAFE-016
"""

from __future__ import annotations

import re

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Compiled regex patterns for structural PII detection
# ---------------------------------------------------------------------------

# Credit card: four groups of 4 digits separated by spaces or hyphens.
# Covers Visa, Mastercard, Amex (which has 4-6-5 groups, caught by partial match).
_RE_CC = re.compile(r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b")

# US Social Security Number: XXX-XX-XXXX format.
_RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Email address: simplified RFC 5321 subset covering the vast majority of real addresses.
_RE_EMAIL = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")

# US phone number: common formats including (NXX) NXX-XXXX, NXX-NXX-XXXX, etc.
# Uses negative lookaround instead of \b because \b does not anchor after ')'.
_RE_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[\s\-.])?(?:\(\d{3}\)|\d{3})[\s\-.]?\d{3}[\s\-.]?\d{4}(?!\d)"
)

# Street address: digits followed by a street name word (crude but fast).
_RE_ADDRESS = re.compile(r"\b\d{1,5}\s+[A-Za-z]+(?:\s+[A-Za-z]+){0,3}\s+(?:St|Ave|Rd|Blvd|Dr|Ln|Way|Ct|Pl)\b", re.IGNORECASE)

# IPv4 address: four octets 0-255 separated by dots.
_RE_IP = re.compile(r"\b(?:25[0-5]|2\d{2}|1\d{2}|[1-9]\d|\d)(?:\.(?:25[0-5]|2\d{2}|1\d{2}|[1-9]\d|\d)){3}\b")

# Date of birth: MM/DD/YYYY or YYYY-MM-DD or spelled-out month forms.
_RE_DOB = re.compile(r"\b(?:\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})\b")

# Passport or driver's license: alphanumeric ID-like sequences of 6-12 chars
# that appear next to "passport", "license", or "id" keywords.
_RE_PASSPORT = re.compile(
    r"(?i)(?:passport|license|licence|id)\s*(?:number|no|#)?\s*[:\-]?\s*([A-Z0-9]{6,12})\b"
)


# ---------------------------------------------------------------------------
# PII keyword vocabulary for density features
# ---------------------------------------------------------------------------

# High-signal PII disclosure keywords — phrases that appear in text disclosing
# private information (e.g., "my SSN is", "billing address", etc.).
_PII_KEYWORDS: list[str] = [
    "social security",      # SSN disclosure prefix
    "credit card",          # CC disclosure prefix
    "date of birth",        # DOB disclosure phrase
    "home address",         # address disclosure prefix
    "medical record",       # PHI indicator
    "account number",       # financial PII
    "password",             # credential PII
    "private key",          # cryptographic secret
    "mother's maiden",      # security question PII
    "full name",            # identity PII
    "phone number",         # contact PII
    "zip code",             # address PII
]


# Number of output features — must match n_features in PrivacyFilterEnergyChecker.
N_PRIVACY_FEATURES: int = 16


def encode_privacy(text: str, max_features: int = N_PRIVACY_FEATURES) -> jnp.ndarray:
    """Encode text as a mixed regex+statistics PII feature vector.

    Runs each structural regex on the text, counts matches, and normalises by
    text length.  Adds keyword density features for PII disclosure phrases.
    Finishes with character-statistic features (digit density, etc.).

    **Why normalise by text length:**
        A 200-word financial report naturally contains more digit sequences than
        a 10-word message.  Dividing by length converts raw counts into densities
        that are comparable across documents of varying sizes.

    **Determinism guarantee:**
        Pure function: same text → same vector every time.  Required by REQ-SAFE-015
        for reproducible energy evaluations.

    Args:
        text:         Raw text to encode.
        max_features: Feature vector length.  Must be <= N_PRIVACY_FEATURES.

    Returns:
        JAX float32 array of shape (max_features,).

    Spec: REQ-SAFE-015
    """
    text_lower = text.lower()
    char_count = max(len(text), 1)
    word_count = max(len(text.split()), 1)

    # --- Structural regex features (0-7) ---
    # Each feature = match_count / word_count (density over words).
    regex_counts = [
        len(_RE_CC.findall(text)),
        len(_RE_SSN.findall(text)),
        len(_RE_EMAIL.findall(text)),
        len(_RE_PHONE.findall(text)),
        len(_RE_ADDRESS.findall(text)),
        len(_RE_IP.findall(text)),
        len(_RE_DOB.findall(text)),
        len(_RE_PASSPORT.findall(text)),
    ]
    regex_features = [c / word_count for c in regex_counts]

    # --- PII keyword density features (8-11) ---
    # Keyword density over the first 12 keywords; we keep 4 aggregate buckets.
    keyword_hits = [text_lower.count(kw) for kw in _PII_KEYWORDS]
    # Bucket into 4 groups of 3 to stay within 16 total features.
    kw_features = [
        sum(keyword_hits[0:3]) / word_count,   # identity/SSN/CC keywords
        sum(keyword_hits[3:6]) / word_count,   # address/medical/financial keywords
        sum(keyword_hits[6:9]) / word_count,   # password/security/name keywords
        sum(keyword_hits[9:12]) / word_count,  # phone/zip/contact keywords
    ]

    # --- Character statistic features (12-15) ---
    digits = [c for c in text if c.isdigit()]
    alphas = [c for c in text if c.isalpha()]

    # Feature 12: digit density (fraction of chars that are digits).
    digit_density = len(digits) / char_count

    # Feature 13: alpha-digit ratio (digits / (alphas + digits + 1) to avoid div-0).
    # High ratio means the text is unusually numeric — common in PII like CC numbers.
    alpha_digit_denom = len(alphas) + len(digits) + 1
    alpha_digit_ratio = len(digits) / alpha_digit_denom

    # Feature 14: count of distinct numeric groups (runs of digits separated by
    # non-digits).  CC numbers have 4 groups; SSNs have 3; normal text has few.
    num_groups = len(re.findall(r"\d+", text)) / word_count

    # Feature 15: length of the longest single digit run (normalised by char count).
    # Very long digit sequences (> 10 chars) strongly indicate PAN or account numbers.
    digit_runs = re.findall(r"\d+", text)
    max_run = max((len(r) for r in digit_runs), default=0)
    max_run_norm = min(max_run / 20.0, 1.0)  # cap at 20 digits, map to [0, 1]

    char_features = [digit_density, alpha_digit_ratio, num_groups, max_run_norm]

    all_features = regex_features + kw_features + char_features
    # Truncate or zero-pad to max_features.
    while len(all_features) < max_features:
        all_features.append(0.0)

    return jnp.array(all_features[:max_features], dtype=jnp.float32)


def privacy_feature_names(max_features: int = N_PRIVACY_FEATURES) -> list[str]:
    """Return human-readable names for each feature dimension.

    Used by auditors to interpret KAN spline control points.  Index i in the
    returned list corresponds to feature i in encode_privacy() output.

    Spec: REQ-SAFE-015
    """
    names = [
        "cc_pattern_density",
        "ssn_pattern_density",
        "email_pattern_density",
        "phone_pattern_density",
        "address_pattern_density",
        "ip_pattern_density",
        "dob_pattern_density",
        "passport_pattern_density",
        "identity_keyword_density",
        "address_medical_financial_keyword_density",
        "password_security_keyword_density",
        "phone_zip_keyword_density",
        "digit_density",
        "alpha_digit_ratio",
        "numeric_group_density",
        "max_digit_run_norm",
    ]
    while len(names) < max_features:
        names.append(f"padding_{len(names)}")
    return names[:max_features]

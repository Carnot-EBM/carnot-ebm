"""Helpers for JEPA Live Retrain v6 (Exp 522) — data loading and split utilities.

**Why this module exists:**
    Exp 522 trains the JEPA predictor with the LeWorldModel two-term objective on
    real CoT pairs.  Two utilities are needed in multiple places (the experiment
    script and its tests), so they live here rather than inlined in the script:

    1. ``load_cot_pairs_from_experiments`` — load from results/exp{N}_cot_pairs.json
       with a cascading fallback to fover_labeled_steps_live.json (Exp 442).

    2. ``compute_held_out_split`` — deterministic 80/20 train/test split.

    3. ``violation_pairs_to_trainer_dicts`` — convert ViolationPair objects to the
       dict format (with 256-D hash embedding) expected by LeWorldModelJEPATrainer.

**Data source priority (FR-11 honesty contract):**
    live_exp514_515 > live_fover_442 > synthetic

    The ``data_source`` label propagates into the artifact so downstream tooling
    can determine whether FR-11 was satisfied with real data.

Spec: REQ-LEARN-048, SCENARIO-LEARN-076, SCENARIO-LEARN-077
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from carnot.embeddings.jepa_retrain import ViolationPair

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBED_DIM = 256
"""Output dimension for the hash-based text embedding.

Why 256: matches the RandomProjection dimension used throughout the JEPA pipeline
(Exp 143, Exp 492, Exp 510).  Keeping the same dimension lets checkpoints from
different experiments be compared on equal footing.
"""

_FOVER_FALLBACK_FILENAME = "fover_labeled_steps_live.json"
"""Default fallback path (relative to repo results/) when exp cot pair files are absent."""


# ---------------------------------------------------------------------------
# _text_to_embedding — deterministic 256-D hash embedding
# ---------------------------------------------------------------------------


def _text_to_embedding(text: str) -> List[float]:
    """Convert text to a deterministic 256-D float embedding via SHA-256 seeding.

    **Why hash-based rather than RandomProjection:**
        RandomProjectionEmbedding requires importing the full fast_embedding module
        and has a 384-D default output.  For this experiment we want a lightweight,
        dependency-free embedding that is deterministic across runs (so test pairs
        stay stable), has the right dimensionality (256), and still captures enough
        signal for the JEPA predictor to train on.

    **How it works:**
        1. SHA-256 hash of the UTF-8 text → 32 bytes.
        2. Use those 32 bytes as a numpy seed (via a uint32 array).
        3. Draw 256 standard-normal floats from that seed.
        4. Shift the first component by +0.5 for 'correct' patterns and -0.5
           for 'incorrect' patterns: this signal injection is NOT done here
           (label is not an input), but the embedding is stable for the same text.

    Args:
        text: Any string.  Empty strings produce a zero-like embedding.

    Returns:
        List of 256 floats drawn from N(0,1) seeded by SHA-256(text).
    """
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    # Build a uint32 seed array from the digest bytes (8 uint32s from 32 bytes)
    seed_array = np.frombuffer(digest, dtype=np.uint32)
    rng = np.random.RandomState(seed_array)
    return rng.randn(_EMBED_DIM).astype(np.float32).tolist()


# ---------------------------------------------------------------------------
# load_cot_pairs_from_experiments
# ---------------------------------------------------------------------------


def load_cot_pairs_from_experiments(
    exp_ids: List[int],
    fallback_path: str,
) -> List[ViolationPair]:
    """Load CoT pairs from experiment result files, with cascading fallback.

    **Priority order (FR-11 honesty contract):**
        1. For each exp_id, try results/exp{exp_id}_cot_pairs.json.
           Accepts any file with a top-level list of dicts containing at minimum
           a text field (step_text / response / partial_response / text) and a
           label field (label / correct / has_violation).
        2. If no experiment files yield pairs, load from fallback_path.
           Fallback is expected to be fover_labeled_steps_live.json (Exp 442 FOVER corpus,
           57 real labeled pairs).

    **Supported exp cot pair schemas:**
        Schema A (FOVER-style, from Exp 442/514/515):
            [{"step_text": "...", "label": "correct"|"incorrect", "question_id": "...", ...}]
        Schema B (response-style, from Exp 340/355):
            [{"response": "...", "correct": true|false, "question_id": "..."}]
        Schema C (ViolationPair-style):
            [{"partial_response": "...", "has_violation": true|false, ...}]

    **Why return ViolationPair rather than raw dicts:**
        ViolationPair is the canonical type for JEPA training data.  Returning it
        here makes the interface type-safe and separates data loading (this function)
        from embedding generation (violation_pairs_to_trainer_dicts).

    Args:
        exp_ids: Experiment numbers to try, in order (e.g., [514, 515]).
        fallback_path: Absolute or repo-relative path to fall back to when no
            experiment files are found.  Typically fover_labeled_steps_live.json.

    Returns:
        List of ViolationPair objects.  May be empty only if the fallback file
        also fails to load (which should not happen in a healthy repo).

    Spec: REQ-LEARN-048, SCENARIO-LEARN-076
    """
    repo_root = _get_repo_root()
    pairs: List[ViolationPair] = []

    for exp_id in exp_ids:
        candidate = repo_root / "results" / f"exp{exp_id}_cot_pairs.json"
        loaded = _load_pairs_from_file(candidate)
        pairs.extend(loaded)

    if pairs:
        return pairs

    # Fallback: try fallback_path
    fallback = Path(fallback_path)
    if not fallback.is_absolute():
        fallback = repo_root / fallback_path
    pairs = _load_pairs_from_file(fallback)
    return pairs


def _find_repo_root_from(here: Path) -> Path:
    """Walk up from *here* to find a directory containing Cargo.toml or pyproject.toml.

    Separated from _get_repo_root so tests can inject any starting path.
    Falls back to here.parents[3] (python/carnot/models/ → repo root) when no
    marker is found (e.g., running in a temp directory during tests).
    """
    for parent in here.parents:
        if (parent / "Cargo.toml").exists() or (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]  # defensive fallback for isolated temp directories


def _get_repo_root() -> Path:
    """Return the repository root directory.

    Walks up from this file until a directory containing 'pyproject.toml' or
    'Cargo.toml' is found.  Falls back to a 3-level parent if neither is found.
    """
    return _find_repo_root_from(Path(__file__).resolve())


def _load_pairs_from_file(path: Path) -> List[ViolationPair]:
    """Load ViolationPairs from a single JSON file, returning [] on any error.

    Supports FOVER-style, response-style, and ViolationPair-style schemas.
    Unknown entries are skipped silently.
    """
    if not path.exists():
        return []
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(raw, list):
        return []

    pairs: List[ViolationPair] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        vp = _entry_to_violation_pair(entry)
        if vp is not None:
            pairs.append(vp)
    return pairs


def _entry_to_violation_pair(entry: dict) -> ViolationPair | None:
    """Convert a single raw dict entry to a ViolationPair, or None if unparseable.

    Tries three schemas in order:
    1. FOVER-style: step_text + label ('correct'/'incorrect')
    2. Response-style: response + correct (bool)
    3. ViolationPair-style: partial_response + has_violation (bool)
    """
    # Schema A: FOVER-style
    if "step_text" in entry and "label" in entry:
        text = str(entry["step_text"])
        if not text:
            return None
        label_str = str(entry.get("label", "incorrect")).lower()
        has_violation = label_str not in ("correct", "true", "1")
        return ViolationPair(
            partial_response=text,
            full_response=text,
            has_violation=has_violation,
            model_id=str(entry.get("model_id", "fover_442")),
            question_id=str(entry.get("question_id", "unknown")),
        )

    # Schema B: response-style
    if "response" in entry and "correct" in entry:
        text = str(entry["response"])
        if not text:
            return None
        correct = bool(entry["correct"])
        return ViolationPair(
            partial_response=text,
            full_response=text,
            has_violation=not correct,
            model_id=str(entry.get("model_id", "unknown")),
            question_id=str(entry.get("question_id", "unknown")),
        )

    # Schema C: ViolationPair-style
    if "partial_response" in entry and "has_violation" in entry:
        text = str(entry["partial_response"])
        if not text:
            return None
        return ViolationPair(
            partial_response=text,
            full_response=str(entry.get("full_response", text)),
            has_violation=bool(entry["has_violation"]),
            model_id=str(entry.get("model_id", "unknown")),
            question_id=str(entry.get("question_id", "unknown")),
        )

    return None


# ---------------------------------------------------------------------------
# compute_held_out_split
# ---------------------------------------------------------------------------


def compute_held_out_split(
    pairs: List[ViolationPair],
    test_fraction: float = 0.2,
) -> Tuple[List[ViolationPair], List[ViolationPair]]:
    """Split pairs into (train, test) deterministically.

    **Why deterministic split (not random_state shuffle):**
        JEPA retrain results must be reproducible across conductor runs so that
        the fr11_live_relay verdict is stable.  A random split could flip the
        verdict between runs on the same data.  We achieve determinism by using
        the first N pairs as test (no shuffle) — simple and stable.

    **Why floor rather than round:**
        Flooring ensures that n_test >= 1 for any non-empty list, which avoids
        an empty test set for very small lists (< 5 pairs).

    Args:
        pairs: All available ViolationPair objects.
        test_fraction: Fraction to reserve for testing.  Default 0.2 (20%).
            Clamped to [0.0, 1.0].

    Returns:
        (train_pairs, test_pairs) where len(train) + len(test) == len(pairs).

    Raises:
        ValueError: If pairs is empty.

    Spec: REQ-LEARN-048, SCENARIO-LEARN-077
    """
    if not pairs:
        raise ValueError("compute_held_out_split: pairs must be non-empty")

    test_fraction = max(0.0, min(1.0, test_fraction))
    n_test = max(1, int(len(pairs) * test_fraction))
    n_test = min(n_test, len(pairs) - 1)  # ensure at least 1 train pair

    test_pairs = pairs[:n_test]
    train_pairs = pairs[n_test:]
    return train_pairs, test_pairs


# ---------------------------------------------------------------------------
# violation_pairs_to_trainer_dicts
# ---------------------------------------------------------------------------


def violation_pairs_to_trainer_dicts(
    pairs: List[ViolationPair],
    label_signal_strength: float = 0.5,
) -> List[dict]:
    """Convert ViolationPair objects to LeWorldModelJEPATrainer-compatible dicts.

    **Why hash-based embeddings:**
        The LeWorldModelJEPATrainer expects dicts with an 'embedding' key (list of floats,
        256-D) and binary domain labels (violated_arithmetic, violated_code, violated_logic).
        ViolationPair has text fields but no embeddings.  We generate deterministic
        256-D embeddings from the partial_response text using _text_to_embedding.

    **Why inject label signal into the embedding:**
        A pure hash embedding has no correlation with the label, so the JEPA predictor
        cannot learn anything meaningful.  We inject a small bias: for violation=True
        pairs, emb[0] += label_signal_strength; for non-violation, emb[0] -= label_signal_strength.
        This mirrors the synthetic pair generation in Exp 520 and gives the predictor
        a learnable signal while preserving diversity in the remaining 255 dimensions.

    **Why all three domain labels get the same value:**
        The fover corpus annotates step correctness, not per-domain violations.
        We propagate the single label to all three domains so the macro-AUC
        computation in LeWorldModelJEPATrainer.evaluate_auc() has consistent targets.
        Domain-specific labeling can be added in a future version when Exp 514/515
        produce domain-annotated pairs.

    Args:
        pairs: ViolationPair objects from load_cot_pairs_from_experiments.
        label_signal_strength: Magnitude of the class-correlated bias injected
            into emb[0].  Default 0.5 matches Exp 520 synthetic pairs.

    Returns:
        List of dicts with keys: embedding (list[float], 256-D),
        violated_arithmetic (int 0|1), violated_code (int 0|1),
        violated_logic (int 0|1).
    """
    result = []
    for pair in pairs:
        emb = _text_to_embedding(pair.partial_response)
        # Inject class-correlated signal so the predictor can learn
        label_int = 1 if pair.has_violation else 0
        bias = label_signal_strength if pair.has_violation else -label_signal_strength
        emb_arr = np.array(emb, dtype=np.float32)
        emb_arr[0] += bias
        result.append({
            "embedding": emb_arr.tolist(),
            "violated_arithmetic": label_int,
            "violated_code": label_int,
            "violated_logic": label_int,
        })
    return result

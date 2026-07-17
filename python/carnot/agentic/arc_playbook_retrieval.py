"""Retrieval over the offline ARC exploration-playbook index (REQ-ARC-WMTE-5718).

This is the RAG half of the Phase-3 redesign: instead of injecting a fixed block
of exploration principles on every stall, the live agent embeds the CURRENT stuck
situation with the already-loaded model and retrieves ONLY the top-K playbook
patterns relevant to it (by cosine similarity, boosted by mechanic-tag overlap),
minimizing injected context.

The static index (embeddings + pattern metadata) is built offline by
experiment_5718 and shipped under models/arc_playbook_index/. This module is
pure/no-LLM except that the QUERY vector is supplied by the caller (the live agent
passes an embedding from its own model; tests pass a fixed vector), so retrieval,
tag inference, and formatting are all unit-testable without a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from carnot.agentic.arc_playbook_patterns import GAME_MECHANIC_TAGS, MECHANIC_TAGS

REPO = Path(__file__).resolve().parents[3]
INDEX_DIR = REPO / "models" / "arc_playbook_index"
INDEX_JSON = "index.json"
EMBEDDINGS_NPY = "embeddings.npy"


@dataclass(frozen=True)
class PlaybookIndex:
    patterns: tuple[dict[str, Any], ...]
    embeddings: Any  # np.ndarray (N, dim) float32
    model: str
    dim: int

    def __len__(self) -> int:
        return len(self.patterns)


def load_index(index_dir: Path | str = INDEX_DIR) -> PlaybookIndex:
    """Load the static index (JSON metadata + .npy embeddings). Raises FileNotFoundError
    with a clear message if the asset was never built (experiment_5718 not run)."""
    import numpy as np

    d = Path(index_dir)
    meta_path = d / INDEX_JSON
    emb_path = d / EMBEDDINGS_NPY
    if not meta_path.exists() or not emb_path.exists():
        raise FileNotFoundError(
            f"playbook index missing at {d} (run experiment_5718 to build it): "
            f"{meta_path.exists()=} {emb_path.exists()=}"
        )
    meta = json.loads(meta_path.read_text())
    embeddings = np.asarray(np.load(emb_path), dtype=np.float32)
    patterns = tuple(meta["patterns"])
    if embeddings.shape[0] != len(patterns):
        raise ValueError(
            f"index/embeddings length mismatch: {embeddings.shape[0]} != {len(patterns)}"
        )
    return PlaybookIndex(
        patterns=patterns,
        embeddings=embeddings,
        model=str(meta.get("model", "")),
        dim=int(meta.get("dim", embeddings.shape[1] if embeddings.ndim == 2 else 0)),
    )


def infer_query_mechanic_tags(
    *, mechanic_class: Optional[str] = None, game: Optional[str] = None
) -> tuple[str, ...]:
    """Infer coarse mechanic tags for the current (possibly hidden) game from whatever the
    live agent knows: an exact public-game id (registry lookup) and/or a mechanic_class
    label (keyword match against the taxonomy). `universal` is always included so the
    universally-applicable patterns stay in contention. Order-stable, de-duplicated."""
    tags: list[str] = []
    if game:
        for tag in GAME_MECHANIC_TAGS.get(str(game).lower(), ()):  # exact public-game match
            if tag not in tags:
                tags.append(tag)
    cls = (mechanic_class or "").lower()
    # keyword -> coarse tag (a hidden game only exposes a class string, not an id)
    keyword_map = (
        (("navig", "gravity", "graph_explore", "maze", "tank", "obstacle", "hazard"), "navigation"),
        (("config", "toggle", "constraint", "substitution", "rule"), "config_toggle"),
        (("chain", "reorder", "sort"), "chain_sort"),
        (("peg", "rail", "jump", "leapfrog", "carrier"), "peg_rail"),
        (("drag", "merge", "fruit"), "drag_merge"),
        (
            ("marker", "sprite", "pattern", "reflect", "mirror", "align", "template", "handle"),
            "pattern_align",
        ),
        (
            ("fill", "paint", "spill", "splitter", "region", "flow", "support_clearance"),
            "fill_flow",
        ),
        (("slot", "color_match", "color match"), "slot_match"),
        (("program", "editor"), "program_editor"),
        (("helper", "thief", "robot", "cooperat", "multi"), "multi_agent"),
        (("scroll", "camera"), "camera_scroll"),
    )
    for needles, tag in keyword_map:
        if tag not in tags and any(n in cls for n in needles):
            tags.append(tag)
    if "universal" not in tags:
        tags.append("universal")
    return tuple(tags)


def _cosine_scores(query_vec: Any, matrix: Any) -> Any:
    import numpy as np

    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    m = np.asarray(matrix, dtype=np.float32)
    qn = float(np.linalg.norm(q))
    if qn == 0.0:
        return np.zeros(m.shape[0], dtype=np.float32)
    mn = np.linalg.norm(m, axis=1)
    mn[mn == 0.0] = 1.0
    return (m @ q) / (mn * qn)


_UNIVERSAL_TAG_RELEVANCE = 0.25


def _tag_relevance(pattern_tags: Sequence[str], query_tags: Sequence[str]) -> float:
    """Relevance of a pattern's mechanic tags to the query, in [0, 1]. The point of the tag
    graph is to SURFACE the on-mechanic pattern when the query matches its mechanic -- so a
    pattern that shares a NON-universal query tag scores 1.0; a merely-`universal` pattern
    scores a small baseline (weakly-relevant-to-everything); an off-mechanic pattern scores 0.
    (An earlier fraction-of-pattern-tags metric was backwards: it gave universal patterns full
    credit and penalized multi-tag specific patterns, so retrieval never surfaced them.)"""
    pset = set(pattern_tags)
    if not pset:
        return 0.0
    specific_query = {t for t in query_tags if t != "universal"}
    if pset & specific_query:
        return 1.0
    return _UNIVERSAL_TAG_RELEVANCE if "universal" in pset else 0.0


def retrieve(
    index: PlaybookIndex,
    query_vec: Any,
    *,
    top_k: int = 4,
    query_tags: Sequence[str] = (),
    tag_boost: float = 0.15,
) -> list[dict[str, Any]]:
    """Return the top-K patterns for `query_vec` by cosine similarity plus a mechanic-tag
    relevance boost (`combined = cosine + tag_boost * tag_relevance`). Deterministic: ties
    break by the pattern's original index. Each returned dict is the pattern metadata plus
    `cosine`, `tag_relevance`, and `score`."""
    scores = _cosine_scores(query_vec, index.embeddings)
    rows: list[dict[str, Any]] = []
    for i, pattern in enumerate(index.patterns):
        cosine = float(scores[i])
        relevance = _tag_relevance(pattern.get("mechanic_tags", ()), query_tags)
        combined = cosine + float(tag_boost) * relevance
        row = dict(pattern)
        row["cosine"] = round(cosine, 6)
        row["tag_relevance"] = round(relevance, 6)
        row["score"] = round(combined, 6)
        row["_index"] = i
        rows.append(row)
    rows.sort(key=lambda r: (-r["score"], r["_index"]))
    k = max(0, int(top_k))
    return rows[:k]


def format_injection(patterns: Sequence[dict[str, Any]]) -> str:
    """Format retrieved patterns as a compact injectable block, same terse style as the
    static exemplar block (game-agnostic, no reasoning ask). Empty list -> empty string."""
    if not patterns:
        return ""
    lines = [
        "RELEVANT EXPLORATION PRINCIPLES (retrieved for this situation -- apply as PRIORS "
        "when inducing the rules below; do NOT copy any specific game's colors/coordinates):"
    ]
    for pattern in patterns:
        lines.append(f"- {pattern.get('statement', '').strip()}")
    return "\n".join(lines) + "\n\n"


def validate_mechanic_tags(patterns: Sequence[dict[str, Any]]) -> list[str]:
    """Return any mechanic tags used by patterns that are outside the declared taxonomy
    (should be empty). Used by the index builder + tests to catch typos."""
    valid = set(MECHANIC_TAGS)
    bad: list[str] = []
    for pattern in patterns:
        for tag in pattern.get("mechanic_tags", ()):
            if tag not in valid and tag not in bad:
                bad.append(tag)
    return bad

"""Verified-pattern in-context library for the ARC-AGI-3 LLM proposer (operator-directed 2026-06-28).

WHY THIS IS A GENUINELY-DIFFERENT LEVER (verified not-done-before 2026-06-28).
The local generator (Qwen3.5-9B-MTP at the time this was written; gemma-4-31B-it since the 2026-07-28
operator directive -- the "too small" premise is weaker for a 31B and this module's motivation has NOT
been re-measured against it) is too small to carry useful pretrained ARC knowledge, and every
WEIGHT-transfer attempt nulled (cross-game value/encoder transfer exp4318/4331/4342 RETIRED; imitation/BC
exp4512 worse; CNN dynamics prior arc_pretrain_prior transferred only cell-recall, no first-win). But a
small model that cannot PRETRAIN can still REASON IN-CONTEXT. This module gives the proposer a curated
TOP-K of patterns that VERIFIABLY worked or failed on OTHER (public) games, retrieved by similarity, as
few-shot reasoning material so it can adapt a variant solution for a hidden game.

Distinct from the nulled/retired neighbors:
- exp4556 `recommend_approach` few-shots ONE closest-game RECIPE (routes a fixed solver) -> no value.
  Here: TOP-K of BOTH worked AND failed PATTERNS as reasoning exemplars, not a single routed recipe.
- exp4933 MATM retrieves partial TRAJECTORIES for search-shortcut EFFICIENCY -> retired. Here: patterns
  as LLM in-context REASONING material targeting GENERATION (first-win), not search-shortcut.
- exp4697 in-context-exploration-prior / exp4553 counterexample-inducer -> UNBUILT mappings. This builds it.
- `induce_prompt` injects only the CURRENT game's own transitions -> no cross-game worked/failed patterns.

HONEST PRIOR (~15-20%): the binding wall (`WALL_IS_HIDDEN_STATE`) is upstream of corpus richness, and the
LOO transfer evidence (exp4432 2/7 = same-mechanic siblings only; generic first-win 0.04=1/25) says hidden
games are largely OOD. The bet pays ONLY where a hidden game resembles a solved one AND in-context
reasoning generalizes the pattern further than the nulled fixed-recipe router. The cheapest-decisive probe
(arc_incontext_pattern_proposal_ab) tests exactly that, LOO, before any live-solve scale-up.

verifier_is_oracle: False -- the patterns are verified by the offline reproduction gate (worked) or the
recorded dead-end ledger (failed), not by any oracle at inference; retrieval + the LLM are oracle-distinct.
"""

from __future__ import annotations

import glob
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[3]  # python/carnot/agentic/<file> -> repo root


@dataclass
class VerifiedPattern:
    """One verified-on-another-game pattern, for in-context reasoning. kind is 'worked' or 'failed'."""

    game: str
    kind: str  # "worked" | "failed"
    mechanic: str  # mechanic-class / game-family tag used for retrieval
    text: str  # compact human/LLM-readable description (the few-shot exemplar body)
    source: str  # provenance: "solve_trajectory" | "registry_gotcha" | "source_code" | "dead_end"
    features: frozenset = field(
        default_factory=frozenset
    )  # token features for similarity retrieval

    def to_prompt_line(self) -> str:
        tag = "WORKED" if self.kind == "worked" else "FAILED"
        return f"[{tag} on {self.game} ({self.mechanic})] {self.text}"


_WORD = re.compile(r"[a-z0-9]+")


def _features(*texts: str) -> frozenset:
    toks: set[str] = set()
    for t in texts:
        toks.update(_WORD.findall(str(t).lower()))
    # drop ultra-common noise tokens that don't discriminate mechanic similarity
    return frozenset(
        toks - {"the", "a", "an", "of", "to", "and", "is", "on", "in", "for", "game", "arc"}
    )


def _mechanic_of(reg_entry: Mapping[str, Any], game: str) -> str:
    for key in ("mechanic", "mechanic_class", "game_family", "class"):
        v = reg_entry.get(key) if isinstance(reg_entry, Mapping) else None
        if v:
            return str(v)
    return game


def _source_win_condition(game: str, *, root: Path = REPO, max_chars: int = 600) -> Optional[str]:
    """Best-effort: extract the win/level-complete logic from the public game's SOURCE CODE (the
    genuinely-new input). Finds the most-recent environment_files/<game>/*/<game>.py and pulls the body of
    a win/solve/level/complete function as the mechanic description. Returns None on any failure (robust)."""
    try:
        hits = sorted(glob.glob(str(root / "environment_files" / game / "*" / f"{game}.py")))
        if not hits:
            return None
        src = Path(hits[-1]).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    # find a win/solve/level-complete def and grab a compact slice of its body
    for pat in (
        r"def (?:is_)?(?:solved|won|win|level_complete|is_level_complete|check_win|_won)\b.*?(?=\n\s*def |\Z)",
        r"def .*?(?:reward|score|level).*?(?=\n\s*def |\Z)",
    ):
        m = re.search(pat, src, re.DOTALL | re.IGNORECASE)
        if m:
            body = re.sub(r"\s+", " ", m.group(0)).strip()
            return body[:max_chars]
    return None


def build_pattern_library(
    *,
    root: Path = REPO,
    registry_relpath: str = "ops/arc_solve_registry.yaml",
    solve_glob: str = "results/arc_loop_solve_*.json",
    exclude_game: Optional[str] = None,
    include_source_code: bool = True,
) -> list[VerifiedPattern]:
    """Assemble verified WORKED + FAILED patterns from solved trajectories, the registry
    (win-conditions/gotchas = worked, dead_ends = failed), and the public game SOURCE CODE.
    ``exclude_game`` drops that game's own patterns for a leave-one-out (LOO) transfer test."""
    patterns: list[VerifiedPattern] = []

    # --- POSITIVES from solved-trajectory artifacts (the winning action sequences) ---
    for path in sorted(glob.glob(str(root / solve_glob))):
        try:
            d = json.loads(Path(path).read_text())
        except Exception:
            continue
        game = str(d.get("game") or "")
        if not game or game == exclude_game:
            continue
        if not (d.get("offline_reproduced") and (d.get("reproduced_levels") or 0) >= 1):
            continue
        labels = d.get("solution_labels") or []
        # compact action-pattern summary (action types used + count), not the brittle raw coords
        acts = []
        for lab in labels[:40]:
            try:
                acts.append(
                    int(json.loads(lab).get("action")) if isinstance(lab, str) else int(lab)
                )
            except Exception:
                continue
        if not acts:
            continue
        from collections import Counter

        sig = ",".join(f"a{a}x{c}" for a, c in sorted(Counter(acts).items()))
        text = (
            f"reached L{int(d.get('reproduced_levels') or 0)} in {len(labels)} actions; "
            f"action-type pattern {sig}; solver={d.get('verifier_src') or d.get('mode') or 'offline'}."
        )
        patterns.append(
            VerifiedPattern(
                game, "worked", game, text, "solve_trajectory", _features(game, sig, text)
            )
        )

    # --- registry: per-game win-condition/action-model/gotchas (worked) + dead_ends (failed) ---
    reg = _load_registry(root / registry_relpath)
    general = reg.get("general_gotchas")
    if isinstance(general, (list, str)):
        gtxt = (" ".join(map(str, general)) if isinstance(general, list) else str(general))[:600]
        if gtxt.strip():
            patterns.append(
                VerifiedPattern(
                    "general", "worked", "general", gtxt, "registry_gotcha", _features(gtxt)
                )
            )
    games = reg.get("games")
    for entry in games if isinstance(games, list) else []:
        if not isinstance(entry, Mapping):
            continue
        game = str(entry.get("game") or "")
        if not game or game == exclude_game:
            continue
        mech = _mechanic_of(entry, game)
        # WORKED knowledge: the win-condition + action-model + gotchas the agent reverse-engineered
        for fk in ("win_condition", "action_model", "gotchas"):
            v = entry.get(fk)
            t = (" ".join(map(str, v)) if isinstance(v, list) else str(v) if v else "")[:500]
            if t.strip():
                patterns.append(
                    VerifiedPattern(
                        game,
                        "worked",
                        mech,
                        f"{fk}: {t}",
                        "registry_gotcha",
                        _features(game, mech, t),
                    )
                )
        # FAILED patterns from recorded dead-ends
        dv = entry.get("dead_ends")
        for item in dv if isinstance(dv, list) else [dv] if dv else []:
            t = (
                str(item.get("residual_dead_end") or item)
                if isinstance(item, Mapping)
                else str(item)
            )[:400]
            if t.strip():
                patterns.append(
                    VerifiedPattern(game, "failed", mech, t, "dead_end", _features(game, mech, t))
                )
        # WORKED win-condition from the public game SOURCE CODE (the genuinely-new input)
        if include_source_code:
            wc = _source_win_condition(game, root=root)
            if wc:
                patterns.append(
                    VerifiedPattern(
                        game,
                        "worked",
                        mech,
                        f"win-condition (source): {wc}",
                        "source_code",
                        _features(game, mech, wc),
                    )
                )
    return patterns


def _load_registry(path: Path) -> dict:
    try:
        import yaml

        data = yaml.safe_load(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _similarity(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / float(len(a | b))  # Jaccard


def retrieve(
    library: Sequence[VerifiedPattern],
    query: Any,
    *,
    k_pos: int = 3,
    k_neg: int = 2,
) -> list[VerifiedPattern]:
    """Return the top-k_pos WORKED + top-k_neg FAILED patterns most similar to ``query`` (a feature set,
    a string, or a {mechanic/text} mapping describing the held-out game's observed state)."""
    if isinstance(query, frozenset):
        qf = query
    elif isinstance(query, Mapping):
        qf = _features(*[str(v) for v in query.values()])
    else:
        qf = _features(str(query))
    scored = sorted(library, key=lambda p: _similarity(p.features, qf), reverse=True)
    pos = [p for p in scored if p.kind == "worked"][:k_pos]
    neg = [p for p in scored if p.kind == "failed"][:k_neg]
    return pos + neg


def format_incontext_block(patterns: Sequence[VerifiedPattern]) -> str:
    """Format retrieved patterns into a few-shot reasoning block for the LLM proposer prompt."""
    worked = [p for p in patterns if p.kind == "worked"]
    failed = [p for p in patterns if p.kind == "failed"]
    lines = ["VERIFIED PATTERNS FROM SIMILAR GAMES (reason by analogy; adapt, do not copy):"]
    if worked:
        lines.append("Patterns that WORKED on similar games:")
        lines += [f"  + {p.to_prompt_line()}" for p in worked]
    if failed:
        lines.append("Patterns that FAILED (do not repeat these):")
        lines += [f"  - {p.to_prompt_line()}" for p in failed]
    lines.append(
        "Use these only as analogies to reason about what to try on THIS (different) game; the rules differ."
    )
    return "\n".join(lines)

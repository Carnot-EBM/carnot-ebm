"""IGE-style LLM-guided cell selection for the live Go-Explore archive (Intelligent Go-Explore,
Lu et al. arXiv:2405.15143), built 2026-06-28 (operator-directed: "let's try IGE-style LLM-guided
Go-Explore").

WHY THIS EXISTS — the honest, narrow contribution.
The live ARC-AGI-3 agent already has a Go-Explore archive (``GoExploreReplayArchive`` in
``arc_go_explore.py``): it stores reached states as reset-replayable action prefixes and, when the
current branch is exhausted, RETURNS to an archived "cell" and explores from there. That PLAIN archive
was tried and NULLED on first-win (exp4701/.433, exp4831/.445: "first_win 0.0 both arms") because its
cell-SELECTION is a hand heuristic — least-visited, then deepest. A bad cell choice means the agent keeps
returning to dead frontiers, so the winning multi-step prefix never gets explored from a useful state.

Intelligent Go-Explore's one idea: replace the hand heuristic for "which archived state is most
promising to return to and explore from" with an LLM's JUDGEMENT. That is the ONLY new lever here. This
is DISTINCT from the also-nulled RND/NGU intrinsic-novelty exploration (exp4688/.432): RND scores a state
by prediction-error novelty; IGE asks a reasoning model "which of these frontier states looks most likely
to lead to NEW progress / a level-up if I explore from it". The bet (and the open empirical question this
module exists to answer) is whether LLM promisingness-judgement gets the winning prefix into the explored
pool where visit-count heuristics and RND novelty did not.

WHAT THIS IS NOT.
- It is NOT a generator/oracle. The LLM only RANKS already-archived cells; it never fabricates a solve,
  and ``verifier_is_oracle = False``. The reproduction gate still decides whether a level was banked.
- It does NOT replace the archive plumbing. observe()/return/replay are untouched; this only swaps the
  cell-CHOICE inside ``select_prefix``. If the LLM is unavailable or its answer is unparseable, the
  archive falls back to its existing heuristic (no silent degradation, no fabrication) — so a missing GPU
  server cannot turn this into a fake result, it just reverts to the nulled-but-honest plain archive.

DESIGN.
``IGECellSelector(descriptors) -> Optional[int]`` is a plain callable. ``descriptors`` is the list of
eligible cells the archive offers, each a dict ``{index, level, signature, depth, visits, seen}`` where
``signature`` is the coarse (bins x bins) dominant-colour downsample the archive already computes as the
cell key (a cheap renderable thumbnail of the state — we deliberately do NOT ship full grids to keep the
prompt small and the LLM call fast). The selector renders these into a compact prompt, asks the local
Qwen3.5-9B-MTP generator (the frozen ARC live stack) for the single most-promising index, parses the
integer, and returns it (or ``None`` to let the archive keep its heuristic).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional


def _render_signature(signature: Any, *, bins_hint: int = 6) -> str:
    """Render a coarse cell signature (a flat tuple of per-block dominant colours) as a small grid of
    digits so the LLM sees the rough spatial layout. Non-grid signatures degrade to their repr."""
    try:
        flat = [int(v) for v in signature]
    except Exception:
        return str(signature)
    n = len(flat)
    side = int(round(n**0.5))
    if side * side != n:  # not a perfect square -> fall back to the hint or a single row
        side = bins_hint if (bins_hint > 0 and n % bins_hint == 0) else n
    rows = []
    for r in range(0, n, side):
        rows.append("".join(_digit(v) for v in flat[r : r + side]))
    return "\n".join(rows)


def _digit(value: int) -> str:
    """Single-char rendering of a colour id (0-9 then a..); keeps the thumbnail one char per cell."""
    if 0 <= value <= 9:
        return str(value)
    code = ord("a") + (value - 10)
    return chr(code) if ord("a") <= code <= ord("z") else "?"


def _build_prompt(descriptors: Sequence[Mapping[str, Any]]) -> str:
    """Compose the IGE promisingness prompt. The model is told the Go-Explore framing and asked for ONE
    integer (the index of the cell most promising to explore from toward a NEW level-up)."""
    lines = [
        "You are guiding a Go-Explore agent on an unfamiliar grid puzzle game.",
        "The agent has reached several distinct frontier states (cells). It will RETURN to ONE of them",
        "and explore new actions from there, trying to reach a NEW level (level-up).",
        "Each cell below shows: its index, the level it is at, how deep its path is (more actions = a",
        "harder-to-reach frontier), how many times it has already been explored from (fewer = more",
        "unexplored), and a coarse thumbnail of the screen (digits are colour regions).",
        "",
        "Pick the SINGLE cell most promising to explore from to make new progress. Prefer cells that are",
        "deep/under-explored and whose layout looks like it has unexplored structure to act on.",
        "",
    ]
    for d in descriptors:
        idx = int(d.get("index", 0))
        lines.append(
            f"CELL {idx}: level={int(d.get('level', 0))} depth={int(d.get('depth', 0))} "
            f"times_explored={int(d.get('visits', 0))} seen={int(d.get('seen', 0))}"
        )
        sig = d.get("signature")
        if sig is not None:
            lines.append(_render_signature(sig))
        lines.append("")
    lines.append(
        "Reply with ONLY the integer index of the single most promising cell (e.g. `3`). No other text."
    )
    return "\n".join(lines)


def _parse_index(text: str, n: int) -> Optional[int]:
    """Extract the chosen cell index from the model's reply. Takes the FIRST integer in range [0, n)."""
    for match in re.finditer(r"-?\d+", text or ""):
        try:
            value = int(match.group())
        except ValueError:
            continue
        if 0 <= value < n:
            return value
    return None


class IGECellSelector:
    """LLM-judged Go-Explore cell selector (callable). Returns the chosen cell index, or ``None`` to let
    the archive fall back to its heuristic. Holds a lazily-constructed local-GGUF proposer so the heavy
    ``arc_executable_world_model`` import (and the GPU server) is only touched when the selector actually
    fires. Inject ``proposer`` (anything with ``complete_text(prompt) -> (ok, text)``) for tests."""

    verifier_is_oracle = False

    def __init__(
        self,
        *,
        proposer: Any = None,
        enabled: bool = True,
        max_cells_in_prompt: int = 12,
        # None -> the canonical live-generator pin (ARC_LIVE_GENERATOR_REPO_SUBSTR, resolved
        # lazily in _get_proposer alongside the LocalGGUFProposer import). Deliberately NOT a
        # string literal here: this default and the one in from_config() below were two separate
        # copies of "Qwen3.5-9B-MTP", which is exactly the shape that lets a generator switch land
        # in one and not the other. Now there is nothing here to forget to update.
        repo_substr: Optional[str] = None,
        max_tokens: int = 16,
        temperature: float = 0.1,
    ) -> None:
        self.enabled = bool(enabled)
        self._proposer = proposer  # may be None -> lazily built on first call
        self._proposer_injected = proposer is not None
        self.max_cells_in_prompt = max(2, int(max_cells_in_prompt))
        # "" means "not overridden" -- _get_proposer() substitutes the canonical live pin. Kept as
        # a string rather than None so every existing reader (diagnostics, artifact dumps) that
        # does str(selector.repo_substr) keeps working unchanged.
        self.repo_substr = str(repo_substr) if repo_substr else ""
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        # diagnostics: every outcome is counted so the A/B can prove the selector actually fired and
        # was not silently a no-op (the discipline that catches a "win" that is really the heuristic).
        self.calls = 0
        self.llm_choices = 0
        self.parse_failures = 0
        self.server_unavailable = 0
        self.fallbacks = 0

    def _get_proposer(self) -> Any:
        if self._proposer is not None:
            return self._proposer
        # Lazy import keeps this module light and avoids an import cycle (arc_go_explore -> here ->
        # arc_executable_world_model would otherwise load at module import time).
        from carnot.agentic.arc_executable_world_model import (
            ARC_LIVE_GENERATOR_MTP_DEFAULT,
            ARC_LIVE_GENERATOR_NO_THINK_PREFIX,
            ARC_LIVE_GENERATOR_REPO_SUBSTR,
            LocalGGUFProposer,
        )

        # This selector defaults to port 8919 -- the SAME port _proposer() uses -- and relies on
        # LocalGGUFProposer's server reuse to share one warm model. That only works if the config
        # matches; a stale model/mtp/prefix here would make the shared server get refused and
        # relaunched on a fresh port, i.e. a second full model load. At 18.3 GB that is no longer
        # merely wasteful, it is an OOM. So all three come from the canonical pin.
        self._proposer = LocalGGUFProposer(
            repo_substr=self.repo_substr or ARC_LIVE_GENERATOR_REPO_SUBSTR,
            model_path=os.environ.get("CARNOT_ARC_GGUF_PATH") or None,
            mtp=(os.environ.get("CARNOT_ARC_MTP", ARC_LIVE_GENERATOR_MTP_DEFAULT) != "0"),
            kv_quant="q8_0",
            no_think_prefix=ARC_LIVE_GENERATOR_NO_THINK_PREFIX,
            max_tokens=max(self.max_tokens, 16),
            n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
            port=int(os.environ.get("CARNOT_IGE_LLM_PORT", "8919")),
        )
        return self._proposer

    def __call__(self, descriptors: Sequence[Mapping[str, Any]]) -> Optional[int]:
        if not self.enabled:
            return None
        descriptors = list(descriptors or [])
        n = len(descriptors)
        if n < 2:
            # Nothing to choose between -> let the archive handle the trivial case.
            return None
        prompt_cells = descriptors[: self.max_cells_in_prompt]
        self.calls += 1
        try:
            proposer = self._get_proposer()
            ok, text = proposer.complete_text(
                _build_prompt(prompt_cells),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception:
            ok, text = False, ""
        if not ok:
            self.server_unavailable += 1
            self.fallbacks += 1
            return None
        choice = _parse_index(text, len(prompt_cells))
        if choice is None:
            self.parse_failures += 1
            self.fallbacks += 1
            return None
        self.llm_choices += 1
        # Map back to the original descriptor's index field (descriptors carry their own archive index).
        return int(prompt_cells[choice].get("index", choice))

    def diagnostics(self) -> dict:
        return {
            "selector": "ige_llm_promisingness",
            "verifier_is_oracle": False,
            "enabled": bool(self.enabled),
            "proposer_injected": bool(self._proposer_injected),
            "calls": int(self.calls),
            "llm_choices": int(self.llm_choices),
            "parse_failures": int(self.parse_failures),
            "server_unavailable": int(self.server_unavailable),
            "fallbacks": int(self.fallbacks),
        }


def coerce_ige_cell_selector(value: Any, *, proposer: Any = None) -> Optional[IGECellSelector]:
    """Normalize a flag/config value into an IGECellSelector | None (mirrors coerce_go_explore_archive).

    - None / False / "" -> None (disabled; archive keeps its heuristic)
    - an IGECellSelector instance -> returned as-is
    - True / "ige" / "llm_promisingness" -> a default selector (lazy GPU proposer)
    - a Mapping -> a selector configured from its keys
    """
    if value is None or value is False or value == "":
        return None
    if isinstance(value, IGECellSelector):
        return value
    if value is True or (
        isinstance(value, str)
        and value.lower() in {"ige", "llm_promisingness", "llm_promisingness_go_explore"}
    ):
        return IGECellSelector(proposer=proposer)
    if isinstance(value, Mapping):
        if not bool(value.get("enabled", True)):
            return None
        return IGECellSelector(
            proposer=proposer,
            enabled=True,
            max_cells_in_prompt=int(value.get("max_cells_in_prompt", 12)),
            # No literal default: an absent/blank key means "use the canonical live pin", resolved
            # in _get_proposer(). See the __init__ signature's comment.
            repo_substr=value.get("repo_substr") or None,
            max_tokens=int(value.get("max_tokens", 16)),
            temperature=float(value.get("temperature", 0.1)),
        )
    return None

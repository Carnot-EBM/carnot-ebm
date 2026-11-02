"""REPRODUCE-ONLY: does the CEGIS refinement feedback path leak the held-out acceptance rows?

WHAT IS BEING CHECKED, IN ONE SENTENCE. `execute_bounded_llm_reinduction` accepts a round on
`heldout_accuracy` -- scored by `select_trusted_world_model` on the LAST THIRD of the transition
list -- while the refinement feedback it hands the LLM on a REJECTED round is built from
`WorldModelVerifier(list(transitions))`, i.e. the FULL corpus including that same last third,
with `true_change` (the observed answer) attached per mismatch. If any of those held-out
mismatches reaches the rendered prompt, then a row that GRADES the engine also TEACHES it.

WHY THIS SCRIPT EXISTS RATHER THAN A CODE READING. Four findings in the session that
commissioned this work were fabricated or mis-derived from greps. So every number below is
produced by CALLING the shipped functions -- `select_trusted_world_model`,
`WorldModelVerifier.score`, `_bounded_mismatches`, `refactor_prompt`, `induce_prompt` -- not by
reasoning about them. Nothing here is changed, flagged, or fixed: this is the reproduce phase.

WHAT IS DELIBERATELY *NOT* CLAIMED. That an LLM actually exploits the leak. Proving that needs a
live generator A/B. What is provable offline, and what this script proves, is the MEASUREMENT
defect: the rows that decide acceptance are not disjoint from the rows that shape refinement,
and the overlap is DELIVERED into the prompt string (not merely present in a dict the prompt
might drop -- see the delivery check, which reads the rendered text).

SUBSTRATE: pure Python/numpy over synthetic transitions. No LLM, no GPU, no network. Synthetic
is the right instrument here because the claim is STRUCTURAL (which row indices reach which
consumer), and a synthetic corpus lets every transition carry a unique, greppable delta so
"this held-out row's answer is in the prompt" is decidable by string search rather than by eye.
The live SHAPE is not invented: n=25 is the shape the shipped
`_induce_transitions_k` docstring tabulates for six captured games, and the prefix/held-out
split is computed by calling the shipped `_split_prefix_heldout`, never by arithmetic here.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_REPO, "python") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "python"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.agentic.arc_executable_world_model import (  # noqa: E402
    Transition,
    WorldModelVerifier,
    _bounded_mismatches,
    _delta,
    _rle_delta_compact,
    induce_prompt,
    refactor_prompt,
)
from carnot.agentic.arc_llm_reinduction import _counterexample_result  # noqa: E402
from carnot.agentic.arc_world_model_trust_energy import (  # noqa: E402
    WorldModelCandidate,
    _split_prefix_heldout,
    select_trusted_world_model,
)

GRID_H, GRID_W = 8, 8
GAME = "SYNTH"


# ---------------------------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------------------------
def build_corpus(n: int, *, terminal_levelup: bool) -> list[Transition]:
    """A realistic single-level trajectory: each step changes exactly ONE cell, to a colour no
    other step uses. Uniqueness is the instrument -- it makes "row i's answer appears in this
    prompt" a decidable string query instead of a judgement call.

    `terminal_levelup` mirrors the real induction window, which ENDS at the level-up
    transition. That row matters twice over: `WorldModelVerifier.score` excludes it from
    grading (so it can never become a mismatch and never enters `n`), while
    `_split_prefix_heldout` does NOT exclude it -- so it consumes one of the held-out slots
    without contributing a gradeable row.
    """
    g = np.zeros((GRID_H, GRID_W), dtype=int)
    rows: list[Transition] = []
    for i in range(n):
        nxt = g.copy()
        nxt[i // GRID_W, i % GRID_W] = i + 3  # colour i+3 occurs on exactly one row
        lvl_after = 1 if (terminal_levelup and i == n - 1) else 0
        rows.append(Transition(g.copy(), i % 4, None, nxt.copy(), 0, lvl_after))
        g = nxt
    return rows


def make_engine(rows: list[Transition], correct_idx: set[int]):
    """An engine that reproduces exactly the transitions in `correct_idx` and is a no-op
    elsewhere. A no-op is a genuinely WRONG prediction here (every row changes a cell), so a
    row outside `correct_idx` is a real mismatch, not a censored one."""
    table = {}
    for i in sorted(correct_idx):
        table[(rows[i].grid.tobytes(), int(rows[i].action))] = rows[i].next_grid.copy()

    def engine(grid, action, data=None):
        key = (np.asarray(grid).tobytes(), int(action))
        hit = table.get(key)
        return hit.copy() if hit is not None else np.asarray(grid).copy()

    return engine


def row_delta_tuples(rows: list[Transition], i: int) -> list[tuple[int, int, int, int]]:
    """Row `i`'s answer as `_delta` encodes it -- the exact tuples that land in a mismatch's
    `true_change`. Each row changes a DIFFERENT cell, so a tuple identifies its row uniquely."""
    return [tuple(x) for x in _delta(rows[i].grid, rows[i].next_grid)]


def row_rle_delta(rows: list[Transition], i: int) -> str:
    """Row `i`'s answer as the INDUCE prompt encodes it (`rNcM:<value>x<count>`). The `rNcM`
    coordinate prefix is unique per row, so an exact substring test is unambiguous -- an
    earlier version of this script matched on the colour alone and reported 24 of 25 rows
    shown at k=8, because "3x1" is a substring of "13x1". Recorded because it is precisely the
    substring-over-match failure this reproduction exists to avoid."""
    return _rle_delta_compact(rows[i].grid, rows[i].next_grid)


def parse_rendered_mismatches(prompt: str) -> list[dict[str, Any]]:
    """Parse the mismatch block back OUT of the rendered prompt STRING.

    This is the delivery check. Reading `_bounded_mismatches(...)`'s return value proves what
    was AVAILABLE; parsing the prompt proves what was DELIVERED. They can differ -- the prompt
    builder could truncate, re-encode or drop the block -- and "availability is not delivery"
    is a documented failure mode of this codebase.
    """
    start = prompt.find("MISMATCHES:\n")
    if start < 0:
        return []
    body = prompt[start + len("MISMATCHES:\n") :]
    end = body.rfind("]")
    if end < 0:
        return []
    try:
        parsed = json.loads(body[: end + 1])
    except json.JSONDecodeError:
        return []
    return [m for m in parsed if isinstance(m, dict)]


# ---------------------------------------------------------------------------------------------
# one configuration through the SHIPPED path
# ---------------------------------------------------------------------------------------------
def run_config(
    rows: list[Transition],
    *,
    n_prefix_correct: int,
    n_heldout_correct: int,
    prefix_pattern: str = "head",
) -> dict[str, Any]:
    prefix, heldout = _split_prefix_heldout(rows)
    n_prefix, n_heldout = len(prefix), len(heldout)
    p_idx = list(range(n_prefix))
    h_idx = list(range(n_prefix, len(rows)))
    if prefix_pattern == "head":
        chosen_p = p_idx[:n_prefix_correct]
    elif prefix_pattern == "tail":
        chosen_p = p_idx[n_prefix - n_prefix_correct :] if n_prefix_correct else []
    elif prefix_pattern == "stride":
        chosen_p = (
            p_idx[::2][:n_prefix_correct]
            + p_idx[1::2][: max(0, n_prefix_correct - len(p_idx[::2]))]
        )
    else:  # pragma: no cover - defensive
        raise ValueError(prefix_pattern)
    chosen_h = h_idx[:n_heldout_correct]
    engine = make_engine(rows, set(chosen_p) | set(chosen_h))

    # --- the SHIPPED acceptance path -----------------------------------------------------
    selection = select_trusted_world_model(
        list(rows),
        [WorldModelCandidate("repro", engine, None)],
        hidden_state=True,
    )
    heldout_accuracy = float(selection.selected_score.heldout_accuracy)
    prefix_accuracy = float(selection.selected_score.prefix_accuracy)
    accepted = heldout_accuracy >= 1.0  # the LIVE threshold (min_heldout_accuracy=1.0)

    # --- the SHIPPED refinement-feedback path (arc_llm_reinduction.py:1585) ---------------
    real_verify = WorldModelVerifier(list(rows), hud_mask=None).score(engine)
    mismatch_idx = [int(m["i"]) for m in real_verify.mismatches if "i" in m]
    rendered = _bounded_mismatches(list(real_verify.mismatches))
    rendered_idx = [int(m["i"]) for m in rendered if "i" in m]
    rendered_heldout = [i for i in rendered_idx if i >= n_prefix]

    # --- DELIVERY: does the answer reach the PROMPT STRING? -------------------------------
    counterexample = {
        "kind": "heldout_transition_verification_failed",
        "real_n": real_verify.n,
        "real_n_correct": real_verify.n_correct,
        "real_accuracy": float(real_verify.accuracy),
        "real_mismatches": list(real_verify.mismatches),
    }
    prompt = refactor_prompt(GAME, _counterexample_result(counterexample))
    # Identify rows by their OWN unique changed cell, parsed out of the delivered prompt text --
    # never by trusting the mismatch's `i` field, which is a label the renderer could drop.
    tuple_to_row = {}
    for i in range(len(rows)):
        for tup in row_delta_tuples(rows, i):
            tuple_to_row[tup] = i
    delivered_rows: set[int] = set()
    for m in parse_rendered_mismatches(prompt):
        for tup in m.get("true_change") or []:
            hit = tuple_to_row.get(tuple(tup))
            if hit is not None:
                delivered_rows.add(hit)
    delivered = sorted(i for i in delivered_rows if i >= n_prefix)
    delivered_prefix = sorted(i for i in delivered_rows if i < n_prefix)

    # --- cost side of design (i): what does prefix-only refinement discard? ---------------
    prefix_only = WorldModelVerifier(list(prefix), hud_mask=None).score(engine)
    prefix_only_idx = [int(m["i"]) for m in prefix_only.mismatches if "i" in m]

    return {
        "n": len(rows),
        "n_prefix": n_prefix,
        "n_heldout": n_heldout,
        "gradeable_n_full": int(real_verify.n),
        "n_levelup_rows_excluded": int(getattr(real_verify, "n_levelup_rows_excluded", 0)),
        "prefix_pattern": prefix_pattern,
        "n_prefix_correct_requested": n_prefix_correct,
        "n_heldout_correct_requested": n_heldout_correct,
        "prefix_accuracy": round(prefix_accuracy, 6),
        "heldout_accuracy": round(heldout_accuracy, 6),
        "accepted_at_live_threshold_1.0": bool(accepted),
        "n_mismatches_collected_full_corpus": len(mismatch_idx),
        "mismatch_idx_full_corpus": mismatch_idx,
        "n_mismatches_rendered": len(rendered_idx),
        "rendered_idx": rendered_idx,
        "n_rendered_from_heldout": len(rendered_heldout),
        "rendered_heldout_idx": rendered_heldout,
        "leaks": bool(rendered_heldout),
        "heldout_answers_delivered_to_prompt_idx": delivered,
        "n_heldout_answers_delivered": len(delivered),
        "prefix_answers_delivered_to_prompt_idx": delivered_prefix,
        "n_mismatches_prefix_only": len(prefix_only_idx),
        "prefix_only_starved": bool(len(prefix_only_idx) == 0 and len(mismatch_idx) > 0),
        "prompt_chars": len(prompt),
    }


# ---------------------------------------------------------------------------------------------
# induce-prompt leak: is the held-out tail SHOWN, with answers, before any refinement?
# ---------------------------------------------------------------------------------------------
def induce_prompt_leak(rows: list[Transition], *, k_env: str | None) -> dict[str, Any]:
    prefix, _heldout = _split_prefix_heldout(rows)
    n_prefix = len(prefix)
    prev = os.environ.get("CARNOT_ARC_INDUCE_TRANSITIONS_K")
    if k_env is None:
        os.environ.pop("CARNOT_ARC_INDUCE_TRANSITIONS_K", None)
    else:
        os.environ["CARNOT_ARC_INDUCE_TRANSITIONS_K"] = k_env
    try:
        text = induce_prompt(GAME, list(rows), 1)
    finally:
        if prev is None:
            os.environ.pop("CARNOT_ARC_INDUCE_TRANSITIONS_K", None)
        else:
            os.environ["CARNOT_ARC_INDUCE_TRANSITIONS_K"] = prev
    shown_prefix, shown_heldout = [], []
    for i in range(len(rows)):
        # EXACT rendered-delta substring, anchored on the row's unique `rNcM:` coordinate.
        hit = f"= {row_rle_delta(rows, i)}" in text
        if hit:
            (shown_prefix if i < n_prefix else shown_heldout).append(i)
    return {
        "CARNOT_ARC_INDUCE_TRANSITIONS_K": k_env if k_env is not None else "<unset:default>",
        "prompt_chars": len(text),
        "n_prefix": n_prefix,
        "n_heldout": len(rows) - n_prefix,
        "shown_prefix_idx": shown_prefix,
        "shown_heldout_idx": shown_heldout,
        "n_shown_heldout": len(shown_heldout),
        "heldout_shown_fraction": round(len(shown_heldout) / max(1, len(rows) - n_prefix), 4),
    }


def three_way_split_sizes(n: int) -> dict[str, Any]:
    """Actual sizes for design (ii) at each corpus size -- reported, not argued."""
    rows = build_corpus(n, terminal_levelup=True)
    prefix, heldout = _split_prefix_heldout(rows)
    # equal halving of the existing tail is the least-invasive three-way split: it leaves the
    # shipped prefix untouched and only divides the rows that already do not train the engine.
    half = len(heldout) // 2
    refine_tail, accept_tail = heldout[:half], heldout[half:]
    gradeable_accept = sum(1 for t in accept_tail if t.level_after <= t.level_before)
    # thirds is the alternative: split the WHOLE corpus three ways.
    t = n // 3
    return {
        "n": n,
        "shipped_two_way": {"prefix": len(prefix), "heldout": len(heldout)},
        "halve_the_tail": {
            "prefix": len(prefix),
            "refine_tail": len(refine_tail),
            "accept_tail": len(accept_tail),
            "gradeable_accept_rows_after_levelup_exclusion": gradeable_accept,
            "accept_accuracy_granularity": (
                round(1.0 / gradeable_accept, 4) if gradeable_accept else None
            ),
        },
        "equal_thirds": {"a": t, "b": t, "c": n - 2 * t},
    }


def main() -> int:
    t0 = time.time()
    out: dict[str, Any] = {"configs": [], "induce": [], "splits": []}

    for n, levelup in ((25, True), (25, False), (12, True)):
        rows = build_corpus(n, terminal_levelup=levelup)
        prefix, heldout = _split_prefix_heldout(rows)
        tag = f"n{n}_levelup{int(levelup)}"
        for pattern in ("head", "tail", "stride"):
            for p in range(0, len(prefix) + 1):
                for h in (0, max(0, len(heldout) - 1)):
                    r = run_config(
                        rows, n_prefix_correct=p, n_heldout_correct=h, prefix_pattern=pattern
                    )
                    r["corpus"] = tag
                    r["terminal_levelup"] = levelup
                    out["configs"].append(r)

    rows25 = build_corpus(25, terminal_levelup=True)
    for k_env in (None, "8"):
        out["induce"].append(induce_prompt_leak(rows25, k_env=k_env))

    for n in (12, 25, 30):
        out["splits"].append(three_way_split_sizes(n))

    out["wall_s"] = round(time.time() - t0, 3)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

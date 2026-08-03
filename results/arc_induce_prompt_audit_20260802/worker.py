#!/usr/bin/env python3
"""ONE game per process, KILLABLE. Render the REAL induce prompt and record what it contains.

WHAT THIS IS. A prompt audit, not an accuracy measurement. It renders `induce_prompt` on real
captured transitions and records structural facts about the resulting string: its token count
against the server's real per-slot budget, which transitions survive selection, what the
instructions ask the model to return, and whether anything in it makes "return the input
unchanged" an easy or licensed answer.

WHAT IT DOES NOT CLAIM. Nothing here scores an engine, so no held-out split protects anything
and none is asserted to. Every number below is a property of a STRING. The one place a split
matters is `_split_prefix_heldout`, and it is used ONLY to reproduce the frozen
`arc_goal_predicate_anatomy_20260801` prompt sha as a fidelity check that this harness renders
the same prompt that run did -- never to score anything.

THE SHAPE THAT MATTERS IS `live`. The live agent calls
`proposer.induce(self.short, active_transitions, self.cell, **kwargs)`
(arc_competition_agent.py:6497) with NO `previous_level_complete_grid` and, since the
2026-08-02 gating, no `win_transition` unless CARNOT_ARC_SUPPLY_WIN_TRANSITION=1. So the live
prompt carries the transitions block and nothing else. `anatomy` reproduces the frozen run's
shape (shown-half only) purely for the sha gate.

NO LLM, NO GPU, NO GENERATION, NO SUBMISSION, NO SCORED GAME. The tokenizer is loaded with
`vocab_only=True` -- vocabulary only, no weights -- per the CLAUDE.md GGUF tokenizer rule
(never AutoTokenizer on a GGUF repo id).
"""

import hashlib
import json
import os
import re
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# A stray write into the live engine store would corrupt evidence another session is reading.
# results/arc_e3 is EVIDENCE: read, never write. Redirect before the import that reads it.
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_prompt_audit_e3_never_written")
sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/python")
sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/scripts")

with open(sys.argv[1]) as _jf:
    job = json.loads(_jf.read())
game = job["game"]
gguf = job["gguf"]
dump_dir = job.get("dump_dir")

out = {"game": game}
t0 = time.time()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


try:
    import numpy as np
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_world_model_trust_energy as wmte

    w = atp.build_progress_window(game)
    if w is None:
        out["status"] = "no_window"
    else:
        window, _full, cell = w
        window = list(window)
        shown, held = wmte._split_prefix_heldout(window)

        # The object-perception flag must be UNSET to match both the shipped default and the
        # frozen anatomy run's `off` arm.
        os.environ.pop("CARNOT_ARC_OBJECT_PERCEPTION", None)

        def render(trans, k_env):
            """Render with an explicit CARNOT_ARC_INDUCE_TRANSITIONS_K, restoring it after.

            The k resolver reads the environment at call time, so the two arms have to be
            rendered under the two settings rather than by passing k= directly -- passing k
            directly would bypass `_induce_transitions_k` and stop testing the wiring that
            REQ-ARC-FCP-5699-23 was missing.
            """
            prev = os.environ.get("CARNOT_ARC_INDUCE_TRANSITIONS_K")
            if k_env is None:
                os.environ.pop("CARNOT_ARC_INDUCE_TRANSITIONS_K", None)
            else:
                os.environ["CARNOT_ARC_INDUCE_TRANSITIONS_K"] = str(k_env)
            try:
                return e3.induce_prompt(game, list(trans), int(cell))
            finally:
                if prev is None:
                    os.environ.pop("CARNOT_ARC_INDUCE_TRANSITIONS_K", None)
                else:
                    os.environ["CARNOT_ARC_INDUCE_TRANSITIONS_K"] = prev

        # ---- the renders ------------------------------------------------------------------
        # window_all  : the whole offline window at today's default (k -> None -> all).
        # window_k8   : the same window under the pre-2026-08-01 cap, for the truncation delta.
        # live        : LIVE-FAITHFUL. `_active_transitions()` returns
        #               `transitions[_episode_transition_start:]` and that start index is set
        #               one PAST the level-up row (`_begin_level_goal_episode`), so a live
        #               induce window CANNOT contain a level-up transition. The offline window
        #               from `build_progress_window` is built the opposite way -- it is sliced
        #               to END at the level-up -- so rendering it directly would fire the WIN
        #               TRANSITION block that the live path provably never fires (measured 0
        #               times across every rebuilt live prompt; see arc_competition_agent.py
        #               :6472). Dropping the level-up rows reproduces the live shape.
        # anatomy     : the frozen run's shape (shown half), for the sha reproduction gate.
        live_trans = [t for t in window if t.level_after <= t.level_before]
        p_window_all = render(window, None)
        p_window_k8 = render(window, 8)
        p_live_all = render(live_trans, None) if live_trans else ""
        p_live_k8 = render(live_trans, 8) if live_trans else ""
        p_anat = render(shown, None)
        out["n_live_trans"] = len(live_trans)

        # What the model is REALLY sent: the code-only directive is prepended and the
        # "```python" primer appended by `generate()` (arc_executable_world_model.py:5993).
        # Auditing the bare `induce_prompt` output alone would miss the single strongest
        # instruction in the payload.
        sent_all = (
            e3._L2_CODEONLY_DIRECTIVE
            + p_live_all
            + "\n\nReturn ONLY one ```python code block with engine + "
            "is_level_complete.\n```python\n"
        )

        # ---- transition accounting --------------------------------------------------------
        def changed_of(t):
            return not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))

        # Accounting is done on the LIVE transition set, because that is the set whose
        # selection the live agent actually experiences.
        n_changed = sum(1 for t in live_trans if changed_of(t))
        n_noop = len(live_trans) - n_changed
        sel_all = e3._select_transitions_for_prompt(
            [t for t in live_trans if changed_of(t)],
            [t for t in live_trans if not changed_of(t)],
            k=None,
        )
        sel_k8 = e3._select_transitions_for_prompt(
            [t for t in live_trans if changed_of(t)],
            [t for t in live_trans if not changed_of(t)],
            k=8,
        )
        out["transitions"] = {
            "n_window": len(window),
            "n_live_trans": len(live_trans),
            "n_changed": int(n_changed),
            "n_noop": int(n_noop),
            "n_shown_all": len(sel_all),
            "n_shown_k8": len(sel_k8),
            "n_changed_shown_all": sum(1 for t in sel_all if changed_of(t)),
            "n_changed_shown_k8": sum(1 for t in sel_k8 if changed_of(t)),
            "n_noop_shown_all": sum(1 for t in sel_all if not changed_of(t)),
            "n_noop_shown_k8": sum(1 for t in sel_k8 if not changed_of(t)),
            "n_changed_dropped_k8": int(n_changed) - sum(1 for t in sel_k8 if changed_of(t)),
            # Did the char budget ever bind under "all"? If every changed transition is present
            # the answer is no and the budget is decorative on this game.
            "char_budget_bound_all": sum(1 for t in sel_all if changed_of(t)) < int(n_changed),
            "grid_shape": [int(v) for v in np.asarray(window[0].grid).shape],
            "levelups_in_window": sum(1 for t in window if t.level_after > t.level_before),
            "n_distinct_actions_live": len({int(t.action) for t in live_trans}),
            "n_distinct_actions_shown_all": len({int(t.action) for t in sel_all}),
            "n_distinct_actions_shown_k8": len({int(t.action) for t in sel_k8}),
        }

        # Order of the FIRST rendered transition: a no-op shown first would teach that the
        # default outcome of acting is that nothing happens.
        first_is_noop = (not changed_of(sel_all[0])) if sel_all else None
        out["first_rendered_transition_is_noop"] = first_is_noop

        # ---- action coverage vs the DECLARED action space ---------------------------------
        # The prompt states "Actions are integers 1-7" and asks for `engine(grid, action, data)`
        # -- a TOTAL function over that space. What fraction of it does the evidence cover?
        # An action never observed has no evidence at all, and the only answer for its branch
        # that the prompt does not contradict is to leave the grid alone.
        obs_actions = sorted({int(t.action) for t in sel_all})
        out["action_space"] = {
            "declared_actions": list(range(1, 8)),
            "observed_actions": obs_actions,
            "n_declared": 7,
            "n_observed": len(obs_actions),
            "coverage_fraction": round(len(obs_actions) / 7.0, 4),
            "unobserved_actions": [a for a in range(1, 8) if a not in obs_actions],
            "single_action_only": len(obs_actions) == 1,
        }

        # ---- change sparsity --------------------------------------------------------------
        # If a transition changes a vanishing fraction of cells, the identity function is
        # already correct on almost every cell. That is a property of the DATA, and it is what
        # makes identity a locally sensible answer under any cell-wise reading of the task.
        cells_total = int(np.asarray(window[0].grid).size)
        fracs = []
        for t in sel_all:
            d = np.asarray(t.grid) != np.asarray(t.next_grid)
            fracs.append(float(d.sum()) / cells_total)
        fracs.sort()
        out["change_sparsity"] = {
            "cells_total": cells_total,
            "n_transitions": len(fracs),
            "changed_cell_fraction_min": round(fracs[0], 6) if fracs else None,
            "changed_cell_fraction_median": round(fracs[len(fracs) // 2], 6) if fracs else None,
            "changed_cell_fraction_max": round(fracs[-1], 6) if fracs else None,
            # The score the identity function earns under a CELL-WISE reading: the fraction of
            # cells it gets right. Stated for the shown transitions only, which is what the
            # model is looking at when it decides what to write.
            "identity_cellwise_accuracy_median": round(1.0 - fracs[len(fracs) // 2], 6)
            if fracs
            else None,
        }

        # ---- what the prompt asks for -----------------------------------------------------
        # Full-grid return vs delta return. The engine contract line is the load-bearing one.
        body = p_live_all
        out["asks"] = {
            "engine_returns_full_grid": "Return the predicted next grid (same shape)" in body,
            "mentions_delta_output_format": bool(re.search(r"[Rr]eturn .{0,60}delta", body)),
            "evidence_is_delta_encoded": "DELTA = the FULL set of changed cells" in body,
            "says_prefer_simple_general": "Prefer SIMPLE GENERAL rules" in body,
            "n_chars": len(body),
        }

        # ---- identity-invitation surface --------------------------------------------------
        out["identity_surface"] = {
            # Does any instruction license leaving cells alone / being conservative?
            "says_all_other_cells_unchanged": "all other cells are unchanged" in body,
            "says_unchanged_count": body.count("unchanged"),
            # A no-op transition renders as this literal, which IS the identity function
            # written out as an example.
            "noop_rendered_as_no_change": "(no change)" in body,
            "n_no_change_examples": body.count("(no change)"),
            # The code-only directive: does the payload actually tell the model not to reason?
            "codeonly_directive_present": "Skip all reasoning" in sent_all,
            "codeonly_forbids_grid_analysis": "Do NOT analyze the grids" in sent_all,
            "no_think_prefix": sent_all.startswith("/no_think"),
            # Anything forbidding a degenerate answer?
            "forbids_identity": bool(
                re.search(
                    r"(must (not|never) .{0,40}identity|do not return .{0,30}unchanged|"
                    r"engine must change|not the identity)",
                    body,
                    re.I,
                )
            ),
            "mentions_word_identity": "identity" in body.lower(),
        }

        # ---- blocks present ---------------------------------------------------------------
        out["blocks"] = {
            "win_transition_block": "WIN TRANSITION (this is how the level was completed)" in body,
            "opening_board_block": "BOARD AT THE START OF THE CURRENT LEVEL" in body,
            "object_structure_block": "OBJECT STRUCTURE" in body,
            "playbook_exemplars": body.startswith("EXEMPLAR") or "EXEMPLAR" in body[:400],
            "initial_grid_block": "INITIAL GRID" in body,
        }

        # ---- goal-only prompt comparison --------------------------------------------------
        # Constructed exactly as `_split_induce`'s fallback does: (game, previous_level_
        # complete_grid, trans). On the LIVE path previous_level_complete_grid is None because
        # the live induce call never passes it, so that is what is rendered here.
        class _P(e3.LocalGGUFProposer):
            def __init__(self):  # no server, no model -- only the prompt method is used
                pass

        gp_off_prev = os.environ.pop("CARNOT_ARC_GOAL_PROMPT_TRANSITIONS", None)
        goal_off = _P()._goal_only_prompt(game, None, list(live_trans))
        os.environ["CARNOT_ARC_GOAL_PROMPT_TRANSITIONS"] = "1"
        goal_on = _P()._goal_only_prompt(game, None, list(live_trans))
        if gp_off_prev is None:
            os.environ.pop("CARNOT_ARC_GOAL_PROMPT_TRANSITIONS", None)
        else:
            os.environ["CARNOT_ARC_GOAL_PROMPT_TRANSITIONS"] = gp_off_prev

        out["goal_prompt"] = {
            "shipped_chars": len(goal_off),
            "with_transitions_chars": len(goal_on),
            "shipped_carries_any_grid": "r0:" in goal_off or "\n0" in goal_off,
            "shipped_carries_transitions": "ACTION" in goal_off,
            "shipped_is_evidence_free": ("ACTION" not in goal_off) and ("r0:" not in goal_off),
            "flag_on_carries_transitions": "ACTION" in goal_on,
            "receives_win_transition": "WIN TRANSITION" in goal_on,
        }

        # ---- tokenize ---------------------------------------------------------------------
        from llama_cpp import Llama

        llm = Llama(model_path=gguf, vocab_only=True, verbose=False)

        def ntok(s: str) -> int:
            return len(llm.tokenize(s.encode(), add_bos=True, special=False))

        out["tokens"] = {
            "live_all": ntok(p_live_all) if p_live_all else None,
            "live_k8": ntok(p_live_k8) if p_live_k8 else None,
            "window_all": ntok(p_window_all),
            "window_k8": ntok(p_window_k8),
            "anatomy_shown": ntok(p_anat),
            "as_sent_all": ntok(sent_all) if p_live_all else None,
            "goal_shipped": ntok(goal_off),
            "goal_with_transitions": ntok(goal_on),
            "codeonly_directive": ntok(e3._L2_CODEONLY_DIRECTIVE),
        }
        out["sha"] = {
            "live_all": sha(p_live_all),
            "live_k8": sha(p_live_k8),
            "window_all": sha(p_window_all),
            "window_k8": sha(p_window_k8),
            "anatomy_shown": sha(p_anat),
        }
        out["prompt_chars"] = {
            "live_all": len(p_live_all),
            "live_k8": len(p_live_k8),
            "window_all": len(p_window_all),
            "window_k8": len(p_window_k8),
            "anatomy_shown": len(p_anat),
            "as_sent_all": len(sent_all),
        }
        out["k8_changes_prompt"] = sha(p_live_all) != sha(p_live_k8)
        out["status"] = "ok"

        if dump_dir:
            d = os.path.join(dump_dir, game)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "induce_all.txt"), "w") as f:
                f.write(p_live_all)
            with open(os.path.join(d, "as_sent_all.txt"), "w") as f:
                f.write(sent_all)
            with open(os.path.join(d, "goal_shipped.txt"), "w") as f:
                f.write(goal_off)
            with open(os.path.join(d, "induce_k8.txt"), "w") as f:
                f.write(p_live_k8)
            with open(os.path.join(d, "window_all.txt"), "w") as f:
                f.write(p_window_all)

except Exception as exc:  # noqa: BLE001 - a failed cell is a COVERAGE GAP, never a zero
    out["status"] = "error"
    out["error"] = f"{type(exc).__name__}: {exc}"[:400]

out["elapsed_s"] = round(time.time() - t0, 2)
with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1)

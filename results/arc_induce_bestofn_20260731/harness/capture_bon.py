#!/usr/bin/env python3
"""BEST-OF-N, STEP 1 -- capture the STALL-path induce prompt AND the planner's start state.

Direct descendant of `results/arc_induce_confirm_20260731/harness/capture_p3.py`, which
established the technique and why it is sound: the 2026-07-30 grid measured LLM-ON and LLM-OFF
action traces byte-identical on 5 of 6 games, so the trajectory leading to an induction point does
not depend on whether a generator exists. A proposer that RECORDS its `induce` arguments instead
of answering them therefore yields the prompt the ON arm would have sent, at seconds of pure-Python
arcade stepping instead of ~1100s of GPU decode.

WHAT THIS ADDS OVER capture_p3, and why Phase 1 cannot run without it.

capture_p3 recorded everything needed to score DYNAMICS (criterion (i)): the prompt, the
transitions, and the three-level split. It did NOT record what is needed to score the GOAL
(criterion (ii)) or the PLAN (criterion (iii)), because those two gates are not functions of the
transitions at all -- both are searches rooted at `root_grid`:

    _goal_satisfiability_check(engine=..., goal=..., start_grid=root_grid)      # (ii)
    plan_in_model(engine, is_level_complete, start_grid=root_grid)              # (iii)

`root_grid` is `E3AgentPolicy.root_grid`, handed to `execute_bounded_llm_reinduction` as a keyword
at the two call sites in `arc_competition_agent`. Nothing downstream of the proposer sees it, so a
proposer-side recorder cannot reach it. This module therefore wraps
`arc_competition_agent.execute_bounded_llm_reinduction` -- the BOUND name in the calling module,
not the definition in `arc_llm_reinduction`, because `arc_competition_agent` does
`from ... import execute_bounded_llm_reinduction` at import time and patching the definition site
would be a no-op. The wrapper records its kwargs and calls through unchanged.

It also captures the COMBINED prompt as the one that is measured. capture_p3 measured
`prompt{i}_engine.txt` because Phase 1-confirm was about engine VALIDITY. Best-of-N has to score
the goal predicate too, and only the combined `engine + is_level_complete` call emits one -- it is
also the SHIPPED happy path (`LocalGGUFProposer.induce` tries combined first and only falls back
to two focused calls when it fails).

THE STALL PATH IS call_index 2, and that is not a convention -- it is the whole point. The
mediation analysis this phase acts on found that 0 of 22 `stall` events (transition_count=25) ever
passed the goal gate while 4 of 6 `level_up_reinduction` events (transition_count=1) did, so a
criterion evaluated on the post-bank path selects for triviality and reproduces the reverse-causal
artifact. Each `execute_bounded_llm_reinduction` call yields exactly one `induce` call here (round
1 breaks on `proposer_failed`), so call_index 2 binds to the second reinduction site -- the stall
one. `n_transitions` is recorded per capture so the binding is CHECKED rather than assumed.

NO GPU IS TOUCHED: `CUDA_VISIBLE_DEVICES` is emptied and every generation entry point on
`LocalGGUFProposer` is replaced by a raiser, so a stray construction site cannot quietly reach a
model and turn this into an unlabelled LLM-ON run.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
HERE = os.path.dirname(os.path.abspath(__file__))

GAME = os.environ.get("CAP_GAME", "ft09")
SEED = int(os.environ.get("CAP_SEED", "1"))
BUDGET = int(os.environ.get("CAP_BUDGET", "60"))
MAX_IND = int(os.environ.get("CAP_MAX_IND", "2"))
EXPLORE_BUDGET = int(os.environ.get("CAP_EXPLORE_BUDGET", "24"))
OUT = os.path.join(HERE, "capture", GAME)
os.makedirs(OUT, exist_ok=True)

# Identical env pins to capture_p3, so this capture reproduces that one's induction points rather
# than a differently-configured neighbour of them.
os.environ.setdefault("CARNOT_ARC_GENERATOR_SEED", "3003")
os.environ["CARNOT_ARC_E3_DIR"] = os.path.join(OUT, "e3")  # results/arc_e3 is EVIDENCE
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.join(REPO, "python"))

ANON = "hg" + hashlib.sha256(f"{GAME}|heldout".encode()).hexdigest()[:6]
ARM_NAME = "frozen_gemma_pin"
ARM_CFG = {
    "desc": "gemma-4-31B live pin: codeonly fence ON, NO /no_think prefix, 4096 n_predict",
    "codeonly": "1",
    "no_think_prefix": "",
    "max_tokens": 4096,
    "tries": 3,
    "retrieval": "0",
    "static": "0",
}

CAPTURED: list[dict] = []
RECORDED: list[str] = []
PREFIX_CALLS: list[dict] = []
REINDUCTIONS: list[dict] = []


def main() -> int:  # noqa: C901
    import numpy as np

    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_competition_agent as aca
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_llm_reinduction as reind

    for mod in (atp, aca, e3, reind):
        assert mod.__file__.startswith(REPO), f"wrong repo code imported: {mod.__file__}"

    # RECORD-ONLY patch. Calls the real function, returns its result untouched.
    _real_prefix = reind._proposal_prefix

    def _recording_prefix(transitions):
        out = _real_prefix(transitions)
        PREFIX_CALLS.append({"full": list(transitions), "prefix": list(out)})
        return out

    reind._proposal_prefix = _recording_prefix

    # THE NEW RECORDER. Patch the name BOUND IN `arc_competition_agent` -- that module did
    # `from arc_llm_reinduction import execute_bounded_llm_reinduction` at import time, so
    # patching `reind.execute_bounded_llm_reinduction` would rebind a name nobody reads.
    _real_exec = aca.execute_bounded_llm_reinduction

    def _recording_exec(*a, **kw):
        trans = kw.get("transitions")
        REINDUCTIONS.append(
            {
                "order": len(REINDUCTIONS) + 1,
                "induce_calls_before": len(CAPTURED),
                "cell": kw.get("cell"),
                "n_transitions": len(list(trans)) if trans is not None else None,
                "root_grid": kw.get("root_grid"),
                "previous_level_complete_grid": kw.get("previous_level_complete_grid"),
                "positional_args": len(a),
            }
        )
        return _real_exec(*a, **kw)

    aca.execute_bounded_llm_reinduction = _recording_exec

    for name in (
        "_ensure_server",
        "_url",
        "generate",
        "complete_text",
        "_chat_complete_request",
        "refactor",
        "induce_programmatic_experts",
    ):
        if hasattr(e3.LocalGGUFProposer, name):

            def _mk(nm):
                def _blocked(*_a, **_kw):
                    raise RuntimeError(f"capture cell reached generation substrate: {nm}")

                return _blocked

            setattr(e3.LocalGGUFProposer, name, _mk(name))

    class CapturingProposer:
        def __init__(self) -> None:
            self.n_induce_calls = 0
            self.no_think_prefix = ""
            self.max_tokens = 0
            self.tries = 0
            self.include_playbook_exemplars = False
            self.timeout = 0

        def induce(self, game, trans, cell, *, previous_level_complete_grid=None):
            self.n_induce_calls += 1
            CAPTURED.append(
                {
                    "call_index": self.n_induce_calls,
                    "game": game,
                    "cell": int(cell),
                    "trans": trans,
                    "prev_grid": previous_level_complete_grid,
                    "include_playbook_exemplars": self.include_playbook_exemplars,
                    # Which reinduction site was on the stack when this fired. Index, not
                    # assumption -- see the module docstring on binding call_index 2 to the stall.
                    "reinduction_order": len(REINDUCTIONS),
                }
            )
            return False, "capture_only_no_generator_present"

        def world_model_candidates(self, _game=None):
            return []

        def __getattr__(self, name):
            raise AttributeError(name)

    _BasePolicy = aca.E3AgentPolicy

    class _TracingPolicy(_BasePolicy):  # type: ignore[misc,valid-type]
        def __init__(self, game_id, *a, **kw):
            super().__init__(ANON, *a, **kw)

        def next_move(self, frames, latest):
            kind, data = super().next_move(frames, latest)
            RECORDED.append(f"{kind}|{data!r}")
            return kind, data

    aca.E3AgentPolicy = _TracingPolicy
    atp.ARM_CONFIGS[ARM_NAME] = dict(ARM_CFG)

    t0 = time.monotonic()
    prop = CapturingProposer()
    res = atp.run_bounded_progress(
        GAME,
        ARM_NAME,
        proposer=prop,
        seed=SEED,
        budget=BUDGET,
        max_inductions=MAX_IND,
        wall_s=900,
        explore_budget=EXPLORE_BUDGET,
    )
    row = res.to_row(include_events=True, include_trace=True)

    def _changed(t) -> bool:
        return not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))

    meta = []
    for cap in CAPTURED:
        base = e3.induce_prompt(
            cap["game"],
            cap["trans"],
            cap["cell"],
            previous_level_complete_grid=cap["prev_grid"],
            k=e3._induce_transitions_k(),
            include_playbook_exemplars=cap["include_playbook_exemplars"],
        )
        engine_only = (
            base
            + "\n\nReturn ONLY one ```python code block defining engine(grid, action, data).\n"
            + "```python\n"
        )
        # THE ONE THIS PHASE MEASURES: the shipped happy path, and the only induce call that
        # emits an `is_level_complete` for criteria (ii)/(iii) to grade.
        combined = (
            base
            + "\n\nReturn ONLY one ```python code block with engine + is_level_complete.\n"
            + "```python\n"
        )

        def _wire(p: str) -> str:
            # generate() prepends the code-only directive + fence when codeonly_eligible, and
            # every induce call is. This is what the SERVER sees.
            return e3._L2_CODEONLY_DIRECTIVE + p + "\n```python\n"

        i = cap["call_index"]
        for fn, txt in (
            (f"prompt{i}_engine.txt", _wire(engine_only)),
            (f"prompt{i}_combined.txt", _wire(combined)),
        ):
            with open(os.path.join(OUT, fn), "w") as fh:
                fh.write(txt)
        with open(os.path.join(OUT, f"transitions{i}.pkl"), "wb") as fh:
            pickle.dump(cap["trans"], fh)

        # THE SPLIT. Match this induce call's prefix back to the `_proposal_prefix` call that
        # produced it. Identity match on list CONTENTS, not call order, and matches are CONSUMED,
        # so two induce calls carrying identity-equal one-row lists cannot both bind to the first.
        full = None
        for j, pc in enumerate(PREFIX_CALLS):
            if pc.get("_used"):
                continue
            if len(pc["prefix"]) == len(cap["trans"]) and all(
                a is b for a, b in zip(pc["prefix"], cap["trans"], strict=False)
            ):
                full = pc["full"]
                PREFIX_CALLS[j]["_used"] = True
                break
        if full is not None:
            with open(os.path.join(OUT, f"full_transitions{i}.pkl"), "wb") as fh:
                pickle.dump(full, fh)

        # THE PLANNER'S START STATE, bound to the reinduction call that was on the stack.
        ri = cap["reinduction_order"]
        root = None
        root_sha = None
        if 1 <= ri <= len(REINDUCTIONS):
            root = REINDUCTIONS[ri - 1]["root_grid"]
        if root is not None:
            arr = np.asarray(root)
            with open(os.path.join(OUT, f"root_grid{i}.pkl"), "wb") as fh:
                pickle.dump(arr, fh)
            root_sha = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
        if cap["prev_grid"] is not None:
            with open(os.path.join(OUT, f"prev_level_grid{i}.pkl"), "wb") as fh:
                pickle.dump(np.asarray(cap["prev_grid"]), fh)

        n_changing = sum(1 for t in cap["trans"] if _changed(t))
        meta.append(
            {
                "call_index": i,
                "game": cap["game"],
                "cell": cap["cell"],
                "n_transitions": len(cap["trans"]),
                "n_changing": int(n_changing),
                "n_full_transitions": len(full) if full is not None else None,
                "n_full_changing": (
                    int(sum(1 for t in full if _changed(t))) if full is not None else None
                ),
                "prefix_matched_full": full is not None,
                "actions": sorted({int(t.action) for t in cap["trans"]}),
                "grid_shape": list(np.asarray(cap["trans"][0].grid).shape) if cap["trans"] else None,
                "has_prev_level_complete_grid": cap["prev_grid"] is not None,
                # The new fields. `root_grid_captured` False on a call_index means criteria
                # (ii)/(iii) are UNMEASURABLE there and must be reported as such, never as a fail.
                "reinduction_order": ri,
                "root_grid_captured": root is not None,
                "root_grid_shape": list(np.asarray(root).shape) if root is not None else None,
                "root_grid_sha256_16": root_sha,
                "reinduction_n_transitions": (
                    REINDUCTIONS[ri - 1]["n_transitions"] if 1 <= ri <= len(REINDUCTIONS) else None
                ),
            }
        )

    payload = {
        "game": GAME,
        "seed": SEED,
        "budget": BUDGET,
        "n_induce_calls": prop.n_induce_calls,
        "n_proposal_prefix_calls": len(PREFIX_CALLS),
        "n_reinduction_calls": len(REINDUCTIONS),
        "captures": meta,
        "trace_len": len(RECORDED),
        "trace_sha256": hashlib.sha256("\n".join(RECORDED).encode()).hexdigest()[:16],
        "wall_s": round(time.monotonic() - t0, 1),
        "total_actions": row.get("total_actions"),
        "levels_gained": row.get("levels_gained"),
    }
    with open(os.path.join(OUT, "capture.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

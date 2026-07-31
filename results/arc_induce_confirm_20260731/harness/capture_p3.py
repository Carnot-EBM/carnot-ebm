#!/usr/bin/env python3
"""PHASE 1 (confirm), STEP 1 -- capture each game's REAL induce prompt, LLM-OFF, WITH the split.

Adapted from `results/arc_engine_validation_20260731/harness/capture_p2.py`, which established
the technique and why it is sound: the 2026-07-30 grid measured LLM-ON and LLM-OFF action traces
byte-identical on 5 of 6 games, so the trajectory leading to an induction point does not depend on
whether a generator exists. A proposer that RECORDS its `induce` arguments instead of answering
them therefore yields the prompt the ON arm would have sent, at seconds of pure-Python arcade
stepping instead of ~1100s of GPU decode.

WHAT THIS ADDS, AND WHY THE WHOLE PHASE DEPENDS ON IT. Phase 2's scoring was IN-SAMPLE and said so
loudly; retiring that caveat is the point of this run, and it cannot be done from the Phase-2
capture because that capture recorded the WRONG list. `arc_llm_reinduction._proposal_prefix`
removes a `round(n/3)` SUFFIX before calling `proposer.induce`, so what the proposer receives --
and therefore what capture_p2 pickled -- is ALREADY the prefix. The live held-out suffix was never
in that file. This capture patches `_proposal_prefix` to record its INPUT and return its output
unchanged, so all three levels survive into the scoring process:

  full     every transition the agent had collected at the induction point.
  prefix   `_proposal_prefix(full)` -- what `induce()` was handed. full minus a round(n/3) tail.
  shown    the <=8 rows `_transitions_block` actually RENDERS into the prompt text
           (`changed[:k-2] + noop[:2]`, k=8). This is all the model ever sees.

The model is accountable for `full \\ shown`, not for `full \\ prefix`: a prefix row whose delta
was never rendered is exactly as unseen as a suffix row. Both are reported, because the strict
suffix is what the production gate grades and the union is what actually tests generalisation.

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

# The 2026-07-30 grid's per-game sampler seed. Irrelevant to an inert proposer (nothing is
# sampled) but pinned so this cell's env matches the cell it reproduces.
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


def main() -> int:
    import numpy as np

    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_competition_agent as aca
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_llm_reinduction as reind

    for mod in (atp, aca, e3, reind):
        assert mod.__file__.startswith(REPO), f"wrong repo code imported: {mod.__file__}"

    # RECORD-ONLY patch. It calls the real function and returns its result untouched, so the
    # agent's behaviour is identical; only the input is remembered.
    _real_prefix = reind._proposal_prefix

    def _recording_prefix(transitions):
        out = _real_prefix(transitions)
        PREFIX_CALLS.append({"full": list(transitions), "prefix": list(out)})
        return out

    reind._proposal_prefix = _recording_prefix

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
        # produced it. Identity match on the list contents, not on call order, so an extra
        # `_proposal_prefix` call anywhere else in the agent cannot silently misalign the pair.
        # Matches are CONSUMED in order, so two induce calls carrying identity-equal one-row
        # lists (vc33) cannot both bind to the first `_proposal_prefix` call and report a
        # phantom suffix for the second.
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
            }
        )

    payload = {
        "game": GAME,
        "seed": SEED,
        "budget": BUDGET,
        "n_induce_calls": prop.n_induce_calls,
        "n_proposal_prefix_calls": len(PREFIX_CALLS),
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

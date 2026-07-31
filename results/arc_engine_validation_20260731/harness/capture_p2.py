#!/usr/bin/env python3
"""PHASE 2, STEP 4a -- capture each game's REAL induce prompt AND its transitions, LLM-OFF.

Adapted from `results/arc_induce_budget_20260731/harness/capture_prompts.py`, which established
the technique and the reason it is sound: the 2026-07-30 grid measured LLM-ON and LLM-OFF action
traces as byte-identical on 5 of 6 games, so the trajectory leading to an induction point does not
depend on whether a generator exists. Running the OFF arm with a proposer that RECORDS its
`induce` arguments instead of answering them yields the prompt the ON arm would have sent, at
seconds of pure-Python arcade stepping instead of ~1100s of GPU decode.

WHAT THIS ADDS over the Phase-1 version: it also PICKLES the transitions. Phase 1 only needed the
prompt (it was measuring token budgets); Phase 2's dry run has to RUN the generated engine against
the transitions the agent actually observed, so those have to survive into the next process.

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


def main() -> int:
    import numpy as np

    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_competition_agent as aca
    from carnot.agentic import arc_executable_world_model as e3

    for mod in (atp, aca, e3):
        assert mod.__file__.startswith(REPO), f"wrong repo code imported: {mod.__file__}"

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
            + "\n\nReturn ONLY one ```python code block defining engine(grid, action, data).\n```python\n"
        )
        combined = (
            base
            + "\n\nReturn ONLY one ```python code block with engine + is_level_complete.\n```python\n"
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
        n_changing = sum(
            1
            for t in cap["trans"]
            if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))
        )
        meta.append(
            {
                "call_index": i,
                "game": cap["game"],
                "cell": cap["cell"],
                "n_transitions": len(cap["trans"]),
                "n_changing": int(n_changing),
                "actions": sorted({int(t.action) for t in cap["trans"]}),
                "grid_shape": list(np.asarray(cap["trans"][0].grid).shape)
                if cap["trans"]
                else None,
                "has_prev_level_complete_grid": cap["prev_grid"] is not None,
            }
        )

    payload = {
        "game": GAME,
        "seed": SEED,
        "budget": BUDGET,
        "n_induce_calls": prop.n_induce_calls,
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

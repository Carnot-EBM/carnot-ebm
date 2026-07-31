#!/usr/bin/env python3
"""PHASE 1, STEP 1 -- capture ft09's REAL induce prompts without spending a single GPU token.

WHY THIS IS SOUND. The 2026-07-30 grid measured ft09's LLM-ON and LLM-OFF action traces as
byte-identical (normalized sha256 06f1e397129a03d9 on both arms, and again on the
reconstructed control). So the trajectory that leads up to induction point 1 -- and therefore
the transition set the induce prompt is built from -- does not depend on whether a generator
exists. Running the OFF arm with a proposer that RECORDS its `induce` arguments instead of
answering them yields exactly the prompt the ON arm would have sent, at ~8s of pure-Python
arcade stepping instead of ~1100s of GPU decode.

WHAT IT WRITES. For induction event 1 (the only one ft09 reaches):
  * prompt_combined.txt  -- the happy-path call: engine + is_level_complete in one shot
  * prompt_engine.txt    -- the split fallback's ENGINE-ONLY call
  * prompt_goal.txt      -- the split fallback's focused win-condition call
  * capture.json         -- transition count, grid shape, and the harness pins

Each is written with the SAME suffix `induce()` appends before calling `generate()`, and with
the code-only directive + opened fence that `generate()` prepends when
`codeonly_eligible=True` -- i.e. the byte-exact string the server receives. A prompt
reconstructed to "about the right shape" would make the Phase-1 token arithmetic a guess.

NO GPU IS TOUCHED: CUDA_VISIBLE_DEVICES is emptied and every generation entry point on
LocalGGUFProposer is replaced by a raiser, so a stray construction site cannot quietly reach a
model and turn this into an unlabelled LLM-ON run.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "prompts")
os.makedirs(OUT, exist_ok=True)

GAME = os.environ.get("CAP_GAME", "ft09")
SEED = int(os.environ.get("CAP_SEED", "1"))
BUDGET = int(os.environ.get("PROBE_BUDGET", "60"))
MAX_IND = int(os.environ.get("PROBE_MAX_IND", "2"))
EXPLORE_BUDGET = int(os.environ.get("PROBE_EXPLORE_BUDGET", "24"))
# The 2026-07-30 grid's per-game sampler seed. Irrelevant to an inert proposer (nothing is
# sampled) but pinned anyway so this cell's env is identical to the cell it is reproducing.
os.environ.setdefault("CARNOT_ARC_GENERATOR_SEED", "3003")

E3_DIR = os.path.join(HERE, "e3_capture", f"{GAME}__s{SEED}")
os.makedirs(E3_DIR, exist_ok=True)
os.environ["CARNOT_ARC_E3_DIR"] = E3_DIR  # results/arc_e3 is EVIDENCE; never written here
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
_T0 = time.monotonic()


def main() -> int:
    import numpy as np

    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_competition_agent as aca
    from carnot.agentic import arc_executable_world_model as e3

    for mod in (atp, aca, e3):
        assert mod.__file__.startswith(REPO), f"wrong repo code imported: {mod.__file__}"

    # Hard block on the generation surface. This cell must be provably LLM-off.
    _GEN = (
        "_ensure_server",
        "_url",
        "generate",
        "complete_text",
        "_complete_request",
        "_chat_complete_request",
        "refactor",
        "propose_many",
        "induce_programmatic_experts",
        "reflect",
    )
    for name in _GEN:
        if hasattr(e3.LocalGGUFProposer, name):

            def _mk(nm):
                def _blocked(*_a, **_kw):
                    raise RuntimeError(f"capture cell reached generation substrate: {nm}")

                return _blocked

            setattr(e3.LocalGGUFProposer, name, _mk(name))

    class CapturingProposer:
        """The inert contract, plus: record every `induce` call's arguments verbatim.

        `include_playbook_exemplars` mirrors the live proposer's value so the captured prompt
        carries the same exemplar block (or absence of one) the ON arm's prompt carried -- an
        exemplar block is thousands of tokens, so getting this wrong would corrupt the very
        arithmetic this capture exists to supply.
        """

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
                    "n_transitions": len(trans),
                    "grid_shape": list(np.asarray(trans[0].grid).shape) if trans else None,
                    "has_prev_level_complete_grid": previous_level_complete_grid is not None,
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

    # ---- rebuild the three prompts exactly as induce() would ------------------------------
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
        combined = (
            base
            + "\n\nReturn ONLY one ```python code block with engine + is_level_complete.\n```python\n"
        )
        engine_only = (
            base
            + "\n\nReturn ONLY one ```python code block defining engine(grid, action, data).\n```python\n"
        )
        # _goal_only_prompt is an instance method that touches no instance state beyond the
        # signature, so calling it unbound on the class is exact rather than approximate.
        goal_only = e3.LocalGGUFProposer._goal_only_prompt(
            None, cap["game"], cap["prev_grid"]
        )

        # generate() prepends the code-only directive + fence when codeonly_eligible. All three
        # induce calls are codeonly_eligible=True, so this is what the SERVER sees.
        def _wire(p: str) -> str:
            return e3._L2_CODEONLY_DIRECTIVE + p + "\n```python\n"

        i = cap["call_index"]
        files = {
            f"prompt{i}_combined.txt": _wire(combined),
            f"prompt{i}_engine.txt": _wire(engine_only),
            f"prompt{i}_goal.txt": _wire(goal_only),
        }
        for fn, txt in files.items():
            with open(os.path.join(OUT, fn), "w") as fh:
                fh.write(txt)
        meta.append(
            {
                "call_index": i,
                "game": cap["game"],
                "cell": cap["cell"],
                "n_transitions": cap["n_transitions"],
                "grid_shape": cap["grid_shape"],
                "has_prev_level_complete_grid": cap["has_prev_level_complete_grid"],
                "include_playbook_exemplars": cap["include_playbook_exemplars"],
                "induce_transitions_k": e3._induce_transitions_k(),
                "files": {fn: len(txt) for fn, txt in files.items()},
            }
        )

    payload = {
        "game": GAME,
        "seed": SEED,
        "anon_game_id": ANON,
        "budget": BUDGET,
        "max_inductions": MAX_IND,
        "explore_budget": EXPLORE_BUDGET,
        "n_induce_calls": prop.n_induce_calls,
        "captures": meta,
        "trace_len": len(RECORDED),
        "trace_sha256": hashlib.sha256("\n".join(RECORDED).encode()).hexdigest()[:16],
        "result_summary": {
            k: row.get(k)
            for k in (
                "total_actions",
                "levels_gained",
                "n_inductions",
                "n_plans_found",
                "reached_level",
            )
        },
        "wall_s": round(time.monotonic() - _T0, 1),
    }
    with open(os.path.join(OUT, "capture.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Does a row where the engine RAISES drop out of the graded metrics?

Construct: su15's source-derived expert (control_su15, frozen) wrapped so that it
answers only held-out row 19 (the block move) and raises on every other held-out
row. Score it with the real WorldModelVerifier and the real change_gate_decision.
Compare with the same engine returning identity instead of raising.
"""
import json
import os

import numpy as np

for _k in ("CARNOT_ARC_TRUST_METRIC", "CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "CARNOT_ARC_WM_HUD_MASK"):
    os.environ.pop(_k, None)
from carnot.agentic import arc_executable_world_model as e3  # noqa: E402
from carnot.agentic.arc_world_model_trust_energy import _split_prefix_heldout  # noqa: E402

S = "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/ad0c053d-41df-446e-99d6-d71368a47030/scratchpad/b2pc"
d = json.load(open("rescore_all.json"))
rows = []
for line in open(d["games"]["su15"]["window_file"]):
    r = json.loads(line)
    rows.append(e3.Transition(np.asarray(r["grid"]), int(r["action"]), r.get("data"), np.asarray(r["next_grid"]),
                              int(r["level_before"]), int(r["level_after"])))
_, heldout = _split_prefix_heldout(rows)
ns = {}
exec(open(f"{S}/control_su15/expert_engine.first_score.py").read(), ns)
expert = ns["engine"]
target = rows[19]


def is_target(grid, action, data):
    return np.array_equal(grid, target.grid) and action == target.action and data == target.data


def raiser(grid, action, data=None):
    if is_target(grid, action, data):
        return expert(grid, action, data)
    raise RuntimeError("abstain")


def identity_else(grid, action, data=None):
    if is_target(grid, action, data):
        return expert(grid, action, data)
    return np.array(grid)


out = {}
for name, eng in (("raise_on_7_of_8", raiser), ("identity_on_7_of_8", identity_else)):
    vr = e3.WorldModelVerifier(list(heldout), hud_mask_enabled=False).score(eng)
    cg = e3.change_gate_decision(vr, enabled=True)
    out[name] = {"accuracy": vr.accuracy, "n": vr.n, "n_engine_raised": vr.n_engine_raised,
                 "n_changing_graded": vr.n_changing, "cell_recall": vr.cell_recall,
                 "change_fidelity": vr.change_fidelity, "change_accuracy": vr.change_accuracy,
                 "n_noop_graded": vr.n_noop, "noop_hallucination_rate": vr.noop_hallucination_rate,
                 "change_gate_passed": cg.get("passed"), "change_gate_reason": cg.get("reason")}
json.dump(out, open("raise_inflation.json", "w"), indent=1)
print(json.dumps(out, indent=1))

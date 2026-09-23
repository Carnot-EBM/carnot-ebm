"""Same raiser engine through the live selector, with and without CARNOT_ARC_TRUST_METRIC=cell_recall."""
import json, os, sys
import numpy as np
os.environ.pop("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", None); os.environ.pop("CARNOT_ARC_WM_HUD_MASK", None)
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_world_model_trust_energy import WorldModelCandidate, select_trusted_world_model
S = "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/ad0c053d-41df-446e-99d6-d71368a47030/scratchpad/b2pc"
d = json.load(open("rescore_all.json"))
rows = [e3.Transition(np.asarray(r["grid"]), int(r["action"]), r.get("data"), np.asarray(r["next_grid"]), int(r["level_before"]), int(r["level_after"])) for r in map(json.loads, open(d["games"]["su15"]["window_file"]))]
ns = {}; exec(open(f"{S}/control_su15/expert_engine.first_score.py").read(), ns); expert = ns["engine"]
t19 = rows[19]
def raiser(grid, action, data=None):
    if np.array_equal(grid, t19.grid) and action == t19.action and data == t19.data:
        return expert(grid, action, data)
    raise RuntimeError("abstain")
out = {}
for metric in (None, "cell_recall"):
    if metric: os.environ["CARNOT_ARC_TRUST_METRIC"] = metric
    else: os.environ.pop("CARNOT_ARC_TRUST_METRIC", None)
    sel = select_trusted_world_model(list(rows), [WorldModelCandidate("raiser", raiser, None)], hidden_state=True)
    h = float(sel.selected_score.heldout_accuracy)
    out[str(metric)] = {"heldout_accuracy_gate": h, "prefix": float(sel.selected_score.prefix_accuracy), "accepted_at_1p0": h >= 1.0}
print(json.dumps(out, indent=1)); json.dump(out, open("raise_trustmetric.json", "w"), indent=1)

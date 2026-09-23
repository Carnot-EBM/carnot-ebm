"""Synthesis rescore: all 12 B2 v3 windows, one harness, three metrics.

Read-only on the repo. For each game: find the gate window, confirm it is
byte-identical across the 3 scored seeds, then score identity, lookup, the
control agent's frozen expert engine, and the 3 recorded first-shot engines on
(A) the live gate, (B) the unmasked held-out verifier, and (C) the proposed
pilot metric (per-game HUD mask on hidden-counter rows, known hidden-state
undo rows excluded).
"""
import glob
import hashlib
import json
import os
import signal
import sys

import numpy as np

for _k in ("CARNOT_ARC_TRUST_METRIC", "CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "CARNOT_ARC_WM_HUD_MASK"):
    os.environ.pop(_k, None)

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402
from carnot.agentic.arc_world_model_trust_energy import (  # noqa: E402
    WorldModelCandidate,
    _split_prefix_heldout,
    cegis_accept_split_enabled,
    select_trusted_world_model,
)

R = "/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_10009_b2_induction_gate_measurement_v3"
S = "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/ad0c053d-41df-446e-99d6-d71368a47030/scratchpad/b2pc"
SEEDS = ("7491001", "7491002", "7491003")
GAMES = ["sb26", "vc33", "su15", "g50t", "m0r0", "dc22", "wa30", "ka59", "bp35", "sp80", "ft09", "ar25"]
EXPERT = {
    "sb26": "control_sb26/expert_engine.py",
    "vc33": "control_vc33/expert_engine.py",
    "su15": "control_su15/expert_engine.first_score.py",
    "g50t": "control_g50t/expert_engine.py",
    "m0r0": "control_m0r0/expert_engine.py",
    "dc22": "control_dc22/expert_engine_frozen_v1.py",
    "wa30": "control_wa30/expert_engine_frozen_v1.py",
    "ka59": "control_ka59/expert_engine_first_score_snapshot.py",
    "bp35": "control_bp35/expert_engine.py",
    "sp80": "control_sp80/expert_engine_first_score_snapshot.py",
    "ft09": "control_ft09/expert_engine.py",
    "ar25": "control_ar25/expert_engine.py",
}
# Rows whose value tracks a hidden step counter (sim-twin proofs in the verifier reports).
MASK_ROWS = {"g50t": [63], "m0r0": [0, 63], "dc22": [63], "wa30": [63], "ka59": [63]}
# Held-out rows whose answer depends on a hidden undo stack / off-screen state.
EXCLUDE = {"sb26": [20], "ar25": [19], "bp35": [17, 18, 19, 21, 23, 24]}


class _TO(Exception):
    pass


def _alarm(*_):
    raise _TO()


signal.signal(signal.SIGALRM, _alarm)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def windows(game, seed):
    base = f"{R}/episodes/{game}__seed-{seed}/fresh_e3/{game}/attempts"
    out = []
    for p in sorted(glob.glob(f"{base}/.evidence-staging/*/transitions.jsonl") + glob.glob(f"{base}/evidence/*/transitions.jsonl")):
        n = sum(1 for _ in open(p))
        out.append((p, n, sha(p)))
    return out


def load_rows(path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        rows.append(e3.Transition(np.asarray(d["grid"]), int(d["action"]), d.get("data"), np.asarray(d["next_grid"]),
                                  int(d["level_before"]), int(d["level_after"])))
    return rows


def response_text(p):
    d = json.load(open(p))
    if isinstance(d.get("content"), str):
        return d["content"]
    try:
        return d["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def load_code(code, name):
    ns = {"__name__": name}
    exec(compile(code, name, "exec"), ns)
    return ns.get("engine")


def vr_dict(vr):
    return {k: getattr(vr, k) for k in ("n", "n_correct", "accuracy", "cell_recall", "change_fidelity", "n_changing",
                                         "n_changes_correct", "noop_hallucination_rate", "n_noop", "n_engine_raised",
                                         "n_levelup_rows_excluded", "hud_mask_status")}


def score(engine_factory, rows, game):
    """engine_factory() returns a FRESH engine for each metric (no state carried across)."""
    prefix, heldout = _split_prefix_heldout(rows)
    n_prefix = len(prefix)
    out = {}
    signal.alarm(120)
    try:
        eng = engine_factory()
        sel = select_trusted_world_model(list(rows), [WorldModelCandidate("c", eng, None)], hidden_state=True)
        out["A_live_gate_heldout"] = float(sel.selected_score.heldout_accuracy)
        out["A_live_gate_prefix"] = float(sel.selected_score.prefix_accuracy)
        out["A_accept_1p0"] = out["A_live_gate_heldout"] >= 1.0
        eng = engine_factory()
        out["B_unmasked"] = vr_dict(e3.WorldModelVerifier(list(heldout), hud_mask_enabled=False).score(eng))
        excl = set(EXCLUDE.get(game, []))
        idx = list(range(n_prefix, len(rows))) if len(rows) >= 2 else [0]
        keep = [rows[i] for i in idx if i not in excl]
        mask = None
        if game in MASK_ROWS:
            mask = np.zeros_like(np.asarray(rows[0].grid), dtype=bool)
            for r in MASK_ROWS[game]:
                mask[r, :] = True
        eng = engine_factory()
        sw = e3.hud_mask_swallow_check(list(rows), mask)
        vr = e3.WorldModelVerifier(keep, hud_mask=mask, hud_mask_enabled=mask is not None, hud_mask_swallow=sw).score(eng)
        out["C_pilot"] = vr_dict(vr)
        out["C_pilot"]["rows_scored"] = [i for i in idx if i not in excl]
        out["C_pilot"]["swallow_reason"] = sw.get("reason")
    except _TO:
        out["error"] = "timeout"
    except Exception as ex:
        out["error"] = repr(ex)[:200]
    finally:
        signal.alarm(0)
    return out


def main():
    res = {"cegis_accept_split_enabled": cegis_accept_split_enabled(), "games": {}}
    for game in GAMES:
        g = {}
        per_seed = {s: windows(game, s) for s in SEEDS}
        maxrows = {s: max(n for _, n, _ in per_seed[s]) for s in SEEDS}
        max_shas = {s: sorted({h for _, n, h in per_seed[s] if n == maxrows[s]}) for s in SEEDS}
        all_sizes = {s: sorted({(n, h[:12]) for _, n, h in per_seed[s]}) for s in SEEDS}
        g["window_rows_per_seed"] = maxrows
        g["window_sha_per_seed"] = {s: [h[:16] for h in v] for s, v in max_shas.items()}
        g["distinct_windows_across_seeds"] = len({h for v in max_shas.values() for h in v})
        g["all_transition_files_per_seed"] = all_sizes
        wpath = [p for p, n, _ in per_seed[SEEDS[0]] if n == maxrows[SEEDS[0]]][0]
        g["window_file"] = wpath
        rows = load_rows(wpath)
        prefix, heldout = _split_prefix_heldout(rows)
        g["split"] = {"n": len(rows), "n_prefix": len(prefix), "n_heldout": len(heldout),
                      "heldout_is_prefix_object": len(rows) < 2}
        keys = {}
        for i, t in enumerate(rows):
            keys.setdefault((np.asarray(t.grid).tobytes(), t.action, json.dumps(t.data, sort_keys=True)), []).append(i)
        table = {k: rows[v[0]].next_grid for k, v in keys.items()}
        g["distinct_keys"] = len(keys)

        def lookup_factory():
            def engine(grid, action, data=None):
                k = (np.asarray(grid).tobytes(), int(action), json.dumps(data, sort_keys=True))
                return np.array(table[k]) if k in table else np.array(grid)
            return engine

        def identity_factory():
            return lambda grid, action, data=None: np.array(grid)

        g["identity"] = score(identity_factory, rows, game)
        g["lookup"] = score(lookup_factory, rows, game)
        ep = f"{S}/{EXPERT[game]}"
        ecode = open(ep).read()
        g["expert_file"] = ep
        g["expert_sha256_16"] = hashlib.sha256(ecode.encode()).hexdigest()[:16]
        g["expert"] = score(lambda: load_code(ecode, f"expert_{game}"), rows, game)
        rec = {}
        for s in SEEDS:
            rp = f"{R}/{game}__seed-{s}/requests/00_response.json"
            r = {"response_file": rp}
            if not os.path.exists(rp):
                r["status"] = "no_response_file"
                rec[s] = r
                continue
            code = e3._extract_python(response_text(rp))
            r["code_sha256_16"] = hashlib.sha256(code.encode()).hexdigest()[:16]
            try:
                compile(code, "x", "exec")
            except SyntaxError as ex:
                r["status"] = f"syntax_error: {ex.msg} line {ex.lineno}"
                rec[s] = r
                continue
            try:
                eng = load_code(code, f"rec_{game}_{s}")
            except Exception as ex:
                r["status"] = f"exec_error: {repr(ex)[:120]}"
                rec[s] = r
                continue
            if eng is None:
                r["status"] = "no_engine_def"
                rec[s] = r
                continue
            r["status"] = "scored"
            r.update(score(lambda: load_code(code, f"rec_{game}_{s}"), rows, game))
            rec[s] = r
        g["recorded_first_shot"] = rec
        res["games"][game] = g
        c = g["expert"].get("C_pilot", {})
        print(game, "rows", maxrows, "distinct", g["distinct_windows_across_seeds"],
              "| expert A", g["expert"].get("A_live_gate_heldout"), "C acc", c.get("accuracy"), "C fid", c.get("change_fidelity"), c.get("hud_mask_status"),
              "| id C", g["identity"].get("C_pilot", {}).get("accuracy"),
              "| rec A", [rec[s].get("A_live_gate_heldout", rec[s].get("status")) for s in SEEDS],
              "| rec C acc", [rec[s].get("C_pilot", {}).get("accuracy") for s in SEEDS],
              "| rec C fid", [None if rec[s].get("C_pilot") is None else round(rec[s]["C_pilot"]["change_fidelity"], 3) for s in SEEDS],
              flush=True)
    json.dump(res, open("rescore_all.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

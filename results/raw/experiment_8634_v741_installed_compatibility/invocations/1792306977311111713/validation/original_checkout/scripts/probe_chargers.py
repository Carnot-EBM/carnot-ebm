"""Probe ARC games for a CHARGING-ENEMY (charger) mechanic like tu93's.

Charger signature:
  1. a DEATH transition: avatar present in pre-death grid, env _game_over in next frame
  2. a NON-AVATAR, non-wall, non-floor colour BLOCK (size ~4-30 cells) whose centroid
     TRANSLATED between the pre-death grid and the death frame (charged to intercept).
  3. centre-marker: does the charger block contain a distinct minority centre colour?
  4. avatar rigid-nav: does the avatar translate by a fixed step (InducedNavWorldModel-fittable)?
"""

from __future__ import annotations

import sys
import json
import copy
import numpy as np
from collections import Counter

import carnot.agentic.arc_solver_kit as kit
import carnot.agentic.arc_graph_explore as ge
from carnot.agentic.arc_executable_world_model import detect_cell, to_logical
from carnot.agentic.arc_nav_world_model import InducedNavWorldModel

from arcengine import GameAction

GAME_IDS = {
    "s5i5": "s5i5-18d95033",
    "ka59": "ka59-38d34dbb",
    "vc33": "vc33-5430563c",
    "g50t": "g50t-5849a774",
    "cn04": "cn04-2fe56bfb",
    "tu93": "tu93-0768757b",  # reference positive
}

rng = np.random.default_rng(42)


def step_act(env, act):
    """Step the env with a trajectory action which may be a dict {'action':id,'data':..} or a bare int."""
    if isinstance(act, dict):
        return env.step(ge._game_action(GameAction, act["action"]), data=act.get("data"))
    return env.step(ge._game_action(GameAction, int(act)), data=None)


def lgrid(frame):
    g = ge.grid_of(frame)
    c = detect_cell(np.asarray(g, dtype=np.int16))
    return to_logical(np.asarray(g, dtype=np.int16), c), c


def blobs_by_color(grid):
    """Return {color: [(cells, centroid_yx), ...]} 4-connected components per non-bg color."""
    from scipy import ndimage  # noqa

    out = {}
    colors = sorted(set(int(v) for v in grid.flatten().tolist()))
    for col in colors:
        mask = grid == col
        lab, n = ndimage.label(mask)
        comps = []
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            comps.append((len(ys), (float(ys.mean()), float(xs.mean())), (ys, xs)))
        out[col] = comps
    return out


def bg_color(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _replay_prefix(env, prefix):
    cur = ge._warm(env, False)
    for a in prefix or []:
        cur = step_act(env, a)
        if ge._game_over(cur):
            return None
    return cur


def collect_random_transitions(env, frame, n=60, max_steps=240, prefix=None):
    """Collect transitions via random keyboard actions 1-4. Records pre/post logical grids,
    action, levels, and game-over flag. Resets-on-death replay back to the captured start prefix
    so we keep sampling at the same level (the charger level)."""
    transitions = []
    deaths = []
    cur = frame
    steps = 0
    start_level = ge._levels_completed(cur)
    while len(transitions) < n and steps < max_steps:
        steps += 1
        avail = ge._available_action_ids(cur)
        kb = [a for a in avail if a in (1, 2, 3, 4)]
        if not kb:
            break
        a = int(rng.choice(kb))
        g0, c0 = lgrid(cur)
        lvl0 = ge._levels_completed(cur)
        try:
            nxt = env.step(ge._game_action(GameAction, a), data=None)
        except Exception:
            break
        over = ge._game_over(nxt)
        g1, c1 = lgrid(nxt)
        lvl1 = ge._levels_completed(nxt)
        transitions.append(
            {
                "grid": g0,
                "action": a,
                "next_grid": g1,
                "level_before": lvl0,
                "level_after": lvl1,
                "game_over": over,
            }
        )
        if over:
            deaths.append({"pre": g0, "post": g1, "action": a, "level": lvl0})
            # re-replay prefix back to the charger level so we keep sampling there
            cur = _replay_prefix(env, prefix)
            if cur is None:
                break
        else:
            cur = nxt
    return transitions, deaths, start_level


def fit_avatar(transitions):
    """Fit InducedNavWorldModel; return (model, is_rigid_nav, diag)."""
    try:
        rows = [
            (t["grid"], t["action"], t["next_grid"], t["level_before"], t["level_after"])
            for t in transitions
            if not t["game_over"]
        ]
        if len(rows) < 3:
            return None, False, {"reason": "too_few_nondeath_transitions", "n": len(rows)}
        m = InducedNavWorldModel.fit(rows)
        disp = m.displacement or {}
        nz = [v for v in disp.values() if v != (0, 0)]
        rigid = len(nz) >= 1 and bool(m.avatar_colors)
        return (
            m,
            rigid,
            {
                "displacement": {int(k): list(v) for k, v in disp.items()},
                "avatar_colors": sorted(int(x) for x in m.avatar_colors),
                "fit_quality": m.fit_quality,
            },
        )
    except Exception as e:
        return None, False, {"reason": f"fit_error: {e}"}


def analyze_death_for_charger(death, avatar_colors, bg, floor=None, wall_colors=None):
    """Check whether a non-avatar, non-floor, non-wall block TRANSLATED between pre and
    death frame by a NON-TRIVIAL amount (>= 2 cells, to exclude 1-cell HUD/footprint jitter)."""
    pre, post = death["pre"], death["post"]
    if pre.shape != post.shape:
        return None
    wall_colors = wall_colors or set()
    try:
        pre_blobs = blobs_by_color(pre)
        post_blobs = blobs_by_color(post)
    except Exception as e:
        return {"error": f"blob_error: {e}"}

    h, w = pre.shape
    movers = []
    colors = set(pre_blobs) | set(post_blobs)
    for col in colors:
        if col == bg or col in avatar_colors or col == floor or col in wall_colors:
            continue
        pre_c = pre_blobs.get(col, [])
        post_c = post_blobs.get(col, [])
        for psz, pcen, _ in pre_c:
            if not (3 <= psz <= 30):
                continue
            # skip bottom-edge HUD bars (centroid on last 2 rows)
            if pcen[0] >= h - 2:
                continue
            best = None
            for qsz, qcen, _ in post_c:
                if abs(qsz - psz) > max(2, psz // 2):
                    continue
                d = abs(pcen[0] - qcen[0]) + abs(pcen[1] - qcen[1])
                if best is None or d < best[0]:
                    best = (d, qcen, qsz)
            if best is not None and best[0] >= 2.0:  # moved >= 2 cells = a charge, not jitter
                movers.append(
                    {
                        "color": col,
                        "size": psz,
                        "pre_centroid": [round(x, 2) for x in pcen],
                        "post_centroid": [round(x, 2) for x in best[1]],
                        "shift_manhattan": round(best[0], 2),
                    }
                )
    return movers


def find_center_marker(grid, charger_color, bg, avatar_colors):
    """Does the charger block contain a distinct minority centre colour?
    Find a charger blob; check if it encloses a different minority colour at/near its centroid."""
    from scipy import ndimage  # noqa

    mask = grid == charger_color
    lab, n = ndimage.label(mask)
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if not (4 <= len(ys) <= 60):
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        sub = grid[y0 : y1 + 1, x0 : x1 + 1]
        inner = Counter(int(v) for v in sub.flatten().tolist())
        for col, cnt in inner.items():
            if col == charger_color or col == bg or col in avatar_colors:
                continue
            # minority colour inside the charger bbox = candidate facing marker
            if cnt < len(ys):
                return {
                    "marker_color": col,
                    "marker_count": cnt,
                    "charger_bbox": [int(y0), int(x0), int(y1), int(x1)],
                }
    return None


def probe_game(name, gid, max_levels=5):
    arc = kit.offline_arcade()
    env = arc.make(gid)
    frame = ge._warm(env, False)
    cell0 = detect_cell(np.asarray(ge.grid_of(frame), dtype=np.int16))
    tags_kb = any(a in (1, 2, 3, 4) for a in ge._available_action_ids(frame))

    report = {
        "game": name,
        "game_id": gid,
        "cell": int(cell0),
        "l1_actions": ge._available_action_ids(frame),
        "l1_keyboard_nav": bool(tags_kb),
        "levels": [],
        "charger_found": False,
        "charger_level": None,
        "charger_colors": [],
        "has_center_marker": False,
        "avatar_rigid_nav": None,
        "charger_moves_at_death": False,
        "notes": [],
    }

    # Blind-solve forward as deep as cheap, probing each level reached.
    prefix = []
    level = 0
    reached_levels = []
    while level <= max_levels:
        # collect transitions at this level
        cur = ge._warm(env, False)
        # replay prefix to reach the level
        ok = True
        for a in prefix:
            try:
                cur = step_act(env, a)
            except Exception:
                ok = False
                break
            if ge._game_over(cur):
                ok = False
                break
        if not ok:
            report["notes"].append(f"prefix_replay_failed_at_level_{level}")
            break
        cur_level = ge._levels_completed(cur)
        g, c = lgrid(cur)
        bg = bg_color(g)
        avail = ge._available_action_ids(cur)
        kb = [a for a in avail if a in (1, 2, 3, 4)]

        transitions, deaths, _ = collect_random_transitions(env, cur, n=60, prefix=prefix)
        m, rigid, diag = fit_avatar(transitions)
        avatar_colors = set(diag.get("avatar_colors", [])) if isinstance(diag, dict) else set()
        floor = int(m.floor_color) if m is not None and m.floor_color is not None else None
        wall_colors = (
            set(int(x) for x in m.wall_colors) if m is not None and m.wall_colors else set()
        )
        if report["avatar_rigid_nav"] is None:
            report["avatar_rigid_nav"] = bool(rigid)

        lvl_rec = {
            "level": cur_level,
            "cell": int(c),
            "keyboard_actions": kb,
            "n_transitions": len(transitions),
            "n_deaths": len(deaths),
            "rigid_nav": bool(rigid),
            "avatar_colors": sorted(avatar_colors),
            "fit_diag": diag,
            "chargers_at_death": [],
        }

        for d in deaths:
            movers = analyze_death_for_charger(
                d, avatar_colors, bg, floor=floor, wall_colors=wall_colors
            )
            if movers:
                lvl_rec["chargers_at_death"].append(movers)
                report["charger_moves_at_death"] = True
                report["charger_found"] = True
                if report["charger_level"] is None:
                    report["charger_level"] = cur_level
                for mv in movers:
                    if mv.get("color") not in report["charger_colors"]:
                        report["charger_colors"].append(mv["color"])
                    # center marker check on the pre-death grid
                    cm = find_center_marker(d["pre"], mv["color"], bg, avatar_colors)
                    if cm:
                        report["has_center_marker"] = True
                        lvl_rec.setdefault("center_markers", []).append(cm)

        report["levels"].append(lvl_rec)
        reached_levels.append(cur_level)

        # try to advance to next level via blind graph-explore
        if not kb:
            report["notes"].append(f"level_{cur_level}_not_keyboard_nav")
            break
        try:
            env2 = arc.make(gid)
            ge._warm(env2, False)
            traj, reached = ge.graph_explore_solve_v2(
                env2,
                start_level=cur_level,
                max_expansions=4000,
                max_depth=120,
                prefix=prefix if prefix else None,
            )
        except Exception as e:
            report["notes"].append(f"graph_explore_error_level_{cur_level}: {e}")
            break
        if traj is None or reached <= cur_level:
            report["notes"].append(f"could_not_advance_past_level_{cur_level} (reached={reached})")
            break
        prefix = list(traj)
        level = reached
        if reached in reached_levels:
            break

    return report


def main():
    targets = sys.argv[1:] or list(GAME_IDS.keys())
    out = {}
    for name in targets:
        gid = GAME_IDS[name]
        print(f"=== probing {name} ({gid}) ===", file=sys.stderr)
        try:
            out[name] = probe_game(name, gid)
        except Exception as e:
            import traceback

            out[name] = {"game": name, "error": str(e), "tb": traceback.format_exc()}
        r = out[name]
        print(
            json.dumps(
                {
                    k: r.get(k)
                    for k in (
                        "game",
                        "charger_found",
                        "charger_level",
                        "charger_colors",
                        "has_center_marker",
                        "avatar_rigid_nav",
                        "charger_moves_at_death",
                        "notes",
                    )
                },
                indent=2,
            ),
            file=sys.stderr,
        )
    with open("/tmp/charger_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE /tmp/charger_probe.json", file=sys.stderr)


if __name__ == "__main__":
    main()

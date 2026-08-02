#!/usr/bin/env python3
"""Probe ONE game's window/full trajectory sizes in a killable subprocess.

build_progress_window steps a real offline env and has no internal bound; the
sibling metric-validity worker records tr87 spinning at 100% CPU forever and
taking two independently-written drivers down. So: one game per process, driver
applies the timeout, a game that does not return is DROPPED with its reason
recorded and is never scored 0.
"""

from __future__ import annotations
import hashlib
import json
import os
import pathlib
import sys

REPO = pathlib.Path(os.environ["CARNOT_REPO"])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# NEVER let anything touch the real store: results/arc_e3 is EVIDENCE.
os.environ["CARNOT_ARC_E3_DIR"] = os.environ["SCRATCH_E3"]
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))


def main() -> int:
    game = sys.argv[1]
    import numpy as np
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_world_model_trust_energy as wmte

    w = atp.build_progress_window(game)
    if w is None:
        print(json.dumps({"status": "no_window", "game": game}))
        return 0
    win, full, cell = w
    shown, held = wmte._split_prefix_heldout(list(win))  # noqa: SLF001

    def sig(t):
        return hashlib.sha256(
            np.ascontiguousarray(np.asarray(t.grid)).tobytes()
            + b"|"
            + str(getattr(t, "action", None)).encode()
            + b"|"
            + np.ascontiguousarray(np.asarray(t.next_grid)).tobytes()
        ).hexdigest()

    shown_sigs = {sig(t) for t in shown}
    full_not_shown = [t for t in full if sig(t) not in shown_sigs]

    def n_chg(ts):
        return sum(1 for t in ts if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid)))

    # WorldModelVerifier `continue`s on level-up rows BEFORE counting them, so a row that
    # straddles a level-up is ungradable: the completing action re-lays the playfield out.
    def n_grad_chg(ts):
        return sum(
            1
            for t in ts
            if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))
            and not (getattr(t, "level_after", 0) > getattr(t, "level_before", 0))
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "game": game,
                "cell": int(cell),
                "n_window": len(win),
                "n_full": len(full),
                "n_shown": len(shown),
                "n_heldout_tail": len(held),
                "n_full_not_shown": len(full_not_shown),
                "tail_changing": n_chg(held),
                "tail_gradable_changing": n_grad_chg(held),
                "fullNS_changing": n_chg(full_not_shown),
                "fullNS_gradable_changing": n_grad_chg(full_not_shown),
                "window_is_subset_of_full": all(sig(t) in {sig(u) for u in full} for t in win),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

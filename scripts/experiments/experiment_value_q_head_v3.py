#!/usr/bin/env python3
"""Verifier-as-Q-head, STEP 3: a POSITION-PRESERVING value net (the architecture fix).

Step 2 diagnosed the bottleneck: ValueNet's AdaptiveAvgPool2d(1) (GLOBAL average pooling) discards
spatial position, but a navigation game's value IS position (avatar distance to goal) -> the value
stays flat (on-path 18.9 ~= off-path 19.1) and routes only modestly (1.21x, non-growing). This step
tests the fix: a SpatialValueNet that keeps a coarse 4x4 spatial map (32*16=512-d head input) so the
value can REPRESENT where the avatar is. Trained head-to-head against the global-pool ValueNet on
IDENTICAL data (blind-trace positives + hard negatives + far negatives). If the spatial net
DISCRIMINATES (on << hard << far) and ROUTES better, the architecture diagnosis is confirmed and we
have a real dense gradient. Honest, OFFLINE, CPU. verifier_is_oracle: false.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
from carnot.agentic.arc_value_net import ValueNet, _to_grid, GRID, NCOLORS

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_value_q_head_v3.json"

_spec = importlib.util.spec_from_file_location(
    "vqh2", str(REPO / "scripts" / "experiments" / "experiment_value_q_head_v2.py"))
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)


class SpatialValueNet:
    """Same interface as ValueNet (frame->float steps-to-go) but a POSITION-PRESERVING head: the conv
    stack is pooled to a coarse 4x4 map (not 1x1), so the MLP head sees WHERE objects are -- the
    spatial signal a navigation value needs and global-average-pooling destroys."""

    def __init__(self, device: str = "cpu") -> None:
        import torch
        import torch.nn as nn
        self.torch = torch
        self.device = device

        class SpatialCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(NCOLORS, 8)
                self.conv = nn.Sequential(
                    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64->32
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
                    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(4),                                       # KEEP a 4x4 map
                )
                self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 16, 64), nn.ReLU(),
                                          nn.Linear(64, 1))

            def forward(self, g):
                x = self.emb(g).permute(0, 3, 1, 2)
                return self.head(self.conv(x)).squeeze(-1)

        self.net = SpatialCNN().to(device)
        self.trained = False

    def fit(self, grids, values, epochs: int = 100, lr: float = 1e-3, batch: int = 64, seed: int = 0):
        torch = self.torch
        torch.manual_seed(seed)
        X = torch.as_tensor(np.stack(grids), dtype=torch.long, device=self.device)
        y = torch.as_tensor(np.asarray(values, dtype=np.float32), device=self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        lossf = torch.nn.SmoothL1Loss()
        n = X.shape[0]
        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = lossf(self.net(X[idx]), y[idx])
                loss.backward()
                opt.step()
        self.net.eval()
        self.trained = True
        self.last_train_loss = float(loss.item())
        return self

    def predict_grid(self, grid: np.ndarray) -> float:
        torch = self.torch
        with torch.no_grad():
            g = torch.as_tensor(grid[None], dtype=torch.long, device=self.device)
            return float(max(0.0, self.net(g).item()))

    def __call__(self, frame) -> float:
        if not self.trained:
            return 0.0
        return self.predict_grid(_to_grid(frame))


def evaluate(name, net, pos_all, hard_all, far_use, game, budget, blind_exp):
    on = float(np.mean([net.predict_grid(g) for g, vv in pos_all if vv <= 2])) if any(v <= 2 for _, v in pos_all) else 0.0
    hard = float(np.mean([net.predict_grid(g) for g, _ in hard_all])) if hard_all else None
    far = float(np.mean([net.predict_grid(g) for g, _ in far_use])) if far_use else None
    routed = v2.search(game, budget, net)
    speedup = (round(blind_exp / routed["expansions"], 2)
               if routed["offline_reproduced"] and routed["expansions"] else None)
    discriminates = hard is not None and far is not None and on < hard < far
    print(f"  [{name:9}] values on={on:.1f} hard={None if hard is None else round(hard,1)} "
          f"far={None if far is None else round(far,1)} discns={discriminates} | routed won="
          f"{routed['offline_reproduced']} exp={routed['expansions']} L{routed['reached_level']} "
          f"speedup={speedup}", flush=True)
    return {"on": round(on, 2), "hard": None if hard is None else round(hard, 2),
            "far": None if far is None else round(far, 2), "discriminates": bool(discriminates),
            "routed_won": routed["offline_reproduced"], "routed_exp": routed["expansions"],
            "routed_level": routed["reached_level"], "speedup": speedup}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=str, default="ls20")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--hard-branch", type=int, default=3)
    ap.add_argument("--hard-penalty", type=float, default=8.0)
    ap.add_argument("--far-rollouts", type=int, default=80)
    ap.add_argument("--max-len", type=int, default=45)
    ap.add_argument("--off-path-value", type=float, default=60.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    t0 = time.time()
    game = args.game
    rng = random.Random(args.seed)

    blind = v2.search(game, args.budget, None)
    print(f"  blind baseline: won={blind['offline_reproduced']} exp={blind['expansions']} L{blind['reached_level']}", flush=True)
    tr = v2.solve_trace(game, args.budget, None)
    if not tr["pos"]:
        OUT.write_text(json.dumps({"experiment": "experiment_value_q_head_v3", "game": game,
                                   "honest_verdict": "blocked_blind_no_L1"}, indent=2))
        print("  blind could not reach L1 -> blocked"); return 0
    pos_all = tr["pos"]
    hard_all = v2.hard_negatives(game, tr["labels"], tr["win_at"], args.hard_branch, args.hard_penalty, rng)
    far_all = v2.far_negatives(game, args.far_rollouts, args.max_len, args.off_path_value, rng)
    rng.shuffle(far_all)
    far_use = far_all[: max(len(hard_all), 20)]
    grids = [g for g, _ in pos_all] + [g for g, _ in hard_all] + [g for g, _ in far_use]
    values = [v for _, v in pos_all] + [v for _, v in hard_all] + [v for _, v in far_use]
    print(f"  data: {len(pos_all)} pos + {len(hard_all)} hard + {len(far_use)} far", flush=True)

    glob = ValueNet(device="cpu").fit(grids, values, epochs=args.epochs, seed=args.seed)
    spat = SpatialValueNet(device="cpu").fit(grids, values, epochs=args.epochs, seed=args.seed)
    res_glob = evaluate("global", glob, pos_all, hard_all, far_use, game, args.budget, blind["expansions"])
    res_spat = evaluate("spatial", spat, pos_all, hard_all, far_use, game, args.budget, blind["expansions"])

    g_sp = res_glob["speedup"] or 0
    s_sp = res_spat["speedup"] or 0
    spatial_wins = res_spat["discriminates"] and (s_sp > g_sp) and s_sp > (g_sp + 0.05)
    if res_spat["routed_level"] >= 2:
        verdict = "success: spatial_value_reached_L2_dense_routing"
    elif spatial_wins:
        verdict = "success: position_preserving_value_discriminates_and_routes_better_arch_fix_confirmed"
    elif res_spat["discriminates"] and not res_glob["discriminates"]:
        verdict = "complete: spatial_value_discriminates_where_global_does_not_modest_routing"
    else:
        verdict = "complete: spatial_arch_no_clear_win_honest_null_gap_sharpened"

    artifact = {"experiment": "experiment_value_q_head_v3", "game": game, "honest_verdict": verdict,
                "verifier_is_oracle": False, "inference_substrate": "offline_arc_search_plus_cpu_cnn_train",
                "random_seed": args.seed, "blind_expansions": blind["expansions"],
                "n_pos": len(pos_all), "n_hard": len(hard_all), "n_far": len(far_use),
                "global_pool": res_glob, "spatial": res_spat, "spatial_better": bool(spatial_wins),
                "duration_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nVERDICT: {verdict}\n  global speedup={g_sp} discns={res_glob['discriminates']} | "
          f"spatial speedup={s_sp} discns={res_spat['discriminates']} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

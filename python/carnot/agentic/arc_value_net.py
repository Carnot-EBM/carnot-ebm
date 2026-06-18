"""A LEARNED grid-CNN value head for the ARC-AGI-3 live explorer -- the higher-capacity successor to
the linear LearnedVerifier (arc_value_learner). The conclusive finding (results/arc_offline_to_live_
bridge_v2.json): a linear head over 5-41 hand-features CANNOT route the live search (it is actively
misleading when given control). The hypothesis here: a small CNN that sees the GRID directly has the
capacity to predict progress-to-win, IF trained on enough data (this is why the corpus builder adds
off-path negatives -- the discrimination the linear head lacked).

Same interface the live StepwiseExplorer already consumes: a callable `frame -> float` (predicted
steps-to-next-level-up; LOWER == closer to advancing a level). Frame-only (live-legal): reads only
grid_of(frame). Trains on CPU by default (the net is tiny -- ~64x64 grid in, one scalar out -- so it
needs no GPU and never contends with the conductor's 3090 experiments).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

GRID = 64          # ARC-AGI-3 frames are 64x64
NCOLORS = 16       # color indices 0-15


def _to_grid(frame: Any) -> np.ndarray:
    """frame -> a fixed 64x64 int grid of color indices (frame-only)."""
    from carnot.agentic.arc_agi3_world_model import grid_of
    g = np.asarray(grid_of(frame))
    if g.ndim == 1:
        s = int(round(g.size ** 0.5))
        g = g.reshape(s, s) if s * s == g.size else g.reshape(1, -1)
    out = np.zeros((GRID, GRID), dtype=np.int64)
    h, w = min(GRID, g.shape[0]), min(GRID, g.shape[1])
    out[:h, :w] = np.clip(g[:h, :w], 0, NCOLORS - 1)
    return out


def _build_net():
    import torch.nn as nn

    class ValueCNN(nn.Module):
        """Tiny grid CNN: embed colors -> 2 conv blocks -> global pool -> MLP -> scalar value."""

        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(NCOLORS, 8)
            self.conv = nn.Sequential(
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 64 -> 32
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, g):                       # g: (B,64,64) int64
            x = self.emb(g).permute(0, 3, 1, 2)     # (B,8,64,64)
            return self.head(self.conv(x)).squeeze(-1)

    return ValueCNN()


class ValueNet:
    """Trainable + savable + loadable grid-CNN value head. __call__(frame)->float for the explorer."""

    def __init__(self, device: str = "cpu") -> None:
        import torch
        self.torch = torch
        self.device = device
        self.net = _build_net().to(device)
        self.trained = False

    def fit(self, grids: Sequence[np.ndarray], values: Sequence[float],
            epochs: int = 60, lr: float = 1e-3, batch: int = 64, seed: int = 0) -> "ValueNet":
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

    def __call__(self, frame: Any) -> float:
        """frame -> predicted steps-to-next-level-up (LOWER == closer). The explorer's value_head."""
        if not self.trained:
            return 0.0
        return self.predict_grid(_to_grid(frame))

    def save(self, path: str | Path, meta: Optional[dict] = None) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # store weights as nested lists (json, mirror-ready per Decentralization Rule 3) + a meta sidecar
        state = {k: v.cpu().numpy().tolist() for k, v in self.net.state_dict().items()}
        p.write_text(json.dumps({
            "schema": "carnot_arc_value_cnn_v1", "kind": "grid_cnn_value_head",
            "grid": GRID, "ncolors": NCOLORS, "state_dict": state,
            "meta": meta or {},
        }))
        return p

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ValueNet":
        import torch
        d = json.loads(Path(path).read_text())
        v = cls(device=device)
        sd = {k: torch.as_tensor(np.asarray(w, dtype=np.float32)) for k, w in d["state_dict"].items()}
        v.net.load_state_dict(sd)
        v.net.eval()
        v.trained = True
        return v

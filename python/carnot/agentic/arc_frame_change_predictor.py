"""Frame-only ARC action-effect predictor and behavior prior.

Spec refs: REQ-ARC-FCP-4490, REQ-ARC-FCP-4491, SCENARIO-ARC-FCP-4490.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of


TERMINAL_ACTION_IDS = (1, 2, 3, 4, 5)
DEFAULT_NUM_COLORS = 16
DEFAULT_FRAME_SIZE = 64


def frame_state_key(frame: Any) -> str:
    """REQ-ARC-FCP-4490: stable state key from rendered frame pixels only."""

    return frame_hash(grid_of(frame))


def frame_to_tensor(
    frame: Any,
    *,
    num_colors: int = DEFAULT_NUM_COLORS,
    size: int = DEFAULT_FRAME_SIZE,
) -> torch.Tensor:
    """REQ-ARC-FCP-4490: convert an ARC frame/grid to a one-hot CHW tensor.

    The input is only the rendered frame data accepted by ``grid_of``. Colors
    outside the configured vocabulary are clipped into the final bucket so the
    live path never depends on private game objects or mirror feature vectors.
    """

    grid = np.asarray(grid_of(frame), dtype=np.int64)
    clipped = np.clip(grid, 0, int(num_colors) - 1)
    base = torch.from_numpy(np.ascontiguousarray(clipped)).long()
    one_hot = functional.one_hot(base, num_classes=int(num_colors)).permute(2, 0, 1).float()
    if one_hot.shape[-2:] != (int(size), int(size)):
        one_hot = functional.interpolate(
            one_hot.unsqueeze(0),
            size=(int(size), int(size)),
            mode="nearest",
        ).squeeze(0)
    return one_hot


class SmallFrameChangeCNN(nn.Module):
    """REQ-ARC-FCP-4490: small CNN with click heatmap and directional heads."""

    def __init__(self, num_colors: int = DEFAULT_NUM_COLORS, hidden_channels: int = 24) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(int(num_colors), int(hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.click_head = nn.Conv2d(int(hidden_channels), 1, kernel_size=1)
        self.directional_pool = nn.AdaptiveAvgPool2d(1)
        self.directional_head = nn.Linear(int(hidden_channels), len(TERMINAL_ACTION_IDS))

    def forward(self, frame_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(frame_tensor)
        click_heatmap = torch.sigmoid(self.click_head(feats))
        pooled = self.directional_pool(feats).flatten(1)
        directional_change = torch.sigmoid(self.directional_head(pooled))
        return click_heatmap, directional_change


@dataclass(frozen=True)
class BehaviorActionPrior:
    """REQ-ARC-FCP-4491: count-based marginal and state-conditioned action prior."""

    marginal_action_counts: Mapping[int, float] = field(default_factory=dict)
    click_cell_counts: Mapping[tuple[int, int], float] = field(default_factory=dict)
    state_action_counts: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    state_click_counts: Mapping[str, Mapping[tuple[int, int], float]] = field(default_factory=dict)
    marginal_weight: float = 0.10
    click_weight: float = 0.25
    state_action_weight: float = 0.45
    state_click_weight: float = 1.00

    @classmethod
    def from_examples(cls, examples: Sequence[Mapping[str, Any]]) -> "BehaviorActionPrior":
        """Build a prior from frame-only examples with ``action_id`` and optional click coords."""

        action_counts: dict[int, float] = {}
        click_counts: dict[tuple[int, int], float] = {}
        state_actions: dict[str, dict[int, float]] = {}
        state_clicks: dict[str, dict[tuple[int, int], float]] = {}

        for row in examples:
            action_id = int(row["action_id"])
            action_counts[action_id] = action_counts.get(action_id, 0.0) + 1.0
            state_key = str(row.get("state_key") or "")
            if state_key:
                per_action = state_actions.setdefault(state_key, {})
                per_action[action_id] = per_action.get(action_id, 0.0) + 1.0
            if action_id == 6 and row.get("x") is not None and row.get("y") is not None:
                cell = (int(row["x"]), int(row["y"]))
                click_counts[cell] = click_counts.get(cell, 0.0) + 1.0
                if state_key:
                    per_click = state_clicks.setdefault(state_key, {})
                    per_click[cell] = per_click.get(cell, 0.0) + 1.0

        return cls(
            marginal_action_counts=action_counts,
            click_cell_counts=click_counts,
            state_action_counts=state_actions,
            state_click_counts=state_clicks,
        )

    @staticmethod
    def _normalized_count(key: Any, counts: Mapping[Any, float]) -> float:
        total = float(sum(float(value) for value in counts.values()))
        if total <= 0.0:
            return 0.0
        return float(counts.get(key, 0.0)) / total

    def score(self, frame: Any, candidate: Any) -> float:
        action_id = int(getattr(candidate, "action_id"))
        score = self.marginal_weight * self._normalized_count(
            action_id,
            self.marginal_action_counts,
        )

        key = frame_state_key(frame)
        state_actions = self.state_action_counts.get(key, {})
        score += self.state_action_weight * self._normalized_count(action_id, state_actions)

        data = getattr(candidate, "data", None) or {}
        if action_id == 6 and "x" in data and "y" in data:
            cell = (int(data["x"]), int(data["y"]))
            score += self.click_weight * self._normalized_count(cell, self.click_cell_counts)
            state_clicks = self.state_click_counts.get(key, {})
            score += self.state_click_weight * self._normalized_count(cell, state_clicks)

        return float(score)


@dataclass
class FrameChangeScorer:
    """Score candidate actions from a trained ``SmallFrameChangeCNN``."""

    model: nn.Module
    num_colors: int = DEFAULT_NUM_COLORS
    size: int = DEFAULT_FRAME_SIZE
    device: str = "cpu"

    def _predict(self, frame: Any) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = frame_to_tensor(frame, num_colors=self.num_colors, size=self.size)
        self.model.eval()
        with torch.no_grad():
            click_heatmap, directional_change = self.model(
                tensor.unsqueeze(0).to(torch.device(self.device))
            )
        return click_heatmap[0, 0].detach().cpu(), directional_change[0].detach().cpu()

    def candidate_score(self, frame: Any, candidate: Any) -> float:
        click_heatmap, directional_change = self._predict(frame)
        action_id = int(getattr(candidate, "action_id"))
        data = getattr(candidate, "data", None) or {}
        if action_id == 6 and "x" in data and "y" in data:
            grid = grid_of(frame)
            h, w = grid.shape
            y = round((int(data["y"]) / max(1, h - 1)) * (self.size - 1))
            x = round((int(data["x"]) / max(1, w - 1)) * (self.size - 1))
            return float(click_heatmap[int(y), int(x)].item())
        if action_id in TERMINAL_ACTION_IDS:
            return float(directional_change[action_id - 1].item())
        return 0.0


def _scorer_value(frame: Any, candidate: Any, scorer: Any) -> float:
    if scorer is None:
        return 0.0
    if hasattr(scorer, "candidate_score"):
        return float(scorer.candidate_score(frame, candidate))
    if isinstance(scorer, Callable):
        return float(scorer(frame, candidate))
    raise TypeError("scorer must expose candidate_score(frame, candidate) or be callable")


def _delta_energy_value(frame: Any, candidate: Any, structural_energy_scorer: Any) -> float:
    if structural_energy_scorer is None:
        return 0.0
    if hasattr(structural_energy_scorer, "candidate_delta_energy"):
        return float(structural_energy_scorer.candidate_delta_energy(frame, candidate))
    if isinstance(structural_energy_scorer, Callable):
        return float(structural_energy_scorer(frame, candidate))
    raise TypeError(
        "structural_energy_scorer must expose candidate_delta_energy(frame, candidate) "
        "or be callable"
    )


def rank_arc_actions(
    frame: Any,
    candidates: Sequence[Any],
    *,
    scorer: Any | None = None,
    prior: BehaviorActionPrior | None = None,
    structural_energy_scorer: Any | None = None,
) -> list[Any]:
    """REQ-ARC-FCP-4491/4493: rank candidates by effect, prior, and optional energy."""

    if scorer is None and prior is None and structural_energy_scorer is None:
        return list(candidates)

    scored: list[tuple[float, int, Any]] = []
    for index, candidate in enumerate(candidates):
        p_change = _scorer_value(frame, candidate, scorer)
        score = p_change
        if structural_energy_scorer is not None:
            delta_energy = _delta_energy_value(frame, candidate, structural_energy_scorer)
            p_for_energy = p_change if scorer is not None else 1.0
            score = p_for_energy * (-delta_energy)
        if prior is not None:
            score += prior.score(frame, candidate)
        scored.append((float(score), index, candidate))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [candidate for _score, _index, candidate in scored]


def efficiency_score(human_actions: int, agent_actions: int) -> float:
    if human_actions <= 0 or agent_actions <= 0:
        return 0.0
    return float(min(float(human_actions) / float(agent_actions), 1.0) ** 2)


def evaluate_positive_control() -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4490: synthetic ranking sanity check with a known click."""

    frame = type("Frame", (), {})()
    frame.frame = np.zeros((8, 8), dtype=np.int16)
    frame.available_actions = [6]
    candidates = [
        type("Action", (), {"action_id": 6, "data": {"x": 1, "y": 1}, "source": "noop_a"})(),
        type("Action", (), {"action_id": 6, "data": {"x": 3, "y": 3}, "source": "noop_b"})(),
        type(
            "Action", (), {"action_id": 6, "data": {"x": 6, "y": 6}, "source": "changing_click"}
        )(),
    ]
    prior = BehaviorActionPrior(state_click_counts={frame_state_key(frame): {(6, 6): 1.0}})
    ranked = rank_arc_actions(frame, candidates, prior=prior)

    def actions_to_change(rows: Sequence[Any]) -> int:
        for index, candidate in enumerate(rows, start=1):
            if getattr(candidate, "source") == "changing_click":
                return index
        return len(rows) + 1  # pragma: no cover - positive control always includes the target.

    baseline_actions = actions_to_change(candidates)
    ranked_actions = actions_to_change(ranked)
    baseline_efficiency = efficiency_score(1, baseline_actions)
    ranked_efficiency = efficiency_score(1, ranked_actions)
    return {
        "baseline_actions_to_first_levelup": int(baseline_actions),
        "ranked_actions_to_first_levelup": int(ranked_actions),
        "actions_reduced": bool(ranked_actions < baseline_actions),
        "baseline_efficiency": baseline_efficiency,
        "ranked_efficiency": ranked_efficiency,
        "implied_efficiency_delta": ranked_efficiency - baseline_efficiency,
    }

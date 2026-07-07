"""Frame-only ARC action-effect predictor and behavior prior.

Spec refs: REQ-ARC-FCP-4490, REQ-ARC-FCP-4491, SCENARIO-ARC-FCP-4490,
REQ-ARC-FCP-5373, SCENARIO-ARC-FCP-5373.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from carnot.agentic.arc_agi3_live_adapter import ArcAction
from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of


TERMINAL_ACTION_IDS = (1, 2, 3, 4, 5)
DEFAULT_NUM_COLORS = 16
DEFAULT_FRAME_SIZE = 64
REPO_ROOT = Path(__file__).resolve().parents[3]
TRANSITION_CORPUS_RELATIVE_DIR = Path("data/arc_transition_corpus")
LIVE_CNN_CHECKPOINT_RELATIVE_PATH = Path("results/experiment_4629_live_frame_change_cnn.pt")


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
    _cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def _predict(self, frame: Any) -> tuple[torch.Tensor, torch.Tensor]:
        key = frame_state_key(frame)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        tensor = frame_to_tensor(frame, num_colors=self.num_colors, size=self.size)
        self.model.eval()
        with torch.no_grad():
            click_heatmap, directional_change = self.model(
                tensor.unsqueeze(0).to(torch.device(self.device))
            )
        result = click_heatmap[0, 0].detach().cpu(), directional_change[0].detach().cpu()
        self._cache[key] = result
        return result

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


@dataclass
class LiveActionEffectScorer:
    """REQ-ARC-FCP-4629: live scorer backed by persistent action effects and optional CNN."""

    memory: Any | None = None
    cnn_scorer: Any | None = None
    memory_weight: float = 1.0
    cnn_weight: float = 0.05
    source: str = "persistent_aem_plus_optional_cnn"

    def candidate_score(self, frame: Any, candidate: Any) -> float:
        score = 0.0
        if self.memory is not None:
            try:
                score += float(self.memory_weight) * float(self.memory.candidate_score(candidate))
            except Exception:
                pass
        if self.cnn_scorer is not None:
            try:
                score += float(self.cnn_weight) * float(
                    self.cnn_scorer.candidate_score(frame, candidate)
                )
            except Exception:
                pass
        return float(score)

    def as_dict(self) -> dict[str, Any]:
        memory = (
            self.memory.as_dict()
            if self.memory is not None and hasattr(self.memory, "as_dict")
            else None
        )
        return {
            "source": self.source,
            "memory": memory,
            "cnn_loaded": bool(self.cnn_scorer is not None),
            "memory_weight": float(self.memory_weight),
            "cnn_weight": float(self.cnn_weight),
        }


@dataclass
class GroundTruthValidatedFrameChangeScorer:
    """Gate frame-diff ranking until live observations validate the scorer.

    "Ground truth" here means the agent's own before/after frame pixels after a
    committed action. The wrapper prevents an unvalidated frame-diff prior from
    choosing probes solely because it is self-consistent on the current frame.
    """

    scorer: Any
    change_threshold: float = 0.0
    score_threshold: float = 0.0
    required_agreements: int = 1
    source: str = "observed_frame_diff_validated"
    _observed_count: int = field(default=0, init=False, repr=False)
    _agreement_count: int = field(default=0, init=False, repr=False)
    _contradiction_count: int = field(default=0, init=False, repr=False)
    _last_observed_delta: float | None = field(default=None, init=False, repr=False)

    @property
    def validated(self) -> bool:
        return bool(
            self._agreement_count >= int(self.required_agreements)
            and self._contradiction_count == 0
        )

    def candidate_score(self, frame: Any, candidate: Any) -> float:
        if not self.validated:
            return 0.0
        return float(_scorer_value(frame, candidate, self.scorer))

    def observe_transition(
        self,
        before: Any,
        action_id: int,
        data: Mapping[str, Any] | None,
        after: Any,
        *,
        source: str = "observed_transition",
    ) -> None:
        candidate = ArcAction(int(action_id), dict(data) if data else None, str(source))
        predicted_score = 0.0
        try:
            predicted_score = float(_scorer_value(before, candidate, self.scorer))
        except Exception:
            predicted_score = 0.0
        try:
            delta = _transition_frame_delta(grid_of(before), grid_of(after))
        except Exception:
            delta = 0.0
        predicted_changed = predicted_score > float(self.score_threshold)
        observed_changed = float(delta) > float(self.change_threshold)
        self._observed_count += 1
        self._last_observed_delta = float(delta)
        if predicted_changed == observed_changed:
            self._agreement_count += 1
        else:
            self._contradiction_count += 1
        if hasattr(self.scorer, "observe_transition"):
            try:
                self.scorer.observe_transition(before, action_id, data, after)
            except Exception:
                pass

    def reset(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self.scorer, "reset"):
            try:
                self.scorer.reset(*args, **kwargs)
            except Exception:
                pass

    def as_dict(self) -> dict[str, Any]:
        base = (
            self.scorer.as_dict()
            if self.scorer is not None and hasattr(self.scorer, "as_dict")
            else None
        )
        return {
            "source": self.source,
            "base_scorer": base,
            "frame_diff_ground_truth_validated": bool(self.validated),
            "observed_count": int(self._observed_count),
            "agreement_count": int(self._agreement_count),
            "contradiction_count": int(self._contradiction_count),
            "last_observed_delta": self._last_observed_delta,
        }

    def diagnostics(self) -> dict[str, Any]:
        return self.as_dict()


def _transition_frame_delta(before: Any, after: Any) -> float:
    lhs = np.asarray(before)
    rhs = np.asarray(after)
    if lhs.shape != rhs.shape:
        return 1.0
    total = int(lhs.size)
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(lhs != rhs) / total)


def load_cached_transition_effect_rows(
    root: Path | str = REPO_ROOT,
    *,
    limit: int | None = None,
    include_frames: bool = False,
) -> list[dict[str, Any]]:
    """REQ-ARC-FCP-4629: load local self-supervised action effects for live ranking."""

    transition_dir = Path(root) / TRANSITION_CORPUS_RELATIVE_DIR
    rows: list[dict[str, Any]] = []
    for path in sorted(transition_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        grids = data["grids"]
        next_grids = data["next_grids"]
        xs = data["xs"] if "xs" in data else np.full((grids.shape[0],), -1)
        ys = data["ys"] if "ys" in data else np.full((grids.shape[0],), -1)
        lb = data["lb"] if "lb" in data else np.zeros((grids.shape[0],), dtype=np.int16)
        la = data["la"] if "la" in data else np.zeros((grids.shape[0],), dtype=np.int16)
        for index in range(int(grids.shape[0])):
            action_id = int(data["actions"][index])
            grid = grids[index]
            delta = _transition_frame_delta(grid, next_grids[index])
            row: dict[str, Any] = {
                "game": path.stem,
                "env": path.stem,
                "state_key": frame_state_key(grid),
                "action_id": action_id,
                "changed": bool(delta > 0.0),
                "frame_delta": float(delta),
                "level_progress": 1.0 if int(la[index]) > int(lb[index]) else 0.0,
                "step_index": int(index),
                "feature_source": "arc_transition_corpus",
            }
            if action_id == 6 and int(xs[index]) >= 0 and int(ys[index]) >= 0:
                row["x"] = int(xs[index])
                row["y"] = int(ys[index])
            if include_frames:
                row["frame"] = grid
            rows.append(row)
            if limit is not None and len(rows) >= int(limit):
                return rows
    return rows


def save_live_frame_change_cnn_checkpoint(
    model: nn.Module,
    path: Path | str,
    *,
    num_colors: int = DEFAULT_NUM_COLORS,
    size: int = DEFAULT_FRAME_SIZE,
    hidden_channels: int = 24,
) -> Path:
    """REQ-ARC-FCP-4629: persist the locally trained small CNN for live loading."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_colors": int(num_colors),
            "size": int(size),
            "hidden_channels": int(hidden_channels),
        },
        out,
    )
    return out


def load_live_frame_change_cnn_scorer(
    root: Path | str = REPO_ROOT,
    *,
    checkpoint_path: Path | str | None = None,
    device: str = "cpu",
) -> FrameChangeScorer | None:
    """REQ-ARC-FCP-4629: load the graduated CNN scorer when the checkpoint exists."""

    path = (
        Path(checkpoint_path) if checkpoint_path is not None else LIVE_CNN_CHECKPOINT_RELATIVE_PATH
    )
    if not path.is_absolute():
        path = Path(root) / path
    if not path.exists():
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch compatibility.
            payload = torch.load(path, map_location="cpu")
        model = SmallFrameChangeCNN(
            num_colors=int(payload.get("num_colors") or DEFAULT_NUM_COLORS),
            hidden_channels=int(payload.get("hidden_channels") or 24),
        )
        model.load_state_dict(payload["state_dict"])
        return FrameChangeScorer(
            model,
            num_colors=int(payload.get("num_colors") or DEFAULT_NUM_COLORS),
            size=int(payload.get("size") or DEFAULT_FRAME_SIZE),
            device=device,
        )
    except Exception:
        return None


def load_live_action_effect_scorer(
    root: Path | str = REPO_ROOT,
    *,
    checkpoint_path: Path | str | None = None,
    exclude_games: Sequence[str] = (),
    use_memory: bool = True,
    use_cnn: bool = True,
) -> LiveActionEffectScorer | None:
    """REQ-ARC-FCP-4629: assemble the live action-effect scorer for E3AgentPolicy."""

    memory = None
    if use_memory:
        try:
            from carnot.agentic.arc_solver_kit import PersistentAEM

            rows = load_cached_transition_effect_rows(root)
            if rows:
                memory = PersistentAEM.from_effect_rows(rows, exclude_games=exclude_games)
        except Exception:
            memory = None
    cnn_scorer = (
        load_live_frame_change_cnn_scorer(root, checkpoint_path=checkpoint_path)
        if use_cnn
        else None
    )
    if memory is None and cnn_scorer is None:
        return None
    return LiveActionEffectScorer(memory=memory, cnn_scorer=cnn_scorer)


@dataclass(frozen=True)
class FrameActionEffectExample:
    """REQ-ARC-FCP-4501: one replay-derived action-effect row from raw frame pixels.

    The staged mirror can contain useful frame/action/delta rows without the
    separate 14,672-row `action_effect_dict.npz`. This object is the live-legal
    contract the predictor consumes: rendered frame pixels, normalized action
    id, optional click coordinates, and labels derived from frame deltas.
    """

    frame: Any
    action_id: int
    frame_delta: float
    level_progress: float
    state_key: str
    x: int | None = None
    y: int | None = None
    env: str = ""
    guid: str = ""
    step_index: int = 0
    feature_source: str = "raw_frame_shard_recomputed"

    @property
    def changed(self) -> bool:
        return bool(float(self.frame_delta) > 0.0)

    def to_prior_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "state_key": self.state_key,
            "action_id": int(self.action_id),
        }
        if self.x is not None and self.y is not None:
            row["x"] = int(self.x)
            row["y"] = int(self.y)
        return row


def normalize_action(action: Any) -> tuple[int | None, int | None, int | None]:
    """REQ-ARC-FCP-4501: normalize replay action encodings into action id and click."""

    data: Any = {}
    raw_id: Any = None
    if isinstance(action, Mapping):
        raw_id = (
            action.get("id")
            if action.get("id") is not None
            else action.get("action_id", action.get("action"))
        )
        data = action.get("data") or action
    else:
        raw_id = action

    try:
        action_id = int(raw_id)
    except (TypeError, ValueError):
        return None, None, None

    x_value = None
    y_value = None
    if isinstance(data, Mapping):
        x_value = data.get("x", data.get("click_x"))
        y_value = data.get("y", data.get("click_y"))
    try:
        x = None if x_value is None else int(x_value)
        y = None if y_value is None else int(y_value)
    except (TypeError, ValueError):
        x = None
        y = None
    return action_id, x, y


def normalize_frame_action_effect_row(row: Mapping[str, Any]) -> FrameActionEffectExample | None:
    """REQ-ARC-FCP-4501: build one frame-only training row from a staged shard row."""

    if "feature_keys" in row:
        raise ValueError("mirror feature_keys are not a frame-only input")
    if "frame" not in row:
        return None
    action_id, x, y = normalize_action(row.get("action"))
    if action_id is None:
        return None
    frame = row["frame"]
    return FrameActionEffectExample(
        frame=frame,
        action_id=int(action_id),
        x=x,
        y=y,
        frame_delta=float(row.get("frame_delta") or 0.0),
        level_progress=float(row.get("level_progress") or 0.0),
        state_key=frame_state_key(frame),
        env=str(row.get("env") or ""),
        guid=str(row.get("guid") or ""),
        step_index=int(row.get("step_index") or 0),
    )


def normalize_frame_action_effect_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int | None = None,
) -> Iterator[FrameActionEffectExample]:
    """REQ-ARC-FCP-4501: stream normalized examples without mirror state vectors."""

    emitted = 0
    for row in rows:
        example = normalize_frame_action_effect_row(row)
        if example is None:
            continue
        yield example
        emitted += 1
        if limit is not None and emitted >= int(limit):
            return


def load_frame_action_effect_examples(
    data_dir: Path | str,
    *,
    limit: int | None = None,
) -> list[FrameActionEffectExample]:
    """REQ-ARC-FCP-4501: load staged frame/action/delta shards from the local cache."""

    from carnot.agentic import arc_human_replay_corpus

    rows = arc_human_replay_corpus.load_training_shards(data_dir, limit=limit)
    return list(normalize_frame_action_effect_rows(rows, limit=limit))


def build_behavior_prior_from_effect_examples(
    examples: Sequence[FrameActionEffectExample],
) -> BehaviorActionPrior:
    """REQ-ARC-FCP-4501: emit the behavior-cloning prior from normalized examples."""

    return BehaviorActionPrior.from_examples([example.to_prior_row() for example in examples])


def _model_cell_for_example(example: FrameActionEffectExample, size: int) -> tuple[int, int] | None:
    if example.x is None or example.y is None:
        return None
    grid = grid_of(example.frame)
    h, w = grid.shape
    y = round((int(example.y) / max(1, h - 1)) * (int(size) - 1))
    x = round((int(example.x) / max(1, w - 1)) * (int(size) - 1))
    return int(y), int(x)


def _effect_loss_for_batch(
    model: nn.Module,
    examples: Sequence[FrameActionEffectExample],
    *,
    num_colors: int,
    size: int,
    device: torch.device,
) -> torch.Tensor | None:
    tensors = [
        frame_to_tensor(example.frame, num_colors=num_colors, size=size)
        for example in examples
        if _example_has_trainable_head(example)
    ]
    trainable = [example for example in examples if _example_has_trainable_head(example)]
    if not trainable:
        return None

    batch = torch.stack(tensors).to(device)
    click_heatmap, directional_change = model(batch)
    losses: list[torch.Tensor] = []
    for index, example in enumerate(trainable):
        target = torch.tensor(float(example.changed), dtype=torch.float32, device=device)
        if example.action_id == 6:
            cell = _model_cell_for_example(example, size)
            if cell is None:
                continue
            y, x = cell
            losses.append(functional.binary_cross_entropy(click_heatmap[index, 0, y, x], target))
        elif example.action_id in TERMINAL_ACTION_IDS:
            losses.append(
                functional.binary_cross_entropy(
                    directional_change[index, int(example.action_id) - 1],
                    target,
                )
            )
    if not losses:
        return None
    return torch.stack(losses).mean()


def _example_has_trainable_head(example: FrameActionEffectExample) -> bool:
    if example.action_id in TERMINAL_ACTION_IDS:
        return True
    return bool(example.action_id == 6 and example.x is not None and example.y is not None)


def _mean_effect_loss(
    model: nn.Module,
    examples: Sequence[FrameActionEffectExample],
    *,
    num_colors: int,
    size: int,
    batch_size: int,
    device: torch.device,
) -> float | None:
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), int(batch_size)):
            loss = _effect_loss_for_batch(
                model,
                examples[start : start + int(batch_size)],
                num_colors=num_colors,
                size=size,
                device=device,
            )
            if loss is not None:
                losses.append(float(loss.detach().cpu().item()))
    if not losses:
        return None
    return float(sum(losses) / len(losses))


def train_frame_change_model(
    examples: Sequence[FrameActionEffectExample],
    *,
    num_colors: int = DEFAULT_NUM_COLORS,
    size: int = DEFAULT_FRAME_SIZE,
    hidden_channels: int = 24,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    seed: int = 4501,
    device: str = "cpu",
) -> tuple[SmallFrameChangeCNN, dict[str, Any]]:
    """REQ-ARC-FCP-4501: train the small CNN on frame/action/frame-delta rows."""

    torch.manual_seed(int(seed))
    torch_device = torch.device(device)
    model = SmallFrameChangeCNN(num_colors=num_colors, hidden_channels=hidden_channels).to(
        torch_device
    )
    trainable = [example for example in examples if _example_has_trainable_head(example)]
    initial_loss = _mean_effect_loss(
        model,
        trainable,
        num_colors=num_colors,
        size=size,
        batch_size=batch_size,
        device=torch_device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    batches_trained = 0

    for _epoch in range(max(0, int(epochs))):
        model.train()
        for start in range(0, len(trainable), int(batch_size)):
            batch = trainable[start : start + int(batch_size)]
            loss = _effect_loss_for_batch(
                model,
                batch,
                num_colors=num_colors,
                size=size,
                device=torch_device,
            )
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batches_trained += 1

    final_loss = _mean_effect_loss(
        model,
        trainable,
        num_colors=num_colors,
        size=size,
        batch_size=batch_size,
        device=torch_device,
    )
    return model.cpu(), {
        "examples_seen": int(len(examples)),
        "examples_used": int(len(trainable)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_channels": int(hidden_channels),
        "num_colors": int(num_colors),
        "frame_size": int(size),
        "learning_rate": float(learning_rate),
        "batches_trained": int(batches_trained),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
    }


def _candidate_from_effect_example(index: int, example: FrameActionEffectExample) -> ArcAction:
    data = (
        {"x": int(example.x), "y": int(example.y)}
        if example.action_id == 6 and example.x is not None and example.y is not None
        else None
    )
    label = "target" if (example.level_progress > 0.0 or example.changed) else "noop"
    return ArcAction(int(example.action_id), data, f"heldout_{index}_{label}")


def _actions_to_first_target(candidates: Sequence[Any]) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if str(getattr(candidate, "source", "")).endswith("_target"):
            return int(index)
    return None


def evaluate_replay_candidate_order(
    examples: Sequence[FrameActionEffectExample],
    *,
    scorer: Any | None = None,
    prior: BehaviorActionPrior | None = None,
    min_candidates: int = 2,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4501: measure before/after rank to first replay-effective action."""

    by_state: dict[str, list[FrameActionEffectExample]] = {}
    for example in examples:
        by_state.setdefault(example.state_key, []).append(example)

    before_ranks: list[int] = []
    after_ranks: list[int] = []
    solved_before = 0
    solved_after = 0
    group_count = 0
    for state_examples in by_state.values():
        if len(state_examples) < int(min_candidates):
            continue
        candidates = [
            _candidate_from_effect_example(index, example)
            for index, example in enumerate(state_examples)
            if _example_has_trainable_head(example)
        ]
        if len(candidates) < int(min_candidates):
            continue
        if not any(str(candidate.source).endswith("_target") for candidate in candidates):
            continue
        group_count += 1
        before = _actions_to_first_target(candidates)
        ranked = rank_arc_actions(state_examples[0].frame, candidates, scorer=scorer, prior=prior)
        after = _actions_to_first_target(ranked)
        if before is not None:
            solved_before += 1
            before_ranks.append(before)
        if after is not None:
            solved_after += 1
            after_ranks.append(after)

    before_median = float(median(before_ranks)) if before_ranks else None
    after_median = float(median(after_ranks)) if after_ranks else None
    before_rate = float(solved_before / group_count) if group_count else 0.0
    after_rate = float(solved_after / group_count) if group_count else 0.0
    delta = (
        efficiency_score(1, int(after_median)) - efficiency_score(1, int(before_median))
        if before_median is not None and after_median is not None
        else None
    )
    return {
        "heldout_group_count": int(group_count),
        "heldout_median_actions_before": before_median,
        "heldout_median_actions_after": after_median,
        "solve_rate_before": before_rate,
        "solve_rate_after": after_rate,
        "solve_rate_dropped": bool(after_rate < before_rate),
        "implied_efficiency_delta": delta,
        "measurement_kind": "frame_only_replay_candidate_order_proxy",
    }


def _scorer_value(frame: Any, candidate: Any, scorer: Any) -> float:
    if scorer is None:
        return 0.0
    if hasattr(scorer, "candidate_score"):
        return float(scorer.candidate_score(frame, candidate))
    if isinstance(scorer, Callable):
        return float(scorer(frame, candidate))
    raise TypeError("scorer must expose candidate_score(frame, candidate) or be callable")


@dataclass
class ActionEffectExpansionPrior:
    """REQ-ARC-FCP-4641: score frontier branches by remaining predicted action effect.

    The candidate ranker answers "which action in this state should go first".
    The expansion prior answers the search-level question "which frontier state
    still has the most promising untested action". Lower returned values expand
    earlier, matching `graph_explore_solve_v2` and `StepwiseExplorer` frontier
    priority conventions.
    """

    scorer: Any
    weight: float = 0.25
    aggregate: str = "max"
    source: str = "persistent_aem_plus_optional_cnn_frontier_prior"
    verifier_is_oracle: bool = False
    _scored_frontiers: int = field(default=0, init=False, repr=False)
    _scored_candidates: int = field(default=0, init=False, repr=False)
    _scoring_errors: int = field(default=0, init=False, repr=False)

    def _effect_score(self, frame: Any, candidate: Any) -> float | None:
        try:
            return float(_scorer_value(frame, candidate, self.scorer))
        except Exception:
            self._scoring_errors += 1
            return None

    def frontier_priority(self, frame: Any, candidates: Sequence[Any]) -> float:
        """Return a lower-is-better branch priority from candidate effect scores."""

        rows = list(candidates or [])
        self._scored_frontiers += 1
        if self.scorer is None or not rows:
            return 0.0
        scores = [
            score
            for score in (self._effect_score(frame, candidate) for candidate in rows)
            if score is not None
        ]
        self._scored_candidates += len(scores)
        if not scores:
            return 0.0
        if self.aggregate == "mean":
            effect = sum(scores) / len(scores)
        else:
            effect = max(scores)
        return -float(self.weight) * max(0.0, float(effect))

    def __call__(self, frame: Any) -> float:
        return self.frontier_priority(frame, [])

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "source": self.source,
            "weight": float(self.weight),
            "aggregate": self.aggregate,
            "verifier_is_oracle": False,
            "scored_frontiers": int(self._scored_frontiers),
            "scored_candidates": int(self._scored_candidates),
            "scoring_errors": int(self._scoring_errors),
        }


def _prior_value(frame: Any, candidate: Any, prior: Any) -> float:
    if prior is None:
        return 0.0
    if hasattr(prior, "score"):
        return float(prior.score(frame, candidate))
    if isinstance(prior, Callable):
        return float(prior(frame, candidate))
    raise TypeError("prior must expose score(frame, candidate) or be callable")


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
            score += _prior_value(frame, candidate, prior)
        scored.append((float(score), index, candidate))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [candidate for _score, _index, candidate in scored]


def prune_arc_actions(
    frame: Any,
    candidates: Sequence[Any],
    *,
    scorer: Any | None,
    threshold: float | None,
    min_candidates: int = 1,
) -> tuple[list[Any], dict[str, Any]]:
    """REQ-ARC-FCP-4511: drop predicted no-op candidates before expansion.

    The pruning gate is deliberately opt-in and conservative. A missing scorer
    or threshold returns the legacy candidate list unchanged. If every candidate
    scores below threshold, the best-scoring legacy candidate is retained so the
    explorer never mistakes an uncertain model for proof that no action exists.
    """

    rows = list(candidates)
    diagnostics: dict[str, Any] = {
        "enabled": bool(scorer is not None and threshold is not None),
        "threshold": None if threshold is None else float(threshold),
        "candidate_count": int(len(rows)),
        "kept_count": int(len(rows)),
        "pruned_count": 0,
        "forced_keep_count": 0,
    }
    if scorer is None or threshold is None or not rows:
        return rows, diagnostics

    scored: list[tuple[float, int, Any]] = [
        (_scorer_value(frame, candidate, scorer), index, candidate)
        for index, candidate in enumerate(rows)
    ]
    kept = [candidate for score, _index, candidate in scored if score >= float(threshold)]
    forced_keep_count = 0
    if not kept and min_candidates > 0:
        forced_keep_count = min(int(min_candidates), len(scored))
        kept = [
            candidate
            for _score, _index, candidate in sorted(
                scored,
                key=lambda row: (-row[0], row[1]),
            )[:forced_keep_count]
        ]
    diagnostics.update(
        {
            "kept_count": int(len(kept)),
            "pruned_count": int(len(rows) - len(kept)),
            "forced_keep_count": int(forced_keep_count),
            "min_score_kept": min(
                (_scorer_value(frame, candidate, scorer) for candidate in kept),
                default=None,
            ),
            "max_score_pruned": max(
                (score for score, _index, candidate in scored if candidate not in kept),
                default=None,
            ),
        }
    )
    return kept, diagnostics


def prune_arc_actions_by_prior_quantile(
    frame: Any,
    candidates: Sequence[Any],
    *,
    prior: Any | None,
    prune_quantile: float | None,
    min_candidates: int = 1,
) -> tuple[list[Any], dict[str, Any]]:
    """REQ-ARC-FCP-4512: drop the bottom prior-likelihood quantile before expansion."""

    rows = list(candidates)
    diagnostics: dict[str, Any] = {
        "enabled": bool(prior is not None and prune_quantile is not None),
        "prune_quantile": None if prune_quantile is None else float(prune_quantile),
        "candidate_count": int(len(rows)),
        "kept_count": int(len(rows)),
        "pruned_count": 0,
        "forced_keep_count": 0,
    }
    if prior is None or prune_quantile is None or not rows:
        return rows, diagnostics

    quantile = max(0.0, min(1.0, float(prune_quantile)))
    max_prunable = max(0, len(rows) - max(0, int(min_candidates)))
    prune_count = min(max_prunable, int(len(rows) * quantile))
    if prune_count <= 0:
        return rows, diagnostics

    scored: list[tuple[float, int, Any]] = [
        (_prior_value(frame, candidate, prior), index, candidate)
        for index, candidate in enumerate(rows)
    ]
    prune_indexes = {
        index
        for _score, index, _candidate in sorted(scored, key=lambda row: (row[0], row[1]))[
            :prune_count
        ]
    }
    kept = [candidate for _score, index, candidate in scored if index not in prune_indexes]
    forced_keep_count = 0
    if not kept and min_candidates > 0:
        forced_keep_count = min(int(min_candidates), len(scored))
        kept = [
            candidate
            for _score, _index, candidate in sorted(
                scored,
                key=lambda row: (-row[0], row[1]),
            )[:forced_keep_count]
        ]
    diagnostics.update(
        {
            "kept_count": int(len(kept)),
            "pruned_count": int(len(rows) - len(kept)),
            "forced_keep_count": int(forced_keep_count),
            "min_score_kept": min(
                (_prior_value(frame, candidate, prior) for candidate in kept),
                default=None,
            ),
            "max_score_pruned": max(
                (score for score, index, _candidate in scored if index in prune_indexes),
                default=None,
            ),
        }
    )
    return kept, diagnostics


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

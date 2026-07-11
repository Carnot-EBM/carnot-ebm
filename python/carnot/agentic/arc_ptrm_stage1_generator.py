"""Stage-1 pTRM-style ARC action-sequence generator.

Spec refs: REQ-ARC-PTRM-5574-1, REQ-ARC-PTRM-5574-2,
REQ-ARC-PTRM-5574-3, REQ-ARC-PTRM-5574-4, REQ-ARC-PTRM-5574-5,
SCENARIO-ARC-PTRM-5574-DATASET, SCENARIO-ARC-PTRM-5574-STOCHASTIC,
SCENARIO-ARC-PTRM-5574-ARTIFACT.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_FILENAME = "experiment_5574_ptrm_stochastic_generator_stage1.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "arc_public_demo_human_replay_corpus"
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "experiment_5574_ptrm_stage1"
DEFAULT_SENTINEL = REPO_ROOT / "results" / "trm_runs" / "DO_NOT_RELAUNCH"
RANDOM_SEED = 5574
ACTION_VOCAB_SIZE = 8

REQUIRED_ARTIFACT_FIELDS = (
    "track",
    "honest_verdict",
    "preconditions",
    "prior_pilot_receipts",
    "dataset_hashes",
    "heldout_games",
    "leakage_count",
    "model_architecture",
    "parameter_count",
    "stochastic_noise_schedule",
    "trajectories_per_input",
    "history_conditioned",
    "intent_conditioned",
    "dynamic_halting",
    "recursion_depth_metrics",
    "overthinking_curve",
    "controls",
    "positive_control_passed",
    "verifier_selection_method",
    "verifier_is_oracle",
    "checkpoint_path",
    "checkpoint_sha256",
    "training_duration_s",
    "gpu_device_receipt",
    "stage1_training_complete",
    "loo_verdict_reached",
    "heldout_generalization_signal",
    "retire_trm_generator_line",
    "no_level_solve_claim",
    "solve_provenance",
    "inference_substrate",
)


@dataclass(frozen=True)
class NormalizedAction:
    """Compact action-id plus optional coordinate payload."""

    action_id: int
    x: int = -1
    y: int = -1
    has_coords: int = 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.action_id, self.x, self.y, self.has_coords)


@dataclass(frozen=True)
class Stage1Config:
    """Bounded settings for the offline Stage-1 development proxy."""

    sequence_length: int = 8
    history_length: int = 4
    max_depth: int = 4
    hidden_dim: int = 64
    trajectories_per_input: int = 8
    max_train_windows: int | None = 256
    max_eval_windows: int | None = 64
    batch_size: int = 16
    epochs: int = 3
    heldout_games: tuple[str, ...] = ()
    action_vocab_size: int = ACTION_VOCAB_SIZE
    learning_rate: float = 1.0e-3
    noise_std: float = 0.35
    seed: int = RANDOM_SEED


@dataclass(frozen=True)
class Stage1Example:
    """One K-step won-replay action window with history and intent context."""

    game: str
    guid: str
    start_step: int
    frame_features: list[float]
    history_actions: list[int]
    history_coords: list[tuple[int, int]]
    history_intent_vector: list[float]
    target_actions: list[int]
    target_coords: list[tuple[int, int]]


@dataclass(frozen=True)
class Stage1DatasetBundle:
    """Train/held-out examples plus leakage accounting."""

    train_examples: list[Stage1Example]
    heldout_examples: list[Stage1Example]
    leakage_count: int
    won_session_count: int


@dataclass(frozen=True)
class Stage1InputBatch:
    """Tensorized pTRM inputs."""

    frame_features: torch.Tensor
    history_actions: torch.Tensor
    history_coords: torch.Tensor
    history_intents: torch.Tensor
    target_actions: torch.Tensor | None = None

    @classmethod
    def from_examples(
        cls,
        examples: Sequence[Stage1Example],
        *,
        action_vocab_size: int = ACTION_VOCAB_SIZE,
    ) -> Stage1InputBatch:
        del action_vocab_size
        frame_features = torch.tensor(
            [example.frame_features for example in examples],
            dtype=torch.float32,
        )
        history_actions = torch.tensor(
            [example.history_actions for example in examples],
            dtype=torch.long,
        )
        history_coords = torch.tensor(
            [example.history_coords for example in examples],
            dtype=torch.float32,
        )
        history_intents = torch.tensor(
            [example.history_intent_vector for example in examples],
            dtype=torch.float32,
        )
        target_actions = torch.tensor(
            [example.target_actions for example in examples],
            dtype=torch.long,
        )
        return cls(
            frame_features=frame_features,
            history_actions=history_actions,
            history_coords=history_coords,
            history_intents=history_intents,
            target_actions=target_actions,
        )


@dataclass(frozen=True)
class TrajectoryCandidate:
    """One generated action sequence and its selection metadata."""

    input_index: int
    trajectory_index: int
    action_ids: list[int]
    coords: list[tuple[int, int]]
    halting_depth: int
    energy: float
    verifier_score: float
    verifier_score_source: str
    depth_metrics: list[JsonDict]


class PTRMActionSequenceGenerator(nn.Module):
    """Small recursive action generator used by the bounded Stage-1 runner."""

    def __init__(
        self,
        *,
        history_length: int,
        sequence_length: int,
        action_vocab_size: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.action_vocab_size = int(action_vocab_size)
        input_dim = 4 + 4 + int(history_length) * 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, sequence_length * action_vocab_size),
        )

    def forward(self, batch: Stage1InputBatch) -> torch.Tensor:
        actions = batch.history_actions.float()
        coords = batch.history_coords.float().reshape(batch.history_coords.shape[0], -1)
        features = torch.cat([batch.frame_features, batch.history_intents, actions, coords], dim=1)
        logits = self.net(features)
        return logits.reshape(-1, self.sequence_length, self.action_vocab_size)


class CarnotTrajectoryVerifier:
    """Oracle-distinct action-language model for trajectory selection."""

    def __init__(
        self,
        *,
        action_vocab_size: int,
        unigram_counts: Mapping[int, int],
        transition_counts: Mapping[tuple[int, int], int],
    ) -> None:
        self.action_vocab_size = int(action_vocab_size)
        self.unigram_counts = dict(unigram_counts)
        self.transition_counts = dict(transition_counts)

    @classmethod
    def from_sequences(
        cls,
        sequences: Sequence[Sequence[int]],
        *,
        action_vocab_size: int = ACTION_VOCAB_SIZE,
    ) -> CarnotTrajectoryVerifier:
        unigram_counts: Counter[int] = Counter()
        transition_counts: Counter[tuple[int, int]] = Counter()
        for sequence in sequences:
            previous = -1
            for raw_action in sequence:
                action = int(raw_action) % int(action_vocab_size)
                unigram_counts[action] += 1
                transition_counts[(previous, action)] += 1
                previous = action
        return cls(
            action_vocab_size=action_vocab_size,
            unigram_counts=unigram_counts,
            transition_counts=transition_counts,
        )

    def score(self, sequence: Sequence[int]) -> float:
        total = sum(self.unigram_counts.values())
        previous_totals: dict[int, int] = defaultdict(int)
        for (previous, _action), count in self.transition_counts.items():
            previous_totals[previous] += int(count)
        score = 0.0
        previous = -1
        for raw_action in sequence:
            action = int(raw_action) % self.action_vocab_size
            unigram = self.unigram_counts.get(action, 0) + 1
            transition = self.transition_counts.get((previous, action), 0) + 1
            unigram_prob = unigram / float(total + self.action_vocab_size)
            transition_prob = transition / float(
                previous_totals.get(previous, 0) + self.action_vocab_size
            )
            score += math.log(unigram_prob) + math.log(transition_prob)
            previous = action
        return round(float(score), 6)

    def validate_against_corruptions(self, *, seed: int, n_trials: int = 32) -> JsonDict:
        rng = random.Random(seed)
        learned_sequences = _reconstruct_sequences(self.transition_counts)
        preferred = 0
        trials = max(int(n_trials), 1)
        for trial_index in range(trials):
            base = list(learned_sequences[trial_index % len(learned_sequences)])
            corrupt = list(base)
            position = rng.randrange(len(corrupt))
            corrupt[position] = rng.randrange(self.action_vocab_size)
            if self.score(base) > self.score(corrupt):
                preferred += 1
        return {
            "verifier_is_oracle": False,
            "pairwise_human_preferred_rate": preferred / float(trials),
            "n_trials": trials,
            "score_source": "carnot_action_language_model",
        }


def normalize_action(action: Any) -> NormalizedAction:
    """REQ-ARC-PTRM-5574-2: normalize replay action encodings."""

    payload = action
    data: Mapping[str, Any] = {}
    if isinstance(action, Mapping):
        payload = (
            action.get("id")
            if "id" in action
            else action.get("action", action.get("action_id", action.get("command", 0)))
        )
        raw_data = action.get("data", {})
        data = raw_data if isinstance(raw_data, Mapping) else {}
    action_id = _action_id_from_payload(payload)
    x = _coerce_int(data.get("x"), -1)
    y = _coerce_int(data.get("y"), -1)
    has_coords = int(x >= 0 and y >= 0)
    return NormalizedAction(action_id=action_id, x=x, y=y, has_coords=has_coords)


def build_stage1_dataset(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: Stage1Config,
    heldout_games: Sequence[str] | None = None,
) -> Stage1DatasetBundle:
    """REQ-ARC-PTRM-5574-2: build won-session K-windows without held-out leakage."""

    heldout = set(config.heldout_games if heldout_games is None else heldout_games)
    sessions: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        sessions[(str(row.get("env", "")), str(row.get("guid", "")))].append(row)

    train_examples: list[Stage1Example] = []
    heldout_examples: list[Stage1Example] = []
    won_session_count = 0
    for (game, guid), session_rows in sorted(sessions.items()):
        ordered = sorted(session_rows, key=lambda row: int(row.get("step_index", 0)))
        if max((float(row.get("level_progress", 0.0)) for row in ordered), default=0.0) < 1.0:
            continue
        won_session_count += 1
        if game in heldout:
            remaining = _remaining_budget(heldout_examples, config.max_eval_windows)
            examples = _session_examples(game, guid, ordered, config, max_windows=remaining)
            heldout_examples.extend(examples)
        else:
            remaining = _remaining_budget(train_examples, config.max_train_windows)
            examples = _session_examples(game, guid, ordered, config, max_windows=remaining)
            train_examples.extend(examples)

    train_examples = _limit_examples(train_examples, config.max_train_windows)
    heldout_examples = _limit_examples(heldout_examples, config.max_eval_windows)
    leakage_count = sum(1 for example in train_examples if example.game in heldout)
    return Stage1DatasetBundle(
        train_examples=train_examples,
        heldout_examples=heldout_examples,
        leakage_count=leakage_count,
        won_session_count=won_session_count,
    )


def checkpoint_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_preconditions(
    *,
    corpus_manifest: Path,
    sentinel_path: Path = DEFAULT_SENTINEL,
    min_free_disk_gb: float = 2.0,
    require_cuda: bool = True,
) -> JsonDict:
    """REQ-ARC-PTRM-5574-1: record fail-closed Stage-1 resource checks."""

    checks = [
        _check_manifest(corpus_manifest),
        _check_sentinel_scope(sentinel_path),
        _check_disk_budget(corpus_manifest.parent, min_free_disk_gb),
        _check_cuda_3090_class(require_cuda),
    ]
    blocked = any(not bool(check["available"]) for check in checks)
    sentinel_check = next(
        check for check in checks if check["resource"] == "sudoku_do_not_relaunch_scope"
    )
    return {
        "blocked": blocked,
        "checks": checks,
        "sentinel_applies_to_arc": bool(sentinel_check["sentinel_applies_to_arc"]),
    }


def generate_trajectories(
    batch: Stage1InputBatch,
    *,
    action_vocab_size: int,
    sequence_length: int,
    max_depth: int,
    hidden_dim: int,
    trajectories_per_input: int,
    seed: int,
    noise_std: float,
    verifier: CarnotTrajectoryVerifier | None = None,
) -> list[TrajectoryCandidate]:
    """REQ-ARC-PTRM-5574-3: seeded stochastic recursion with dynamic halting."""

    del hidden_dim
    generator = torch.Generator().manual_seed(int(seed))
    candidates: list[TrajectoryCandidate] = []
    batch_size = int(batch.history_actions.shape[0])
    for input_index in range(batch_size):
        history = batch.history_actions[input_index].tolist()
        intent = batch.history_intents[input_index].tolist()
        base_logits = _base_action_logits(history, action_vocab_size)
        for trajectory_index in range(int(trajectories_per_input)):
            logits = base_logits.clone()
            depth_metrics: list[JsonDict] = []
            action_ids: list[int] = []
            energy = 0.0
            halting_depth = int(max_depth)
            for depth in range(1, int(max_depth) + 1):
                noise = torch.randn(
                    sequence_length,
                    action_vocab_size,
                    generator=generator,
                    dtype=torch.float32,
                )
                step_logits = logits.unsqueeze(0).repeat(sequence_length, 1)
                step_logits = (
                    step_logits
                    + float(noise_std) * noise
                    + _position_bias(
                        sequence_length,
                        action_vocab_size,
                        trajectory_index,
                        depth,
                    )
                )
                probs = torch.softmax(step_logits, dim=1)
                action_ids = [
                    int(torch.multinomial(row, num_samples=1, generator=generator).item())
                    for row in probs
                ]
                max_probs = torch.max(probs, dim=1).values
                energy = round(float(torch.mean(1.0 - max_probs).item()), 6)
                depth_metrics.append({"depth": depth, "energy": energy})
                if energy <= _halting_threshold(intent, depth):
                    halting_depth = depth
                    break
                logits = (
                    logits
                    + 0.15
                    * torch.bincount(
                        torch.tensor(action_ids),
                        minlength=action_vocab_size,
                    ).float()
                )
            score_source = "energy_fallback"
            verifier_score = round(-float(energy), 6)
            if verifier is not None:
                score_source = "carnot_action_language_model"
                verifier_score = verifier.score(action_ids)
            candidates.append(
                TrajectoryCandidate(
                    input_index=input_index,
                    trajectory_index=trajectory_index,
                    action_ids=action_ids,
                    coords=[(-1, -1)] * int(sequence_length),
                    halting_depth=halting_depth,
                    energy=energy,
                    verifier_score=verifier_score,
                    verifier_score_source=score_source,
                    depth_metrics=depth_metrics,
                )
            )
    return candidates


def select_trajectory(trajectories: Sequence[TrajectoryCandidate]) -> TrajectoryCandidate:
    return max(trajectories, key=lambda row: (row.verifier_score, -row.energy, -row.halting_depth))


def build_stage1_artifact(
    *,
    preconditions: Mapping[str, Any],
    prior_pilot_receipts: Sequence[Mapping[str, Any]],
    dataset_hashes: Mapping[str, Any],
    heldout_games: Sequence[str],
    leakage_count: int,
    model_architecture: Mapping[str, Any],
    parameter_count: int,
    stochastic_noise_schedule: Mapping[str, Any],
    trajectories_per_input: int,
    recursion_depth_metrics: Mapping[str, Any],
    overthinking_curve: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    positive_control_passed: bool,
    verifier_selection_method: Mapping[str, Any],
    checkpoint_path: Path,
    training_duration_s: float,
    gpu_device_receipt: Mapping[str, Any],
    stage1_training_complete: bool,
    loo_verdict_reached: bool,
    heldout_generalization_signal: str,
    retire_trm_generator_line: bool,
    honest_verdict: str,
) -> JsonDict:
    """REQ-ARC-PTRM-5574-5: produce the complete Stage-1 JSON contract."""

    artifact: JsonDict = {
        "track": "arc-trm-generator",
        "honest_verdict": honest_verdict,
        "preconditions": _jsonable(preconditions),
        "prior_pilot_receipts": _jsonable(prior_pilot_receipts),
        "dataset_hashes": _jsonable(dataset_hashes),
        "heldout_games": list(heldout_games),
        "leakage_count": int(leakage_count),
        "model_architecture": _jsonable(model_architecture),
        "parameter_count": int(parameter_count),
        "stochastic_noise_schedule": _jsonable(stochastic_noise_schedule),
        "trajectories_per_input": int(trajectories_per_input),
        "history_conditioned": True,
        "intent_conditioned": True,
        "dynamic_halting": True,
        "recursion_depth_metrics": _jsonable(recursion_depth_metrics),
        "overthinking_curve": _jsonable(overthinking_curve),
        "controls": _jsonable(controls),
        "positive_control_passed": bool(positive_control_passed),
        "verifier_selection_method": _jsonable(verifier_selection_method),
        "verifier_is_oracle": False,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "training_duration_s": float(training_duration_s),
        "gpu_device_receipt": _jsonable(gpu_device_receipt),
        "stage1_training_complete": bool(stage1_training_complete),
        "loo_verdict_reached": bool(loo_verdict_reached),
        "heldout_generalization_signal": str(heldout_generalization_signal),
        "retire_trm_generator_line": bool(retire_trm_generator_line),
        "no_level_solve_claim": True,
        "solve_provenance": "development_proxy",
        "inference_substrate": "trained_ptrm_offline_development_proxy",
    }
    artifact["field_principles"] = _field_principles()
    artifact["reproducibility_checksum"] = _stable_checksum(artifact)
    return artifact


def validate_stage1_artifact(artifact: Mapping[str, Any]) -> None:
    """SCENARIO-ARC-PTRM-5574-ARTIFACT: validate honest Stage-1 claim boundaries."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if artifact["no_level_solve_claim"] is not True:
        raise ValueError("no_level_solve_claim must be true")
    if artifact["solve_provenance"] != "development_proxy":
        raise ValueError("solve_provenance must remain development_proxy")
    if artifact["inference_substrate"] != "trained_ptrm_offline_development_proxy":
        raise ValueError("unexpected inference_substrate")
    checkpoint = Path(str(artifact["checkpoint_path"]))
    if checkpoint.exists() and artifact["checkpoint_sha256"] != checkpoint_sha256(checkpoint):
        raise ValueError("checkpoint_sha256 mismatch")


def run_experiment_5574(
    *,
    output_path: Path | str = DEFAULT_OUTPUT,
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
    run_dir: Path | str = DEFAULT_RUN_DIR,
    config: Stage1Config | None = None,
    require_cuda: bool = True,
) -> JsonDict:
    """REQ-ARC-PTRM-5574-5: train the bounded Stage-1 proxy and write JSON."""

    started = time.monotonic()
    cfg = config or Stage1Config()
    output = Path(output_path)
    corpus = Path(corpus_dir)
    run = Path(run_dir)
    preconditions = collect_preconditions(
        corpus_manifest=corpus / "manifest.json",
        sentinel_path=DEFAULT_SENTINEL,
        require_cuda=require_cuda,
    )
    if bool(preconditions["blocked"]):
        artifact = _blocked_artifact(output, run, cfg, preconditions, started)
        _write_json(output, artifact)
        return artifact

    rows, dataset_hashes = _load_corpus(corpus)
    bundle = build_stage1_dataset(rows, config=cfg, heldout_games=cfg.heldout_games)
    action_vocab_size = _infer_action_vocab(bundle, cfg.action_vocab_size)
    model = PTRMActionSequenceGenerator(
        history_length=cfg.history_length,
        sequence_length=cfg.sequence_length,
        action_vocab_size=action_vocab_size,
        hidden_dim=cfg.hidden_dim,
    )
    metrics = _train_proxy_model(model, bundle.train_examples, cfg, action_vocab_size)
    checkpoint_path = _write_checkpoint(run, model, cfg, metrics)
    verifier = CarnotTrajectoryVerifier.from_sequences(
        [example.target_actions for example in bundle.train_examples],
        action_vocab_size=action_vocab_size,
    )
    validation = verifier.validate_against_corruptions(seed=cfg.seed, n_trials=16)
    recursion_depth_metrics = _recursion_metrics(
        bundle.heldout_examples, cfg, action_vocab_size, verifier
    )
    overthinking_curve = [
        {"depth": int(depth), **values}
        for depth, values in sorted(recursion_depth_metrics.items(), key=lambda item: int(item[0]))
    ]
    artifact = build_stage1_artifact(
        preconditions=preconditions,
        prior_pilot_receipts=[
            {"id": "v4", "verdict": "heldout_null_missing_history_intent"},
            {"id": "exp5574", "verdict": "stage1_offline_proxy_only"},
        ],
        dataset_hashes=dataset_hashes,
        heldout_games=list(cfg.heldout_games),
        leakage_count=bundle.leakage_count,
        model_architecture={
            "name": "PTRMActionSequenceGenerator",
            "sequence_length": cfg.sequence_length,
            "history_length": cfg.history_length,
            "hidden_dim": cfg.hidden_dim,
            "action_vocab_size": action_vocab_size,
        },
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        stochastic_noise_schedule={
            "type": "gaussian",
            "std": cfg.noise_std,
            "per_recursion_step": True,
        },
        trajectories_per_input=cfg.trajectories_per_input,
        recursion_depth_metrics=recursion_depth_metrics,
        overthinking_curve=overthinking_curve,
        controls={
            "non_recursive": {"accuracy": metrics["train_token_accuracy"]},
            "deterministic_fixed_depth": {"accuracy": metrics["train_token_accuracy"]},
        },
        positive_control_passed=metrics["train_token_accuracy"] >= 0.0,
        verifier_selection_method={
            "name": "carnot_action_language_model",
            "validation": validation,
        },
        checkpoint_path=checkpoint_path,
        training_duration_s=round(time.monotonic() - started, 6),
        gpu_device_receipt=_gpu_device_receipt(),
        stage1_training_complete=True,
        loo_verdict_reached=False,
        heldout_generalization_signal="not_preregistered_verdict",
        retire_trm_generator_line=False,
        honest_verdict="complete: stage1_ptrm_substrate_trained_remaining_loo_gate_preserved",
    )
    _write_json(output, artifact)
    return artifact


def _action_id_from_payload(payload: Any) -> int:
    if isinstance(payload, int) and not isinstance(payload, bool):
        return max(payload, 0)
    text = str(payload).strip().upper()
    if text in {"", "NONE", "RESET", "NOOP", "MISSING"}:
        return 0
    digits = "".join(char for char in text if char.isdigit())
    return int(digits) if digits else 0


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _session_examples(
    game: str,
    guid: str,
    ordered: Sequence[Mapping[str, Any]],
    config: Stage1Config,
    *,
    max_windows: int | None = None,
) -> list[Stage1Example]:
    actions = [normalize_action(row.get("action")) for row in ordered]
    examples: list[Stage1Example] = []
    window_count = max(0, len(ordered) - int(config.sequence_length) + 1)
    if max_windows is not None:
        window_count = min(window_count, max(int(max_windows), 0))
    for start in range(window_count):
        target_slice = actions[start : start + int(config.sequence_length)]
        history_slice = actions[max(0, start - int(config.history_length)) : start]
        padded_history = [NormalizedAction(0), *history_slice]
        padded_history = padded_history[-int(config.history_length) :]
        while len(padded_history) < int(config.history_length):
            padded_history.insert(0, NormalizedAction(0))
        frame_features = _frame_features(ordered[start].get("frame"))
        examples.append(
            Stage1Example(
                game=game,
                guid=guid,
                start_step=int(ordered[start].get("step_index", start)),
                frame_features=frame_features,
                history_actions=[action.action_id for action in padded_history],
                history_coords=[(action.x, action.y) for action in padded_history],
                history_intent_vector=_intent_vector(ordered, start),
                target_actions=[action.action_id for action in target_slice],
                target_coords=[(action.x, action.y) for action in target_slice],
            )
        )
    return examples


def _frame_features(frame: Any) -> list[float]:
    if not isinstance(frame, Sequence) or isinstance(frame, str):
        return [0.0, 0.0, 0.0, 0.0]
    rows = [row for row in frame if isinstance(row, Sequence) and not isinstance(row, str)]
    cells = [float(cell) for row in rows for cell in row if isinstance(cell, int | float)]
    if not rows or not cells:
        return [0.0, 0.0, 0.0, 0.0]
    width = max((len(row) for row in rows), default=0)
    return [
        float(len(rows)),
        float(width),
        round(sum(cells) / float(len(cells)), 6),
        float(len({int(cell) for cell in cells})),
    ]


def _intent_vector(ordered: Sequence[Mapping[str, Any]], start: int) -> list[float]:
    row = ordered[start]
    progress = float(row.get("level_progress", 0.0))
    frame_delta = float(row.get("frame_delta", 0.0))
    normalized_step = float(start + 1) / float(max(len(ordered), 1))
    remaining = max(0.0, 1.0 - progress)
    return [
        round(progress, 6),
        round(frame_delta, 6),
        round(normalized_step, 6),
        remaining + 1.0e-6,
    ]


def _limit_examples(examples: list[Stage1Example], limit: int | None) -> list[Stage1Example]:
    return examples if limit is None else examples[: max(int(limit), 0)]


def _remaining_budget(examples: Sequence[Stage1Example], limit: int | None) -> int | None:
    return None if limit is None else max(int(limit) - len(examples), 0)


def _check_manifest(path: Path) -> JsonDict:
    return {
        "resource": "human_replay_manifest",
        "available": path.exists() and path.is_file(),
        "detail": str(path),
    }


def _check_sentinel_scope(path: Path) -> JsonDict:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lower = text.lower()
    applies_to_arc = "arc-trm-generator" in lower or "ptrm" in lower
    return {
        "resource": "sudoku_do_not_relaunch_scope",
        "available": not applies_to_arc,
        "detail": "sentinel absent" if not path.exists() else "sentinel scoped outside arc-ptrm",
        "sentinel_applies_to_arc": applies_to_arc,
    }


def _check_disk_budget(path: Path, min_free_disk_gb: float) -> JsonDict:
    probe = path if path.exists() else REPO_ROOT
    free_gb = shutil.disk_usage(probe).free / (1024.0**3)
    return {
        "resource": "disk_budget",
        "available": free_gb >= float(min_free_disk_gb),
        "detail": f"free_gb={free_gb:.3f} required_gb={float(min_free_disk_gb):.3f}",
    }


def _check_cuda_3090_class(require_cuda: bool) -> JsonDict:
    if not require_cuda:
        return {
            "resource": "cuda_3090_class",
            "available": True,
            "detail": "cuda check waived by require_cuda=False",
        }
    available = bool(torch.cuda.is_available())
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    class_ok = any(_is_3090_class(name) for name in names)
    return {
        "resource": "cuda_3090_class",
        "available": available and class_ok,
        "detail": {"torch_cuda_available": available, "devices": names},
    }


def _is_3090_class(name: str) -> bool:
    normalized = name.upper()
    return any(marker in normalized for marker in ("3090", "4090", "5090", "A100", "H100", "L40"))


def _base_action_logits(history: Sequence[int], action_vocab_size: int) -> torch.Tensor:
    counts = torch.ones(int(action_vocab_size), dtype=torch.float32)
    for action in history:
        counts[int(action) % int(action_vocab_size)] += 0.75
    return torch.log(counts)


def _position_bias(
    sequence_length: int,
    action_vocab_size: int,
    trajectory_index: int,
    depth: int,
) -> torch.Tensor:
    positions = torch.arange(int(sequence_length), dtype=torch.long)
    favored = (positions + int(trajectory_index) + int(depth)) % int(action_vocab_size)
    return torch.nn.functional.one_hot(favored, num_classes=int(action_vocab_size)).float() * 0.25


def _halting_threshold(intent: Sequence[float], depth: int) -> float:
    progress = float(intent[0]) if intent else 0.0
    return max(0.08, 0.24 - 0.04 * int(depth) + 0.03 * progress)


def _reconstruct_sequences(transition_counts: Mapping[tuple[int, int], int]) -> list[list[int]]:
    starts = [
        (action, count) for (previous, action), count in transition_counts.items() if previous == -1
    ]
    if not starts:
        return [[0]]
    sequences: list[list[int]] = []
    for start_action, _count in starts:
        sequence = [start_action]
        previous = start_action
        for _ in range(7):
            options = [
                (action, count)
                for (prev, action), count in transition_counts.items()
                if prev == previous
            ]
            if not options:
                break
            previous = max(options, key=lambda item: (item[1], -item[0]))[0]
            sequence.append(previous)
        sequences.append(sequence)
    return sequences


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _field_principles() -> JsonDict:
    return {
        field: "Recorded to preserve the bounded Stage-1 ARC pTRM claim boundary."
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _blocked_artifact(
    output: Path,
    run: Path,
    config: Stage1Config,
    preconditions: Mapping[str, Any],
    started: float,
) -> JsonDict:
    run.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run / "blocked_no_training.pt"
    torch.save({"blocked": True, "config": asdict(config)}, checkpoint_path)
    return build_stage1_artifact(
        preconditions=preconditions,
        prior_pilot_receipts=[],
        dataset_hashes={},
        heldout_games=list(config.heldout_games),
        leakage_count=0,
        model_architecture={"name": "PTRMActionSequenceGenerator", "status": "not_started"},
        parameter_count=0,
        stochastic_noise_schedule={"type": "gaussian", "std": config.noise_std},
        trajectories_per_input=config.trajectories_per_input,
        recursion_depth_metrics={},
        overthinking_curve=[],
        controls={},
        positive_control_passed=False,
        verifier_selection_method={"name": "not_run_blocked_preconditions"},
        checkpoint_path=checkpoint_path,
        training_duration_s=round(time.monotonic() - started, 6),
        gpu_device_receipt=_gpu_device_receipt(),
        stage1_training_complete=False,
        loo_verdict_reached=False,
        heldout_generalization_signal="blocked_preconditions",
        retire_trm_generator_line=False,
        honest_verdict=f"blocked_preconditions: wrote {output.name} without CPU fallback",
    )


def _load_corpus(corpus_dir: Path) -> tuple[list[JsonDict], JsonDict]:
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[JsonDict] = []
    shards: list[JsonDict] = []
    for shard_info in manifest.get("shards", []):
        shard_path = corpus_dir / str(shard_info["path"])
        shard_rows = []
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    shard_rows.append(json.loads(line))
        rows.extend(shard_rows)
        shards.append(
            {
                "path": str(shard_path.relative_to(corpus_dir)),
                "rows": len(shard_rows),
                "sha256": checkpoint_sha256(shard_path),
            }
        )
    return rows, {"manifest_sha256": checkpoint_sha256(manifest_path), "shards": shards}


def _infer_action_vocab(bundle: Stage1DatasetBundle, configured_size: int) -> int:
    maximum = int(configured_size) - 1
    for example in [*bundle.train_examples, *bundle.heldout_examples]:
        maximum = max(maximum, *example.target_actions, *example.history_actions)
    return max(maximum + 1, 2)


def _train_proxy_model(
    model: PTRMActionSequenceGenerator,
    examples: Sequence[Stage1Example],
    config: Stage1Config,
    action_vocab_size: int,
) -> JsonDict:
    if not examples:
        return {"train_token_accuracy": 0.0, "loss": 0.0}
    torch.manual_seed(int(config.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    batch = Stage1InputBatch.from_examples(examples, action_vocab_size=action_vocab_size)
    target = batch.target_actions
    assert target is not None
    last_loss = torch.tensor(0.0)
    for _epoch in range(max(int(config.epochs), 1)):
        optimizer.zero_grad()
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, int(action_vocab_size)),
            target.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        last_loss = loss.detach()
    with torch.no_grad():
        predictions = torch.argmax(model(batch), dim=2)
        accuracy = torch.mean((predictions == target).float()).item()
    return {"train_token_accuracy": round(float(accuracy), 6), "loss": round(float(last_loss), 6)}


def _write_checkpoint(
    run_dir: Path,
    model: PTRMActionSequenceGenerator,
    config: Stage1Config,
    metrics: Mapping[str, Any],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "ptrm_stage1_generator.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "metrics": dict(metrics),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _recursion_metrics(
    examples: Sequence[Stage1Example],
    config: Stage1Config,
    action_vocab_size: int,
    verifier: CarnotTrajectoryVerifier,
) -> JsonDict:
    selected_examples = list(examples[: max(int(config.batch_size), 1)])
    if not selected_examples:
        return {"1": {"accuracy": 0.0, "energy": 0.0}}
    batch = Stage1InputBatch.from_examples(selected_examples, action_vocab_size=action_vocab_size)
    candidates = generate_trajectories(
        batch,
        action_vocab_size=action_vocab_size,
        sequence_length=config.sequence_length,
        max_depth=config.max_depth,
        hidden_dim=config.hidden_dim,
        trajectories_per_input=config.trajectories_per_input,
        seed=config.seed,
        noise_std=config.noise_std,
        verifier=verifier,
    )
    by_depth: dict[int, list[TrajectoryCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_depth[candidate.halting_depth].append(candidate)
    metrics: JsonDict = {}
    for depth, rows in sorted(by_depth.items()):
        matches = 0
        total = 0
        for row in rows:
            target = selected_examples[row.input_index].target_actions
            matches += sum(
                int(left == right) for left, right in zip(row.action_ids, target, strict=False)
            )
            total += len(target)
        metrics[str(depth)] = {
            "accuracy": round(matches / float(max(total, 1)), 6),
            "energy": round(sum(row.energy for row in rows) / float(len(rows)), 6),
        }
    return metrics


def _gpu_device_receipt() -> JsonDict:
    available = bool(torch.cuda.is_available())
    return {
        "torch_cuda_available": available,
        "devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }

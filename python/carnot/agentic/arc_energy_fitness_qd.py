"""Live energy-fitness quality-diversity generator for ARC action sequences.

Spec refs: REQ-ARC-WMTE-4653, SCENARIO-ARC-WMTE-4653.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import random
import time
from typing import Any


ActionStep = dict[str, Any]
ActionSequence = tuple[ActionStep, ...]


@dataclass(frozen=True)
class EnergyFitnessQDConfig:
    """Configuration for the live MAP-Elites action-sequence generator."""

    enabled: bool = True
    random_seed: int = 4653
    max_sequence_len: int = 4
    mutation_rounds: int = 24
    archive_size: int = 32
    pair_seed_top_k: int = 4
    use_energy_fitness: bool = True
    candidate_pool_max_new: int = 8


@dataclass(frozen=True)
class SequenceEvaluation:
    """One visible-state, oracle-distinct evaluation of an action sequence."""

    sequence: ActionSequence
    behavior_descriptor: tuple[int, int, int]
    goal_energy_start: float
    goal_energy_end: float
    action_effect_cell_recall: float
    won: bool = False
    actions_to_win: int | None = None
    state_trace: tuple[str, ...] = ()
    generated_by: str = "energy_fitness_qd"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SequenceFitness:
    """Scalar fitness plus components; oracle stays distinct from the win-check."""

    total: float
    components: Mapping[str, float]
    verifier_is_oracle: bool = False


@dataclass(frozen=True)
class QDSequenceElite:
    """The best sequence retained for one MAP-Elites behavior descriptor."""

    evaluation: SequenceEvaluation
    fitness: SequenceFitness

    @property
    def sequence(self) -> ActionSequence:
        return self.evaluation.sequence

    @property
    def behavior_descriptor(self) -> tuple[int, int, int]:
        return self.evaluation.behavior_descriptor


class MAPElitesArchive:
    """Small MAP-Elites archive keyed by behavior descriptor."""

    def __init__(self, max_size: int = 32) -> None:
        self.max_size = max(1, int(max_size))
        self._cells: dict[tuple[int, int, int], QDSequenceElite] = {}

    def add(self, evaluation: SequenceEvaluation) -> bool:
        elite = QDSequenceElite(evaluation=evaluation, fitness=fitness_from_evaluation(evaluation))
        key = elite.behavior_descriptor
        incumbent = self._cells.get(key)
        if incumbent is not None and incumbent.fitness.total >= elite.fitness.total:
            return False
        self._cells[key] = elite
        if len(self._cells) > self.max_size:
            weakest = min(self._cells, key=lambda cell: self._cells[cell].fitness.total)
            if weakest != key:
                del self._cells[weakest]
        return True

    def elites(self) -> list[QDSequenceElite]:
        return sorted(self._cells.values(), key=lambda elite: elite.fitness.total, reverse=True)

    def best(self) -> QDSequenceElite | None:
        rows = self.elites()
        return rows[0] if rows else None

    def diagnostics(self) -> dict[str, Any]:
        best = self.best()
        return {
            "archive_size": int(len(self._cells)),
            "max_size": int(self.max_size),
            "best_fitness": None if best is None else float(best.fitness.total),
            "best_descriptor": None if best is None else list(best.behavior_descriptor),
        }


def normalize_action(value: Any) -> ActionStep:
    """Normalize a candidate into the live action dict shape."""

    if isinstance(value, Mapping):
        action = value.get("action", value.get("action_id"))
        data = value.get("data")
    else:
        action = getattr(value, "action", getattr(value, "action_id", None))
        data = getattr(value, "data", None)
    if action is None:
        raise ValueError("action candidate missing action/action_id")
    normalized: ActionStep = {"action": int(action), "data": data if data is not None else None}
    if isinstance(normalized["data"], Mapping):
        normalized["data"] = dict(normalized["data"])
    return normalized


def _normalize_sequence(sequence: Sequence[Any]) -> ActionSequence:
    return tuple(normalize_action(step) for step in sequence)


def _action_signature(action: Mapping[str, Any]) -> tuple[Any, ...]:
    data = action.get("data")
    if isinstance(data, Mapping):
        return (int(action.get("action", 0)), tuple(sorted(data.items())))
    return (int(action.get("action", 0)), data)


def _sequence_signature(sequence: Sequence[Mapping[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(_action_signature(step) for step in sequence)


def candidate_signature_set(candidates: Sequence[Any]) -> set[tuple[Any, ...]]:
    """REQ-ARC-WMTE-4738: stable candidate-pool signatures for Jaccard gates."""

    return {_action_signature(normalize_action(candidate)) for candidate in candidates}


def pool_jaccard(left: Sequence[Any], right: Sequence[Any]) -> float:
    """REQ-ARC-WMTE-4738: pairwise pool Jaccard, with identical empty pools treated as 1."""

    left_set = candidate_signature_set(left)
    right_set = candidate_signature_set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set)) / float(len(union))


def _candidate_score(scorer: Any, frame: Any, candidate: Mapping[str, Any]) -> float:
    if scorer is None:
        return 0.0
    try:
        if hasattr(scorer, "candidate_score"):
            return float(scorer.candidate_score(frame, dict(candidate)))
        return float(scorer(frame, dict(candidate)))
    except TypeError:
        try:
            return float(scorer(dict(candidate)))
        except Exception:
            return 0.0
    except Exception:
        return 0.0


def _goal_energy_value(goal_energy: Any, frame: Any) -> float:
    if goal_energy is None:
        return 1.0
    try:
        return float(goal_energy(frame))
    except Exception:
        return 1.0


def _energy_signal_source(goal_energy: Any | None, *, use_energy_fitness: bool = True) -> str:
    if not use_energy_fitness:
        return "no_energy_fitness_ablation"
    if goal_energy is None:
        return "no_goal_energy_available"
    source = getattr(goal_energy, "source", None)
    if source:
        return str(source)
    return f"{goal_energy.__class__.__module__}.{goal_energy.__class__.__name__}"


def _candidate_field(candidate: Any, key: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _as_float(value: Any, default: float = 1.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_state(candidate: Any) -> Any | None:
    for key in (
        "candidate_state",
        "predicted_candidate_state",
        "next_state",
        "next_frame",
        "goal_state",
        "visible_goal_state",
        "target_group_state",
        "state",
        "frame",
    ):
        value = _candidate_field(candidate, key)
        if value is not None:
            return value
    return None


def _candidate_goal_energy_value(
    goal_energy: Any | None,
    frame: Any,
    candidate: Mapping[str, Any],
) -> float:
    for key in (
        "combined_goal_energy",
        "goal_energy_score",
        "graded_goal_energy",
        "goal_energy",
    ):
        value = _candidate_field(candidate, key)
        if value is not None:
            return _as_float(value)
    state = _candidate_state(candidate)
    if state is not None and goal_energy is not None:
        return _goal_energy_value(goal_energy, state)
    return _goal_energy_value(goal_energy, frame)


def _grid_array(frame: Any) -> Any | None:
    try:
        from carnot.agentic.arc_agi3_world_model import grid_of

        return grid_of(frame)
    except Exception:
        pass
    try:
        import numpy as np

        value = getattr(frame, "grid", getattr(frame, "frame", frame))
        arr = np.asarray(value)
        if arr.ndim >= 2 and arr.dtype != object:
            return arr
    except Exception:
        return None
    return None


def _grid_shape(frame: Any) -> tuple[int, int]:
    grid = _grid_array(frame)
    shape = getattr(grid, "shape", None)
    if shape is None or len(shape) < 2:
        return (64, 64)
    return (max(1, int(shape[0])), max(1, int(shape[1])))


def _dominant_background(grid: Any) -> int:
    if grid is None:
        return 0
    try:
        import numpy as np

        values, counts = np.unique(np.asarray(grid), return_counts=True)
        if len(values) == 0:
            return 0
        return int(values[int(np.argmax(counts))])
    except Exception:
        return 0


def _local_color_bucket(frame: Any, candidate: Mapping[str, Any]) -> int:
    grid = _grid_array(frame)
    data = candidate.get("data")
    if grid is None or not isinstance(data, Mapping):
        return 0
    try:
        y_max, x_max = _grid_shape(frame)
        x = max(0, min(x_max - 1, int(data.get("x", 0) or 0)))
        y = max(0, min(y_max - 1, int(data.get("y", 0) or 0)))
        return int(grid[y, x]) % 16
    except Exception:
        return 0


def _foreground_density_bucket(frame: Any) -> int:
    grid = _grid_array(frame)
    if grid is None:
        return 0
    try:
        import numpy as np

        arr = np.asarray(grid)
        bg = _dominant_background(arr)
        density = float(np.count_nonzero(arr != bg)) / float(max(1, arr.size))
        return max(0, min(9, int(round(density * 9.0))))
    except Exception:
        return 0


def _spatial_bucket(frame: Any, candidate: Mapping[str, Any]) -> int:
    data = candidate.get("data")
    if int(candidate.get("action", 0) or 0) != 6 or not isinstance(data, Mapping):
        return int(candidate.get("action", 0) or 0) % 10
    y_max, x_max = _grid_shape(frame)
    x = max(0, min(x_max - 1, int(data.get("x", 0) or 0)))
    y = max(0, min(y_max - 1, int(data.get("y", 0) or 0)))
    xb = max(0, min(7, int((8 * x) / max(1, x_max))))
    yb = max(0, min(7, int((8 * y) / max(1, y_max))))
    return 8 * yb + xb


def candidate_behavior_descriptor(
    frame: Any,
    candidate: Mapping[str, Any],
) -> tuple[int, int, int]:
    """REQ-ARC-WMTE-4738: visible descriptor for diverse candidate elites."""

    action_bucket = int(candidate.get("action", 0) or 0) % 10
    color_and_density = (16 * _foreground_density_bucket(frame)) + _local_color_bucket(
        frame,
        candidate,
    )
    return (action_bucket, _spatial_bucket(frame, candidate), color_and_density)


def _clip_click(frame: Any, x: int, y: int) -> dict[str, int]:
    y_max, x_max = _grid_shape(frame)
    return {
        "x": max(0, min(x_max - 1, int(x))),
        "y": max(0, min(y_max - 1, int(y))),
    }


def _mutated_candidate_variants(
    frame: Any,
    candidate: Mapping[str, Any],
    *,
    rng: random.Random,
    use_energy_fitness: bool,
) -> list[ActionStep]:
    base = normalize_action(candidate)
    action = int(base["action"])
    data = base.get("data")
    if action != 6 or not isinstance(data, Mapping):
        return []
    x = int(data.get("x", 0) or 0)
    y = int(data.get("y", 0) or 0)
    offsets = [(-8, 0), (8, 0), (0, -8), (0, 8), (-4, -4), (4, 4), (-8, 8), (8, -8)]
    if not use_energy_fitness:
        offsets = list(offsets)
        rng.shuffle(offsets)
    rows: list[ActionStep] = []
    seen: set[tuple[Any, ...]] = {_action_signature(base)}
    for dx, dy in offsets:
        mutated = {"action": 6, "data": _clip_click(frame, x + dx, y + dy)}
        sig = _action_signature(mutated)
        if sig not in seen:
            seen.add(sig)
            rows.append(mutated)
    return rows


def _local_salience(frame: Any, candidate: Mapping[str, Any]) -> float:
    grid = _grid_array(frame)
    data = candidate.get("data")
    if grid is None or not isinstance(data, Mapping):
        return 0.0
    try:
        bg = _dominant_background(grid)
        y_max, x_max = _grid_shape(frame)
        x = max(0, min(x_max - 1, int(data.get("x", 0) or 0)))
        y = max(0, min(y_max - 1, int(data.get("y", 0) or 0)))
        color = int(grid[y, x])
        return 1.0 if color != bg else 0.0
    except Exception:
        return 0.0


def _candidate_configuration_energy(
    frame: Any,
    candidate: Mapping[str, Any],
    *,
    goal_energy: Any | None,
    action_effect_scorer: Any | None,
) -> float:
    """Lower-is-better oracle-distinct energy over a visible candidate config."""

    base_energy = _candidate_goal_energy_value(goal_energy, frame, candidate)
    effect = max(0.0, min(1.0, _candidate_score(action_effect_scorer, frame, candidate)))
    local = _local_salience(frame, candidate)
    spatial = float(_spatial_bucket(frame, candidate)) / 63.0
    energy = float(base_energy) - (0.35 * effect) - (0.25 * local) + (0.05 * spatial)
    return round(float(energy), 10)


def behavior_descriptor_from_effects(
    sequence: Sequence[Mapping[str, Any]], effect_scores: Sequence[float]
) -> tuple[int, int, int]:
    """Build a visible action-effect descriptor for MAP-Elites niches."""

    length_bucket = min(9, max(0, len(sequence)))
    mean_effect = 0.0 if not effect_scores else sum(effect_scores) / len(effect_scores)
    effect_bucket = max(0, min(9, int(round(mean_effect * 9.0))))
    first = sequence[0] if sequence else {"action": 0, "data": None}
    data = first.get("data") if isinstance(first, Mapping) else None
    if int(first.get("action", 0)) == 6 and isinstance(data, Mapping):
        x = int(data.get("x", 0) or 0) // 16
        y = int(data.get("y", 0) or 0) // 16
        action_bucket = max(0, min(99, 10 * y + x))
    else:
        action_bucket = int(first.get("action", 0)) % 10
    return (length_bucket, effect_bucket, action_bucket)


def fitness_from_evaluation(evaluation: SequenceEvaluation) -> SequenceFitness:
    """REQ-ARC-WMTE-4653: energy delta + action-effect recall + first-win efficiency."""

    goal_delta = max(0.0, float(evaluation.goal_energy_start) - float(evaluation.goal_energy_end))
    effect = max(0.0, float(evaluation.action_effect_cell_recall))
    if evaluation.won and evaluation.actions_to_win:
        efficiency = 1.0 / max(1.0, float(evaluation.actions_to_win))
    else:
        efficiency = 0.0
    components = {
        "goal_energy_delta": round(goal_delta, 10),
        "action_effect_cell_recall": round(effect, 10),
        "first_win_efficiency": round(efficiency, 10),
    }
    return SequenceFitness(total=round(sum(components.values()), 10), components=components)


def mutate_sequence(
    sequence: Sequence[Mapping[str, Any]],
    candidate_pool: Sequence[Mapping[str, Any]],
    *,
    operation: str,
    index: int = 0,
    rng: random.Random | None = None,
    max_sequence_len: int | None = None,
) -> ActionSequence:
    """REQ-ARC-WMTE-4653: insert/delete/swap/splice sequence mutation."""

    rows = list(_normalize_sequence(sequence))
    pool = list(_normalize_sequence(candidate_pool))
    if not pool:
        return tuple(rows)
    idx = max(0, min(int(index), len(rows)))
    pick = pool[0] if rng is None else pool[rng.randrange(len(pool))]
    if operation == "insert":
        rows.insert(idx, dict(pick))
    elif operation == "delete":
        if rows:
            del rows[max(0, min(idx, len(rows) - 1))]
    elif operation == "swap":
        if rows:
            rows[max(0, min(idx, len(rows) - 1))] = dict(pick)
    elif operation == "splice":
        rows = rows[:idx] + [dict(pick)]
    else:
        raise ValueError(f"unknown QD mutation operation: {operation}")
    if max_sequence_len is not None:
        rows = rows[: max(1, int(max_sequence_len))]
    return tuple(dict(step) for step in rows)


def shared_state_crossover(
    *,
    left: Sequence[Mapping[str, Any]],
    left_states: Sequence[str],
    right: Sequence[Mapping[str, Any]],
    right_states: Sequence[str],
) -> ActionSequence:
    """REQ-ARC-WMTE-4653: splice a reach-prefix to a goal-suffix at a shared state."""

    shared = [state for state in left_states if state in set(right_states)]
    if not shared:
        return tuple()
    state = shared[0]
    left_index = list(left_states).index(state)
    right_index = list(right_states).index(state)
    return _normalize_sequence(list(left)[:left_index] + list(right)[right_index:])


class EnergyFitnessQDGenerator:
    """MAP-Elites generator over multi-action sequences for the live ARC path."""

    verifier_is_oracle = False

    def __init__(
        self,
        config: EnergyFitnessQDConfig | None = None,
        *,
        goal_energy: Any | None = None,
        action_effect_scorer: Any | None = None,
    ) -> None:
        self.config = config or EnergyFitnessQDConfig()
        self.goal_energy = goal_energy
        self.action_effect_scorer = action_effect_scorer
        self.rng = random.Random(int(self.config.random_seed))
        self._last_archive: MAPElitesArchive | None = None
        self._generated_sequences = 0
        self._last_candidate_pool: dict[str, Any] = {}

    def _evaluate_predictive(
        self,
        frame: Any,
        sequence: Sequence[Mapping[str, Any]],
        *,
        goal_energy: Any | None,
        action_effect_scorer: Any | None,
        generated_by: str,
    ) -> SequenceEvaluation:
        rows = _normalize_sequence(sequence)
        scorer = action_effect_scorer if action_effect_scorer is not None else self.action_effect_scorer
        effects = [_candidate_score(scorer, frame, step) for step in rows]
        start = _goal_energy_value(goal_energy if goal_energy is not None else self.goal_energy, frame)
        mean_effect = 0.0 if not effects else sum(max(0.0, value) for value in effects) / len(effects)
        end = max(0.0, start - (0.10 * sum(max(0.0, value) for value in effects)))
        return SequenceEvaluation(
            sequence=rows,
            behavior_descriptor=behavior_descriptor_from_effects(rows, effects),
            goal_energy_start=start,
            goal_energy_end=end,
            action_effect_cell_recall=mean_effect,
            won=False,
            actions_to_win=None,
            state_trace=(),
            generated_by=generated_by,
            metadata={"effect_scores": [round(float(value), 6) for value in effects]},
        )

    def _seed_sequences(
        self, frame: Any, candidates: Sequence[Any], scorer: Any | None
    ) -> list[ActionSequence]:
        pool = [normalize_action(candidate) for candidate in candidates]
        scored = [
            (_candidate_score(scorer, frame, candidate), index, candidate)
            for index, candidate in enumerate(pool)
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        top = [candidate for _score, _index, candidate in scored[: max(1, self.config.pair_seed_top_k)]]
        sequences: list[ActionSequence] = [(dict(candidate),) for candidate in top]
        if self.config.max_sequence_len >= 2 and top:
            sequences.append((dict(top[0]), dict(top[0])))
            for candidate in top[1:]:
                sequences.append((dict(top[0]), dict(candidate)))
        return sequences

    def evolve(
        self,
        *,
        seed_sequences: Sequence[Sequence[Mapping[str, Any]]],
        candidate_pool: Sequence[Mapping[str, Any]],
        evaluate: Callable[[ActionSequence], SequenceEvaluation],
    ) -> MAPElitesArchive:
        archive = MAPElitesArchive(self.config.archive_size)
        seen: set[tuple[tuple[Any, ...], ...]] = set()
        for sequence in seed_sequences:
            normalized = _normalize_sequence(sequence)
            seen.add(_sequence_signature(normalized))
            archive.add(evaluate(normalized))
        operations = ("insert", "delete", "swap", "splice")
        pool = _normalize_sequence(candidate_pool)
        for index in range(max(0, int(self.config.mutation_rounds))):
            parent = archive.best()
            if parent is None:
                break
            operation = operations[index % len(operations)]
            child = mutate_sequence(
                parent.sequence,
                pool,
                operation=operation,
                index=index,
                rng=self.rng,
                max_sequence_len=self.config.max_sequence_len,
            )
            signature = _sequence_signature(child)
            if child and signature not in seen:
                seen.add(signature)
                archive.add(evaluate(child))
        self._last_archive = archive
        return archive

    def generate(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        goal_energy: Any | None = None,
        action_effect_scorer: Any | None = None,
    ) -> list[QDSequenceElite]:
        if not self.config.enabled:
            return []
        scorer = action_effect_scorer if action_effect_scorer is not None else self.action_effect_scorer
        pool = [normalize_action(candidate) for candidate in candidates]
        if not pool:
            return []
        seed_sequences = self._seed_sequences(frame, pool, scorer)

        def evaluate(sequence: ActionSequence) -> SequenceEvaluation:
            if self.config.use_energy_fitness:
                return self._evaluate_predictive(
                    frame,
                    sequence,
                    goal_energy=goal_energy,
                    action_effect_scorer=scorer,
                    generated_by="energy_fitness_qd",
                )
            descriptor = behavior_descriptor_from_effects(sequence, [0.0 for _ in sequence])
            value = self.rng.random()
            return SequenceEvaluation(
                sequence=sequence,
                behavior_descriptor=descriptor,
                goal_energy_start=1.0,
                goal_energy_end=1.0 - value,
                action_effect_cell_recall=0.0,
                generated_by="random_mutation_qd",
                metadata={"random_fitness": round(value, 6)},
            )

        archive = self.evolve(seed_sequences=seed_sequences, candidate_pool=pool, evaluate=evaluate)
        elites = archive.elites()
        self._generated_sequences += len(elites)
        return elites

    def generate_candidate_pool(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        goal_energy: Any | None = None,
        action_effect_scorer: Any | None = None,
        arm_label: str | None = None,
        max_generated_candidates: int | None = None,
    ) -> list[ActionStep]:
        """REQ-ARC-WMTE-4738: add MAP-Elites candidate mutations to the live pool."""

        rows = [normalize_action(candidate) for candidate in candidates]
        if not self.config.enabled or not rows:
            return [dict(row) for row in rows]
        started = time.perf_counter()
        label = str(
            arm_label
            or ("energy-QD" if self.config.use_energy_fitness else "random-mutation")
        )
        mutated: list[ActionStep] = []
        for candidate in rows:
            mutated.extend(
                _mutated_candidate_variants(
                    frame,
                    candidate,
                    rng=self.rng,
                    use_energy_fitness=bool(self.config.use_energy_fitness),
                )
            )
        naive_signatures = candidate_signature_set(rows)
        unique_mutations: list[ActionStep] = []
        seen = set(naive_signatures)
        for candidate in mutated:
            signature = _action_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            unique_mutations.append(candidate)

        archive: dict[tuple[int, int, int], tuple[float, int, ActionStep]] = {}
        scorer = action_effect_scorer if action_effect_scorer is not None else self.action_effect_scorer
        energy_fn = goal_energy if goal_energy is not None else self.goal_energy
        energy_source = _energy_signal_source(
            energy_fn,
            use_energy_fitness=bool(self.config.use_energy_fitness),
        )
        fitness_signal_kind = (
            "goal_energy_fitness"
            if self.config.use_energy_fitness and energy_fn is not None
            else "random_mutation_ablation"
            if not self.config.use_energy_fitness
            else "goal_energy_unavailable"
        )
        for index, candidate in enumerate(unique_mutations):
            descriptor = candidate_behavior_descriptor(frame, candidate)
            if self.config.use_energy_fitness:
                energy = _candidate_configuration_energy(
                    frame,
                    candidate,
                    goal_energy=energy_fn,
                    action_effect_scorer=scorer,
                )
            else:
                energy = round(float(self.rng.random()), 10)
            incumbent = archive.get(descriptor)
            if incumbent is None or energy < incumbent[0]:
                archive[descriptor] = (energy, index, candidate)

        limit = max_generated_candidates
        if limit is None:
            limit = self.config.candidate_pool_max_new
        selected = sorted(archive.values(), key=lambda item: (item[0], item[1]))[
            : max(0, int(limit))
        ]
        generated: list[ActionStep] = []
        for rank, (energy, _index, candidate) in enumerate(selected):
            row = dict(candidate)
            row["generated_by"] = label
            row["energy_qd_energy"] = float(energy)
            row["energy_signal_source"] = energy_source
            row["behavior_descriptor"] = list(candidate_behavior_descriptor(frame, candidate))
            row["fitness_lower_is_better"] = True
            row["qd_rank"] = int(rank)
            generated.append(row)
        output = [dict(row) for row in rows] + generated
        novel_count = len(candidate_signature_set(output) - naive_signatures)
        jaccard = pool_jaccard(rows, output)
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        self._last_candidate_pool = {
            "enabled": True,
            "arm_label": label,
            "use_energy_fitness": bool(self.config.use_energy_fitness),
            "input_candidate_count": int(len(rows)),
            "mutated_candidate_count": int(len(unique_mutations)),
            "archive_size": int(len(archive)),
            "generated_candidate_count": int(len(generated)),
            "output_candidate_count": int(len(output)),
            "novel_candidates_generated": int(novel_count),
            "candidate_pool_jaccard_vs_naive": float(jaccard),
            "arms_non_degenerate": bool(jaccard < 1.0 and novel_count > 0),
            "cpu_generation_ms": float(elapsed_ms),
            "energy_signal_source": energy_source,
            "fitness_signal_kind": fitness_signal_kind,
            "behavior_descriptors": [row["behavior_descriptor"] for row in generated],
            "verifier_is_oracle": False,
        }
        return output

    def best_sequence(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        goal_energy: Any | None = None,
        action_effect_scorer: Any | None = None,
        min_len: int = 2,
    ) -> ActionSequence:
        for elite in self.generate(
            frame,
            candidates,
            goal_energy=goal_energy,
            action_effect_scorer=action_effect_scorer,
        ):
            if len(elite.sequence) >= int(min_len):
                return elite.sequence
        return tuple()

    def diagnostics(self) -> dict[str, Any]:
        archive = self._last_archive.diagnostics() if self._last_archive is not None else {}
        last_pool_source = self._last_candidate_pool.get("energy_signal_source")
        return {
            "enabled": bool(self.config.enabled),
            "use_energy_fitness": bool(self.config.use_energy_fitness),
            "energy_signal_source": str(last_pool_source)
            if last_pool_source
            else _energy_signal_source(
                self.goal_energy,
                use_energy_fitness=bool(self.config.use_energy_fitness),
            ),
            "random_seed": int(self.config.random_seed),
            "max_sequence_len": int(self.config.max_sequence_len),
            "mutation_rounds": int(self.config.mutation_rounds),
            "generated_sequences": int(self._generated_sequences),
            "archive": archive,
            "candidate_pool": dict(self._last_candidate_pool),
            "verifier_is_oracle": False,
        }


def coerce_qd_generator(
    value: Any,
    *,
    action_effect_scorer: Any | None = None,
    goal_energy: Any | None = None,
) -> EnergyFitnessQDGenerator | None:
    """Normalize live-path QD configuration into a generator instance."""

    if value is None or value is False:
        return None
    if isinstance(value, EnergyFitnessQDGenerator):
        return value
    if isinstance(value, EnergyFitnessQDConfig):
        return EnergyFitnessQDGenerator(
            value,
            goal_energy=goal_energy,
            action_effect_scorer=action_effect_scorer,
        )
    if value is True:
        return EnergyFitnessQDGenerator(
            EnergyFitnessQDConfig(),
            goal_energy=goal_energy,
            action_effect_scorer=action_effect_scorer,
        )
    if hasattr(value, "best_sequence") or hasattr(value, "generate_candidate_pool"):
        return value
    return None

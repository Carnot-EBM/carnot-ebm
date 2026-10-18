"""Continual-learning utilities for Carnot experiments."""

from carnot.learning.kan_cl import (
    KanClLearner,
    SplitTask,
    build_split_task_benchmark_payload,
    make_split_task_constraint_tasks,
    write_split_task_benchmark_artifact,
)
from carnot.learning.fast_slow import (
    FastSlowTrainer,
    FastWeights,
    SlowWeights,
    VerifiedConstraint,
)

__all__ = [
    "FastSlowTrainer",
    "FastWeights",
    "KanClLearner",
    "SlowWeights",
    "SplitTask",
    "VerifiedConstraint",
    "build_split_task_benchmark_payload",
    "make_split_task_constraint_tasks",
    "write_split_task_benchmark_artifact",
]

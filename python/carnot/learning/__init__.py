"""Continual-learning utilities for Carnot experiments."""

from carnot.learning.kan_cl import (
    KanClLearner,
    SplitTask,
    build_split_task_benchmark_payload,
    make_split_task_constraint_tasks,
    write_split_task_benchmark_artifact,
)

__all__ = [
    "KanClLearner",
    "SplitTask",
    "build_split_task_benchmark_payload",
    "make_split_task_constraint_tasks",
    "write_split_task_benchmark_artifact",
]

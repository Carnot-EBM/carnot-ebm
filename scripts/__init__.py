"""Experiment scripts for Carnot research."""

from __future__ import annotations

import importlib
from types import ModuleType


def __getattr__(name: str) -> ModuleType:
    if name == "research_conductor":
        module = importlib.import_module(f"{__name__}.{name}")
        from .arc_nocov_precondition_lint import install_research_conductor_activation_guard

        install_research_conductor_activation_guard(module)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

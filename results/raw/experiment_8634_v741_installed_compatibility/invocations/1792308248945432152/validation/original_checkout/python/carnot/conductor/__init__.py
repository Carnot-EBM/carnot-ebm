"""Conductor-side runtime helpers for the autoresearch loop.

Submodules in this package are loaded by `scripts/research_conductor.py`
or by experiment scripts spawned by the conductor. Code here is *runtime
infrastructure* for the autoresearch loop — single-run guards, host
health probes, eventual orphan-tracker registry, eventual supervisor
hooks. Distinct from `python/carnot/autoresearch/` which is the
research-loop *content* (planner, evaluator, lessons, skills).

Spec: REQ-INFRA-067 (single-run guard), REQ-INFRA-072 (process-isolation
proposals).
"""

from carnot.conductor.single_run_guard import SingleRunHeld, acquire

__all__ = ["SingleRunHeld", "acquire"]

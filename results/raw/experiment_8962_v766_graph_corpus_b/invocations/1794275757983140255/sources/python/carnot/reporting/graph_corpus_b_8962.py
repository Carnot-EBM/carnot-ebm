"""REQ-REPORT-8962: reuse qualified custody with an independent shard B identity."""

from __future__ import annotations

import sys
from typing import Any

from carnot.reporting import graph_corpus_a_8961 as shared
from carnot.reporting import graph_corpus_runner_8961 as runner
from carnot.reporting.graph_corpus_a_8961 import (
    ExperimentTemplate as ExperimentTemplate,
    authenticate as authenticate,
    raw_complete as raw_complete,
)

Json = dict[str, Any]
ROOT = shared.ROOT
NAME = "experiment_8962_v766_graph_corpus_b"
TASK = "exp8962-graph-corpus-b"
EXPERIMENT_ID = 8962
SHARD = "B"
SEED = 766062
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [*shared.OWNED[:-1], "python/carnot/reporting/graph_corpus_b_8962.py", CLI]
TEST = "tests/python/test_graph_corpus_b_8962.py"
TESTS = [shared.TEST, TEST]
INPUTS = shared.INPUTS


def ordered_units(protocol: Json) -> list[Json]:
    """Select B from frozen methods without reading shard A's outcome."""
    return shared.ordered_units(protocol, SHARD)


def reduce(units: list[Json], rows: list[Json]) -> Json:
    """Keep independent labels separate from the response confidence view."""
    return shared.reduce(units, rows, SHARD)


def qualified(validation: Json) -> bool:
    """B readiness needs measured statements for its adapter and direct CLI."""
    return shared.qualified(validation, OWNED)


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Bind the result to B without changing the shared readiness gates."""
    return shared.build(work, validation, fixture, producer=sys.modules[__name__])


def main(argv: list[str] | None = None) -> int:
    """Use the qualified publication and replay path with B's lease owner."""
    return runner.main(argv, producer=sys.modules[__name__])

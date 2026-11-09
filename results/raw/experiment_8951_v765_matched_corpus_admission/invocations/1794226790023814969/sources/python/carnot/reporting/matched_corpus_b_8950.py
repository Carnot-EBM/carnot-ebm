"""REQ-REPORT-8950: reuse shard custody with an independent B scheduler identity."""

from __future__ import annotations

import sys
from typing import Any

from carnot.reporting import matched_corpus_a_8949 as shared
from carnot.reporting import matched_corpus_runner_8949 as runner
from carnot.reporting.matched_corpus_a_8949 import (
    ExperimentTemplate as ExperimentTemplate,
    authenticate as authenticate,
    raw_complete as raw_complete,
)

Json = dict[str, Any]
ROOT = shared.ROOT
NAME = "experiment_8950_v765_matched_corpus_b"
TASK = "exp8950-matched-corpus-b"
EXPERIMENT_ID = 8950
SHARD = "B"
SEED = 765050
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [*shared.OWNED[:-1], "python/carnot/reporting/matched_corpus_b_8950.py", CLI]
TEST = "tests/python/test_matched_corpus_b_8950.py"
TESTS = [shared.TEST, TEST]
INPUTS = shared.INPUTS


def ordered_units(protocol: Json) -> list[Json]:
    """Select B from the original roster without reading shard A's outcome."""
    return shared.ordered_units(protocol, SHARD)


def reduce(units: list[Json], rows: list[Json]) -> Json:
    """Use the same independent labels and response-only confidence schema."""
    return shared.reduce(units, rows, SHARD)


def qualified(validation: Json) -> bool:
    """Require measured statements for the B adapter as well as shared code."""
    return shared.qualified(validation, OWNED)


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Bind the artifact to B while preserving shared readiness assertions."""
    return shared.build(work, validation, fixture, producer=sys.modules[__name__])


def main(argv: list[str] | None = None) -> int:
    """Keep publication, private checks and cold replay on the established path."""
    return runner.main(argv, producer=sys.modules[__name__])

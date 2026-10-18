"""REQ-REPORT-8527: qualify original source custody without training a new head.

The existing strongest-baseline reader retains the numerical work, source pins
and prediction-before-target seal order. These adapters change producer identity,
current authority and diagnostic schema; exposed historical losses stay descriptive.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import v732_strongest_baseline_audit as baseline
from carnot.reporting import v733_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v733_source_diagnostics import (
    DIAGNOSTIC_FIELDS as DIAGNOSTIC_FIELDS,
    diagnostic_rows,
)

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = (
    "experiment_8527_v733_source_consumer_qualification",
    "exp8527-source-consumer-qualification",
)
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v733_source_consumer_8527.py"
OWNED = [
    f"python/carnot/reporting/v733_{p}.py"
    for p in ["source_diagnostics", "source_consumer_qualification", "source_consumer_runner"]
] + [CLI]
COMPARATORS = (
    "raw_p2",
    "legacy_full_source",
    "brier_scalar",
    "brier_full",
    "brier_source_permuted",
)
MODEL_SPECS: list[Json] = []
read, reference, checksum = baseline.read, baseline.reference, baseline.checksum
read_bound_sidecar = baseline.read_bound_sidecar
LITERATURE = f"results/raw/{NAME}/literature/mapping.json"
LITERATURE_PIN = "sha256:88f3e3f8b17055266e2dbd6a34dccad55c6bbc1fd23c5a9e1b956f70be786df0"
METHODS_PIN = "sha256:5185ba15b6e23c5a8185553fe4c878921255768bd7d12d88ebf16202d3f5c004"
CUSTODY = f"results/raw/{NAME}/original_custody.json"
CUSTODY_PIN = "sha256:efc9561824661b8ed190e98ed9a9cb1eafa52e4725288f548b755223ef03c724"
_MEASURE, _BUILD, _REPLAY, _MANIFEST = (
    baseline.measure,
    baseline.build,
    baseline.replay,
    baseline.comparator_manifest,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual progress because CPU readers have no model activity to observe."""
    print(f"[exp8527] phase={phase} completed={completed} pending={pending}", flush=True)


def comparator_manifest(root: Path) -> Json:
    """Name the V733 arms while keeping original source vectors, roles and donor maps.

    Brier heads have not run. Their input slots are registered future work and
    cannot be presented as fitted predictions just because custody qualifies.
    """
    value = dict(_MANIFEST(root))
    value.update(
        schema="carnot.v733.source_comparators.v1", future_producer="exp8528-identity-brier-fit"
    )
    for row in value["slots"]:
        future = row["arm"].startswith("brier_")
        row.update(
            status="future_dependency" if future else "authenticated_cached_input",
            missing_reason="Exp8528_not_run" if future else None,
        )
    return value


@contextmanager
def binding() -> Iterator[None]:
    """Keep shipped checks intact while routing this invocation to its own producer.

    Scoped bindings are restored after every call so historical readers retain
    their original identities and no historical publication is rewritten.
    """
    proxy = SimpleNamespace(**dict(vars(methods), legacy=baseline.shipped.methods.legacy))
    with patch.multiple(
        baseline,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        methods=methods,
        _METHODS=proxy,
        METHODS_PIN=METHODS_PIN,
        COMPARATORS=COMPARATORS,
        LITERATURE=LITERATURE,
        LITERATURE_PIN=LITERATURE_PIN,
        comparator_manifest=comparator_manifest,
        progress=progress,
    ):
        yield


def measure(root: Path, raw: Path) -> Json:
    """Reuse the real CPU source reader and retain immutable historical failure evidence."""
    with binding():
        work = dict(_MEASURE(root, raw))
    work["diagnostic_schema_rows"] = diagnostic_rows(
        work["authority"].get("required_gate_observations", []) + work["failures"]
    )
    work["original_failure_reference"] = read(dict(path=str(ROOT / CUSTODY), sha256=CUSTODY_PIN))
    work["source_refs"].append(dict(path=str(ROOT / CUSTODY), sha256=CUSTODY_PIN))
    work["source_refs"].extend(work["original_failure_reference"])
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Readiness grants fitting permission only; it never establishes source benefit."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    value.update(
        experiment_id=8527,
        task_id=TASK,
        milestone="2026.10.733",
        run_date="20261015",
        honest_verdict=f"complete_{value['verdict_class']}_v733_source_consumer_qualification",
        gate_check_summary=diagnostic_rows(value["gate_check_summary"]),
        diagnostic_schema_rows=work["diagnostic_schema_rows"],
        original_failure_reference=work["original_failure_reference"],
        consumer_receipt=dict(
            scope="prepublication_private_candidate",
            cold_read_passed=any(r.get("name") == "cold" and r["passed"] for r in receipts),
            cold_replay_receipts=[r for r in receipts if r.get("name") == "cold"],
            exact_final_bytes_receipt_path=str(
                output.parent / "raw" / output.stem / "terminal_validation.json"
            ),
        ),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in [8528, 8529, 8535]
        ],
    )
    value["field_principles"].update(
        {
            k: "Bind original custody and actual invocation; readiness is separate from source value."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Recompute sealed primitives and reject locally repaired diagnostic or source evidence."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        if work["diagnostic_schema_rows"] != diagnostic_rows(
            work["authority"].get("required_gate_observations", []) + work["failures"]
        ):
            return False
        if work["original_failure_reference"] != read(
            dict(path=str(ROOT / CUSTODY), sha256=CUSTODY_PIN)
        ):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    with binding(), patch.object(baseline, "build", build):
        return bool(_REPLAY(path))

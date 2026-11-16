"""REQ-REPORT-9137-V778: reconcile all scheduled outcomes without model work."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v777_capstone as lifecycle
from carnot.reporting import v778_capstone_evidence as evidence
from carnot.reporting import v775_capstone as legacy
from carnot.reporting import scoped_provenance_9014 as runner
from carnot.reporting.encoder_head_diagnosis_8972 import (
    checksum as checksum,
    reference as reference,
)

Json = dict[str, Any]
ROOT = evidence.ROOT
NAME, TASK = "experiment_9137_v778_capstone", "exp9137-capstone"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v778_capstone_9137.py"
OWNED = [
    "python/carnot/reporting/v778_capstone.py",
    "python/carnot/reporting/v778_capstone_evidence.py",
    CLI,
]
DIRECT = lifecycle.DIRECT
REUSED = [
    *lifecycle.REUSED,
    *lifecycle.OWNED,
    "python/carnot/reporting/v778_contract.py",
    "openspec/change-proposals/v778-capstone-inputs.json",
]
MODEL_SPECS: list[Json] = []
_collect, _build, _replay, _main = (
    lifecycle.collect,
    lifecycle.build,
    lifecycle.replay,
    lifecycle.main,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries; the reused child supervisor supplies thirty-second heartbeats."""
    print(f"[exp9137] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Reuse the established lifecycle while freezing this producer's own validation scope."""
    fields: Json = dict(
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        DIRECT=DIRECT,
        REUSED=REUSED,
        evidence=evidence,
        progress=progress,
        collect=collect,
        build=build,
        replay=replay,
    )
    with patch.multiple(legacy, **fields), patch.multiple(lifecycle, **fields, binding=binding):
        yield


def collect(raw: Path, run_date: str, root: Path = ROOT) -> Json:
    """Freeze commands and authenticate source bytes before computing any headline."""
    with binding():
        return dict(_collect(raw, run_date, root))


def build(work: Json) -> Json:
    """Administrative completion never promotes fixtures or absent scientific predictions."""
    with binding():
        value = _build(work)
    value.update(
        experiment_id=9137,
        milestone="2026.11.778",
        random_seed=778137,
        claim_scope="Fourteen administrative outcomes; two static and seven online bounds remain unmeasured. No independent verifier benefit or ARC solve is claimed.",
        exposure_scope="Exposed LPHB development only. Cached model evidence is historical provenance; gauge and recovery fixtures grant no scientific credit.",
        staged_authority_status="Authenticated active V778 queue, complete objects and visible design",
        sample_size_budget=dict(
            administrative_slots=14,
            static_test_qids=100,
            online_future_qids=40,
            repeated_retention_unique_qids=100,
        ),
        scientific_scope_ledger=[
            dict(branch=b, status=s, independent_generalization_score=0)
            for b, s in [
                ("static_energy", "unmeasured"),
                ("online_energy", "unmeasured"),
                ("gauge_controls", "fixture_only"),
                ("donor_labels", "exposed_development"),
                ("cost", "unmeasured_service"),
                ("graph", "historical_circular_positive_only"),
                ("ARC", "historical_only_no_solve_claim"),
            ]
        ],
    )
    value["task_outcome_rows"][-1].update(experiment_id=9137, primary_path=f"results/{NAME}.json")
    value["archive_lag"].update(
        current_milestone="2026.11.778",
        archive_lagged=value["archive_lag"]["observed_archive_end"] != "2026.11.778",
    )
    for row in value["rows"]:
        row.update(qid=None, sibling=None)
    value.setdefault("hardware_limits", {})
    value["field_principles"].update(
        {
            "two_bound_rows": "Both original static energy contrasts are conjunctive; absent predictions cannot be reported as null results.",
            "seven_bound_rows": "Five future comparisons and two retention limits remain independent of static qualification.",
            "gauge_control_rows": "Constant and alternating gauges test susceptibility; normalized probability invariance grants no verifier benefit.",
            "donor_label_change_rows": "Actual changed donor labels distinguish label displacement from a merely nonidentity donor map.",
            "donor_changed_label_fraction": "Changed-label numerator and original support prevent an inflated control contrast.",
            "scientific_scope_ledger": "Fixture validity, custody, exposed science and exact-oracle evidence retain separate scopes.",
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from retained immutable rows, rejecting even freshly rehashed inventions."""
    with binding():
        return dict(_replay(path))


def main(argv: list[str] | None = None) -> int:
    """Run real scoped checks and publish only bytes accepted by the unchanged checkers."""
    with binding():
        return int(_main(argv))

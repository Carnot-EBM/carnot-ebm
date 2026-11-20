"""REQ-REPORT-9193-V782: unconditional reconciliation with no current model work."""

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v781_capstone as prior
from carnot.reporting import v782_capstone_evidence as evidence
from carnot.reporting import scoped_provenance_9014 as runner
from carnot.reporting import v775_capstone as legacy
from carnot.reporting.encoder_head_diagnosis_8972 import checksum as checksum
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = evidence.ROOT
NAME, TASK = "experiment_9193_v782_capstone", "exp9193-capstone"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v782_capstone_9193.py"
OWNED = [
    "python/carnot/reporting/v782_capstone.py",
    "python/carnot/reporting/v782_capstone_evidence.py",
    "python/carnot/reporting/v782_capstone_reduction.py",
    CLI,
]
REUSED = [
    *prior.REUSED,
    *prior.OWNED,
    "python/carnot/reporting/v782_contract_evidence.py",
    evidence.CONFIG,
]
DIRECT = prior.DIRECT
MODEL_SPECS: list[Json] = []
_collect, _build, _replay, _main = prior.collect, prior._build, prior._replay, prior.main
_reduce = evidence.reduce


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase transitions while the child supervisor reports long checks."""
    print(f"[exp9193] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Use established validation and publication under this consumer's exact scope."""
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
    with (
        patch.multiple(legacy, **fields),
        patch.multiple(prior, **fields, binding=binding, reduce=evidence.reduce),
    ):
        yield


def collect(raw: Path, run_date: str, root: Path = ROOT) -> Json:
    """Freeze operands, resource receipts and unchanged commands before reduction."""
    with binding():
        return dict(_collect(raw, run_date, root))


def build(work: Json) -> Json:
    """Administrative completeness grants no new scientific or independent readiness."""
    normalized = dict(work, tasks=[dict(t, gated_on=t.get("gated_on", [])) for t in work["tasks"]])
    with binding(), patch.object(evidence, "reduce", lambda _: _reduce(work)):
        value = _build(normalized)
    value.update(
        experiment_id=9193,
        milestone="2026.11.782",
        random_seed=evidence.PROTOCOL["seed"],
        requested_inference_substrate_class="no_model_load",
        generated_tokens=0,
        verifier_is_oracle=False,
        preconditions_checked=work["preconditions_checked"],
        model_invocation_status="not_invoked_empty_requested_list",
        exposure_scope="exposed_development",
        outcome_rows=value["task_outcome_rows"],
        sample_size_budget=dict(
            intended=14,
            attempted=14,
            completed=14,
            failed=0,
            censored=0,
            effective_independent_groups=0,
        ),
        open_gaps={"GAP-ORACLE-DISTINCT": "open", "GAP-DETECTOR-AUROC-4208": "open"},
        staged_authority_status="Fourteen full active objects compared with visible and embedded V782 design; staged next queue absent after activation",
        full_task_objects=work["tasks"],
        full_task_digest=evidence.canonical_hash(work["tasks"]),
        claim_scope=dict(
            exposure_scope="Original exposed LPHB900/300 development panel",
            independent_generalization_score=0,
            allowed_claims=[
                "fourteen terminal administrative outcomes",
                "authenticated disqualified checker and ARC observations",
                "unmeasured primary science and secondary CES recorded separately",
            ],
        ),
        acceptance_gates={
            k: dict(
                condition=v,
                principle="Frozen conditions establish accounting only; secondary evidence never rescues primary science",
            )
            for k, v in dict(
                contract="fourteen exact ordered tasks",
                static="both original Brier bounds and defined accepted risk",
                learning="all seven original learning bounds and exact recovery",
                CES="completeness separate from three ranking bounds",
                validation="all affected checks;100 percent owned statements; zero exclusions",
                independence="exposed panel; independent score zero",
            ).items()
        },
    )
    if any(not r["passed"] for r in work["checks"]):
        value["inference_substrate_class"] = "blocked_no_run"
    value["model_invocation_counts"].update(
        current_llm_invocations=0,
        loads_attempted=0,
        loads_completed=0,
        loads_failed=0,
        forward_calls=0,
        generated_tokens=0,
    )
    value["task_outcome_rows"][-1].update(
        experiment_id=9193,
        primary_path=f"results/{NAME}.json",
        primary_status="self_slot",
        scheduler_alternative=None,
        quarantine=None,
        gate_operands=[],
        gate_check_summary=[],
        readiness=dict(reconciliation_ready_score=value["reconciliation_ready_score"]),
    )
    value["publication_gates"] = dict(
        **{k: value[k] for k in ["G1", "G2", "G3", "G4", "paper_ready", "unmet_gates"]},
        publication_scope="historical_FoVer",
        current_science_paper_ready=False,
    )
    value["scientific_scope_ledger"] = [
        dict(branch=b, scope=s, independent_generalization_score=0)
        for b, s in [
            ("validation", "Disqualified correction; complete qualification absent"),
            ("static", "Two original bounds unmeasured"),
            ("learning", "Seven original bounds and recovery unmeasured"),
            ("CES", "Secondary completeness and three bounds unmeasured"),
            ("cost", "Complete scientific service clocks unmeasured"),
            ("ARC", "Disqualified fixture compatibility; no live generalization"),
        ]
    ]
    for row in value["rows"]:
        row.update(qid=None, sibling=None, seed=evidence.PROTOCOL["seed"])
    value["archive_lag"].update(
        current_milestone="2026.11.782",
        archive_lagged=value["archive_lag"]["observed_archive_end"] != "2026.11.782",
    )
    value["source_references"] = [
        dict(
            r,
            producer_path=r.get("original_path", r["path"]),
            imported_fields=["original_terminal_outcomes_and_independent_unit_evidence"],
        )
        for r in value["source_references"]
    ]
    value["gate_failure_rows"] = [r for r in value["gate_check_summary"] if not r["passed"]]
    value["timing_budget"] = dict(
        measurement_limit_s=3300,
        validation_reserve_s=1200,
        publication_reserve_s=300,
        hard_cap_s=4800,
        deadline_monotonic_s=work["started_ns"] / 1e9 + 4800,
        elapsed_is_watchdog_ceiling=False,
    )
    value["methodology_note"] = (
        "Authenticate complete V782 contracts and original producer bytes; recompute independent human groups, obligation union and compatibility denominators. Retain V781 retirement and all original diagnostic/ARC disqualifications. Missing predictions leave original bounds undefined. CES is secondary and cannot promote energy. No LLM, generation, training, gameplay, physical hardware, queue activation or external publication."
    )
    value["field_principles"].update(
        {
            k: "Exact retained operands and executed receipts support accounting; absent science remains blocked"
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild retained inputs so rehashed invented science still fails cold replay."""
    with binding():
        return dict(_replay(path))


def main(argv: list[str] | None = None) -> int:
    """Share one monotonic deadline across descendants and checked atomic publication."""
    started = time.monotonic()
    deadline = min(
        float(os.environ.get("CARNOT_V782_CAPSTONE_DEADLINE", started + 4800)), started + 4800
    )
    supervisor = child

    def bounded(name: str, *args: Any, **kwargs: Any) -> Json:
        reserve = 0 if name in {"cold_replay", "adversarial", "strict_rows"} else 300
        remaining = deadline - reserve - time.monotonic()
        if remaining <= 0:
            raise ValueError("shared_deadline")
        kwargs["deadline"] = min(kwargs.get("deadline", 600), remaining)
        kwargs["heartbeat"] = min(kwargs.get("heartbeat", 60), 60)
        return dict(supervisor(name, *args, **kwargs))

    with (
        binding(),
        patch.object(runner, "child", bounded),
        patch.dict(os.environ, CARNOT_V782_CAPSTONE_DEADLINE=str(deadline)),
    ):
        return int(_main(argv))

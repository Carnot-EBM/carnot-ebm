"""REQ-REPORT-9221-V784: unconditional accounting with zero current model work."""

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v781_capstone as lifecycle
from carnot.reporting import v783_capstone as previous
from carnot.reporting import v784_capstone_evidence as evidence
from carnot.reporting import scoped_provenance_9014 as runner
from carnot.reporting import v775_capstone as legacy
from carnot.reporting.encoder_head_diagnosis_8972 import (
    checksum as checksum,
    reference as reference,
)
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = evidence.ROOT
NAME, TASK = "experiment_9221_v784_capstone", "exp9221-capstone"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v784_capstone_9221.py"
OWNED = [
    "python/carnot/reporting/v784_capstone.py",
    "python/carnot/reporting/v784_capstone_evidence.py",
    "python/carnot/reporting/v784_capstone_reduction.py",
    CLI,
]
REUSED = [*previous.REUSED, *previous.OWNED, evidence.CONFIG]
DIRECT = previous.DIRECT
MODEL_SPECS: list[Json] = []
_collect, _build, _replay, _main = lifecycle.collect, legacy.build, legacy.replay, lifecycle.main
_reduce = evidence.reduce


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so the existing supervisor can expose long subprocess waits."""
    print(f"[exp9221] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Bind this producer's exact scope without changing historical producers or checkers."""
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
        patch.multiple(lifecycle, **fields, binding=binding, reduce=evidence.reduce),
    ):
        yield


def collect(raw: Path, run_date: str = "20261118", root: Path = ROOT) -> Json:
    """Freeze source, resource and command receipts before any outcome reduction."""
    mapped = "20261118" if run_date in {"20261117", "20261118"} else run_date
    with binding():
        return dict(_collect(raw, mapped, root))


def build(work: Json) -> Json:
    """Administrative completeness stays separate from exposed, unavailable scientific results."""
    normalized = dict(work, tasks=[dict(t, gated_on=t.get("gated_on", [])) for t in work["tasks"]])
    with binding(), patch.object(evidence, "reduce", lambda _: _reduce(work)):
        value = _build(normalized)
    value.update(
        experiment_id=9221,
        milestone="2026.11.784",
        random_seed=evidence.PROTOCOL["seed"],
        requested_inference_substrate_class="no_model_load",
        generated_tokens=0,
        verifier_is_oracle=False,
        full_task_objects=work["tasks"],
        full_task_digest=evidence.canonical_hash(work["tasks"]),
        preconditions_checked=work["preconditions_checked"],
        model_invocation_status="not_invoked_empty_requested_list",
        claim_scope=dict(
            exposure_scope="Original exposed LPHB900/300 development panel",
            independent_generalization_score=0,
            allowed_claims=[
                "fourteen terminal administrative outcomes",
                "authenticated historical V782/V783 failures and Exp9208 replay-authority disqualification",
                "independently qualified Exp9211 target-identifiability and affine label-ambiguity method without model predictions",
                "unmeasured human science, secondary CES and two label sensitivity branches",
            ],
        ),
        sample_size_budget=dict(
            intended=14,
            attempted=14,
            completed=14,
            failed=0,
            censored=0,
            effective_independent_groups=0,
        ),
        staged_authority_status="Fourteen complete V784 objects compared with visible and embedded design; V782 and V783 contracts retained",
        open_gaps={"GAP-ORACLE-DISTINCT": "open", "GAP-DETECTOR-AUROC-4208": "open"},
        acceptance_gates={
            k: dict(
                condition=v,
                principle="Frozen conditions establish accounting only; secondary evidence never rescues primary science",
            )
            for k, v in dict(
                contract="fourteen exact ordered objects and protocol-owner replay",
                corpus="lossless full input parity; unchanged checker semantics",
                forwards="all original674 plus affected obligations",
                static="both original Brier bounds and defined accepted risk",
                learning="seven original bounds and exact recovery",
                CES="completeness separate from three ranking bounds",
                ambiguity="independent target/ambiguity method and two separate conditional envelopes",
                validation="all affected checks;100 percent owned statements; zero exclusions",
                independence="exposed panel; independent score zero",
            ).items()
        },
        timing_budget=dict(
            measurement_limit_s=3300,
            validation_reserve_s=1200,
            publication_reserve_s=300,
            hard_cap_s=4800,
            deadline_monotonic_s=work["started_ns"] / 1e9 + 4800,
            elapsed_is_watchdog_ceiling=False,
        ),
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
        experiment_id=9221,
        primary_path=f"results/{NAME}.json",
        primary_status="self_slot",
        scheduler_alternative=None,
        gate_operands=[],
        gate_check_summary=[],
        readiness=dict(reconciliation_ready_score=value["reconciliation_ready_score"]),
    )
    value["outcome_rows"] = value["task_outcome_rows"]
    value["publication_gates"] = dict(
        **{k: value[k] for k in ["G1", "G2", "G3", "G4", "paper_ready", "unmet_gates"]},
        publication_scope="historical_FoVer",
        current_science_paper_ready=False,
    )
    value["scientific_scope_ledger"] = [
        dict(branch=b, scope=s, independent_generalization_score=0)
        for b, s in [
            ("replay_authority", "Disqualified on direct-consumer timeout; V783 zeros preserved"),
            ("corpus", "Lossless optimization unmeasured; checker semantics unchanged"),
            ("forward_qualification", "Complete obligation execution absent"),
            ("static", "Both original bounds unmeasured"),
            ("learning", "Seven bounds and recovery unmeasured"),
            ("CES", "Secondary completeness and three ranking bounds unmeasured"),
            (
                "label_ambiguity",
                "Independent Exp9211 target and endpoint method qualified; both conditional sensitivity branches unmeasured without model predictions",
            ),
            ("cost", "Complete scientific service clocks unmeasured"),
            ("ARC", "Historical Exp9192 disqualified compatibility only; no experiment scheduled"),
        ]
    ]
    for row in value["rows"]:
        row.update(qid=None, sibling=None, seed=evidence.PROTOCOL["seed"])
    value["archive_lag"].update(
        current_milestone="2026.11.784",
        archive_lagged=value["archive_lag"]["observed_archive_end"] != "2026.11.784",
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
    value["methodology_note"] = (
        "Authenticate complete V784 contracts and original producer bytes; recompute V782/V783 independent human groups, obligation union, cost denominators and Exp9211 target/ambiguity evidence. Preserve two-executed/twelve-skipped V783 closure, four-executed/ten-skipped V782 closure, V781 retirement and diagnostic/ARC disqualifications. Separately account for protocol-owner replay, corpus-input optimization, full qualification, primary human science, secondary CES, qualified Exp9211 target/ambiguity method and two unmeasured conditional label sensitivity branches. Missing predictions remain undefined. No LLM, generation, training, gameplay, hardware, queue activation or external publication."
    )
    value["field_principles"].update(
        {
            k: "Exact retained operands and executed receipts support accounting; absent science stays blocked"
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild retained inputs so fresh checksums cannot authorize invented science."""
    with binding():
        return dict(_replay(path))


def main(argv: list[str] | None = None) -> int:
    """Share one monotonic deadline across descendants and checked atomic publication."""
    started = time.monotonic()
    deadline = min(
        float(os.environ.get("CARNOT_V784_CAPSTONE_DEADLINE", started + 4800)), started + 4800
    )

    def bounded(name: str, *args: Any, **kwargs: Any) -> Json:
        """A child cannot spend the publication reserve on another measurement attempt."""
        remaining = (
            deadline
            - (0 if name in {"cold_replay", "adversarial", "strict_rows"} else 300)
            - time.monotonic()
        )
        if remaining <= 0:
            raise ValueError("shared_deadline")
        kwargs["deadline"] = min(kwargs.get("deadline", 600), remaining)
        kwargs["heartbeat"] = min(kwargs.get("heartbeat", 60), 60)
        return dict(child(name, *args, **kwargs))

    mapped = [
        "20261117" if a == "20261118" else "invalid" if a == "20261117" else a
        for a in (sys.argv[1:] if argv is None else argv)
    ]
    with (
        binding(),
        patch.object(runner, "child", bounded),
        patch.dict(os.environ, CARNOT_V784_CAPSTONE_DEADLINE=str(deadline)),
    ):
        return int(_main(mapped))

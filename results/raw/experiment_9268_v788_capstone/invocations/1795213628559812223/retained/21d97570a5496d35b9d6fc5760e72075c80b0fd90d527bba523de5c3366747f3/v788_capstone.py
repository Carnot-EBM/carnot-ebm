"""REQ-REPORT-9268-V788: unconditional V788 accounting and V787 history preservation with zero current model work."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import scoped_provenance_9014 as runner
from carnot.reporting import v775_capstone as legacy
from carnot.reporting import v781_capstone as lifecycle
from carnot.reporting import v786_capstone as previous
from carnot.reporting import v788_capstone_evidence as evidence
from carnot.reporting.encoder_head_diagnosis_8972 import (
    checksum as checksum,
    reference as reference,
)
from carnot.reporting.v709_execution import child
from carnot.reporting.v763_capstone_reduction import require

Json = dict[str, Any]
ROOT = evidence.ROOT
NAME, TASK = "experiment_9268_v788_capstone", "exp9268-capstone"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v788_capstone_9268.py"
OWNED = [
    "python/carnot/reporting/v788_capstone.py",
    "python/carnot/reporting/v788_capstone_evidence.py",
    "python/carnot/reporting/v788_capstone_reduction.py",
    CLI,
]
REUSED = [*previous.REUSED, *previous.OWNED, evidence.CONFIG]
DIRECT = previous.DIRECT
MODEL_SPECS: list[Json] = []
_collect, _build, _main = lifecycle.collect, legacy.build, lifecycle.main
_reduce = evidence.reduce


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so the existing supervisor can expose long subprocess waits."""
    print(f"[exp9268] phase={phase} completed={completed} pending={pending}", flush=True)


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


def collect(raw: Path, run_date: str = "20261120", root: Path = ROOT) -> Json:
    """Freeze source, resource and command receipts before any outcome reduction."""
    mapped = "20261120" if run_date in {"20261117", "20261120"} else run_date
    with binding():
        return dict(_collect(raw, mapped, root))


def build(work: Json) -> Json:
    """Administrative completeness stays separate from exposed, unavailable scientific results."""
    normalized = dict(work, tasks=[dict(t, gated_on=t.get("gated_on", [])) for t in work["tasks"]])
    with binding(), patch.object(evidence, "reduce", lambda _: _reduce(work)):
        value = _build(normalized)
    value.update(
        experiment_id=9268,
        milestone="2026.11.788",
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
                "fourteen terminal V788 administrative outcomes: Exp9255 seven-shard replay and 14-task contract pass, Exp9256 lossless traversal with full-suite timeout disqualification, Exp9257 pre-gate skip, Exp9258 Victris Arena Gibbs state-energy harness pass with missing-wheel block, Exp9259-Exp9267 upstream-retired skips, and Exp9268 reconciliation",
                "authenticated V787 record: Exp9241 six-shard pass with 7-task truncation/missing-YAML and coverage-list mismatch block, Exp9244 qualified conformal-risk/CA-OCRC/Mycelium protocol, Exp9242/9243/9245-9247 skips, and seven omitted V787 slots",
                "authenticated historical V782-V786 closures and independently qualified Exp9211 target/ambiguity, Exp9225 selective-deferral/spline-support, and Exp9244 conformal/CA-OCRC/Mycelium methods",
                "unmeasured human science, secondary CES, descriptive ambiguity/deferral/spline-support/conformal/CA-OCRC branches, and blocked Victris simulator matches",
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
        staged_authority_status="Fourteen complete V788 objects compared with visible table and embedded YAML; V782-V787 contracts retained",
        open_gaps={"GAP-ORACLE-DISTINCT": "open", "GAP-DETECTOR-AUROC-4208": "open"},
        acceptance_gates={
            k: dict(
                condition=v,
                principle="Frozen conditions establish accounting only; secondary evidence never rescues primary science",
            )
            for k, v in dict(
                contract="fourteen exact ordered objects, embedded-YAML parity, seven consumer shards and coverage-list parity",
                corpus="lossless full input parity; unchanged checker semantics; all required checks pass",
                forwards="all original 674 plus affected V780-V788 obligations",
                victris="sub-10ms Gibbs state-energy bot, untrusted-shout isolation, DEV/HELDOUT disjointness, and honest absent-wheel blocking",
                conformal="qualified Exp9244 qid-cluster conformal risk bounds, CA-OCRC tracking and Mycelium provenance ledger",
                static="both original Brier bounds and defined accepted risk",
                learning="seven original bounds, exact E2E-020 recovery, and non-interfering Exp9225/Exp9244 shadows",
                CES="completeness separate from three ranking bounds",
                ambiguity="independent target/ambiguity method and two separate conditional envelopes",
                selective_deferral="independent Exp9225 masks and Horvitz-Thompson IPW risk controls",
                spline_support="label-free 57-column cubic B-spline support-density slices",
                validation="all affected checks; 100 percent owned statements; zero exclusions",
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
        experiment_id=9268,
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
            (
                "replay_authority",
                "Exp9255 qualified V788 seven-shard consumer replay (replay_authority_ready_score=1) and 14-task embedded-YAML contract parity (contract_ready_score=1); V787 Exp9241 six-shard pass and 7-task truncation/coverage-list block preserved",
            ),
            (
                "corpus",
                "Exp9256 traversed all 764,150 recursive JSON files with zero parity discrepancies and passed target_module (1544.2s), but disqualified corpus_scan_ready_score=0 on full_python_suite timeout (-9)",
            ),
            (
                "forward_qualification",
                "Exp9257 pre-gate skipped on exp9256-corpus-prefilter.corpus_scan_ready_score==0; complete obligation execution absent",
            ),
            (
                "victris_arena_rsi",
                "Exp9258 qualified victris_harness_ready_score=1 across six sub-10ms Gibbs state-energy archive versions and blocked honestly on absent arena_sim wheel (heldout_matches_played=0)",
            ),
            (
                "conformal_risk_ca_ocrc_and_provenance",
                "Independent Exp9244 qid-cluster conformal risk bounds, CA-OCRC tracking and Mycelium provenance ledger qualified; empirical branches unmeasured without model predictions",
            ),
            ("static", "Both original bounds unmeasured"),
            ("learning", "Seven bounds, recovery and shadow streams unmeasured"),
            ("CES", "Secondary completeness and three ranking bounds unmeasured"),
            (
                "label_ambiguity",
                "Independent Exp9211 target and endpoint method qualified; both conditional sensitivity branches unmeasured without model predictions",
            ),
            (
                "selective_deferral_and_spline_support",
                "Independent Exp9225 selective-deferral masks, IPW estimators and cubic B-spline support-density method qualified; empirical branches unmeasured without model predictions",
            ),
            ("cost", "Complete scientific service clocks unmeasured"),
            ("ARC", "Historical Exp9192 disqualified compatibility only; no experiment scheduled"),
        ]
    ]
    for row in value["rows"]:
        row.update(qid=None, sibling=None, seed=evidence.PROTOCOL["seed"])
    value["archive_lag"].update(
        current_milestone="2026.11.788",
        archive_lagged=value["archive_lag"]["observed_archive_end"] != "2026.11.788",
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
        "Authenticate complete V788 contracts and original producer bytes; recompute V782-V787 independent human groups, obligation union, cost denominators, Exp9211 target/ambiguity evidence, Exp9225 selective-deferral/spline-support evidence, Exp9244 conformal-risk/CA-OCRC/Mycelium evidence, and Exp9258 Victris Arena state-energy bot/archive evidence. Preserve two-executed/five-skipped/seven-omitted V787 closure, one-executed/thirteen-skipped V786 closure, two-executed/three-skipped/nine-omitted V785 closure, three-executed/eleven-skipped V784 closure, two-executed/twelve-skipped V783 closure, four-executed/ten-skipped V782 closure, V781 retirement and diagnostic/ARC disqualifications. Separately account for seven-shard consumer replay and 14-task parity (Exp9255), lossless corpus traversal and full-suite timeout disqualification (Exp9256), pre-gate forward skip (Exp9257), Victris Arena state-energy bot and absent-wheel block (Exp9258), primary human science, secondary CES, qualified Exp9211/Exp9225/Exp9244 methods, and unmeasured conditional branches. Missing predictions remain undefined. No LLM, generation, training, gameplay, hardware, queue activation or external publication."
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
    try:
        value = json.loads(path.read_bytes())
        require(checksum(value) == value["reproducibility_checksum"], "checksum_drift")
        ref = value["work_reference"]
        require(reference(Path(ref["path"])) == ref, "work_drift")
        work = json.loads(Path(ref["path"]).read_bytes())
        for name, saved in work["snapshots"].items():
            require(saved["sha256"] == evidence.PINS[name], "original_pin")
        refs = [
            *work["snapshots"].values(),
            *work["operands"].values(),
            *work["code"],
            work["current_work_reference"],
            *work["repository_health_references"],
            *work["publication_sources"],
        ]
        for saved in refs:
            require(reference(Path(saved["path"]))["sha256"] == saved["sha256"], "operand_drift")
        for entry in work["values"].values():
            require(
                entry["reference"] == work["snapshots"][entry["selected_name"]]
                and entry["value"] == evidence.load_projected(Path(entry["reference"]["path"])),
                "producer_drift",
            )
        for receipt in work["publication_receipts"]:
            require(
                reference(Path(receipt["log_path"]))["sha256"] == receipt["log_sha256"]
                and work.get("publication_gate")
                == json.loads(Path(receipt["log_path"]).read_bytes()),
                "publication_gate_drift",
            )
        require(build(work) == value, "reduction_drift")
        return dict(passed=True, replay_passed=True, slots=14)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(passed=False, replay_passed=False, reason=str(error))


def main(argv: list[str] | None = None) -> int:
    """Share one monotonic deadline across descendants and checked atomic publication."""
    started = time.monotonic()
    deadline = min(
        float(os.environ.get("CARNOT_V788_CAPSTONE_DEADLINE", started + 4800)), started + 4800
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
        "20261117" if a == "20261120" else "invalid" if a == "20261117" else a
        for a in (sys.argv[1:] if argv is None else argv)
    ]
    with (
        binding(),
        patch.object(runner, "child", bounded),
        patch.dict(os.environ, CARNOT_V788_CAPSTONE_DEADLINE=str(deadline)),
    ):
        return int(_main(mapped))

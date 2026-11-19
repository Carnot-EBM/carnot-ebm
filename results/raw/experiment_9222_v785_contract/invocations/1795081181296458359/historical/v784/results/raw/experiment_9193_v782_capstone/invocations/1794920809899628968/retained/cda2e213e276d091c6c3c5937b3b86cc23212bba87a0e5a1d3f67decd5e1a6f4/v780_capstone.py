"""REQ-REPORT-9165-V780: reconcile fourteen outcomes with zero current model work."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v777_capstone as lifecycle
from carnot.reporting import v780_capstone_evidence as evidence
from carnot.reporting import v775_capstone as legacy
from carnot.reporting.encoder_head_diagnosis_8972 import (
    checksum as checksum,
    reference as reference,
)
from carnot.reporting.v780_capstone_reduction import reduce
from carnot.reporting.v778_capstone_evidence import bounds

Json = dict[str, Any]
ROOT = evidence.ROOT
NAME, TASK = "experiment_9165_v780_capstone", "exp9165-capstone"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v780_capstone_9165.py"
OWNED = [
    "python/carnot/reporting/v780_capstone.py",
    "python/carnot/reporting/v780_capstone_evidence.py",
    "python/carnot/reporting/v780_capstone_reduction.py",
    CLI,
]
DIRECT = legacy.DIRECT
REUSED = [
    *lifecycle.REUSED,
    *lifecycle.OWNED,
    "python/carnot/reporting/v779_contract_evidence.py",
    "python/carnot/reporting/v780_contract_evidence.py",
    "openspec/change-proposals/v780-contract-inputs.json",
    "python/carnot/reporting/human_contrast_9141.py",
    evidence.CONFIG,
]
MODEL_SPECS: list[Json] = []
_collect, _build, _replay, _main = lifecycle.collect, legacy.build, legacy.replay, lifecycle.main


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed phases and the shared child supervisor keep blocked work visible."""
    print(f"[exp9165] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Reuse validation and publication while freezing only this producer's new statements."""
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
    """Authenticate source bytes and freeze all commands before computing outcomes."""
    with binding():
        return dict(_collect(raw, run_date, root))


def build(work: Json) -> Json:
    """Administrative readiness stays separate from blocked exposed-panel science."""
    # The shared builder expects every task to carry a gate list, including empty lists.
    normalized = dict(work, tasks=[dict(t, gated_on=t.get("gated_on", [])) for t in work["tasks"]])
    with binding(), patch.object(evidence, "reduce", lambda _: reduce(work)):
        value = _build(normalized)
    value.update(
        experiment_id=9165,
        milestone="2026.11.780",
        random_seed=evidence.PROTOCOL["seed"],
        requested_inference_substrate_class="no_model_load",
        generated_tokens=0,
        verifier_is_oracle=False,
        claim_scope=dict(
            scope="Fourteen administrative outcomes. Scientific static and online predictions and complete costs remain unmeasured; measured validation remains blocked. No ARC solve, hardware benefit or general theorem.",
            exposure_scope="Original exposed LPHB900/300 development panel; no independent human confirmation.",
            independent_generalization_score=0,
        ),
        exposure_scope="exposed_development",
        sample_size_budget=dict(
            intended=14,
            attempted=14,
            completed=14,
            failed=0,
            censored=0,
            effective_independent_groups=0,
            static_test_qids=100,
            online_future_qids=40,
            repeated_retention_unique_qids=100,
        ),
        open_gaps={"GAP-ORACLE-DISTINCT": "open", "GAP-DETECTOR-AUROC-4208": "open"},
        staged_authority_status="V780 full ordered objects compared with active YAML and visible/embedded design",
        full_task_objects=work["tasks"],
        full_task_digest=evidence.canonical_hash(work["tasks"]),
        branch_readiness=[
            dict(
                task_id=r["task_id"],
                readiness=r.get("readiness", {}),
                verdict_class=r["verdict_class"],
                qualified=r["qualified"],
            )
            for r in value["task_outcome_rows"][:-1]
        ],
    )
    value["model_invocation_counts"].update(
        current_llm_invocations=0,
        loads_attempted=0,
        loads_completed=0,
        loads_failed=0,
        forward_calls=0,
        generated_tokens=0,
    )
    value["task_outcome_rows"][-1].update(
        experiment_id=9165,
        primary_path=f"results/{NAME}.json",
        primary_status="self_slot",
        scheduler_alternative=None,
        readiness=dict(reconciliation_ready_score=value["reconciliation_ready_score"]),
    )
    value["publication_gate"] = dict(
        **{k: value[k] for k in ["G1", "G2", "G3", "G4", "paper_ready", "unmet_gates"]},
        publication_scope="historical_FoVer",
        current_science_paper_ready=False,
    )
    for key, rows in bounds().items():
        value.setdefault(key, rows)
    for row in value["rows"]:
        row.update(qid=None, sibling=None)
    value["archive_lag"].update(
        current_milestone="2026.11.780",
        archive_lagged=value["archive_lag"]["observed_archive_end"] != "2026.11.780",
    )
    value["source_references"] = [
        dict(
            r,
            producer_path=r.get("original_path", r.get("source_path", r["path"])),
            imported_fields=["original_outcomes_and_per_unit_evidence"],
        )
        for r in value["source_references"]
    ]
    health = work["repository_health_references"]
    if health:
        receipt = json.loads(Path(health[0]["path"]).read_bytes())
        value["repository_health"].update(
            passed=receipt["passed"],
            timed_out=receipt["timed_out"],
            exit_code=receipt["exit_code"],
        )
    value["gate_failure_rows"] = [r for r in value["gate_check_summary"] if not r["passed"]]
    value.setdefault("science_decisions", dict(status="blocked_missing_authenticated_inputs"))
    value["field_principles"].update(
        {
            k: "Exact retained operands and executed checks establish accounting only; absent science remains blocked."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["methodology_note"] = (
        "Authenticate complete V780 objects and exact gate operands, then recompute human question groups and preserve all unfinished science and costs. No model load, generation, theorem, hardware run, ARC solve or external publication. Prototype correctness, administrative null, historical disqualification and unmeasured science retain separate scope."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from retained authenticated rows so a fresh hash cannot bless invented science."""
    with binding():
        return dict(_replay(path))


def main(argv: list[str] | None = None) -> int:
    """Publish only the candidate bytes accepted by actual terminal checker receipts."""
    with binding():
        return int(_main(argv))

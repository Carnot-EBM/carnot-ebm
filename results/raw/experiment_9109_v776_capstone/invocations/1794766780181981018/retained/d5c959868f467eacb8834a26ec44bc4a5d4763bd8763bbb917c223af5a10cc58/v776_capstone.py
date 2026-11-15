"""REQ-REPORT-9109-V776: reuse the checked capstone lifecycle with new evidence."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v775_capstone as legacy
from carnot.reporting import v776_capstone_evidence as evidence
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.encoder_head_diagnosis_8972 import checksum, reference

Json = dict[str, Any]
ROOT = evidence.ROOT
NAME, TASK = "experiment_9109_v776_capstone", "exp9109-capstone"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v776_capstone_9109.py"
OWNED = [
    "python/carnot/reporting/v776_capstone.py",
    "python/carnot/reporting/v776_capstone_evidence.py",
    CLI,
]
DIRECT = legacy.DIRECT
REUSED = list(
    dict.fromkeys([*legacy.REUSED, *legacy.OWNED, "python/carnot/reporting/qwen_judge_9107.py"])
)
MODEL_SPECS: list[Json] = []
_collect, _build, _replay, _main = legacy.collect, legacy.build, legacy.replay, legacy.main


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed boundaries and inherited child heartbeats make long work observable."""
    print(f"[exp9109] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Reuse the lifecycle without editing historical producers or their source pins."""
    with patch.multiple(
        legacy,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        DIRECT=DIRECT,
        REUSED=REUSED,
        evidence=evidence,
        progress=progress,
        HEALTH=ROOT / "results/raw" / NAME / "development/global_python_suite.receipt.json",
        collect=collect,
        build=build,
        replay=replay,
    ):
        yield


def collect(raw: Path, run_date: str, root: Path = ROOT) -> Json:
    """Freeze validation before row reduction and label the empty current ledger precisely."""
    with binding():
        work = _collect(raw, run_date, root)
    path = Path(work["current_work_reference"]["path"])
    current = json.loads(path.read_bytes())
    current["inference_substrate_details"]["scope"] = "V776 administrative reconciliation only"
    atomic_json(path, current)
    work["current_work_reference"] = reference(path)
    atomic_json(raw / "work.json", work)
    return work


def build(work: Json) -> Json:
    """Administrative completion cannot qualify absent or disqualified scientific evidence."""
    work = dict(work, values=dict(sorted(work["values"].items())))
    with binding():
        value = _build(work)
    value.update(
        experiment_id=9109,
        milestone="2026.11.776",
        random_seed=776109,
        sample_size_budget=dict(administrative_slots=14),
        verifier_is_oracle=False,
        claim_scope="Fourteen administrative outcomes and retained descriptive evidence only; no independent factuality, learned-head or causal benefit claim.",
        exposure_scope="LPHB is exposed development. The disqualified Qwen judge is diagnostic only. Graph executable-oracle and CPU susceptibility results remain historical.",
        publication_scope="historical_FoVer",
        current_science_paper_ready=False,
        open_gaps={"GAP-ORACLE-DISTINCT": "open", "GAP-DETECTOR-AUROC-4208": "open"},
        staged_authority_status="V776 task objects from immutable Exp9096; historical authorities are never resolved through live vNEXT",
        scientific_scope_ledger=[
            dict(
                branch=r["branch"],
                claim_scope=r["next_prerequisite"],
                scientific_readiness=0,
                independent_generalization_score=0,
            )
            for r in value.get("evidence_table", [])
        ],
    )
    self_row = value["task_outcome_rows"][-1]
    self_row.update(
        experiment_id=9109,
        primary_path=f"results/{NAME}.json",
        primary_status="self_slot",
        scheduler_alternative=None,
        readiness=dict(reconciliation_ready_score=value["reconciliation_ready_score"]),
        failed_gate_value=None,
    )
    value["archive_lag"].update(
        current_milestone="2026.11.776",
        archive_lagged=value["archive_lag"]["observed_archive_end"] != "2026.11.776",
    )
    value["gate_failure_rows"] = [r for r in value["gate_check_summary"] if not r["passed"]]
    health = work["repository_health_references"]
    if health:
        receipt = json.loads(Path(health[0]["path"]).read_bytes())
        value["repository_health"].update(
            passed=receipt["passed"], timed_out=receipt["timed_out"], exit_code=receipt["exit_code"]
        )
    for row in value["rows"]:
        row["original_qid"] = None
    value["methodology_note"] = (
        "Recompute original human groups and diagnostic judge metrics from retained rows. Missing scientific branches stay terminal blocked. Qualified source custody and CPU methods do not imply useful verification. Preserve prior nulls and hardware limits. No model load, generation, gameplay, roadmap activation or publication occurs."
    )
    value["field_principles"].update(
        {
            "task_outcome_rows": "Conductor slots are administrative units; a scheduler alternative cannot replace the declared scientific primary.",
            "gate_failure_rows": "Exact failed operands distinguish missing_file, missing_field, null and observed zero.",
            "reconciliation_ready_score": "All owned checks may qualify complete accounting even when scientific prerequisites are blocked.",
            "independent_generalization_score": "Exposed development, sibling answers and fixture repetitions establish no independent generalization.",
            "judge_summary": "A disqualified producer may supply diagnostic rows but cannot support an accepted factuality comparison.",
            "retirement_recommendations": "Retire only the declared unchanged scope after comparing actual prior verdict and changed mechanism.",
            "sample_size_budget": "Fourteen task slots are administrative units, not fourteen independent scientific samples.",
            "current_work_receipts": "The current empty event ledger stays separate from historical model loads and evaluations.",
            "publication_scope": "G1-G4 concern historical FoVer publication and do not qualify this milestone's new science.",
            "duration_s": "Monotonic elapsed time measures current accounting work, not historical inference time.",
            "code_config_hashes": "Frozen source hashes bind validation to the implementation that produced this accounting.",
            "owned_statement_coverage": "Measured execution must cover every new statement with no exclusions.",
            "MODEL_SPECS": "An empty current model list forbids treating historical model use as new inference.",
            "inference_substrate_class": "No model load is needed to reconcile retained bytes on the CPU.",
            "current_llm_invocations": "Only current owned model events count; cited prior calls cannot fill this zero.",
            "model_invocation_counts": "Attempted and terminal current events must balance independently of historical provenance.",
            "source_references": "Exact byte hashes preserve original authority without repinning historical producers.",
            "validation_receipts": "Actual command exits and retained log hashes establish validation, not asserted flags.",
            "required_checks_passed": "Failed affected checks disqualify readiness; unrelated repository health stays separate.",
            "acceptance_gates": "Administrative completion and scientific readiness have different prerequisites.",
            "random_seed": "A fixed identifier records the frozen protocol; seeds add no independent samples.",
            "reproducibility_checksum": "A complete content hash detects changed fields; retained-row replay also rejects rehashed inventions.",
            "rows": "Every administrative slot retains its own identity, denominator and censor state.",
            "effective_independent_groups": "Repeated fixtures and administrative slots establish zero independent scientific groups.",
            "verifier_is_oracle": "Human factual labels are not a cheap executable oracle; historical graph rows retain their own oracle tags.",
            "evidence_table": "Each branch keeps its own observed scope and exact next prerequisite; missing measurements stay null.",
            "scientific_scope_ledger": "Custody, fixtures, exposed development and historical mechanisms have separate claim boundaries.",
            "current_science_paper_ready": "Historical publication gates cannot qualify this milestone's unavailable new science.",
            "open_gaps": "Exposed data and diagnostic improvements cannot close independent oracle-distinct verification gaps.",
            "archive_lag": "Scheduling progress does not authorize archive or roadmap activation.",
            "preserved_cpu_reference": "Recomputation of retained CPU rows is not a new physical measurement.",
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Cold replay reduces pinned rows and rejects altered aggregates even after rehashing."""
    with binding():
        return _replay(path)


def main(argv: list[str] | None = None) -> int:
    """Keep atomic publication behind the existing strict validators and time limits."""
    with binding():
        return _main(argv)

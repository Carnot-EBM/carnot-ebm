"""REQ-REPORT-8621: qualify a fresh invocation without rewriting blocked history.

The old reducer remains the authority for numerical and process behavior. This
adapter adds explicit current permission; it cannot change what old evidence meant.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import coverage

from carnot.reporting import lean_qualification_8609 as previous
from carnot.reporting import v740_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import progress

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8621_v740_lean_resident_qualification"
TASK = "exp8621-lean-resident-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_lean_qualification_8621.py", "tests/python/test_lean_runner_8621.py"]
OWNED = [
    f"python/carnot/reporting/lean_{s}_8621.py" for s in ["qualification", "runner", "validation"]
] + [CLI]
MODEL_SPECS: list[Json] = []
METHODS_REFERENCE = dict(
    path=str(ROOT / "results/experiment_8615_v740_contract_methods.json"),
    sha256="sha256:6f7ea729a61997543357bc926f558510d0dbfdb95cabcd42aecbce936267209e",
)
REQUIRED = previous.REQUIRED | {"current_methods", "historical_substitution_rejected"}


def producer_map() -> Json:
    """Use the immutable registration, never filenames inferred from future outputs."""
    return dict(methods.protocol()["producer_map_v740"])


def bind_current(work: Json, reference: Json, producers: Json) -> Json:
    """An independently registered invocation must match all old scientific operands.

    The invocation identity and fresh clocks prevent treating a blocked old measurement
    as a new experiment merely by changing its methods reference.
    """
    if work.get("invocation_task_id") != TASK or reference != METHODS_REFERENCE:
        raise ValueError("historical_methods_substitution")
    current = json.loads(previous.authenticate(reference).read_bytes())
    published, _ = previous.upstream(ROOT, methods.NAME)
    protocol = methods.protocol()
    historical, _ = previous.upstream(ROOT, previous.NAME)
    prior = json.loads(previous.authenticate(historical["work_reference"]).read_bytes())
    if (
        current != published
        or previous.reference(ROOT / "results" / (previous.NAME + ".json"))
        != protocol["original_producers"]["8609"]
        or current["methods_protocol_reference"]
        != dict(path=str(ROOT / methods.PROTOCOL), sha256=methods.PROTOCOL_PIN)
        or current["methods_ready_score"] != 1
        or current["required_checks_passed"] is not True
        or producers != protocol["producer_map_v740"]
        or producers[TASK] != f"results/{NAME}.json"
        or current["lean_worker_2x2"] != protocol["lean_worker_2x2"]
        or work.get("registered_lean_methods") != protocol["lean_worker_2x2"]
        or work["target_model"] != historical["target_model"]
        or work["started_ns"] <= prior["ended_ns"]
        or work["raw"] == prior["raw"]
    ):
        raise ValueError("current_methods_or_operand_drift")
    if work["sessions"]:
        actual = json.loads(previous.authenticate(work["config_reference"]).read_bytes())
        original = json.loads(previous.authenticate(prior["config_reference"]).read_bytes())
        if actual != original:
            raise ValueError("head_scaler_temperature_build_operands")
    return dict(
        methods_reference=reference,
        producer_map=producers,
        protocol_reference=current["methods_protocol_reference"],
        operands_equal=True,
        historical_reference=previous.reference(ROOT / "results" / (previous.NAME + ".json")),
    )


def measure(raw: Path, run_date: str, *, methods_reference: Json = METHODS_REFERENCE) -> Json:
    """Run original real workers afresh, then bind explicit current authority separately."""
    progress("8621_start_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    work: Json = previous.measure(raw, run_date)
    work.update(
        raw=str(raw),
        invocation_task_id=TASK,
        current_methods_reference=methods_reference,
        current_gate_check_summary=[],
        historical_replay_rejection_rows=[],
    )
    raw.mkdir(parents=True, exist_ok=True)
    history, refs = previous.upstream(ROOT, previous.NAME)
    old_work = json.loads(previous.authenticate(history["work_reference"]).read_bytes())
    try:
        bind_current(old_work, methods_reference, producer_map())
    except ValueError as error:
        work["historical_replay_rejection_rows"].append(
            dict(
                unit_id="old8609-current8615-substitution",
                passed=True,
                error=str(error),
                original_readiness=history["lean_measurement_ready_score"],
                historical_methods="Exp8602",
                attempted_methods=methods_reference,
            )
        )
    work["sources"].extend(refs)
    try:
        previous.authenticate(methods_reference)
        work["current_binding"] = bind_current(work, methods_reference, producer_map())
        work["sources"].extend([methods_reference, work["current_binding"]["protocol_reference"]])
    except (OSError, ValueError, KeyError) as error:
        work["current_gate_check_summary"].append(
            dict(
                check="current_methods",
                upstream=methods.TASK,
                **methods_reference,
                field="methods_ready_score",
                operator="==",
                expected=1,
                observed="missing" if isinstance(error, OSError) else str(error),
            )
        )
    work["code_config_hashes"].extend(previous.reference(ROOT / p) for p in OWNED + TESTS)
    atomic_json(raw / "work.json", work)
    progress("8621_fresh_primitives_sealed")
    return work


def artifact(work: Json, receipts: list[Json], output: Path) -> Json:
    """Grant permission for future clocks only when current authority and mechanics pass."""
    value: Json = previous.artifact(work, receipts, output)
    gates = work["current_gate_check_summary"] + [
        g
        for g in work["gate_check_summary"]
        if g["upstream"] != "experiment_8602_v739_contract_methods"
    ]
    passed = value["required_checks_passed"]
    ready = int(
        passed
        and not gates
        and bool(work["parity_rows"])
        and work.get("child_statement_coverage", {}).get("passed") is True
        and work.get("original_122_node_union", {}).get("passed") is True
        and REQUIRED <= {r.get("name") for r in receipts if r["passed"]}
    )
    verdict = (
        "disqualified"
        if not passed
        else "blocked"
        if gates
        else "null"
        if ready
        else "disqualified"
    )
    value.update(
        experiment_id=8621,
        task_id=TASK,
        milestone="2026.10.740",
        honest_verdict=f"complete_{verdict}_v740_lean_resident_qualification",
        verdict_class=verdict,
        gate_check_summary=gates,
        lean_measurement_ready_score=ready,
        branch_gate_values=dict(lean_measurement_ready_score=ready),
        current_methods_reference=work["current_methods_reference"],
        current_binding=work.get("current_binding"),
        historical_replay_rejection_rows=work["historical_replay_rejection_rows"],
        original_122_node_union=work.get("original_122_node_union", {}),
        historical_gate_check_summary=work["gate_check_summary"],
        methodology_note="Fresh four-arm mechanics under immutable Exp8615; Exp8609 stays blocked at zero. No benchmark or learner change.",
    )
    value["field_principles"].update(
        {
            k: "Current invocation binding; historical Exp8602/8609 pins and zero readiness remain unchanged."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = previous.checksum(value)
    return value


def coverage_matches(proof: Json) -> bool:
    """Use Coverage.py's executable statements, preserving continuation-line semantics.

    The raw database is retained alongside the human-readable report so a rehashed
    summary cannot invent executed statements or require a continuation as a statement.
    """
    refs = proof["data_references"]
    for ref in refs:
        previous.authenticate(ref)
    aggregate = next(ref for ref in refs if Path(ref["path"]).name == ".coverage")
    recorded = json.loads(previous.authenticate(proof["report"]).read_bytes())
    cov = coverage.Coverage(data_file=aggregate["path"], config_file=False)
    cov.set_option("report:exclude_lines", [])
    cov.load()
    for filename, row in recorded["files"].items():
        _, statements, _, missing, _ = cov.analysis2(str(ROOT / filename))
        if missing != row["missing_lines"] or len(statements) != row["summary"]["num_statements"]:
            return False
    measured = {str((ROOT / p).resolve()): row for p, row in recorded["files"].items()}
    owned = {str((ROOT / p).resolve()) for p in OWNED}
    passed = owned <= set(measured) and all(
        not measured[p]["missing_lines"] and measured[p]["summary"]["num_statements"] > 0
        for p in owned
    )
    return bool(recorded["files"] == proof["files"] and passed == proof["passed"])


def replay(path: Path) -> None:
    """Reconstruct both meanings: historical mechanics and independent V740 authority.

    The temporary old-format view deliberately keeps its Exp8602 blockage. Only the
    new-format reduction can use current authority, after independent binding checks.
    """
    value = json.loads(path.read_bytes())
    if value.get("reproducibility_checksum") != previous.checksum(value):
        raise ValueError("artifact_checksum")
    work = json.loads(previous.authenticate(value["work_reference"]).read_bytes())
    if work.get("child_statement_coverage") and not coverage_matches(
        work["child_statement_coverage"]
    ):
        raise ValueError("raw_child_statement_coverage_drift")
    with TemporaryDirectory(prefix="carnot8621-cold-", dir="/var/tmp") as scratch:
        mechanical = Path(scratch) / (previous.NAME + ".json")
        atomic_json(mechanical, previous.artifact(work, value["validation_receipts"], mechanical))
        previous.replay(mechanical)
    if not work["current_gate_check_summary"]:
        binding = bind_current(work, work["current_methods_reference"], producer_map())
        if binding != work["current_binding"] or not work["historical_replay_rejection_rows"]:
            raise ValueError("current_binding_drift")
    if "original_122_node_union" in work:
        original, _ = previous.upstream(ROOT, previous.NAME)
        union = work["original_122_node_union"]
        if (
            union["collected"] != original["original_node_union"]["collected"]
            or len(union["collected"]) != 122
        ):
            raise ValueError("original_122_node_union_drift")
    if artifact(work, value["validation_receipts"], Path(value["publication_output"])) != value:
        raise ValueError("reduction_drift")
    progress("8621_cold_replay_passed")

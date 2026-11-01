"""REQ-REPORT-8372: preserve missing source bytes and the independent board obligation.

A reopening request records work for an external operator. It cannot authenticate
unavailable history or replace an actual device transcript with a passing score.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import kv260_local_cost_boundary_8315 as base
from carnot.reporting import v721_contract_methods as authority
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.v717_contract_methods import PIN as PROTOCOL_PIN

Json = dict[str, Any]
ROOT = base.ROOT
NAME = "experiment_8372_v721_gatemate_missing_evidence"
TASK = "exp8372-gatemate-missing-evidence"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_gatemate_missing_evidence_8372.py"
OWNED = [
    "python/carnot/reporting/gatemate_missing_evidence_8372.py",
    "python/carnot/reporting/gatemate_missing_runner_8372.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:07324350f091226c5ca14cd88395d95f506e63bd270d95304e9edefe87cb33ff"
HISTORY = "results/experiment_8357_v720_gatemate_change_ledger.json"
HISTORY_PIN = "sha256:fe3e29d775966a50bef90040dbb64a0a492ef913d76ebf32389316af3ae35071"
FINAL = "results/experiment_8358_v720_terminal_replay_qualification.json"
FINAL_PIN = "sha256:fc405e9f7ea05883a886d4b0211b25643f0dc993bb3ede6c30d523365615d8ee"
SEALS = {
    HISTORY_PIN: (
        "sha256:45c165a82e5bfa39a9f42701ba216f42bef4309a10bd8ad04eee611dafc5dfc4",
        "sha256:dbc8723970fad720debe612489f414194c215cc0a4c8df226546f1ccc51e4fd4",
    ),
    FINAL_PIN: (
        "sha256:b077aeeabb83f8c02cc7737f5710c2952711621be25494e6a54800785dbe69ae",
        "sha256:95959eabd4b2d3e568521304f82d714845eb226416b3eae0b6688c74e3ccf7f1",
    ),
}
TRANSCRIPT = "results/experiment_6559_gatemate_changed_state_continuity.json"
TRANSCRIPT_PIN = "sha256:59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66"
REOPEN = "docs/research-notes/v721-gatemate-reopen-condition.md"
RECEIPT_SCHEMA = "carnot.gatemate.supplied_evidence.v1"
MISSING = [
    dict(
        path="python/carnot/reporting/gatemate_change_ledger_8344.py",
        sha256="sha256:4034850aa978b9fc3f63fe960ad2c586069a46eee826923cc9cfae94bca73f03",
    ),
    dict(
        path="tests/python/test_gatemate_change_ledger_8344.py",
        sha256="sha256:72a713565f099e2885ecb73c72c44305270bf950e89a6079ae2c9a6acde206e6",
    ),
]
PHYSICAL = [
    "dated operator cable/port/power change",
    "GM1Ax IDCODE 0x20000001",
    "authenticated n16 bitstream flash",
    "device sample/hash smoke parity",
]
EXTERNAL = ["operator_supplied_snapshot", "producer_archive", "operator_physical_receipt"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts because external evidence inspection must not look like a stalled job."""
    print(f"[exp8372] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """A null operand records absence; zero remains an actual observed value."""
    return dict(
        upstream=path.stem,
        path=str(path),
        hash=base.sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


failure = gate


def sealed(path: Path, pin: str, raw: Path, work: Json) -> Json:
    """Authenticate terminal custody without replaying historical producer code."""
    if base.sha256_file(path) != pin:
        raise ValueError("historical_primary_hash")
    value: Json = json.loads(base.pin(path, raw, work["refs"]).read_bytes())
    validate_primary(value, path)
    terminal = Path(value["terminal_validation_sidecar_path"])
    receipt = json.loads(base.pin(terminal, raw, work["refs"]).read_bytes())
    publication = receipt["publication"]
    sidecar = Path(publication["sidecar_path"])
    report = read_bound_sidecar(path, sidecar)
    base.pin(sidecar, raw, work["refs"])
    if (
        publication["primary_sha256"] != pin
        or publication["primary_path"] != str(path.absolute())
        or report["primary_path"] != str(path.absolute())
        or report["report"]["passed"] is not True
        or (base.sha256_file(terminal), base.sha256_file(sidecar)) != SEALS[pin]
    ):
        raise ValueError("historical_terminal_custody")
    return value


def read_failure(root: Path, raw: Path, work: Json) -> Json:
    """Import the authenticated failure once; unchanged history is never searched again."""
    value = sealed(root / HISTORY, HISTORY_PIN, raw, work)
    missing = [
        dict(path=str(Path(r["path"]).relative_to(ROOT)), sha256=r["sha256"])
        for r in value["history_authentication"]["missing_hashes"]
    ]
    if value["history_authentication"]["passed"] is not False or missing != MISSING:
        raise ValueError("historical_missing_hashes")
    producer = value["history_authentication"]["bundles"][1]
    return dict(
        missing=missing,
        producer_receipts=producer["rows"][:3],
        historical_model_provenance=value["historical_model_provenance"],
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
    )


def consume_receipt(path: Path | None, raw: Path, missing: list[Json]) -> list[Json]:
    """Only explicitly supplied fresh receipts are read; private positives never become natural evidence."""
    if path is None:
        return []
    refs: list[Json] = []
    try:
        value = json.loads(base.pin(path, raw, refs).read_bytes())
        if (
            value["schema"] != RECEIPT_SCHEMA
            or value["upstream_primary_sha256"] != HISTORY_PIN
            or not "20261009" < value["received_date"] <= "20261010"
            or value["authorized_location"] not in EXTERNAL
        ):
            raise ValueError("receipt_authority_or_frontier")
        expected = {r["path"]: r["sha256"] for r in missing}
        rows = []
        seen = set()
        for source in value["source_rows"]:
            original = source["original_path"]
            if original in seen or expected.get(original) != source["sha256"]:
                raise ValueError("receipt_source_membership")
            seen.add(original)
            snap = base.pin(base.checked(source), raw, refs)
            rows.append(
                dict(
                    disposition="supplied_exact_bytes",
                    original_path=original,
                    sha256=source["sha256"],
                    reference=base.reference(snap),
                )
            )
        physical = value.get("physical_change")
        if physical is not None:
            if (
                not "20261009" < physical["date"] <= "20261010"
                or not physical["operator"]
                or not physical["changed_fields"]
                or not set(physical["changed_fields"])
                <= {"cable", "port", "power", "board", "dirtyjtag"}
                or physical["original_transcript_sha256"] != TRANSCRIPT_PIN
            ):
                raise ValueError("physical_receipt_contract")
            rows.append(dict(disposition="queued_next_hardware_task", physical_change=physical))
        return [dict(r, receipt_reference=refs[0]) for r in rows]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [dict(disposition="rejected", path=str(path), reason=str(error), references=refs)]


def measure(root: Path, raw: Path, supplied: Path | None = None) -> Json:
    """Freeze current authority and old failure receipts before recording any readiness."""
    work: Json = dict(
        root=str(root.absolute()),
        refs=[],
        checks=[],
        history={},
        final={},
        contract={},
        supplied_evidence_rows=[],
        started_monotonic_ns=time.monotonic_ns(),
    )
    raw.mkdir(parents=True, exist_ok=True)
    progress("authority_before", 0, 3)
    try:
        for name in [authority.DESIGN, authority.ACTIVE, authority.legacy.PROTOCOL]:
            base.pin(root / name, raw, work["refs"])
        actual = authority.authority(root, raw / "authority")
        task = next(t for t in actual["tasks"] if t["id"] == TASK)
        work["contract"] = dict(
            activated=actual["activated"],
            task_sha256=canonical_hash(task),
            canonical_tasks_sha256=actual["canonical_tasks_sha256"],
        )
        for field, expected, observed in [
            ("authority.activated", True, actual["activated"]),
            ("full_task_sha256", TASK_PIN, canonical_hash(task)),
            ("protocol.sha256", PROTOCOL_PIN, base.sha256_file(root / authority.legacy.PROTOCOL)),
        ]:
            work["checks"].append(gate(root / authority.DESIGN, field, expected, observed))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        work["checks"].append(
            gate(root / authority.DESIGN, "authority", "activated V721 task", str(error))
        )
    progress("authenticated_failure_before", 1, 2)
    try:
        work["history"] = read_failure(root, raw, work)
        final = sealed(root / FINAL, FINAL_PIN, raw, work)
        work["final"] = {
            k: final[k] for k in ["honest_verdict", "verdict_class", "required_checks_passed"]
        }
        transcript = base.pin(root / TRANSCRIPT, raw, work["refs"])
        work["checks"].append(
            gate(
                root / TRANSCRIPT,
                "original_transcript.sha256",
                TRANSCRIPT_PIN,
                base.sha256_file(transcript),
            )
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        work["checks"].append(
            gate(root / HISTORY, "sealed_failure", "authenticated V720 failure", str(error))
        )
    progress("authenticated_failure_after", 2, 1)
    work["supplied_evidence_rows"] = consume_receipt(supplied, raw, MISSING)
    if supplied is not None and supplied.is_file():
        base.pin(supplied, raw, work["refs"])
    for row in work["supplied_evidence_rows"]:
        if row["disposition"] == "rejected":
            work["checks"].append(
                gate(
                    supplied or raw,
                    "supplied_receipt.authentication",
                    "fresh exact authenticated evidence",
                    row["reason"],
                )
            )
    for missing in MISSING:
        present = next(
            (
                r
                for r in work["supplied_evidence_rows"]
                if r.get("original_path") == missing["path"]
            ),
            None,
        )
        work["checks"].append(
            gate(
                root / missing["path"],
                "newly_supplied_source.sha256",
                missing["sha256"],
                present["sha256"] if present else None,
            )
        )
    physical = next(
        (
            r
            for r in work["supplied_evidence_rows"]
            if r["disposition"] == "queued_next_hardware_task"
        ),
        None,
    )
    work["checks"].append(
        gate(
            supplied or root / HISTORY,
            "new_physical_change_receipt",
            "queued_next_hardware_task",
            physical["disposition"] if physical else None,
        )
    )
    request = dict(
        schema="carnot.gatemate.reopening_request.v1",
        task_id=TASK,
        upstream_primary_sha256=HISTORY_PIN,
        missing_source_hashes=MISSING,
        producer_receipts=work["history"].get("producer_receipts", []),
        authorized_external_evidence_locations=EXTERNAL,
        explicitly_supplied_receipt=str(supplied) if supplied else None,
        prior_frontier_date="20261009",
        no_repeated_history_search=True,
        planning_search_result="No matching bytes in reachable commits for either file; no search repeated.",
        required_next_evidence=PHYSICAL,
        original_transcript_sha256=TRANSCRIPT_PIN,
        history_replay_required_in_future=True,
        execution_ready_score=0,
    )
    request_path = raw / "requests" / (canonical_hash(request)[7:] + ".json")
    atomic_json(request_path, request)
    work["reopening_request_reference"] = base.reference(request_path)
    work["code_config_hashes"] = []
    for name in [
        *OWNED,
        "python/carnot/reporting/primary_publication.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "python/carnot/reporting/v721_contract_methods.py",
        "python/carnot/reporting/v709_execution.py",
        "ops/exclusion_manifest.yaml",
        REOPEN,
    ]:
        frozen: list[Json] = []
        base.pin(ROOT / name, raw / "code", frozen)
        work["code_config_hashes"].extend(frozen)
    work["ended_monotonic_ns"] = time.monotonic_ns()
    progress("external_obligation_recorded", 3, 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """An authenticated obligation scores documentation only; unavailable history remains blocked."""
    owned = (
        bool(receipts) and all(r["passed"] for r in receipts) and work.get("preflight_passed", True)
    )
    recorded = bool(
        work["history"]
        and work["contract"].get("activated")
        and work["contract"].get("task_sha256") == TASK_PIN
        and all(
            c["passed"]
            for c in work["checks"]
            if c["artifact_field"] in {"protocol.sha256", "original_transcript.sha256"}
        )
    )
    kind = "blocked" if owned else "disqualified"
    rows = [
        dict(
            source_id="gatemate_reopening_obligation",
            arm="read_only_obligation",
            intended=1,
            completed=recorded,
            failed=not owned,
            censored=not recorded,
            excluded=False,
            independent=0,
            numerator=int(recorded),
            denominator=1,
        )
    ]
    for missing in MISSING:
        present = any(
            r.get("original_path") == missing["path"] for r in work["supplied_evidence_rows"]
        )
        rows.append(
            dict(
                source_id=missing["path"],
                arm="newly_supplied_exact_source",
                intended=1,
                completed=present,
                failed=False,
                censored=not present,
                excluded=False,
                independent=0,
                numerator=int(present),
                denominator=1,
            )
        )
    rows.append(
        dict(
            source_id="gatemate_physical_execution",
            arm="future_hardware_obligation",
            intended=1,
            completed=False,
            failed=False,
            censored=True,
            excluded=False,
            independent=0,
            numerator=0,
            denominator=1,
        )
    )
    value: Json = dict(
        experiment_id=8372,
        task_id=TASK,
        milestone="2026.10.721",
        run_date="20261010",
        honest_verdict="complete_" + kind + "_gatemate_missing_evidence",
        verdict_class=kind,
        gate_check_summary=[
            *work["checks"],
            dict(
                upstream="exp8357-gatemate-change-ledger",
                path=str(Path(work["root"]) / HISTORY),
                hash=HISTORY_PIN,
                artifact_field="history_authentication",
                op="==",
                expected=True,
                observed=False,
                passed=False,
            ),
            dict(
                upstream="gatemate_physical_setup",
                path=str(Path(work["root"]) / TRANSCRIPT),
                hash=TRANSCRIPT_PIN,
                artifact_field="future_physical_execution",
                op="==",
                expected=True,
                observed=None,
                passed=False,
            ),
        ],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["history"].get("historical_model_provenance", []),
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["completed"] for r in rows),
        failed_count=sum(r["failed"] for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            obligation=1,
            missing_sources=2,
            physical_execution=1,
            timing_repeats_are_independent=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed_development_and_constructed_private_controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates=dict(
            owned_checks=owned,
            obligation_recorded=recorded,
            history_authenticated=False,
            hardware_execution=False,
            scientific_benefit=False,
        ),
        validation_receipts=receipts,
        adversarial_findings=work.get("adversarial_findings", []),
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        preconditions_checked=True,
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(
                phase="authenticate_and_record",
                started_monotonic_ns=work["started_monotonic_ns"],
                ended_monotonic_ns=work["ended_monotonic_ns"],
            )
        ],
        random_seed=7218372,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[
            base.reference(raw / "measurement.json"),
            work["reopening_request_reference"],
        ],
        cited_upstream_artifacts=[
            dict(r, imported_fields=["authority, terminal custody or historical failure"])
            for r in work["refs"]
        ],
        obligation_recorded_score=int(recorded and owned),
        history_authentication=False,
        missing_source_hashes=MISSING,
        supplied_evidence_rows=work["supplied_evidence_rows"],
        physical_change_frontier=dict(
            prior_frontier_date="20261009",
            inspection="explicit_new_receipts_only",
            required_next_evidence=PHYSICAL,
            original_idcode="0xffffffff",
            future_idcode="0x20000001",
            current_execution=False,
        ),
        execution_ready_score=0,
        current_jtag_retry_count=0,
        original_transcript_sha256=TRANSCRIPT_PIN,
        reopen_contract_path=REOPEN,
        reopening_request_reference=work["reopening_request_reference"],
        work_reference=base.reference(raw / "measurement.json"),
        publication_output=str(output),
        historical_dispositions=dict(exp8357=work["history"], exp8358=work["final"]),
        methodology="Authenticate sealed historical failure once; record exact missing bytes and independent physical obligation; inspect explicit fresh receipts only; no model, full history replay, JTAG or flash.",
    )
    value["field_principles"] = {
        k: "Bind this field to frozen primitives; recording evidence does not grant execution or scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        field_principles="Explain why each field is required for an auditable read-only obligation.",
        reproducibility_checksum="Bind the complete reduced record; checksums do not establish semantic correctness.",
        obligation_recorded_score="Score a correct authenticated obligation, independently of unavailable source recovery.",
        execution_ready_score="A read-only request cannot qualify device execution.",
        history_authentication="Missing exact producer bytes keep historical authentication false.",
        supplied_evidence_rows="Keep rejected evidence and queue physical receipts without executing them.",
        historical_model_provenance="Imported observations are separate from this invocation's zero LLM calls.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rehashed claims must still match frozen authority and sealed historical meaning."""
    try:
        value = json.loads(path.read_bytes())
        for ref in [
            value["work_reference"],
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            base.checked(ref)
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        original = {r["original_path"]: r for r in work["refs"]}
        root = Path(work["root"])
        for ref in work["code_config_hashes"]:
            if base.sha256_file(Path(ref["original_path"])) != ref["sha256"]:
                return False
        for row in work["supplied_evidence_rows"]:
            refs = [
                *row.get("references", []),
                *([row["receipt_reference"]] if "receipt_reference" in row else []),
            ]
            for ref in refs:
                base.checked(ref)
            if "receipt_reference" in row:
                receipt = json.loads(base.checked(row["receipt_reference"]).read_bytes())
                if (
                    receipt["schema"] != RECEIPT_SCHEMA
                    or receipt["upstream_primary_sha256"] != HISTORY_PIN
                    or not "20261009" < receipt["received_date"] <= "20261010"
                    or receipt["authorized_location"] not in EXTERNAL
                ):
                    return False
            if row["disposition"] == "supplied_exact_bytes":
                base.checked(row["reference"])
                if (
                    row["reference"]["sha256"] != row["sha256"]
                    or not any(
                        r["path"] == row["original_path"] and r["sha256"] == row["sha256"]
                        for r in MISSING
                    )
                    or not any(
                        r["original_path"] == row["original_path"] and r["sha256"] == row["sha256"]
                        for r in receipt["source_rows"]
                    )
                ):
                    return False
            if row["disposition"] == "queued_next_hardware_task":
                physical = row["physical_change"]
                if (
                    physical != receipt["physical_change"]
                    or not "20261009" < physical["date"] <= "20261010"
                    or not physical["operator"]
                    or not physical["changed_fields"]
                    or not set(physical["changed_fields"])
                    <= {"cable", "port", "power", "board", "dirtyjtag"}
                    or physical["original_transcript_sha256"] != TRANSCRIPT_PIN
                ):
                    return False
        if work["contract"]:
            with TemporaryDirectory(prefix="exp8372-cold-", dir="/var/tmp") as directory:
                private = Path(directory)
                for name in [authority.DESIGN, authority.ACTIVE]:
                    output = private / name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(base.checked(original[str(root / name)]).read_bytes())
                actual = authority.authority(private, private / "authority")
                task = next(t for t in actual["tasks"] if t["id"] == TASK)
                if (
                    work["contract"]
                    != dict(
                        activated=actual["activated"],
                        task_sha256=canonical_hash(task),
                        canonical_tasks_sha256=actual["canonical_tasks_sha256"],
                    )
                    or canonical_hash(task) != TASK_PIN
                    or original[str(root / authority.legacy.PROTOCOL)]["sha256"] != PROTOCOL_PIN
                ):
                    return False
        if work["history"]:
            ref = original[str(root / HISTORY)]
            history = json.loads(base.checked(ref).read_bytes())
            if (
                ref["sha256"] != HISTORY_PIN
                or history["history_authentication"]["passed"] is not False
            ):
                return False
            missing = [
                dict(path=str(Path(r["path"]).relative_to(ROOT)), sha256=r["sha256"])
                for r in history["history_authentication"]["missing_hashes"]
            ]
            expected = dict(
                missing=missing,
                producer_receipts=history["history_authentication"]["bundles"][1]["rows"][:3],
                historical_model_provenance=history["historical_model_provenance"],
                honest_verdict=history["honest_verdict"],
                verdict_class=history["verdict_class"],
            )
            if missing != MISSING or work["history"] != expected:
                return False
            for name, pin in [(HISTORY, HISTORY_PIN), (FINAL, FINAL_PIN)]:
                primary = json.loads(base.checked(original[str(root / name)]).read_bytes())
                terminal = json.loads(
                    base.checked(original[primary["terminal_validation_sidecar_path"]]).read_bytes()
                )
                publication = terminal["publication"]
                side = json.loads(base.checked(original[publication["sidecar_path"]]).read_bytes())
                if (
                    original[str(root / name)]["sha256"] != pin
                    or publication["primary_sha256"] != pin
                    or side["primary_sha256"] != pin
                    or side["report"]["passed"] is not True
                    or (
                        original[primary["terminal_validation_sidecar_path"]]["sha256"],
                        original[publication["sidecar_path"]]["sha256"],
                    )
                    != SEALS[pin]
                ):
                    return False
            final = json.loads(base.checked(original[str(root / FINAL)]).read_bytes())
            if work["final"] != {
                k: final[k] for k in ["honest_verdict", "verdict_class", "required_checks_passed"]
            }:
                return False
            if original[str(root / TRANSCRIPT)]["sha256"] != TRANSCRIPT_PIN:
                return False
        request = json.loads(base.checked(work["reopening_request_reference"]).read_bytes())
        if (
            request["missing_source_hashes"] != MISSING
            or request["required_next_evidence"] != PHYSICAL
            or request["original_transcript_sha256"] != TRANSCRIPT_PIN
            or request["upstream_primary_sha256"] != HISTORY_PIN
            or request["producer_receipts"] != work["history"].get("producer_receipts", [])
        ):
            return False
        for field in ["execution_manifest_reference", "owned_coverage_reference"]:
            if work.get(field):
                base.checked(work[field])
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if stream + "_path" in receipt:
                    base.checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    )
        raw = Path(value["work_reference"]["path"]).parent
        return bool(
            value
            == build(work, value["validation_receipts"], raw, Path(value["publication_output"]))
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False

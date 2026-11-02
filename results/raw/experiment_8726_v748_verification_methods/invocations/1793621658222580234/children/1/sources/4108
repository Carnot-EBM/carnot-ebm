"""REQ-VERIFY-8453: original publication bytes keep their original authority.

A private restored publication gives readers checked bytes without changing the
flagged historical primary. Its validation uses the original producer's CLI.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import arc_supervisor_support_evidence_8439 as evidence
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v726_feature_panel_runner import scan

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "scripts/experiments/experiment_8412_v724_arc_panel_qualification.py"


def recover(root: Path, raw: Path, private: Path) -> Json:
    """Only the named sealed candidate is inspected; the current hash is never a seal."""
    started = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    receipts: list[Json] = []
    findings: list[Json] = []
    try:
        source = evidence.producer(root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(
            passed=False, external_missing=True, reason=str(error), receipts=[], findings=[]
        )
    current_path = root / evidence.UPSTREAM
    current = json.loads(current_path.read_bytes()) if current_path.is_file() else {}
    flags = {
        role: {key: value.get(key) for key in ["flagged_adversarial", "adversarial_findings"]}
        for role, value in [("original", source), ("current", current)]
    }
    inspection_duration = time.monotonic() - started
    cli = [sys.executable, "-u", str(CLI), "--cold-replay"]
    for name in ["original", "missing", "forged", "rehashed"]:
        value = deepcopy(source)
        if name == "forged":
            value["arc_panel_ready_score"] = 999
        if name == "rehashed":
            work = evidence.read(value["work_reference"])
            work["rows"][0]["action_count"] += 1
            operand = raw / "rehashed_work.json"
            atomic_json(operand, work)
            value["work_reference"] = dict(path=str(operand), sha256=sha256_file(operand))
            value["reproducibility_checksum"] = canonical_hash(work)
        path = raw / (name + ".json")
        if name != "missing":
            atomic_json(path, value)
        receipts.append(
            child(
                name,
                [*cli, str(path)],
                raw / "controls",
                expected=int(name != "original"),
                deadline=20,
            )
        )
    restored = private / "restored" / Path(evidence.UPSTREAM).name

    def validator(path: Path) -> Json:
        """Exact historical replay and both unchanged scanners qualify restored bytes."""
        receipts.append(child("restored_cold", [*cli, str(path)], raw / "validators", deadline=20))
        checks, flags = scan(path, raw / "validators")
        receipts.extend(checks)
        findings.extend(flags)
        return dict(
            passed=all(r["passed"] for r in receipts) and not findings,
            findings=findings,
            receipts=list(receipts),
        )

    try:
        publication = publish_primary(restored, source, validator)
        # Retain the checked bytes and report after temporary private scratch expires.
        retained = raw / restored.name
        retained.write_bytes(restored.read_bytes())
        proof = json.loads(Path(publication["sidecar_path"]).read_bytes())
        atomic_json(raw / "restored_validation.json", proof)
        consumer = reader_receipt(source["task_id"], restored.parent, field="arc_panel_ready_score")
        passed = consumer["passed"] and sha256_file(retained) == evidence.CANDIDATE_PIN
    except ValueError as error:
        publication, consumer, passed = dict(error=str(error)), {}, False
    return dict(
        passed=passed,
        external_missing=False,
        original_validated_hash=evidence.CANDIDATE_PIN,
        current_observed_hash=sha256_file(current_path) if current_path.is_file() else None,
        flags=flags,
        publication=publication,
        retained_path=str(raw / restored.name),
        proof_path=str(raw / "restored_validation.json"),
        consumer=consumer,
        receipts=receipts,
        findings=findings,
        inspection_cap_s=120,
        inspection_duration_s=inspection_duration,
        duration_s=time.monotonic() - started,
        original_authority_references=evidence.read(source["work_reference"])[
            "source_input_receipts"
        ],
    )


def replay(custody: Json) -> bool:
    """A self-rehashed custody claim must still have genuine restored bytes and checks."""
    if custody["external_missing"]:
        return custody["passed"] is False and custody["receipts"] == []
    for receipt in custody["receipts"]:
        saved = json.loads(Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes())
        if saved != receipt:
            return False
        for stream in ["stdout", "stderr"]:
            if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                return False
    if not custody["passed"]:
        return not all(r["passed"] for r in custody["receipts"])
    proof = json.loads(Path(custody["proof_path"]).read_bytes())
    return bool(
        sha256_file(Path(custody["retained_path"])) == evidence.CANDIDATE_PIN
        and custody["original_validated_hash"] == evidence.CANDIDATE_PIN
        and proof["primary_sha256"] == evidence.CANDIDATE_PIN
        and proof["report"]["passed"]
        and proof["report"]["receipts"] == custody["receipts"]
        and not custody["findings"]
        and all(r["passed"] for r in custody["receipts"])
    )

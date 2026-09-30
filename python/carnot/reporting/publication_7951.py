"""REQ-REPORT-7951: seal terminal bytes and verify both actual primary readers.

Final validators bind a candidate hash. Nested sidecars cannot replace the
primary selected by the unchanged gate and document readers.
"""

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7951_v689_hardware_evidence as q
from carnot.reporting import qualification_7926 as runner
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt


def qualify(root: Path, run_date: str, output: Path, raw_root: Path) -> int:
    """Reuse qualified supervision, then bind final publication to actual readers."""
    from carnot.reporting import validation_7951 as plan
    from carnot.reporting.experiment_7913_v686_hardware_evidence import progress

    started = time.monotonic()
    output = output.absolute()
    publication_raw = output.parent / "raw" / output.stem
    publication_raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="carnot-7951-publish-", dir="/tmp") as folder:
        private = Path(folder)
        terminal = plan.terminal_manifest(private)
        for spec in terminal:
            spec["argv"] = [
                arg.replace(
                    str(private / "terminal-candidate.json"),
                    str(publication_raw / "terminal_candidate.json"),
                )
                for arg in spec["argv"]
            ]
        atomic_json(
            publication_raw / "publication_command_manifest.json",
            {
                "task_id": "exp7951-hardware-evidence",
                "commands": terminal,
                "dependency_hashes": {
                    name: sha256_file(root / name)
                    for name in (*plan.MEASURED, *plan.TESTS, *plan.LIBRARIES)
                },
                "coverage_includes": plan.INCLUDE,
            },
        )
        runner.qualify(
            root,
            run_date,
            private / "qualified.json",
            raw_root,
            evidence=q,
            validation=plan,
        )
        value = json.loads((private / "qualified.json").read_text())
        receipt_path = publication_raw / "primary_resolution_receipt.json"
        terminal_path = publication_raw / "terminal_validation_reports.json"
        value["primary_resolution_receipt"] = {
            "path": str(receipt_path),
            "binding": "exact final primary bytes",
        }
        value["terminal_validation_sidecar_path"] = str(terminal_path)
        value["duration_s"] = time.monotonic() - started
        value["phase_spans"] = [
            {"phase": "custody_and_owned_validation", "start_s": 0.0, "end_s": value["duration_s"]}
        ]
        value["duration_scope"] = (
            "Monotonic work until final candidate freeze; final validator durations are in its bound sidecar."
        )
        value["field_principles"]["duration_scope"] = (
            "Keep final validation time separate from the already frozen candidate."
        )

        def validator(candidate: Path) -> dict[str, Any]:
            progress(started, "final_validation", "before_cold_reduction", 0)
            reduced = q.cold_reduce(root, json.loads(candidate.read_text()))
            reports = [
                runner.run_child(spec, private, publication_raw / "terminal_logs", started, index)
                for index, spec in enumerate(terminal)
            ]
            flagged = json.loads(Path(reports[0]["log_path"]).read_text())["flagged_count"]
            passed = not flagged and all(r["passed"] for r in reports)
            report = {
                "passed": passed,
                "flagged_adversarial": bool(flagged),
                "candidate_sha256": sha256_file(candidate),
                "reports": reports,
                "cold_reduction": reduced,
            }
            atomic_json(terminal_path, report)
            progress(started, "final_validation", "after_validators", len(reports))
            return report

        published = publish_primary(output, value, validator)
        challenge = publication_raw / "reader_mtime_challenge.json"
        atomic_json(
            challenge,
            {
                "primary_sha256": published["primary_sha256"],
                "scope": "nested sidecar selection check",
            },
        )
        stamp = max(time.time_ns(), output.stat().st_mtime_ns + 1)
        os.utime(challenge, ns=(stamp, stamp))
        receipt = reader_receipt(
            value["task_id"],
            output.parent,
            field="hardware_evidence_ready_score",
            expected=value["hardware_evidence_ready_score"],
        )
        if (
            not receipt["passed"]
            or receipt["gate_path"] != str(output)
            or receipt["gate_sha256"] != published["primary_sha256"]
        ):
            raise ValueError("reader_identity")
        receipt["readiness_open"] = value["hardware_evidence_ready_score"] == 1
        receipt["newer_sidecar_path"] = str(challenge)
        receipt["primary_resolution"] = published
        atomic_json(receipt_path, receipt)
    progress(started, "final", "published_checked_bytes_and_reader_hashes", 3)
    return 0

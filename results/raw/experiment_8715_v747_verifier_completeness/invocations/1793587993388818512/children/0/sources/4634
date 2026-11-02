"""REQ-REPORT-8248: small adapters reuse qualified execution and publication.

The inherited parent already records clocks, bounded subprocess receipts,
coverage and atomic terminal publication. This adapter binds its current task.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import patch

from carnot.reporting import decision_margin_runner_8234 as inherited
from carnot.reporting import evidence_intervention_methods_8248 as q
from carnot.reporting import v685_authority_lifecycle as authority
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.v710_contract_replay import failure, require_reference
from carnot.verify.sentence_transport_methods_8179 import tokenizer
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
INHERITED_COMMANDS = inherited.commands


def commands(private: Path) -> list[Json]:
    """Scope coverage to current statements while retaining both private E2E checks."""
    with patch.object(inherited, "q", q):
        return list(INHERITED_COMMANDS(private))


def replay(path: Path) -> Json:
    """Rebuild source views and authority rows rather than trusting claimed totals."""
    v = json.loads(path.read_bytes())
    for ref in [
        v["work_reference"],
        *v["source_artifact_hashes"],
        *v["code_config_hashes"],
        *v["raw_shard_hashes"],
    ]:
        require_reference(ref)
    for receipt in v["validation_receipts"]:
        for stream in ["stdout", "stderr"]:
            if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                raise ValueError("receipt_drift")
    w = json.loads(Path(v["work_reference"]["path"]).read_bytes())
    if v["reproducibility_checksum"] != canonical_hash(
        [v["work_reference"], v["code_config_hashes"], q.PIN]
    ):
        raise ValueError("checksum_drift")
    if w["protocol"] != q.PROTOCOL_VALUE or v["protocol_sha256"] != q.PIN:
        raise ValueError("protocol_drift")
    refs = {r["path"]: r for r in w["refs"]}
    expected_failures = []
    for spec in [
        dict(path=str(q.ROOT / q.PROTOCOL), sha256=q.PIN),
        *q.PROTOCOL_VALUE["source_artifact_hashes"],
    ]:
        operand = Path(w["root"]) / Path(spec["path"]).relative_to(q.ROOT)
        observed = refs[str(operand)]["sha256"]
        if observed != spec["sha256"]:
            expected_failures.append(failure(operand, "sha256", spec["sha256"], observed, observed))
    for name in q.INPUTS:
        operand = Path(w["root"]) / name
        if not refs[str(operand)]["exists"] and name != q.ACTIVE:
            expected_failures.append(failure(operand, "exists", True, None))
    recorded = [f for f in w["failures"] if f["artifact_field"] in {"sha256", "exists"}]
    if recorded != expected_failures:
        raise ValueError("source_gate_drift")
    with tempfile.TemporaryDirectory(prefix="carnot8248-replay-") as directory:
        raw = Path(directory)
        snapshots = w["contract"]["authority_snapshots"]
        if snapshots:
            paths = [
                Path(snapshots[k].get("snapshot_path", raw / ("absent-" + k)))
                for k in ["design", "staged", "active"]
            ]
            contract = authority.assess_authorities(
                *paths, raw, milestone=q.MILESTONE, first_id=8248, count=14
            )
            for key in ["activated", "contract_rows", "canonical_tasks_sha256"]:
                if contract[key] != w["contract"][key]:
                    raise ValueError("authority_drift")
        if not w["failures"]:
            read = lambda ref: json.loads(
                Path(
                    refs[str(Path(w["root"]) / Path(ref["path"]).relative_to(q.ROOT))][
                        "snapshot_path"
                    ]
                ).read_bytes()
            )
            count, receipt = tokenizer(dict(checks=[]), fixture=v["fixture_protocol_only"])
            if (
                receipt != w["tokenizer_receipt"]
                or q.primitives(read, w["protocol"], count) != w["public_rows"]
            ):
                raise ValueError("source_primitive_drift")
            if json.loads(Path(w["label_reference"]["path"]).read_bytes()) != q.sealed_labels(
                read, w["protocol"]
            ):
                raise ValueError("label_seal_drift")
    rebuilt = q.reduce(w, v["validation_receipts"])
    for key, observed in rebuilt.items():
        if v[key] != observed:
            raise ValueError("primitive_reduction_drift:" + key)
    return dict(passed=True, rows_checksum=canonical_hash(rebuilt["rows"]))


def normalize(value: Json) -> Json:
    """Bind the current hypothesis seed and explain the added evidence fields."""
    value["random_seed"] = 7138248
    value["field_principles"].update(
        intervention_protocol_ready_score="Authenticated frozen methods and owned checks; no measured benefit.",
        source_view_rows="Complete original source slots, lexical choices and full answer bytes; targets remain sealed.",
        feature_definitions="Model sensitivity predictions need independent human targets; no entailment certificate.",
        continuous_mechanism="Delayed released Beta counts use coarse reusable keys and global backoff.",
        role_manifest_hashes="Original source hashes separate fit, calibration, selection, stream and retention.",
    )
    return dict(normalize_artifact_for_template_write(value))


def main(argv: list[str] | None = None) -> int:
    """Reuse the qualified bounded parent without changing the historical producer."""
    with (
        patch.object(inherited, "q", q),
        patch.object(inherited, "commands", commands),
        patch.object(inherited, "replay", replay),
        patch.object(inherited, "normalize_artifact_for_template_write", normalize),
    ):
        return int(inherited.main(argv))

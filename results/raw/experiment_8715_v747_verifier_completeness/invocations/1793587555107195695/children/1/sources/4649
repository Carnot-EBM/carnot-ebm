"""Reusable source-feature extraction and cold reconstruction.

Every group remains visible when the structural grammar cannot check its prose.
Spec: REQ-REPORT-7646 and SCENARIO-REPORT-7646-COVERAGE.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file
from carnot.verify.source_claim_witness import parse_numbered_blocks, verify_claim

PROTOCOL = Path("results/raw/experiment_7602_v664_evidence_requalification/protocol.json")
ARMS = ("original_source", "evidence_erasure", "within_role_derangement")


def digest(text: str) -> str:
    """Bind a source or answer to its exact UTF-8 bytes before extraction."""

    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def validate_model_row(row: dict[str, Any]) -> None:
    """Reject label access and byte drift before a witness sees predictor text."""

    if row.get("labels_accessible", False) is not False or any(
        ("label" in key.lower() and key != "labels_accessible") or "evaluator" in key.lower()
        for key in row
    ):
        raise ValueError("label_access_rejected")
    source, answer = row["complete_source"], row["complete_answer"]
    if row["source_sha256"] != digest(source):
        raise ValueError("source_hash_mismatch")
    if row["answer_sha256"] != digest(answer):
        raise ValueError("answer_hash_mismatch")
    data = answer.encode()
    for sentence in row["answer_sentences"]:
        start, end = sentence["byte_start"], sentence["byte_end"]
        if data[start:end] != sentence["text"].encode() or sentence["text_sha256"] != digest(
            sentence["text"]
        ):
            raise ValueError("answer_offset_mismatch")


def extract_role(rows: list[dict], role: str, derangement: dict[str, str]) -> list[dict]:
    """Keep every inherited group in each arm, including unsupported prose."""

    groups = [row["component_hash"] for row in rows]
    if len(groups) != len(set(groups)) or any(row["role"] != role for row in rows):
        raise ValueError("duplicate_or_wrong_role")
    if set(derangement) != set(groups) or set(derangement.values()) != set(groups):
        raise ValueError("derangement_roster_mismatch")
    lookup = {row["component_hash"]: row for row in rows}
    results = []
    for row in rows:
        validate_model_row(row)
        group = row["component_hash"]
        for arm in ARMS:
            source_row = row if arm == "original_source" else lookup[derangement[group]]
            source = "" if arm == "evidence_erasure" else source_row["complete_source"]
            witnesses = [
                verify_claim(source, sentence["text"]) for sentence in row["answer_sentences"]
            ]
            blocks = parse_numbered_blocks(source)
            complete_blocks = sum(bool(block["complete"] and block["closed"]) for block in blocks)
            checked = sum(witness["status"] != "unknown" for witness in witnesses)
            unknown = len(witnesses) - checked
            results.append(
                {
                    "unit_id": group,
                    "feature_unit_id": digest(
                        group + ":" + row["source_sha256"] + ":" + row["answer_sha256"]
                    ),
                    "role": role,
                    "partition": row["learning_partition"],
                    "arm": arm,
                    "source_group_id": None
                    if arm == "evidence_erasure"
                    else source_row["component_hash"],
                    "source_sha256": digest(source),
                    "original_source_sha256": row["source_sha256"],
                    "answer_sha256": row["answer_sha256"],
                    "sentence_offsets": [
                        [part["byte_start"], part["byte_end"]] for part in row["answer_sentences"]
                    ],
                    "witnesses": witnesses,
                    "syntactic_parse_coverage": complete_blocks,
                    "source_scope_count": len(blocks),
                    "checked_predicates": checked,
                    "unknown_predicates": unknown,
                    "unchecked_prose": sum(
                        witness["residual_unverified_span"] or witness["status"] == "unknown"
                        for witness in witnesses
                    ),
                    "absolute_metric": checked / len(witnesses) if witnesses else 0.0,
                    "numerator": checked,
                    "denominator": len(witnesses),
                    "raw_provenance": str(PROTOCOL.parent / f"{role}_model_inputs.jsonl"),
                    "excluded": False,
                    "censored": unknown == len(witnesses),
                    "dataset_hallucination_label": None,
                }
            )
    return results


def read_jsonl(path: Path) -> list[dict]:
    """Read immutable line records without interpreting evaluator labels."""

    return [json.loads(line) for line in path.read_text().splitlines()]


def immutable_jsonl(path: Path, rows: list[dict]) -> str:
    """Write each completed unit once and reject later byte changes."""

    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("immutable_feature_drift")
    else:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.replace(path)
    return sha256_file(path)


def cold_reconstruct(manifest: dict, root: Path) -> dict:
    """Recompute every source offset and witness from frozen predictor bytes."""

    observed = 0
    all_groups: set[str] = set()
    for role, info in manifest["roles"].items():
        input_path, feature_path = Path(info["input_path"]), Path(info["feature_path"])
        if not input_path.is_absolute():
            input_path = root / input_path
        if not feature_path.is_absolute():
            feature_path = root / feature_path
        if (
            sha256_file(input_path) != info["input_sha256"]
            or sha256_file(feature_path) != info["feature_sha256"]
        ):
            raise ValueError("sidecar_hash_mismatch")
        inputs = read_jsonl(input_path)
        groups = [row["component_hash"] for row in inputs]
        if (
            groups != info["group_ids"]
            or len(groups) != len(set(groups))
            or any(group in all_groups for group in groups)
        ):
            raise ValueError("duplicate_or_altered_group")
        all_groups.update(groups)
        mapping = {group: manifest["derangement"][group] for group in groups}
        expected = extract_role(inputs, role, mapping)
        if read_jsonl(feature_path) != expected:
            raise ValueError("feature_reconstruction_mismatch")
        observed += len(groups)
    return {"passed": True, "groups": observed, "rows": observed * len(ARMS)}



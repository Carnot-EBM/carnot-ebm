"""Qualify exposed source bytes and evidence views (REQ-REPORT-7768)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

from carnot.experiment_7727_v673_development_corpus import COUNTS, validate_public
from carnot import experiment_7740_v674_sentence_label_protocol as prior
from carnot import experiment_7754_v675_sentence_protocol as custody
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.verify import evidence_views as views

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7768_v676_source_view_qualification"
RAW = ROOT / "results/raw" / NAME
OUTPUT = ROOT / "results" / f"{NAME}.json"
MANIFEST = ROOT / "results/raw/experiment_7727_v673_development_corpus/development_manifest.json"
SCOPE = RAW / "frozen_affected_scope.json"
WRAPPER = f"scripts/experiments/{NAME}.py"
PRINCIPLES = {
    **custody.PRINCIPLES,
    "evidence_view_ready_score": "Fitting needs qualified bytes and features.",
    "source_view_manifest_path": "Consumers must load the same records.",
}


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Report real work at each boundary so a silent child remains visible."""
    print(
        f"[exp7768] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def prepare_basetemp(parent: Path) -> None:
    """Pytest removes its leaf, so the nested parent must already exist."""
    parent.mkdir(parents=True, exist_ok=True)


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    """Seal records with deterministic UTF-8 bytes for independent replay."""
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return sha256_file(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _check(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    return {
        "upstream_id": upstream,
        "artifact_path": str(path),
        "artifact_hash": sha256_file(path) if path.is_file() else None,
        "field": field,
        "expected": expected,
        "observed": observed,
        "operator": "==",
        "passed": expected == observed,
    }


def preflight(root: Path) -> list[dict[str, Any]]:
    """Check the declared producer and conductor receipt as separate inputs."""
    producer = root / "results/experiment_7727_v673_development_corpus.json"
    manifest = (
        root / "results/raw/experiment_7727_v673_development_corpus/development_manifest.json"
    )
    gate = root / "results/experiment_7753_v675_contract_methods.json"
    checks = [
        _check("exp7727", producer, "is_file", True, producer.is_file()),
        _check("exp7727", manifest, "is_file", True, manifest.is_file()),
        _check("exp7753_conductor_pre_gate", gate, "is_file", True, gate.is_file()),
    ]
    if producer.is_file():
        source = json.loads(producer.read_text())
        checks.extend(
            (
                _check(
                    "exp7727",
                    producer,
                    "development_cohort_ready_score",
                    1,
                    source.get("development_cohort_ready_score"),
                ),
                _check(
                    "exp7727",
                    producer,
                    "development_manifest_sha256",
                    sha256_file(manifest) if manifest.is_file() else None,
                    source.get("development_manifest_sha256"),
                ),
            )
        )
    if gate.is_file():
        checks.append(
            _check(
                "exp7753_conductor_pre_gate",
                gate,
                "contract_ready_score",
                1,
                json.loads(gate.read_text()).get("contract_ready_score"),
            )
        )
    if manifest.is_file():
        inventory = json.loads(manifest.read_text())
        checks.append(_check("exp7727", manifest, "counts", COUNTS, inventory.get("counts")))
        for role in COUNTS:
            meta = inventory.get("roles", {}).get(role, {})
            checks.append(
                _check("exp7727", manifest, f"{role}.count", COUNTS[role], meta.get("count"))
            )
            for kind in ("public", "evaluator"):
                path = manifest.parent / meta.get(f"{kind}_path", f"missing_{role}_{kind}")
                checks.append(
                    _check(
                        "exp7727",
                        path,
                        f"{role}.{kind}_sha256",
                        meta.get(f"{kind}_sha256"),
                        sha256_file(path) if path.is_file() else None,
                    )
                )
    return checks


def qualify_views(source: bytes, answer: bytes) -> dict[str, dict[str, Any]]:
    """Use the registered complete-byte builder for both source layouts."""
    return views.prepare_views(source, answer)


def checked_targets(answer: bytes, annotations: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Accept exact repeated marks but reject conflicting overlapping spans."""
    mapped = custody.map_byte_targets(answer, annotations)
    spans = sorted((item["start"], item["end"], item["text"]) for item in annotations or [])
    duplicates = 0
    for left, right in zip(spans, spans[1:], strict=False):
        if right[0] < left[1]:
            if right == left:
                duplicates += 1
            else:
                raise ValueError("overlapping_annotation_spans")
    return {**mapped, "duplicate_span_count": duplicates}


def arm_decisions(pair: dict[str, dict[str, Any]], label: int | None) -> dict[str, dict[str, Any]]:
    """Retain every registered arm while leaving fitted benefit unmeasured."""
    if not pair["a"]["abstention"]:
        return {arm: {"action": None, "probability_unsupported": None} for arm in views.ARMS}
    return {
        arm: {key: decision[key] for key in ("action", "probability_unsupported")}
        for arm in views.ARMS
        for decision in [views.decision(arm, None, None, 1.0, label)]
    }


def prepare_public(row: dict[str, Any]) -> dict[str, Any]:
    """Prepare role and view data from authenticated public bytes alone."""
    if "label" in row or "annotations" in row:
        raise ValueError("unauthorized_public_label")
    validate_public(row, row["role"])
    pair = qualify_views(row["complete_source"].encode(), row["complete_response"].encode())
    serialized = json.loads(json.dumps(views.serialize_pair(pair)))
    return {
        "family_id": row["family_id"],
        "role": row["role"],
        "source_sha256": row["source_sha256"],
        "response_sha256": row["response_sha256"],
        "prior_exposure": row["previously_exposed"],
        "fresh_generalization_eligible": row["fresh_generalization_eligible"],
        "view_a": serialized["a"],
        "view_b": serialized["b"],
        "arms": arm_decisions(pair, None),
        "abstention": pair["a"]["abstention"],
    }


def map_evaluator(public: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    """Bind response labels to the same answer bytes after public sealing."""
    if (
        set(label) != {"family_id", "response_id", "annotations", "label"}
        or label["family_id"] != public["family_id"]
        or label["response_id"] != public["response_id"]
    ):
        raise ValueError("evaluator_join")
    mapped = checked_targets(public["complete_response"].encode(), label["annotations"])
    expected = int(any(not bool(item.get("implicit_true")) for item in label["annotations"]))
    if label["label"] != expected:
        raise ValueError("response_label_mapping")
    return {
        "family_id": public["family_id"],
        "role": public["role"],
        "response_sha256": public["response_sha256"],
        "response_label": expected,
        "sentence_targets": mapped["targets"],
        "sentence_byte_offsets": mapped["sentence_byte_offsets"],
        "annotation_byte_offsets": mapped["annotation_byte_offsets"],
        "duplicate_span_count": mapped["duplicate_span_count"],
        "reason": mapped["reason"],
    }


def prepare_corpus(
    manifest_path: Path, raw: Path, counts: dict[str, int] = COUNTS
) -> dict[str, Any]:
    """Seal all public roles and views before opening evaluator shards."""
    public, manifest, _ = prior.authenticate(manifest_path, counts)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    last = started
    for row in public:
        rows.append(prepare_public(row))
        if time.monotonic() - last >= 30:
            progress(started, "prepare", "heartbeat", len(rows))
            last = time.monotonic()
    if len(rows) != sum(counts.values()):
        raise ValueError("family_count")
    rows_hash = _jsonl(raw / "rows.jsonl", rows)
    source_view_manifest = {
        "schema": "carnot.exp7768.source_view_manifest.v1",
        "development_manifest_path": str(manifest_path),
        "development_manifest_sha256": sha256_file(manifest_path),
        "role_counts": counts,
        "role_hashes": {
            role: {
                kind: manifest["roles"][role][f"{kind}_sha256"] for kind in ("public", "evaluator")
            }
            for role in counts
        },
        "evaluator_paths": {
            role: str(manifest_path.parent / manifest["roles"][role]["evaluator_path"])
            for role in counts
        },
        "rows_path": str(raw / "rows.jsonl"),
        "rows_sha256": rows_hash,
        "arms": views.ARMS,
        "window_policy": {
            "a": "singles_and_adjacent_triples",
            "b": "singles_and_adjacent_pairs",
            "null": "one_per_view",
            "max_windows": 128,
            "max_answer_units": 16,
        },
        "output_span_authority": "local_response_only",
        "prior_exposure": True,
        "fresh_generalization_eligible": False,
    }
    atomic_json(raw / "source_view_manifest.json", source_view_manifest)
    targets: list[dict[str, Any]] = []
    by_role = {
        role: {row["family_id"]: row for row in public if row["role"] == role} for role in counts
    }
    for role in counts:
        evaluator = _read_jsonl(manifest_path.parent / manifest["roles"][role]["evaluator_path"])
        if len(evaluator) != counts[role]:
            raise ValueError("evaluator_count")
        for label in evaluator:
            targets.append(map_evaluator(by_role[role][label["family_id"]], label))
        progress(started, "targets", role, len(targets))
    _jsonl(raw / "targets.jsonl", targets)
    os.chmod(raw / "targets.jsonl", 0o600)
    coverage = [
        {
            "family_id": target["family_id"],
            "role": target["role"],
            "mapping_eligible": target["reason"] == "mapped",
            "reason": target["reason"],
            "known_sentence_count": sum(value is not None for value in target["sentence_targets"]),
            "sentence_count": len(target["sentence_targets"]),
            "duplicate_span_count": target["duplicate_span_count"],
            "abstention": row["abstention"],
        }
        for row, target in zip(rows, targets, strict=True)
    ]
    return {
        "rows": rows,
        "annotation_coverage_rows": coverage,
        "targets": targets,
        "source_view_manifest_path": str(raw / "source_view_manifest.json"),
    }


def replay_corpus(
    manifest_path: Path, raw: Path, counts: dict[str, int] = COUNTS
) -> dict[str, Any]:
    """Recompute every record from authenticated shards in a fresh reader."""
    public, manifest, _ = prior.authenticate(manifest_path, counts)
    rows = _read_jsonl(raw / "rows.jsonl")
    targets = _read_jsonl(raw / "targets.jsonl")
    sealed = json.loads((raw / "source_view_manifest.json").read_text())
    if len(rows) != sum(counts.values()) or len(targets) != len(rows):
        raise ValueError("family_count")
    if (
        sealed["development_manifest_sha256"] != sha256_file(manifest_path)
        or sealed["rows_sha256"] != sha256_file(raw / "rows.jsonl")
        or sealed["role_counts"] != counts
    ):
        raise ValueError("manifest_or_rows_hash")
    expected_targets = {}
    for role in counts:
        for label in _read_jsonl(manifest_path.parent / manifest["roles"][role]["evaluator_path"]):
            expected_targets[label["family_id"]] = label
    started = time.monotonic()
    last = started
    for index, (public_row, row, target) in enumerate(zip(public, rows, targets, strict=True), 1):
        if row != prepare_public(public_row):
            raise ValueError("public_view_replay_mismatch")
        if target != map_evaluator(public_row, expected_targets[public_row["family_id"]]):
            raise ValueError("evaluator_replay_mismatch")
        if time.monotonic() - last >= 30:
            progress(started, "cold_replay", "heartbeat", index)
            last = time.monotonic()
    return {"families": len(rows), "roles": counts}

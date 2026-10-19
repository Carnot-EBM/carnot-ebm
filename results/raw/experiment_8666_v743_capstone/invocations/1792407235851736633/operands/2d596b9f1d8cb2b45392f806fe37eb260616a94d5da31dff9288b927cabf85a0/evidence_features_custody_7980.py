"""Bind public research inputs directly without reopening old fit gates.

REQ-REPORT-7980. History searches identify exposure; they cannot prove that
model pretraining or unrecorded research never saw a public source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file
from carnot.verify import evidence_features_7980 as features

Json = dict[str, Any]
INPUTS = {
    7968: (
        "results/experiment_7968_v691_response_role_targets.json",
        "exp7968-response-role-targets",
        "20261001",
        "2026.10.691",
        "response_roles_ready_score",
    ),
    7969: (
        "results/experiment_7969_v691_qwen_calibration_capture.json",
        "exp7969-qwen-calibration-capture",
        "20261001",
        "2026.10.691",
        "qwen_capture_ready_score",
    ),
    7958: (
        "results/experiment_7958_v690_qwen_response_risk.json",
        "exp7958-qwen-response-risk",
        "20261001",
        "2026.09.690",
        "qwen_response_measurement_ready_score",
    ),
    7972: (
        "results/experiment_7972_v691_qwen_energy_calibration.json",
        "exp7972-qwen-energy-calibration",
        "20261001",
        "2026.10.691",
        "qwen_calibration_ready_score",
    ),
}
PINS = {
    7968: "sha256:5978a0302945b4111afd06ee5f756ff0d8d810a3d84e2cf4fec4d88acb8f06d1",
    7969: "sha256:3f25b2e4b43d50536525e64ace321db55e9d6889aab97cc2581e4665014f2f51",
    7958: "sha256:3342ad4f66c5613482ac7da1da2b6f87099dfa3135647fa559b65748d7f62cad",
    7972: "sha256:a47d19ca7af500014117ff083249469597cd7e6bc0ff57778098fa39cfa41c63",
}


def reference(path: Path) -> Json:
    """Exact byte hashes make inherited provenance independently checkable."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def checked(item: Json) -> Path:
    """Reject missing evidence distinctly from changed or zero-valued evidence."""
    path = Path(item["path"])
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise ValueError(f"hash:{path}")
    return path


def operand(eid: Any, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """A reader needs actual operands to distinguish absence from a failed gate."""
    return dict(
        upstream_id=f"exp{eid}",
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Check only the four qualified producers and their declared byte custody."""
    upstream, checks, refs = {}, [], []
    for eid, (relative, task, date, milestone, readiness) in INPUTS.items():
        path = root / relative
        checks.append(
            operand(eid, path, "sha256", PINS[eid], sha256_file(path) if path.is_file() else None)
        )
        value = json.loads(path.read_text()) if checks[-1]["passed"] else {}
        expected = dict(
            experiment_id=eid,
            task_id=task,
            run_date=date,
            milestone=milestone,
            flagged_adversarial=False,
            **{readiness: 1},
        )
        checks.extend(operand(eid, path, k, v, value.get(k)) for k, v in expected.items())
        checks.append(
            operand(
                eid,
                path,
                "eligible_verdict",
                True,
                value.get("verdict_class") in {"null", "positive"},
            )
        )
        upstream[eid] = value
        if value:
            refs.append(
                dict(
                    reference(path),
                    producer_id=eid,
                    producer_invocation_date=value["run_date"],
                    imported_fields=[readiness, "rows", "public_role_manifests", "heads_seal"],
                )
            )
            items = list(value.get("raw_response_shards", []))
            for field in ("public_role_manifests", "evaluator_role_manifests"):
                items.extend(value.get(field, {}).values())
            if value.get("heads_seal"):
                items.append(value["heads_seal"])
            for item in items:
                p = Path(item["path"])
                checks.append(
                    operand(
                        eid, p, "sha256", item["sha256"], sha256_file(p) if p.is_file() else None
                    )
                )
                refs.append(item)
    for name in ("source_info.jsonl", "response.jsonl"):
        path = root / "data/ragtruth" / name
        checks.append(operand("official_train", path, "exists", True, path.is_file()))
        if path.is_file():
            refs.append(
                dict(
                    reference(path),
                    imported_fields=[
                        "TRAIN_source_text" if name.startswith("source") else "TRAIN_response_text"
                    ],
                    label_access="reserved_evaluator_child_only",
                )
            )
    return [r for r in checks if not r["passed"]], dict(
        upstream=upstream, checks=checks, references=refs
    )


def train_public(root: Path) -> tuple[list[Json], list[Json]]:
    """Project TRAIN text in a public process without looking up outcome fields."""
    responses = []
    with (root / "data/ragtruth/response.jsonl").open() as stream:
        for line in stream:
            if not re.search(r'"split"\s*:\s*"train"', line):
                continue
            row = json.loads(line)
            responses.append({k: row[k] for k in ("id", "source_id", "split", "response")})
    ids = {r["source_id"] for r in responses}
    sources = selected_sources(root / "data/ragtruth/source_info.jsonl", ids)
    return sources, responses


def selected_sources(path: Path, ids: set[str]) -> list[Json]:
    """Skip unrelated source records before decoding official TEST text."""
    result = []
    with path.open() as stream:
        for line in stream:
            match = re.search(r'"source_id"\s*:\s*"([^"\\]+)"', line)
            if match and match[1] in ids:
                result.append(json.loads(line))
    return result


def exposure(root: Path, owned: Path) -> tuple[set[str], set[str], Json]:
    """Scan all discoverable local research JSON captures without reading labels."""
    inventory, discoveries, ids, hashes, invalid = [], [], set(), set(), []
    roots = [root / p for p in ("results", "data", ".harness/captures", ".harness/manifests")]
    paths = sorted(
        {
            p
            for base in roots
            for p in base.rglob("*")
            if p.is_file()
            and p.suffix in {".json", ".jsonl", ".yaml", ".yml"}
            and not p.resolve().is_relative_to(owned.resolve())
            and not p.resolve().is_relative_to((root / "data/ragtruth").resolve())
        }
    )
    started = time.monotonic()
    last_progress = started
    for index, path in enumerate(paths):
        digest = hashlib.sha256()
        found_ids, found_hashes = set(), set()
        invalid_hex = []
        with path.open("rb") as stream:
            for line in stream:
                now = time.monotonic()
                if now - last_progress >= 30:
                    print(
                        f"[exp7980-exposure] active_file={path} scanned={index}/{len(paths)} elapsed_s={now - started:.3f}",
                        flush=True,
                    )
                    last_progress = now
                digest.update(line)
                for match in re.finditer(rb'"(?:source_id|source_group)"\s*:\s*"([^"\\]+)"', line):
                    value = match[1].decode()
                    found_ids.add(value.removeprefix("ragtruth:").removeprefix("source:"))
                for match in re.finditer(rb'"source_bytes"\s*:\s*"([a-fA-F0-9]*)"', line):
                    try:
                        found_hashes.add(features.normalized(bytes.fromhex(match[1].decode())))
                    except ValueError as error:
                        invalid_hex.append(
                            dict(
                                raw_hex_sha256=hashlib.sha256(match[1]).hexdigest(),
                                error=str(error),
                            )
                        )
                for match in re.finditer(
                    rb'"(?:source_cluster_id|source_sha256|normalized_source_hash|source_normalized_hash)"\s*:\s*"(?:sha256:)?([a-f0-9]{64})"',
                    line,
                ):
                    found_hashes.add(match[1].decode())
        item = dict(path=str(path.absolute()), sha256="sha256:" + digest.hexdigest())
        inventory.append(item)
        invalid.extend(dict(item, **record) for record in invalid_hex)
        if found_ids or found_hashes:
            discoveries.append(
                dict(item, source_ids=sorted(found_ids), source_hashes=sorted(found_hashes))
            )
        ids.update(found_ids)
        hashes.update(found_hashes)
        if index % 128 == 0:
            print(
                f"[exp7980-exposure] scanned={index + 1}/{len(paths)} elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    return (
        ids,
        hashes,
        dict(
            search_roots=[str(p) for p in roots],
            search_inventory=inventory,
            discovered_exclusions=discoveries,
            invalid_source_byte_records=invalid,
            complete_history_custody=False,
            custody_limit="Local discoverable records cannot exclude unrecorded prior access.",
        ),
    )


def q_rows(upstream: dict[int, Json], public: list[Json]) -> list[Json]:
    """Absent cached probability stays absent and never becomes a human target."""
    indexed = {}
    for eid in (7969, 7958):
        for row in upstream[eid]["rows"]:
            if eid == 7958 and row.get("arm") != "full_source":
                continue
            q = row.get("parsed", {}).get("probability") if row["status"] == "completed" else None
            if q is not None and (not isinstance(q, (int, float)) or not 0 <= q <= 1):
                raise ValueError("q_value")
            fid = row["family_id"]
            if fid in indexed:
                raise ValueError("q_duplicate")
            indexed[fid] = dict(q=q, origin=f"exp{eid}")
    return [
        dict(family_id=r["family_id"], **indexed.get(r["family_id"], dict(q=None, origin=None)))
        for r in public
    ]


def reserved_join(root: Path, seal_path: Path, output: Path) -> None:
    """Only this evaluator child opens selected annotations after public sealing."""
    from carnot.reporting.current_work_receipt import atomic_json
    from carnot.verify import response_targets_7955 as targets
    from carnot.verify.sentence_labels_7942 import digest

    seal = json.loads(seal_path.read_text())
    public = json.loads(checked(seal["public"]).read_text())["request_rows"]
    checked(seal["features"])
    roster = json.loads(checked(seal["roster"]).read_text())["rows"]
    selected = {r["response_id"] for r in roster}
    responses = []
    with (root / "data/ragtruth/response.jsonl").open() as stream:
        for line in stream:
            if not re.search(r'"split"\s*:\s*"train"', line):
                continue
            row = json.loads(line)
            if row["id"] in selected:
                responses.append(row)
    sources = selected_sources(
        root / "data/ragtruth/source_info.jsonl", {r["source_id"] for r in responses}
    )
    roles = [
        dict(
            family_id=r["family_id"],
            role="reserved_development",
            status="completed",
            source_cluster_id=digest(bytes.fromhex(p["source_bytes"])),
        )
        for r, p in zip(roster, public, strict=True)
    ]
    evaluators = [
        dict(family_id=r["family_id"], response_id=r["response_id"], role="reserved_development")
        for r in roster
    ]
    frozen, audit = targets.freeze(public)
    rows, annotations = targets.join(
        frozen,
        audit,
        dict(roles=roles, evaluators=evaluators, responses=responses, sources=sources),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.parent.chmod(0o700)
    atomic_json(
        output,
        dict(
            rows=rows,
            annotation_rows=annotations,
            access_policy=dict(consumer="exp7983", requires="policy_seal", predictor_access=False),
            public_seal_sha256=sha256_file(seal_path),
        ),
    )
    output.chmod(0o600)

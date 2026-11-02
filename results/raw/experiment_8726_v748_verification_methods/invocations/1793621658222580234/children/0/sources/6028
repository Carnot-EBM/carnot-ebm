"""Reuse qualified numerical callables with current custody (REQ-VERIFY-7930-V688)."""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import threading
import time
from types import ModuleType
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)
from carnot.reporting import primary_publication as publication
from carnot.verify import energy_fit_7894 as custody
from carnot.verify import evidence_views, natural_training, training_runtime
from scripts.experiments import experiment_7894_v685_energy_fit as prior

ROOT = prior.ROOT
ARMS, SEEDS = prior.ARMS, prior.SEEDS
OWNED = (
    "python/carnot/verify/energy_fit_7930.py",
    "python/carnot/verify/energy_fit_7930_run.py",
    "scripts/experiments/experiment_7930_v688_energy_fit.py",
)
RUNTIME = ROOT / "results/experiment_7916_v687_training_qualification.json"
UPSTREAM = prior.UPSTREAM
START = time.monotonic()


def external_json(path: Path) -> dict[str, Any]:
    """Unavailable or malformed upstream bytes are external blocks, never owned work."""
    try:
        value: dict[str, Any] = json.loads(path.read_text())
        return value
    except (ValueError, OSError) as exc:
        row = custody.operand(path, "artifact.valid_json", "==", True, str(exc))
        row["upstream_id"] = (
            "exp7916-training-qualification"
            if "runtime" in path.name or "7916" in path.name
            else "exp7892-source-boundary"
        )
        raise custody.InputBlocked([row]) from exc


def progress(phase: str, event: str = "boundary", units: int = 0) -> None:
    """Measured counters distinguish ongoing numerical work from a stalled child."""
    print(
        f"[exp7930] phase={phase} event={event} completed={units} elapsed_s={time.monotonic() - START:.3f}",
        flush=True,
    )


def authenticate(
    primary: Path, upstream: Path
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Read readiness from primary bytes; validation only attests their identity."""

    def require(path: Path, field: str, expected: Any, observed: Any) -> None:
        if expected != observed:
            row = custody.operand(path, field, "==", expected, observed)
            row["upstream_id"] = "exp7916-training-qualification"
            raise custody.InputBlocked([row])

    require(primary, "artifact.exists", True, primary.is_file())
    artifact = external_json(primary)
    for field, expected in (
        ("experiment_id", 7916),
        ("training_runtime_ready_score", 1),
        ("flagged_adversarial", False),
    ):
        require(primary, field, expected, artifact.get(field))
    require(
        primary,
        "terminal_eligible",
        True,
        artifact.get("verdict_class") in ("positive", "circular_positive", "null")
        and str(artifact.get("honest_verdict", "")).startswith("complete_"),
    )
    sidecar = Path(artifact["terminal_validation_sidecar_path"])
    require(primary, "sidecar.exists", True, sidecar.is_file())
    attestation = json.loads(sidecar.read_text())
    require(
        primary,
        "sidecar.candidate_sha256",
        sha256_file(primary),
        attestation.get("candidate_sha256"),
    )
    require(
        primary,
        "sidecar.validation",
        True,
        bool(attestation.get("receipts"))
        and all(row.get("passed") is True for row in attestation["receipts"]),
    )
    require(
        primary,
        "runtime_receipt.validation_errors",
        [],
        validate_current_work_receipt(artifact.get("current_work_receipt", {}), root=ROOT),
    )
    dependencies = artifact["training_dependency_hashes"]
    require(primary, "training_dependency_hashes.nonempty", True, bool(dependencies))
    for name, expected in dependencies.items():
        path = ROOT / name
        require(
            primary, f"dependency.{name}", expected, sha256_file(path) if path.is_file() else None
        )
    reference = next(
        (
            row
            for row in artifact["source_artifact_hashes"]
            if Path(row["path"]).resolve() == upstream.resolve()
        ),
        {},
    )
    require(primary, "qualified_source_path", str(upstream.resolve()), reference.get("path"))
    require(
        primary,
        "qualified_source_sha256",
        reference.get("sha256"),
        sha256_file(upstream) if upstream.is_file() else None,
    )
    return (
        [
            {
                "role": "qualified_runtime_primary",
                "path": str(primary),
                "sha256": sha256_file(primary),
            },
            {
                "role": "validation_attestation",
                "path": str(sidecar),
                "sha256": sha256_file(sidecar),
            },
        ],
        dependencies,
        artifact,
    )


def public_records(
    upstream: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Cohort roles and public bytes suffice to recompute exclusions without labels."""
    artifact = external_json(upstream)
    checks = (
        ("experiment_id", 7892),
        ("source_boundary_ready_score", 1),
        ("flagged_adversarial", False),
    )
    failed = [
        custody.operand(upstream, k, "==", v, artifact.get(k))
        for k, v in checks
        if artifact.get(k) != v
    ]
    if artifact.get("verdict_class") not in ("positive", "circular_positive", "null") or not str(
        artifact.get("honest_verdict", "")
    ).startswith("complete_"):
        failed.append(custody.operand(upstream, "terminal_eligible", "==", True, False))
    if failed:
        raise custody.InputBlocked(failed)
    public, sources = custody._shard_rows(upstream, artifact, "public_shards")
    cohort = Path(artifact["cohort_manifest_path"])
    manifest = external_json(cohort)
    if sha256_file(cohort) != artifact["cohort_manifest_sha256"]:
        raise custody.InputBlocked(
            [
                custody.operand(
                    upstream,
                    "cohort_manifest_sha256",
                    "==",
                    artifact["cohort_manifest_sha256"],
                    sha256_file(cohort),
                )
            ]
        )
    members = {row["family_id"]: row for row in manifest["rows"]}
    records, excluded = [], []
    for index, row in enumerate(public):
        if index % 32 == 0:
            progress("public_custody", "recompute exclusions", index)
        if (
            set(row) != {"family_id", "source_bytes", "answer_bytes"}
            or row["family_id"] not in members
        ):
            raise custody.InputBlocked(
                [
                    custody.operand(
                        upstream, "public_columns_and_join", "valid", True, row["family_id"]
                    )
                ]
            )
        member = members[row["family_id"]]
        record = {
            "id": row["family_id"],
            "group": row["family_id"],
            "role": member["role"],
            "source": bytes.fromhex(row["source_bytes"]),
            "answer": bytes.fromhex(row["answer_bytes"]),
            "source_cluster_id": member["source_cluster_id"],
        }
        reason = evidence_views.prepare_views(record["source"], record["answer"])["a"]["abstention"]
        if member["exclusion_reasons"] != ([reason] if reason else []):
            raise custody.InputBlocked(
                [
                    custody.operand(
                        upstream,
                        "exclusion_reasons",
                        "==",
                        member["exclusion_reasons"],
                        [reason] if reason else [],
                    )
                ]
            )
        if reason:
            excluded.append(
                {
                    "family_id": record["id"],
                    "role": record["role"],
                    "reason": reason,
                    "source_cluster_id": record["source_cluster_id"],
                }
            )
        else:
            records.append(record)
    counts = Counter(row["role"] for row in [*records, *excluded])
    if (
        counts != prior.ROLE_BUDGET
        or len(members) != len(public)
        or len({r["family_id"] for r in public}) != len(public)
    ):
        raise custody.InputBlocked(
            [
                custody.operand(
                    upstream, "role_counts_unique_families", "==", prior.ROLE_BUDGET, dict(counts)
                )
            ]
        )
    sources.extend(
        [
            {"role": "upstream", "path": str(upstream), "sha256": sha256_file(upstream)},
            {"role": "cohort_manifest", "path": str(cohort), "sha256": sha256_file(cohort)},
        ]
    )
    for reference in [*artifact["evaluator_shards"], *artifact["feature_shards"]]:
        path = Path(reference["path"])
        if not path.is_file() or sha256_file(path) != reference["sha256"]:
            raise custody.InputBlocked(
                [
                    custody.operand(
                        upstream,
                        "evaluator_shard_sha256",
                        "==",
                        reference["sha256"],
                        sha256_file(path) if path.is_file() else None,
                    )
                ]
            )
        sources.append(
            {
                "role": "evaluator_shards"
                if reference in artifact["evaluator_shards"]
                else "feature_shards",
                **reference,
            }
        )
    return records, excluded, sources


def attach_labels(
    records: list[dict[str, Any]], upstream: Path, roles: set[str]
) -> list[dict[str, Any]]:
    """Access target fields only for the authorized roles, after public preparation."""
    artifact = json.loads(upstream.read_text())
    evaluator, _ = custody._shard_rows(upstream, artifact, "evaluator_shards")
    selected = {row["family_id"]: row for row in evaluator if row["role"] in roles}
    result = []
    for record in records:
        if record["role"] not in roles:
            continue
        row = selected[record["id"]]
        offsets = row["annotation_byte_offsets"]
        if (
            row.get("observed") != 1
            or row.get("human_label") not in (0, 1)
            or any(not 0 <= a < b <= len(record["answer"]) for a, b in offsets)
        ):
            raise custody.InputBlocked(
                [custody.operand(upstream, "evaluator_label", "valid", True, record["id"])]
            )
        result.append(
            {
                **record,
                "label": row["human_label"],
                "known": custody._known(record["answer"], row["human_label"], offsets),
            }
        )
    return result


def fit_heads(
    records: list[dict[str, Any]], raw: Path, dependency_hash: str, budget_s: float
) -> tuple[list[dict[str, Any]], Path]:
    """Reuse only current checkpoints with the complete frozen dependency identity."""
    path = raw / "checkpoint_manifest.json"
    saved = json.loads(path.read_text()) if path.is_file() else {}
    manifest = (
        saved.get("checkpoints", []) if saved.get("dependency_hash") == dependency_hash else []
    )
    manifest = [
        row
        for row in manifest
        if Path(row["path"]).is_file() and sha256_file(Path(row["path"])) == row["sha256"]
    ]
    deadline = time.monotonic() + budget_s
    for arm in ARMS:
        for seed in SEEDS:
            if any(row["arm"] == arm and row["seed"] == seed for row in manifest):
                progress("fit", f"resume arm={arm} seed={seed}", len(manifest))
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("single numerical budget exhausted")
            began = time.monotonic()
            progress("fit", f"before arm={arm} seed={seed}", len(manifest))
            stop = threading.Event()

            def heartbeat() -> None:
                while not stop.wait(30):
                    progress("fit", f"heartbeat arm={arm} seed={seed}", len(manifest))

            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                head = natural_training.fit(
                    [r for r in records if r["role"] == "fit"],
                    [r for r in records if r["role"] == "tune"],
                    arm,
                    seed,
                    0.01,
                    16,
                )
            finally:
                stop.set()
                thread.join(timeout=1)
            if head["parameter_count"] > 4096 or len(head["curve"]) != 16:
                raise ValueError("head budget mismatch")
            checkpoint = raw / "checkpoints" / dependency_hash.split(":")[-1] / f"{arm}_{seed}.json"
            training_runtime.save(checkpoint, head)
            manifest.append(
                {
                    **{
                        k: v for k, v in head.items() if k not in ("params", "curve", "arm", "seed")
                    },
                    "arm": arm,
                    "seed": seed,
                    "path": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                    "epochs": head["curve"],
                    "duration_s": time.monotonic() - began,
                    "dependency_hash": dependency_hash,
                }
            )
            atomic_json(path, {"dependency_hash": dependency_hash, "checkpoints": manifest})
            progress("fit", f"after arm={arm} seed={seed}", len(manifest))
    return manifest, path


def library(upstream: Path, output: Path) -> ModuleType:
    """Use the existing scorer and reducers without invoking a historical publisher."""
    spec = importlib.util.spec_from_file_location("_carnot7930_callables", Path(prior.__file__))
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    value.UPSTREAM, value.OUTPUT, value.OWNED = upstream, output, list(OWNED)
    value.START, value.progress = START, progress
    return value


def replay(path: Path) -> None:
    """Cold reconstruction uses checkpoint parameters and exact public bytes."""
    value = json.loads(path.read_text())
    if value["verdict_class"] == "blocked":
        if not value["gate_check_summary"] or value["energy_fit_ready_score"] != 0:
            raise ValueError("blocked gate drift")
        progress("replay", "blocked operands preserved")
        return
    q = library(Path(value["upstream_path"]), path)
    q.replay(path)
    for reference in value["source_artifact_hashes"]:
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("source hash drift")
    records, _, _ = public_records(Path(value["upstream_path"]))
    by_id = {r["id"]: r for r in records}
    rows = value["rows"]
    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in json.loads(Path(value["checkpoint_manifest_path"]).read_text())["checkpoints"]:
        head = training_runtime.load(Path(spec["path"]))
        for role in prior.ROLE_BUDGET:
            selected = [
                r
                for r in rows
                if r["arm"] == spec["arm"] and r["seed"] == spec["seed"] and r["role"] == role
            ]
            if not selected:
                continue
            public = [by_id[r["family_id"]] for r in selected]
            view_class = (
                spec["arm"]
                if spec["arm"]
                in ("source_erased_constrained_set", "complete_static_constrained_set")
                else "local_set"
            )
            key = role, view_class
            if key not in prepared:
                prepared[key], _ = natural_training.prepare(public, view_class)
            progress("replay", f"before arm={spec['arm']} seed={spec['seed']} role={role}")
            predictions = q.predict_batch(head, public, prepared[key])
            if not np.allclose(
                [r["probability"] for r in selected],
                [r["probability_unsupported"] for r in predictions],
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError("cold probability drift")
            progress(
                "replay", f"after arm={spec['arm']} seed={spec['seed']} role={role}", len(selected)
            )
    progress("replay", "public-byte reconstruction passed", len(rows))

"""Qualify public source features and observed local labels (REQ-REPORT-7824)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import source_alignment

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7824_v680_source_feature_isolation"
RAW = ROOT / "results/raw" / NAME
OUTPUT = ROOT / "results" / f"{NAME}.json"
COMMAND_MANIFEST = RAW / "validation_command_manifest.json"
COMMAND_SHA256 = "sha256:d19bb4a29e951c2084e44299ed37ff43e2b92d4f12fa165e120dadc6140c8a88"
UPSTREAM = ROOT / "results/experiment_7810_v679_source_view_qualification.json"
ROLES = {
    "fit": 256,
    "tune": 64,
    "policy": 64,
    "online_update": 96,
    "online_admission": 64,
    "evaluation": 64,
    "retention": 32,
}
FORBIDDEN = (
    "role",
    "family_id",
    "label",
    "response_label",
    "sentence_targets",
    "error_annotations",
    "annotations",
    "confidence",
    "generator_identity",
    "source_family_id",
)
VIEW_FIELDS = (
    "abstention",
    "source_bytes",
    "answer_bytes",
    "source_sentences",
    "answer_units",
    "source_offsets",
    "answer_offsets",
    "window_offsets",
    "windows",
    "group_ids",
)
MODEL_SPECS: list[dict[str, Any]] = []


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Print the real elapsed time and completed unit count at every phase edge."""
    print(
        f"[exp7824] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def digest(value: Any) -> str:
    """Hash canonical JSON so a changed public record changes its identity."""
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def project(row: dict[str, Any]) -> dict[str, Any]:
    """Copy only the original byte fields and structural view fields."""
    return {
        "family_id": row.get("family_id"),
        "view_a": {k: row["view_a"][k] for k in VIEW_FIELDS},
        "view_b": {k: row["view_b"][k] for k in VIEW_FIELDS},
    }


def extract(public: dict[str, Any]) -> dict[str, Any]:
    """Validate an exact public allowlist before computing 132-feature tensors."""
    if set(public) != {"family_id", "view_a", "view_b"} or not isinstance(
        public["family_id"], (str, type(None))
    ):
        raise TypeError("invalid public record fields")
    answer: dict[str, Any] = {"family_id": public["family_id"], "feature_dim": 132}
    for arm in ("a", "b"):
        view = public[f"view_{arm}"]
        if not isinstance(view, dict) or set(view) != set(VIEW_FIELDS):
            raise TypeError("invalid view structure")
        if not isinstance(view["source_bytes"], str) or not isinstance(view["answer_bytes"], str):
            raise TypeError("invalid byte encoding")
        source = bytes.fromhex(view["source_bytes"])
        response = bytes.fromhex(view["answer_bytes"])
        if not isinstance(view["windows"], list) or not isinstance(view["answer_units"], list):
            raise TypeError("invalid public view types")
        windows = [bytes.fromhex(value) for value in view["windows"]]
        units = [bytes.fromhex(value) for value in view["answer_units"]]
        if (
            b"".join(bytes.fromhex(x) for x in view["source_sentences"]) != source
            or b"".join(units) != response
            or len(windows) != len(view["group_ids"])
        ):
            raise ValueError("invalid public view bytes")
        tensor = (
            []
            if view["abstention"]
            else [source_alignment.pair_features(w, units) for w in windows]
        )
        if any(len(vector) != 132 for vector in tensor):
            raise ValueError("feature dimension drift")
        answer[f"view_{arm}_tensor_sha256"] = digest(tensor)
        answer[f"view_{arm}_tensor_rows"] = len(tensor)
    return answer


def masked_local_loss(
    logits: list[float], labels: list[int], observed: list[int]
) -> tuple[float, list[float]]:
    """Average observed Bernoulli loss; unknown positions have zero gradient."""
    if (
        len(logits) != len(labels)
        or len(logits) != len(observed)
        or any(x not in (0, 1) for x in (*labels, *observed))
    ):
        raise ValueError("invalid label mask")
    count = max(sum(observed), 1)
    losses = []
    gradients = []
    for logit, label, mask in zip(logits, labels, observed, strict=True):
        probability = 1 / (1 + math.exp(-logit))
        losses.append(mask * (math.log1p(math.exp(logit)) - label * logit) / count)
        gradients.append(mask * (probability - label) / count)
    return sum(losses), gradients


def load_command_manifest() -> dict[str, Any]:
    """Read the exact prospectively sealed validation command bytes."""
    if sha256_file(COMMAND_MANIFEST) != COMMAND_SHA256:
        raise ValueError("validation_manifest_drift")
    return json.loads(COMMAND_MANIFEST.read_text())


def validate_command_manifest(value: dict[str, Any]) -> None:
    """Reject altered names, argv, classes, order or added children."""
    if value != load_command_manifest():
        raise ValueError("validation_manifest_drift")


def validate_log_receipt(receipt: dict[str, Any]) -> None:
    """Trust a child exit only while its sealed log bytes are unchanged."""
    path = Path(receipt["log_path"])
    if not path.is_file() or sha256_file(path) != receipt["log_sha256"]:
        raise ValueError("validation_log_drift")


def check_public_manifest(value: dict[str, Any]) -> None:
    """Cold-check the published public file against its sealed byte hash."""
    path = Path(value["public_path"])
    if not path.is_file() or sha256_file(path) != value["public_sha256"]:
        raise ValueError("public_hash_drift")


def execute_child(command: dict[str, Any], index: int, scope: dict[str, Any]) -> dict[str, Any]:
    """Run one declared command, then seal its closed log at a unique digest path."""
    private = Path(scope["private_root"])
    spec = CommandSpec(
        command["name"],
        tuple(command["argv"]),
        command["classification"],
        timeout_s=float(command["timeout_s"]),
    )
    receipt = run_commands(
        ROOT,
        [spec],
        log_dir=private / "logs" / f"{index:02d}_{command['name']}",
        extra_env={"CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )[0]
    source = ROOT / receipt["log_path"]
    hash_value = sha256_file(source)
    destination = RAW / "validation_logs" / f"{index:02d}_{command['name']}_{hash_value[7:]}.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("validation_log_path_reused")
    shutil.copyfile(source, destination)
    receipt.update(
        classification=command["classification"],
        log_path=str(destination),
        log_sha256=sha256_file(destination),
    )
    validate_log_receipt(receipt)
    return receipt


def dispatch(
    scope: dict[str, Any],
    executor: Callable[[dict[str, Any], int, dict[str, Any]], dict[str, Any]] = execute_child,
) -> list[dict[str, Any]]:
    """Run every and only frozen child, preserving exit and log identities."""
    validate_command_manifest(scope)
    if executor is execute_child:
        private = Path(scope["private_root"])
        for part in ("pytest", "coverage", "logs"):
            (private / part).mkdir(parents=True, exist_ok=True)
    receipts = []
    for index, command in enumerate(scope["commands"]):
        receipt = executor(command, index, scope)
        if (receipt["name"], receipt["command_argv"], receipt["classification"]) != (
            command["name"],
            command["argv"],
            command["classification"],
        ):
            raise ValueError("observed_child_command_drift")
        validate_log_receipt(receipt)
        receipts.append(receipt)
    return receipts


def failed_operand(
    upstream: str, path: Path, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Record the literal failed precondition and exact available input hash."""
    return {
        "upstream_id": upstream,
        "path": str(path),
        "hash": sha256_file(path) if path.is_file() else None,
        "field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
    }


def read_lines(path: Path) -> list[dict[str, Any]]:
    """Read an explicitly named JSONL shard without changing row order."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def preflight() -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Authenticate the qualified science producer and every original role shard."""
    failed = []
    sources = []
    if not UPSTREAM.is_file():
        return None, [failed_operand("exp7810", UPSTREAM, "is_file", True, False)], sources
    qualified = json.loads(UPSTREAM.read_text())
    for field, expected in (
        ("verdict_class", "circular_positive"),
        ("evidence_view_ready_score", 1),
        ("run_date", "20260928"),
    ):
        if qualified.get(field) != expected:
            failed.append(
                failed_operand("exp7810", UPSTREAM, field, expected, qualified.get(field))
            )
    sources.append(
        {
            "upstream_id": "exp7810_science",
            "path": str(UPSTREAM),
            "sha256": sha256_file(UPSTREAM),
            "date": "20260928",
            "role": "qualified_canonical",
            "eligibility": "exposed_development",
        }
    )
    path = Path(qualified.get("source_view_manifest_path") or "missing")
    if not path.is_file():
        return None, [*failed, failed_operand("exp7810", path, "is_file", True, False)], sources
    manifest = json.loads(path.read_text())
    sources.append(
        {
            "upstream_id": "exp7810_science",
            "path": str(path),
            "sha256": sha256_file(path),
            "date": "20260928",
            "role": "source_view_manifest",
            "eligibility": "exposed_development",
        }
    )
    for field, expected in (("role_counts", ROLES), ("role_hashes", qualified.get("role_hashes"))):
        if manifest.get(field) != expected:
            failed.append(failed_operand("exp7810", path, field, expected, manifest.get(field)))
    for field, hash_field in (
        ("rows_path", "rows_sha256"),
        ("development_manifest_path", "development_manifest_sha256"),
    ):
        shard = Path(manifest[field])
        observed = sha256_file(shard) if shard.is_file() else None
        if observed != manifest[hash_field]:
            failed.append(
                failed_operand("exp7810", shard, hash_field, manifest[hash_field], observed)
            )
        sources.append(
            {
                "upstream_id": "exp7810_science",
                "path": str(shard),
                "sha256": observed,
                "date": "20260928",
                "role": field,
                "eligibility": "exposed_development",
            }
        )
    for role in ROLES:
        for kind in ("public", "evaluator"):
            shard = (
                Path(manifest[f"{kind}_paths"][role])
                if f"{kind}_paths" in manifest
                else Path(manifest["development_manifest_path"]).parent
                / json.loads(Path(manifest["development_manifest_path"]).read_text())["roles"][
                    role
                ][f"{kind}_path"]
            )
            observed = sha256_file(shard) if shard.is_file() else None
            expected = manifest["role_hashes"][role][kind]
            if observed != expected:
                failed.append(
                    failed_operand("exp7727", shard, f"{role}.{kind}.sha256", expected, observed)
                )
            sources.append(
                {
                    "upstream_id": "exp7727_science",
                    "path": str(shard),
                    "sha256": observed,
                    "date": "20260926",
                    "role": f"{role}_{kind}",
                    "eligibility": "exposed_development",
                }
            )
    target_path = Path(manifest["rows_path"]).with_name("targets.jsonl")
    if not target_path.is_file():
        failed.append(failed_operand("exp7810", target_path, "is_file", True, False))
    else:
        sources.append(
            {
                "upstream_id": "exp7810_science",
                "path": str(target_path),
                "sha256": sha256_file(target_path),
                "date": "20260928",
                "role": "evaluator_targets",
                "eligibility": "exposed_development",
            }
        )
    return manifest, failed, sources


def scan_canonical(
    manifest: dict[str, Any], start: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare every sealed view with original source and evaluator shards."""
    rows = read_lines(Path(manifest["rows_path"]))
    targets = read_lines(Path(manifest["rows_path"]).with_name("targets.jsonl"))
    development = json.loads(Path(manifest["development_manifest_path"]).read_text())
    public_original = []
    evaluator_original = []
    expected_ids = []
    for role in ROLES:
        entry = development["roles"][role]
        base = Path(manifest["development_manifest_path"]).parent
        role_public = read_lines(base / entry["public_path"])
        role_evaluator = read_lines(base / entry["evaluator_path"])
        if len(role_public) != ROLES[role] or len(role_evaluator) != ROLES[role]:
            raise ValueError("role_count_drift")
        expected_ids.extend(entry["families"])
        public_original.extend(role_public)
        evaluator_original.extend(role_evaluator)
        progress(start, "custody", role, len(expected_ids))
    if len(rows) != 640 or len(targets) != 640 or len(set(expected_ids)) != 640:
        raise ValueError("canonical_roster_drift")
    for index, (row, target, public, evaluator, family) in enumerate(
        zip(rows, targets, public_original, evaluator_original, expected_ids, strict=True), 1
    ):
        if any(item["family_id"] != family for item in (row, target, public, evaluator)):
            raise ValueError(f"family_order_drift:{index}")
        if row["role"] != public["role"] or target["role"] != public["role"]:
            raise ValueError(f"role_drift:{index}")
        for arm in ("a", "b"):
            view = row[f"view_{arm}"]
            if (
                bytes.fromhex(view["source_bytes"]) != public["complete_source"].encode()
                or bytes.fromhex(view["answer_bytes"]) != public["complete_response"].encode()
                or view["source_sha256"] != public["source_sha256"]
                or view["answer_sha256"] != public["response_sha256"]
            ):
                raise ValueError(f"source_answer_drift:{index}")
        if (
            target["response_label"] != evaluator["label"]
            or target["response_sha256"] != public["response_sha256"]
        ):
            raise ValueError(f"annotation_drift:{index}")
        if index % 64 == 0:
            progress(start, "custody", "rows", index)
    return rows, targets


def write_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    """Write one complete deterministic shard before sealing its byte digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return sha256_file(path)


def run_extractor(public_path: Path, output_path: Path, start: float) -> None:
    """Give a fresh process only the public path and a minimal environment."""
    argv = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(ROOT / "scripts/experiments" / f"{NAME}.py"),
        "--extract",
        str(public_path),
        str(output_path),
    ]
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin"),
        "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}",
        "JAX_PLATFORMS": "cpu",
        "PYTHONUNBUFFERED": "1",
    }
    progress(start, "extraction", "before_subprocess")
    child = subprocess.Popen(argv, cwd=ROOT, env=environment, close_fds=True)
    while child.poll() is None:
        progress(start, "extraction", "subprocess_outstanding")
        if time.monotonic() - start > 600:
            child.terminate()
            raise TimeoutError("extractor_deadline")
        time.sleep(30)
    progress(start, "extraction", "after_subprocess", 640 if child.returncode == 0 else 0)
    if child.returncode != 0:
        raise ValueError("public_extraction_failed")


def extract_file(public_path: Path, output_path: Path) -> None:
    """Compute feature hashes with no evaluator file descriptor or metadata."""
    start = time.monotonic()
    rows = read_lines(public_path)
    features = []
    for index, public in enumerate(rows, 1):
        features.append(extract(public))
        if index % 64 == 0:
            progress(start, "public_feature_extraction", "rows", index)
    write_lines(output_path, features)


def measure(
    rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    features: list[dict[str, Any]],
    start: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Record every metadata mutation and observed-mask intervention by family."""
    invariance = []
    masked = []
    for index, (row, target, feature) in enumerate(zip(rows, targets, features, strict=True), 1):
        baseline = project(row)
        public_hash = digest({arm: baseline[arm] for arm in ("view_a", "view_b")})
        for field in FORBIDDEN:
            for intervention in ("mutate", "omit"):
                changed = dict(row)
                if intervention == "mutate":
                    changed[field] = {"poisoned": index}
                else:
                    changed.pop(field, None)
                changed_public = project(changed)
                same = (
                    digest({arm: changed_public[arm] for arm in ("view_a", "view_b")})
                    == public_hash
                )
                invariance.append(
                    {
                        "family_id": row["family_id"],
                        "role": row["role"],
                        "field": field,
                        "intervention": intervention,
                        "public_sha256": public_hash,
                        "tensor_sha256_a": feature["view_a_tensor_sha256"],
                        "tensor_sha256_b": feature["view_b_tensor_sha256"],
                        "equal": same,
                    }
                )
        values = target["sentence_targets"]
        observed = [int(value is not None) for value in values]
        zero = [0 if value is None else value for value in values]
        one = [1 if value is None else value for value in values]
        logits = [0.7 + 0.01 * j for j in range(len(values))]
        loss_zero, grad_zero = masked_local_loss(logits, zero, observed)
        loss_one, grad_one = masked_local_loss(logits, one, observed)
        masked.append(
            {
                "family_id": row["family_id"],
                "role": row["role"],
                "unknown_count": observed.count(0),
                "observed_count": sum(observed),
                "loss_zero": loss_zero,
                "loss_one": loss_one,
                "gradient_zero_sha256": digest(grad_zero),
                "gradient_one_sha256": digest(grad_one),
                "equal": loss_zero == loss_one and grad_zero == grad_one,
            }
        )
        if index % 64 == 0:
            progress(start, "invariance", "rows", index)
    return invariance, masked


def cold_reduce(candidate: Path) -> dict[str, Any]:
    """Reopen current raw rows, manifests and sealed receipts in a fresh process."""
    value = json.loads(candidate.read_text())
    public_manifest = json.loads(Path(value["public_feature_manifest_path"]).read_text())
    sidecar_manifest = json.loads(Path(value["label_sidecar_manifest_path"]).read_text())
    check_public_manifest(public_manifest)
    for manifest, path_key, hash_key in (
        (public_manifest, "feature_path", "feature_sha256"),
        (sidecar_manifest, "sidecar_path", "sidecar_sha256"),
    ):
        path = Path(manifest[path_key])
        if not path.is_file() or sha256_file(path) != manifest[hash_key]:
            raise ValueError("manifest_shard_drift")
    canonical, failed, _ = preflight()
    if failed or canonical is None:
        raise ValueError("canonical_custody_drift")
    source_rows, targets = scan_canonical(canonical, time.monotonic())
    public = read_lines(Path(public_manifest["public_path"]))
    features = read_lines(Path(public_manifest["feature_path"]))
    sidecar = read_lines(Path(sidecar_manifest["sidecar_path"]))
    if (
        len(public) != 640
        or len(features) != 640
        or len(sidecar) != 640
        or len(value["rows"]) != 640
    ):
        raise ValueError("candidate_row_count_drift")
    for raw, target, exposed, feature, private, result in zip(
        source_rows, targets, public, features, sidecar, value["rows"], strict=True
    ):
        if (
            exposed != project(raw)
            or feature != extract(exposed)
            or private["family_id"] != raw["family_id"]
            or private["sentence_targets"] != target["sentence_targets"]
            or result["family_id"] != raw["family_id"]
        ):
            raise ValueError("candidate_row_drift")
    for receipt in value.get("validation_receipts", []):
        validate_log_receipt(receipt)
    return {
        "families": 640,
        "public_sha256": public_manifest["public_sha256"],
        "sidecar_sha256": sidecar_manifest["sidecar_sha256"],
    }


def base_result(
    start: float,
    failed: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a complete terminal shape even if an external input is absent."""
    blocked = bool(failed)
    fields = (
        "experiment_id",
        "milestone",
        "run_date",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "gate_check_summary",
        "rows",
        "acceptance_gate_results",
        "duration_s",
        "phase_spans",
        "random_seed",
        "reproducibility_checksum",
        "sample_size_budget",
        "source_artifact_hashes",
        "preconditions_checked",
        "validation_receipts",
        "validation_command_manifest_path",
        "observed_child_commands",
        "repository_health",
        "verifier_is_oracle",
        "claim_scope",
        "field_principles",
        "inference_substrate",
        "inference_substrate_class",
        "MODEL_SPECS",
        "model_specs",
        "model_invocation_counts",
        "source_isolation_ready_score",
        "source_view_manifest_path",
        "public_feature_manifest_path",
        "label_sidecar_manifest_path",
        "role_hashes",
        "feature_invariance_rows",
        "masked_loss_rows",
    )
    identity = {
        "inputs": sources,
        "seed": 7824,
        "code": sha256_file(Path(__file__)),
        "wrapper": sha256_file(ROOT / "scripts/experiments" / f"{NAME}.py"),
        "commands": sha256_file(COMMAND_MANIFEST),
    }
    return {
        "schema": "carnot.exp7824.source_feature_isolation.v1",
        "experiment_id": "exp7824-source-feature-isolation",
        "milestone": "2026.09.680",
        "run_date": "20260928",
        "honest_verdict": "complete_blocked_external_input"
        if blocked
        else "complete_circular_positive_source_isolation",
        "verdict_class": "blocked" if blocked else "circular_positive",
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "rows": [],
        "acceptance_gate_results": {
            "validity": not blocked,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - start,
        "phase_spans": spans,
        "random_seed": 7824,
        "reproducibility_checksum": digest(identity),
        "sample_size_budget": {
            "intended": 640,
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "excluded": 0,
            "censored": 0,
            "independent_n": 640,
            "role_counts": ROLES,
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": failed,
        "validation_receipts": [],
        "validation_command_manifest_path": str(COMMAND_MANIFEST),
        "observed_child_commands": [],
        "repository_health": {
            "historical_full_suite_obligation": "failed",
            "broad_suite_exit": None,
            "classification": "diagnostic",
        },
        "verifier_is_oracle": True,
        "claim_scope": "640 exposed development families; fixture invariance only; no oracle-distinct win",
        "field_principles": {field: "Preserve the stated evidence boundary." for field in fields},
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": [],
        "model_invocation_counts": {"loads": 0, "calls": 0, "tokens": 0, "loaded_file_hashes": []},
        "source_isolation_ready_score": 0,
        "source_view_manifest_path": None,
        "public_feature_manifest_path": None,
        "label_sidecar_manifest_path": None,
        "role_hashes": {},
        "feature_invariance_rows": [],
        "masked_loss_rows": [],
    }


def run_experiment(date: str) -> dict[str, Any]:
    """Qualify canonical public features and publish after owned validation."""
    start = time.monotonic()
    progress(start, "start", "begin")
    if date != "20260928":
        raise ValueError("run_date_mismatch")
    scope = load_command_manifest()
    validate_command_manifest(scope)
    private = Path(scope["private_root"])
    if private.exists():
        raise ValueError("attempt_root_reused")
    private.mkdir(parents=True)
    spans = []
    phase = time.monotonic()
    progress(start, "preconditions", "begin")
    canonical, failed, sources = preflight()
    spans.append({"phase": "preconditions", "duration_s": time.monotonic() - phase})
    progress(start, "preconditions", "complete", len(sources))
    value = base_result(start, failed, sources, spans)
    if failed or canonical is None:
        atomic_json(OUTPUT, value)
        progress(start, "publish", "blocked")
        return value
    phase = time.monotonic()
    progress(start, "custody", "begin")
    rows, targets = scan_canonical(canonical, start)
    spans.append({"phase": "custody", "duration_s": time.monotonic() - phase})
    progress(start, "custody", "complete", len(rows))
    public = [project(row) for row in rows]
    sidecar = [
        dict(target, evaluator_only=True, source_role=row["role"])
        for row, target in zip(rows, targets, strict=True)
    ]
    public_path = RAW / "public_records.jsonl"
    feature_path = RAW / "public_features.jsonl"
    sidecar_path = RAW / "label_sidecar.jsonl"
    phase = time.monotonic()
    progress(start, "projection", "begin")
    public_hash = write_lines(public_path, public)
    sidecar_hash = write_lines(sidecar_path, sidecar)
    run_extractor(public_path, feature_path, start)
    features = read_lines(feature_path)
    if len(features) != 640:
        raise ValueError("feature_row_count_drift")
    public_manifest = {
        "schema": "carnot.exp7824.public_features.v1",
        "source_view_manifest_path": str(
            Path(value["source_view_manifest_path"] or canonical["rows_path"]).with_name(
                "source_view_manifest.json"
            )
        ),
        "public_path": str(public_path),
        "public_sha256": public_hash,
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "rows": 640,
        "allowlist": ["family_id", "view_a", "view_b"],
        "view_allowlist": list(VIEW_FIELDS),
    }
    sidecar_manifest = {
        "schema": "carnot.exp7824.label_sidecar.v1",
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sidecar_hash,
        "rows": 640,
        "custody": "evaluator_only",
        "role_hashes": canonical["role_hashes"],
    }
    public_manifest_path = RAW / "public_feature_manifest.json"
    sidecar_manifest_path = RAW / "label_sidecar_manifest.json"
    atomic_json(public_manifest_path, public_manifest)
    atomic_json(sidecar_manifest_path, sidecar_manifest)
    spans.append({"phase": "projection", "duration_s": time.monotonic() - phase})
    progress(start, "projection", "complete", len(features))
    phase = time.monotonic()
    progress(start, "invariance", "begin")
    invariance, masked = measure(rows, targets, features, start)
    fixture = masked_local_loss([0.7], [0], [1])
    flipped = masked_local_loss([0.7], [1], [1])
    valid = (
        all(row["equal"] for row in invariance)
        and all(row["equal"] for row in masked)
        and fixture[0] != flipped[0]
        and fixture[1][0] != 0
    )
    spans.append({"phase": "invariance", "duration_s": time.monotonic() - phase})
    progress(start, "invariance", "complete", len(invariance))
    value.update(
        source_view_manifest_path=str(
            Path(canonical["rows_path"]).with_name("source_view_manifest.json")
        ),
        public_feature_manifest_path=str(public_manifest_path),
        label_sidecar_manifest_path=str(sidecar_manifest_path),
        role_hashes=canonical["role_hashes"],
        feature_invariance_rows=invariance,
        masked_loss_rows=masked,
        rows=[
            {
                "family_id": row["family_id"],
                "role": row["role"],
                "raw_path": canonical["rows_path"],
                "status": "completed",
                "source_sha256": row["source_sha256"],
                "answer_sha256": row["response_sha256"],
                "view_a_tensor_sha256": feature["view_a_tensor_sha256"],
                "view_b_tensor_sha256": feature["view_b_tensor_sha256"],
                "unknown_count": local["unknown_count"],
                "invariance_passed": local["equal"],
            }
            for row, feature, local in zip(rows, features, masked, strict=True)
        ],
        sample_size_budget={
            "intended": 640,
            "eligible": 640,
            "started": 640,
            "completed": 640,
            "excluded": 0,
            "censored": 0,
            "independent_n": 640,
            "role_counts": ROLES,
        },
        preconditions_checked=[
            {
                "field": "exp7810_qualified",
                "passed": True,
                "path": str(UPSTREAM),
                "hash": sha256_file(UPSTREAM),
            },
            {"field": "all_role_shards", "passed": True, "count": 14},
        ],
        source_isolation_ready_score=int(valid),
        acceptance_gate_results={
            "validity": valid,
            "readiness": int(valid),
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        duration_s=time.monotonic() - start,
        phase_spans=spans,
        text_sensitivity={
            "kind": "descriptive_only",
            "source_answer_bytes_are_feature_inputs": True,
        },
    )
    atomic_json(Path(scope["candidate_path"]), value)
    phase = time.monotonic()
    progress(start, "validation", "begin")
    receipts = dispatch(scope)
    spans.append({"phase": "validation", "duration_s": time.monotonic() - phase})
    progress(start, "validation", "complete", len(receipts))
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = [
        {"name": r["name"], "argv": r["command_argv"], "classification": r["classification"]}
        for r in receipts
    ]
    value["repository_health"]["broad_suite_exit"] = next(
        r["exit_code"] for r in receipts if r["name"] == "repository_health"
    )
    required_failures = [
        r for r in receipts if r["classification"] == "required" and not r["passed"]
    ]
    if required_failures or not valid:
        value["honest_verdict"] = "complete_disqualified_required_validation"
        value["verdict_class"] = "disqualified"
        value["source_isolation_ready_score"] = 0
        value["acceptance_gate_results"]["readiness"] = 0
        value["acceptance_gate_results"]["validity"] = False
        value["gate_check_summary"] = [
            failed_operand("exp7824", Path(r["log_path"]), r["name"], 0, r["exit_code"])
            for r in required_failures
        ]
    value["duration_s"] = time.monotonic() - start
    atomic_json(OUTPUT, value)
    progress(start, "publish", "complete", len(value["rows"]))
    return value


def main(argv: list[str] | None = None) -> int:
    """Dispatch one dated producer or an explicit public extraction/replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--extract", nargs=2, metavar=("PUBLIC", "FEATURES"))
    parser.add_argument("--cold-replay", metavar="CANDIDATE")
    args = parser.parse_args(argv)
    if args.extract:
        extract_file(Path(args.extract[0]), Path(args.extract[1]))
        return 0
    if args.cold_replay:
        print(json.dumps(cold_reduce(Path(args.cold_replay)), sort_keys=True), flush=True)
        return 0
    if args.date is None:
        parser.error("--date is required for a producer run")
    return int(run_experiment(args.date)["verdict_class"] == "disqualified")

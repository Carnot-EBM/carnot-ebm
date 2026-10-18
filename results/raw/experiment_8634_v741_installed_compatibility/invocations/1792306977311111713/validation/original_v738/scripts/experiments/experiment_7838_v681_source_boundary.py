"""Qualify the current public source boundary (REQ-REPORT-7838)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import source_projection

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7838_v681_source_boundary"
RAW = ROOT / "results/raw" / NAME
ATTEMPT = RAW / "attempts/current"
OUTPUT = ROOT / "results" / f"{NAME}.json"
SCOPE = RAW / "validation_command_manifest.json"
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
REQUIRED = (
    "worktree_imports",
    "affected_pytest",
    "changed_coverage",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec",
    "cli_e2e",
    "cold_replay",
    "adversarial_verify",
    "strict_rows",
)


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Show elapsed time and completed work at each real phase edge."""
    print(
        f"[exp7838] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def operand(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Keep a wrong value distinct from a missing external file."""
    return {
        "upstream_id": upstream,
        "path": str(path),
        "hash": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
    }


def checked_file(
    path: Path,
    expected: str | None,
    upstream: str,
    role: str,
    sources: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> bool:
    """Check an exact source path and record its byte identity and role."""
    observed = sha256_file(path) if path.is_file() else None
    sources.append(
        {
            "upstream_id": upstream,
            "path": str(path),
            "sha256": observed,
            "date": "20260928" if upstream == "exp7810" else "20260926",
            "role": role,
            "eligibility": "exposed_development",
        }
    )
    if observed is None or (expected is not None and observed != expected):
        failed.append(operand(upstream, path, "sha256", expected or "present", observed))
        return False
    return True


def preflight(
    start: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Authenticate Exp7810 and every original public and evaluator role shard."""
    sources: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    if not checked_file(UPSTREAM, None, "exp7810", "science_producer", sources, failed):
        return None, sources, failed
    qualified = json.loads(UPSTREAM.read_text())
    for field, expected in (
        ("experiment_id", "exp7810-source-view-qualification"),
        ("run_date", "20260928"),
        ("verdict_class", "circular_positive"),
        ("evidence_view_ready_score", 1),
        ("flagged_adversarial", False),
    ):
        if qualified.get(field) != expected:
            failed.append(operand("exp7810", UPSTREAM, field, expected, qualified.get(field)))
    path = Path(qualified.get("source_view_manifest_path") or "missing")
    if not checked_file(path, None, "exp7810", "canonical_manifest", sources, failed):
        return None, sources, failed
    manifest = json.loads(path.read_text())
    if manifest.get("role_counts") != ROLES:
        failed.append(operand("exp7810", path, "role_counts", ROLES, manifest.get("role_counts")))
    for field, hash_field in (
        ("rows_path", "rows_sha256"),
        ("development_manifest_path", "development_manifest_sha256"),
    ):
        checked_file(Path(manifest[field]), manifest[hash_field], "exp7810", field, sources, failed)
    development_path = Path(manifest["development_manifest_path"])
    if failed:
        return None, sources, failed
    development = json.loads(development_path.read_text())
    for role, count in ROLES.items():
        entry = development["roles"][role]
        if entry["count"] != count or len(entry["families"]) != count:
            failed.append(
                operand("exp7727", development_path, f"{role}.count", count, entry["count"])
            )
        for kind in ("public", "evaluator"):
            shard = development_path.parent / entry[f"{kind}_path"]
            expected = manifest["role_hashes"][role][kind]
            checked_file(shard, expected, "exp7727", f"{role}_{kind}", sources, failed)
    targets = Path(manifest["rows_path"]).with_name("targets.jsonl")
    checked_file(targets, None, "exp7810", "evaluator_targets", sources, failed)
    progress(start, "preflight", "checked", len(sources))
    return (manifest if not failed else None), sources, failed


def custody(
    manifest: dict[str, Any], start: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Join every canonical view to exact original source, answer and annotation bytes."""
    row_path = Path(manifest["rows_path"])
    rows = source_projection.read_jsonl(row_path)
    targets = source_projection.read_jsonl(row_path.with_name("targets.jsonl"))
    development_path = Path(manifest["development_manifest_path"])
    development = json.loads(development_path.read_text())
    public: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    index = 0
    for role, count in ROLES.items():
        entry = development["roles"][role]
        originals = source_projection.read_jsonl(development_path.parent / entry["public_path"])
        evaluators = source_projection.read_jsonl(development_path.parent / entry["evaluator_path"])
        if len(originals) != count or len(evaluators) != count:
            raise ValueError(f"role_count_drift:{role}")
        for original, evaluator, family in zip(
            originals, evaluators, entry["families"], strict=True
        ):
            row, target = rows[index], targets[index]
            if any(item["family_id"] != family for item in (original, evaluator, row, target)):
                raise ValueError(f"family_order_drift:{index}")
            if row["role"] != role or target["role"] != role or original["role"] != role:
                raise ValueError(f"role_drift:{index}")
            source = original["complete_source"].encode()
            answer = original["complete_response"].encode()
            if any(
                bytes.fromhex(row[f"view_{arm}"]["source_bytes"]) != source
                or bytes.fromhex(row[f"view_{arm}"]["answer_bytes"]) != answer
                for arm in ("a", "b")
            ):
                raise ValueError(f"source_answer_drift:{index}")
            if evaluator["label"] != target["response_label"]:
                raise ValueError(f"annotation_drift:{index}")
            offsets = [
                [
                    len(original["complete_response"][: item["start"]].encode()),
                    len(original["complete_response"][: item["end"]].encode()),
                ]
                for item in evaluator["annotations"]
            ]
            if offsets != target["annotation_byte_offsets"] or any(
                original["complete_response"][item["start"] : item["end"]] != item["text"]
                for item in evaluator["annotations"]
            ):
                raise ValueError(f"annotation_bytes_drift:{index}")
            if target["response_sha256"] != original["response_sha256"]:
                raise ValueError(f"target_answer_drift:{index}")
            if (
                original["source_sha256"] != row["source_sha256"]
                or original["response_sha256"] != row["response_sha256"]
            ):
                raise ValueError(f"source_hash_drift:{index}")
            public.append(
                {"family_id": family, "source_bytes": source.hex(), "answer_bytes": answer.hex()}
            )
            sidecar.append(
                {
                    "family_id": family,
                    "role": role,
                    "evaluator": evaluator,
                    "target": target,
                    "generator_identity": original.get("generator_identity"),
                    "confidence": original.get("confidence"),
                }
            )
            evidence.append(
                {
                    "family_id": family,
                    "role": role,
                    "status": "completed",
                    "arm": "public_projection",
                    "seed": 68138,
                    "source_sha256": original["source_sha256"],
                    "answer_sha256": original["response_sha256"],
                    "eligible": True,
                }
            )
            index += 1
            if index % 32 == 0:
                progress(start, "custody", "rows", index)
    if (
        index != 640
        or len(rows) != 640
        or len(targets) != 640
        or len({r["family_id"] for r in public}) != 640
    ):
        raise ValueError("canonical_roster_drift")
    return public, sidecar, evidence


def seal(receipt: dict[str, Any], index: int) -> dict[str, Any]:
    """Copy one closed child log to a unique path keyed by its exact bytes."""
    source = ROOT / receipt["log_path"]
    hash_value = sha256_file(source)
    destination = RAW / "validation_logs" / f"{index:02d}_{receipt['name']}_{hash_value[7:]}.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise ValueError("sealed_log_collision")
    if not destination.exists():
        shutil.copyfile(source, destination)
    receipt["log_path"] = str(destination)
    receipt["log_sha256"] = sha256_file(destination)
    if receipt["log_sha256"] != hash_value:
        raise ValueError("sealed_log_drift")
    return receipt


def child(spec: dict[str, Any], index: int) -> dict[str, Any]:
    """Run a bounded owned validation child and preserve its real exit."""
    command = CommandSpec(
        spec["name"],
        tuple(spec["argv"]),
        spec["classification"],
        timeout_s=float(spec["timeout_s"]),
    )
    receipt = run_commands(
        ROOT,
        [command],
        log_dir=ATTEMPT / "logs" / f"{index:02d}",
        extra_env={"CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"},
        heartbeat_s=30,
    )[0]
    return seal(receipt, index)


def isolated_extract(public_path: Path, features_path: Path, start: float) -> dict[str, Any]:
    """Launch public extraction with only public paths and a minimal environment."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}",
        "JAX_PLATFORMS": "cpu",
        "PYTHONUNBUFFERED": "1",
    }
    argv = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        "-m",
        "carnot.verify.source_projection",
        "extract",
        str(public_path),
        str(features_path),
    ]
    progress(start, "extraction", "before_subprocess")
    began = time.monotonic()
    process = subprocess.Popen(
        argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    chunks = []
    while True:
        try:
            output, _ = process.communicate(timeout=30)
            chunks.append(output)
            break
        except subprocess.TimeoutExpired:
            progress(start, "extraction", "heartbeat")
            if time.monotonic() - began >= 600:
                process.terminate()
                output, _ = process.communicate(timeout=10)
                chunks.append(output)
                break
    log = ATTEMPT / "extract.log"
    log.write_text("".join(chunks))
    receipt = {
        "name": "public_extraction",
        "command_argv": argv,
        "classification": "owned_compute",
        "exit_code": process.returncode,
        "passed": process.returncode == 0,
        "duration_s": time.monotonic() - began,
        "log_path": str(log),
        "log_sha256": sha256_file(log),
    }
    progress(start, "extraction", "after_subprocess")
    return seal(receipt, 99)


def artifact_base(
    start: float, sources: list[dict[str, Any]], failed: list[dict[str, Any]]
) -> dict[str, Any]:
    """Give blocked and completed outcomes the same explicit evidence contract."""
    receipt = build_current_work_receipt(
        run_id="exp7838-current",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_details={"model_calls": 0, "tokens": 0},
        inference_substrate_class="no_model_load",
        execution_venue="cpu",
        started_monotonic_ns=int(start * 1_000_000_000),
        ended_monotonic_ns=time.monotonic_ns(),
    )
    return {
        "schema": "carnot.exp7838.source_boundary.v1",
        "experiment_id": 7838,
        "task_id": "exp7838-source-boundary",
        "milestone": "2026.09.681",
        "run_date": "20260928",
        "honest_verdict": "complete_blocked_required_source_evidence"
        if failed
        else "complete_disqualified_required_checks",
        "verdict_class": "blocked" if failed else "disqualified",
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "rows": [],
        "sample_size_budget": {
            "intended": 640,
            "eligible": 0,
            "started": 0,
            "completed": 0,
            "censored": 0,
            "excluded": 0,
            "independent": 0,
        },
        "acceptance_gate_results": {
            "validity": False,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - start,
        "phase_spans": [],
        "random_seed": 68138,
        "reproducibility_checksum": None,
        "source_artifact_hashes": sources,
        "preconditions_checked": {"failed": failed, "source_count": len(sources)},
        "validation_receipts": [],
        "validation_command_manifest_path": str(SCOPE),
        "observed_child_commands": [],
        "repository_health": {
            "historical_exp7824": {
                "path": str(ROOT / "results/experiment_7824_v680_source_feature_isolation.json"),
                "verdict_class": "disqualified",
                "required_full_suite": "failed",
                "coverage_percent": 27,
            }
        },
        "verifier_is_oracle": True,
        "claim_scope": "exposed_development_boundary_only",
        "field_principles": {
            "source_boundary_ready_score": "Only current passed checks open compute.",
            "gate_check_summary": "Missing and wrong operands remain distinct.",
            "rows": "Each family counts once across both views.",
            "acceptance_gate_results": "No predictive benefit is measured.",
        },
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": receipt["invocation_counts"],
        "current_work_receipt": receipt,
        "source_boundary_ready_score": 0,
        "public_manifest_path": None,
        "evaluator_sidecar_path": None,
        "corpus_exposure": {
            "historically_exposed": True,
            "fresh_generalization_eligible": False,
            "family_count": 640,
        },
        "import_inventory": {"resolved_imports": {}},
    }


def run_experiment(date: str) -> dict[str, Any]:
    """Qualify canonical public features without using historical dispatchers."""
    start = time.monotonic()
    progress(start, "start", "begin")
    if date != "20260928":
        raise ValueError("run_date_mismatch")
    scope = json.loads(SCOPE.read_text())
    if [item["name"] for item in scope["commands"]] != list(REQUIRED):
        raise ValueError("validation_command_manifest_drift")
    if scope["task_id"] != "exp7838-source-boundary":
        raise ValueError("validation_task_drift")
    manifest, sources, failed = preflight(start)
    artifact = artifact_base(start, sources, failed)
    artifact["validation_command_manifest_sha256"] = sha256_file(SCOPE)
    if manifest is None:
        atomic_json(OUTPUT, artifact)
        progress(start, "terminal", "blocked")
        return artifact
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    progress(start, "custody", "begin")
    public, sidecar, evidence = custody(manifest, start)
    public_path = ATTEMPT / "public.jsonl"
    sidecar_path = ATTEMPT / "evaluator_sidecar.jsonl"
    features_path = ATTEMPT / "features.jsonl"
    for path, records in ((public_path, public), (sidecar_path, sidecar)):
        if path.exists() and source_projection.read_jsonl(path) != records:
            raise ValueError(f"checkpoint_input_changed:{path}")
        if not path.exists():
            source_projection.write_jsonl(path, records)
    if features_path.exists():
        source_projection.replay_file(public_path, features_path)
        progress(start, "extraction", "checkpoint_reused", 640)
    else:
        extraction = isolated_extract(public_path, features_path, start)
        artifact["observed_child_commands"].append(extraction)
        if not extraction["passed"]:
            raise ValueError("public_extraction_failed")
    source_projection.replay_file(public_path, features_path)
    features = source_projection.read_jsonl(features_path)
    source_projection.validate_features(public, features, [row["family_id"] for row in evidence])
    progress(start, "isolation", "begin")
    mutations = 0
    for index, (row, private, feature) in enumerate(zip(public, sidecar, features, strict=True), 1):
        altered = {
            **row,
            "role": "mutated",
            "label": 999,
            "confidence": -1,
            "generator_identity": "mutated",
            "annotations": [{"start": 999}],
            "unknown_label": -999,
        }
        if (
            source_projection.public_row(altered) != row
            or source_projection.extract_row(row)["feature_hash"] != feature["feature_hash"]
        ):
            raise ValueError(f"metadata_leak:{index}")
        if private["family_id"] != row["family_id"]:
            raise ValueError(f"sidecar_join_drift:{index}")
        evidence[index - 1].update(
            public_path=str(public_path),
            evaluator_path=str(sidecar_path),
            feature_path=str(features_path),
            feature_sha256=feature["feature_hash"],
            view_a_windows=feature["view_a_windows"],
            view_b_windows=feature["view_b_windows"],
            abstention=feature["abstention"],
        )
        mutations += 1
        if index % 32 == 0:
            progress(start, "isolation", "rows", index)
    artifact["rows"] = evidence
    artifact["sample_size_budget"] = {
        "intended": 640,
        "eligible": 640,
        "started": 640,
        "completed": 640,
        "censored": 0,
        "excluded": 0,
        "independent": 640,
    }
    artifact["public_manifest_path"] = str(features_path.with_suffix(".manifest.json"))
    artifact["evaluator_sidecar_path"] = str(sidecar_path)
    artifact["public_manifest_sha256"] = sha256_file(features_path.with_suffix(".manifest.json"))
    artifact["evaluator_sidecar_sha256"] = sha256_file(sidecar_path)
    artifact["public_sha256"] = sha256_file(public_path)
    artifact["features_sha256"] = sha256_file(features_path)
    artifact["metadata_mutation_count"] = mutations
    artifact["import_inventory"] = {
        "direct_calls": [
            "source_projection.public_row",
            "source_projection.extract_row",
            "source_projection.replay_file",
            "source_projection.validate_features",
            "source_projection.masked_loss",
            "evidence_views.prepare_views",
            "source_alignment.pair_features",
            "current_work_receipt.build_current_work_receipt",
        ],
        "resolved_imports": {},
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "code": [sha256_file(ROOT / path) for path in scope["affected_source_closure"]],
            "inputs": [(item["path"], item["sha256"]) for item in sources],
            "config": sha256_file(SCOPE),
            "seed": 68138,
        }
    )
    progress(start, "validation", "begin")
    candidate_path = ATTEMPT / "candidate.json"
    for index, item in enumerate(scope["commands"]):
        if item["name"] in {"adversarial_verify", "strict_rows"}:
            atomic_json(candidate_path, artifact)
        receipt = child(item, index)
        artifact["validation_receipts"].append(receipt)
        artifact["observed_child_commands"].append(receipt)
        if item["name"] == "worktree_imports":
            artifact["import_inventory"]["resolved_imports"] = receipt.get("resolved_imports", {})
        progress(start, "validation", "completed", index + 1)
    health = child(scope["repository_health"], len(scope["commands"]))
    artifact["repository_health"]["current_repository_health_180s"] = health
    artifact["observed_child_commands"].append(health)
    artifact["flagged_adversarial"] = any(
        receipt["name"] == "adversarial_verify" and not receipt["passed"]
        for receipt in artifact["validation_receipts"]
    )
    checks_pass = all(receipt["passed"] for receipt in artifact["validation_receipts"])
    artifact["acceptance_gate_results"]["validity"] = checks_pass
    artifact["acceptance_gate_results"]["readiness"] = int(checks_pass)
    artifact["source_boundary_ready_score"] = int(checks_pass)
    artifact["verdict_class"] = "circular_positive" if checks_pass else "disqualified"
    artifact["honest_verdict"] = (
        "complete_circular_positive_source_boundary_readiness"
        if checks_pass
        else "complete_disqualified_required_checks"
    )
    artifact["duration_s"] = time.monotonic() - start
    atomic_json(OUTPUT, artifact)
    progress(start, "terminal", "published", 640)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Run one dated qualification or a bounded fixture CLI child."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--fixture-public", type=Path)
    parser.add_argument("--fixture-output", type=Path)
    args = parser.parse_args(argv)
    if args.fixture_public is not None and args.fixture_output is not None:
        source_projection.extract_file(args.fixture_public, args.fixture_output)
        source_projection.replay_file(args.fixture_public, args.fixture_output)
        return 0
    if args.date is None:
        parser.error("--date is required")
    run_experiment(args.date)
    return 0


if __name__ == "__main__":
    sys.exit(main())

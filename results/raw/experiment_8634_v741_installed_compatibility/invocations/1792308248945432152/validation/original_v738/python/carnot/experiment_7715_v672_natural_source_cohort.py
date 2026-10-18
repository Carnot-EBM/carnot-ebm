"""Authenticate and reduce the V672 natural source cohort (REQ-REPORT-7715)."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import socket
import time
from typing import Any

from carnot.experiment_7423_v651_annotated_protocol import (
    DEFAULT_CACHE_ROOT,
    RAGTRUTH_COMMIT,
    authenticate_assets,
    load_release,
)
from carnot.reporting import natural_source_cohort as cohort
from carnot.reporting import fresh_relation_cohort as base
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7715_v672_natural_source_cohort")
OUTPUT = Path("results/experiment_7715_v672_natural_source_cohort.json")
SALT = "v672-natural-20260926"
ROLE_COUNTS = {
    "fit": 128,
    "tune": 40,
    "policy": 40,
    "retention": 32,
    "online_update": 60,
    "online_admission": 60,
    "evaluation": 40,
}
SCOPE = {
    "test_paths": ["tests/python/test_experiment_7715_v672_natural_source_cohort.py"],
    "changed_modules": [
        "python/carnot/reporting/natural_source_cohort.py",
        "python/carnot/experiment_7715_v672_natural_source_cohort.py",
    ],
    "static_paths": ["scripts/experiments/experiment_7715_v672_natural_source_cohort.py"],
    "specs": ["REQ-REPORT-7715", "REQ-VERIFY-7715"],
}
MODEL_SPECS: list[dict] = []
PRINCIPLE = "Measured evidence bounds the claim and prevents invalid downstream use."
FIELDS = (
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "sample_size_budget",
    "inference_substrate",
    "inference_substrate_class",
    "MODEL_SPECS",
    "model_invoked",
    "execution_venue",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "preconditions_checked",
    "validation_receipts",
    "verifier_is_oracle",
    "natural_cohort_ready_score",
    "fresh_source_score",
    "role_manifest_path",
    "label_definition",
)
GATES = (
    "validity",
    "readiness",
    "probability",
    "utility",
    "coverage",
    "source_dependence",
    "retention",
    "efficiency",
)


def progress(started: float, phase: str, event: str, units: int = 0) -> None:
    """Flush a phase boundary or loop heartbeat with elapsed time and units."""
    print(
        f"[exp7715] {phase} {event} units={units} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )


def check(name: str, upstream: str, path: Path, field: str, expected: Any, observed: Any) -> dict:
    """Keep exact blocking operands beside their external authority."""
    return {
        "check": name,
        "upstream": upstream,
        "path": str(path.resolve()),
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def _phase(
    spans: list[dict], started: float, previous: float, name: str, units: int, checkpoint: Path
) -> float:
    """Close one disjoint monotonic phase and retain its durable checkpoint."""
    now = time.monotonic()
    spans.append(
        {
            "phase": name,
            "start_s": previous - started,
            "end_s": now - started,
            "duration_s": now - previous,
            "heartbeat_timestamp_s": now - started,
            "completed_units": units,
            "checkpoint": str(checkpoint),
        }
    )
    progress(started, name, "complete", units)
    return now


def _v651_exposure(root: Path, started: float) -> tuple[set[str], list[dict], list[dict]]:
    """Read only source IDs from every authenticated V651 evaluator shard."""
    manifest_path = (
        root / "results/raw/experiment_7423_v651_annotated_protocol/corpus_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = [
        check(
            "v651_manifest_schema",
            "exp7423",
            manifest_path,
            "schema",
            "carnot.exp7423.corpus_manifest.v1",
            manifest["schema"],
        )
    ]
    receipts = [
        {"path": str(manifest_path.relative_to(root)), "sha256": sha256_file(manifest_path)}
    ]
    ids: set[str] = set()
    for shard in manifest["shards"]:
        if shard["kind"] != "evaluator":
            continue
        path = manifest_path.parent / shard["path"]
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            check("v651_source_role_hash", "exp7423", path, "sha256", shard["sha256"], observed)
        )
        if observed != shard["sha256"]:
            continue
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                # The projection uses only the public source_id column.
                source_id = json.loads(line)["source_id"]
                ids.add(source_id)
                count += 1
                if count % 5000 == 0:
                    progress(started, "exposure", "v651_rows", len(ids))
        checks.append(check("v651_source_role_rows", "exp7423", path, "rows", shard["rows"], count))
        receipts.append({"path": str(path.relative_to(root)), "sha256": observed, "rows": count})
        progress(started, "exposure", "v651_shard", len(ids))
    return ids, checks, receipts


def _later_exposure_inventory(root: Path, started: float) -> tuple[set[str], list[dict]]:
    """Enumerate V651–V671 public source-role and inspected-pilot files."""
    ids: set[str] = set()
    receipts: list[dict] = []
    marker = re.compile(r"^experiment_\d+_v6(\d{2})_")
    for directory in sorted((root / "results/raw").glob("experiment_*_v6*_*")):
        match = marker.match(directory.name)
        if match is None or not 51 <= int(match.group(1)) <= 71:
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
                continue
            if not any(
                token in path.name
                for token in (
                    "manifest",
                    "protocol",
                    "model_inputs",
                    "predictor",
                    "pilot",
                    "panel",
                )
            ):
                continue
            payload = path.read_bytes()
            for source_id in re.findall(rb'"source_id"\s*:\s*"([^"]+)"', payload):
                if source_id.isdigit():
                    ids.add(source_id.decode("ascii"))
            receipts.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": sha256_file(path),
                    "bytes": len(payload),
                }
            )
            progress(started, "exposure", "manifest_checked", len(receipts))
    return ids, receipts


def _gates(valid: bool, ready: bool, eligible: int, fresh: bool) -> list[dict]:
    """Report measured cohort operands without inventing scientific quality."""
    observed = {
        "validity": valid,
        "readiness": ready,
        "probability": None,
        "utility": None,
        "coverage": eligible,
        "source_dependence": None,
        "retention": None,
        "efficiency": None,
    }
    expected = {
        "validity": True,
        "readiness": True,
        "probability": "measured",
        "utility": "measured",
        "coverage": 400,
        "source_dependence": "measured",
        "retention": "measured",
        "efficiency": "measured",
    }
    return [
        {
            "gate": name,
            "observed": observed[name],
            "operator": "==" if observed[name] is not None else "measured",
            "expected": expected[name],
            "passed": observed[name] == expected[name],
            "principle": PRINCIPLE,
        }
        for name in GATES
    ]


def build_candidate(root: Path, raw_dir: Path, run_date: str) -> dict:
    """Authenticate actual bytes, inventory families, and freeze a candidate."""
    started = time.monotonic()
    progress(started, "startup", "begin")
    root = root.resolve(strict=True)
    if root != ROOT or run_date != "20260926":
        raise ValueError("repo_or_date_mismatch")
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(raw_dir / "frozen_affected_scope.json", SCOPE)
    spans: list[dict] = []
    previous = started
    checks: list[dict] = []
    hashes: dict[str, Any] = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    progress(started, "preconditions", "before_release_authentication")
    receipt = authenticate_assets(DEFAULT_CACHE_ROOT)
    for file in receipt["files"]:
        path = Path(file["cache_path"])
        checks.append(
            check(
                "ragtruth_release_hash",
                RAGTRUTH_COMMIT,
                path,
                "sha256",
                file["sha256"],
                sha256_file(path),
            )
        )
        hashes["valid_producers"][str(path)] = file["sha256"]
    manifest_path = (
        root / "results/raw/experiment_7423_v651_annotated_protocol/corpus_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks.append(
        check("ragtruth_license", "exp7423", manifest_path, "license", "MIT", manifest["license"])
    )
    checks.append(
        check(
            "ragtruth_commit",
            "exp7423",
            manifest_path,
            "commit",
            RAGTRUTH_COMMIT,
            manifest["commit"],
        )
    )
    checks.append(check("cpu_resource", "host", root, "cpu_available", True, bool(os.cpu_count())))
    previous = _phase(
        spans,
        started,
        previous,
        "preconditions",
        len(checks),
        raw_dir / "frozen_affected_scope.json",
    )
    progress(started, "public_inventory", "before_release_decode")
    source_rows, response_rows = load_release(receipt, started=started)
    families, excluded = cohort.build_public_families(source_rows, response_rows, SALT)
    previous = _phase(
        spans,
        started,
        previous,
        "public_inventory",
        len(families),
        raw_dir / "frozen_affected_scope.json",
    )
    progress(started, "exposure", "before_manifest_scan")
    v651_ids, v651_checks, v651_files = _v651_exposure(root, started)
    checks.extend(v651_checks)
    later_ids, later_files = _later_exposure_inventory(root, started)
    for item in (*v651_files, *later_files):
        hashes["valid_producers"][item["path"]] = item["sha256"]
    exposed = v651_ids | later_ids
    planned = cohort.subtract_and_assign(families, exposed, ROLE_COUNTS, SALT)
    source_ids = {row["source_id"] for row in source_rows}
    checks.append(
        check(
            "v651_complete_source_exposure",
            "exp7423",
            manifest_path,
            "exposed_source_ids",
            len(source_ids),
            len(source_ids & v651_ids),
        )
    )
    for split in ("train", "test"):
        required = sum(
            n for role, n in ROLE_COUNTS.items() if (role == "evaluation") == (split == "test")
        )
        checks.append(
            check(
                "fresh_role_capacity",
                "RAGTruth official " + split,
                manifest_path,
                f"eligible_{split}_families",
                required,
                planned["candidate_counts"][split],
            )
        )
    previous = _phase(
        spans,
        started,
        previous,
        "exposure_and_roles",
        len(exposed),
        raw_dir / "frozen_affected_scope.json",
    )
    windows_path = root / "results/raw/experiment_7714_v672_alignment_protocol/protocol.json"
    windows_link = {
        "path": str(windows_path.relative_to(root)),
        "sha256": sha256_file(windows_path) if windows_path.is_file() else None,
    }
    checks.append(
        check(
            "windows_protocol_exists",
            "exp7714",
            windows_path,
            "exists",
            True,
            windows_path.is_file(),
        )
    )
    for name in ("7701", "7704", "7705", "7709", "7712"):
        matches = sorted((root / "results").glob(f"experiment_{name}_v671_*.json"))
        for path in matches:
            prior = json.loads(path.read_text(encoding="utf-8"))
            hashes["flagged_historical_evidence"][str(path.relative_to(root))] = {
                "sha256": sha256_file(path),
                "honest_verdict": prior.get("honest_verdict"),
                "verdict_class": prior.get("verdict_class"),
                "flagged_adversarial": prior.get("flagged_adversarial"),
            }
    rows = [
        {
            "unit_id": family["family_id"],
            "role": None,
            "arm": "natural_source_inventory",
            "official_split": family["official_split"],
            "raw_metrics": {
                "response_count": family["member_count"],
                "source_count": len(family["source_ids"]),
            },
            "denominator": 1,
            "excluded": True,
            "censored": False,
            "exclusion_reason": "prior_exposure"
            if set(family["source_ids"]) & exposed
            else "unselected",
            "provenance": "authenticated RAGTruth public source and response identity",
        }
        for family in families
    ]
    candidate_manifest = {
        "schema": "carnot.exp7715.v672.candidate_manifest.v1",
        "salt": SALT,
        "role_counts": ROLE_COUNTS,
        "observed_candidate_counts": planned["candidate_counts"],
        "shortages": planned["shortages"],
        "source_count": len(source_ids),
        "v651_exposed_source_ids": len(v651_ids),
        "all_exposed_source_ids": len(exposed),
        "excluded_families": len(excluded),
        "family_ids": [row["family_id"] for row in families],
        "source_role_files": [*v651_files, *later_files],
        "windows_protocol": windows_link,
        "label_definition": "1 iff at least one human-annotated unsupported response span",
    }
    atomic_json(raw_dir / "candidate_manifest.json", candidate_manifest)
    manifest_hash = sha256_file(raw_dir / "candidate_manifest.json")
    previous = _phase(
        spans,
        started,
        previous,
        "candidate_manifest",
        len(families),
        raw_dir / "candidate_manifest.json",
    )
    failures = [item for item in checks if not item["passed"]]
    blocked = bool(failures or any(planned["shortages"].values()))
    artifact = {
        "schema": "carnot.exp7715.v672.natural_source_cohort.v1",
        "experiment_id": "exp7715-natural-source-cohort",
        "milestone": "2026.09.672",
        "date": run_date,
        "honest_verdict": "complete_blocked_fresh_role_shortage"
        if blocked
        else "complete_null_natural_cohort_ready",
        "verdict_class": "blocked" if blocked else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": _gates(
            not blocked, not blocked, sum(planned["candidate_counts"].values()), not blocked
        ),
        "rows": rows,
        "sample_size_budget": {
            "intended_families": 400,
            "observed_families": len(families),
            "eligible_families": sum(planned["candidate_counts"].values()),
            "excluded_families": len(families)
            - sum(planned["candidate_counts"].values())
            + len(excluded),
            "censored_families": 0,
            "intended_roles": ROLE_COUNTS,
            "observed_roles": Counter(row["role"] for row in planned["selected"]),
            "candidate_counts": planned["candidate_counts"],
            "shortages": planned["shortages"],
            "prior_exposure_families": planned["excluded_prior_exposure_count"],
            "effective_blocks": len(planned["selected"]),
            "seeds_windows_arms_are_independent_families": False,
        },
        "inference_substrate": "no_model_load",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": MODEL_SPECS,
        "model_specs": MODEL_SPECS,
        "model_invoked": False,
        "invocation_counts": {
            name: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
            for name in ("model_loads", "forward_calls", "generation_calls", "tokens")
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "backend": "cpu",
            "gpu_uuid": None,
        },
        "phase_spans": spans,
        "random_seed": {"role_salt": SALT, "purpose": "label-blind hash order"},
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "predeclared_scope": SCOPE,
            "required_checks": [],
            "terminal_readers": [],
            "repository_health": {"global_debt": "not measured by this task"},
        },
        "verifier_is_oracle": False,
        "natural_cohort_ready_score": int(not blocked),
        "fresh_source_score": int(not blocked),
        "role_manifest_path": str((raw_dir / "candidate_manifest.json").relative_to(root))
        if raw_dir.is_relative_to(root)
        else str(raw_dir / "candidate_manifest.json"),
        "role_manifest_sha256": manifest_hash,
        "label_definition": "1 iff at least one human-annotated unsupported response span",
        "windows_protocol_link": windows_link,
        "exposure_ledger": {
            "v651_exposed_source_ids": len(v651_ids),
            "later_exposed_source_ids": len(later_ids),
            "source_role_files": [*v651_files, *later_files],
            "uncertainties": [],
        },
        "excluded_source_families": excluded,
        "field_principles": {
            **{name: PRINCIPLE for name in FIELDS},
            **{name: PRINCIPLE for name in GATES},
        },
    }
    artifact["reproducibility_checksum"] = base.stable_hash(
        {
            "release": receipt["files"],
            "manifest_sha256": manifest_hash,
            "salt": SALT,
            "roles": ROLE_COUNTS,
            "reducer_sha256": sha256_file(
                root / "python/carnot/reporting/natural_source_cohort.py"
            ),
        }
    )
    return artifact


def cold_reduce(artifact: dict, raw_dir: Path) -> dict:
    """Check saved roster and blocked operands without trusting producer counts."""
    manifest_path = raw_dir / "candidate_manifest.json"
    if sha256_file(manifest_path) != artifact["role_manifest_sha256"]:
        raise ValueError("role_manifest_hash_mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = artifact["rows"]
    if [row["unit_id"] for row in rows] != manifest["family_ids"]:
        raise ValueError("family_roster_mismatch")
    if (
        sum(manifest["observed_candidate_counts"].values())
        != artifact["sample_size_budget"]["eligible_families"]
    ):
        raise ValueError("eligible_count_mismatch")
    failures = [item for item in artifact["preconditions_checked"] if not item["passed"]]
    if failures != artifact["gate_check_summary"]:
        raise ValueError("gate_summary_mismatch")
    if any(manifest["shortages"].values()):
        if artifact["verdict_class"] != "blocked" or artifact["natural_cohort_ready_score"] != 0:
            raise ValueError("shortage_promoted")
    return {
        "verdict_class": artifact["verdict_class"],
        "families": len(rows),
        "eligible": sum(manifest["observed_candidate_counts"].values()),
        "shortages": manifest["shortages"],
    }


def run_experiment(root: Path, run_date: str) -> dict:
    """Validate current work, cold-read a candidate, then publish atomically."""
    started = time.monotonic()
    progress(started, "run", "begin")
    root = root.resolve(strict=True)
    raw_dir = root / RAW
    artifact = build_candidate(root, raw_dir, run_date)
    progress(started, "validation", "before_scoped_subprocesses", len(artifact["rows"]))
    private = Path(f"/tmp/carnot7715-validation-{os.getpid()}")
    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    scoped = validation.run_scoped_validation(
        root,
        SCOPE["test_paths"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=private / "basetemp",
        coverage_file=private / ".coverage",
        log_dir=raw_dir / "validation" / "affected",
        historical_failures=[
            {
                "experiment_id": "exp7673",
                "issue": "prior required coverage failed",
                "resolved": False,
            }
        ],
    )
    artifact["validation_receipts"]["required_checks"] = scoped["validation_receipts"]
    artifact["validation_receipts"]["required_checks_passed"] = scoped["required_checks_passed"]
    artifact["validation_receipts"]["repository_health"] = scoped["repository_health"]
    progress(started, "validation", "after_scoped_subprocesses", len(scoped["validation_receipts"]))
    if not scoped["required_checks_passed"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["natural_cohort_ready_score"] = 0
        artifact["fresh_source_score"] = 0
        artifact["acceptance_gate_results"] = _gates(
            False, False, artifact["sample_size_budget"]["eligible_families"], False
        )
    candidate = raw_dir / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    terminal = [
        validation.CommandSpec(
            "independent_cold_replay",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/experiments/experiment_7715_v672_natural_source_cohort.py",
                "--cold-replay",
                str(candidate),
            ),
            "exact_candidate",
            900,
        ),
        validation.CommandSpec(
            "adversarial_verify",
            (str(root / ".venv/bin/python"), "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            900,
        ),
        validation.CommandSpec(
            "verdict_row_consistency_strict",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            900,
        ),
    ]
    progress(started, "terminal_readers", "before_subprocesses", len(terminal))
    receipts = validation.run_commands(root, terminal, log_dir=raw_dir / "validation" / "terminal")
    artifact["validation_receipts"]["terminal_readers"] = receipts
    artifact["flagged_adversarial"] = not receipts[1]["passed"]
    if not all(item["passed"] for item in receipts):
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["verdict_class"] = "disqualified"
        artifact["natural_cohort_ready_score"] = 0
        artifact["fresh_source_score"] = 0
        artifact["acceptance_gate_results"] = _gates(
            False, False, artifact["sample_size_budget"]["eligible_families"], False
        )
    progress(started, "terminal_readers", "after_subprocesses", len(receipts))
    atomic_json(root / OUTPUT, artifact)
    progress(started, "publication", "complete", len(artifact["rows"]))
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Run the dated experiment or replay one saved candidate in a new process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--cold-replay")
    args = parser.parse_args(argv)
    if args.cold_replay:
        path = Path(args.cold_replay).resolve(strict=True)
        candidate = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(cold_reduce(candidate, path.parent), sort_keys=True), flush=True)
        return 0
    artifact = run_experiment(ROOT, args.date)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "verdict_class": artifact["verdict_class"],
            }
        ),
        flush=True,
    )
    return 0

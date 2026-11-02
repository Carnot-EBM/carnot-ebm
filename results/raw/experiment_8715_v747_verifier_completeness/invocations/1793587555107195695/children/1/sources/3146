"""Seal exposed RAGTruth development families (REQ-REPORT-7727)."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

from carnot.experiment_7423_v651_annotated_protocol import (
    DEFAULT_CACHE_ROOT,
    RAGTRUTH_COMMIT,
    authenticate_assets,
    join_release,
    load_release,
    reload_corpus,
)
from carnot.reporting import natural_source_cohort as cohort
from carnot.reporting import fresh_relation_cohort as base
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7727_v673_development_corpus")
OUTPUT = Path("results/experiment_7727_v673_development_corpus.json")
SALT = "v673-development-20260926"
COUNTS = {
    "fit": 256,
    "tune": 64,
    "policy": 64,
    "online_update": 96,
    "online_admission": 64,
    "evaluation": 64,
    "retention": 32,
}
SCOPE = {
    "test_paths": ["tests/python/test_experiment_7727_v673_development_corpus.py"],
    "changed_modules": ["python/carnot/experiment_7727_v673_development_corpus.py"],
    "static_paths": ["scripts/experiments/experiment_7727_v673_development_corpus.py"],
    "specs": ["REQ-REPORT-7727", "REQ-VERIFY-7727"],
}
PUBLIC = frozenset(
    {
        "family_id",
        "role",
        "official_split",
        "source_id",
        "response_id",
        "complete_source",
        "complete_response",
        "source_sha256",
        "response_sha256",
        "previously_exposed",
        "fresh_generalization_eligible",
    }
)
PRINCIPLE = "Measured evidence bounds the claim and downstream use."


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Flush a real phase boundary or completed-unit heartbeat."""
    print(
        f"[exp7727] {phase} {event} units={units} elapsed_s={time.monotonic() - start:.3f}",
        flush=True,
    )


def select_development(
    sources: list[dict], responses: list[dict], counts: dict[str, int], pilot_response_ids: set[str]
) -> tuple[list[dict], dict[str, int]]:
    """Choose one response per deduplicated family without reading labels."""
    families, _ = cohort.build_public_families(sources, responses, SALT)
    pilot_sources = {row["source_id"] for row in responses if row["id"] in pilot_response_ids}
    candidates = [row for row in families if not set(row["source_ids"]) & pilot_sources]
    capacity = Counter(row["official_split"] for row in candidates)
    needed = {
        "train": sum(n for role, n in counts.items() if role != "evaluation"),
        "test": counts.get("evaluation", 0),
    }
    observed = {split: capacity[split] for split in needed}
    if any(observed[split] < needed[split] for split in needed):
        return [], observed
    selected = []
    for split in ("train", "test"):
        ranked = sorted(
            (row for row in candidates if row["official_split"] == split),
            key=lambda row: base.digest(SALT + ":" + row["family_id"]),
        )
        offset = 0
        for role, count in counts.items():
            if (role == "evaluation") == (split == "test"):
                selected.extend({**row, "role": role} for row in ranked[offset : offset + count])
                offset += count
    return selected, observed


def validate_public(row: dict, role: str) -> None:
    """Deny evaluator fields, missing text, role swaps, and byte drift."""
    if set(row) != PUBLIC or row["role"] != role:
        raise ValueError("public_fields_or_role_mismatch")
    if (role == "evaluation") != (row["official_split"] == "test"):
        raise ValueError("official_split_mismatch")
    if not row["complete_source"] or not row["complete_response"]:
        raise ValueError("missing_source_or_response_bytes")
    if row["source_sha256"] != base.digest(row["complete_source"]):
        raise ValueError("source_hash_mismatch")
    if row["response_sha256"] != base.digest(row["complete_response"]):
        raise ValueError("response_hash_mismatch")
    if row["previously_exposed"] is not True or row["fresh_generalization_eligible"] is not False:
        raise ValueError("exposure_mismatch")


def _jsonl(path: Path, rows: list[dict], mode: int) -> str:
    """Write deterministic row bytes and return their exact digest."""
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.chmod(path, mode)
    return sha256_file(path)


def seal_development(
    raw: Path, selected: list[dict], responses: list[dict], counts: dict[str, int]
) -> dict:
    """Freeze public rows before opening validated human annotations."""
    if len({row["family_id"] for row in selected}) != len(selected):
        raise ValueError("duplicate_family")
    if Counter(row["role"] for row in selected) != Counter(counts):
        raise ValueError("role_count_mismatch")
    raw.mkdir(parents=True, exist_ok=True)
    roles: dict[str, dict] = {}
    for role, count in counts.items():
        members = [row for row in selected if row["role"] == role]
        public = []
        for member in members:
            view = member["view"]
            row = {
                key: view[key]
                for key in (
                    "source_id",
                    "response_id",
                    "complete_source",
                    "complete_response",
                    "official_split",
                )
            }
            row.update(
                family_id=member["family_id"],
                role=role,
                source_sha256=base.digest(view["complete_source"]),
                response_sha256=base.digest(view["complete_response"]),
                previously_exposed=True,
                fresh_generalization_eligible=False,
            )
            validate_public(row, role)
            public.append(row)
        name = f"{role}_public.jsonl"
        roles[role] = {
            "count": count,
            "public_path": name,
            "public_sha256": _jsonl(raw / name, public, 0o644),
            "families": [row["family_id"] for row in public],
        }
    atomic_json(raw / "public_manifest.json", {"roles": roles, "counts": counts, "salt": SALT})
    # The shipped loader validates quality and every human span before labels are copied.
    by_id = {row["id"]: row for row in responses}
    joined, _ = join_release(
        [
            {
                "source_id": row["view"]["source_id"],
                "task_type": "Summary",
                "source_info": row["view"]["complete_source"],
            }
            for row in selected
        ],
        [by_id[row["view"]["response_id"]] for row in selected],
    )
    labels = {row["response_id"]: row for row in joined}
    for role in counts:
        members = [row for row in selected if row["role"] == role]
        evaluator = []
        for member in members:
            response_id = member["view"]["response_id"]
            annotations = labels[response_id]["annotations"]
            evaluator.append(
                {
                    "family_id": member["family_id"],
                    "response_id": response_id,
                    "annotations": annotations,
                    "label": int(any(not bool(span.get("implicit_true")) for span in annotations)),
                }
            )
        name = f"{role}_evaluator.jsonl"
        roles[role].update(
            evaluator_path=name, evaluator_sha256=_jsonl(raw / name, evaluator, 0o600)
        )
    manifest = {
        "schema": "carnot.exp7727.development_manifest.v1",
        "salt": SALT,
        "counts": counts,
        "roles": roles,
        "public_manifest_sha256": sha256_file(raw / "public_manifest.json"),
        "initially_open_evaluator_roles": ["fit"],
        "online_blocks": {
            "update": [
                roles["online_update"]["families"][i : i + 12]
                for i in range(0, counts.get("online_update", 0), 12)
            ]
            if "online_update" in roles
            else [],
            "admission": [
                roles["online_admission"]["families"][i : i + 8]
                for i in range(0, counts.get("online_admission", 0), 8)
            ]
            if "online_admission" in roles
            else [],
        },
    }
    atomic_json(raw / "development_manifest.json", manifest)
    atomic_json(
        raw / "evaluator_custody_manifest.json",
        {
            "roles": {
                role: {"path": item["evaluator_path"], "sha256": item["evaluator_sha256"]}
                for role, item in roles.items()
            },
            "initially_open_roles": ["fit"],
        },
    )
    return manifest


def cold_reduce(path: Path, counts: dict[str, int]) -> dict:
    """Reopen exact public and evaluator bytes and independently check roles."""
    raw = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["counts"] != counts or manifest["salt"] != SALT:
        raise ValueError("manifest_contract_mismatch")
    if sha256_file(raw / "public_manifest.json") != manifest["public_manifest_sha256"]:
        raise ValueError("public_manifest_hash_mismatch")
    public_manifest = json.loads((raw / "public_manifest.json").read_text(encoding="utf-8"))
    if public_manifest["counts"] != counts or public_manifest["roles"] != {
        role: {key: value for key, value in item.items() if not key.startswith("evaluator_")}
        for role, item in manifest["roles"].items()
    }:
        raise ValueError("public_manifest_mismatch")
    families: set[str] = set()
    source_hashes: set[str] = set()
    response_hashes: set[str] = set()
    for role, count in counts.items():
        info = manifest["roles"][role]
        public_path = raw / info["public_path"]
        evaluator_path = raw / info["evaluator_path"]
        if sha256_file(public_path) != info["public_sha256"]:
            raise ValueError("public_hash_mismatch")
        if sha256_file(evaluator_path) != info["evaluator_sha256"]:
            raise ValueError("evaluator_hash_mismatch")
        rows = [json.loads(line) for line in public_path.read_text(encoding="utf-8").splitlines()]
        labels = [
            json.loads(line) for line in evaluator_path.read_text(encoding="utf-8").splitlines()
        ]
        if (
            len(rows) != count
            or len(labels) != count
            or info["families"] != [r["family_id"] for r in rows]
        ):
            raise ValueError("role_count_or_roster_mismatch")
        for row, label in zip(rows, labels, strict=True):
            validate_public(row, role)
            if (
                row["family_id"] in families
                or row["source_sha256"] in source_hashes
                or row["response_sha256"] in response_hashes
            ):
                raise ValueError("duplicate_family")
            families.add(row["family_id"])
            source_hashes.add(row["source_sha256"])
            response_hashes.add(row["response_sha256"])
            if (
                label["family_id"] != row["family_id"]
                or label["response_id"] != row["response_id"]
                or label["label"]
                != int(any(not bool(a.get("implicit_true")) for a in label["annotations"]))
            ):
                raise ValueError("label_join_mismatch")
    return {"families": len(families), "role_counts": counts, "isolated": True}


def _check(name: str, upstream: str, path: Path, field: str, expected: Any, observed: Any) -> dict:
    """Name the exact operand that could block an external prerequisite."""
    return {
        "check": name,
        "upstream_id": upstream,
        "artifact_path": str(path),
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def build_candidate(root: Path, raw: Path, run_date: str) -> dict:
    """Authenticate inputs, seal a fixed roster, and record an administrative null."""
    start = time.monotonic()
    progress(start, "preconditions", "begin")
    if root.resolve(strict=True) != ROOT or run_date != "20260926":
        raise ValueError("root_or_date_mismatch")
    raw.mkdir(parents=True, exist_ok=True)
    scope_path = raw / "frozen_affected_scope.json"
    atomic_json(scope_path, SCOPE)
    checks: list[dict] = []
    hashes: dict[str, Any] = {
        "eligible_producers": {},
        "flagged_historical_inputs": {},
        "pre_gate_receipts": {},
        "absent_sources": ["SciHal private corpus"],
    }
    spans: list[dict] = []
    previous = start

    def phase(name: str, units: int, checkpoint: Path) -> None:
        nonlocal previous
        now = time.monotonic()
        spans.append(
            {
                "phase": name,
                "start_s": previous - start,
                "end_s": now - start,
                "duration_s": now - previous,
                "run_date": run_date,
                "heartbeat_times_s": [now - start],
                "completed_units": units,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
            }
        )
        previous = now
        progress(start, name, "complete", units)

    progress(start, "preconditions", "before_release_authentication")
    receipt = authenticate_assets(DEFAULT_CACHE_ROOT)
    progress(start, "preconditions", "after_release_authentication", len(receipt["files"]))
    for item in receipt["files"]:
        path = Path(item["cache_path"])
        checks.append(
            _check(
                "release_hash", RAGTRUTH_COMMIT, path, "sha256", item["sha256"], sha256_file(path)
            )
        )
        hashes["eligible_producers"][str(path)] = item["sha256"]
    v651 = root / "results/raw/experiment_7423_v651_annotated_protocol/corpus_manifest.json"
    progress(start, "preconditions", "before_v651_custody_reload")
    v651_reloaded = reload_corpus(v651.parent)
    progress(start, "preconditions", "after_v651_custody_reload", len(v651_reloaded["evaluators"]))
    v651_data = json.loads(v651.read_text(encoding="utf-8"))
    checks.extend(
        [
            _check(
                "v651_schema",
                "exp7423",
                v651,
                "schema",
                "carnot.exp7423.corpus_manifest.v1",
                v651_data.get("schema"),
            ),
            _check(
                "v651_commit", "exp7423", v651, "commit", RAGTRUTH_COMMIT, v651_data.get("commit")
            ),
            _check("v651_license", "exp7423", v651, "license", "MIT", v651_data.get("license")),
            _check("host_cpu", "host", root, "cpu_available", True, bool(os.cpu_count())),
        ]
    )
    hashes["eligible_producers"][str(v651.relative_to(root))] = sha256_file(v651)
    prior = root / "results/experiment_7715_v672_natural_source_cohort.json"
    prior_data = json.loads(prior.read_text(encoding="utf-8"))
    checks.append(
        _check(
            "v672_exposed_families",
            "exp7715",
            prior,
            "observed_families",
            2894,
            prior_data["sample_size_budget"]["observed_families"],
        )
    )
    checks.append(
        _check(
            "v672_fresh_eligible",
            "exp7715",
            prior,
            "eligible_families",
            0,
            prior_data["sample_size_budget"]["eligible_families"],
        )
    )
    hashes["pre_gate_receipts"][str(prior.relative_to(root))] = sha256_file(prior)
    pilot_path = root / "results/raw/experiment_7716_v672_qwen_semantic_pilot/frozen_panel.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    checks.append(_check("pilot_families", "exp7716", pilot_path, "count", 24, len(pilot)))
    hashes["flagged_historical_inputs"][str(pilot_path.relative_to(root))] = sha256_file(pilot_path)
    phase("preconditions", len(checks), scope_path)

    progress(start, "inventory", "before_release_decode")
    sources, responses = load_release(receipt, started=start)
    families, excluded = cohort.build_public_families(sources, responses, SALT)
    checks.append(_check("exposed_inventory", "exp7715", prior, "families", 2894, len(families)))
    pilot_ids = {row["response_id"] for row in pilot}
    selected, capacity = select_development(sources, responses, COUNTS, pilot_ids)
    for split, needed in (("train", 576), ("test", 64)):
        checks.append(
            _check(
                "development_role_capacity",
                "RAGTruth",
                v651,
                f"{split}_families_at_least",
                True,
                capacity[split] >= needed,
            )
        )
    phase("inventory", len(families), scope_path)
    blocked = any(not row["passed"] for row in checks)
    if not blocked:
        progress(start, "seal", "before_evaluator_materialization", len(selected))
        manifest = seal_development(raw, selected, responses, COUNTS)
        replay = cold_reduce(raw / "development_manifest.json", COUNTS)
        progress(start, "seal", "after_evaluator_materialization", replay["families"])
        manifest_path = raw / "development_manifest.json"
    else:
        manifest = {"roles": {}, "counts": COUNTS}
        manifest_path = raw / "development_manifest.json"
        atomic_json(manifest_path, {"blocked": True, "checks": checks, "capacity": capacity})
    phase("seal", len(selected), manifest_path)
    rows = [
        {
            "unit_id": item["family_id"],
            "family_id": item["family_id"],
            "role": item["role"],
            "official_split": item["official_split"],
            "source_id": item["view"]["source_id"],
            "response_id": item["view"]["response_id"],
            "source_sha256": base.digest(item["view"]["complete_source"]),
            "response_sha256": base.digest(item["view"]["complete_response"]),
            "raw_metrics": {
                "source_bytes": len(item["view"]["complete_source"].encode()),
                "response_bytes": len(item["view"]["complete_response"].encode()),
            },
            "denominator": 1,
            "censored": False,
            "excluded": False,
            "previously_exposed": True,
            "fresh_generalization_eligible": False,
            "exposure_reason": "V651 evaluator inventory; V672 full exposure accounting",
            "exposure_evidence_paths": [str(v651.relative_to(root)), str(prior.relative_to(root))],
        }
        for item in selected
    ]
    rows_path = raw / "rows.jsonl"
    _jsonl(rows_path, rows, 0o644)
    phase("raw_rows", len(rows), rows_path)
    gate_values = {
        "validity": not blocked,
        "readiness": not blocked,
        "brier_score": None,
        "decision_cost": None,
        "coverage": None,
        "retention": None,
        "efficiency": None,
    }
    gates = [
        {
            "gate": key,
            "observed": value,
            "expected": True if key in ("validity", "readiness") else "measured",
            "operator": "==" if value is not None else "measured",
            "passed": value is True,
            "principle": PRINCIPLE,
        }
        for key, value in gate_values.items()
    ]
    field_names = [
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "sample_size_budget",
        "claim_scope",
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
        "development_cohort_ready_score",
        "development_manifest_path",
        "fresh_generalization_eligible",
        "exposure_accounting",
    ]
    artifact = {
        "schema": "carnot.exp7727.v673.development_corpus.v1",
        "experiment_id": "exp7727-development-corpus",
        "milestone": "2026.09.673",
        "run_date": run_date,
        "honest_verdict": "complete_blocked_development_capacity_or_custody"
        if blocked
        else "complete_null_development_corpus_ready",
        "verdict_class": "blocked" if blocked else "null",
        "flagged_adversarial": False,
        "gate_check_summary": [row for row in checks if not row["passed"]],
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended_families": 640,
            "observed_families": len(families),
            "eligible_development_families": sum(capacity.values()),
            "eligible_fresh_families": 0,
            "excluded_families": len(excluded) + 24,
            "censored_families": 0,
            "effective_independent_families": len(rows),
            "intended_roles": COUNTS,
            "observed_roles": dict(Counter(r["role"] for r in rows)),
            "candidate_counts": capacity,
            "shortages": {
                "train": max(0, 576 - capacity["train"]),
                "test": max(0, 64 - capacity["test"]),
            },
            "seeds_windows_arms_increase_independent_n": False,
        },
        "claim_scope": "development_only",
        "fresh_generalization_eligible": False,
        "inference_substrate": "no_model_load",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": {
            name: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
            for name in (
                "model_loads",
                "forward_calls",
                "generation_calls",
                "input_tokens",
                "output_tokens",
            )
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "backend": "cpu",
            "gpu_uuid": None,
        },
        "phase_spans": spans,
        "duration_s": time.monotonic() - start,
        "random_seed": {"role_hash_salt": SALT, "purpose": "label-blind deterministic role order"},
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks
        + [
            {
                "check": "effective_coding_backend",
                "observed": os.environ.get(
                    "CODEX_MODEL", "coding-agent backend not exposed to process"
                ),
                "passed": True,
            }
        ],
        "validation_receipts": {
            "predeclared_scope": SCOPE,
            "required_checks": [],
            "terminal_readers": [],
            "repository_health": {"global_suite_debt": "separate; not a science-readiness gate"},
        },
        "verifier_is_oracle": False,
        "development_cohort_ready_score": int(not blocked),
        "development_manifest_path": str(manifest_path.relative_to(root))
        if manifest_path.is_relative_to(root)
        else str(manifest_path),
        "development_manifest_sha256": sha256_file(manifest_path),
        "development_manifest": {
            "role_counts": COUNTS,
            "source_level_overlap_count": 0,
            "sha256": sha256_file(manifest_path),
        },
        "exposure_accounting": {
            "observed_exposed_families": len(families),
            "fresh_eligible": 0,
            "pilot_excluded": 24,
            "per_family": {
                row["family_id"]: {
                    "reason": row["exposure_reason"],
                    "evidence_paths": row["exposure_evidence_paths"],
                }
                for row in rows
            },
            "inventory": len(families),
            "evaluator_materialized": len(rows),
            "fitting": 0,
            "inspected_predictions": 0,
        },
        "deferred_corpus_work": {
            "SciHal": "access and license review deferred; no private download or registration",
            "synthetic_labels": "not a substitute for human annotations",
        },
        "field_principles": {name: PRINCIPLE for name in (*field_names, *gate_values)},
    }
    artifact["reproducibility_checksum"] = base.stable_hash(
        {
            "salt": SALT,
            "counts": COUNTS,
            "release": receipt["files"],
            "manifest_sha256": artifact["development_manifest_sha256"],
            "rows_sha256": sha256_file(rows_path),
            "reducer_sha256": sha256_file(
                root / "python/carnot/experiment_7727_v673_development_corpus.py"
            ),
        }
    )
    return artifact


def cold_replay(path: Path) -> dict:
    """Reduce a saved terminal candidate from exact raw rows in a fresh process."""
    value = json.loads(path.read_text(encoding="utf-8"))
    declared = Path(value["development_manifest_path"])
    raw = (declared if declared.is_absolute() else ROOT / declared).parent
    manifest_path = raw / "development_manifest.json"
    if sha256_file(manifest_path) != value["development_manifest_sha256"]:
        raise ValueError("development_manifest_hash_mismatch")
    rows = [
        json.loads(line) for line in (raw / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if rows != value["rows"]:
        raise ValueError("raw_row_mismatch")
    if value["verdict_class"] != "blocked":
        reduced = cold_reduce(manifest_path, COUNTS)
        if reduced["families"] != len(rows) or len(rows) != 640:
            raise ValueError("family_count_mismatch")
    elif not value["gate_check_summary"]:
        raise ValueError("blocked_without_gate")
    if any(row["fresh_generalization_eligible"] for row in rows):
        raise ValueError("fresh_claim_mismatch")
    return {"families": len(rows), "verdict_class": value["verdict_class"]}


def run_experiment(root: Path, run_date: str) -> dict:
    """Run registered checks and exact terminal readers before atomic publication."""
    started = time.monotonic()
    progress(started, "run", "begin")
    raw = root / RAW
    artifact = build_candidate(root, raw, run_date)
    private = Path(f"/tmp/carnot7727-validation-{os.getpid()}")
    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    progress(started, "validation", "before_scoped_subprocesses", len(artifact["rows"]))
    scoped = validation.run_scoped_validation(
        root,
        SCOPE["test_paths"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=private / "basetemp",
        coverage_file=private / ".coverage",
        log_dir=raw / "validation" / "affected",
        historical_failures=[],
    )
    artifact["validation_receipts"]["required_checks"] = scoped["validation_receipts"]
    artifact["validation_receipts"]["required_checks_passed"] = scoped["required_checks_passed"]
    artifact["validation_receipts"]["repository_health"] = scoped["repository_health"]
    progress(started, "validation", "after_scoped_subprocesses", len(scoped["validation_receipts"]))
    if not scoped["required_checks_passed"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["development_cohort_ready_score"] = 0
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    python = str(root / ".venv/bin/python")
    terminal = [
        validation.CommandSpec(
            "independent_cold_replay",
            (
                python,
                "-u",
                "scripts/experiments/experiment_7727_v673_development_corpus.py",
                "--cold-replay",
                str(candidate),
            ),
            "exact_candidate",
            900,
        ),
        validation.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            900,
        ),
        validation.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            900,
        ),
    ]
    progress(started, "terminal", "before_readers", len(terminal))
    receipts = validation.run_commands(root, terminal, log_dir=raw / "validation" / "terminal")
    progress(started, "terminal", "after_readers", len(receipts))
    artifact["validation_receipts"]["terminal_readers"] = receipts
    artifact["flagged_adversarial"] = (
        not receipts[1]["passed"] or "FLAGGED" in receipts[1]["output_tail"]
    )
    if not all(row["passed"] for row in receipts) or artifact["flagged_adversarial"]:
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["verdict_class"] = "disqualified"
        artifact["development_cohort_ready_score"] = 0
    for gate in artifact["acceptance_gate_results"]:
        if gate["gate"] == "readiness":
            gate["observed"] = bool(artifact["development_cohort_ready_score"])
            gate["passed"] = gate["observed"]
    artifact["duration_s"] = time.monotonic() - started
    atomic_json(root / OUTPUT, artifact)
    progress(started, "publication", "complete", len(artifact["rows"]))
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Run the dated producer or cold replay the exact saved candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        print(json.dumps(cold_replay(args.cold_replay), sort_keys=True), flush=True)
        return 0
    run_experiment(ROOT, args.date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

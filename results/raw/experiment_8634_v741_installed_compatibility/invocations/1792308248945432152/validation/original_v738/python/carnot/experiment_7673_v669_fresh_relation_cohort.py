"""Freeze a label-isolated fresh relation cohort (REQ-REPORT-7673)."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import time
from typing import Any

from carnot.experiment_7533_v659_tool_protocol import (
    DATA_ROOT,
    DATASET_REVISION,
    PINNED_SHARD_HASHES,
    README_HASH,
    annotation_binary_label,
)
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting import fresh_relation_cohort as cohort


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7673_v669_fresh_relation_cohort")
OUTPUT = Path("results/experiment_7673_v669_fresh_relation_cohort.json")
SALT = "v669-fresh-relations-20260926"
ROLE_COUNTS = {
    "fit": 128,
    "retention": 32,
    "tune": 40,
    "policy": 40,
    "online_update": 80,
    "online_admission": 80,
    "evaluation": 80,
}
SCOPE = {
    "test_paths": ["tests/python/test_experiment_7673_v669_fresh_relation_cohort.py"],
    "changed_modules": [
        "python/carnot/reporting/fresh_relation_cohort.py",
        "python/carnot/experiment_7673_v669_fresh_relation_cohort.py",
    ],
    "static_paths": ["scripts/experiments/experiment_7673_v669_fresh_relation_cohort.py"],
    "specs": ["REQ-REPORT-7673", "REQ-VERIFY-7673"],
}
MODEL_SPECS: list[dict[str, Any]] = []
ZERO_CALLS = {
    name: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
    for name in ("model_loads", "forward_calls", "generation_calls", "tokens")
}
PRINCIPLES = {
    "honest_verdict": "Completion describes the task, not a verifier benefit.",
    "verdict_class": "Blocked resources and invalid checks have different meanings.",
    "flagged_adversarial": "A reader flag cannot open a gate.",
    "gate_check_summary": "Every block names an exact failed operand.",
    "acceptance_gate_results": "Validity, readiness, freshness, and benefit use separate operands.",
    "rows": "One family per source arm; controls never enlarge n.",
    "sample_size_budget": "Count independent source families before and after exclusions.",
    "inference_substrate_class": "Actual current work loads no model.",
    "MODEL_SPECS": "Current model identities are empty, distinct from historical provenance.",
    "model_invoked": "Zero current loads, forwards, generations, and tokens.",
    "execution_venue": "The host and owned PID identify this CPU run.",
    "phase_spans": "Monotonic disjoint spans describe actual work, without padding.",
    "random_seed": "The salt determines every public role before labels open.",
    "reproducibility_checksum": "Bind input bytes, configuration, and reducer code.",
    "source_artifact_hashes": "Input receipts exclude this run's planned outputs.",
    "preconditions_checked": "A missing upstream resource is reported with exact operands.",
    "validation_receipts": "Only actual subprocess exits and byte hashes count.",
    "verifier_is_oracle": "Injected-error truth is an evaluator annotation, not natural truth.",
    "fresh_source_score": "Fresh credit requires 480 unexposed disjoint families.",
    "relation_features_ready_score": "Feature readiness requires isolated complete three-arm rows.",
    "protocol_path": "The protocol freezes roles and claim scope before labels open.",
    "role_counts": "Counts are source families, not rows or arms.",
    "exposure_ledger": "Prior identities and uncertainty govern fresh eligibility.",
    "label_provenance": "Dataset labels describe injected errors only.",
}


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Flush each boundary and periodic count before long CPU work."""
    print(
        f"[exp7673] {phase} {event} units={units} elapsed_s={time.monotonic() - start:.3f}",
        flush=True,
    )


def check(
    check_name: str, upstream: str, path: Path, field: str, expected: Any, observed: Any
) -> dict:
    """Keep exact blocked operands beside the resource that can fix them."""
    return {
        "check": check_name,
        "upstream": upstream,
        "path": str(path.resolve()),
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preconditions(root: Path, start: float) -> tuple[list[dict], dict]:
    """Authenticate immutable corpus and repository files without labels."""
    checks: list[dict] = []
    receipts: dict[str, Any] = {"producers": {}, "pre_gate_receipts": {}, "missing_evidence": []}
    for name, expected in PINNED_SHARD_HASHES.items():
        path = DATA_ROOT / "data" / name
        observed = sha256_file(path).removeprefix("sha256:") if path.is_file() else None
        checks.append(
            check("pinned_shard_hash", DATASET_REVISION, path, "sha256", expected, observed)
        )
        if observed is None:
            receipts["missing_evidence"].append(str(path))
        else:
            receipts["producers"][str(path)] = "sha256:" + observed
        progress(start, "preconditions", "shard_checked", len(checks))
    readme = DATA_ROOT / "README.md"
    observed = sha256_file(readme).removeprefix("sha256:") if readme.is_file() else None
    checks.append(
        check("pinned_readme_hash", DATASET_REVISION, readme, "sha256", README_HASH, observed)
    )
    if observed is not None:
        receipts["producers"][str(readme)] = "sha256:" + observed
    else:
        receipts["missing_evidence"].append(str(readme))
    for label in ("AGENTS.md", "CODEX.md", "CLAUDE.md", "ops/exclusion_manifest.yaml"):
        path = root / label
        checks.append(check("repo_input_exists", "worktree", path, "exists", True, path.is_file()))
        if path.is_file():
            receipts["producers"][label] = sha256_file(path)
    return checks, receipts


EXPOSURE_INPUTS = {
    "experiment_7533_v659_tool_protocol": ["predictor.jsonl"],
    "experiment_7575_v662_cached_learning_protocol": ["exposure_manifest.json"],
    "experiment_7602_v664_evidence_requalification": ["*_model_inputs.jsonl", "protocol.json"],
    "experiment_7616_v665_evidence_schema": ["canonical_inputs.json"],
    "experiment_7646_v667_source_feature_corpus": ["manifest.json"],
    "experiment_7659_v668_atom_corpus": ["manifest.json"],
}


def _scan_identity(value: Any, found: dict[str, set[str]]) -> None:
    """Collect only public identity fields from registered lineage records."""
    if isinstance(value, dict):
        for key, item in value.items():
            if "label" in key.lower() or key in {"truth", "probability", "diagnostics"}:
                continue
            if isinstance(item, str):
                if key in {
                    "context",
                    "source",
                    "source_text",
                    "complete_source",
                    "original_source",
                }:
                    found["source_hashes"].add(cohort.digest(cohort.normalized(item)))
                    found["template_hashes"].add(cohort.digest(cohort.source_template(item)))
                    found["exact_source_hashes"].add(cohort.digest(item))
                elif key in {"answer", "complete_answer", "response_text"}:
                    found["answer_hashes"].add(cohort.digest(cohort.normalized(item)))
                    found["exact_answer_hashes"].add(cohort.digest(item))
                elif key in {"context_sha256", "source_sha256", "original_source_sha256"}:
                    found["exact_source_hashes"].add(item)
                elif key in {"answer_sha256", "response_sha256"}:
                    found["exact_answer_hashes"].add(item)
                elif key in {"source_hash", "normalized_source_hash"}:
                    found["source_hashes"].add(item)
                elif key in {"answer_hash", "response_hash"}:
                    found["answer_hashes"].add(item)
                elif key in {"instance_id", "source_id"}:
                    found["instance_ids"].add(item)
            elif isinstance(item, (dict, list)):
                _scan_identity(item, found)
    elif isinstance(value, list):
        for item in value:
            _scan_identity(item, found)


def exposure_ledger(root: Path, start: float) -> tuple[dict, dict[str, set[str]], list[dict]]:
    """Hash and scan registered public lineage; name every missing manifest."""
    found = {
        name: set()
        for name in (
            "source_hashes",
            "template_hashes",
            "exact_source_hashes",
            "answer_hashes",
            "exact_answer_hashes",
            "family_ids",
            "instance_ids",
        )
    }
    files: list[dict] = []
    missing: list[dict] = []
    for dirname, patterns in EXPOSURE_INPUTS.items():
        directory = root / "results/raw" / dirname
        for pattern in patterns:
            paths = sorted(directory.glob(pattern))
            if not paths:
                missing.append(
                    check(
                        "prior_manifest_exists",
                        dirname,
                        directory / pattern,
                        "matches",
                        True,
                        False,
                    )
                )
            for path in paths:
                before = {name: len(values) for name, values in found.items()}
                with path.open(encoding="utf-8") as stream:
                    if path.suffix == ".jsonl":
                        for line in stream:
                            if line.strip():
                                _scan_identity(json.loads(line), found)
                    else:
                        _scan_identity(json.load(stream), found)
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                        "new_identities": {
                            name: len(values) - before[name] for name, values in found.items()
                        },
                    }
                )
                progress(start, "exposure", "file_checked", len(files))
    ledger = {
        "files": files,
        "prior_source_hashes": sorted(found["source_hashes"] | found["exact_source_hashes"]),
        "prior_family_hashes": sorted(found["template_hashes"] | found["family_ids"]),
        "prior_answer_hashes": sorted(found["answer_hashes"] | found["exact_answer_hashes"]),
        "known_exposures": {name: len(values) for name, values in found.items()},
        "uncertainties": [item["path"] for item in missing],
        "lineage_note": "V660-V668 derived captures reuse V659 or V664 rosters; outcome-bearing stores are excluded from pre-label reads.",
    }
    return ledger, found, missing


def _load_public_rows(start: float) -> list[dict]:
    """Read only public Parquet columns, preserving row addresses for later labels."""
    import pyarrow.parquet as parquet

    output = []
    for name in PINNED_SHARD_HASHES:
        path = DATA_ROOT / "data" / name
        table = parquet.read_table(
            path, columns=["context", "question", "answer", "split", "dataset", "metadata"]
        )
        for index, row in enumerate(table.to_pylist()):
            if row["dataset"] == "lettucedetect-tool-output":
                row["official_split"] = row.pop("split")
                row["_shard"] = name
                row["_row_index"] = index
                output.append(row)
        progress(start, "inventory", "shard_complete", len(output))
    return output


def write_jsonl(path: Path, rows: list[dict]) -> str:
    """Replace one owned store only after all row bytes are complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)
    return sha256_file(path)


def read_jsonl(path: Path) -> list[dict]:
    """Read a sealed role store without broadening to another role."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seal_public_cohort(root: Path, selected: list[dict], start: float) -> tuple[dict, list[dict]]:
    """Write input and feature stores and seal the protocol before labels open."""
    raw_dir = root / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    roles: dict[str, dict] = {}
    all_features = []
    for role, expected in ROLE_COUNTS.items():
        members = [item for item in selected if item["role"] == role]
        if len(members) != expected:
            raise ValueError(f"underfilled_{role}:{len(members)}<{expected}")
        inputs = [cohort.predictor_view(item["view"], item["family_id"], role) for item in members]
        features = cohort.feature_rows(inputs, role)
        input_path = raw_dir / f"{role}_model_inputs.jsonl"
        feature_path = raw_dir / f"{role}_features.jsonl"
        input_hash = write_jsonl(input_path, inputs)
        feature_hash = write_jsonl(feature_path, features)
        roster = [item["family_id"] for item in members]
        roles[role] = {
            "families": roster,
            "role_hash": cohort.stable_hash(roster),
            "model_inputs": input_path.relative_to(root).as_posix(),
            "model_inputs_sha256": input_hash,
            "features": feature_path.relative_to(root).as_posix(),
            "features_sha256": feature_hash,
            "group_count": len(roster),
            "row_count": len(features),
        }
        all_features.extend(features)
        atomic_json(raw_dir / "checkpoints" / f"{role}.json", roles[role])
        progress(start, "feature_rows", "role_complete", len(roles))
    protocol = {
        "schema": "carnot.exp7673.v669.protocol.v1",
        "dataset_revision": DATASET_REVISION,
        "selection_salt": SALT,
        "role_counts": ROLE_COUNTS,
        "roles": roles,
        "feature_rule": "verify_relations on intact, erased, and next-family same-role source",
        "feature_module_sha256": sha256_file(
            root / "python/carnot/reporting/fresh_relation_cohort.py"
        ),
        "relation_module_sha256": sha256_file(
            root / "python/carnot/verify/tool_source_relations.py"
        ),
        "label_semantics": "binary presence of injected-error annotation spans; evaluator only",
        "claim_restrictions": [
            "no natural-response accuracy claim",
            "no inherited Qwen probabilities",
            "no whole-answer certification",
        ],
        "accessor_boundary": {
            "features": "model_inputs only",
            "labels": "evaluator_stores after protocol freeze",
        },
    }
    atomic_json(raw_dir / "protocol.json", protocol)
    progress(start, "protocol", "frozen", 480)
    return protocol, all_features


def seal_evaluators(root: Path, selected: list[dict], protocol: dict, start: float) -> dict:
    """Open annotation columns only after the public protocol exists on disk."""
    if not (root / RAW / "protocol.json").is_file():
        raise ValueError("protocol_not_frozen")
    import pyarrow.parquet as parquet

    labels_by_shard = {}
    for name in PINNED_SHARD_HASHES:
        labels_by_shard[name] = (
            parquet.read_table(DATA_ROOT / "data" / name, columns=["labels"])
            .column("labels")
            .to_pylist()
        )
        progress(start, "evaluator_labels", "shard_opened", len(labels_by_shard))
    receipts = {}
    for role, details in protocol["roles"].items():
        members = [item for item in selected if item["role"] == role]
        rows = []
        for item in members:
            view = item["view"]
            answer = view["answer"]
            spans = labels_by_shard[view["_shard"]][view["_row_index"]]
            rows.append(
                {
                    "family_id": item["family_id"],
                    "role": role,
                    "role_hash": details["role_hash"],
                    "answer_sha256": cohort.digest(answer),
                    "label": annotation_binary_label(answer, spans),
                    "provenance": "LettuceDetect injected-error character spans",
                }
            )
        path = root / RAW / f"{role}_evaluator_store.jsonl"
        receipts[role] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": write_jsonl(path, rows),
            "count": len(rows),
        }
        progress(start, "evaluator_labels", "role_complete", len(receipts))
    atomic_json(root / RAW / "evaluator_manifest.json", receipts)
    return receipts


def cold_reduce(root: Path) -> dict:
    """Recompute every feature from sealed public inputs in a fresh process."""
    protocol = json.loads((root / RAW / "protocol.json").read_text())
    manifest = json.loads((root / RAW / "evaluator_manifest.json").read_text())
    groups = set()
    row_count = 0
    for role, details in protocol["roles"].items():
        inputs = root / details["model_inputs"]
        features = root / details["features"]
        evaluator = root / manifest[role]["path"]
        if any(
            (
                sha256_file(inputs) != details["model_inputs_sha256"],
                sha256_file(features) != details["features_sha256"],
                sha256_file(evaluator) != manifest[role]["sha256"],
            )
        ):
            raise ValueError("sealed_store_hash_mismatch")
        model_rows = read_jsonl(inputs)
        if (
            cohort.stable_hash([row["component_hash"] for row in model_rows])
            != details["role_hash"]
        ):
            raise ValueError("role_hash_mismatch")
        if cohort.feature_rows(model_rows, role) != read_jsonl(features):
            raise ValueError("feature_replay_mismatch")
        evaluator_rows = read_jsonl(evaluator)
        if len(evaluator_rows) != len(model_rows):
            raise ValueError("label_join_mismatch")
        for model, label in zip(model_rows, evaluator_rows, strict=True):
            if (model["component_hash"], role, details["role_hash"], model["answer_sha256"]) != (
                label["family_id"],
                label["role"],
                label["role_hash"],
                label["answer_sha256"],
            ):
                raise ValueError("label_join_mismatch")
            if label["label"] not in (0, 1) or model["component_hash"] in groups:
                raise ValueError("label_or_family_collision")
            groups.add(model["component_hash"])
        row_count += len(model_rows) * len(cohort.ARMS)
    if len(groups) != 480 or row_count != 1440:
        raise ValueError("roster_count_mismatch")
    return {
        "families": len(groups),
        "rows": row_count,
        "role_hashes": {role: value["role_hash"] for role, value in protocol["roles"].items()},
    }


def gate_results(
    ready: bool, validity: bool, freshness: bool, groups: int, rows: int
) -> list[dict]:
    """Keep infrastructure, probability, and utility evidence independent."""
    operands = {
        "validity": (validity, True),
        "readiness": (ready, True),
        "coverage": (rows, 1440),
        "freshness": (groups if freshness else 0, 480),
        "probability": (None, "measured fresh-row probabilities"),
        "decision_utility": (None, "measured fresh-row paired utility"),
        "retention": (None, "measured delayed retention"),
        "efficiency": (None, "measured model or service cost"),
    }
    return [
        {
            "gate": name,
            "observed": observed,
            "expected": expected,
            "operator": "=="
            if name in {"validity", "readiness", "coverage", "freshness"}
            else "measured",
            "passed": observed == expected
            if name in {"validity", "readiness", "coverage", "freshness"}
            else False,
            "principle": PRINCIPLES["acceptance_gate_results"],
        }
        for name, (observed, expected) in operands.items()
    ]


def base_artifact(
    root: Path, run_date: str, checks: list[dict], hashes: dict, start: float
) -> dict:
    """Make blocked and ready terminal records share one complete schema."""
    failures = [item for item in checks if not item["passed"]]
    return {
        "experiment_id": "7673",
        "milestone": "2026.09.669",
        "date": run_date,
        "schema": "carnot.exp7673.v669.fresh_relation_cohort.v1",
        "honest_verdict": "complete_blocked_precondition"
        if failures
        else "complete_null_fresh_relation_cohort",
        "verdict_class": "blocked" if failures else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": gate_results(False, not failures, False, 0, 0),
        "rows": [],
        "sample_size_budget": {
            "intended_independent_groups": 480,
            "observed_independent_groups": 0,
            "eligible_independent_groups": 0,
            "excluded_independent_groups": 0,
            "censored_independent_groups": 0,
            "prior_exposure_groups": 0,
            "effective_blocks": 0,
            "limits": "480 fixed families; no role shrinkage",
        },
        "inference_substrate": "no_model_load",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": [],
        "model_specs": [{"declaration": "no_current_model", "model_id": None}],
        "model_invoked": False,
        "invocation_counts": ZERO_CALLS,
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
        },
        "phase_spans": [],
        "duration_s": time.monotonic() - start,
        "random_seed": {
            "role_salt": SALT,
            "purpose": "public family hash order; no stochastic inference",
        },
        "reproducibility_checksum": None,
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": SCOPE,
            "required_checks": [],
            "terminal_readers": [],
        },
        "verifier_is_oracle": False,
        "field_principles": PRINCIPLES,
        "fresh_source_score": 0,
        "relation_features_ready_score": 0,
        "protocol_path": RAW.joinpath("protocol.json").as_posix(),
        "role_counts": ROLE_COUNTS,
        "exposure_ledger": {
            "files": [],
            "prior_source_hashes": [],
            "prior_family_hashes": [],
            "prior_answer_hashes": [],
            "known_exposures": {},
            "uncertainties": [],
            "exclusions": [],
            "selected_family_ids": [],
        },
        "label_provenance": {
            "dataset": "LettuceDetect tool-output",
            "semantics": "injected-error spans",
            "natural_hallucination_truth": False,
        },
        "claim_scope": "cohort infrastructure only; no learned-verifier improvement",
        "historical_model_provenance": "unsloth/Qwen3.8-27B-GGUF in V659-V668; not invoked now",
        "retire_if_same_verdict": {
            "mechanism": "fresh relation cohort",
            "same_verdict": False,
            "resource_absence_is_science": False,
        },
    }


def validate_candidate(path: Path) -> dict:
    """Cold-check result custody and forbid output as an immutable input."""
    value = json.loads(path.read_text())
    if not str(value["honest_verdict"]).startswith("complete_"):
        raise ValueError("nonterminal_verdict")
    if (
        value["MODEL_SPECS"]
        or value["model_invoked"]
        or value["inference_substrate_class"] != "no_model_load"
    ):
        raise ValueError("false_model_provenance")
    if str(path.resolve()) in value["source_artifact_hashes"].get("producers", {}):
        raise ValueError("output_self_input")
    if value["verdict_class"] == "null":
        reduced = cold_reduce(ROOT)
        if reduced["families"] != 480 or len(value["rows"]) != 1440:
            raise ValueError("candidate_row_mismatch")
        if Counter(row["arm"] for row in value["rows"]) != Counter(
            {arm: 480 for arm in cohort.ARMS}
        ):
            raise ValueError("candidate_arm_mismatch")
    return {"verdict_class": value["verdict_class"], "row_count": len(value["rows"])}


def run_experiment(root: Path, run_date: str, output: Path) -> dict:
    """Authenticate, freeze, reduce, validate, and atomically publish V669."""
    started = time.monotonic()
    progress(started, "startup", "begin")
    root = root.resolve(strict=True)
    if root != ROOT or run_date != "20260926":
        raise ValueError("repo_or_date_mismatch")
    raw_dir = root / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(raw_dir / "frozen_affected_scope.json", SCOPE)
    checks, hashes = preconditions(root, started)
    artifact = base_artifact(root, run_date, checks, hashes, started)
    spans = []
    previous = started

    def finish(phase: str, units: int) -> None:
        nonlocal previous
        now = time.monotonic()
        spans.append(
            {
                "phase": phase,
                "start_s": previous - started,
                "end_s": now - started,
                "duration_s": now - previous,
                "completed_units": units,
                "heartbeat_timestamp_s": now - started,
            }
        )
        previous = now
        progress(started, phase, "complete", units)

    finish("preconditions", len(checks))
    selected: list[dict] = []
    if all(item["passed"] for item in checks):
        public = _load_public_rows(started)
        families, collisions = cohort.cluster_public_rows(public)
        finish("public_inventory", len(public))
        ledger, exposed, missing = exposure_ledger(root, started)
        for item in missing:
            checks.append(item)
        eligible, exclusions = cohort.subtract_exposure(families, exposed)
        try:
            selected = cohort.assign_roles(eligible, ROLE_COUNTS, SALT)
        except ValueError as exc:
            checks.append(
                check(
                    "fixed_role_counts",
                    "pinned public families",
                    DATA_ROOT,
                    "role_allocation",
                    "480 disjoint families",
                    str(exc),
                )
            )
        ledger["exclusions"] = collisions + exclusions
        ledger["selected_family_ids"] = [item["family_id"] for item in selected]
        artifact["exposure_ledger"] = ledger
        artifact["sample_size_budget"] = {
            "intended_independent_groups": 480,
            "observed_independent_groups": len(selected),
            "eligible_independent_groups": len(eligible),
            "excluded_independent_groups": len(collisions) + len(exclusions),
            "censored_independent_groups": 0,
            "prior_exposure_groups": len(exclusions),
            "effective_blocks": len(selected),
            "limits": "480 fixed families; no role shrinkage; three views share one denominator",
            "official_inventory_rows": len(public),
            "clustered_families": len(families),
            "cross_split_collisions": len(collisions),
        }
        hashes["producers"].update({item["path"]: item["sha256"] for item in ledger["files"]})
        finish("exposure_and_roles", len(eligible))
    if all(item["passed"] for item in checks) and len(selected) == 480:
        protocol, features = seal_public_cohort(root, selected, started)
        artifact["rows"] = features
        artifact["role_hashes"] = {
            role: value["role_hash"] for role, value in protocol["roles"].items()
        }
        artifact["sample_size_budget"]["censored_independent_groups"] = len(
            {row["unit_id"] for row in features if row["censored"]}
        )
        finish("public_protocol", len(features))
        evaluator_receipts = seal_evaluators(root, selected, protocol, started)
        artifact["evaluator_stores"] = evaluator_receipts
        finish("evaluator_isolation", sum(item["count"] for item in evaluator_receipts.values()))
    else:
        failures = [item for item in checks if not item["passed"]]
        artifact["honest_verdict"] = (
            "complete_blocked_external_absence"
            if any(
                item["observed"] is None or item["check"] == "prior_manifest_exists"
                for item in failures
            )
            else "complete_disqualified_authentication"
        )
        artifact["verdict_class"] = (
            "blocked" if "blocked" in artifact["honest_verdict"] else "disqualified"
        )
        artifact["gate_check_summary"] = failures
        finish("blocked_or_disqualified", len(failures))
    hashes["pre_gate_receipts"]["frozen_affected_scope.json"] = sha256_file(
        raw_dir / "frozen_affected_scope.json"
    )
    artifact["preconditions_checked"] = checks
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = cohort.stable_hash(
        {
            "inputs": hashes["producers"],
            "scope": SCOPE,
            "salt": SALT,
            "reducer": sha256_file(
                root / "python/carnot/experiment_7673_v669_fresh_relation_cohort.py"
            ),
        }
    )
    basetemp = Path(tempfile.mkdtemp(prefix="carnot-7673-pytest-"))
    coverage_file = basetemp / ".coverage"
    commands = validation.build_scoped_commands(
        root,
        SCOPE["test_paths"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=basetemp,
        coverage_file=coverage_file,
    )
    progress(started, "affected_validation", "before_subprocess", len(commands))
    receipts = validation.run_commands(
        root,
        commands,
        log_dir=raw_dir / "validation" / "affected",
        extra_env={"COVERAGE_FILE": str(coverage_file)},
    )
    artifact["validation_receipts"]["required_checks"] = receipts
    artifact["validation_receipts"]["required_checks_passed"] = validation.reduce_required_checks(
        receipts
    )["required_checks_passed"]
    finish("affected_validation", len(receipts))
    if not artifact["validation_receipts"]["required_checks_passed"]:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["fresh_source_score"] = 0
        artifact["relation_features_ready_score"] = 0
    elif artifact["verdict_class"] == "null":
        artifact["fresh_source_score"] = int(
            len(selected) == 480 and not artifact["exposure_ledger"]["uncertainties"]
        )
        artifact["relation_features_ready_score"] = 1
    artifact["acceptance_gate_results"] = gate_results(
        artifact["relation_features_ready_score"] == 1,
        artifact["validation_receipts"]["required_checks_passed"],
        artifact["fresh_source_score"] == 1,
        len(selected),
        len(artifact["rows"]),
    )
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    candidate = raw_dir / "exact_terminal_candidate.json"
    atomic_json(candidate, artifact)
    progress(started, "terminal_readers", "candidate_frozen", len(artifact["rows"]))
    terminal = [
        validation.CommandSpec(
            "cold_replay",
            (
                str(root / ".venv/bin/python"),
                "-u",
                str(root / "scripts/experiments/experiment_7673_v669_fresh_relation_cohort.py"),
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
    reader_receipts = validation.run_commands(
        root, terminal, log_dir=raw_dir / "validation" / "terminal"
    )
    artifact["validation_receipts"]["terminal_readers"] = reader_receipts
    artifact["flagged_adversarial"] = not reader_receipts[1]["passed"]
    if not all(item["passed"] for item in reader_receipts):
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["fresh_source_score"] = 0
        artifact["relation_features_ready_score"] = 0
    finish("terminal_readers", len(reader_receipts))
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    atomic_json(root / output, artifact)
    progress(started, "publication", "complete", len(artifact["rows"]))
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Run V669, or cold-read one immutable candidate without model work."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--cold-replay")
    args = parser.parse_args(argv)
    if args.cold_replay:
        print(
            json.dumps(validate_candidate(Path(args.cold_replay).resolve()), sort_keys=True),
            flush=True,
        )
        return 0
    run_experiment(ROOT, args.date, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())

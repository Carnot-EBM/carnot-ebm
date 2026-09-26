"""Seal a fresh label-isolated cohort for REQ-REPORT-7701."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import socket
import time
from typing import Any

from carnot.experiment_7533_v659_tool_protocol import (
    DATA_ROOT,
    PINNED_SHARD_HASHES,
    annotation_binary_label,
)
from carnot.experiment_7673_v669_fresh_relation_cohort import (
    _load_public_rows,
    check,
    exposure_ledger,
    preconditions,
)
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting import fresh_relation_cohort as base
from carnot.reporting import sealed_source_cohort as sealed
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7701_v671_sealed_cohort")
OUTPUT = Path("results/experiment_7701_v671_sealed_cohort.json")
SALT = "v671-20260926"
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
    "test_paths": [
        "tests/python/test_experiment_7701_v671_sealed_cohort.py",
        "tests/python/test_experiment_7673_v669_fresh_relation_cohort.py",
    ],
    "changed_modules": [
        "python/carnot/reporting/sealed_source_cohort.py",
        "python/carnot/experiment_7701_v671_sealed_cohort.py",
    ],
    "static_paths": ["scripts/experiments/experiment_7701_v671_sealed_cohort.py"],
    "specs": ["REQ-REPORT-7701", "REQ-VERIFY-7701"],
}
MODEL_SPECS: list[dict] = []
ZERO_CALLS = {
    name: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
    for name in ("model_loads", "forward_calls", "generation_calls", "tokens")
}
FIELDS = [
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
    "cohort_ready_score",
    "fresh_source_score",
    "selected_prior_exposure_count",
    "excluded_prior_exposure_count",
    "role_manifest_path",
    "evaluator_stores",
]
PRINCIPLES = {name: "Exact current evidence limits downstream claims." for name in FIELDS}
PRINCIPLES.update(
    {
        "honest_verdict": "A terminal disposition prevents retries of unchanged external blocks.",
        "verdict_class": "A closed enum carries claim eligibility into downstream readers.",
        "flagged_adversarial": "Disqualified evidence must not pass a downstream readiness gate.",
        "cohort_ready_score": "Only 400 valid, disjoint, label-isolated families count as ready.",
        "fresh_source_score": "Selected exposure and custody uncertainty must both be empty.",
        "selected_prior_exposure_count": "Count contamination among selected families only.",
        "excluded_prior_exposure_count": "Count families removed before selection separately.",
    }
)
GATE_PRINCIPLES = {
    "validity": "Invalid evidence must not propagate.",
    "readiness": "Administrative readiness needs a complete isolated cohort.",
    "coverage": "Fixed role sizes prevent outcome-driven shrinkage.",
    "freshness": "Prior exposure cannot earn fresh credit.",
    "probability": "Unmeasured quality cannot follow from plumbing.",
    "utility": "Unmeasured benefit cannot follow from plumbing.",
    "retention": "A gain cannot be inferred while forgetting is unmeasured.",
    "efficiency": "Unmeasured compute cost cannot support efficiency claims.",
}


def progress(started: float, phase: str, event: str, units: int = 0) -> None:
    """Emit each boundary with elapsed time and completed units."""
    print(
        f"[exp7701] {phase} {event} units={units} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )


def gates(valid: bool, ready: bool, groups: int, fresh: bool) -> list[dict]:
    """Keep infrastructure gates apart from unmeasured scientific quality."""
    values = {
        "validity": (valid, True),
        "readiness": (ready, True),
        "coverage": (groups, 400),
        "freshness": (fresh, True),
        "probability": (None, "measured calibrated probability"),
        "utility": (None, "measured paired decision utility"),
        "retention": (None, "measured delayed retention"),
        "efficiency": (None, "measured service cost"),
    }
    return [
        {
            "gate": name,
            "observed": observed,
            "operator": "=="
            if name in {"validity", "readiness", "coverage", "freshness"}
            else "measured",
            "expected": expected,
            "passed": observed == expected if observed is not None else False,
            "principle": GATE_PRINCIPLES[name],
        }
        for name, (observed, expected) in values.items()
    ]


def prior_selected(root: Path) -> tuple[set[str], list[dict], dict]:
    """Authenticate all V669 selections even though its coverage gate failed."""
    path = root / "results/raw/experiment_7673_v669_fresh_relation_cohort/protocol.json"
    result_path = root / "results/experiment_7673_v669_fresh_relation_cohort.json"
    checks = []
    receipts: dict[str, str] = {}
    for item in (path, result_path):
        checks.append(
            check("prior_producer_exists", "exp7673", item, "exists", True, item.is_file())
        )
        if item.is_file():
            receipts[item.relative_to(root).as_posix()] = sha256_file(item)
    if not all(item["passed"] for item in checks):
        return set(), checks, receipts
    protocol = json.loads(path.read_text(encoding="utf-8"))
    prior_result = json.loads(result_path.read_text(encoding="utf-8"))
    ids = [item for role in protocol["roles"].values() for item in role["families"]]
    checks.append(
        check("prior_selected_count", "exp7673", path, "selected_families", 480, len(set(ids)))
    )
    checks.append(
        check(
            "prior_result_roster_matches",
            "exp7673",
            result_path,
            "selected_family_ids",
            True,
            set(ids) == set(prior_result["exposure_ledger"]["selected_family_ids"]),
        )
    )
    return set(ids), checks, receipts


def base_artifact(run_date: str, checks: list[dict], hashes: dict, started: float) -> dict:
    """Give blocked and measured terminal results the same auditable schema."""
    failures = [item for item in checks if not item["passed"]]
    return {
        "experiment_id": "7701",
        "milestone": "2026.09.671",
        "date": run_date,
        "schema": "carnot.exp7701.v671.sealed_cohort.v1",
        "honest_verdict": "complete_blocked_precondition"
        if failures
        else "complete_null_cohort_readiness",
        "verdict_class": "blocked" if failures else "null",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": gates(not failures, False, 0, False),
        "rows": [],
        "sample_size_budget": {
            "intended_groups": 400,
            "observed_groups": 0,
            "eligible_groups": 0,
            "excluded_groups": 0,
            "censored_groups": 0,
            "effective_blocks": 0,
            "prior_exposure_groups": 0,
            "inference_limits": "Data readiness only; no probability, utility, retention or efficiency claim.",
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
        "effective_agent_backend": {
            "observed": "codex",
            "routing_request": "claude/opus",
            "forced_by_environment": os.environ.get("CODEX_FORCE_EXPERIMENTS") == "1",
            "successful_current_invocation_receipt": "current Codex agent tool execution; no backend activation claim",
            "activation_verified": False,
        },
        "phase_spans": [],
        "duration_s": time.monotonic() - started,
        "random_seed": {"role_salt": SALT, "purpose": "public family hash ordering"},
        "reproducibility_checksum": None,
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": SCOPE,
            "required_checks": [],
            "terminal_readers": [],
        },
        "verifier_is_oracle": True,
        "field_principles": {
            **PRINCIPLES,
            **{f"acceptance_gate_{name}": value for name, value in GATE_PRINCIPLES.items()},
        },
        "cohort_ready_score": 0,
        "fresh_source_score": 0,
        "selected_prior_exposure_count": 0,
        "excluded_prior_exposure_count": 0,
        "role_manifest_path": RAW.joinpath("protocol.json").as_posix(),
        "evaluator_stores": {},
        "role_counts": ROLE_COUNTS,
        "claim_scope": "Administrative data validity only; no source informativeness or natural-hallucination claim.",
        "historical_model_provenance": "V659-V669 Qwen lineage only; no current model invocation",
        "prior_failures": [
            {
                "experiment_id": "exp7687-sealed-cohort",
                "custody": "not_emitted_usage_limit_three_attempts",
            },
            {
                "experiment_id": "exp7673-fresh-relation-cohort",
                "verdict": "complete_disqualified_required_validation",
            },
        ],
    }


def label_reader(started: float):
    """Create a late-opening evaluator callback after public protocol freeze."""
    import pyarrow.parquet as parquet

    cache: dict[str, list] = {}

    def read(item: dict) -> int:
        view = item["view"]
        name = view["_shard"]
        if name not in cache:
            progress(started, "evaluator", "before_label_shard_open", len(cache))
            cache[name] = (
                parquet.read_table(DATA_ROOT / "data" / name, columns=["labels"])
                .column("labels")
                .to_pylist()
            )
            progress(started, "evaluator", "after_label_shard_open", len(cache))
        return annotation_binary_label(view["answer"], cache[name][view["_row_index"]])

    return read


def validate_candidate(path: Path) -> dict:
    """Cold-check the candidate and rehash its sealed role stores."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value["honest_verdict"].startswith("complete_"):
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
        protocol_path = ROOT / value["role_manifest_path"]
        if sha256_file(protocol_path) != value["role_manifest_sha256"]:
            raise ValueError("protocol_hash_mismatch")
        reduced = sealed.cold_reduce(protocol_path, ROLE_COUNTS)
        if reduced["families"] != 400 or len(value["rows"]) != 400:
            raise ValueError("candidate_row_mismatch")
        if Counter(row["role"] for row in value["rows"]) != Counter(ROLE_COUNTS):
            raise ValueError("candidate_role_mismatch")
        if value["selected_prior_exposure_count"] != 0:
            raise ValueError("selected_prior_exposure")
        return reduced
    return {"families": 0, "verdict_class": value["verdict_class"]}


def run_experiment(root: Path, run_date: str, output: Path) -> dict:
    """Authenticate, freeze, replay, validate, and atomically publish V671."""
    started = time.monotonic()
    progress(started, "startup", "begin")
    root = root.resolve(strict=True)
    if root != ROOT or run_date != "20260926":
        raise ValueError("repo_or_date_mismatch")
    raw_dir = root / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(raw_dir / "frozen_affected_scope.json", SCOPE)
    checks, hashes = preconditions(root, started)
    ids, prior_checks, prior_hashes = prior_selected(root)
    checks.extend(prior_checks)
    hashes["producers"].update(prior_hashes)
    artifact = base_artifact(run_date, checks, hashes, started)
    spans: list[dict] = []
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
                "heartbeat_timestamp_s": now - started,
                "completed_units": units,
                "checkpoint": str(raw_dir / "frozen_affected_scope.json")
                if phase == "preconditions"
                else None,
            }
        )
        previous = now
        progress(started, phase, "complete", units)

    finish("preconditions", len(checks))
    if all(item["passed"] for item in checks):
        progress(started, "public_inventory", "before_shard_reads")
        public = _load_public_rows(started)
        finish("public_inventory", len(public))
        ledger, exposed, missing = exposure_ledger(root, started)
        checks.extend(missing)
        exposed["family_ids"].update(ids)
        if not missing:
            try:
                planned = sealed.plan(public, exposed, ids, ROLE_COUNTS, SALT)
            except ValueError as exc:
                checks.append(
                    check(
                        "fixed_role_counts",
                        "pinned public families",
                        DATA_ROOT,
                        "role_allocation",
                        "400 disjoint families",
                        str(exc),
                    )
                )
            else:
                selected = planned["selected"]
                artifact["excluded_prior_exposure_count"] = planned["excluded_prior_exposure_count"]
                artifact["selected_prior_exposure_count"] = planned["selected_prior_exposure_count"]
                artifact["sample_size_budget"].update(
                    {
                        "observed_groups": len(selected),
                        "eligible_groups": planned["inventory"]["eligible_families"],
                        "excluded_groups": len(planned["collisions"]) + len(planned["exclusions"]),
                        "effective_blocks": len(selected),
                        "prior_exposure_groups": planned["excluded_prior_exposure_count"],
                        "inventory": planned["inventory"],
                    }
                )
                atomic_json(
                    raw_dir / "public_exclusions.json",
                    {"collisions": planned["collisions"], "prior_exposure": planned["exclusions"]},
                )
                hashes["producers"].update(
                    {item["path"]: item["sha256"] for item in ledger["files"]}
                )
                artifact["exposure_ledger"] = {
                    **ledger,
                    "uncertainties": [],
                    "prior_selected_count": len(ids),
                }
                finish("exposure_and_roles", len(selected))
                progress(started, "public_protocol", "before_seal", len(selected))
                protocol = sealed.seal(raw_dir, selected, ROLE_COUNTS, SALT, label_reader(started))
                finish("public_protocol_and_evaluators", len(selected))
                artifact["evaluator_stores"] = protocol["evaluator_stores"]
                artifact["role_manifest_sha256"] = sha256_file(raw_dir / "protocol.json")
                artifact["rows"] = [
                    {
                        "unit_id": item["family_id"],
                        "role": item["role"],
                        "arm": "public_cohort_selection",
                        "raw_metrics": {
                            "source_bytes": len(item["view"]["context"].encode()),
                            "answer_bytes": len(item["view"]["answer"].encode()),
                        },
                        "counts": {"family": 1, "members": item["member_count"]},
                        "excluded": False,
                        "censored": False,
                        "provenance": "authenticated pinned LettuceDetect public columns",
                    }
                    for item in selected
                ]
                progress(started, "cold_reduction", "before_subprocess", len(selected))
                # The independent process uses the on-disk protocol through --cold-replay below.
                reduced = sealed.cold_reduce(raw_dir / "protocol.json", ROLE_COUNTS)
                artifact["cold_reduction"] = reduced
                finish("cold_reduction", reduced["families"])
    failures = [item for item in checks if not item["passed"]]
    if failures:
        artifact["honest_verdict"] = "complete_blocked_precondition"
        artifact["verdict_class"] = "blocked"
    artifact["gate_check_summary"] = failures
    artifact["source_artifact_hashes"] = hashes
    artifact["reproducibility_checksum"] = base.stable_hash(
        {
            "source_hashes": hashes["producers"],
            "salt": SALT,
            "counts": ROLE_COUNTS,
            "reducer_sha256": sha256_file(root / "python/carnot/reporting/sealed_source_cohort.py"),
        }
    )
    progress(started, "validation", "before_scoped_subprocesses", len(artifact["rows"]))
    private = Path(f"/tmp/carnot7701-validation-{os.getpid()}")
    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    scope_result = validation.run_scoped_validation(
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
                "issue": "prior required coverage reported 92 percent",
                "resolved": False,
            }
        ],
    )
    artifact["validation_receipts"]["required_checks"] = scope_result["validation_receipts"]
    artifact["validation_receipts"]["required_checks_passed"] = scope_result[
        "required_checks_passed"
    ]
    artifact["validation_receipts"]["repository_health"] = scope_result["repository_health"]
    finish("validation", len(scope_result["validation_receipts"]))
    ready = not failures and scope_result["required_checks_passed"]
    if not scope_result["required_checks_passed"]:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
    artifact["cohort_ready_score"] = int(ready)
    artifact["fresh_source_score"] = int(ready and artifact["selected_prior_exposure_count"] == 0)
    artifact["acceptance_gate_results"] = gates(
        not failures and scope_result["required_checks_passed"],
        ready,
        len(artifact["rows"]) if ready else 0,
        ready and artifact["selected_prior_exposure_count"] == 0,
    )
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    candidate = raw_dir / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    terminal = [
        validation.CommandSpec(
            "cold_reduction",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/experiments/experiment_7701_v671_sealed_cohort.py",
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
        artifact["cohort_ready_score"] = 0
        artifact["fresh_source_score"] = 0
        artifact["acceptance_gate_results"] = gates(False, False, 0, False)
    finish("terminal_readers", len(receipts))
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    atomic_json(root / output, artifact)
    progress(started, "publication", "complete", len(artifact["rows"]))
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Run the owned cohort or cold-read one serialized candidate."""
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

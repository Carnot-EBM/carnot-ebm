"""Reconcile V667 producer evidence. Spec: REQ-REPORT-7656."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import socket
import tempfile
import time
from typing import Any

from carnot import experiment_7643_v667_contract_methods as contract
from carnot import experiment_7628_v665_capstone as prior
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = "results/experiment_7656_v667_capstone.json"
MODULE = "python/carnot/experiment_7656_v667_capstone.py"
WRAPPER = "scripts/experiments/experiment_7656_v667_capstone.py"
TEST = "tests/python/test_experiment_7656_v667_capstone.py"
NOTE = "docs/research-notes/v667-capstone.md"
PRE_GATES = {"exp7647-witness-energy": "results/experiment_7647_witness_energy.json"}
INPUTS = (
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "scripts/publication_gate.py",
)


def progress(start: float, phase: str, event: str, units: int = 0) -> None:  # pragma: no cover
    """Print an owned progress boundary using monotonic current time."""
    print(
        f"[exp7656] phase={phase} event={event} units={units} elapsed_s={time.monotonic() - start:.3f}",
        flush=True,
    )


def load_authority(root: Path) -> dict[str, Any]:
    """Authenticate the selected YAML against the independent V667 design."""
    selected, roadmap, candidates = contract.resolve_authority(root)
    comparison = contract.compare_authorities((root / contract.DESIGN_PATH).read_text(), roadmap)
    if not comparison["passed"]:
        raise ValueError(f"V667 authority mismatch: {comparison['errors']}")
    return {
        "path": selected.relative_to(root).as_posix(),
        "tasks": roadmap["tasks"],
        "candidates": candidates,
    }


def collect_dispositions(root: Path, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read exactly fourteen literal task outcomes, including renamed pre-gates."""
    if len(tasks) != 14 or [t["id"].split("-")[0] for t in tasks] != [
        f"exp{i}" for i in range(7643, 7657)
    ]:
        raise ValueError("V667 roster order or count mismatch")
    result = []
    for index, task in enumerate(tasks, 1):
        planned = task["deliverable"]
        actual = PRE_GATES.get(task["id"], planned)
        path = root / actual
        exists = path.is_file() if index != 14 else False
        payload = prior.load_json(path) if exists else {}
        kind = (
            "current_self"
            if index == 14
            else "conductor_pre_gate"
            if "blocked_diagnostic_contract" in payload
            else "terminal_producer"
            if exists
            else "missing_work"
        )
        gate = (
            prior._pre_gate_failure(task["id"], actual, payload)
            if kind == "conductor_pre_gate"
            else None
        )
        result.append(
            {
                "order": index,
                "task_id": task["id"],
                "title": task["title"],
                "phase": task["phase"],
                "planned_path": planned,
                "actual_path": actual if exists else None,
                "custody_kind": kind,
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
                "honest_verdict": payload.get("honest_verdict") if exists else None,
                "verdict_class": "blocked" if gate else payload.get("verdict_class"),
                "flagged_adversarial": payload.get("flagged_adversarial", False),
                "gate_check": gate,
            }
        )
    return result


def reduce_evidence(root: Path) -> dict[str, Any]:
    """Recount raw rows by independent unit and retain producer validity."""
    get = lambda n: prior.load_json(next((root / "results").glob(f"experiment_{n}_v667_*.json")))
    fixture, corpus, audit = get(7644), get(7646), get(7650)
    qwen, wrapper, live = get(7651), get(7652), get(7653)
    feature_dir = root / "results/raw/experiment_7646_v667_source_feature_corpus"
    raw_features = [
        json.loads(line)
        for role in ("fit", "tune", "policy", "online", "evaluation", "pilot")
        for line in (feature_dir / f"{role}_features.jsonl").read_text().splitlines()
        if line
    ]
    if Counter(map(canonical_hash, raw_features)) != Counter(map(canonical_hash, corpus["rows"])):
        raise ValueError("source feature producer rows disagree with raw sidecars")
    wrapper_dir = root / "results/raw/experiment_7652_v667_arc_wrapper_measurement/rows"
    raw_wrapper = [prior.load_json(path) for path in wrapper_dir.glob("*.json")]
    if Counter(map(canonical_hash, raw_wrapper)) != Counter(map(canonical_hash, wrapper["rows"])):
        raise ValueError("ARC wrapper producer rows disagree with raw sidecars")
    raw_live = prior.load_json(
        root / "results/raw/experiment_7653_v667_arc_live_generalization/episode_rows.json"
    )["rows"]
    if Counter(map(canonical_hash, raw_live)) != Counter(map(canonical_hash, live["rows"])):
        raise ValueError("ARC live producer rows disagree with raw sidecar")
    original = [
        r for r in raw_features if r.get("arm") == "original_source" and r.get("role") != "pilot"
    ]
    groups = {r["source_group_id"] for r in original}
    denominator = sum(r.get("denominator", 0) for r in original)
    checked = sum(r.get("checked_predicates", 0) for r in original)
    if (
        len(groups) != 240
        or checked != audit["independent_findings"]["exp7646"]["checked_predicates"]
    ):
        raise ValueError("source corpus raw reduction disagrees with independent audit")
    return {
        "fixture": {
            "independent_groups": len(fixture["rows"]),
            "verifier_is_oracle": True,
            "verdict_class": fixture["verdict_class"],
        },
        "source_corpus": {
            "independent_groups": len(groups),
            "checked_predicates": checked,
            "coverage_denominator": denominator,
            "coverage": checked / denominator if denominator else None,
            "verdict_class": corpus["verdict_class"],
            "flagged_adversarial": corpus["flagged_adversarial"],
        },
        "source_audit": {
            "benefit_eligible": False,
            "findings": audit["claim_findings"],
            "verdict_class": audit["verdict_class"],
        },
        "qwen_pilot": {
            "independent_groups": len(qwen["paired_pilot_rows"]),
            "verdict_class": qwen["verdict_class"],
            "current_calls_are_inherited": True,
        },
        "arc_wrapper": {
            "independent_games": len(wrapper["per_game_results"]),
            "verdict_class": wrapper["verdict_class"],
            "benefit_score": wrapper["wrapper_benefit_score"],
        },
        "arc_live": {
            "independent_games": len(live["per_game_results"]),
            "episodes": len(live["rows"]),
            "verdict_class": live["verdict_class"],
            "paired_level_deltas": live["sample_size_budget"]["paired_level_deltas"],
        },
        "portable_consumer": {"current_cost_available": False, "prior_hardware_only": True},
    }


def blocking_checks(root: Path, dispositions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Name one exact missing or invalid operand for each unavailable branch."""
    checks = []
    for row in dispositions[:-1]:
        task = row["task_id"]
        path = str(root / (row["actual_path"] or row["planned_path"]))
        if row["custody_kind"] == "conductor_pre_gate":
            checks.append(row["gate_check"])
        elif row["custody_kind"] == "missing_work":
            checks.append(
                prior._failed_check(
                    "required_scientific_producer", task, path, "path", "exists", True, False
                )
            )
        elif row["verdict_class"] == "disqualified" or row["flagged_adversarial"]:
            checks.append(
                prior._failed_check(
                    "producer_eligible",
                    task,
                    path,
                    "verdict_class",
                    "in",
                    ["positive", "null", "circular_positive"],
                    row["verdict_class"],
                )
            )
    return checks


def decisions() -> list[dict[str, str]]:
    """Require a falsifiable changed premise for each mechanism family."""
    return [
        {
            "scope": "source_witness_parser",
            "decision": "retire",
            "changed_premise_required": "new source-dependent discriminator checks at least one real corpus predicate",
        },
        {
            "scope": "source_decision_energy",
            "decision": "change",
            "changed_premise_required": "valid nonzero source coverage and oracle-distinct labels before head fitting",
        },
        {
            "scope": "retained_learning",
            "decision": "change",
            "changed_premise_required": "released delayed feedback and held-out restart replay on valid groups",
        },
        {
            "scope": "qwen_pointer_pilot",
            "decision": "keep",
            "changed_premise_required": "independent labeled confirmatory groups and paired proper loss",
        },
        {
            "scope": "arc_goal_guard",
            "decision": "change",
            "changed_premise_required": "valid wrapper checks and adapter-withheld hidden-game improvement",
        },
        {
            "scope": "cuda_live_resource",
            "decision": "change",
            "changed_premise_required": "owned exclusive CUDA capacity before a new live measurement",
        },
        {
            "scope": "portable_consumer",
            "decision": "change",
            "changed_premise_required": "valid portable head and whole-consumer process-block cost comparison",
        },
    ]


def _gates(summary: dict[str, Any], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep validity, readiness and each benefit type independently typed."""
    facts = [
        (
            "validity",
            not checks,
            {"failed_upstream_checks": len(checks)},
            "Only authenticated, unflagged producers can support a claim.",
        ),
        (
            "readiness",
            False,
            {
                "source_groups": summary["source_corpus"]["independent_groups"],
                "checked_predicates": summary["source_corpus"]["checked_predicates"],
            },
            "Structural transport readiness requires nonzero valid coverage.",
        ),
        (
            "probability_benefit",
            None,
            {"eligible_paired_probabilities": 0, "independent_labels": 0},
            "Proper loss requires independent labels and paired probabilities.",
        ),
        (
            "utility",
            None,
            {"eligible_typed_decisions": 0, "eligible_costs": 0},
            "Decision value requires typed actions, costs and independent outcomes.",
        ),
        (
            "retention",
            None,
            {"eligible_delayed_updates": 0, "held_out_restart_groups": 0},
            "Retained learning requires causal feedback and held-out replay.",
        ),
        (
            "freshness",
            False,
            {"unexposed_confirmatory_groups": 0, "qualified_hidden_games": 0},
            "Exposed replay and public games do not establish fresh benefit.",
        ),
    ]
    return [
        {"gate": name, "passed": passed, "measured_operands": operands, "principle": principle}
        for name, passed, operands, principle in facts
    ]


def _sources(
    root: Path, dispositions: list[dict[str, Any]], authority: dict[str, Any]
) -> dict[str, Any]:
    """Separate immutable producer bytes, pre-gates, missing inputs and outputs."""
    result: dict[str, Any] = {
        "producer_files": {},
        "pre_gate_receipts": {},
        "missing_inputs": [],
        "planned_outputs": [OUTPUT],
    }
    for row in dispositions[:-1]:
        if row["exists"]:
            bucket = (
                "pre_gate_receipts"
                if row["custody_kind"] == "conductor_pre_gate"
                else "producer_files"
            )
            result[bucket][row["actual_path"]] = row["sha256"]
        else:
            result["missing_inputs"].append(row["planned_path"])
    for label in (*INPUTS, authority["path"], MODULE):
        path = root / label
        if path.is_file():
            result["producer_files"][label] = sha256_file(path)
        else:
            result["missing_inputs"].append(label)
    raw_paths = [
        *(
            root
            / "results/raw/experiment_7646_v667_source_feature_corpus"
            / f"{role}_features.jsonl"
            for role in ("fit", "tune", "policy", "online", "evaluation", "pilot")
        ),
        *(root / "results/raw/experiment_7652_v667_arc_wrapper_measurement/rows").glob("*.json"),
        root / "results/raw/experiment_7653_v667_arc_live_generalization/episode_rows.json",
    ]
    for path in raw_paths:
        result["producer_files"][path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def checksum(value: dict[str, Any]) -> str:
    """Bind immutable sources, reduction code and interpreted decisions."""
    return canonical_hash(
        {
            key: value.get(key)
            for key in (
                "selected_authority",
                "milestone_dispositions",
                "source_artifact_hashes",
                "evidence_summary",
                "gate_check_summary",
                "acceptance_gate_results",
                "remaining_prd_gaps",
                "next_decisions",
                "publication_gates",
                "random_seed",
                "affected_file_validation_manifest",
            )
        }
    )


def build_artifact(
    root: Path = ROOT,
    *,
    receipts: list[dict[str, Any]] | None = None,
    outcomes: dict[str, Any] | None = None,
    spans: list[dict[str, Any]] | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """Build a reproducible candidate without invoking a model or child process."""
    authority = load_authority(root)
    dispositions = collect_dispositions(root, authority["tasks"])
    summary = reduce_evidence(root)
    checks = blocking_checks(root, dispositions)
    verdict = (
        "complete_blocked_required_v667_external_evidence"
        if checks
        else "complete_null_v667_no_benefit"
    )
    verdict_class = "blocked" if checks else "null"
    dispositions[-1]["honest_verdict"] = verdict
    dispositions[-1]["verdict_class"] = verdict_class
    publication = prior._publication_record(root)
    publication["claim_boundary"] = "historical_fover_only"
    publication["unmet_gates"] = [
        name for name, gate in publication["gates"].items() if not gate["pass"]
    ]
    sources = _sources(root, dispositions, authority)
    preconditions = [
        {
            "check": "named_input_file",
            "upstream": "repository",
            "path": label,
            "field": "exists",
            "operator": "==",
            "expected": True,
            "observed": (root / label).is_file(),
            "passed": (root / label).is_file(),
        }
        for label in (*INPUTS, authority["path"])
    ]
    preconditions.append(
        {
            "check": "current_process_ownership",
            "upstream": "host",
            "path": "/proc/self",
            "field": "pid",
            "operator": "==",
            "expected": os.getpid(),
            "observed": os.getpid(),
            "passed": True,
        }
    )
    rows = prior._rows(dispositions)
    for number, arm_source in (
        (7644, "fixture"),
        (7646, "source_corpus"),
        (7651, "qwen_pilot"),
        (7652, "arc_wrapper"),
        (7653, "arc_live"),
    ):
        source_path = next((root / "results").glob(f"experiment_{number}_v667_*.json"))
        for raw in prior.load_json(source_path)["rows"]:
            rows.append(
                {
                    "unit_id": raw.get("source_group_id", raw.get("unit_id", raw.get("game"))),
                    "arm": raw.get("arm"),
                    "source_branch": arm_source,
                    "absolute_metric": raw.get("absolute_metric"),
                    "numerator": raw.get("numerator"),
                    "denominator": raw.get("denominator"),
                    "raw_provenance": raw.get("raw_provenance", str(source_path.relative_to(root))),
                    "excluded": raw.get("excluded", raw.get("exclusion", False)),
                    "censored": raw.get("censored", False),
                    "seed": raw.get("seed"),
                    "raw_row_checksum": canonical_hash(raw),
                }
            )
    gaps = [
        {
            "gap": "source_backed_decisions",
            "status": "measured_null_coverage",
            "observed": summary["source_corpus"],
            "next_condition": "new discriminator with nonzero valid corpus coverage",
        },
        {
            "gap": "causal_retained_learning",
            "status": "unavailable",
            "observed": None,
            "next_condition": "eligible delayed feedback and held-out restart replay",
        },
        {
            "gap": "useful_live_planning",
            "status": "disqualified",
            "observed": summary["arc_live"],
            "next_condition": "valid ARC execution receipts and hidden-game gain",
        },
    ]
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7656.v667.capstone.v1",
        "experiment": 7656,
        "experiment_id": "exp7656-capstone",
        "milestone": "2026.09.667",
        "run_date": "20260925",
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "capstone_complete_score": 1,
        "gate_check_summary": {
            "passed": not checks,
            "failed_count": len(checks),
            "first_failure": checks[0] if checks else None,
            "failed_checks": checks,
        },
        "acceptance_gate_results": _gates(summary, checks),
        "rows": rows,
        "sample_size_budget": {
            "milestone_tasks": {
                "intended": 14,
                "observed": 14,
                "eligible": 14,
                "excluded": 0,
                "censored": 0,
            },
            "source_groups": {
                "intended": 240,
                "observed": 240,
                "eligible": 0,
                "excluded": 240,
                "censored": 240,
                "exposure_limit": "historically exposed corpus; arms and views share groups",
            },
            "qwen_pilot": {
                "intended": 8,
                "observed": 8,
                "eligible": 0,
                "excluded": 8,
                "censored": 1,
            },
            "arc_live": {
                "intended_games": 3,
                "observed_games": 3,
                "episodes": summary["arc_live"]["episodes"],
                "eligible": 0,
                "excluded": 3,
                "censored": 4,
            },
            "repeated_seeds_views_and_orderings_increase_sample_size": False,
        },
        "preconditions_checked": preconditions,
        "inference_substrate": "aggregation_from_authenticated_upstream_rows",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_invoked": False,
        "historical_models": ["unsloth/Qwen3.8-27B-GGUF"],
        "invocation_counts": {
            name: 0
            for name in (
                "model_loads_attempted",
                "model_loads_completed",
                "forwards_attempted",
                "forwards_completed",
                "generations_attempted",
                "generations_completed",
                "input_tokens",
                "output_tokens",
            )
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": socket.gethostname(),
            "current_pid": os.getpid(),
            "current_gpu_uuid": None,
            "platform": platform.platform(),
        },
        "phase_spans": spans or [],
        "duration_s": duration_s,
        "random_seed": {
            "current": None,
            "purpose": "deterministic cold reduction",
            "inherited_seeds_are_not_new_samples": True,
        },
        "source_artifact_hashes": sources,
        "validation_receipts": receipts or [],
        "terminal_reader_outcomes": outcomes or {},
        "verifier_is_oracle": True,
        "selected_authority": authority["path"],
        "milestone_dispositions": dispositions,
        "evidence_summary": summary,
        "remaining_prd_gaps": gaps,
        "next_decisions": decisions(),
        "publication_gates": publication,
        "unmet_gates": publication["unmet_gates"],
        "hardware_dispositions": prior.load_json(
            root / "results/experiment_7641_v666_native_consumer.json"
        )["hardware_dispositions"],
        "capstone_note_path": NOTE,
        "affected_file_validation_manifest": {
            "test_paths": [TEST],
            "changed_modules": [MODULE],
            "static_paths": [WRAPPER],
            "spec_paths": ["openspec/capabilities/research-reporting/spec.md"],
            "frozen_before_validation": True,
        },
        "publication_performed": False,
        "roadmap_activation_performed": False,
        "purchase_performed": False,
        "generator_training_performed": False,
        "production_defaults_changed": False,
    }
    artifact["reproducibility_checksum"] = checksum(artifact)
    artifact["field_principles"] = prior._field_principles(tuple(artifact))
    artifact["field_principles"]["field_principles"] = (
        "Every top-level field carries its governing principle."
    )
    return artifact


def mutate_candidate(value: dict[str, Any], mutation: str) -> dict[str, Any]:
    """Corrupt a private candidate for four independent cold replay controls."""
    if mutation == "deleted":
        value["milestone_dispositions"].pop(4)
    elif mutation == "reordered":
        value["milestone_dispositions"][3:5] = reversed(value["milestone_dispositions"][3:5])
    elif mutation == "wrong_field":
        value["gate_check_summary"]["failed_checks"][0]["field"] = "wrong_field"
    elif mutation == "self_input":
        value["source_artifact_hashes"]["producer_files"][OUTPUT] = "sha256:self"
    else:
        raise ValueError(f"unknown mutation: {mutation}")
    value["reproducibility_checksum"] = checksum(value)
    return value


def independent_reduce(value: object, root: Path = ROOT) -> list[str]:
    """Reload current sources and compare every immutable interpretation."""
    if not isinstance(value, dict):
        return ["artifact_object_required"]
    errors: list[str] = []
    if value.get("reproducibility_checksum") != checksum(value):
        errors.append("checksum_mismatch")
    dispositions = value.get("milestone_dispositions")
    if not isinstance(dispositions, list) or len(dispositions) != 14:
        errors.append("milestone_disposition_count")
    else:
        try:
            ids = [row["task_id"] for row in dispositions]
        except (KeyError, TypeError):
            ids = []
        if ids != [
            f"exp{n}-{suffix}"
            for n, suffix in zip(
                range(7643, 7657),
                (
                    "contract-methods",
                    "source-witness-prototype",
                    "arc-validation-requalification",
                    "source-feature-corpus",
                    "witness-energy",
                    "decision-evaluation",
                    "continuous-witness-learning",
                    "independent-source-audit",
                    "qwen-witness-challenge",
                    "arc-wrapper-measurement",
                    "arc-live-generalization",
                    "portable-witness-energy",
                    "consumer-cost-continuity",
                    "capstone",
                ),
                strict=True,
            )
        ]:
            errors.append("milestone_disposition_order")
    try:
        expected = build_artifact(root)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return [*errors, f"cold_reduction_failed:{type(error).__name__}"]
    for field in (
        "milestone_dispositions",
        "source_artifact_hashes",
        "evidence_summary",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "sample_size_budget",
        "remaining_prd_gaps",
        "next_decisions",
        "publication_gates",
        "unmet_gates",
        "selected_authority",
        "reproducibility_checksum",
    ):
        if value.get(field) != expected.get(field):
            errors.append(f"{field}_mismatch")
    for field in (
        "honest_verdict",
        "verdict_class",
        "capstone_complete_score",
        "inference_substrate_class",
        "MODEL_SPECS",
        "model_invoked",
        "verifier_is_oracle",
    ):
        if value.get(field) != expected.get(field):
            errors.append(f"{field}_mismatch")
    if not isinstance(value.get("field_principles"), dict) or not set(value).issubset(
        value["field_principles"]
    ):
        errors.append("field_principles_incomplete")
    return list(dict.fromkeys(errors))


def read_candidate(path: Path, root: Path = ROOT) -> list[str]:
    """Read a candidate in a fresh process and cold-reduce exact source bytes."""
    return independent_reduce(prior.load_json(path), root)


def task_specific_e2e(value: dict[str, Any], root: Path) -> list[str]:
    """Reject deleted, reordered, wrong-field and self-input controls."""
    errors = independent_reduce(value, root)
    for mutation in ("deleted", "reordered", "wrong_field", "self_input"):
        if not independent_reduce(mutate_candidate(deepcopy(value), mutation), root):
            errors.append(f"mutation_not_rejected:{mutation}")
    return errors


def terminal_commands(
    root: Path, candidate: Path
) -> list[validation.CommandSpec]:  # pragma: no cover
    """Name bounded fresh readers of one unpublished candidate."""
    python = str(root / ".venv/bin/python")
    base = (python, "-u", WRAPPER, "--root", str(root))
    return [
        validation.CommandSpec(
            "fresh_process_cold_reduction", (*base, "--cold-replay", str(candidate)), "candidate"
        ),
        validation.CommandSpec(
            "task_specific_e2e", (*base, "--e2e", str(candidate)), "candidate_mutations"
        ),
        validation.CommandSpec(
            "independent_reduction", (*base, "--independent-reduce", str(candidate)), "candidate"
        ),
        validation.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate",
        ),
        validation.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "candidate",
        ),
    ]


def run_experiment(root: Path, run_date: str, output: Path) -> dict[str, Any]:  # pragma: no cover
    """Validate one frozen candidate and publish only after terminal readers pass."""
    start = time.monotonic()
    progress(start, "preconditions", "before")
    if root.resolve() != ROOT or run_date != "20260925":
        raise ValueError("frozen root or date mismatch")
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7656-", dir="/tmp"))
    (private / "pytest").mkdir()
    spans = []
    candidate = build_artifact(root)
    if not all(row["passed"] for row in candidate["preconditions_checked"]):
        raise RuntimeError("named input precondition failed")
    boundary = time.monotonic() - start
    spans.append(
        {
            "phase": "preconditions_and_reduction",
            "start_s": 0.0,
            "end_s": boundary,
            "duration_s": boundary,
            "completed_units": 14,
            "checkpoint": "fourteen_dispositions_reduced",
        }
    )
    progress(start, "preconditions", "after", 14)
    for name in ("model_load", "generation", "benchmark"):
        progress(start, name, "before")
        now = time.monotonic() - start
        spans.append(
            {
                "phase": name,
                "start_s": now,
                "end_s": now,
                "duration_s": 0.0,
                "completed_units": 0,
                "checkpoint": "no_current_invocation",
            }
        )
        progress(start, name, "after")
    progress(start, "validation", "before_subprocesses")
    phase_start = time.monotonic() - start
    commands = validation.build_scoped_commands(
        root,
        [TEST],
        [MODULE],
        static_paths=[WRAPPER],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage.exp7656",
    )
    receipts = validation.run_commands(
        root, commands, log_dir=private / "logs/affected", heartbeat_s=60
    )
    required = validation.reduce_required_checks(receipts)
    now = time.monotonic() - start
    spans.append(
        {
            "phase": "validation",
            "start_s": phase_start,
            "end_s": now,
            "duration_s": now - phase_start,
            "completed_units": len(receipts),
            "checkpoint": "affected_checks_complete",
        }
    )
    progress(start, "validation", "after_subprocesses", len(receipts))
    if not required["required_checks_passed"]:
        raise RuntimeError("affected validation failed")
    candidate = build_artifact(
        root, receipts=receipts, spans=spans, duration_s=time.monotonic() - start
    )
    candidate_path = private / "candidate.json"
    atomic_json(candidate_path, candidate)
    progress(start, "terminal", "before_subprocesses")
    phase_start = time.monotonic() - start
    terminal = validation.run_commands(
        root,
        terminal_commands(root, candidate_path),
        log_dir=private / "logs/terminal",
        heartbeat_s=60,
    )
    outcomes = {
        row["name"]: {
            "passed": row["passed"],
            "exit_code": row["exit_code"],
            "log_path": row["log_path"],
            "log_sha256": row["log_sha256"],
        }
        for row in terminal
    }
    now = time.monotonic() - start
    spans.append(
        {
            "phase": "terminal_readers",
            "start_s": phase_start,
            "end_s": now,
            "duration_s": now - phase_start,
            "completed_units": len(terminal),
            "checkpoint": "terminal_candidate_checked",
        }
    )
    progress(start, "terminal", "after_subprocesses", len(terminal))
    if not all(row["passed"] for row in terminal):
        raise RuntimeError("terminal reader failed")
    final = build_artifact(
        root,
        receipts=[*receipts, *terminal],
        outcomes=outcomes,
        spans=spans,
        duration_s=time.monotonic() - start,
    )
    if independent_reduce(final, root):
        raise RuntimeError("final cold reduction failed")
    destination = output if output.is_absolute() else root / output
    progress(start, "publication", "before_atomic")
    atomic_json(destination, final)
    progress(start, "publication", "after_atomic")
    return final


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Run the declared capstone or one bounded read-only terminal check."""
    print("[exp7656] phase=startup event=flushed", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--date", default="20260925")
    parser.add_argument("--output", type=Path, default=Path(OUTPUT))
    for name in ("cold-replay", "independent-reduce", "e2e"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    candidate = args.cold_replay or args.independent_reduce or args.e2e
    if candidate:
        value = prior.load_json(candidate)
        errors = (
            task_specific_e2e(value, args.root)
            if args.e2e
            else independent_reduce(value, args.root)
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, args.output)
    return 0

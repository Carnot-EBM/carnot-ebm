#!/usr/bin/env python3
"""Qualify complete-byte evidence views under REQ-REPORT-7756."""

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

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.verify import evidence_views as views
from carnot.verify import source_set_energy as energy

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7756_v675_evidence_view_protocol"
RAW = ROOT / "results/raw" / NAME
OUTPUT = ROOT / "results" / f"{NAME}.json"
TEST = f"tests/python/test_{NAME}.py"
MODULE = "python/carnot/verify/evidence_views.py"
WRAPPER = f"scripts/experiments/{NAME}.py"
REQUIRED = (
    (
        "exp7714-alignment",
        "results/experiment_7714_v672_alignment_protocol.json",
        "alignment_protocol_ready_score",
        1,
    ),
    (
        "exp7728-set-energy",
        "results/experiment_7728_v673_set_energy_protocol.json",
        "set_protocol_ready_score",
        1,
    ),
    (
        "exp7753-contract",
        "results/experiment_7753_v675_contract_methods.json",
        "contract_ready_score",
        1,
    ),
)
HISTORICAL = (
    (
        "exp7754-sentence",
        "results/experiment_7754_v675_sentence_protocol.json",
        "sentence_protocol_ready_score",
        1,
    ),
    (
        "exp7755-training",
        "results/experiment_7755_v675_training_runtime.json",
        "training_runtime_ready_score",
        1,
    ),
    (
        "exp7741-declared-producer",
        "results/experiment_7741_v674_training_qualification.json",
        "training_runtime_ready_score",
        1,
    ),
    (
        "exp7741-conductor-pre-gate",
        "results/experiment_7741_training_qualification.json",
        "honest_verdict",
        "complete_qualified_training_runtime",
    ),
)
PRINCIPLES = {
    "experiment_id": "An artifact must have a unique current owner.",
    "honest_verdict": "A terminal record must not waste attempts on unchanged inputs.",
    "verdict_class": "The claim class travels with the evidence.",
    "flagged_adversarial": "Invalid evidence must not open downstream gates.",
    "gate_check_summary": "Missing producers and failed scientific thresholds are different causes.",
    "rows": "Aggregates must be recomputable without rerunning science.",
    "acceptance_gate_results": "A working protocol is not evidence of benefit.",
    "duration_s": "Duration must describe actual work without padding.",
    "random_seed": "A third party needs the same experiment inputs.",
    "sample_size_budget": "Repeated views and seeds do not increase independent family count.",
    "source_artifact_hashes": "A missing producer cannot be replaced with a convenient old result.",
    "preconditions_checked": "Access and validity must be established before expensive work.",
    "validation_receipts": "All registered checks must pass before readiness opens.",
    "verifier_is_oracle": "Execution truth and independent semantic verification are distinct claims.",
    "inference_substrate_class": "Duration floors must match the invoked substrate.",
    "MODEL_SPECS": "A cited upstream model is not a current model invocation.",
    "evidence_view_ready_score": "A proposed regularizer needs a real, truth-preserving intervention.",
    "evidence_view_protocol_path": "Identical input information makes the comparison interpretable.",
    "view_witness_rows": "A name change alone is not a changed technique.",
}


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Flush each phase and model boundary so a long run stays observable."""
    print(
        f"[exp7756] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def _span(phase: str, began: float, units: int) -> dict[str, Any]:
    return {"phase": phase, "duration_s": time.monotonic() - began, "completed_units": units}


def check_sources() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep required producer checks apart from historical input observations."""
    sources, failures, observations = [], [], []
    for upstream, relative, field, expected in (*REQUIRED, *HISTORICAL):
        path = ROOT / relative
        value = json.loads(path.read_text()) if path.is_file() else {}
        observed = value.get(field)
        item = {
            "upstream_id": upstream,
            "artifact_path": relative,
            "artifact_hash": sha256_file(path) if path.is_file() else None,
            "run_date": value.get("run_date"),
            "imported_fields": {field: observed},
            "eligibility": bool(
                path.is_file()
                and observed == expected
                and value.get("flagged_adversarial") is False
            ),
        }
        sources.append(item)
        if not item["eligibility"]:
            check = {
                "upstream_id": upstream,
                "artifact_path": relative,
                "artifact_hash": item["artifact_hash"],
                "field": field if path.is_file() else "exists",
                "expected": expected if path.is_file() else True,
                "observed": observed if path.is_file() else False,
                "operator": "==",
            }
            (
                failures if (upstream, relative, field, expected) in REQUIRED else observations
            ).append(check)
    return sources, failures, observations


def fixtures() -> list[dict[str, Any]]:
    """Exercise changed windows, duplicates, null evidence, and both limits."""
    return [
        {
            "family": "changed_unicode",
            "source": "Élan! 東京? Third value is 12. final fragment".encode(),
            "answer": "東京? Answer continues".encode(),
            "label": 0,
        },
        {"family": "duplicate", "source": b"A. A.", "answer": b"A.", "label": 0},
        {"family": "empty_source", "source": b"", "answer": b"Maybe 7.", "label": 1},
        {"family": "shared_over_budget", "source": b"A. " * 65, "answer": b"A.", "label": 0},
        {"family": "invalid_utf8", "source": b"\xff", "answer": b"A.", "label": 0},
    ]


def _logistic_parameters() -> dict[str, Any]:
    """Use a fixed control head so the protocol test opens no natural labels."""
    weights = [[0.0, 0.0] for _ in range(132)]
    weights[128][1] = 3.0
    return {"w": weights, "b": [0.0, 0.0]}


def build_raw(start: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the real prepare-to-distribution path on fixed byte fixtures."""
    manifest = {
        "view_a": "all original sentences plus adjacent triples",
        "view_b": "all original sentences plus adjacent pairs",
        "feature_dim": 132,
        "feature_source": "source_alignment.pair_features",
        "normalization": "finite location-label softmax with equal duplicate-group and null mass",
        "max_windows": 128,
        "max_answer_units": 16,
        "shared_abstention": True,
        "empty_source": "explicit null location",
        "arms": views.ARMS,
        "temperature": "one response-level temperature after raw A/B arithmetic mean",
        "tune_selection": "uncalibrated deployed response NLL",
        "local_loss": "mean known-sentence NLL once per response; response_set has no local term",
        "augmentation": "mean of A/B supervised losses",
        "constraints": {
            "symmetric_bernoulli_kl_half_max": 0.01,
            "view_b_label_ce_max": 0.70,
            "dual_step": 0.01,
            "dual_range": [0, 10],
        },
        "logistic": "average per-sentence pooled features before shared linear head",
        "policy": {"accept": "5p", "reject": "1-p", "escalate": 0.25, "ties": "escalate"},
        "online_learning": "same frozen inference and cost rule as offline deployment",
        "predictor_exclusions": [
            "labels",
            "annotation_offsets",
            "source_ids",
            "roles",
            "generator_ids",
        ],
        "parameter_max": 4096,
    }
    atomic_json(RAW / "protocol.json", manifest)
    rows, witnesses = [], []
    params = energy.zero_parameters()
    params["weights"][0][128] = 3.0
    params["weights"][1][128] = -3.0
    for index, fixture in enumerate(fixtures()):
        pair = views.prepare_views(fixture["source"], fixture["answer"])
        serialized = views.serialize_pair(pair)
        view_path = RAW / f"view_{index}.json"
        atomic_json(view_path, serialized)
        eligible = pair["a"]["abstention"] is None
        risks = {}
        normalization = {}
        for key in ("a", "b"):
            if eligible:
                distribution = energy.distribution(pair[key], params)
                risks[key] = distribution["response_unsupported"]
                normalization[key] = max(
                    abs(sum(map(sum, joint)) - 1) for joint in distribution["joint"]
                )
            else:
                risks[key], normalization[key] = None, None
        difference = eligible and pair["a"]["pair_features"] != pair["b"]["pair_features"]
        witness = {
            "family": fixture["family"],
            "raw_path": str(view_path.relative_to(ROOT)),
            "source_covered": all(
                view["source_bytes"] == fixture["source"]
                and (
                    view["abstention"] == "invalid_utf8"
                    or b"".join(view["source_sentences"]) == fixture["source"]
                )
                for view in pair.values()
            ),
            "answer_covered": all(
                view["answer_bytes"] == fixture["answer"]
                and (
                    view["abstention"] == "invalid_utf8"
                    or b"".join(view["answer_units"]) == fixture["answer"]
                )
                for view in pair.values()
            ),
            "different_features": bool(difference),
            "normalization_error": normalization,
            "duplicate_prior": views.location_prior(pair["a"])
            if fixture["family"] == "duplicate"
            else None,
            "abstention": pair["a"]["abstention"],
            "invariance_error": 0.0
            if views.prepare_views(fixture["source"], fixture["answer"]) == pair
            else 1.0,
        }
        witnesses.append(witness)
        for arm in views.ARMS:
            arm_risks = dict(risks)
            if arm == "local_logistic" and eligible:
                arm_risks = {
                    "a": views.logistic_averaged_feature_risk(_logistic_parameters(), pair),
                    "b": None,
                }
            outcome = views.decision(arm, arm_risks["a"], arm_risks["b"], 1.0, fixture["label"])
            rows.append(
                {
                    "family_id": fixture["family"],
                    "arm": arm,
                    "seed": 67501,
                    "raw_path": str(view_path.relative_to(ROOT)),
                    "source_sha256": views.digest(fixture["source"]),
                    "answer_sha256": views.digest(fixture["answer"]),
                    "label": fixture["label"],
                    "label_scope": "synthetic_fixture",
                    "risk_a": arm_risks["a"],
                    "risk_b": arm_risks["b"],
                    **outcome,
                    "excluded": False,
                    "censored": not eligible,
                    "exclusions": [pair["a"]["abstention"]] if not eligible else [],
                    "denominator": 1,
                }
            )
        progress(start, "fixtures", "completed_family", index + 1)
    atomic_json(RAW / "rows.json", rows)
    atomic_json(RAW / "witnesses.json", witnesses)
    return rows, witnesses, manifest


def reduce_raw(raw: Path, candidate: Path | None = None) -> dict[str, Any]:
    """Rebuild serialized views and independently recompute every raw score."""
    rows = json.loads((raw / "rows.json").read_text())
    witnesses = json.loads((raw / "witnesses.json").read_text())
    params = energy.zero_parameters()
    params["weights"][0][128] = 3.0
    params["weights"][1][128] = -3.0
    for row in rows:
        encoded = json.loads((ROOT / row["raw_path"]).read_text())
        pair = views.deserialize_pair(encoded)
        original = views.prepare_views(pair["a"]["source_bytes"], pair["a"]["answer_bytes"])
        if pair != original or pair["a"]["source_sha256"] != row["source_sha256"]:
            raise ValueError("serialized view or input bytes changed")
        if row["eligible"]:
            risks = {
                key: energy.distribution(pair[key], params)["response_unsupported"]
                for key in ("a", "b")
            }
        else:
            risks = {"a": None, "b": None}
        if row["arm"] == "local_logistic" and row["eligible"]:
            risks = {
                "a": views.logistic_averaged_feature_risk(_logistic_parameters(), pair),
                "b": None,
            }
        expected = views.decision(row["arm"], risks["a"], risks["b"], 1.0, row["label"])
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError("raw decision changed")
    if candidate is not None:
        value = json.loads(candidate.read_text())
        if value["rows"] != rows or value["view_witness_rows"] != witnesses:
            raise ValueError("candidate metrics changed")
    return {
        "rows": rows,
        "witnesses": witnesses,
        "eligible": sum(row["eligible"] for row in rows),
        "family_count": len(witnesses),
    }


def _commands() -> list[CommandSpec]:
    """Use the frozen affected scope and one full-suite gate."""
    scope = json.loads((RAW / "validation_scope.json").read_text())
    scoped = build_scoped_commands(
        ROOT,
        scope["test_paths"],
        scope["changed_modules"],
        static_paths=scope["static_paths"],
        basetemp=RAW / "tmp",
        coverage_file=RAW / ".coverage",
    )
    full = CommandSpec(
        "full_python_suite",
        (
            str(ROOT / ".venv/bin/pytest"),
            "tests/python",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={RAW / 'tmp/full'}",
        ),
        "all_python_tests",
        900,
    )
    return [full, *scoped]


def _failure(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "upstream_id": NAME,
        "artifact_path": receipt["log_path"],
        "artifact_hash": receipt["log_sha256"],
        "field": f"{receipt['name']}.exit_code",
        "expected": 0,
        "observed": receipt["exit_code"],
        "operator": "==",
    }


def _artifact(
    start: float,
    date: str,
    sources: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    reduced: dict[str, Any] | None,
    receipts: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep fixture validity distinct from validation readiness and benefit."""
    rows = reduced["rows"] if reduced else []
    witnesses = reduced["witnesses"] if reduced else []
    validity = bool(
        witnesses
        and all(
            row["source_covered"] and row["answer_covered"] and row["invariance_error"] == 0
            for row in witnesses
        )
        and any(row["different_features"] for row in witnesses)
        and all(
            all(value is None or value <= 1e-12 for value in row["normalization_error"].values())
            for row in witnesses
        )
    )
    ready = validity and not failures and all(receipt["passed"] for receipt in receipts)
    verdict_class = "blocked" if failures else "circular_positive" if ready else "disqualified"
    bound = {
        "sources": sources,
        "protocol_hash": sha256_file(RAW / "protocol.json")
        if (RAW / "protocol.json").is_file()
        else None,
        "code_hash": sha256_file(ROOT / MODULE),
        "seed": 67501,
        "roles": "private_fixture_only",
        "arms": views.ARMS,
    }

    return {
        "experiment_id": "exp7756-evidence-view-protocol",
        "milestone": "2026.09.675",
        "run_date": date,
        "honest_verdict": f"complete_{verdict_class}_evidence_view_protocol",
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "view_witness_rows": witnesses,
        "acceptance_gate_results": {
            "validity": validity,
            "readiness": int(ready),
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - start,
        "phase_spans": spans,
        "random_seed": [67501],
        "reproducibility_checksum": "sha256:"
        + hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest(),
        "sample_size_budget": {
            "intended": 5,
            "eligible": sum(row["abstention"] is None for row in witnesses),
            "started": len(witnesses),
            "completed": len(witnesses),
            "excluded": 0,
            "censored": sum(row["abstention"] is not None for row in witnesses),
            "effective_independent_n": len(witnesses),
        },
        "source_artifact_hashes": sources,
        "preconditions_checked": {
            "path": str(ROOT),
            "custody": "private_synthetic_fixtures",
            "backend": "jax_cpu",
            "cpu_count": os.cpu_count(),
            "host": platform.node(),
            "frozen_scope_path": str((RAW / "validation_scope.json").relative_to(ROOT)),
            "frozen_scope_hash": sha256_file(RAW / "validation_scope.json"),
            "historical_input_observations": observations,
        },
        "validation_receipts": {
            "frozen_affected_scope": json.loads((RAW / "validation_scope.json").read_text()),
            "commands": receipts,
            "cold_replay": None,
            "terminal_readers": None,
        },
        "verifier_is_oracle": True,
        "claim_scope": "fixture_only; natural_probability_and_decision_benefit_unmeasured",
        "field_principles": PRINCIPLES,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            "loads": 0,
            "generation_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "loaded_files": [],
        },
        "evidence_view_ready_score": int(ready),
        "evidence_view_protocol_path": {
            "path": str((RAW / "protocol.json").relative_to(ROOT)),
            "sha256": sha256_file(RAW / "protocol.json")
            if (RAW / "protocol.json").is_file()
            else None,
        },
        "overall_metrics": {
            "brier": sum(row["brier"] for row in rows if row["arm"] == "constrained_set")
            / len(witnesses)
            if witnesses
            else None,
            "realized_cost": sum(
                row["realized_cost"] for row in rows if row["arm"] == "constrained_set"
            )
            / len(witnesses)
            if witnesses
            else None,
        },
        "eligible_only_metrics": {
            "brier": sum(
                row["brier"] for row in rows if row["arm"] == "constrained_set" and row["eligible"]
            )
            / sum(row["eligible"] for row in rows if row["arm"] == "constrained_set")
            if witnesses
            else None,
            "realized_cost": sum(
                row["realized_cost"]
                for row in rows
                if row["arm"] == "constrained_set" and row["eligible"]
            )
            / sum(row["eligible"] for row in rows if row["arm"] == "constrained_set")
            if witnesses
            else None,
        },
    }


def run(date: str) -> dict[str, Any]:
    """Run owned fixture work, bounded validation, then publish exact bytes."""
    start = time.monotonic()
    RAW.mkdir(parents=True, exist_ok=True)
    for name in ("focused", "coverage", "full"):
        (RAW / "tmp" / name).parent.mkdir(parents=True, exist_ok=True)
    spans: list[dict[str, Any]] = []
    progress(start, "preconditions", "start")
    began = time.monotonic()
    sources, failures, observations = check_sources()
    scope = json.loads((RAW / "validation_scope.json").read_text())
    if scope["changed_modules"] != [MODULE] or scope["test_paths"] != [TEST]:
        failures.append(
            {
                "upstream_id": NAME,
                "artifact_path": str((RAW / "validation_scope.json").relative_to(ROOT)),
                "artifact_hash": sha256_file(RAW / "validation_scope.json"),
                "field": "affected_scope",
                "expected": [MODULE, TEST],
                "observed": [scope["changed_modules"], scope["test_paths"]],
                "operator": "==",
            }
        )
    spans.append(_span("preconditions", began, len(sources)))
    progress(start, "preconditions", "end", len(sources))
    reduced = None
    if not failures:
        progress(start, "fixtures", "start")
        began = time.monotonic()
        build_raw(start)
        reduced = reduce_raw(RAW)
        spans.append(_span("fixtures", began, reduced["family_count"]))
        progress(start, "fixtures", "end", reduced["family_count"])
    else:
        progress(start, "fixtures", "blocked")
    receipts: list[dict[str, Any]] = []
    if not failures:
        progress(start, "validation", "start")
        began = time.monotonic()
        receipts = run_commands(
            ROOT,
            _commands(),
            log_dir=RAW / "validation_logs",
            extra_env={"COVERAGE_FILE": str(RAW / ".coverage"), "JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        spans.append(_span("validation", began, len(receipts)))
        progress(start, "validation", "end", len(receipts))
    check_failures = [
        *failures,
        *(_failure(receipt) for receipt in receipts if not receipt["passed"]),
    ]
    artifact = _artifact(
        start, date, sources, observations, reduced, receipts, check_failures, spans
    )
    if check_failures and not failures:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["evidence_view_ready_score"] = 0
        artifact["acceptance_gate_results"]["readiness"] = 0
    candidate = RAW / "candidate.json"
    atomic_json(candidate, artifact)
    if reduced is not None:
        progress(start, "cold_replay", "before_subprocess")
        began = time.monotonic()
        cold = run_commands(
            ROOT,
            [
                CommandSpec(
                    "cold_replay",
                    (
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        WRAPPER,
                        "--cold-replay",
                        str(RAW),
                        "--candidate",
                        str(candidate),
                    ),
                    "serialized_views",
                    300,
                )
            ],
            log_dir=RAW / "cold_logs",
            heartbeat_s=30,
        )
        spans.append(_span("cold_replay", began, 1))
        artifact["validation_receipts"]["cold_replay"] = cold[0]
        progress(start, "cold_replay", "after_subprocess", 1)
        progress(start, "terminal", "start")
        began = time.monotonic()
        terminal = run_commands(
            ROOT,
            [
                CommandSpec(
                    "adversarial_verify",
                    (
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        "scripts/adversarial_verify.py",
                        "--json",
                        str(candidate),
                    ),
                    "exact_candidate",
                    300,
                ),
                CommandSpec(
                    "verdict_row_consistency_strict",
                    (
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ),
                    "exact_candidate",
                    300,
                ),
            ],
            log_dir=RAW / "terminal_logs",
            heartbeat_s=30,
        )
        spans.append(_span("terminal", began, len(terminal)))
        artifact["validation_receipts"]["terminal_readers"] = terminal
        artifact["flagged_adversarial"] = not terminal[0]["passed"]
        for receipt in [*cold, *terminal]:
            if not receipt["passed"]:
                artifact["gate_check_summary"].append(_failure(receipt))
        if artifact["gate_check_summary"]:
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = "complete_disqualified_required_validation"
            artifact["evidence_view_ready_score"] = 0
            artifact["acceptance_gate_results"]["readiness"] = 0
        progress(start, "terminal", "end", len(terminal))
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - start
    atomic_json(candidate, artifact)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_bytes(candidate.read_bytes())
    os.replace(temporary, OUTPUT)
    progress(start, "publish", "end", 1)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        reduce_raw(args.cold_replay, args.candidate)
        print("cold replay passed", flush=True)
        return 0
    result = run(args.date)
    return 0 if result["verdict_class"] in {"circular_positive", "blocked"} else 1


if __name__ == "__main__":
    sys.exit(main())

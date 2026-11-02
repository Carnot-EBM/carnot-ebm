"""Qualify the finite evidence-set protocol on exact fixtures. REQ-REPORT-7728."""

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
from carnot.verify import source_alignment as old
from carnot.verify import source_set_energy as energy

EXPERIMENT_ID = "experiment_7728_v673_set_energy_protocol"
RAW_PATH = Path("results/raw") / EXPERIMENT_ID
RESULT_PATH = Path("results") / f"{EXPERIMENT_ID}.json"
TEST_PATH = f"tests/python/test_{EXPERIMENT_ID}.py"
CHANGED_MODULES = (
    "python/carnot/verify/source_set_energy.py",
    f"python/carnot/{EXPERIMENT_ID}.py",
)
WRAPPER_PATH = f"scripts/experiments/{EXPERIMENT_ID}.py"
INPUTS = (
    ("results/experiment_7714_v672_alignment_protocol.json", "alignment_protocol_ready_score", 1),
    ("results/experiment_7727_v673_development_corpus.json", "development_cohort_ready_score", 1),
)


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Give the conductor an honest flushed boundary and completed count."""
    print(
        f"[exp7728] {phase} {event} elapsed_s={time.monotonic() - start:.3f} completed={units}",
        flush=True,
    )


def digest(data: bytes) -> str:
    """Hash exact bytes in the same syntax as existing receipts."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def check_preconditions(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Measure resource and producer schemas without treating our output as input."""
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        "eligible_producers": [],
        "flagged_historical_inputs": [],
        "pre_gate_receipts": [],
        "absent_sources": [],
    }
    for relative, field, expected in INPUTS:
        path = root / relative
        observed: Any = False
        valid = False
        sha = None
        if path.is_file():
            sha = sha256_file(path)
            try:
                value = json.loads(path.read_text())
                valid = isinstance(value, dict) and isinstance(value.get("honest_verdict"), str)
                observed = value.get(field) if valid else "invalid_schema"
                valid = valid and value.get("flagged_adversarial") is False
            except (ValueError, OSError):
                observed = "invalid_schema"
        passed = valid and observed == expected
        checks.append(
            {
                "path": relative,
                "exists": path.is_file(),
                "schema_valid": valid,
                "sha256": sha,
                "field": field,
                "expected": expected,
                "observed": observed,
                "passed": passed,
            }
        )
        if passed:
            hashes["eligible_producers"].append({"path": relative, "sha256": sha})
        else:
            failures.append(
                {
                    "check": "input_eligibility",
                    "upstream_id": Path(relative).stem,
                    "artifact_path": relative,
                    "field": field if path.is_file() else "exists",
                    "operator": "==",
                    "expected": expected if path.is_file() else True,
                    "observed": observed,
                }
            )
            hashes["absent_sources"].append(relative)
    checks.append(
        {
            "path": str(root),
            "resource": "repo_root",
            "exists": root.is_dir(),
            "effective_coding_backend": os.environ.get("CODEX_MODEL", "gpt-6 (session)"),
        }
    )
    return checks, failures, hashes


def frozen_scope(root: Path) -> dict[str, Any]:
    """Name every affected check before producing fixture results."""
    commands = build_scoped_commands(
        root,
        [TEST_PATH],
        CHANGED_MODULES,
        static_paths=[WRAPPER_PATH],
        basetemp=root / RAW_PATH / "tmp",
        coverage_file=root / RAW_PATH / ".coverage",
    )
    return {
        "test_paths": [TEST_PATH],
        "changed_modules": list(CHANGED_MODULES),
        "static_paths": [WRAPPER_PATH],
        "required_names": [c.name for c in commands],
    }


def _parameters() -> dict[str, Any]:
    params = energy.zero_parameters()
    params["weights"][0][128] = 3.0
    params["weights"][1][128] = -3.0
    return params


def build_protocol(raw_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Freeze the no-label method, then evaluate 64 fit and 32 held fixtures."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "feature_source": "source_alignment.pair_features",
        "feature_dim": old.FEATURE_DIM,
        "hash_seed_hex": old.HASH_SEED.hex(),
        "max_windows": old.MAX_WINDOWS,
        "max_answer_units": old.MAX_ANSWER_UNITS,
        "max_parameters": energy.MAX_PARAMETERS,
        "parameter_counts": {
            "set_energy": 266,
            "shared_energy": 266,
            "pooled_logistic": 266,
            "pooled_mlp": energy.MLP_PARAMETERS,
        },
        "fit_tune_budget": {
            "seeds": [67301, 67302, 67303, 67304, 67305],
            "max_epochs": 200,
            "same_four_configurations_each_arm": True,
        },
        "aggregation_assumption": "conditional_independence",
        "null_state": True,
        "duplicate_prior": "equal_mass_per_equivalent_group",
        "predictor_exclusions": [
            "labels",
            "annotation_offsets",
            "source_ids",
            "generator_ids",
            "roles",
        ],
        "control_arms": ["shared_location", "pooled_logistic", "pooled_mlp", "source_erased"],
        "fixture_roles": {"development": 64, "held": 32},
        "labels_opened": False,
    }
    atomic_json(raw_dir / "manifest.json", manifest)
    rows = []
    start = time.monotonic()
    for index, fixture in enumerate(old.fixture_groups()):
        source, answer = fixture["source"], fixture["answer"]
        view = energy.prepare(source, answer)
        result = energy.distribution(view, _parameters())
        rows.append(
            {
                "family_id": fixture["family_id"],
                "role": fixture["role"],
                "source_hex": source.hex(),
                "answer_hex": answer.hex(),
                "source_sha256": digest(source),
                "answer_sha256": digest(answer),
                "response_support": result["response_support"],
                "response_unsupported": result["response_unsupported"],
                "shared_control": energy.shared_control(view, old.zero_parameters()),
                "source_erased_control": energy.distribution(
                    energy.prepare(b"", answer), _parameters()
                )["response_support"],
                "normalization_error": max(
                    abs(sum(map(sum, joint)) - 1) for joint in result["joint"]
                ),
                "byte_retained": b"".join(view["source_sentences"]) == source
                and b"".join(view["answer_units"]) == answer,
                "denominator": 1,
                "censored": False,
                "excluded": False,
                "exclusions": [],
                "claim_scope": "fixture_only",
            }
        )
        if (index + 1) % 24 == 0:
            progress(start, "fixtures", "checkpoint", index + 1)
    atomic_json(raw_dir / "rows.json", rows)
    return manifest, rows


def reduce_raw(raw_dir: Path) -> dict[str, Any]:
    """Recompute raw fixtures and tiny explicit state enumeration from bytes."""
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    rows = json.loads((raw_dir / "rows.json").read_text())
    fixtures = old.fixture_groups()
    if (
        len(rows) != 96
        or len(fixtures) != 96
        or manifest["fixture_roles"] != {"development": 64, "held": 32}
    ):
        raise ValueError("fixture count changed")
    for row, fixture in zip(rows, fixtures, strict=True):
        source, answer = bytes.fromhex(row["source_hex"]), bytes.fromhex(row["answer_hex"])
        view = energy.prepare(source, answer)
        probability = energy.distribution(view, _parameters())
        if (
            (row["family_id"], row["role"], source, answer)
            != (fixture["family_id"], fixture["role"], fixture["source"], fixture["answer"])
            or row["source_sha256"] != digest(source)
            or row["answer_sha256"] != digest(answer)
        ):
            raise ValueError("fixture input changed")
        if (
            row["response_support"] != probability["response_support"]
            or row["response_unsupported"] != probability["response_unsupported"]
        ):
            raise ValueError("fixture probability changed")
        if not row["byte_retained"] or row["normalization_error"] > 1e-12:
            raise ValueError("fixture validity changed")
    tiny = [
        ("two_locations", b"Alpha is 12. Beta is 30.", b"Alpha is 12. Beta is 30."),
        ("duplicate", b"Alpha is 12. Alpha is 12.", b"Alpha is 12."),
        ("missing", b"", b"Maybe 7."),
        ("reordered", b"Beta is 30. Alpha is 12.", b"Alpha is 12."),
    ]
    normalizations = []
    for name, source, answer in tiny:
        view = energy.prepare(source, answer)
        result = energy.distribution(view, _parameters())
        explicit = energy.enumerate_response(view, _parameters())
        normalizations.append(
            {
                "case": name,
                "factorized_support": result["response_support"],
                "explicit_support": explicit,
                "absolute_error": abs(result["response_support"] - explicit),
                "normalization_error": max(
                    abs(sum(map(sum, joint)) - 1) for joint in result["joint"]
                ),
            }
        )
    original = energy.distribution(energy.prepare(b"Alpha is 12.", b"Alpha is 12."), _parameters())
    copied = energy.distribution(
        energy.prepare(b"Alpha is 12. Alpha is 12.", b"Alpha is 12."), _parameters()
    )
    duplicate_error = abs(original["response_support"] - copied["response_support"])
    boundary = {
        "empty_answer": energy.prepare(b"", b"")["abstention"],
        "empty_source_null_mass": energy.distribution(
            energy.prepare(b"", b"Maybe 7."), _parameters()
        )["null_mass"],
        "129_windows": energy.prepare(b"A. " * 66, b"B.")["abstention"],
        "17_sentences": energy.prepare(b"A.", b"B. " * 17)["abstention"],
    }
    ready = (
        all(r["byte_retained"] and r["normalization_error"] <= 1e-12 for r in rows)
        and all(
            n["absolute_error"] <= 1e-12 and n["normalization_error"] <= 1e-12
            for n in normalizations
        )
        and duplicate_error <= 1e-8
        and boundary["empty_answer"] == "empty_answer"
        and boundary["129_windows"] == "source_windows_over_budget"
        and boundary["17_sentences"] == "answer_units_over_budget"
        and boundary["empty_source_null_mass"] == [1.0]
    )
    return {
        "ready": ready,
        "rows": rows,
        "normalization_rows": normalizations,
        "boundary_fixtures": boundary,
        "duplicate_error": duplicate_error,
        "observed_families": len(rows),
    }


def cold_check(raw_dir: Path, candidate: Path) -> None:
    """Make the cold CLI reject changed candidate metrics or raw source bytes."""
    value = json.loads(candidate.read_text())
    reduction = reduce_raw(raw_dir)
    if (
        value["rows"] != reduction["rows"]
        or value["normalization_rows"] != reduction["normalization_rows"]
    ):
        raise ValueError("candidate rows differ from raw reduction")
    if value.get("set_protocol_manifest_path", {}).get("sha256") not in (
        None,
        sha256_file(raw_dir / "manifest.json"),
    ):
        raise ValueError("manifest hash changed")


def _span(
    name: str, start: float, end: float, units: int, checkpoint: Any, run_date: str
) -> dict[str, Any]:
    return {
        "phase": name,
        "start_monotonic_s": start,
        "end_monotonic_s": end,
        "duration_s": end - start,
        "run_date": run_date,
        "heartbeat_times": [start, end],
        "completed_units": units,
        "checkpoint_hash": checkpoint,
    }


def build_artifact(
    root: Path,
    date: str,
    checks: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    hashes: dict[str, Any],
    reduced: dict[str, Any] | None,
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind observed fixture limits and current validation without science claims."""
    scope = frozen_scope(root)
    passed = len(receipts) == len(scope["required_names"]) and all(
        r.get("passed") for r in receipts
    )
    ready = bool(reduced and reduced["ready"] and passed and not failures)
    verdict = "blocked" if failures else "circular_positive" if ready else "disqualified"
    gates = {
        key: None for key in ("brier_score", "decision_cost", "coverage", "retention", "efficiency")
    }
    gates.update({"validity": bool(reduced and reduced["ready"]), "readiness": None})
    raw = root / RAW_PATH
    for name in ("manifest.json", "rows.json"):
        path = raw / name
        hashes["pre_gate_receipts"].append(
            {
                "path": (RAW_PATH / name).as_posix(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    bound = json.dumps(
        {
            "manifest": hashes["pre_gate_receipts"],
            "inputs": hashes["eligible_producers"],
            "reducer": sha256_file(root / CHANGED_MODULES[1]),
            "seed": old.HASH_SEED.hex(),
        },
        sort_keys=True,
    ).encode()
    fields = [
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
        "field_principles",
        "set_protocol_ready_score",
        "set_protocol_manifest_path",
        "normalization_rows",
    ]
    return {
        "experiment_id": "exp7728-set-energy-protocol",
        "milestone": "2026.09.673",
        "run_date": date,
        "honest_verdict": f"complete_{verdict}_set_energy_protocol",
        "verdict_class": verdict,
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": gates,
        "rows": reduced["rows"] if reduced else [],
        "normalization_rows": reduced["normalization_rows"] if reduced else [],
        "boundary_fixtures": reduced["boundary_fixtures"] if reduced else {},
        "sample_size_budget": {
            "intended": 96,
            "observed": reduced["observed_families"] if reduced else 0,
            "eligible": reduced["observed_families"] if reduced and reduced["ready"] else 0,
            "excluded": 0,
            "censored": 0,
            "effective_independent_families": reduced["observed_families"] if reduced else 0,
        },
        "claim_scope": {"value": "fixture_only", "fresh_generalization_eligible": False},
        "inference_substrate": "deterministic_cpu_feature_protocol_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "model_invocation_counts": {
            key: 0
            for key in (
                "loads",
                "forwards",
                "generations",
                "input_tokens",
                "output_tokens",
                "failures",
                "cancellations",
            )
        },
        "execution_venue": "host",
        "execution_venue_details": {"host": platform.node(), "pid": os.getpid(), "gpu_uuid": None},
        "phase_spans": spans,
        "random_seed": {
            "signed_hash": old.HASH_SEED.hex(),
            "fixture_order": 0,
            "future_fit_seeds": [67301, 67302, 67303, 67304, 67305],
        },
        "reproducibility_checksum": digest(bound),
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "affected_scope": scope,
            "commands": receipts,
            "cold_replay": None,
            "terminal_readers": None,
            "global_suite_debt": "not a fixture readiness gate",
        },
        "verifier_is_oracle": True,
        "set_protocol_ready_score": int(ready),
        "set_protocol_manifest_path": {
            "path": (RAW_PATH / "manifest.json").as_posix(),
            "sha256": sha256_file(raw / "manifest.json")
            if (raw / "manifest.json").is_file()
            else None,
        },
        "field_principles": {
            name: "Measured evidence bounds the claim and downstream use."
            for name in [*fields, *gates]
        },
    }


def run_experiment(root: Path, date: str) -> dict[str, Any]:
    """Run CPU fixtures and bounded readers before publishing checked bytes."""
    root = root.resolve(strict=True)
    started = time.monotonic()
    progress(started, "preconditions", "start")
    spans: list[dict[str, Any]] = []
    phase = time.monotonic()
    checks, failures, hashes = check_preconditions(root)
    raw = root / RAW_PATH
    (raw / "tmp").mkdir(parents=True, exist_ok=True)
    scope = frozen_scope(root)
    atomic_json(raw / "validation_scope.json", scope)
    spans.append(
        _span(
            "preconditions",
            phase,
            time.monotonic(),
            len(checks),
            sha256_file(raw / "validation_scope.json"),
            date,
        )
    )
    progress(started, "preconditions", "end", len(checks))
    reduced = None
    if not failures:
        progress(started, "fixtures", "start")
        phase = time.monotonic()
        build_protocol(raw)
        reduced = reduce_raw(raw)
        spans.append(
            _span(
                "fixtures",
                phase,
                time.monotonic(),
                reduced["observed_families"],
                sha256_file(raw / "rows.json"),
                date,
            )
        )
        progress(started, "fixtures", "end", reduced["observed_families"])
    else:
        progress(started, "fixtures", "blocked")
    progress(started, "validation", "start")
    phase = time.monotonic()
    commands = build_scoped_commands(
        root,
        [TEST_PATH],
        CHANGED_MODULES,
        static_paths=[WRAPPER_PATH],
        basetemp=raw / "tmp",
        coverage_file=raw / ".coverage",
    )
    receipts = run_commands(
        root,
        commands,
        log_dir=raw / "validation_logs",
        extra_env={"COVERAGE_FILE": str(raw / ".coverage")},
        heartbeat_s=30,
    )
    spans.append(
        _span(
            "validation",
            phase,
            time.monotonic(),
            len(receipts),
            digest(json.dumps(receipts, sort_keys=True).encode()),
            date,
        )
    )
    progress(started, "validation", "end", len(receipts))
    artifact = build_artifact(root, date, checks, failures, hashes, reduced, receipts, spans)
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    terminal = [
        CommandSpec(
            "cold_replay",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "-m",
                "carnot.experiment_7728_v673_set_energy_protocol",
                "--cold-reduce",
                str(raw),
                "--candidate",
                str(candidate),
            ),
            "exact_candidate",
            300,
        ),
        CommandSpec(
            "adversarial_verify",
            (
                str(root / ".venv/bin/python"),
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
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            300,
        ),
    ]
    progress(started, "terminal", "start")
    terminal_receipts = run_commands(root, terminal, log_dir=raw / "terminal_logs", heartbeat_s=30)
    if not all(item["passed"] for item in terminal_receipts):
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["set_protocol_ready_score"] = 0
        artifact["flagged_adversarial"] = not terminal_receipts[1]["passed"]
        artifact["gate_check_summary"] = [
            {
                "check": item["name"],
                "upstream_id": "exp7728-set-energy-protocol",
                "artifact_path": str(candidate.relative_to(root)),
                "field": "exit_code",
                "operator": "==",
                "expected": 0,
                "observed": item["exit_code"],
            }
            for item in terminal_receipts
            if not item["passed"]
        ]
        atomic_json(candidate, artifact)
        terminal_receipts = run_commands(
            root, terminal, log_dir=raw / "terminal_logs", heartbeat_s=30
        )
    atomic_json(
        raw / "terminal_checks.json",
        {"candidate_sha256": sha256_file(candidate), "commands": terminal_receipts},
    )
    progress(started, "terminal", "end", len(terminal_receipts))
    result = root / RESULT_PATH
    temporary = result.with_suffix(".json.tmp")
    temporary.write_bytes(candidate.read_bytes())
    os.replace(temporary, result)
    progress(started, "publish", "end", 1)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        if args.candidate is None:
            parser.error("--candidate is required")
        cold_check(args.cold_reduce, args.candidate)
        print("cold reduction passed", flush=True)
        return 0
    result = run_experiment(Path(__file__).resolve().parents[2], args.date)
    return 0 if result["verdict_class"] in {"circular_positive", "null", "blocked"} else 1


if __name__ == "__main__":
    sys.exit(main())

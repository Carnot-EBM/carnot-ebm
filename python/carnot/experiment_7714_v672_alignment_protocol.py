"""Qualify the CPU source-alignment protocol without natural-label claims.

REQ-REPORT-7714. Raw fixture rows are immutable inputs to a fresh-process
reducer. Existing V671 verdicts are provenance, never relabeled by this run.
"""

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
from carnot.verify import source_alignment as alignment

EXPERIMENT_ID = "experiment_7714_v672_alignment_protocol"
RAW_PATH = Path("results/raw") / EXPERIMENT_ID
RESULT_PATH = Path("results") / f"{EXPERIMENT_ID}.json"
TEST_PATH = "tests/python/test_experiment_7714_v672_alignment_protocol.py"
CHANGED_MODULES = (
    "python/carnot/verify/source_alignment.py",
    "python/carnot/experiment_7714_v672_alignment_protocol.py",
)
WRAPPER_PATH = "scripts/experiments/experiment_7714_v672_alignment_protocol.py"
UPSTREAM = (
    ("results/experiment_7700_v671_record_span_protocol.json", "record_protocol_ready_score", 1),
    ("results/experiment_7704_v671_heldout_decisions.json", "verdict_class", "null"),
)
PRINCIPLE_DEFAULT = "Measured evidence bounds the claim and prevents invalid downstream use."
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal custody prevents retries of unchanged external blocks.",
    "verdict_class": "Claim eligibility travels with the result.",
    "gate_check_summary": "Exact operands distinguish scientific failure from a broken interface.",
    "rows": "Every comparison must be reproducible without rerunning the experiment.",
    "inference_substrate_class": "Duration floors must match real computation.",
    "MODEL_SPECS": "Experimental model identity must match actual invocation.",
    "validation_receipts": "Required checks must pass before a result opens downstream execution.",
    "verifier_is_oracle": "Fixture correctness does not prove an independent learned advantage.",
}


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def progress(started: float, phase: str, event: str, completed: int = 0) -> None:
    print(
        f"[exp7714] {phase} {event} elapsed_s={time.monotonic() - started:.3f} completed={completed}",
        flush=True,
    )


def check_preconditions(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Check exact upstream operands and record absent bytes as a block."""
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        "valid_producers": [],
        "flagged_historical_evidence": [],
        "pre_gate_receipts": [],
        "missing_custody": [],
    }
    for relative, field, expected in UPSTREAM:
        path = root / relative
        exists = path.is_file()
        check: dict[str, Any] = {
            "artifact_path": relative,
            "exists": exists,
            "field": field,
            "expected": expected,
        }
        observed: Any = None
        if exists:
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, dict):
                    raise ValueError("upstream schema must be an object")
                observed = data.get(field)
                check.update(
                    {
                        "schema_object": isinstance(data, dict),
                        "sha256": sha256_file(path),
                        "verdict_class": data.get("verdict_class"),
                        "flagged_adversarial": data.get("flagged_adversarial"),
                    }
                )
            except (ValueError, OSError):
                check["schema_object"] = False
        check["observed"] = observed
        eligible = (
            exists
            and check.get("schema_object") is True
            and observed == expected
            and check.get("flagged_adversarial") is False
        )
        check["eligible"] = eligible
        checks.append(check)
        if eligible:
            hashes["valid_producers"].append({"path": relative, "sha256": check["sha256"]})
        else:
            failures.append(
                {
                    "check": "upstream_eligibility",
                    "upstream_id": Path(relative).stem,
                    "artifact_path": relative,
                    "field": field if exists else "exists",
                    "operator": "==",
                    "expected": expected if exists else True,
                    "observed": observed if exists else False,
                }
            )
            hashes["missing_custody"].append(relative)
    return checks, failures, hashes


def _parameters() -> dict[str, list[Any]]:
    params = alignment.zero_parameters()
    params["weights"][0][128:131] = [-2.0, 0.8, 1.1]
    params["weights"][1][128:131] = [2.0, -0.8, -1.1]
    return params


def build_protocol(raw_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Seal fixtures and feature choices before any fresh label is opened."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    params = _parameters()
    started = time.monotonic()
    for index, fixture in enumerate(alignment.fixture_groups()):
        source, answer = fixture["source"], fixture["answer"]
        view = alignment.prepare(source, answer)
        distribution = alignment.distribution(view, params)
        enumerated = alignment.enumerate_marginal(view, params)
        erased = alignment.prepare(b"", answer)
        shuffled = alignment.prepare(b"".join(reversed(view["source_sentences"])), answer)
        rows.append(
            {
                "family_id": fixture["family_id"],
                "role": fixture["role"],
                "source_hex": source.hex(),
                "answer_hex": answer.hex(),
                "source_sha256": _digest(source),
                "answer_sha256": _digest(answer),
                "source_windows": len(view["windows"]),
                "answer_units": len(view["answer_units"]),
                "abstention": view["abstention"],
                "byte_reconstruction": b"".join(view["source_sentences"]) == source
                and b"".join(view["answer_units"]) == answer,
                "marginal": distribution["marginal"],
                "null_mass": distribution["null_mass"],
                "normalization_sum": sum(sum(pair) for pair in distribution["joint"]),
                "enumeration_error": max(
                    abs(a - b) for a, b in zip(distribution["marginal"], enumerated, strict=True)
                ),
                "erased_feature_changed": view["pair_features"] != erased["pair_features"],
                "shuffled_feature_changed": view["pair_features"] != shuffled["pair_features"],
                "provenance": "deterministic_fixture_oracle",
                "denominator": 1,
                "excluded": False,
                "censored": False,
            }
        )
        if (index + 1) % 24 == 0:
            progress(started, "fixtures", "checkpoint", index + 1)
    row_bytes = _json_bytes(rows)
    (raw_dir / "rows.json").write_bytes(row_bytes)
    manifest = {
        "protocol_version": 1,
        "fixture_count": 96,
        "held_count": 32,
        "rows_sha256": _digest(row_bytes),
        "hash_seed_hex": alignment.HASH_SEED.hex(),
        "token_pattern": alignment.TOKEN_PATTERN,
        "hash_bins": alignment.HASH_BINS,
        "feature_dim": alignment.FEATURE_DIM,
        "max_windows": alignment.MAX_WINDOWS,
        "max_answer_units": alignment.MAX_ANSWER_UNITS,
        "labels_opened": False,
        "label_semantics": ["human_unsupported_present", "human_unsupported_absent"],
        "latent_states": "one visible source window or null evidence",
        "duplicate_prior": "equal unique-window-group prior",
        "controls": ["same_input_pooled_logistic", "same_input_pooled_mlp", "source_erased"],
    }
    atomic_json(raw_dir / "protocol.json", manifest)
    return manifest, rows


def reduce_raw(raw_dir: Path) -> dict[str, Any]:
    """Reopen sealed bytes and recompute fixture truth from original inputs."""
    manifest = json.loads((raw_dir / "protocol.json").read_text())
    raw_bytes = (raw_dir / "rows.json").read_bytes()
    if _digest(raw_bytes) != manifest["rows_sha256"]:
        raise ValueError("raw rows hash changed")
    rows = json.loads(raw_bytes)
    fixtures = alignment.fixture_groups()
    if len(rows) != 96 or len(fixtures) != 96 or manifest["held_count"] != 32:
        raise ValueError("fixture count changed")
    normalization_rows = []
    for row, fixture in zip(rows, fixtures, strict=True):
        source = bytes.fromhex(row["source_hex"])
        answer = bytes.fromhex(row["answer_hex"])
        if (row["family_id"], row["role"], source, answer) != (
            fixture["family_id"],
            fixture["role"],
            fixture["source"],
            fixture["answer"],
        ):
            raise ValueError("fixture source or answer changed")
        view = alignment.prepare(source, answer)
        computed = alignment.distribution(view, _parameters())
        enumerated = alignment.enumerate_marginal(view, _parameters())
        error = max(abs(a - b) for a, b in zip(computed["marginal"], enumerated, strict=True))
        if (
            not row["byte_reconstruction"]
            or b"".join(view["source_sentences"]) != source
            or b"".join(view["answer_units"]) != answer
        ):
            raise ValueError("byte reconstruction failed")
        if row["source_sha256"] != _digest(source) or row["answer_sha256"] != _digest(answer):
            raise ValueError("source hash changed")
        if row["marginal"] != computed["marginal"] or row["enumeration_error"] != error:
            raise ValueError("normalization row changed")
        if not all(
            math_isfinite(value) for value in [*computed["marginal"], computed["null_mass"], error]
        ):
            raise ValueError("nonfinite probability")
        normalization_rows.append(
            {
                "family_id": row["family_id"],
                "role": row["role"],
                "marginal": computed["marginal"],
                "null_mass": computed["null_mass"],
                "normalization_sum": sum(sum(pair) for pair in computed["joint"]),
                "absolute_error": error,
                "enumerated_marginal": enumerated,
            }
        )
    ready = all(
        row["abstention"] is None
        and row["erased_feature_changed"]
        and row["shuffled_feature_changed"]
        and abs(row["normalization_sum"] - 1.0) <= 1e-12
        and row["enumeration_error"] <= 1e-12
        for row in rows
    )
    return {
        "ready": ready,
        "rows": rows,
        "normalization_rows": normalization_rows,
        "fixture_count": len(rows),
    }


def math_isfinite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def _historical_hashes(root: Path, hashes: dict[str, Any]) -> None:
    for relative, category in (
        ("results/experiment_7712_v671_capstone.json", "flagged_historical_evidence"),
        ("results/experiment_7713_v672_contract_methods.json", "pre_gate_receipts"),
    ):
        path = root / relative
        if path.is_file():
            hashes[category].append({"path": relative, "sha256": sha256_file(path)})
        else:
            hashes["missing_custody"].append(relative)


def build_artifact(
    root: Path,
    run_date: str,
    checks: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    hashes: dict[str, Any],
    reduction: dict[str, Any] | None,
    receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind measurements and required check exits to one terminal claim."""
    required_passed = len(receipts) == 8 and all(receipt["passed"] for receipt in receipts)
    ready = bool(reduction and reduction["ready"] and not failures and required_passed)
    if failures:
        verdict_class, honest_verdict = "blocked", "complete_blocked_upstream_eligibility"
    elif not required_passed or not reduction or not reduction["ready"]:
        verdict_class, honest_verdict = "disqualified", "complete_disqualified_required_validation"
    else:
        verdict_class, honest_verdict = (
            "circular_positive",
            "complete_circular_positive_alignment_protocol_ready",
        )
    protocol_path = root / RAW_PATH / "protocol.json"
    source_hashes = dict(hashes)
    _historical_hashes(root, source_hashes)
    source_hashes["current_protocol"] = (
        {
            "path": RAW_PATH.joinpath("protocol.json").as_posix(),
            "sha256": sha256_file(protocol_path),
        }
        if protocol_path.is_file()
        else None
    )
    source_hashes["current_rows"] = (
        {
            "path": RAW_PATH.joinpath("rows.json").as_posix(),
            "sha256": sha256_file(root / RAW_PATH / "rows.json"),
        }
        if (root / RAW_PATH / "rows.json").is_file()
        else None
    )
    reproducibility = _digest(
        _json_bytes(
            {
                "source_hashes": source_hashes,
                "protocol": source_hashes["current_protocol"],
                "reducer": sha256_file(root / CHANGED_MODULES[1]),
            }
        )
    )
    gates = {
        key: None
        for key in (
            "probability",
            "utility",
            "coverage",
            "source_dependence",
            "retention",
            "efficiency",
        )
    }
    gates.update(
        {"measured_validity": bool(reduction and reduction["ready"]), "readiness": int(ready)}
    )
    scope = {
        "test_paths": [TEST_PATH],
        "changed_modules": list(CHANGED_MODULES),
        "static_paths": [WRAPPER_PATH],
        "required_names": [
            item.name
            for item in build_scoped_commands(
                root,
                [TEST_PATH],
                CHANGED_MODULES,
                static_paths=[WRAPPER_PATH],
                basetemp=root / RAW_PATH / "tmp",
                coverage_file=root / RAW_PATH / ".coverage",
            )
        ],
    }
    fields = [
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
        "source_artifact_hashes",
        "preconditions_checked",
        "validation_receipts",
        "verifier_is_oracle",
        "field_principles",
        "alignment_protocol_ready_score",
        "alignment_protocol_path",
        "normalization_rows",
    ]
    return {
        "experiment_id": "exp7714-alignment-protocol",
        "milestone": "2026.09.672",
        "run_date": run_date,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failures
        if failures
        else (
            []
            if ready
            else [
                {
                    "check": "required_validation",
                    "upstream_id": "exp7714-alignment-protocol",
                    "artifact_path": RESULT_PATH.as_posix(),
                    "field": "alignment_protocol_ready_score",
                    "operator": "==",
                    "expected": 1,
                    "observed": 0,
                }
            ]
        ),
        "acceptance_gate_results": gates,
        "rows": reduction["rows"] if reduction else [],
        "normalization_rows": reduction["normalization_rows"] if reduction else [],
        "sample_size_budget": {
            "intended_families": 96,
            "observed_families": reduction["fixture_count"] if reduction else 0,
            "eligible_families": reduction["fixture_count"]
            if reduction and reduction["ready"]
            else 0,
            "excluded_families": 0,
            "censored_families": 0,
            "roles": {"qualification": 64, "held": 32},
            "exposure": "fixture only; no natural labels",
            "effective_blocks": reduction["fixture_count"] if reduction else 0,
        },
        "inference_substrate": "deterministic_cpu_feature_protocol_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": {
            key: 0
            for key in ("loads", "forwards", "generations", "tokens", "failures", "cancellations")
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": platform.node(),
            "pid": os.getpid(),
            "backend": "cpu",
            "gpu_uuid": None,
        },
        "phase_spans": spans,
        "random_seed": {
            "hash_seed_hex": alignment.HASH_SEED.hex(),
            "fixture_order": "sequential 0..95",
            "purposes": ["signed_hashing", "fixture_identity"],
        },
        "reproducibility_checksum": reproducibility,
        "source_artifact_hashes": source_hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "affected_scope": scope,
            "commands": receipts,
            "e2e": "fixture qualification and cold replay",
            "cold_replay": None,
            "terminal_check_receipt_path": (RAW_PATH / "terminal_checks.json").as_posix(),
            "global_debt": "full-suite check is separate from scoped qualification",
        },
        "verifier_is_oracle": True,
        "alignment_protocol_ready_score": int(ready),
        "alignment_protocol_path": {
            "path": (RAW_PATH / "protocol.json").as_posix(),
            "sha256": sha256_file(protocol_path) if protocol_path.is_file() else None,
        },
        "field_principles": {
            key: FIELD_PRINCIPLES.get(key, PRINCIPLE_DEFAULT) for key in [*fields, *gates]
        },
    }


def cold_check(raw_dir: Path, candidate_path: Path) -> None:
    """A fresh interpreter must derive the candidate's rows from raw bytes."""
    value = json.loads(candidate_path.read_text())
    reduced = reduce_raw(raw_dir)
    if (
        value["rows"] != reduced["rows"]
        or value["normalization_rows"] != reduced["normalization_rows"]
    ):
        raise ValueError("candidate rows differ from cold reduction")
    if value["alignment_protocol_path"]["sha256"] != sha256_file(raw_dir / "protocol.json"):
        raise ValueError("protocol hash changed")
    if value["verdict_class"] == "circular_positive" and (
        not reduced["ready"] or not value["verifier_is_oracle"]
    ):
        raise ValueError("fixture verdict exceeds evidence")


def _span(
    spans: list[dict[str, Any]],
    name: str,
    start: float,
    end: float,
    completed: int,
    checkpoint: str | None,
) -> None:
    spans.append(
        {
            "phase": name,
            "start_monotonic_s": start,
            "end_monotonic_s": end,
            "duration_s": end - start,
            "heartbeat_timestamps": [start, end],
            "completed_units": completed,
            "checkpoint": checkpoint,
        }
    )


def run_experiment(root: Path, run_date: str) -> dict[str, Any]:
    """Run a bounded CPU qualification and publish the exact checked bytes."""
    root = root.resolve(strict=True)
    started = time.monotonic()
    progress(started, "preconditions", "start")
    spans: list[dict[str, Any]] = []
    phase_start = time.monotonic()
    checks, failures, hashes = check_preconditions(root)
    raw_dir = root / RAW_PATH
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "tmp").mkdir(exist_ok=True)
    _span(spans, "preconditions", phase_start, time.monotonic(), len(checks), None)
    progress(started, "preconditions", "end", len(checks))
    reduction = None
    if not failures:
        progress(started, "fixtures", "start")
        phase_start = time.monotonic()
        build_protocol(raw_dir)
        reduction = reduce_raw(raw_dir)
        _span(
            spans,
            "fixtures",
            phase_start,
            time.monotonic(),
            reduction["fixture_count"],
            "rows.json",
        )
        progress(started, "fixtures", "end", reduction["fixture_count"])
    else:
        progress(started, "fixtures", "blocked")
    progress(started, "validation", "start")
    phase_start = time.monotonic()
    commands = build_scoped_commands(
        root,
        [TEST_PATH],
        CHANGED_MODULES,
        static_paths=[WRAPPER_PATH],
        basetemp=raw_dir / "tmp",
        coverage_file=raw_dir / ".coverage",
    )
    receipts = run_commands(
        root,
        commands,
        log_dir=raw_dir / "validation_logs",
        extra_env={"COVERAGE_FILE": str(raw_dir / ".coverage")},
        heartbeat_s=30.0,
    )
    _span(spans, "validation", phase_start, time.monotonic(), len(receipts), "validation_logs")
    progress(started, "validation", "end", len(receipts))
    artifact = build_artifact(root, run_date, checks, failures, hashes, reduction, receipts, spans)
    candidate = raw_dir / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    terminal_commands = [
        CommandSpec(
            "independent_cold_replay",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "-m",
                "carnot.experiment_7714_v672_alignment_protocol",
                "--cold-reduce",
                str(raw_dir),
                "--candidate",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
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
            300.0,
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
            300.0,
        ),
    ]
    progress(started, "terminal", "start")
    terminal = run_commands(
        root, terminal_commands, log_dir=raw_dir / "terminal_logs", heartbeat_s=30.0
    )
    if not all(item["passed"] for item in terminal):
        artifact["flagged_adversarial"] = not terminal[1]["passed"]
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["alignment_protocol_ready_score"] = 0
        artifact["acceptance_gate_results"]["readiness"] = 0
        artifact["gate_check_summary"] = [
            {
                "check": item["name"],
                "upstream_id": "exp7714-alignment-protocol",
                "artifact_path": str(candidate.relative_to(root)),
                "field": "exit_code",
                "operator": "==",
                "expected": 0,
                "observed": item["exit_code"],
            }
            for item in terminal
            if not item["passed"]
        ]
        atomic_json(candidate, artifact)
        terminal = run_commands(
            root, terminal_commands, log_dir=raw_dir / "terminal_logs", heartbeat_s=30.0
        )
    atomic_json(
        raw_dir / "terminal_checks.json",
        {"candidate_sha256": sha256_file(candidate), "commands": terminal},
    )
    progress(started, "terminal", "end", len(terminal))
    result = root / RESULT_PATH
    result.parent.mkdir(parents=True, exist_ok=True)
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
            parser.error("--candidate is required for cold reduction")
        cold_check(args.cold_reduce, args.candidate)
        print("cold reduction passed", flush=True)
        return 0
    value = run_experiment(Path(__file__).resolve().parents[2], args.date)
    return 0 if value["verdict_class"] in {"circular_positive", "null", "blocked"} else 1


if __name__ == "__main__":
    sys.exit(main())

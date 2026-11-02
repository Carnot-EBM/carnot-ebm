"""Freeze V664 source features with no evaluator access in the feature worker.

The worker sees only predictor records. The parent authenticates evaluator
bytes, but never passes labels into extraction. Spec: REQ-REPORT-7646.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7303_validation_scope as checks
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7646_source_features import (
    cold_reconstruct,
    digest,
    extract_role,
    immutable_jsonl,
    read_jsonl,
    validate_model_row,
)


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path("results/experiment_7602_v664_evidence_requalification.json")
PROTOCOL = Path("results/raw/experiment_7602_v664_evidence_requalification/protocol.json")
RAW = Path("results/raw/experiment_7646_v667_source_feature_corpus")
RESULT = Path("results/experiment_7646_v667_source_feature_corpus.json")
ROLES = ("fit", "tune", "policy", "online", "evaluation", "pilot")
COUNTS = {"fit": 80, "tune": 20, "policy": 20, "online": 80, "evaluation": 40, "pilot": 8}
ARMS = ("original_source", "evidence_erasure", "within_role_derangement")
TEST = "tests/python/test_experiment_7646_v667_source_feature_corpus.py"
MODULE = "python/carnot/experiment_7646_v667_source_feature_corpus.py"
BEHAVIOR = "python/carnot/reporting/experiment_7646_source_features.py"
WRAPPER = "scripts/experiments/experiment_7646_v667_source_feature_corpus.py"


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover - E2E
    """Make slow phase boundaries and completed units visible to the owner."""

    print(
        f"[exp7646] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {details}",
        flush=True,
    )


def check(name: str, upstream: str, path: Path, field: str, expected: Any, observed: Any) -> dict:  # pragma: no cover - E2E
    """Retain exact failed operands so an external block can be repaired."""

    return {
        "check": name,
        "upstream": upstream,
        "path": str(path.resolve()),
        "field": field,
        "operator": "eq",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def authenticate_inputs(root: Path) -> dict:  # pragma: no cover - E2E
    """Authenticate source custody without treating a planned output as input."""

    root = root.resolve()
    checks_list: list[dict] = []
    upstream_path, protocol_path = root / UPSTREAM, root / PROTOCOL
    for path in (upstream_path, protocol_path):
        checks_list.append(check("input_exists", "exp7602", path, "is_file", True, path.is_file()))
    if any(not item["passed"] for item in checks_list):
        return {"checks": checks_list, "protocol": None, "source_hashes": {}}
    upstream = json.loads(upstream_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    checks_list.extend(
        (
            check(
                "upstream_verdict",
                "exp7602",
                upstream_path,
                "verdict_class",
                "null",
                upstream.get("verdict_class"),
            ),
            check(
                "upstream_flag",
                "exp7602",
                upstream_path,
                "flagged_adversarial",
                False,
                upstream.get("flagged_adversarial"),
            ),
            check(
                "upstream_ready",
                "exp7602",
                upstream_path,
                "evidence_protocol_ready_score",
                1,
                upstream.get("evidence_protocol_ready_score"),
            ),
            check(
                "protocol_hash",
                "exp7602",
                protocol_path,
                "sha256",
                upstream.get("protocol_sha256"),
                sha256_file(protocol_path),
            ),
            check(
                "role_counts",
                "exp7602",
                protocol_path,
                "role_counts",
                COUNTS,
                protocol.get("role_counts"),
            ),
            check(
                "scored_groups",
                "exp7602",
                protocol_path,
                "scored_group_count",
                240,
                protocol.get("scored_group_count"),
            ),
            check(
                "pilot_groups",
                "exp7602",
                protocol_path,
                "pilot_group_count",
                8,
                protocol.get("pilot_group_count"),
            ),
            check(
                "fit_partition",
                "exp7602",
                protocol_path,
                "fit_partition_counts",
                {"optimization": 64, "old_distribution_anchor": 16},
                protocol.get("fit_partition_counts"),
            ),
        )
    )
    receipts = protocol.get("reader_sidecars", {})
    hashes: dict[str, Any] = {
        "producer_files": {
            str(UPSTREAM): sha256_file(upstream_path),
            str(PROTOCOL): sha256_file(protocol_path),
        },
        "pre_gate_receipts": {},
        "missing_inputs": [],
        "planned_outputs": [str(RESULT), str(RAW / "manifest.json")],
    }
    for store in ("model_inputs", "evaluator_stores"):
        for role in ROLES:
            receipt = receipts.get(store, {}).get(role, {})
            label = receipt.get("path", str(PROTOCOL.parent / f"{role}_{store}.jsonl"))
            path = root / label
            observed = sha256_file(path) if path.is_file() else None
            checks_list.append(
                check(
                    "sidecar_hash",
                    f"exp7602:{store}:{role}",
                    path,
                    "sha256",
                    receipt.get("sha256"),
                    observed,
                )
            )
            checks_list.append(
                check(
                    "sidecar_rows",
                    f"exp7602:{store}:{role}",
                    path,
                    "rows",
                    COUNTS[role],
                    receipt.get("rows"),
                )
            )
            if observed is None:
                hashes["missing_inputs"].append(label)
            else:
                hashes["producer_files"][label] = observed
    return {"checks": checks_list, "protocol": protocol, "source_hashes": hashes}


def _worker(
    input_path: Path, output_path: Path, role: str, mapping: dict[str, str], denied_path: Path
) -> None:  # pragma: no cover - isolated E2E
    """Run under a mount boundary that hides all evaluator sidecars."""

    if denied_path.exists():
        raise ValueError("evaluator_store_visible")
    rows = read_jsonl(input_path)
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in extract_role(rows, role, mapping)
    )
    output_path.write_text(payload)


def _gate(name: str, passed: bool | None, principle: str, operands: dict) -> dict:  # pragma: no cover - E2E
    return {"gate": name, "passed": passed, "principle": principle, "measured_operands": operands}


def build_artifact(
    auth: dict,
    manifest: dict | None,
    rows: list[dict],
    receipts: list[dict],
    spans: list[dict],
    duration: float,
    run_date: str,
) -> dict:  # pragma: no cover - E2E
    """Separate complete corpus accounting from unmeasured scientific benefit."""

    failures = [item for item in auth["checks"] if not item["passed"]]
    complete = not failures and len(rows) == 248 * len(ARMS)
    valid = complete and all(receipt.get("passed") for receipt in receipts)
    class_name = "blocked" if failures else "null" if valid else "disqualified"
    verdict = (
        "complete_blocked_external_input"
        if failures
        else "complete_null_source_features_ready"
        if valid
        else "complete_disqualified_required_validation"
    )
    coverage = {}
    for role in ROLES:
        original = [row for row in rows if row["role"] == role and row["arm"] == "original_source"]
        coverage[role] = {
            "groups": len(original),
            "valid_source_scopes": sum(row["syntactic_parse_coverage"] for row in original),
            "checked_predicates": sum(row["checked_predicates"] for row in original),
            "unknown_groups": sum(row["censored"] for row in original),
            "unchecked_prose": sum(row["unchecked_prose"] for row in original),
        }
    gates = [
        _gate(
            "validity",
            valid,
            "Authenticated bytes and passing current readers govern validity.",
            {
                "failed_preconditions": len(failures),
                "required_receipts": len(receipts),
                "passed_receipts": sum(bool(r.get("passed")) for r in receipts),
            },
        ),
        _gate(
            "readiness",
            complete,
            "All inherited groups need honest feature rows, regardless of coverage.",
            {
                "observed_groups": len(rows) // len(ARMS),
                "intended_groups": 248,
                "authenticated_sidecars": not failures,
            },
        ),
        _gate(
            "probability_benefit",
            None,
            "Proper loss needs independent paired probabilities and labels.",
            {"paired_probabilities": 0},
        ),
        _gate(
            "utility",
            None,
            "Decision value needs typed actions, costs and outcomes.",
            {"typed_decisions": 0},
        ),
        _gate(
            "retention",
            None,
            "Retention needs delayed feedback and later replay.",
            {"delayed_feedback_events": 0},
        ),
        _gate(
            "freshness",
            None,
            "Historically exposed groups cannot be fresh confirmation.",
            {"fresh_independent_groups": 0},
        ),
    ]
    source_hashes = auth["source_hashes"]
    checksum_payload = {
        "source_hashes": source_hashes,
        "manifest": manifest,
        "reduction_code": sha256_file(ROOT / MODULE),
        "roles": COUNTS,
    }
    artifact = {
        "schema": "carnot.exp7646.v667.source_feature_corpus.v1",
        "experiment_id": "exp7646-v667-source-feature-corpus",
        "milestone": "2026.09.667",
        "run_date": run_date,
        "honest_verdict": verdict,
        "verdict_class": class_name,
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended_independent_groups": 240,
            "observed_independent_groups": len(
                {r["unit_id"] for r in rows if r["role"] != "pilot"}
            ),
            "pilot_groups": len({r["unit_id"] for r in rows if r["role"] == "pilot"}),
            "eligible": 240 if complete else 0,
            "excluded": 0,
            "censored": sum(
                r["censored"]
                for r in rows
                if r["arm"] == "original_source" and r["role"] != "pilot"
            ),
            "exposure_limits": "All groups historically exposed; three arms are repeated views, not new samples.",
        },
        "preconditions_checked": auth["checks"],
        "inference_substrate": "deterministic source-witness replay on host CPU",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {"loads": 0, "forwards": 0, "generations": 0, "tokens": 0},
        "execution_venue": "host",
        "execution_venue_details": {"hostname": socket.gethostname(), "owned_pid": os.getpid()},
        "phase_spans": spans,
        "duration_s": duration,
        "random_seed": {"source_derangement": "inherited frozen V664 map", "resampling": None},
        "reproducibility_checksum": digest(
            json.dumps(checksum_payload, sort_keys=True, default=str)
        ),
        "source_artifact_hashes": source_hashes,
        "validation_receipts": receipts,
        "verifier_is_oracle": True,
        "field_principles": {
            "honest_verdict": "Completion is separate from scientific benefit.",
            "verdict_class": "Required checks govern the terminal class.",
            "rows": "Each group counts once; controls are repeated views.",
            "source_features_ready_score": "Honest complete rows and authenticated sidecars govern readiness.",
            "probability_benefit": "Proper loss needs oracle-distinct labels and probabilities.",
            "utility": "Decision value needs typed costs and outcomes.",
            "retention": "Retention needs delayed feedback.",
            "freshness": "Historical exposure bars fresh claims.",
            "flagged_adversarial": "A flagged reader cannot open a downstream gate.",
        },
        "source_features_ready_score": int(complete),
        "feature_manifest_path": str(RAW / "manifest.json"),
        "role_counts": {key: COUNTS[key] for key in ROLES if key != "pilot"},
        "pilot_group_count": 8,
        "coverage_by_role": coverage,
        "fresh_confirmatory_claim_allowed": False,
        "historical_model_id": "unsloth/Qwen3.8-27B-GGUF",
    }
    return artifact


def _span(started: float, name: str, begin: float, units: int, checkpoint: str) -> dict:  # pragma: no cover - E2E
    end = time.monotonic() - started
    return {
        "phase": name,
        "start_offset_s": begin,
        "end_offset_s": end,
        "duration_s": end - begin,
        "completed_units": units,
        "checkpoint": checkpoint,
    }


def _mapping(protocol: dict, role: str) -> dict[str, str]:  # pragma: no cover - E2E
    groups = [item["component_hash"] for item in protocol["roster"] if item["role"] == role]
    if len(groups) != COUNTS[role] or len(set(groups)) != len(groups):
        raise ValueError("roster_role_count_invalid")
    if role == "pilot":
        return dict(zip(groups, groups[1:] + groups[:1], strict=True))
    mapping = {group: protocol["within_role_derangement"][group] for group in groups}
    if set(mapping.values()) != set(groups) or any(key == value for key, value in mapping.items()):
        raise ValueError("derangement_role_invalid")
    return mapping


def _reader_plan(candidate: Path, isolated: list[str], isolated_candidate: Path) -> list[checks.CommandSpec]:  # pragma: no cover - E2E
    python = str(ROOT / ".venv/bin/python")
    module = "carnot.experiment_7646_v667_source_feature_corpus"
    return [
        checks.CommandSpec(
            "cold_replay",
            tuple([*isolated, python, "-u", "-m", module, "--cold-replay", str(isolated_candidate)]),
            "exact_candidate",
        ),
        checks.CommandSpec(
            "independent_reduce",
            tuple(
                [*isolated, python, "-u", "-m", module, "--independent-reduce", str(isolated_candidate)]
            ),
            "exact_candidate",
        ),
        checks.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
        ),
        checks.CommandSpec(
            "verdict_row_consistency",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
        ),
    ]


def run_experiment(root: Path, run_date: str, output: Path) -> dict:  # pragma: no cover - E2E
    """Run authenticated CPU extraction and publish only after terminal readers."""

    started = time.monotonic()
    root = root.resolve()
    progress(started, "preconditions", "begin", root=str(root))
    auth = authenticate_inputs(root)
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    spans = [_span(started, "preconditions", 0.0, len(auth["checks"]), str(PROTOCOL))]
    progress(started, "preconditions", "end", failed=sum(not c["passed"] for c in auth["checks"]))
    if any(not item["passed"] for item in auth["checks"]):
        blocked = build_artifact(auth, None, [], [], spans, time.monotonic() - started, run_date)
        atomic_json(output, blocked)
        progress(started, "publication", "after_atomic", verdict=blocked["honest_verdict"])
        return blocked
    protocol = auth["protocol"]
    progress(started, "features", "begin")
    begin = time.monotonic() - started
    roles: dict[str, dict] = {}
    mapping_all: dict[str, str] = {}
    rows: list[dict] = []
    worker_receipts: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="exp7646-isolated-") as temporary:
        private = Path(temporary)
        snapshot = private / "predictor_inputs"
        snapshot.mkdir()
        shutil.copyfile(root / PROTOCOL, snapshot / "protocol.json")
        for index, role in enumerate(ROLES):
            mapping = _mapping(protocol, role)
            mapping_all.update(mapping)
            input_path = root / PROTOCOL.parent / f"{role}_model_inputs.jsonl"
            snapshot_input = snapshot / input_path.name
            shutil.copyfile(input_path, snapshot_input)
            output_path = private / f"{role}.features.jsonl"
            output_path.touch()
            denied = root / PROTOCOL.parent / f"{role}_evaluator_store.jsonl"
            argv = (
                "bwrap",
                "--ro-bind",
                "/",
                "/",
                "--tmpfs",
                str(root / "results"),
                "--bind",
                str(output_path),
                str(output_path),
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--",
                str(root / ".venv/bin/python"),
                "-u",
                "-m",
                "carnot.experiment_7646_v667_source_feature_corpus",
                "--worker",
                str(snapshot_input),
                str(output_path),
                role,
                json.dumps(mapping),
                str(denied),
            )
            progress(started, "features", "before_subprocess", role=role)
            worker_receipts.extend(
                checks.run_commands(
                    root,
                    [checks.CommandSpec(f"worker_{role}", argv, "isolated_predictor", 600)],
                    log_dir=raw / "validation" / "workers",
                    heartbeat_s=30,
                )
            )
            progress(started, "features", "after_subprocess", role=role)
            if not worker_receipts[-1]["passed"]:
                raise RuntimeError(f"isolated_feature_worker_failed:{role}")
            role_rows = read_jsonl(output_path)
            feature_path = raw / f"{role}_features.jsonl"
            feature_hash = immutable_jsonl(feature_path, role_rows)
            group_ids = [
                item["component_hash"] for item in protocol["roster"] if item["role"] == role
            ]
            if [row["unit_id"] for row in role_rows[::3]] != group_ids:
                raise ValueError("worker_group_roster_mismatch")
            roles[role] = {
                "input_path": str(input_path.relative_to(root)),
                "input_sha256": sha256_file(input_path),
                "feature_path": str(feature_path.relative_to(root)),
                "feature_sha256": feature_hash,
                "group_ids": group_ids,
                "rows": len(role_rows),
            }
            rows.extend(role_rows)
            progress(
                started, "features", "checkpoint", completed_units=index + 1, groups=len(rows) // 3
            )
        manifest = {
            "schema": "carnot.exp7646.feature_manifest.v1",
            "roles": roles,
            "derangement": mapping_all,
            "protocol_sha256": sha256_file(root / PROTOCOL),
            "roster_sha256": protocol["roster_sha256"],
            "worker_isolation": "bubblewrap results tmpfs; only predictor snapshots mounted",
        }
        atomic_json(raw / "manifest.json", manifest)
        spans.append(_span(started, "features", begin, len(rows), str(RAW / "manifest.json")))
        progress(started, "features", "end", rows=len(rows))
        affected = {
            "experiment_id": "exp7646-v667-source-feature-corpus",
            "test_paths": [TEST],
            "changed_modules": [MODULE],
            "static_paths": [WRAPPER],
            "protocol_sha256": manifest["protocol_sha256"],
        }
        atomic_json(raw / "affected_validation_manifest.json", affected)
        progress(started, "affected_validation", "before_subprocess")
        begin = time.monotonic() - started
        base = private / "basetemp"
        base.mkdir()
        plan = checks.build_scoped_commands(
            root,
            [TEST],
            [MODULE],
            static_paths=[WRAPPER],
            basetemp=base,
            coverage_file=private / "coverage.data",
        )
        receipts = checks.run_commands(
            root,
            plan,
            log_dir=raw / "validation" / "affected",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / "coverage.data")},
            heartbeat_s=30,
        )
        spans.append(
            _span(
                started,
                "affected_validation",
                begin,
                len(receipts),
                str(RAW / "validation/affected"),
            )
        )
        progress(
            started,
            "affected_validation",
            "after_subprocess",
            passed=sum(r["passed"] for r in receipts),
        )
        candidate = raw / "exact_terminal_candidate.json"
        artifact = build_artifact(
            auth,
            manifest,
            rows,
            [*worker_receipts, *receipts],
            spans,
            time.monotonic() - started,
            run_date,
        )
        atomic_json(candidate, artifact)
        isolated_candidate = private / "candidate.json"
        shutil.copyfile(candidate, isolated_candidate)
        isolated = [
            "bwrap",
            "--ro-bind",
            "/",
            "/",
            "--tmpfs",
            str(root / "results"),
            "--dir",
            str(root / "results/raw"),
            "--ro-bind",
            str(snapshot),
            str(root / PROTOCOL.parent),
            "--ro-bind",
            str(raw),
            str(raw),
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--",
        ]
        terminal_plan = _reader_plan(candidate, isolated, isolated_candidate)
        progress(started, "terminal_readers", "before_subprocess")
        begin = time.monotonic() - started
        terminal = checks.run_commands(
            root,
            terminal_plan,
            log_dir=raw / "validation" / "terminal",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        spans.append(
            _span(
                started, "terminal_readers", begin, len(terminal), str(RAW / "validation/terminal")
            )
        )
        progress(
            started,
            "terminal_readers",
            "after_subprocess",
            passed=sum(r["passed"] for r in terminal),
        )
        final = build_artifact(
            auth,
            manifest,
            rows,
            [*worker_receipts, *receipts, *terminal],
            spans,
            time.monotonic() - started,
            run_date,
        )
        final["flagged_adversarial"] = not next(
            r["passed"] for r in terminal if r["name"] == "adversarial_verify"
        )
        atomic_json(candidate, final)
        shutil.copyfile(candidate, isolated_candidate)
        progress(started, "exact_terminal_replay", "before_subprocess")
        begin = time.monotonic() - started
        exact = checks.run_commands(
            root,
            terminal_plan,
            log_dir=raw / "validation" / "terminal_exact",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        atomic_json(raw / "exact_terminal_reader_outcomes.json", {"outcomes": exact})
        spans.append(
            _span(
                started,
                "exact_terminal_replay",
                begin,
                len(exact),
                str(RAW / "exact_terminal_reader_outcomes.json"),
            )
        )
        progress(
            started,
            "exact_terminal_replay",
            "after_subprocess",
            passed=sum(r["passed"] for r in exact),
        )
        if [(r["name"], r["passed"]) for r in exact] != [
            (r["name"], r["passed"]) for r in terminal
        ]:
            raise RuntimeError("terminal_reader_result_changed")
        final["terminal_reader_outcomes_path"] = str(RAW / "exact_terminal_reader_outcomes.json")
        final["phase_spans"] = spans
        final["duration_s"] = time.monotonic() - started
        final["flagged_adversarial"] = not next(
            r["passed"] for r in exact if r["name"] == "adversarial_verify"
        )
        progress(started, "publication", "before_atomic")
        atomic_json(output, final)
        progress(started, "publication", "after_atomic", verdict=final["honest_verdict"])
        return final


def cold_reader(candidate: Path, *, independent: bool = False) -> dict:  # pragma: no cover - E2E
    """Recompute exact rows in a fresh process with evaluator stores hidden."""

    artifact = json.loads(candidate.read_text())
    manifest_path = ROOT / artifact["feature_manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    result = cold_reconstruct(manifest, ROOT)
    reconstructed = [
        row for role in ROLES for row in read_jsonl(ROOT / manifest["roles"][role]["feature_path"])
    ]
    if artifact["rows"] != reconstructed or result["groups"] != 248:
        raise ValueError("terminal_rows_mismatch")
    if independent:
        counts = {
            role: sum(
                row["role"] == role and row["arm"] == "original_source" for row in reconstructed
            )
            for role in ROLES
        }
        if counts != COUNTS:
            raise ValueError("independent_role_count_mismatch")
        for role in ROLES:
            original = [
                row
                for row in reconstructed
                if row["role"] == role and row["arm"] == "original_source"
            ]
            actual = artifact["coverage_by_role"][role]
            if actual["unknown_groups"] != sum(row["censored"] for row in original):
                raise ValueError("independent_coverage_mismatch")
    return result


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - E2E
    """Select the producer or one isolated, read-only verification mode."""

    print("[exp7646] startup flushed", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260925")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--worker", nargs=5)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        input_name, output_name, role, mapping, denied = args.worker
        _worker(Path(input_name), Path(output_name), role, json.loads(mapping), Path(denied))
        return 0
    if args.cold_replay or args.independent_reduce:
        candidate = args.cold_replay or args.independent_reduce
        assert candidate is not None
        print(
            json.dumps(cold_reader(candidate, independent=args.independent_reduce is not None)),
            flush=True,
        )
        return 0
    output = args.output if args.output.is_absolute() else ROOT / args.output
    run_experiment(ROOT, args.date, output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())

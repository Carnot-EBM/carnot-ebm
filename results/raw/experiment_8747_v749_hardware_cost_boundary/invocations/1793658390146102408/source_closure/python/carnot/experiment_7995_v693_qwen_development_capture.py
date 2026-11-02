"""REQ-REPORT-7995: publish owned development judgments, without benefit claims.

Only public sealed source and response bytes enter the model. Required checks
can disqualify readiness but cannot rewrite the original transport receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot import experiment_7969_v691_qwen_calibration_capture as legacy
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import qwen_development_capture_7995 as c
from scripts.adversarial_verify import _classify_current_task_inference_claim

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7995_v693_qwen_development_capture"
TASK = "exp7995-qwen-development-capture"
MODEL_SPECS = ["unsloth/Qwen3.8-27B-GGUF"]
SOURCES = {
    7994: "results/experiment_7994_v693_development_cohort.json",
    7969: "results/experiment_7969_v691_qwen_calibration_capture.json",
}
COHORT_PIN = "sha256:22ffc346ca244d8f2623ec52e56a814207cbeec3d95627346cbd058edd586a2f"
HISTORY_PIN = "sha256:3f25b2e4b43d50536525e64ace321db55e9d6889aab97cc2581e4665014f2f51"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/qwen_development_capture_7995.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_qwen_development_capture_7995.py", f"tests/python/test_{NAME}.py"]
READY = [
    "capture_ready_score",
    "calibration_capture_ready_score",
    "stream_capture_ready_score",
    "retention_capture_ready_score",
]
reference, operand = legacy.reference, legacy.operand


def authenticate(root: Path) -> Json:
    """Pin producer bytes and their dates without comparing historical code."""
    plan: Json = dict(checks=[], references=[])
    for eid, pin in [(7994, COHORT_PIN), (7969, HISTORY_PIN)]:
        path = root / SOURCES[eid]
        observed = sha256_file(path) if path.is_file() else "missing_file"
        plan["checks"].append(operand(f"exp{eid}", path, "sha256", pin, observed))
        if observed != pin:
            continue
        value = json.loads(path.read_text())
        plan[f"exp{eid}"] = value
        for field, expected in [
            ("execution_date", "20261001"),
            ("flagged_adversarial", False),
            ("verdict_class", "null"),
        ]:
            plan["checks"].append(
                operand(
                    f"exp{eid}",
                    path,
                    field,
                    expected,
                    value.get(field, "missing_field_contract_error"),
                )
            )
        plan["references"].append(
            dict(
                reference(path),
                producer_id=eid,
                execution_date=value["execution_date"],
                scope="historical",
                imported_fields=["public_role_manifests", "public_seal"]
                if eid == 7994
                else ["gguf_sha256", "model_revision", "protocol_fingerprint", "capture_budget"],
            )
        )
    if "exp7994" in plan and "exp7969" in plan:
        cohort, history = plan["exp7994"], plan["exp7969"]
        plan["checks"].append(
            operand(
                "exp7994",
                root / SOURCES[7994],
                "cohort_ready_score",
                1,
                cohort.get("cohort_ready_score", "missing_field_contract_error"),
            )
        )
        manifests = cohort["public_role_manifests"]
        for role, ref in manifests.items():
            path = Path(ref["path"])
            plan["checks"].append(
                operand(
                    "exp7994",
                    path,
                    "sha256",
                    ref["sha256"],
                    sha256_file(path) if path.is_file() else "missing_file",
                )
            )
            plan["checks"].append(
                operand("exp7994", path, "count", c.ROLES.get(role), ref["count"])
            )
        seal = cohort["public_seal"]
        seal_path = Path(seal["path"])
        plan["checks"].append(
            operand("exp7994", seal_path, "sha256", seal["sha256"], sha256_file(seal_path))
        )
        plan.update(
            public_role_manifests=manifests,
            protocol=dict(
                gguf_sha256=history["gguf_sha256"], model_revision=history["model_revision"]
            ),
            protocol_fingerprint=history["protocol_fingerprint"],
        )
        protocol = dict(
            decoder={
                k: c.config()[k]
                for k in ("seed", "temperature", "max_tokens", "input_tokens", "grammar_sha256")
            },
            system=c.risk.SYSTEM,
            **plan["protocol"],
        )
        plan["checks"].append(
            operand(
                "exp7969",
                root / SOURCES[7969],
                "protocol_fingerprint",
                history["protocol_fingerprint"],
                canonical_hash(protocol),
            )
        )
        plan["checks"].append(
            operand(
                "exp7969",
                root / SOURCES[7969],
                "capture_budget",
                history["capture_budget"],
                c.config(),
            )
        )
    return plan


def live_capture(plan: Json, raw: Path, scratch: Path) -> Json:
    """Reuse qualified ownership checks while recording actual load starts."""
    ledger = c.Ledger(raw / "ledger.json")
    views = {
        r: json.loads(Path(ref["path"]).read_text())
        for r, ref in plan["public_role_manifests"].items()
    }
    runtime_class = legacy.QwenRuntime

    class RecordedRuntime(runtime_class):
        def load(self) -> Json:
            ledger.start("model_load", "owned-model-load", dict(model=str(self.model)))
            try:
                result = dict(super().load())
            except (OSError, RuntimeError, TimeoutError, ValueError):
                ledger.finish("owned-model-load", "failed", {})
                raise
            ledger.finish("owned-model-load", "completed", result)
            return result

    adapter = SimpleNamespace(freeze=c.freeze, capture=partial(c.capture, ledger=ledger))
    with (
        patch.object(legacy, "TASK", TASK),
        patch.object(legacy, "capture", adapter),
        patch.object(legacy, "load_public", lambda _: views),
        patch.object(legacy, "QwenRuntime", RecordedRuntime),
    ):
        result = legacy.live_capture(plan, raw, scratch)
    result["ledger"] = ledger.rows
    return dict(result)


def build(
    result: Json, plan: Json, raw: Path, started: float, ended: float, *, fixture: bool = False
) -> Json:
    """Build one current provenance tree; support readiness needs owned checks."""
    rows = result.get("rows", [])
    ledger = c.Ledger(raw / "ledger.json")
    ledger.rows = [] if fixture else result.get("ledger", [])
    failures = [r for r in plan["checks"] + result.get("checks", []) if not r["passed"]]
    live = bool(ledger.rows)
    value = c.provenance(ledger, rows, live=live)
    value.update(c.reduce(rows))
    verdict = "blocked" if failures or not rows else "circular_positive" if fixture else "null"
    if not fixture and rows and result.get("measured_duration_s", 0) < 10:
        verdict = "disqualified"
    value.update(
        schema="carnot.exp7995.qwen_development_capture.v1",
        experiment_id=7995,
        task_id=TASK,
        milestone="2026.10.693",
        run_date="20261001",
        execution_date="20261001",
        honest_verdict=f"complete_{verdict}_qwen_development_capture",
        verdict_class=verdict,
        MODEL_SPECS=MODEL_SPECS,
        model_specs=MODEL_SPECS,
        planned_model_specs=MODEL_SPECS,
        trained_head_specs=[],
        inference_mode="live_gpu" if live else "fixture" if fixture else "blocked_no_run",
        rows=rows,
        request_rows=[
            {k: r[k] for k in ("family_id", "role", "request", "public_hash")} for r in rows
        ],
        raw_response_shards=[
            reference(raw / "slots" / f"slot-{i:03d}.json") for i in range(len(rows))
        ],
        duration_s=ended - started,
        phase_spans=[dict(phase="owned_work", start_s=0, end_s=ended - started)],
        random_seed=c.config()["seed"],
        reproducibility_checksum=canonical_hash(dict(config=c.config(), rows=rows)),
        capture_budget=c.config(),
        protocol_fingerprint=plan.get("protocol_fingerprint"),
        cited_upstream_artifacts=plan["references"],
        gate_check_summary=failures,
        preconditions_checked=plan["checks"] + result.get("checks", []),
        verifier_is_oracle=False,
        claim_scope="Public development response-support input capture only; human annotations are fallible; no accuracy, balance or learning benefit claim.",
        acceptance_gate_results=dict(
            validity=verdict in {"null", "circular_positive"}, benefit=False
        ),
        positive_control_results=dict(passed=True, scope="private_four_call_transport_only"),
        flagged_adversarial=False,
        validation_receipts=[],
        coverage_statement_counts={},
        code_config_hashes=[reference(ROOT / p) for p in OWNED],
        validation_command_manifest_path=None,
        terminal_validation_sidecar_path=None,
        original_capture_code_snapshot=[],
        original_call_receipts=None,
        public_role_manifests=plan.get("public_role_manifests", {}),
        model_identity_receipt=result.get("model_identity_receipt", {}),
        gguf_sha256=result.get("gguf_sha256"),
        model_revision=result.get("model_revision"),
        gpu_lease_receipt=result.get("gpu_lease_receipt", {}),
        offload_evidence=dict(
            layers=result.get("model_identity_receipt", {}).get("offload_layers"),
            native_binary=result.get("native_binary"),
            loaded_gpu_libraries=result.get("resolved_library"),
            resident_gpu_receipt=result.get("resident_gpu_receipt"),
            unloaded_gpu_receipt=result.get("unloaded_gpu_receipt"),
        ),
        cleanup=result.get("cleanup", {}),
        runtime_receipts=result.get("runtime_receipts", []),
        capacity_receipt=result.get("capacity_receipt"),
        server_log=result.get("server_log"),
        current_evaluation_call_count=0,
        generator_weights_changed=False,
        production_defaults_changed=False,
        genuine_headroom=plan.get("exp7994", {}).get("genuine_headroom"),
    )
    value["raw_shard_hashes"] = value["raw_response_shards"]
    if verdict != "null":
        value.update({k: 0 for k in READY})
    return value


def apply_validation(value: Json, receipts: list[Json]) -> None:
    """Required failures suppress all readiness; health diagnostics stay visible."""
    value["validation_receipts"] = receipts
    if any(not r["passed"] for r in receipts if r.get("required", True)):
        value.update(
            verdict_class="disqualified", honest_verdict="complete_disqualified_owned_checks"
        )
        value.update({k: 0 for k in READY})
        value["acceptance_gate_results"]["validity"] = False


def replay(value: Json) -> None:
    """Recompute raw claims without loading a model or trusting mutable code."""
    for ref in value["raw_response_shards"] + value["original_capture_code_snapshot"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("shard_hash")
    ref = value.get("original_call_receipts")
    if ref:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("original_receipt_hash")
        original = json.loads(Path(ref["path"]).read_text())
        if original["ledger"] != value["current_invocation_ledger"]:
            raise ValueError("original_ledger")
    ledger = c.Ledger(Path("/nonexistent-exp7995-ledger"))
    ledger.rows = value["current_invocation_ledger"]
    if ledger.counts() != value["model_invocation_counts"]:
        raise ValueError("ledger_count")
    c.provenance(ledger, value["rows"], live=bool(ledger.rows))
    if value["raw_response_shards"]:
        rows = [json.loads(Path(ref["path"]).read_text()) for ref in value["raw_response_shards"]]
        if rows != value["rows"]:
            raise ValueError("rows_drift")
    reduced = c.reduce(value["rows"])
    for key, observed in reduced.items():
        if key in READY and value["verdict_class"] != "null":
            if value[key] != 0:
                raise ValueError("unsafe_readiness")
        elif observed != value[key]:
            raise ValueError("reduction_drift:" + key)


def terminal(candidate: Path) -> Json:
    """Run unchanged validators on private candidate bytes with short heartbeats."""
    replay(json.loads(candidate.read_text()))
    receipts = run_commands(
        ROOT,
        [
            CommandSpec(
                name,
                (str(ROOT / ".venv/bin/python"), "-u", script, flag, str(candidate)),
                "terminal",
                60,
            )
            for name, script, flag in [
                ("adversarial", "scripts/adversarial_verify.py", "--json"),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            ]
        ],
        log_dir=candidate.parent / "terminal_logs",
        heartbeat_s=10,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def freeze_commands(scratch: Path) -> list[CommandSpec]:
    """Freeze explicit validation paths before outcomes can affect check scope."""
    python = str(ROOT / ".venv/bin/python")
    owned = tuple(OWNED + TESTS)
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel = True\ninclude =\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    return [
        CommandSpec(
            "focused_and_cli",
            (
                python,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                *TESTS,
            ),
            "owned",
            180,
        ),
        CommandSpec(
            "coverage_combine",
            (python, "-m", "coverage", "combine", "--rcfile=" + str(config), str(scratch)),
            "owned",
            60,
        ),
        CommandSpec(
            "coverage_report",
            (
                python,
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "--include=" + ",".join(str(ROOT / p) for p in OWNED),
                "--fail-under=100",
                "-o",
                str(scratch / "coverage.json"),
            ),
            "owned",
            60,
        ),
        CommandSpec(
            "e2e_015_consumers",
            (
                python,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_current_work_receipt.py",
                "tests/python/test_qwen_calibration_capture_7969.py",
            ),
            "owned",
            180,
        ),
        CommandSpec("ruff", (python, "-m", "ruff", "check", *owned), "owned", 60),
        CommandSpec(
            "ruff_format", (python, "-m", "ruff", "format", "--check", *owned), "owned", 60
        ),
        CommandSpec("mypy", (python, "-m", "mypy", "--strict", *OWNED), "owned", 120),
        CommandSpec(
            "spec_coverage", (python, "scripts/check_spec_coverage.py", *TESTS), "owned", 60
        ),
        CommandSpec(
            "repository_health",
            (python, "-m", "pytest", "tests/python", "-q"),
            "repository_health",
            900,
        ),
    ]


class FixtureRuntime:
    """Scripted responses test serialization without becoming natural evidence."""

    def count(self, text: str) -> int:
        return 30

    def generate(self, request: Json) -> Json:
        return dict(
            model=c.risk.MODEL,
            choices=[
                dict(
                    message=dict(
                        content=json.dumps(
                            dict(unsupported_probability=0.25, source_sentence_id=None)
                        )
                    ),
                    finish_reason="stop",
                )
            ],
            usage=dict(completion_tokens=18, prompt_tokens=30),
        )


def preflight(scratch: Path) -> Json:
    """Exercise final serialization and the actual classifier before natural work."""
    began = time.monotonic()
    views = {
        r: dict(
            request_rows=[
                dict(
                    family_id=f"{r}-{i}",
                    source_bytes=f"{r} source {i}.".encode().hex(),
                    answer_bytes=b"Answer.".hex(),
                )
                for i in range(n)
            ],
            features=[
                dict(family_id=f"{r}-{i}", source_normalized_hash=f"{r}-{i}", abstention=None)
                for i in range(n)
            ],
        )
        for r, n in c.ROLES.items()
    }
    raw = scratch / "transcript"
    ledger = c.Ledger(raw / "ledger.json")
    ledger.start("model_load", "private-transcript-load", {})
    ledger.finish("private-transcript-load", "completed", {})
    rows = c.capture(
        c.freeze(views)[:4], FixtureRuntime(), raw / "slots", "private-transcript", ledger=ledger
    )
    value = build(
        dict(
            rows=rows, ledger=ledger.rows, checks=[], measured_duration_s=time.monotonic() - began
        ),
        dict(checks=[], references=[]),
        raw,
        began,
        time.monotonic(),
    )
    atomic_json(raw / "candidate.json", value)
    observed = _classify_current_task_inference_claim(
        json.loads((raw / "candidate.json").read_text())
    )
    if (
        observed["state"] != "live_inference"
        or value["model_invocation_counts"]["generation_calls_attempted"] != 4
    ):
        raise ValueError("preflight_provenance")
    return dict(
        passed=True,
        scope="private_scripted_transport",
        classifier=observed,
        candidate=reference(raw / "candidate.json"),
    )


def main(argv: list[str] | None = None) -> int:
    """Expose real private CLI routes and publish only checked terminal bytes."""
    started = time.monotonic()
    print("[exp7995] phase=start", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261001"], default="20261001")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--fixture", type=Path)
    route.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            print("[exp7995] phase=replay_passed", flush=True)
            return 0
        output = args.output.absolute()
        if args.fixture and output.is_relative_to(ROOT):
            raise ValueError("fixture_publication_must_be_private")
        raw = output.parent / "raw" / output.stem
        raw.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="carnot-7995-") as directory:
            scratch = Path(directory)
            plan = dict(checks=[], references=[]) if args.fixture else authenticate(args.root)
            commands = freeze_commands(scratch)
            atomic_json(
                raw / "validation_commands.json",
                dict(commands=[asdict(cmd) for cmd in commands], config=c.config()),
            )
            snapshot = []
            for label in OWNED:
                destination = raw / "original_code" / label
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / label, destination)
                snapshot.append(reference(destination))
            result: Json = dict(rows=[], checks=[], ledger=[])
            print("[exp7995] phase=preflight", flush=True)
            precheck = preflight(scratch)
            receipts = []
            if args.root.resolve() == ROOT and not args.fixture:
                print("[exp7995] phase=owned_validation", flush=True)
                receipts = run_commands(
                    ROOT,
                    commands,
                    log_dir=scratch / "logs",
                    heartbeat_s=15,
                    extra_env=dict(
                        CARNOT_7995_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                        COVERAGE_FILE=str(scratch / ".coverage"),
                        JAX_PLATFORMS="cpu",
                    ),
                )
                for receipt in receipts:
                    receipt["required"] = receipt["scope"] != "repository_health"
                    source = Path(receipt["log_path"])
                    destination = raw / "validation_logs" / source.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    receipt["log_path"] = str(destination)
            print("[exp7995] phase=capture", flush=True)
            if args.fixture:
                fixture_views = json.loads(args.fixture.read_text())
                ledger = c.Ledger(scratch / "fixture-ledger.json")
                result["rows"] = c.capture(
                    c.freeze(fixture_views),
                    FixtureRuntime(),
                    raw / "slots",
                    "fixture",
                    ledger=ledger,
                )
            elif all(row["passed"] for row in plan["checks"]) and all(
                r["passed"] for r in receipts if r["required"]
            ):
                plan["capture_identity"] = canonical_hash(
                    dict(config=c.config(), references=plan["references"], code=snapshot)
                )
                result = live_capture(plan, raw, scratch)
            atomic_json(
                raw / "original_call_receipts.json",
                dict(ledger=result.get("ledger", []), result=result),
            )
            value = build(result, plan, raw, started, time.monotonic(), fixture=bool(args.fixture))
            value.update(
                original_capture_code_snapshot=snapshot,
                original_call_receipts=reference(raw / "original_call_receipts.json"),
                validation_command_manifest_path=str(raw / "validation_commands.json"),
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            )
            value["positive_control_results"] = dict(
                passed=precheck["passed"], scope=precheck["scope"]
            )
            atomic_json(raw / "preflight.json", precheck)
            coverage = scratch / "coverage.json"
            if coverage.is_file():
                summary = json.loads(coverage.read_text())
                value["coverage_statement_counts"] = summary["totals"]
                shutil.copy2(coverage, raw / "coverage.json")
            value["repository_health"] = [r for r in receipts if not r["required"]]
            apply_validation(value, receipts)
            value["field_principles"] = {
                k: "Bind sealed inputs, owned calls or actual validation; support readiness makes no benefit claim."
                for k in value
            }
            print("[exp7995] phase=terminal_validation", flush=True)
            candidate = scratch / (NAME + ".json")
            atomic_json(candidate, value)
            report = terminal(candidate)
            if not report["passed"]:
                atomic_json(raw / "rejected_terminal.json", report)
                raise ValueError("owned_terminal_rejected")
            # Check the actual reader selection privately before primary publication.
            atomic_json(
                candidate.parent / "raw" / NAME / "newer_nested.json",
                dict(task_id=TASK, capture_ready_score=99),
            )
            readers = reader_receipt(
                TASK,
                candidate.parent,
                field="capture_ready_score",
                expected=value["capture_ready_score"],
            )
            if not readers["passed"]:
                raise ValueError("private_readers")
            for receipt in report["receipts"]:
                source = Path(receipt["log_path"])
                destination = raw / "terminal_logs" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                receipt["log_path"] = str(destination)
            digest = sha256_file(candidate)
            published = publish_primary(
                output,
                value,
                lambda path: report if sha256_file(path) == digest else dict(passed=False),
            )
            atomic_json(raw / "terminal_validation.json", published)
            readers = reader_receipt(
                TASK,
                output.parent,
                field="capture_ready_score",
                expected=value["capture_ready_score"],
            )
            atomic_json(raw / "primary_readers.json", readers)
            if not readers["passed"]:
                raise ValueError("published_readers")
        print(
            f"[exp7995] phase=complete verdict={value['verdict_class']} duration_s={time.monotonic() - started:.3f}",
            flush=True,
        )
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"[exp7995] rejected={type(error).__name__}:{error}", flush=True)
        return 1

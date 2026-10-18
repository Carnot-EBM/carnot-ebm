"""Publish a bounded complete-response measurement with independent human targets.

REQ-REPORT-7958. Public admission and raw replies are sealed before evaluator
access. Existing source exposure keeps this experiment in development scope.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_7932_v688_qwen_completion as completion
from carnot import experiment_7955_v690_response_targets as targets
from carnot.inference.gguf_metadata import read_gguf_metadata
from carnot.inference.qwen_sufficiency_7920 import QwenRuntime, bounded
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import qwen_response_risk_7958 as risk

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7958_v690_qwen_response_risk"
TASK = "exp7958-qwen-response-risk"
MODEL_SPECS = [risk.MODEL]
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/qwen_response_risk_7958.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = [
    "tests/python/test_qwen_response_risk_7958.py",
    f"tests/python/test_{NAME}.py",
    "tests/python/test_source_boundary_7852.py",
    "tests/python/test_experiment_7942_v689_sentence_labels.py",
    "tests/python/test_primary_publication_7928.py",
]
INCLUDE = ",".join(str(ROOT / p) for p in OWNED)
UPSTREAM = "results/experiment_7955_v690_response_targets.json"
PIN = "sha256:dbfac2d991450601add8506a00452c9b6d39a594ad91d3c291f742cfe378c62a"
MODEL_PIN = "sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
IMPORTS = [
    *targets.IMPORTS,
    "carnot.experiment_7955_v690_response_targets",
    "carnot.experiment_7932_v688_qwen_completion",
    "carnot.verify.qwen_completion_7932",
    "carnot.verify.qwen_response_risk_7958",
    "carnot.inference.qwen_sufficiency_7920",
    "carnot.inference.llama_cpp_process",
    "carnot.inference.gguf_metadata",
    "carnot.inference.sota_models",
    "carnot.gpu_lease_phase_journal",
]
reference, operand = targets.reference, targets.operand


def progress(phase: str, started: float, units: int = 0) -> None:
    """Keep phase boundaries visible without manufacturing elapsed work."""
    print(
        f"[exp7958] phase={phase} completed_units={units} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Read the declared primary and public manifest, never evaluator labels."""
    path = root / UPSTREAM
    checks = [
        operand(targets.TASK, path, "sha256", PIN, sha256_file(path) if path.is_file() else None)
    ]
    upstream: Json = {}
    if checks[0]["passed"]:
        value = json.loads(path.read_text())
        for field, expected in (
            ("response_targets_ready_score", 1),
            ("flagged_adversarial", False),
        ):
            checks.append(operand(targets.TASK, path, field, expected, value.get(field)))
        verdict = operand(
            targets.TASK,
            path,
            "verdict_class",
            ["positive", "circular_positive", "null"],
            value.get("verdict_class"),
        )
        verdict.update(op="in", passed=value.get("verdict_class") in verdict["expected"])
        checks.append(verdict)
        for key in ("public_manifest",):
            manifest = Path(value[key + "_path"])
            checks.append(
                operand(
                    targets.TASK,
                    manifest,
                    "sha256",
                    value[key + "_sha256"],
                    sha256_file(manifest) if manifest.is_file() else None,
                )
            )
        if all(c["passed"] for c in checks):
            role_ref = next(
                r
                for r in value["source_artifact_hashes"]
                if r["path"].endswith("experiment_7892_v685_source_boundary.json")
            )
            role_path = targets.prior.checked_reference(role_ref)
            roles = json.loads(role_path.read_text())["rows"]
            ids = {r["family_id"] for r in roles if r["role"] == "evaluation"}
            public = [
                r
                for r in json.loads(manifest.read_text())["predictor_inputs"]
                if r["family_id"] in ids
            ]
            checks.append(
                operand(targets.TASK, manifest, "intended_evaluation_slots", 64, len(public))
            )
            upstream = dict(
                artifact=value,
                public=public,
                checks=checks,
                references=[reference(path), reference(manifest), role_ref],
            )
    else:
        checks.extend(
            operand(targets.TASK, path, field, expected, None)
            for field, expected in (
                ("response_targets_ready_score", 1),
                ("flagged_adversarial", False),
                ("verdict_class", ["positive", "circular_positive", "null"]),
            )
        )
        checks[-1]["op"] = "in"
    return [r for r in checks if not r["passed"]], upstream


def human_targets(upstream: Json) -> list[Json]:
    """Rebuild the original annotation union only after raw replies are sealed."""
    targets.reconstruct(upstream["artifact"])
    annotations = upstream["artifact"]["annotation_rows"]
    return [
        dict(
            r,
            annotation_types=sorted(
                {a["label_type"] for a in annotations if a["family_id"] == r["family_id"]}
            ),
        )
        for r in upstream["artifact"]["response_union_rows"]
        if r["role"] == "evaluation"
    ]


def base(failures: list[Json]) -> Json:
    """Keep all required fields explicit even when no model can be invoked."""
    value = dict(
        schema="carnot.exp7958.qwen_response_risk.v1",
        experiment_id=7958,
        task_id=TASK,
        milestone="2026.09.690",
        run_date="20261001",
        honest_verdict="complete_blocked_external_prerequisite",
        verdict_class="blocked",
        flagged_adversarial=False,
        gate_check_summary=failures,
        preconditions_checked=failures,
        qwen_response_measurement_ready_score=0,
        qwen_response_benefit_score=0,
        verifier_is_oracle=False,
        claim_scope="exposed_development",
        inference_substrate="blocked_no_run",
        inference_substrate_class="blocked_no_run",
        planned_inference_substrate_class="model_bounded_generation",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        target_model=None,
        trained_head_specs=[],
        planned_model_specs=MODEL_SPECS,
        model_invocation_counts=dict(
            model_loads_attempted=0,
            model_loads_completed=0,
            model_loads_failed=0,
            generation_calls_attempted=0,
            generation_calls_completed=0,
            generation_calls_failed=0,
        ),
        duration_s=0.0,
        phase_spans=[],
        random_seed=risk.SEED,
        reproducibility_checksum=None,
        source_artifact_hashes=[],
        resolved_imports={},
        validation_receipts=[],
        validation_command_manifest_path=None,
        observed_child_commands=[],
        coverage_statement_counts={},
        historical_required_failures=[],
        repository_health={},
        primary_resolution_receipt=None,
        terminal_validation_sidecar_path=None,
        response_target_definition="any authenticated human unsupported span anywhere in the original complete response, including implicit_true and due_to_null",
        human_label_lineage={},
        raw_response_shards=[],
        request_rows=[],
        decoder_order=[],
        token_budget=risk.config(),
        gguf_sha256=None,
        model_revision=None,
        quantization=None,
        model_identity_receipt={},
        offload_evidence={},
        grammar_sha256=risk.config()["grammar_sha256"],
        gpu_lease_receipt={},
        cited_upstream_artifacts=[],
        acceptance_gate_results=dict(
            validity=False,
            readiness=False,
            probability_quality=False,
            calibration=False,
            decision_benefit=False,
            retention="not_measured",
            efficiency="descriptive_only",
        ),
        oracle_distinct_corrigendum="September 28 preserved: human support labels are model-independent, not formal truth certificates; GAP-ORACLE-DISTINCT remains open.",
        production_defaults_changed=False,
        generator_weights_changed=False,
    )
    value.update(risk.reduce([], []))
    return value


def live_capture(public: list[Json], raw: Path) -> tuple[Json, list[Json], list[Json], list[Json]]:
    """Own a CUDA worker and measure actual bounded calls with fixed weights."""
    from carnot.gpu_lease_phase_journal import GpuLease
    from carnot.inference.sota_models import cached_current_model

    started = time.monotonic()
    progress("model_preconditions", started)
    identity: Json = dict(model_loads_attempted=0, model_loads_completed=0)
    rows: list[Json] = []
    frozen: list[Json] = []
    spec = cached_current_model() or {}
    path = Path(spec.get("model_path", raw / "missing-model"))
    checks = [
        operand("qwen_cache", path, "hf_id", risk.MODEL, spec.get("hf_id")),
        operand("qwen_cache", path, "exists", True, path.is_file()),
    ]
    if not all(c["passed"] for c in checks):
        return identity, rows, checks, frozen
    progress("before_model_hash", started)
    try:
        digest = bounded(lambda: sha256_file(path), 120)
        metadata = read_gguf_metadata(path)
    except (OSError, ValueError, TimeoutError) as error:
        checks.append(operand("qwen_cache", path, "readable_authenticated_GGUF", True, str(error)))
        return identity, rows, checks, frozen
    checks += [
        operand("qwen_cache", path, "sha256", MODEL_PIN, digest),
        operand("qwen_cache", path, "quantization", "Q4_K_M", metadata["quantization"]),
    ]
    identity.update(
        model_path=str(path),
        gguf_sha256=digest,
        revision=path.parent.name,
        quantization=metadata["quantization"],
        gguf_metadata=metadata,
        backend="llama.cpp native CUDA",
        tokenizer="embedded_GGUF",
    )
    progress("after_model_hash", started)
    if not all(c["passed"] for c in checks):
        return identity, rows, checks, frozen
    capacity = run_commands(
        ROOT,
        [
            CommandSpec(
                "gpu_capacity",
                (
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ),
                "precondition",
                10,
            )
        ],
        log_dir=raw / "gpu",
        heartbeat_s=5,
    )[0]
    candidates = [
        line.split(",")
        for line in capacity["output_tail"].splitlines()
        if len(line.split(",")) == 4
        and int(line.split(",")[2]) < 1000
        and int(line.split(",")[3]) >= 20000
    ]
    checks.append(
        operand(
            "gpu_capacity",
            ROOT / capacity["log_path"],
            "idle_cuda_capacity",
            True,
            capacity["passed"] and bool(candidates),
        )
    )
    identity["capacity_receipt"] = capacity
    if not candidates:
        return identity, rows, checks, frozen
    gpu, uuid, used, _ = candidates[0]
    lease: Any = None
    runtime: Any = None
    work_start = time.monotonic()
    try:
        lease = GpuLease.acquire(
            runtime_dir="/tmp/carnot-gpu-leases",
            task_id=TASK,
            device_uuid=uuid.strip(),
            expected_model=str(path),
            vram_before_mb=int(used),
            ttl_s=3600,
        )
        identity["gpu_lease"] = lease.owner_receipt()
        lease.transition("admitted")
        lease.transition("loading")
        scratch = raw / "owned-model"
        scratch.mkdir(parents=True, exist_ok=True)
        runtime = QwenRuntime(path, scratch, int(gpu))
        identity["model_loads_attempted"] = 1
        progress("before_model_load", started)
        identity.update(runtime.load())
        identity["model_loads_completed"] = 1
        identity["device"] = dict(
            index=int(gpu), uuid=uuid.strip(), pid=runtime.worker.receipt.get("pid")
        )
        identity["native_binary"] = reference(Path(runtime.command[0]))
        progress("after_model_load", started)
        resident, receipt = completion.prior.gpu_memory(int(gpu), scratch, raw)
        identity["resident_gpu_receipt"] = receipt
        lease.transition("resident", vram_mb=resident)
        lease.transition("inferencing")
        progress("before_public_token_admission", started)
        frozen = risk.freeze(public, runtime.count)
        atomic_json(raw / "frozen_requests.json", dict(rows=frozen, config=risk.config()))
        progress("after_public_token_admission", started, len(frozen))
        rows = risk.capture(frozen, runtime, raw / "pairs", started=work_start)
        identity["request_receipts"] = runtime.receipts
    except (RuntimeError, OSError, TimeoutError, ValueError) as error:
        checks.append(
            operand(
                "owned_runtime",
                path,
                "authenticated_cuda_execution",
                True,
                f"{type(error).__name__}:{error}",
            )
        )
    finally:
        progress("before_model_unload", started)
        identity["measured_duration_s"] = time.monotonic() - work_start
        identity["cleanup"] = runtime.close() if runtime else dict(leak_free=True)
        checks.append(
            operand("owned_cleanup", path, "leak_free", True, identity["cleanup"]["leak_free"])
        )
        if lease:
            if lease.document["phase"] in ("resident", "inferencing"):
                lease.transition("unloading")
                after, receipt = completion.prior.gpu_memory(int(gpu), scratch, raw)
                identity["unloaded_gpu_receipt"] = receipt
                lease.transition(
                    "validating",
                    vram_mb=after,
                    exit_code=0,
                    unload_observed=identity["cleanup"]["leak_free"],
                )
            lease.transition(
                "terminal_complete" if identity.get("authenticated") else "terminal_blocked"
            )
            identity["gpu_lease"]["release"] = lease.release()
        if runtime and runtime.log.is_file():
            identity["server_log"] = reference(runtime.log)
        progress("after_model_unload", started)
    return identity, rows, checks, frozen


def freeze_commands(raw: Path) -> Json:
    """Freeze explicit checks and private routes before model outcomes exist."""
    plan = targets.freeze_commands(raw)
    for c in plan["commands"]:
        c["argv"] = [
            a.replace(targets.NAME, NAME)
            .replace("response_targets_7955", "qwen_response_risk_7958")
            .replace(targets.INCLUDE, INCLUDE)
            for a in c["argv"]
        ]
        if c["name"] == "unit_consumer_e2e015":
            c["argv"] = (
                c["argv"][: c["argv"].index("tests/python/test_qwen_response_risk_7958.py")]
                + TESTS
                + ["-q"]
            )
        if c["name"] in ("ruff_check", "ruff_format", "strict_mypy"):
            c["argv"] = c["argv"][: 3 if c["name"] in ("ruff_format", "strict_mypy") else 2] + OWNED
            if c["name"] != "strict_mypy":
                c["argv"] += TESTS[:2]
            else:
                c["argv"] = [
                    str(ROOT / ".venv/bin/mypy"),
                    "--strict",
                    "--follow-imports=skip",
                    *OWNED,
                ]
        if c["name"] == "spec_coverage":
            c["argv"] = c["argv"][:2] + TESTS
    py = str(ROOT / ".venv/bin/python")
    cov = str(ROOT / ".venv/bin/coverage")
    checkpoint = str(raw / "capture_checkpoint.json")
    cold = dict(
        name="cold_recorded_replies",
        argv=[
            cov,
            "run",
            "--parallel-mode",
            "--data-file=" + plan["coverage_file"],
            "--include=" + INCLUDE,
            str(ROOT / OWNED[2]),
            "--date",
            "20261001",
            "--cold-replay",
            checkpoint,
            "--output",
            str(raw / "private-terminal-recheck" / "replay.json"),
        ],
        expected_exit=0,
        failure_reason=None,
        required=True,
        deadline_s=180,
    )
    plan["commands"].insert(
        next(i for i, c in enumerate(plan["commands"]) if c["name"] == "coverage_combine"), cold
    )
    # A real small-model CPU call has its own receipt and never supplies headline rows.
    plan["commands"].append(
        dict(
            name="cpu_grammar_transport",
            argv=[
                py,
                "-u",
                "-c",
                "from pathlib import Path; from carnot.experiment_7932_v688_qwen_completion import cpu_grammar_fixture; "
                f"r=cpu_grammar_fixture(Path({str(raw / 'cpu-worker')!r}), Path({str(raw / 'cpu-transport')!r})); print(r, flush=True); assert r['passed']",
            ],
            expected_exit=0,
            failure_reason=None,
            required=True,
            deadline_s=180,
        )
    )
    plan.update(
        affected_files=OWNED,
        explicit_tests=TESTS,
        transitive_consumers=IMPORTS,
        coverage_includes=INCLUDE,
    )
    plan["terminal_commands"] = [
        {**c, "argv": [a.replace(targets.NAME, NAME) for a in c["argv"]]}
        for c in plan["terminal_commands"]
    ]
    atomic_json(raw / "validation_command_manifest.json", plan)
    return plan


def execute_commands(manifest: Json, raw: Path) -> list[Json]:
    """Reuse the bounded subprocess supervisor and retain expected failures."""
    return targets.prior.execute_commands(manifest, raw / "validation_logs")


def apply_validation(value: Json, receipts: list[Json]) -> None:
    """Current owned failures disqualify; historical health stays historical."""
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = [r["command_argv"] for r in receipts]
    value["repository_health"]["current"] = [r for r in receipts if not r.get("required", True)]
    if any(not r["passed"] for r in receipts if r.get("required", True)):
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_required_validation",
            qwen_response_measurement_ready_score=0,
            qwen_response_benefit_score=0,
        )
        value["acceptance_gate_results"].update(
            validity=False, readiness=False, decision_benefit=False
        )


def build(
    rows: list[Json],
    labels: list[Json],
    raw: Path,
    identity: Json,
    upstream: Json,
    *,
    fixture_path: Path | None = None,
) -> Json:
    """Keep current invocation counts separate from cited and scripted evidence."""
    value = base([])
    value.update(risk.reduce(rows, labels))
    atomic_json(raw / "sealed_rows.json", dict(rows=rows))
    value["raw_response_shards"] = [reference(raw / "sealed_rows.json")]
    value["model_identity_receipt"] = identity
    loads = identity.get("model_loads_attempted", 0)
    completed = identity.get("model_loads_completed", 0)
    value["model_invocation_counts"] = dict(
        model_loads_attempted=loads,
        model_loads_completed=completed,
        model_loads_failed=loads - completed,
        generation_calls_attempted=sum(r["started"] for r in rows) if loads else 0,
        generation_calls_completed=sum(bool(r["raw_response"]) for r in rows) if loads else 0,
        generation_calls_failed=sum(r["started"] and not r["raw_response"] for r in rows)
        if loads
        else 0,
    )
    value.update(
        honest_verdict="complete_null_qwen_response_risk",
        verdict_class="null",
        inference_substrate="live_llm_inference"
        if loads
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="model_bounded_generation" if loads else "no_model_load",
        MODEL_SPECS=MODEL_SPECS if loads else [],
        model_specs=MODEL_SPECS if loads else [],
        target_model=risk.MODEL if loads else None,
        gguf_sha256=identity.get("gguf_sha256"),
        model_revision=identity.get("revision"),
        quantization=identity.get("quantization"),
        gpu_lease_receipt=identity.get("gpu_lease", {}),
        offload_evidence=dict(
            layers=identity.get("offload_layers"),
            device=identity.get("device"),
            resident=identity.get("resident_gpu_receipt"),
        ),
        request_rows=[
            dict(
                family_id=r["family_id"],
                arm=r["arm"],
                seed=r["seed"],
                request_sha256=canonical_hash(r["request"]),
                input_tokens=r["input_tokens"],
                usage=r["parsed"]["usage"],
            )
            for r in rows
        ],
        decoder_order=[
            dict(family_id=r["family_id"], arm=r["arm"], position=r["order"]) for r in rows
        ],
    )
    value["acceptance_gate_results"].update(
        validity=True,
        probability_quality=value["comparison_status"] == "registered_comparison",
        calibration=value["comparison_status"] == "registered_comparison",
        decision_benefit=bool(value["qwen_response_benefit_score"]),
    )
    if value["qwen_response_benefit_score"]:
        value.update(
            honest_verdict="complete_positive_development_response_risk", verdict_class="positive"
        )
    if fixture_path:
        value.update(
            fixture_input=reference(fixture_path),
            verdict_class="circular_positive",
            honest_verdict="complete_circular_positive_fixture_transport",
            qwen_response_benefit_score=0,
        )
        value["acceptance_gate_results"]["decision_benefit"] = False
    else:
        value["source_artifact_hashes"] = upstream["references"]
        value["human_label_lineage"] = dict(
            primary=upstream["references"][0],
            original_annotations=upstream["artifact"]["original_input_manifest"],
            reconstructed=True,
            model_independent=True,
            target_definition=upstream["artifact"]["target_definition"],
        )
        value["cited_upstream_artifacts"] = [
            dict(
                experiment_id=7955,
                fields_imported=[
                    "public_manifest_path",
                    "target_definition",
                    "response_union_rows",
                ],
                **upstream["references"][0],
            )
        ]
        value["historical_required_failures"] = upstream["artifact"]["historical_required_failures"]
        value["repository_health"]["historical"] = upstream["artifact"]["repository_health"]
    return value


def replay(value: Json) -> Json:
    """Reconstruct claims from sealed replies and original annotation custody."""
    for item in value.get("code_config_hashes", []) + value["source_artifact_hashes"]:
        targets.prior.checked_reference(item)
    if not value["raw_response_shards"]:
        if value["qwen_response_measurement_ready_score"]:
            raise ValueError("unsafe_readiness")
        return risk.reduce([], [])
    rows = [
        r
        for item in value["raw_response_shards"]
        for r in json.loads(targets.prior.checked_reference(item).read_text())["rows"]
    ]
    if value.get("fixture_input"):
        data = json.loads(targets.prior.checked_reference(value["fixture_input"]).read_text())
        labels = data["labels"]
        public = data["public"]
    else:
        failures, upstream = authenticate(Path(value.get("custody_root", ROOT)))
        if failures:
            raise ValueError("cold_custody")
        labels = human_targets(upstream)
        public = upstream["public"]
    counts = {canonical_hash(r["request"]["messages"]): r["input_tokens"] for r in rows}
    frozen = risk.freeze(public, lambda text: counts[canonical_hash(json.loads(text))])
    requests = {(r["family_id"], arm): r["requests"][arm] for r in frozen for arm in risk.ARMS}
    for row in rows:
        if row["request"] != requests.get((row["family_id"], row["arm"])):
            raise ValueError("request_drift")
    if value.get("frozen_manifest"):
        manifest = json.loads(targets.prior.checked_reference(value["frozen_manifest"]).read_text())
        if manifest != dict(rows=frozen, config=risk.config()):
            raise ValueError("request_manifest_drift")
    reduced = risk.reduce(rows, labels)
    for key in reduced:
        if key == "qwen_response_benefit_score" and value["verdict_class"] in (
            "disqualified",
            "circular_positive",
        ):
            continue
        if reduced[key] != value[key]:
            raise ValueError("reduction_drift:" + key)
    if value["qwen_response_measurement_ready_score"]:
        targets.check_receipts(value)
        if value["flagged_adversarial"] or value["verdict_class"] in ("blocked", "disqualified"):
            raise ValueError("unsafe_readiness")
    return reduced


def terminal_check(candidate: Path) -> Json:
    """Cold-reduce final bytes before running both mandatory artifact validators."""
    replay(json.loads(candidate.read_text()))
    commands = [
        CommandSpec(c["name"], tuple(c["argv"]), "terminal_candidate", c["deadline_s"])
        for c in json.loads((candidate.parent / "validation_command_manifest.json").read_text())[
            "terminal_commands"
        ]
    ]
    receipts = run_commands(
        ROOT, commands, log_dir=candidate.parent / "terminal_logs", heartbeat_s=10
    )
    return dict(
        passed=all(r["passed"] for r in receipts),
        receipts=receipts,
        flagged_adversarial=any(
            r["name"] == "adversarial" and r["exit_code"] != 0 for r in receipts
        ),
    )


def publish(output: Path, value: Json) -> None:
    """Expose only checked bytes and prove both live readers select the primary."""
    raw = output.parent / "raw" / NAME
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["primary_resolution_receipt"] = dict(path=str(raw / "primary_resolution.json"))
    value["field_principles"] = {
        key: "Bind current producer custody, bounded work, or independently reduced evidence; success alone is not benefit."
        for key in value
    }
    published = publish_primary(output, value, terminal_check)
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            primary_path=str(output),
            primary_sha256=published["primary_sha256"],
            validator=published["sidecar_path"],
        ),
    )
    receipt = reader_receipt(
        TASK,
        output.parent,
        field="qwen_response_measurement_ready_score",
        expected=value["qwen_response_measurement_ready_score"],
    )
    if not receipt["passed"]:
        raise ValueError("primary_resolution")
    atomic_json(raw / "primary_resolution.json", receipt)


class FixtureRuntime:
    """Supply explicit scripted transport without creating model evidence."""

    def generate(self, request: Json) -> Json:
        return dict(
            model=risk.MODEL,
            choices=[
                dict(
                    finish_reason="stop",
                    message=dict(
                        content='{"unsupported_probability":0.5,"source_sentence_id":null}'
                    ),
                )
            ],
            usage=dict(prompt_tokens=100, completion_tokens=20),
        )


def run(
    root: Path, output: Path, *, fixture_path: Path | None = None, validate: bool = True
) -> int:
    """Freeze custody, capture once, validate and atomically publish a terminal result."""
    started = time.monotonic()
    progress("start", started)
    raw = output.parent / "raw" / NAME
    manifest = freeze_commands(raw)
    identity: Json = {}
    upstream: Json = {}
    checks: list[Json] = []
    if fixture_path:
        data = json.loads(fixture_path.read_text())
        rows = risk.capture(risk.freeze(data["public"], len), FixtureRuntime(), raw / "pairs")
        value = build(rows, data["labels"], raw, identity, upstream, fixture_path=fixture_path)
    else:
        try:
            failures, upstream = authenticate(root)
        except (ValueError, OSError, KeyError) as error:
            failures = [
                operand(targets.TASK, root / UPSTREAM, "custody_readable", True, str(error))
            ]
        progress("authentication_complete", started)
        value = base(failures)
        if not failures:
            identity, rows, checks, frozen = live_capture(upstream["public"], raw)
            value["preconditions_checked"] = upstream["checks"] + checks
            failed = [c for c in checks if not c["passed"]]
            if not rows:
                value = base(failed)
                value["model_identity_receipt"] = identity
                if identity.get("model_loads_attempted"):
                    value.update(
                        verdict_class="disqualified",
                        honest_verdict="complete_disqualified_owned_model_load",
                        inference_substrate="live_llm_inference",
                        inference_substrate_class="model_bounded_generation",
                        model_specs=MODEL_SPECS,
                        MODEL_SPECS=MODEL_SPECS,
                        target_model=risk.MODEL,
                    )
                    value["model_invocation_counts"].update(
                        model_loads_attempted=identity["model_loads_attempted"],
                        model_loads_completed=identity.get("model_loads_completed", 0),
                        model_loads_failed=identity["model_loads_attempted"]
                        - identity.get("model_loads_completed", 0),
                    )
            else:
                atomic_json(raw / "sealed_rows.json", dict(rows=rows))
                progress("raw_replies_sealed_before_labels", started, len(rows))
                label_rows = human_targets(upstream)
                value = build(rows, label_rows, raw, identity, upstream)
                value["gate_check_summary"] = upstream["checks"] + checks
                if failed:
                    apply_validation(
                        value,
                        [dict(name="owned_runtime", passed=False, required=True, command_argv=[])],
                    )
                elif identity.get("authenticated") and identity.get("measured_duration_s", 0) >= 10:
                    value["qwen_response_measurement_ready_score"] = 1
                    value["acceptance_gate_results"]["readiness"] = True
                value["frozen_manifest"] = (
                    reference(raw / "frozen_requests.json") if frozen else None
                )
            value["preconditions_checked"] = upstream["checks"] + checks
            value["source_artifact_hashes"] = upstream["references"]
        value["custody_root"] = str(root)
    value["resolved_imports"] = {
        name: str(Path(importlib.import_module(name).__file__).resolve()) for name in IMPORTS
    }
    value["code_config_hashes"] = [reference(ROOT / p) for p in OWNED] + [
        reference(Path(p)) for p in set(value["resolved_imports"].values())
    ]
    value["prior_mechanism_change"] = (
        "Complete response union replaces the three-positive first-sentence target; grammar and generator weights stay fixed."
    )
    value["retire_if_same_verdict"] = (
        "Retire unchanged prior response-risk and sensitivity verdicts; no additional calls or tuning follow these labels."
    )
    value["validation_command_manifest_path"] = str(raw / "validation_command_manifest.json")
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            code=value["code_config_hashes"],
            config=risk.config(),
            inputs=value["source_artifact_hashes"],
            seed=risk.SEED,
        )
    )
    value["phase_spans"] = [
        dict(phase="capture_and_reduction", duration_s=time.monotonic() - started)
    ]
    progress("reduction_complete", started, len(value["rows"]))
    value["duration_s"] = time.monotonic() - started
    atomic_json(
        raw / "capture_checkpoint.json", {**value, "qwen_response_measurement_ready_score": 0}
    )
    if validate and not fixture_path and value["rows"]:
        # Private transport data is generated from declared fixtures, never model labels.
        fixture = dict(
            public=[
                dict(
                    family_id=str(i),
                    source_bytes=f"Source {i}.".encode().hex(),
                    answer_bytes=b"Complete answer.".hex(),
                )
                for i in range(64)
            ],
            labels=[
                dict(
                    family_id=str(i),
                    source_cluster_id=str(i),
                    y=i % 2,
                    implicit_true_excluded_y=i % 2,
                    annotation_count=i % 2,
                )
                for i in range(64)
            ],
        )
        atomic_json(raw / "validation_input.json", fixture)
        before = time.monotonic()
        receipts = execute_commands(manifest, raw)
        apply_validation(value, receipts)
        coverage_path = raw / "coverage.json"
        if coverage_path.is_file():
            report = json.loads(coverage_path.read_text())
            value["coverage_statement_counts"] = report["files"]
        value["phase_spans"].append(
            dict(phase="required_validation", duration_s=time.monotonic() - before)
        )
    value["duration_s"] = time.monotonic() - started
    progress("before_terminal_publication", started)
    publish(output, value)
    progress("terminal_complete", started, len(value["rows"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Provide separate private fixture, negative and cold-replay routes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261001"], default="20261001")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        try:
            reduced = replay(json.loads(args.cold_replay.read_text()))
            atomic_json(args.output, dict(replay_passed=True, rows_sha256=canonical_hash(reduced)))
            print("[exp7958] replay_passed", flush=True)
            return 0
        except (ValueError, OSError, KeyError) as error:
            print(f"[exp7958] replay_failed:{error}", flush=True)
            return 1
    return run(args.root, args.output, fixture_path=args.fixture_input)

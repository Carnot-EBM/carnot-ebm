"""Run a paired transport comparison on exposed development source families.

REQ-REPORT-7932-V688. Fixed pretrained weights and typed grammar measure
completion mechanics. They do not supply independent factual verification.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

from carnot import experiment_7920_v687_qwen_sufficiency as prior
from carnot.inference.gguf_metadata import read_gguf_metadata
from carnot.inference.llama_cpp_process import OwnedLlamaCppProcess
from carnot.inference.qwen_sufficiency_7920 import QwenRuntime, bounded
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import qwen_completion_7932 as protocol
from carnot.verify import source_interventions as source

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7932_v688_qwen_completion"
TASK = "exp7932-qwen-completion"
MODULE = f"python/carnot/{NAME}.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = f"tests/python/test_{NAME}.py"
VERIFY = "python/carnot/verify/qwen_completion_7932.py"
INCLUDE = f"*/{NAME}.py,*/qwen_completion_7932.py"
MODEL_SPECS = [source.MODEL_ID]


def progress(phase: str, units: int = 0) -> None:
    """Flush boundaries so slow owned work remains visible to supervisors."""
    print(
        f"[exp7932] phase={phase} completed_units={units} monotonic_s={time.monotonic():.3f}",
        flush=True,
    )


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Reuse exact primary and transitive hash checks without newest-file selection."""
    try:
        return prior.authenticate(root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [
            prior.operand(
                "exp7917_exp7892",
                root / "results",
                "custody_readable",
                True,
                f"{type(error).__name__}:{error}",
            )
        ], {}


def cpu_grammar_fixture(scratch: Path, raw: Path, *, model: Path | None = None) -> Json:
    """Prove native grammar enforcement on CPU before admitting the large model."""
    paths = sorted(
        (Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots").glob(
            "*/*Q4_K_M.gguf"
        )
    )
    selected = model or (paths[0] if paths else scratch / "missing-smoke-model")
    if not selected.is_file():
        return dict(passed=False, reason="missing_cpu_fixture", cleanup=dict(leak_free=True))
    scratch.mkdir(parents=True, exist_ok=True)
    runtime = QwenRuntime(selected, scratch, 0)
    command = runtime.command.copy()
    command[command.index("-ngl") + 1] = "0"
    command[command.index("--alias") + 1] = "cpu-grammar-fixture"
    worker = OwnedLlamaCppProcess(
        command=command,
        port=runtime.port,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        log_path=runtime.log,
        state_path=scratch / "owner.json",
    )
    receipt: Json = dict(
        passed=False,
        model_path=str(selected),
        model_sha256=sha256_file(selected),
        scope="cpu_smoke_only_no_measurement_rows",
        command_argv=command,
    )
    progress("before_cpu_fixture_model_load")
    try:
        receipt["owner"] = worker.launch()
        if not bounded(lambda: worker.wait_for_health(60), 60)["ok"]:
            raise RuntimeError("cpu_fixture_load")
        progress("after_cpu_fixture_model_load")
        # An exact grammar with an adversarial prompt proves enforcement instead
        # of merely observing that an instruction-following model printed JSON.
        fixed = 'root ::= "{\\"unsupported_probability\\":0,\\"source_sentence_id\\":null}"'
        progress("before_cpu_fixture_generation")
        reply = bounded(
            lambda: worker.post_json(
                "/completion",
                dict(
                    prompt="Print only BANANA, never JSON.",
                    grammar=fixed,
                    n_predict=96,
                    temperature=0,
                    seed=67801,
                ),
                30,
            ),
            30,
        )
        receipt["response"] = reply
        receipt["passed"] = (
            reply.get("content") == '{"unsupported_probability":0,"source_sentence_id":null}'
        )
        progress("after_cpu_fixture_generation")
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        receipt["reason"] = f"{type(error).__name__}:{error}"
    finally:
        progress("before_cpu_fixture_model_unload")
        receipt["cleanup"] = worker.cleanup()
        receipt["passed"] = receipt["passed"] and receipt["cleanup"]["leak_free"]
        raw.mkdir(parents=True, exist_ok=True)
        if runtime.log.is_file():
            sealed = raw / "cpu-fixture-closed.log"
            sealed.write_bytes(runtime.log.read_bytes())
            receipt["log"] = dict(path=str(sealed), sha256=sha256_file(sealed))
        atomic_json(raw / "cpu-grammar-fixture.json", receipt)
        progress("after_cpu_fixture_model_unload")
    return receipt


def live_capture(
    families: list[Json], scratch: Path, raw: Path
) -> tuple[Json, list[Json], list[Json], list[Json]]:
    """Own one idle GPU and authenticate the serving process without eviction."""
    from carnot.gpu_lease_phase_journal import GpuLease
    from carnot.inference.sota_models import cached_current_model

    identity: Json = {}
    rows: list[Json] = []
    manifest: list[Json] = []
    spec = cached_current_model()
    path = Path(spec["model_path"]) if spec else scratch / "missing-model"
    checks = [
        prior.operand(
            "qwen_cache", path, "hf_id", source.MODEL_ID, spec.get("hf_id") if spec else None
        ),
        prior.operand("qwen_cache", path, "exists", True, path.is_file()),
    ]
    if not all(c["passed"] for c in checks):
        return identity, rows, checks, manifest
    try:
        metadata = read_gguf_metadata(path)
    except (OSError, ValueError) as error:
        checks.append(prior.operand("qwen_gguf", path, "header_readable", True, str(error)))
        return identity, rows, checks, manifest
    checks.extend(
        [
            prior.operand("qwen_gguf", path, "quantization", "Q4_K_M", metadata["quantization"]),
            prior.operand(
                "qwen_gguf",
                path,
                "embedded_tokenizer",
                True,
                metadata["tokenizer_metadata"]["token_count"] > 0
                and metadata["tokenizer_metadata"]["chat_template_present"],
            ),
        ]
    )
    if not all(c["passed"] for c in checks):
        return identity, rows, checks, manifest
    progress("before_model_hash")
    identity.update(
        gguf_metadata=metadata,
        gguf_sha256=bounded(lambda: sha256_file(path), 120),
        revision=path.parent.name,
        quantization="Q4_K_M",
        native_binary_sha256=sha256_file(
            Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
        ),
    )
    progress("after_model_hash")
    snapshot = prior.execute(
        dict(
            commands=[
                dict(
                    name="gpu_capacity",
                    argv=[
                        "nvidia-smi",
                        "--query-gpu=index,uuid,memory.used,memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout_s=10,
                    expected_exit_code=0,
                    expected_failure_reason="",
                    scope="required",
                )
            ]
        ),
        scratch,
        raw,
    )[0]
    candidates = [
        line.split(",")
        for line in Path(snapshot["log_path"]).read_text().splitlines()
        if len(line.split(",")) == 4
        and int(line.split(",")[3]) >= 20000
        and int(line.split(",")[2]) < 1000
    ]
    checks.append(
        prior.operand(
            "gpu_capacity",
            Path(snapshot["log_path"]),
            "idle_capacity",
            True,
            snapshot["passed"] and bool(candidates),
        )
    )
    if not candidates:
        return identity, rows, checks, manifest
    gpu, uuid, used, _ = candidates[0]
    lease: Any = None
    runtime: Any = None
    started = time.monotonic()
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
        runtime = QwenRuntime(path, scratch, int(gpu))
        identity["load_attempted"] = True
        identity.update(runtime.load())
        resident, receipt = prior.gpu_memory(int(gpu), scratch, raw)
        identity["resident_gpu_receipt"] = receipt
        lease.transition("resident", vram_mb=resident)
        lease.transition("inferencing")
        progress("before_view_tokenization")
        manifest = protocol.freeze_views(families, runtime.count)
        atomic_json(
            raw / "view_manifest.json", dict(rows=manifest, config=protocol.freeze_config())
        )
        progress("after_view_tokenization", len(manifest))
        rows = protocol.capture(manifest, runtime, raw, started_s=started)
        identity["request_receipts"] = runtime.receipts
        identity["measured_duration_s"] = time.monotonic() - started
    except (RuntimeError, OSError, TimeoutError, ValueError) as error:
        checks.append(
            prior.operand(
                "owned_runtime",
                path,
                "identity_execution",
                "authenticated",
                f"{type(error).__name__}:{error}",
            )
        )
    finally:
        progress("before_model_unload")
        identity["cleanup"] = runtime.close() if runtime else dict(leak_free=True)
        checks.append(
            prior.operand(
                "owned_cleanup", path, "leak_free", True, identity["cleanup"]["leak_free"]
            )
        )
        if lease:
            if lease.document["phase"] in ["resident", "inferencing"]:
                lease.transition("unloading")
                after, receipt = prior.gpu_memory(int(gpu), scratch, raw)
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
            raw.mkdir(parents=True, exist_ok=True)
            sealed = raw / "server-closed.log"
            sealed.write_bytes(runtime.log.read_bytes())
            identity["server_log"] = dict(path=str(sealed), sha256=sha256_file(sealed))
        progress("after_model_unload")
    return identity, rows, checks, manifest


def command_manifest(scratch: Path) -> Json:
    """Parameterize the existing bounded check plan and preserve required failures."""
    plan = prior.command_manifest(scratch)
    files = [MODULE, VERIFY, CLI, TEST]
    tests = [
        TEST,
        prior.TEST,
        "tests/python/test_experiment_7917_v687_intervention_qualification.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_experiment_7770_v676_qwen_runner_qualification.py",
    ]
    # Keep the old success, rejection and historical-date routes, changing only
    # the task-owned CLI, unit selectors and coverage includes.
    for spec in plan["commands"]:
        argv = spec["argv"]
        if spec["name"] == "unit_coverage":
            argv[argv.index(prior.TEST) :] = tests
        spec["argv"] = [
            a.replace(prior.INCLUDE, INCLUDE)
            .replace(prior.CLI, CLI)
            .replace(
                str(scratch / "blocked.json"), str(scratch / "experiment_7932_fixture_blocked.json")
            )
            for a in argv
        ]
        if spec["name"] in {"ruff_check", "ruff_format", "mypy"}:
            spec["argv"] = spec["argv"][: (3 if spec["name"] in {"ruff_format", "mypy"} else 2)] + (
                files if spec["name"] != "mypy" else [MODULE, VERIFY, CLI]
            )
        if spec["name"] == "scoped_spec":
            spec["argv"] = spec["argv"][:2] + tests
    py = str(ROOT / ".venv/bin/python")
    commands = [
        dict(
            name="model_free_checks",
            argv=[
                py,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(scratch / "model-free"),
                "-q",
                TEST,
                "-k",
                "not orchestration and not owned_runtime and not cpu_grammar_fixture",
            ],
            timeout_s=60,
            expected_exit_code=0,
            expected_failure_reason="",
            scope="required",
        )
    ]
    commands.extend(plan["commands"][:2])
    commands.append(
        dict(
            name="e2e_014",
            argv=[
                py,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(scratch / "e2e-014"),
                "-q",
                tests[-1],
            ],
            timeout_s=120,
            expected_exit_code=0,
            expected_failure_reason="",
            scope="required",
        )
    )
    for name, argv in [
        (
            "e2e_014_replay",
            [
                py,
                "-u",
                "scripts/experiments/experiment_7770_v676_qwen_runner_qualification.py",
                "--cold-replay",
                str(ROOT / "results/experiment_7770_v676_qwen_runner_qualification.json"),
            ],
        ),
        (
            "e2e_014_adversarial",
            [
                py,
                "scripts/adversarial_verify.py",
                "--json",
                str(ROOT / "results/experiment_7770_v676_qwen_runner_qualification.json"),
            ],
        ),
        (
            "e2e_014_strict",
            [
                py,
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(ROOT / "results/experiment_7770_v676_qwen_runner_qualification.json"),
            ],
        ),
    ]:
        commands.append(
            dict(
                name=name,
                argv=argv,
                timeout_s=60,
                expected_exit_code=0,
                expected_failure_reason="",
                scope="required",
            )
        )
    commands.extend(plan["commands"][2:])
    fixture = str(scratch / "paired-fixture.json")
    for name, args in [
        ("fixture", ["--fixture-e2e", fixture]),
        ("fixture_replay", ["--cold-replay", fixture]),
    ]:
        commands.insert(
            next(i for i, c in enumerate(commands) if c["name"] == "coverage_combine"),
            dict(
                name="cli_" + name,
                argv=[
                    str(ROOT / ".venv/bin/coverage"),
                    "run",
                    "--data-file=" + str(scratch / (".coverage." + name)),
                    "--include=" + INCLUDE,
                    CLI,
                    *args,
                ],
                timeout_s=60,
                expected_exit_code=0,
                expected_failure_reason="",
                scope="required",
            ),
        )
    combine = next(c for c in commands if c["name"] == "coverage_combine")
    combine["argv"].extend(str(scratch / (".coverage." + n)) for n in ["fixture", "fixture_replay"])
    plan.update(
        commands=commands,
        coverage_include=INCLUDE,
        affected_tests=tests,
        source_hashes={
            p: sha256_file(ROOT / p)
            for p in files
            + [
                prior.MODULE,
                prior.RUNTIME,
                "python/carnot/verify/source_interventions.py",
                "python/carnot/verify/source_alignment.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/inference/llama_cpp_process.py",
                "python/carnot/inference/gguf_metadata.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/experiment_7303_validation_scope.py",
            ]
        },
        configuration=protocol.freeze_config(),
    )
    return plan


def replay(path: Path) -> Json:
    """Cold-read primitive requests and replies before accepting derived claims."""
    artifact = json.loads(path.read_text())
    if artifact.get("validation_command_manifest_path"):
        if (
            sha256_file(Path(artifact["validation_command_manifest_path"]))
            != artifact["validation_command_manifest_sha256"]
        ):
            raise ValueError("command_manifest_drift")
    manifest = artifact.get("view_manifest")
    frozen: Json = {}
    if manifest and manifest.get("sha256"):
        if sha256_file(Path(manifest["path"])) != manifest["sha256"]:
            raise ValueError("view_manifest_drift")
        frozen = {v["family_id"]: v for v in json.loads(Path(manifest["path"]).read_text())["rows"]}
    rng = random.Random(68832)
    for at in range(0, len(artifact["rows"]), 2):
        order = list(protocol.DECODERS)
        rng.shuffle(order)
        for position, row in enumerate(artifact["rows"][at : at + 2]):
            if (
                row["decoder"] != order[position]
                or row["paired_order"] != order
                or row["order_position"] != position
            ):
                raise ValueError("decoder_order_drift")
    for ref in artifact["raw_response_shards"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("shard_drift")
        if json.loads(Path(ref["path"]).read_text())["rows"] != artifact["rows"]:
            raise ValueError("primitive_drift")
    for row in artifact["rows"]:
        if frozen and row.get("request_bytes"):
            expected = protocol.payload(frozen[row["family_id"]], row["view"], row["decoder"])
            if (
                json.loads(row["request_bytes"]) != expected
                or row["visible_ids"]
                != frozen[row["family_id"]]["requests"][row["view"]]["visible_ids"]
            ):
                raise ValueError("frozen_request_drift")
        for side in ["request", "response"]:
            if (
                row.get(side + "_bytes")
                and source.digest(row[side + "_bytes"].encode()) != row[side + "_sha256"]
            ):
                raise ValueError("primitive_hash_drift")
        if row.get("response_bytes"):
            parsed = protocol.parse_response(json.loads(row["response_bytes"]), row["visible_ids"])
            if any(row[k] != v for k, v in parsed.items()):
                raise ValueError("parse_drift")
    reduced = protocol.reduce_rows(artifact["rows"])
    if any(artifact.get(k) != v for k, v in reduced.items()):
        raise ValueError("reduction_drift")
    return reduced


def build_artifact(
    rows: list[Json],
    checks: list[Json],
    authorities: Json,
    identity: Json,
    receipts: list[Json],
    coverage: Json,
    spans: list[Json],
    started_ns: int,
) -> Json:
    """Keep execution readiness and transport gain distinct from scientific benefit."""
    reduced = protocol.reduce_rows(rows)
    failed = [c for c in checks if not c["passed"]]
    owned_failure = any(not r["passed"] for r in receipts if r["scope"] == "required") or any(
        c["upstream_id"] == "owned_cleanup" for c in failed
    )
    ready = bool(
        identity.get("authenticated")
        and not owned_failure
        and not failed
        and len(rows) == 384
        and receipts
        and coverage
        and all(v["statements"] > 0 and v["missing"] == 0 for v in coverage.values())
    )
    benefit = reduced["paired_completion_delta"]["protocol_benefit"] and ready
    verdict = (
        "disqualified"
        if owned_failure
        else "blocked"
        if failed
        else "circular_positive"
        if benefit
        else "null"
    )
    invoked = identity.get("load_attempted", False)
    artifact = dict(
        reduced,
        schema="carnot.exp7932.qwen_completion.v1",
        experiment_id=7932,
        task_id=TASK,
        milestone="2026.09.688",
        run_date="20260930",
        honest_verdict="complete_" + verdict + "_paired_decoder_completion",
        verdict_class=verdict,
        flagged_adversarial=False,
        rows=rows,
        request_rows=rows,
        gate_check_summary=failed,
        preconditions_checked=checks,
        qwen_measurement_ready_score=int(ready),
        acceptance_gate_results=dict(
            validity=not owned_failure,
            readiness=ready,
            calibration=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
            protocol_benefit=bool(benefit),
            scientific_benefit=None,
        ),
        duration_s=(time.monotonic_ns() - started_ns) / 1e9,
        phase_spans=spans,
        random_seed=67801,
        reproducibility_checksum=canonical_hash(
            dict(
                code={p: sha256_file(ROOT / p) for p in [MODULE, VERIFY, CLI]},
                configuration=protocol.freeze_config(),
                checks=checks,
                rows=rows,
            )
        ),
        source_artifact_hashes=[
            dict(
                path=c["artifact_path"],
                sha256=c["artifact_sha256"],
                role=c["upstream_id"],
                exposure_status="exposed_development",
            )
            for c in checks
            if c.get("artifact_sha256")
        ],
        resolved_imports={
            m: str(Path(importlib.import_module(m).__file__).resolve())
            for m in [
                "carnot." + NAME,
                "carnot.verify.qwen_completion_7932",
                "carnot.inference.qwen_sufficiency_7920",
                "carnot.reporting.primary_publication",
            ]
        },
        validation_receipts=[r for r in receipts if r["scope"] == "required"],
        observed_child_commands=[r["command_argv"] for r in receipts]
        + ([identity["command"]] if "command" in identity else []),
        coverage_statement_counts=coverage,
        historical_required_failures=[
            r for a in authorities.values() for r in a.get("historical_required_failures", [])
        ],
        repository_health=dict(
            status="observed_debt",
            current_receipts=[r for r in receipts if r["scope"] == "diagnostic"],
            affects_required_checks=False,
        ),
        verifier_is_oracle=True,
        claim_scope="exposed_development; grammar/schema benefit circular; semantic sensitivity descriptive",
        inference_substrate="live_llm_inference"
        if invoked
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="model_bounded_generation" if invoked else "blocked_no_run",
        planned_inference_substrate_class="model_bounded_generation",
        execution_venue="host",
        MODEL_SPECS=MODEL_SPECS if invoked else [],
        model_specs=MODEL_SPECS if invoked else [],
        target_model=source.MODEL_ID if invoked else None,
        trained_head_specs=[],
        model_invocation_counts=dict(
            model_loads_attempted=int(invoked),
            model_loads_completed=int(identity.get("authenticated", False)),
            generation_calls_attempted=sum(r["started"] for r in rows) if invoked else 0,
            generation_calls_completed=sum(bool(r["response_bytes"]) for r in rows)
            if invoked
            else 0,
        ),
        model_identity_receipt=identity,
        gguf_sha256=identity.get("gguf_sha256"),
        quantization=identity.get("quantization"),
        model_revision=identity.get("revision"),
        offload_evidence=identity.get("offload_layers"),
        raw_response_shards=[],
        view_manifest=None,
        eligibility_rows=[],
        decoder_order=[],
        grammar_sha256=source.digest(protocol.GRAMMAR.encode()),
        methodology="Fixed label-free lexical witness; identical byte views and messages; paired plain and enforced grammar; no sentence truth labels.",
        prior_mechanism_change="Exp7920 model-nominated witness replaced by frozen lexical witness and paired grammar control",
        retire_if_same_verdict=True,
        production_defaults_changed=False,
    )
    artifact["field_principles"] = {
        k: "Bind current producer evidence, custody and honest claim scope."
        for k in [*artifact, "field_principles"]
    }
    artifact["field_principles"].update(
        qwen_measurement_ready_score="Complete accounting and owned validation can qualify a valid null.",
        complete_family_fraction="All 48 intended families stay in each decoder denominator.",
        semantic_sensitivity="Paired probability changes are descriptive and do not prove entailment.",
        natural_brier="Missing independent sentence truth labels prohibit a Brier claim.",
        natural_cost="Syntax and sensitivity do not supply natural decision costs.",
        verifier_is_oracle="Grammar and schema acceptance are exact protocol authority, so benefits are circular.",
    )
    return artifact


def fixture(path: Path) -> None:
    """Exercise CLI transport accounting without supplying natural measurement rows."""
    text = (
        "Other words. Other words. Target fact. Other words. Other words. Other words. Other words."
    )
    answer = "Target fact. More text."
    family = dict(
        family_id="fixture",
        source_group="fixture",
        complete_source=text,
        complete_response=answer,
        source_sha256=source.digest(text.encode()),
        response_sha256=source.digest(answer.encode()),
        eligible=True,
    )

    class Runtime:
        def count(self, text: str) -> int:
            return len(text.split())

        def generate(self, payload: Json) -> Json:
            return dict(
                model=source.MODEL_ID,
                choices=[
                    dict(
                        finish_reason="stop",
                        message=dict(
                            content='{"unsupported_probability":0.3,"source_sentence_id":null}'
                        ),
                    )
                ],
                usage=dict(prompt_tokens=100, completion_tokens=20),
            )

    runtime = Runtime()
    raw = path.parent / (path.stem + "-raw")
    rows = protocol.capture(protocol.freeze_views([family], runtime.count), runtime, raw)
    atomic_json(
        path,
        dict(
            protocol.reduce_rows(rows),
            rows=rows,
            raw_response_shards=[],
            verifier_is_oracle=True,
            verdict_class="circular_positive",
            model_invocation_counts=dict(model_loads_attempted=0, generation_calls_attempted=0),
        ),
    )


def run(root: Path, scratch: Path, output: Path) -> Json:
    """Freeze custody and checks, run owned work, then publish exact checked bytes."""
    progress("start")
    started_ns = phase = time.monotonic_ns()
    scratch.mkdir(parents=True, exist_ok=True)
    raw = output.parent / "raw" / NAME
    raw.mkdir(parents=True, exist_ok=True)
    plan = command_manifest(scratch)
    manifest_path = raw / "validation_command_manifest.json"
    atomic_json(manifest_path, plan)
    checks, authorities = authenticate(root)
    identity: Json = {}
    rows: list[Json] = []
    views: list[Json] = []
    receipts: list[Json] = []
    coverage: Json = {}
    spans = [
        dict(
            name="preconditions", started_monotonic_ns=phase, ended_monotonic_ns=time.monotonic_ns()
        )
    ]
    progress("after_preconditions", len(checks))
    if all(c["passed"] for c in checks):
        phase = time.monotonic_ns()
        families = prior.freeze_families(authorities["7892"])
        atomic_json(
            raw / "frozen_families.json",
            dict(
                rows=families, config=protocol.freeze_config(), source_hashes=plan["source_hashes"]
            ),
        )
        receipts.extend(prior.execute(dict(commands=plan["commands"][:7]), scratch, raw))
        spans.append(
            dict(
                name="model_free_validation",
                started_monotonic_ns=phase,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        )
        if all(r["passed"] for r in receipts):
            phase = time.monotonic_ns()
            smoke = cpu_grammar_fixture(scratch / "cpu-fixture", raw)
            spans.append(
                dict(
                    name="cpu_grammar_fixture",
                    started_monotonic_ns=phase,
                    ended_monotonic_ns=time.monotonic_ns(),
                )
            )
            checks.append(
                prior.operand(
                    "native_grammar_fixture",
                    raw / "cpu-grammar-fixture.json",
                    "passed",
                    True,
                    smoke["passed"],
                )
            )
            if smoke["passed"]:
                phase = time.monotonic_ns()
                identity, rows, resource_checks, views = live_capture(families, scratch, raw)
                identity["cpu_smoke_receipt"] = smoke
                checks.extend(resource_checks)
                spans.append(
                    dict(
                        name="live_panel",
                        started_monotonic_ns=phase,
                        ended_monotonic_ns=time.monotonic_ns(),
                    )
                )
        phase = time.monotonic_ns()
        receipts.extend(prior.execute(dict(commands=plan["commands"][7:]), scratch, raw))
        spans.append(
            dict(
                name="owned_validation",
                started_monotonic_ns=phase,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        )
        if (scratch / "coverage.json").is_file():
            data = json.loads((scratch / "coverage.json").read_text())
            coverage = {
                p: dict(
                    statements=v["summary"]["num_statements"],
                    covered=v["summary"]["covered_lines"],
                    missing=v["summary"]["missing_lines"],
                )
                for p, v in data["files"].items()
            }
    artifact = build_artifact(
        rows, checks, authorities, identity, receipts, coverage, spans, started_ns
    )
    shard = raw / "raw_rows.json"
    atomic_json(shard, dict(rows=rows))
    artifact.update(
        raw_response_shards=[dict(path=str(shard), sha256=sha256_file(shard))],
        validation_command_manifest_path=str(manifest_path),
        validation_command_manifest_sha256=sha256_file(manifest_path),
        view_manifest=dict(
            path=str(raw / "view_manifest.json"),
            sha256=sha256_file(raw / "view_manifest.json") if views else None,
        ),
        eligibility_rows=[
            dict(
                family_id=v["family_id"],
                source_group=v["source_group"],
                eligible=v["eligible"],
                exclusion_reason=v["exclusion_reason"],
            )
            for v in views
        ],
        decoder_order=[
            dict(
                family_id=r["family_id"],
                view=r["view"],
                decoder=r["decoder"],
                order_position=r["order_position"],
            )
            for r in rows
        ],
        primary_resolution_receipt=dict(
            path=str(raw / "primary_resolution_receipt.json"),
            scope="actual readers bound to final primary hash outside self-referential primary",
        ),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
    )
    artifact["field_principles"].update(
        {
            k: "Seal external receipts against final bytes without a self-referential hash."
            for k in artifact
            if k not in artifact["field_principles"]
        }
    )
    candidate = scratch / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    replay(candidate)
    for attempt in range(2):
        ok, flagged, binding = prior.terminal(candidate, raw / "attempts", attempt)
        atomic_json(raw / f"terminal-attempt-{attempt}.json", binding)
        if ok:
            artifact["flagged_adversarial"] = flagged
            break
        artifact.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_terminal_checks",
            flagged_adversarial=flagged,
            qwen_measurement_ready_score=0,
        )
        artifact["acceptance_gate_results"].update(
            validity=False, readiness=False, protocol_benefit=False
        )
        atomic_json(candidate, artifact)
    else:
        raise ValueError("terminal_revalidation_failed")

    def validate(path: Path) -> Json:
        replay(path)
        passed, actual_flag, final = prior.terminal(path, raw / "final_validation", 2)
        atomic_json(raw / "terminal_validation.json", final)
        return dict(passed=passed and actual_flag == artifact["flagged_adversarial"], binding=final)

    publication = publish_primary(output, artifact, validate)
    atomic_json(raw / "publication_receipt.json", publication)
    os.utime(raw / "terminal_validation.json", None)
    resolution = reader_receipt(
        TASK,
        output.parent,
        field="qwen_measurement_ready_score",
        expected=artifact["qwen_measurement_ready_score"],
    )
    resolution["identity_passed"] = resolution["gate_path"] == resolution["document_path"] == str(
        output.absolute()
    ) and resolution["gate_sha256"] == resolution["document_sha256"] == sha256_file(output)
    atomic_json(raw / "primary_resolution_receipt.json", resolution)
    if not resolution["identity_passed"]:
        raise ValueError("primary_resolution_failed")
    progress("complete", len(rows))
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Offer private fixture and replay routes without historical publication."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260930")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    progress("cli_start")
    if args.date != "20260930":
        raise ValueError("run_date_mismatch")
    if args.cold_replay:
        print(json.dumps(replay(args.cold_replay), sort_keys=True), flush=True)
    elif args.fixture_e2e:
        fixture(args.fixture_e2e)
    else:
        run(args.root, Path(tempfile.mkdtemp(prefix="carnot-7932-")), args.output)
    return 0

"""REQ-REPORT-9098-V776: publish independently reduced, validated runtime capabilities."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import embedding_runtime_9098 as d
from carnot.reporting import mdh_human_release_runner_9056 as prior
from carnot.reporting.current_work_receipt import atomic_json, build_current_work_receipt
from carnot.reporting.natural_label_runner_8975 import read_ref
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
DIRECT = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_inference_sota_models.py",
    "tests/python/test_gguf_metadata.py",
]
REUSED = [
    *prior.REUSED,
    "python/carnot/reporting/human_methods_9084.py",
    "python/carnot/inference/sota_models.py",
    "python/carnot/inference/gguf_metadata.py",
    "python/carnot/inference/bfs_pilot_runtime_8918.py",
]
DEFAULT_INPUT = Path("/tmp/carnot-9098-private")


def scope_manifest(commands: list[Json]) -> Json:
    """Freeze actual command vectors and code bytes before touching the runtime."""
    with patch.multiple(prior, e=d, DIRECT=DIRECT, REUSED=REUSED):
        return prior.scope_manifest(commands)


def receipts_pass(scope: Json, validation: Json) -> bool:
    """Use the established normal-exit, byte-hash and SQLite statement checks."""
    with patch.object(prior, "e", d):
        return prior.receipts_pass(scope, validation)


def run_worker(raw: Path, deadline: float) -> Json:
    """A supervised child isolates native aborts while preserving started model work."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    call = child(
        raw.name,
        [
            "env",
            "-u",
            "PYTHONPATH",
            "-C",
            str(raw),
            "CUDA_VISIBLE_DEVICES=0",
            str(d.ROOT / ".venv/bin/python"),
            "-u",
            str(d.ROOT / d.CLI),
            "--worker",
            str(raw),
        ],
        raw / "logs",
        deadline=max(0.001, deadline - time.monotonic()),
        heartbeat=30,
    )
    result_path = raw / "runtime.json"
    if call["passed"] and result_path.is_file():
        result = json.loads(result_path.read_bytes())
    else:
        event_path = raw / "events.json"
        events = json.loads(event_path.read_bytes())["events"] if event_path.is_file() else []
        if events:
            events.append(dict(events[-1], state="cancelled", monotonic_ns=time.monotonic_ns()))
        receipt = build_current_work_receipt(
            run_id=raw.name,
            owner_pid=events[0]["owner_pid"] if events else os.getpid(),
            events=events,
            inference_substrate="frozen_gguf_forward_scoring",
            inference_substrate_details=dict(native_child_exit=call["exit_code"]),
            inference_substrate_class="model_load_no_generation",
            execution_venue="host_gpu0",
            started_monotonic_ns=call["started_monotonic_ns"],
            ended_monotonic_ns=call["ended_monotonic_ns"],
        )
        result = dict(
            blocker="native_worker",
            runtime_ok=False,
            rows=d.censored("native_worker"),
            embedding_calls=0,
            model_loads=int(bool(events)),
            generated_tokens=0,
            runtime_reference={},
            tokenizer_metadata={},
            source_references=[],
            gate_check_summary=[
                d.gate(
                    Path(call["stderr_path"]),
                    "normal_exit",
                    0,
                    call["exit_code"],
                    "native_worker",
                    d.TASK,
                )
            ],
            current_work_receipt=receipt,
            duration_s=call["duration_s"],
        )
        atomic_json(result_path, result)
    result["process_receipt"] = call
    return result


def measure(raw: Path, date: str, scope: Json) -> Json:
    """Seal the protocol before two cold model processes; never run a natural-label job."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    start = time.monotonic()
    frozen = d.methods.protocol()["embedding"]
    atomic_json(raw / "protocol.json", d.protocol())
    atomic_json(raw / "validation_scope_manifest.json", scope)
    warm = run_worker(raw / "warm", start + 3300)
    cold = (
        run_worker(raw / "cold", start + 3300)
        if warm["runtime_ok"] and all(r["embedding_passed"] for r in warm["rows"])
        else None
    )
    primitive = dict(
        warm=warm,
        cold=cold,
        run_date=date,
        duration_s=time.monotonic() - start,
        validation_scope_manifest=scope,
        protocol_reference=d.ref(raw / "protocol.json"),
        pooling_contract=frozen,
        root=str(d.ROOT),
    )
    atomic_json(raw / "work.json", primitive)
    return d.ref(raw / "work.json")


def build(reference: Json, validation: Json) -> Json:
    """A complete refusal retains every fixture and never inherits a historical model call."""
    work = read_ref(reference)
    warm, cold = work["warm"], work["cold"]
    rows = deepcopy(warm["rows"])
    reduced = d.reduce(rows, cold["rows"] if cold else [], runtime_ok=warm["runtime_ok"])
    passed = receipts_pass(work["validation_scope_manifest"], validation)
    blocker = warm["blocker"] or (cold["blocker"] if cold else None)
    checks = deepcopy(warm["gate_check_summary"] + (cold["gate_check_summary"] if cold else []))
    for capability in ["embeddings", "logits"]:
        check = d.gate(
            Path(reference["path"]),
            capability + "_ready_score",
            1,
            reduced[capability + "_ready_score"],
            capability + "_capability",
            d.TASK,
        )
        checks.append(check)
        if not check["passed"] and blocker is None:
            blocker = check["check"]
    processes = [warm] + ([cold] if cold else [])
    model_loads = sum(p["model_loads"] for p in processes)
    embedding_calls = sum(p["embedding_calls"] for p in processes)
    if not passed:
        reduced.update(embeddings_ready_score=0, logits_ready_score=0)
    completed = sum(r["status"] == "completed" for r in rows)
    value: Json = dict(
        reduced,
        schema="carnot.embedding_runtime.v1",
        experiment_id=9098,
        task_id=d.TASK,
        milestone="2026.11.776",
        run_date=work["run_date"],
        honest_verdict="complete_disqualified_owned_validation"
        if not passed
        else "complete_blocked_" + blocker
        if blocker
        else "complete_positive_runtime_capabilities",
        verdict_class="disqualified" if not passed else "blocked" if blocker else "positive",
        gate_check_summary=checks,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope="Administrative runtime capability on twelve fixed texts only. CPU fixtures establish no scientific benefit. No current ARC solve is claimed.",
        exposure_scope="LPHB remains exposed development. No human labels or natural answers are read.",
        MODEL_SPECS=d.MODEL_SPECS,
        inference_substrate="frozen_gguf_forward_scoring",
        inference_substrate_class="model_load_no_generation",
        generated_tokens=0,
        model_loads=model_loads,
        embedding_calls=embedding_calls,
        current_llm_invocations=model_loads,
        model_invocation_counts=dict(
            model_loads=model_loads,
            generations=0,
            embedding_calls=embedding_calls,
            total=model_loads + embedding_calls,
        ),
        current_work_receipts=[p["current_work_receipt"] for p in processes],
        duration_s=work["duration_s"],
        random_seed=7769098,
        runtime_reference=warm["runtime_reference"],
        tokenizer_metadata=warm["tokenizer_metadata"],
        pooling_contract=work["pooling_contract"],
        fixture_rows=rows,
        rows=rows,
        raw_logit_checks=[
            dict(unit_id=r["unit_id"], checks=r.get("raw_logit_checks", {})) for r in rows
        ],
        intended_count=12,
        attempted_count=completed,
        completed_count=completed,
        failed_count=0,
        censored_count=12 - completed,
        independent_count=0,
        effective_independent_groups=dict(fixture_texts=12, scientific_independent_groups=0),
        sample_size_budget=dict(
            intended_fixtures=12,
            independent_evaluation_groups=0,
            repeats_siblings_and_seeds_multiply_units=False,
        ),
        source_references=[r for p in processes for r in p["source_references"]],
        code_config_hashes=work["validation_scope_manifest"].get("code_test_hashes", []),
        primitive_reference=reference,
        protocol_reference=work["protocol_reference"],
        validation_scope_manifest=work["validation_scope_manifest"],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=validation.get("repository_health", {}),
        acceptance_gates=dict(
            owned_validation=passed,
            frozen_protocol=True,
            actual_embeddings=bool(reduced["embeddings_ready_score"]),
            actual_logits=bool(reduced["logits_ready_score"]),
        ),
        methodology_note="Embedded tokenizer, no BOS/chat template, frozen raw last-layer states, masked mean and separate L2. Serial/batch/padded/cold parity at1e-5/1e-4. Fresh full-vocabulary Yes/No teacher forcing sums all literal token log probabilities; zero generated tokens. No CUDA repair or model fallback.",
    )
    principles = [
        (
            "schema experiment_id task_id milestone run_date",
            "Bind this result to one declared task, schema and execution date.",
        ),
        (
            "honest_verdict verdict_class gate_check_summary",
            "Terminal outcomes retain exact failed operands; missing files, missing fields and zero are distinct.",
        ),
        (
            "required_checks_passed flagged_adversarial acceptance_gates validation_scope_manifest validation_receipts owned_statement_coverage repository_health",
            "Frozen affected commands and measured statements qualify readiness; historical health failures remain separate.",
        ),
        (
            "verifier_is_oracle independent_generalization_score claim_scope exposure_scope",
            "Administrative fixtures establish no scientific benefit, independent generalization or current ARC solve.",
        ),
        (
            "MODEL_SPECS inference_substrate inference_substrate_class model_loads embedding_calls current_llm_invocations model_invocation_counts current_work_receipts generated_tokens",
            "Count only owned frozen forward work; teacher forcing generates zero tokens and changes no weights.",
        ),
        (
            "runtime_reference tokenizer_metadata source_references code_config_hashes primitive_reference protocol_reference",
            "Exact model, embedded tokenizer, build, code and primitive bytes prevent silent operand substitution.",
        ),
        (
            "pooling_contract fixture_rows rows raw_logit_checks",
            "Keep column normalization, raw-state masks and complete continuation likelihoods auditable per scheduled fixture.",
        ),
        (
            "intended_count attempted_count completed_count failed_count censored_count independent_count effective_independent_groups sample_size_budget",
            "Retain scheduled denominators; repeated processes, seeds and sibling fixtures create no independent scientific sample.",
        ),
        (
            "duration_s random_seed reproducibility_checksum",
            "Actual measured time and fixed configuration identify reproducible work without padding clocks.",
        ),
    ]
    value["field_principles"] = {key: why for keys, why in principles for key in keys.split()}
    value["field_principles"].update(
        gate_check_summary="Exact paths, hashes and expected/observed operands separate missing_file, missing_field and zero.",
        rows="Every scheduled fixture remains in the denominator; repeats add no scientific units.",
        current_work_receipts="Only owned current events count as model work; historical generation is absent.",
        embeddings_ready_score="Actual hidden states and every fixed parity check must qualify independently.",
        logits_ready_score="Raw unmasked vocabulary logits and complete fresh continuation likelihoods qualify independently.",
        duration_projection="Retained cap-length timings bound both450-row jobs without changing the protocol.",
        validation_receipts="Normal affected command exits and100 percent owned statements precede publication.",
    )
    value["reproducibility_checksum"] = d.canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild all aggregates and bind original process evidence before trusting readiness."""
    value = json.loads(path.read_bytes())
    work = read_ref(value["primitive_reference"])
    if read_ref(work["protocol_reference"]) != d.protocol():
        raise ValueError("protocol_drift")
    for reference in value["source_references"] + value["code_config_hashes"]:
        if d.ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    for result in [work["warm"]] + ([work["cold"]] if work["cold"] else []):
        call = result["process_receipt"]
        for stream in ["stdout", "stderr"]:
            if d.ref(Path(call[stream + "_path"]))["sha256"] != call[stream + "_sha256"]:
                raise ValueError("runtime_log_hash")
        original = Path(call["stdout_path"]).parent.parent / "runtime.json"
        retained = read_ref(d.ref(original))
        if retained != {k: v for k, v in result.items() if k != "process_receipt"}:
            raise ValueError("runtime_reduction_drift")
    validation = dict(
        checks=value["validation_receipts"],
        owned_statement_coverage=value["owned_statement_coverage"],
        repository_health=value["repository_health"],
    )
    if build(value["primitive_reference"], validation) != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True, primary_sha256=d.ref(path)["sha256"])


def validate(candidate: Path, logs: Path) -> Json:
    """Keep the unchanged scanners and actual foreign-directory CLI as publication gates."""
    with patch.object(prior, "e", d):
        return prior.validate(candidate, logs)


def main(argv: list[str] | None = None) -> int:
    """Publish only when frozen owned checks qualify; native workers cannot publish directly."""
    d.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261115")
    parser.add_argument("--output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json"))
    parser.add_argument(
        "--scope", type=Path, default=DEFAULT_INPUT / "validation_scope_manifest.json"
    )
    parser.add_argument("--receipts", type=Path, default=DEFAULT_INPUT / "validation.json")
    parser.add_argument("--work", type=Path, default=DEFAULT_INPUT / "work_reference.json")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.date != "20261115":
            raise ValueError("run_date")
        if args.worker:
            from carnot.inference.embedding_runtime_9098 import worker

            worker(args.worker)
            return 0
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        scope = json.loads(args.scope.read_bytes()) if args.scope.is_file() else scope_manifest([])
        validation = json.loads(args.receipts.read_bytes()) if args.receipts.is_file() else {}
        raw = Path(tempfile.mkdtemp(prefix="carnot-9098-work-"))
        reference = (
            json.loads(args.work.read_bytes())
            if args.work.is_file()
            else measure(raw, args.date, scope)
        )
        value = build(reference, validation)
        if not value["required_checks_passed"]:
            d.progress("owned_validation_failed_primary_preserved")
            return 1
        d.progress("publication_before")
        print(
            json.dumps(
                publish_primary(args.output, value, lambda p: validate(p, raw / "terminal"))
            ),
            flush=True,
        )
        d.progress("complete", len(value["rows"]), 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        d.progress("refused:" + str(error))
        return 1

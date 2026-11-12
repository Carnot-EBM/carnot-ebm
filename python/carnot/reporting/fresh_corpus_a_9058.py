"""REQ-REPORT-9058-V773: retain first transport evidence without opening graph truth."""

from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

from carnot.inference import graph_corpus_runtime_8961 as runtime
from carnot.reporting import fresh_seal_custody_9042 as c
from carnot.reporting import fresh_seal_methods_9042 as methods
from carnot.reporting import graph_corpus_a_8961 as shard
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import invocation_provenance_9000 as p
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

__all__ = ["p", "c", "runtime", "canonical_hash"]

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_9058_v773_fresh_corpus_a", "exp9058-fresh-corpus-a"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/fresh_corpus_a_9058.py",
    "python/carnot/reporting/fresh_corpus_runner_9058.py",
    CLI,
]
TEST = "tests/python/test_fresh_corpus_a_9058.py"
ROSTER = "results/experiment_9055_v773_prospective_roster.json"
PINS = {
    ROSTER: "sha256:80aa684f1cc271897823983b66bc2d673073223413233438ad135a81edfbc867",
    "results/experiment_9028_v771_historical_provenance_qualification.json": "sha256:a2cb53fd67a42e3bb9614849c42f27fc1d694a65bdc5d508ac5943a77e774067",
    "results/raw/experiment_9042_v772_fresh_seal_methods/invocations/1794476323104834541/production_custody/checkpoint.json": "sha256:ebd3524db0d662059a2f3ac288d991d5e6a07a32aa69fdc95fe1a7a00eb40d05",
}
MODEL_SPECS = [dict(hf_id=source.SETTINGS["model"], quantization="Q4_K_M", gpu=0)]
PUBLIC_FIELDS = {
    "unit_id",
    "group_id",
    "arm",
    "condition",
    "status",
    "metric",
    "value",
    "numerator",
    "denominator",
    "exclusion_reason",
    "response_sha256",
    "request_id",
    "token_count",
    "duration_s",
}
progress = source.progress


def authenticate(root: Path, raw: Path) -> Json:
    """Independent pins bind the completed source and original scientific state components."""
    evidence: Json = dict(checks=[], references=[], units=[], manifest={}, state_hashes={})
    values = {}
    for name, pin in PINS.items():
        path = root / name
        observed = p.reference(path)["sha256"] or "missing_file"
        evidence["checks"].append(p.gate(path, "sha256", pin, observed, "authority_bytes"))
        if path.is_file():
            evidence["references"].append(p.retain(path, raw))
        if observed == pin:
            values[name] = json.loads(path.read_bytes())
    if len(values) != len(PINS):
        return evidence
    primary = values[ROSTER]
    for field, expected in [
        ("fresh_source_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        evidence["checks"].append(
            p.gate(
                root / ROSTER, field, expected, primary.get(field, "missing_field"), "roster_gate"
            )
        )
    refs = [
        primary["fresh_role_manifest"],
        *primary["raw_shard_hashes"],
        *primary["code_config_hashes"],
    ]
    for ref in refs:
        path = root / Path(ref["path"]).relative_to(ROOT)
        observed = p.reference(path)["sha256"] or "missing_file"
        evidence["checks"].append(p.gate(path, "sha256", ref["sha256"], observed, "source_operand"))
        if observed == ref["sha256"]:
            evidence["references"].append(p.retain(path, raw))
    if any(not r["passed"] for r in evidence["checks"]):
        return evidence
    original = methods.authenticate(root, raw)
    evidence["checks"].extend(original["checks"])
    evidence["references"].extend(original["references"])
    manifest = json.loads(
        (root / Path(primary["fresh_role_manifest"]["path"]).relative_to(ROOT)).read_bytes()
    )
    for field, expected in [
        ("complete", True),
        ("fixture_only", False),
        ("first_response_budget", 1),
        ("generator_settings", source.SETTINGS),
        ("state_hashes", original["state_hashes"]),
    ]:
        evidence["checks"].append(
            p.gate(
                Path(primary["fresh_role_manifest"]["path"]),
                field,
                expected,
                manifest.get(field, "missing_field"),
                "frozen_manifest",
            )
        )
    units = source.scheduled(manifest, "A")
    shard.check_units(units)
    exposed = [
        p.reference(path)
        for number in (9045, 9046, 9058, 9059)
        for directory in sorted((root / "results/raw").glob(f"experiment_{number}_*"))
        for pattern in ("intent.json", "wire_response.json", "response.json")
        for path in sorted(directory.rglob(pattern))
    ]
    evidence["checks"].append(
        p.gate(root / "results/raw", "prior_response_attempts", [], exposed, "no_response_exposure")
    )
    evidence.update(units=units, manifest=manifest, state_hashes=original["state_hashes"])
    return evidence


def public_rows(units: list[Json], checkpoint: c.Checkpoint) -> list[Json]:
    """Only identity and scalar transport facts may enter the early primary."""
    rows = []
    for unit in units:
        uid = unit["unit_id"]
        transport = checkpoint.state["transport"].get(uid, {})
        status = transport.get("status", "censored")
        rows.append(
            dict(
                unit_id=uid,
                group_id=unit["graph_id"],
                arm="first_response",
                condition=unit["role"],
                status=status,
                metric="transport_completed",
                value=int(status == "completed"),
                numerator=int(status == "completed"),
                denominator=1,
                exclusion_reason=None if status == "completed" else checkpoint.state["status"][uid],
                **{k: v for k, v in transport.items() if k != "status"},
            )
        )
    return rows


def collect(
    units: list[Json],
    transport: Any,
    raw: Path,
    model: Json,
    *,
    checkpoint: c.Checkpoint,
    budget_s: float = 2700,
) -> list[Json]:
    """Spend intent before each call; lost attempts cannot earn replacement responses."""
    started = time.monotonic()
    for index, unit in enumerate(units):
        uid = unit["unit_id"]
        if checkpoint.state["status"][uid] != "pending":
            continue
        remaining = budget_s - (time.monotonic() - started)
        if remaining <= 0:
            checkpoint.state["status"][uid] = "censored"
            atomic_json(checkpoint.path, checkpoint.state)
            continue
        folder = runtime.folder(raw, index, unit)
        payload = dict(source.SETTINGS, messages=[dict(role="user", content=unit["prompt"])])
        checkpoint.start(uid)
        atomic_json(folder / "intent.json", dict(unit_id=uid, request=payload, model=model))
        progress("before_generation", index, len(units) - index)
        began = time.monotonic()
        try:
            response = transport(payload, min(30, remaining))
            duration = time.monotonic() - began
            atomic_json(
                folder / "response.json",
                dict(raw_response=response, duration_s=duration),
            )
            receipt = dict(
                status="completed",
                response_sha256=p.reference(folder / "response.json")["sha256"],
                request_id=response["id"],
                token_count=response["usage"]["completion_tokens"],
                duration_s=duration,
            )
        except (RuntimeError, OSError, ValueError, KeyError, TimeoutError) as error:
            atomic_json(folder / "failure.json", dict(error=str(error)))
            receipt = dict(status="failed", duration_s=time.monotonic() - began)
        checkpoint.finish(uid, receipt)
        progress("after_generation", index + 1, len(units) - index - 1)
        if (index + 1) % 16 == 0:
            progress("durable_checkpoint", index + 1, len(units) - index - 1)
    return public_rows(units, checkpoint)


def build(work: Json, validation: Json, accepted: bool) -> Json:
    """Readiness certifies complete transport custody, never scientific label balance."""
    raw = Path(work["raw"])
    evidence, runtime_result = work["evidence"], work["runtime"]
    rows = work["rows"]
    counts = Counter(r["status"] for r in rows)
    invocation = p.serialize_invocation(runtime_result["counts"], runtime_result["ledger"], raw)
    failures = [r for r in evidence["checks"] + work["environment"]["checks"] if not r["passed"]]
    failures += runtime_result["checks"]
    owned = accepted and not runtime_result["owned_failure"]
    ready = int(
        owned
        and not failures
        and len(rows) == counts["completed"] == 512
        and not work["fixture_only"]
        and runtime_result["counts"]["generation_calls_completed"] == 512
        and runtime_result["counts"]["model_loads_completed"] == 1
        and runtime_result["model_receipt"].get("authenticated") is True
    )
    cls = "blocked" if failures else "circular_positive" if ready else "null"
    if not owned and not work["fixture_only"]:
        cls = "disqualified"
    receipt = runtime_result["model_receipt"]
    value = dict(
        experiment_id=9058,
        task_id=TASK,
        milestone="2026.11.773",
        run_date="20261112",
        honest_verdict="complete_"
        + cls
        + "_"
        + (failures[0]["check"] if cls == "blocked" else "fresh_corpus_a"),
        verdict_class=cls,
        status="blocked" if cls == "blocked" else "error" if cls == "disqualified" else "success",
        shard_id="A",
        shard_ready_score=ready,
        required_checks_passed=owned,
        verifier_is_oracle=True,
        independent_generalization_score=0,
        claim_scope="First-response custody only. Executable graph truth is an oracle; no detector benefit or independent generalization is measured.",
        exposure_scope="Private raw responses and logprobs; primary contains identity and scalar transport only. No TEST truth or predictions opened.",
        inference_substrate="live_llm_inference"
        if runtime_result["counts"]["generation_calls_attempted"]
        else "model_loading"
        if runtime_result["counts"]["model_loads_attempted"]
        else "blocked_no_run",
        **invocation,
        planned_inference_substrate_class="model_bounded_generation",
        minimum_duration_s=10 if runtime_result["counts"]["generation_calls_attempted"] else 0,
        MODEL_SPECS=MODEL_SPECS,
        model_specs=MODEL_SPECS if runtime_result["counts"]["model_loads_attempted"] else [],
        trained_head_specs=[],
        numerical_training_calls=0,
        model_invoked=bool(runtime_result["counts"]["model_loads_attempted"]),
        rows=rows,
        first_response_records=rows,
        intended_count=512,
        completed_count=counts["completed"],
        failed_count=counts["failed"],
        censored_count=512 - counts["completed"] - counts["failed"],
        excluded_count=0,
        independent_count=counts["completed"],
        censoring_rows=[r for r in rows if r["status"] != "completed"],
        gate_check_summary=evidence["checks"]
        + work["environment"]["checks"]
        + runtime_result["checks"],
        state_hashes=evidence["state_hashes"],
        custody_reference=p.reference(raw / "custody/checkpoint.json"),
        model_tokenizer_hashes=dict(
            model_sha256=receipt.get("gguf_sha256"),
            tokenizer=receipt.get("tokenizer"),
            tokenizer_container_sha256=receipt.get("gguf_sha256"),
            quantization=receipt.get("quantization"),
            server_executable_reference=work["server_executable_reference"],
        ),
        actual_offload=receipt.get("offload_layers", []),
        model_receipt=receipt,
        resource_lease=runtime_result.get("resource_lease", {}),
        per_request_receipts=[
            r for r in runtime_result["ledger"] if r["operation"] == "generation"
        ],
        label_access_policy=evidence["manifest"].get("label_access_policy", {}),
        generator_settings=source.SETTINGS,
        duration_s=work["duration_s"],
        random_seed=methods.SEED,
        phase_spans=work["phase_spans"] + runtime_result.get("phase_spans", []),
        preconditions_checked=dict(
            environment=work["environment"],
            input_checks=evidence["checks"],
            runtime=runtime_result.get("preconditions_checked", {}),
        ),
        raw_shard_hashes=work["raw_references"],
        source_artifact_hashes=evidence["references"],
        code_config_hashes=work["code_references"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=int(name.split("experiment_")[1].split("_")[0]),
                sha256=pin,
                fields_imported=["source_identity", "historical_verdict_and_custody_only"],
            )
            for name, pin in PINS.items()
        ],
        validation_scope_manifest=validation["scope"],
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        repository_health=dict(
            passed=False,
            full_suite_repeated=False,
            historical_reference=p.reference(
                ROOT / "results/experiment_9000_v769_invocation_provenance.json"
            ),
        ),
        methodology_note="Authenticate the unchanged completed V773 roster and fits. Freeze shard A512 first attempts,64 CAL/64 TEST/384 online. Qwen Q4_K_M, temperature0, reasoning disabled,16 tokens, actual logprobs. Preflight/load each240s, each request30s, collection2700s. No retries, repair, fallback, fitting or label reads. Admission checks class support independently.",
        fixture_only=work["fixture_only"],
        flagged_adversarial=False,
    )
    value["reproducibility_checksum"] = p.checksum(value)
    return value

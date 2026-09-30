"""Measure source sensitivity while preserving sentence and source custody.

REQ-REPORT-7920-V687 and REQ-VERIFY-7920-V687. Edited-context risks cannot
establish truth. Missing sentence labels therefore leave natural scores null.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import context_sufficiency_7854 as context
from carnot.verify import intervention_protocol_7868 as protocol
from carnot.verify import source_interventions as source

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7920_v687_qwen_sufficiency"
MODULE = f"python/carnot/{NAME}.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = f"tests/python/test_{NAME}.py"
RUNTIME = "python/carnot/inference/qwen_sufficiency_7920.py"
INCLUDE = f"*/{NAME}.py,*/qwen_sufficiency_7920.py"
MODEL_SPECS = [source.MODEL_ID]


def progress(phase: str, units: int = 0) -> None:
    """Flush boundaries so supervisors can distinguish slow work from a stall."""
    print(
        f"[exp7920] phase={phase} monotonic_s={time.monotonic():.3f} completed_units={units}",
        flush=True,
    )


def operand(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Keep failed thresholds distinct from absent evidence for downstream gates."""
    return dict(
        upstream_id=upstream,
        artifact_path=str(path.resolve()),
        artifact_sha256=sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Inspect current authorities directly rather than inherit a conductor gate."""
    checks: list[Json] = []
    authorities: Json = {}
    for number, name, score in [
        (7917, "intervention_qualification", "intervention_protocol_ready_score"),
        (7892, "source_boundary", "source_boundary_ready_score"),
    ]:
        path = (
            root / "results" / f"experiment_{number}_v{687 if number == 7917 else 685}_{name}.json"
        )
        checks.append(operand(f"exp{number}", path, "exists", True, path.is_file()))
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        authorities[str(number)] = value
        for field, expected in [
            ("experiment_id", number),
            (score, 1),
            ("flagged_adversarial", False),
            ("verdict_class", "circular_positive"),
        ]:
            checks.append(operand(f"exp{number}", path, field, expected, value.get(field)))
        references = list(value.get("source_artifact_hashes", []))
        if isinstance(value.get("source_artifact_hashes"), dict):
            references = list(value["source_artifact_hashes"].values())
        for key in [
            "protocol_manifest",
            "fixture_rows",
            "cohort_manifest",
            "validation_command_manifest",
        ]:
            if key + "_path" in value:
                p = Path(value[key + "_path"])
                references.append(
                    dict(
                        path=str(p),
                        sha256=value.get(key + "_sha256", sha256_file(p) if p.is_file() else None),
                    )
                )
        references.extend(value.get("public_shards", []))
        for ref in references:
            p = Path(ref["path"])
            checks.append(
                operand(
                    f"exp{number}",
                    p,
                    "sha256",
                    ref.get("sha256"),
                    sha256_file(p) if p.is_file() else None,
                )
            )
        manifest_path = value.get("validation_command_manifest_path")
        if manifest_path and Path(manifest_path).is_file():
            manifest = json.loads(Path(manifest_path).read_text())
            for label, expected in manifest.get("source_hashes", {}).items():
                p = root / label
                checks.append(
                    operand(
                        f"exp{number}_code",
                        p,
                        "sha256",
                        expected,
                        sha256_file(p) if p.is_file() else None,
                    )
                )
    return checks, authorities


def freeze_families(authority: Json) -> list[Json]:
    """Select source groups before outputs and retain every selected exclusion."""
    paths = [
        *authority["public_shards"],
        dict(path=authority["cohort_manifest_path"], sha256=authority["cohort_manifest_sha256"]),
    ]
    for ref in paths:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("custody_drift")
    public = {
        r["family_id"]: r
        for ref in authority["public_shards"]
        for r in map(json.loads, Path(ref["path"]).read_text().splitlines())
    }
    cohort = json.loads(Path(authority["cohort_manifest_path"]).read_text())["rows"]
    selected = sorted(
        (r for r in cohort if r["role"] == "evaluation"),
        key=lambda r: (r["source_group"], r["family_id"]),
    )[:48]
    rows = []
    for item in selected:
        raw = public[item["family_id"]]
        answer = raw["answer_bytes"]
        text = raw["source_bytes"]
        first = source.target_span(answer.encode()) if answer else None
        eligible = bool(
            text
            and first
            and answer.encode()[: first["end_byte"]].strip().endswith((b".", b"!", b"?"))
        )
        rows.append(
            dict(
                family_id=item["family_id"],
                source_group=item["source_group"],
                complete_source=text,
                complete_response=answer,
                source_sha256=source.digest(text.encode()),
                response_sha256=source.digest(answer.encode()),
                eligible=eligible,
                sentence_label_eligibility="unknown_no_independent_original_sentence_label",
            )
        )
    return rows


def measure(
    families: list[Json], frozen: Json, runtime: Any, raw: Path, *, latest_launch_s: float = 2400
) -> list[Json]:
    """Capture qualified views and keep budget stops in their original positions."""
    started = time.monotonic()
    rows: list[Json] = []

    def transport(payload: Json) -> Json:
        if time.monotonic() - started >= latest_launch_s:
            raise InterruptedError("launch_budget")
        progress("before_generation", len(rows))
        try:
            return runtime.generate(payload)
        finally:
            progress("after_generation", len(rows))

    for family in families:
        progress("family_begin", len(rows) // 4)
        if family["eligible"]:
            captured = context.capture_family(family, frozen, transport, runtime.count)
        else:
            captured = [
                dict(
                    family_id=family["family_id"],
                    arm=arm,
                    status="excluded_ineligible",
                    started=False,
                    completed=False,
                    censored=False,
                    excluded=True,
                    probability=None,
                )
                for arm in context.ARMS
            ]
        requests = [json.loads(r["request_bytes"]) for r in captured if r.get("request_bytes")]
        fidelity = protocol.audit_views(family, requests)
        for row in captured:
            if row["status"] == "interrupted":
                row.update(started=False, censored=False, status="unstarted_launch_budget")
            row.update(
                source_group=family["source_group"],
                intended=True,
                eligible=not row["excluded"],
                failed=bool(row["started"] and not row["completed"] and not row["censored"]),
                seed=67801,
                source_byte_fidelity=fidelity,
                syntax_valid=row["completed"] if row["started"] else None,
                sentence_label_eligibility="unknown_no_independent_original_sentence_label",
                generated_token_truncation='"finish_reason":"length"'
                in (row.get("response_bytes") or ""),
                parse_error=row["status"]
                if row["status"] in ["invalid_parse", "invalid_witness"]
                else None,
            )
        rows.extend(captured)
        atomic_json(
            raw / "checkpoint.json",
            dict(
                input_hash=canonical_hash(families),
                config_hash=canonical_hash(frozen),
                code_hash=sha256_file(Path(__file__)),
                rows=rows,
            ),
        )
    return rows


def reduce_rows(rows: list[Json]) -> Json:
    """Recompute unit counts and paired cluster sensitivity without truth labels."""
    counts = {
        key: sum(bool(row.get(key)) for row in rows)
        for key in [
            "intended",
            "eligible",
            "started",
            "completed",
            "failed",
            "censored",
            "excluded",
        ]
    }
    families: Json = {}
    for row in rows:
        families.setdefault(row["family_id"], {})[row["arm"]] = row
    clusters: Json = {}
    for arms in families.values():
        if set(arms) == set(context.ARMS) and all(r["completed"] for r in arms.values()):
            group = arms["full_source"]["source_group"]
            delta = (
                arms["witness_neighbors"]["probability"] - arms["matched_control"]["probability"]
            )
            clusters.setdefault(group, []).append(delta)
    differences = [sum(values) / len(values) for values in clusters.values()]
    counts["independent"] = len(differences)
    mean = sum(differences) / len(differences) if differences else None
    interval = None
    if len(differences) >= 32:
        assert mean is not None
        se = math.sqrt(
            sum((d - mean) ** 2 for d in differences) / ((len(differences) - 1) * len(differences))
        )
        interval = [mean - 1.96 * se, mean + 1.96 * se]
    return dict(
        sample_size_budget=dict(
            counts,
            intended_families=len(families),
            family_limit=48,
            call_limit=192,
            output_token_limit=24576,
            actual_output_tokens=sum(r.get("usage", {}).get("completion_tokens", 0) for r in rows),
        ),
        syntax_valid=all(r["syntax_valid"] for r in rows if r["started"]),
        source_byte_fidelity=all(r["source_byte_fidelity"] for r in rows),
        natural_brier=None,
        natural_cost=None,
        sentence_label_eligibility=dict(eligible=0, reason="no_independent_sentence_labels"),
        semantic_sensitivity=dict(
            independent_families=len(differences),
            mean_neighbors_minus_filler=mean,
            interval=interval,
            interval_method="paired_source_cluster_normal_95",
            interpretation="source sensitivity only; no correctness or evidence sufficiency certification",
        ),
    )


def replay(path: Path) -> Json:
    """Reduce primitive rows again and reject changed shards or headline counts."""
    artifact = json.loads(path.read_text())
    for ref in artifact["raw_response_shards"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("shard_drift")
    reduced = reduce_rows(artifact["rows"])
    if any(artifact.get(key) != value for key, value in reduced.items()):
        raise ValueError("reduction_drift")
    return reduced


def command_manifest(scratch: Path) -> Json:
    """Freeze exact checks and historical dates before any model is loaded."""
    py = str(ROOT / ".venv/bin/python")
    cov = str(ROOT / ".venv/bin/coverage")
    tests = [
        TEST,
        "tests/python/test_experiment_7917_v687_intervention_qualification.py",
        "tests/python/test_experiment_7893_v685_intervention_protocol.py",
        "tests/python/test_source_boundary_7892.py",
    ]
    specs = []

    def add(
        name: str,
        argv: list[str],
        timeout: int = 120,
        expected: int = 0,
        reason: str = "",
        scope: str = "required",
    ) -> None:
        specs.append(
            dict(
                name=name,
                argv=argv,
                timeout_s=timeout,
                expected_exit_code=expected,
                expected_failure_reason=reason,
                scope=scope,
            )
        )

    fixture = str(scratch / "e2e-016.json")
    for route in ["fixture-e2e", "cold-replay"]:
        add(
            "e2e_016_" + route,
            [
                py,
                "-u",
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--" + route,
                fixture,
            ],
            60,
        )
    add(
        "unit_coverage",
        [
            cov,
            "run",
            "--data-file=" + str(scratch / ".coverage.unit"),
            "--include=" + INCLUDE,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--basetemp=" + str(scratch / "pytest"),
            "-q",
            *tests,
        ],
        180,
    )
    for name, args, exit_code, reason in [
        (
            "success",
            ["--root", str(scratch / "absent"), "--output", str(scratch / "blocked.json")],
            0,
            "",
        ),
        ("wrong_date", ["--date", "20260929"], 1, "run_date_mismatch"),
        ("replay", ["--cold-replay", str(scratch / "blocked.json")], 0, ""),
        (
            "missing_replay",
            ["--cold-replay", str(scratch / "missing.json")],
            1,
            "FileNotFoundError",
        ),
    ]:
        add(
            "cli_" + name,
            [
                cov,
                "run",
                "--data-file=" + str(scratch / (".coverage." + name)),
                "--include=" + INCLUDE,
                CLI,
                *args,
            ],
            60,
            exit_code,
            reason,
        )
    files = [MODULE, CLI, RUNTIME, TEST]
    add(
        "coverage_combine",
        [
            cov,
            "combine",
            "--data-file=" + str(scratch / ".coverage.combined"),
            *[
                str(scratch / (".coverage." + name))
                for name in ["unit", "success", "wrong_date", "replay", "missing_replay"]
            ],
        ],
    )
    add(
        "coverage_json",
        [
            cov,
            "json",
            "--data-file=" + str(scratch / ".coverage.combined"),
            "--include=" + INCLUDE,
            "-o",
            str(scratch / "coverage.json"),
        ],
    )
    add(
        "coverage_report",
        [
            cov,
            "report",
            "--data-file=" + str(scratch / ".coverage.combined"),
            "--include=" + INCLUDE,
            "--show-missing",
            "--fail-under=100",
        ],
    )
    add("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *files])
    add("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *files])
    add(
        "mypy",
        [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", MODULE, RUNTIME, CLI],
    )
    add("scoped_spec", [py, "scripts/check_spec_coverage.py", *tests])
    add(
        "repository_health",
        [
            str(ROOT / ".venv/bin/pytest"),
            "tests/python",
            "-q",
            "--basetemp=" + str(scratch / "repository-pytest"),
        ],
        180,
        scope="diagnostic",
    )
    return dict(
        commands=specs,
        coverage_include=INCLUDE,
        affected_tests=tests,
        source_hashes={p: sha256_file(ROOT / p) for p in files},
        historical_fixture_date="20260929",
        execution_date="20260930",
    )


def execute(plan: Json, scratch: Path, raw: Path) -> list[Json]:
    """Seal exited children and preserve expected rejections as explicit checks."""
    receipts = []
    for index, spec in enumerate(plan["commands"]):
        child = CommandSpec(spec["name"], tuple(spec["argv"]), spec["scope"], spec["timeout_s"])
        receipt = run_commands(ROOT, [child], log_dir=scratch / f"logs-{index}", heartbeat_s=15)[0]
        original = ROOT / receipt["log_path"]
        log = original.read_text()
        receipt.update(
            expected_exit_code=spec["expected_exit_code"],
            deadline_s=spec["timeout_s"],
            expected_failure_reason=spec["expected_failure_reason"],
            passed=receipt["exit_code"] == spec["expected_exit_code"]
            and not receipt["timed_out"]
            and spec["expected_failure_reason"] in log,
        )
        sealed = raw / "logs" / f"{spec['name']}-{receipt['log_sha256'][7:]}.log"
        sealed.parent.mkdir(parents=True, exist_ok=True)
        sealed.write_bytes(original.read_bytes())
        receipt["log_path"] = str(sealed)
        receipts.append(receipt)
    return receipts


def terminal(candidate: Path, scratch: Path, attempt: int) -> tuple[bool, bool, Json]:
    """Reuse qualified readers and bind both reports to the exact candidate bytes."""
    from carnot.experiment_7893_v685_intervention_protocol import validate_terminal

    return validate_terminal(candidate, scratch, time.monotonic(), attempt)


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
    """Keep readiness, validity, scientific benefit and current compute separate."""
    failed_gates = [c for c in checks if not c["passed"]]
    owned_failure = any(not r["passed"] for r in receipts if r["scope"] == "required")
    verdict = "disqualified" if owned_failure else "blocked" if failed_gates else "null"
    invoked = bool(identity.get("load_attempted"))
    complete = sum(r["started"] for r in rows)
    counts = dict(
        model_loads_attempted=int(invoked),
        model_loads_completed=int(identity.get("authenticated", False)),
        generation_calls_attempted=complete,
        generation_calls_completed=sum(r.get("response_bytes") is not None for r in rows),
    )
    result = dict(
        reduce_rows(rows),
        schema="carnot.exp7920.qwen_sufficiency.v1",
        experiment_id=7920,
        task_id="exp7920-qwen-sufficiency",
        milestone="2026.09.687",
        run_date="20260930",
        honest_verdict="complete_" + verdict + "_source_sensitivity",
        verdict_class=verdict,
        flagged_adversarial=False,
        rows=rows,
        request_rows=rows,
        gate_check_summary=failed_gates,
        preconditions_checked=checks,
        acceptance_gate_results=dict(
            validity=not owned_failure,
            readiness=verdict == "null",
            scientific_benefit=None,
            probability_quality=None,
            decision_benefit=None,
        ),
        qwen_measurement_ready_score=int(
            verdict == "null" and identity.get("authenticated", False)
        ),
        duration_s=(time.monotonic_ns() - started_ns) / 1e9,
        phase_spans=spans,
        random_seed=67801,
        reproducibility_checksum=canonical_hash(
            dict(
                rows=rows,
                checks=checks,
                code_hashes={p: sha256_file(ROOT / p) for p in [MODULE, RUNTIME, CLI]},
                configuration=context.freeze_protocol(seed=67801),
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
            if c["artifact_sha256"] is not None
        ],
        resolved_imports={
            "carnot." + NAME: str(Path(__file__).resolve()),
            "carnot.verify.context_sufficiency_7854": str(Path(context.__file__).resolve()),
            "carnot.verify.source_interventions": str(Path(source.__file__).resolve()),
        },
        validation_receipts=[r for r in receipts if r["scope"] == "required"],
        observed_child_commands=[r["command_argv"] for r in receipts]
        + ([identity["command"]] if "command" in identity else [])
        + [
            identity[k]["command_argv"]
            for k in ["resident_gpu_receipt", "unloaded_gpu_receipt"]
            if k in identity and "command_argv" in identity[k]
        ],
        coverage_statement_counts=coverage,
        historical_required_failures=[
            r for a in authorities.values() for r in a.get("historical_required_failures", [])
        ],
        repository_health=dict(
            status="observed_debt",
            current_receipts=[r for r in receipts if r["scope"] == "diagnostic"],
            affects_required_checks=False,
        ),
        verifier_is_oracle=False,
        claim_scope="exposed_development source sensitivity only",
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
        model_invocation_counts=counts,
        model_identity_receipt=identity,
        model_revision=identity.get("revision"),
        gguf_sha256=identity.get("gguf_sha256"),
        quantization=identity.get("quantization"),
        offload_evidence=identity.get("offload_layers"),
        raw_response_shards=[],
        validation_command_manifest_path=None,
        methodology="Fixed first sentence; four qualified source views; no independent sentence truth labels.",
        terminal_validation_chain=[],
    )
    result["field_principles"] = {
        key: "Bind current evidence, exact custody, and honest gate scope."
        for key in [*result, "field_principles"]
    }
    result["field_principles"].update(
        sample_size_budget="Arms never increase independent source count.",
        sentence_label_eligibility="Whole-answer labels cannot score the unchanged sentence.",
        semantic_sensitivity="Edited risk differences measure sensitivity rather than truth.",
        qwen_measurement_ready_score="Authenticated complete accounting can be ready even below the power floor.",
        acceptance_gate_results="Execution validity and readiness do not prove scientific benefit.",
    )
    return result


def run(root: Path, scratch: Path, output: Path) -> Json:
    """Freeze inputs, perform owned bounded work, and publish only checked bytes."""
    progress("start")
    started_ns = time.monotonic_ns()
    scratch.mkdir(parents=True, exist_ok=True)
    raw = output.parent / "raw" / NAME
    raw.mkdir(parents=True, exist_ok=True)
    checks, authorities = authenticate(root)
    identity: Json = {}
    rows: list[Json] = []
    spans: list[Json] = []
    receipts: list[Json] = []
    coverage: Json = {}
    plan = command_manifest(scratch)
    manifest_path = raw / "validation_command_manifest.json"
    atomic_json(manifest_path, plan)
    frozen = context.freeze_protocol(seed=67801)
    if all(c["passed"] for c in checks):
        families = freeze_families(authorities["7892"])
        atomic_json(
            raw / "frozen_families.json",
            dict(
                families=families,
                protocol=frozen,
                source_hashes=plan["source_hashes"],
                authority_checks=checks,
            ),
        )
        phase_start = time.monotonic_ns()
        receipts.extend(execute(dict(commands=plan["commands"][:2]), scratch, raw))
        spans.append(
            dict(
                name="historical_e2e",
                started_monotonic_ns=phase_start,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        )
        if all(r["passed"] for r in receipts):
            capture_begin = time.monotonic_ns()
            identity, rows, resource_checks = live_capture(families, frozen, scratch, raw)
            spans.append(
                dict(
                    name="owned_capture",
                    started_monotonic_ns=capture_begin,
                    ended_monotonic_ns=time.monotonic_ns(),
                )
            )
            checks.extend(resource_checks)
        phase_start = time.monotonic_ns()
        receipts.extend(execute(dict(commands=plan["commands"][2:]), scratch, raw))
        spans.append(
            dict(
                name="validation",
                started_monotonic_ns=phase_start,
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
    artifact["raw_response_shards"] = [dict(path=str(shard), sha256=sha256_file(shard))]
    artifact["validation_command_manifest_path"] = str(manifest_path)
    artifact["validation_command_manifest_sha256"] = sha256_file(manifest_path)
    artifact["field_principles"]["validation_command_manifest_sha256"] = (
        "Freeze exact validation scope before outcomes."
    )
    candidate = scratch / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    replay(candidate)
    for attempt in range(2):
        ok, flagged, binding = terminal(candidate, scratch, attempt)
        atomic_json(raw / f"terminal-{attempt}.json", binding)
        if ok:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".checked.tmp")
            temporary.write_bytes(candidate.read_bytes())
            temporary.replace(output)
            progress("complete", len(rows))
            return artifact
        artifact.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_terminal_checks",
            flagged_adversarial=flagged,
            qwen_measurement_ready_score=0,
        )
        artifact["acceptance_gate_results"].update(validity=False, readiness=False)
        atomic_json(candidate, artifact)
    raise ValueError("terminal_revalidation_failed")


def live_capture(
    families: list[Json], frozen: Json, scratch: Path, raw: Path
) -> tuple[Json, list[Json], list[Json]]:
    """Acquire an owned idle GPU and retain capacity or identity failures explicitly."""
    from carnot.gpu_lease_phase_journal import GpuLease
    from carnot.inference.qwen_sufficiency_7920 import QwenRuntime, bounded
    from carnot.inference.sota_models import cached_current_model

    checks: list[Json] = []
    identity: Json = {}
    rows: list[Json] = []
    spec = cached_current_model()
    path = Path(spec["model_path"]) if spec else scratch / "missing-model"
    checks.append(
        operand("qwen_cache", path, "hf_id", source.MODEL_ID, spec.get("hf_id") if spec else None)
    )
    checks.append(operand("qwen_cache", path, "exists", True, path.is_file()))
    if not all(c["passed"] for c in checks):
        return identity, rows, checks
    progress("model_hash_begin")
    model_hash = bounded(lambda: sha256_file(path), 120)
    identity.update(gguf_sha256=model_hash, revision=path.parent.name, quantization="Q4_K_M")
    progress("model_hash_complete")
    snapshot = execute(
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
        if len(line.split(",")) == 4 and int(line.split(",")[3].strip()) >= 20000
    ]
    checks.append(
        operand(
            "gpu_capacity",
            Path(snapshot["log_path"]),
            "idle_capacity",
            True,
            snapshot["passed"] and bool(candidates),
        )
    )
    if not candidates:
        return identity, rows, checks
    gpu, uuid, used, _free = candidates[0]
    lease = None
    runtime = None
    started = time.monotonic()
    try:
        lease = GpuLease.acquire(
            runtime_dir="/tmp/carnot-gpu-leases",
            task_id="exp7920-qwen-sufficiency",
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
        resident_mb, resident_receipt = gpu_memory(int(gpu), scratch, raw)
        identity["resident_gpu_receipt"] = resident_receipt
        lease.transition("resident", vram_mb=resident_mb)
        lease.transition("inferencing")
        rows = measure(
            families,
            frozen,
            runtime,
            raw,
            latest_launch_s=max(0, 2400 - (time.monotonic() - started)),
        )
        identity["request_receipts"] = runtime.receipts
        identity["measured_duration_s"] = time.monotonic() - started
    except (RuntimeError, OSError, TimeoutError, ValueError) as error:
        checks.append(
            operand(
                "owned_runtime",
                path,
                "capacity_identity_execution",
                "authenticated",
                f"{type(error).__name__}:{error}",
            )
        )
    finally:
        progress("before_model_unload")
        cleanup = runtime.close() if runtime else dict(leak_free=True)
        identity["cleanup"] = cleanup
        if lease:
            if lease.document["phase"] in ["resident", "inferencing"]:
                lease.transition("unloading")
                after_mb, after_receipt = gpu_memory(int(gpu), scratch, raw)
                identity["unloaded_gpu_receipt"] = after_receipt
                lease.transition(
                    "validating",
                    vram_mb=after_mb,
                    exit_code=0,
                    unload_observed=cleanup["leak_free"],
                )
            lease.transition(
                "terminal_complete" if identity.get("authenticated") else "terminal_blocked"
            )
            identity["gpu_lease"]["release"] = lease.release()
        if runtime and runtime.log.is_file():
            sealed = raw / "server-closed.log"
            raw.mkdir(parents=True, exist_ok=True)
            sealed.write_bytes(runtime.log.read_bytes())
            identity["server_log"] = dict(path=str(sealed), sha256=sha256_file(sealed))
        progress("after_model_unload")
    return identity, rows, checks


def main(argv: list[str] | None = None) -> int:
    """Expose private replay and explicit authorities through a small CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260930")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / f"{NAME}.json")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20260930":
        raise ValueError("run_date_mismatch")
    if args.cold_replay:
        print(json.dumps(replay(args.cold_replay), sort_keys=True), flush=True)
    else:
        scratch = Path(tempfile.mkdtemp(prefix="carnot-7920-"))
        run(args.root, scratch, args.output)
    return 0


def gpu_memory(gpu: int, scratch: Path, raw: Path) -> tuple[int, Json]:
    """Measure actual resident memory rather than invent it from a model size."""
    receipt = execute(
        dict(
            commands=[
                dict(
                    name="gpu_memory",
                    argv=[
                        "nvidia-smi",
                        "--id=" + str(gpu),
                        "--query-gpu=memory.used",
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
    return int(Path(receipt["log_path"]).read_text().strip()), receipt

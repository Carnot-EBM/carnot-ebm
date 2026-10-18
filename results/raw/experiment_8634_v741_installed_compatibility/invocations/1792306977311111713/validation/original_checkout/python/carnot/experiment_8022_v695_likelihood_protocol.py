"""REQ-REPORT-8022: qualify a bounded scorer and freeze an exposed public panel.

Readiness authenticates likelihood access. It supplies no human-label accuracy,
new corpus or calibrated energy-fit result. Scientific hypotheses remain open.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot import experiment_8010_v694_source_intervention_protocol as upstream
from carnot.experiment_8011_v694_qwen_source_sensitivity import verify_references
from carnot.inference import fixed_answer_likelihood_8022 as s
from carnot.inference import likelihood_runtime_8022 as runtime
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt

Json = dict[str, Any]
ROOT = runtime.ROOT
NAME = "experiment_8022_v695_likelihood_protocol"
TASK = runtime.TASK
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/inference/fixed_answer_likelihood_8022.py",
    "python/carnot/inference/likelihood_runtime_8022.py",
    CLI,
]
TEST = "tests/python/test_likelihood_protocol_8022.py"


def authenticate(root: Path) -> Json:
    """Read public roles through existing custody, without any fitted-head gate."""
    plan = upstream.authenticate(root)
    plan["public"] = {}
    path = root / "results/experiment_7980_v692_evidence_features.json"
    if not path.is_file():
        plan["checks"].append(
            dict(
                upstream_id="exp7980",
                path=str(path),
                hash=None,
                artifact_field="public_role_manifests",
                expected="present",
                observed="missing_file",
                passed=False,
            )
        )
        return plan
    value = json.loads(path.read_text())
    plan["references"].append(
        dict(reference(path), imported_fields=["public_role_manifests"], scope="historical")
    )
    if all(c["passed"] for c in plan["checks"]):
        for role in s.ROLES:
            ref = (
                plan["public_role_manifests"]["stream"]
                if role == "evaluation"
                else value["public_role_manifests"][role]
            )
            plan["public"][role] = json.loads(checked(ref).read_text())["request_rows"]
            plan["references"].append(ref)
    return plan


def commands(scratch: Path) -> list[CommandSpec]:
    """Reuse validation supervision while covering only the added statements."""
    with patch.object(upstream, "OWNED", OWNED), patch.object(upstream, "TEST", TEST):
        return upstream.commands(scratch)


def build(plan: Json, result: Json, raw: Path, receipts: list[Json], elapsed: float) -> Json:
    """Keep protocol readiness separate from benefit and incomplete external gates."""
    qualification, panel = result.get("qualification", {}), result.get("panel", {})
    failures = [
        dict(c, artifact_field=c.get("artifact_field", c.get("field")))
        for c in plan["checks"] + result.get("checks", [])
        if not c["passed"]
    ]
    if not panel.get("complete", False) or panel.get("counts") != s.ROLES:
        failures.append(
            dict(
                upstream_id="exp8022-public-panel",
                path=str(raw / "capture.json"),
                hash=sha256_file(raw / "capture.json"),
                artifact_field="panel.counts",
                expected=s.ROLES,
                observed=panel.get("counts", "missing_field_contract_error"),
                passed=False,
            )
        )
    owned = [r for r in receipts if r["scope"] == "owned"]
    qualified = qualification.get("passed", False) and qualification.get("forward_pass_counts") == 8
    ready = int(
        not failures
        and panel.get("complete", False)
        and qualified
        and bool(owned)
        and all(r["passed"] for r in owned)
        and result.get("duration_s", 0) >= 2
    )
    verdict = (
        "blocked"
        if failures or not panel.get("complete", False)
        else "null"
        if ready
        else "disqualified"
    )
    rows = qualification.get("rows", [])
    attempts = result.get("forward_attempts", rows)
    loads = result.get("model_invocation_counts", ZERO_INVOCATION_COUNTS)["model_loads_attempted"]
    counters = result.get("model_invocation_counts", ZERO_INVOCATION_COUNTS)
    operation_counts = {k: v for k, v in counters.items() if not k.startswith("generation_calls_")}
    operation_counts["generation"] = {
        k.removeprefix("generation_calls_"): v
        for k, v in counters.items()
        if k.startswith("generation_calls_")
    }
    value = dict(
        experiment_id=8022,
        task_id=TASK,
        milestone="2026.10.695",
        run_date="20261002",
        execution_date="20261002",
        schema="carnot.exp8022.fixed_answer_likelihood.v1",
        claim_scope="Bounded teacher-forced scorer qualification and exposed-development panel only; no correctness, fitted-head or learning benefit claim.",
        honest_verdict=f"complete_likelihood_protocol_{verdict}"
        if loads and verdict == "blocked"
        else f"complete_{verdict}_likelihood_protocol",
        verdict_class=verdict,
        gate_check_summary=failures,
        preconditions_checked=plan["checks"] + result.get("checks", []),
        rows=rows,
        sample_size_budget=dict(
            unit="private_fixed_answer_scoring_pass",
            intended=8,
            eligible=8 if panel else 0,
            started=len(attempts),
            completed=len(rows),
            excluded=0,
            failed=sum(r.get("status") == "failed" for r in attempts),
            censored=8 - len(attempts),
            independent=0,
            public_panel=dict(
                intended=768,
                eligible=4 * len(panel.get("rows", [])),
                started=0,
                completed=0,
                excluded=len(panel.get("exclusions", [])),
                failed=0,
                censored=0,
                independent=0,
                frozen_source_groups=len(panel.get("rows", [])),
                measurement_scheduled_here=False,
            ),
        ),
        random_seed=69522,
        reproducibility_checksum=canonical_hash(dict(panel=panel, qualification=qualification)),
        verifier_is_oracle=False,
        genuine_headroom=None,
        acceptance_gate_results=dict(
            runtime_qualification=bool(qualified),
            panel_complete=panel.get("complete", False),
            owned_checks=bool(owned) and all(r["passed"] for r in owned),
            scientific_benefit=False,
        ),
        positive_control_results=dict(
            scope="private_fixed_answer_runtime_plumbing_only", **qualification
        ),
        generalized_learning_benefit_score=0,
        likelihood_protocol_ready_score=ready,
        token_scoring_ready_score=int(ready),
        MODEL_SPECS=[runtime.MODEL],
        model_specs=[runtime.MODEL],
        trained_head_specs=[],
        inference_substrate="live_llm_inference",
        inference_substrate_class="model_load_no_generation"
        if loads
        else "blocked_no_run"
        if verdict == "blocked"
        else "no_model_load",
        planned_inference_substrate_class="model_load_no_generation",
        model_invocation_counts=operation_counts,
        model_invocation_counts_schema="carnot.operation_counters.separate_generation.v1",
        current_invocation_ledger=result.get("current_invocation_ledger", []),
        duration_s=elapsed,
        phase_spans=[dict(phase="bounded_protocol_and_owned_validation", start_s=0, end_s=elapsed)],
        measured_duration_s=result.get("duration_s", 0),
        forward_pass_counts=qualification.get("forward_pass_counts", 0),
        scored_tokens=qualification.get("scored_tokens", 0),
        generated_tokens=0,
        model_identity_receipt=result.get("model_identity_receipt", {}),
        gguf_sha256=result.get("gguf_sha256"),
        model_revision=result.get("model_revision"),
        gpu_lease_receipt=result.get("gpu_lease_receipt", {}),
        offload_evidence=result.get("offload_evidence", {}),
        cleanup=result.get("cleanup", {}),
        public_panel_manifest=panel,
        fixed_answer_rows=panel.get("rows", []),
        response_token_offsets={
            r["family_id"]: r["response_token_offsets"] for r in panel.get("rows", [])
        },
        view_hashes={r["family_id"]: r["view_hashes"] for r in panel.get("rows", [])},
        removal_masks={r["family_id"]: r["removal_mask"] for r in panel.get("rows", [])},
        protocol_fingerprint=canonical_hash(dict(methods=s.METHODS, public=plan.get("public", {}))),
        method_map=s.METHODS,
        evaluation_labels_opened=False,
        cited_upstream_artifacts=plan["references"],
        raw_shard_hashes=[
            reference(raw / p)
            for p in [
                "public.json",
                "plan.json",
                "capture.json",
                "validation_commands.json",
                "validation_receipts.json",
            ]
        ],
        checkpoint_references=[reference(raw / "capture.json")]
        + [reference(p) for p in sorted((raw / "forwards").glob("*.json"))],
        code_config_hashes=json.loads((raw / "validation_commands.json").read_text())[
            "code_config_hashes"
        ],
        validation_receipts=owned,
        repository_health=[r for r in receipts if r["scope"] == "repository_health"],
        coverage_statement_counts={},
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        flagged_adversarial=False,
        limitations=[
            "Restricted GASP lexical single-chunk adaptation, not reproduction.",
            "GASP reported short-answer RAGBench transfer failure.",
            "Evidence acceptance and verbal verification can diverge; sensitivity is not correctness.",
            "No new corpus; all source roles retain development exposure.",
        ],
    )
    value["raw_shard_hashes"] += [
        reference(p)
        for p in sorted(raw.rglob("*.log"))
        if "terminal" not in str(p) and "published" not in str(p) and "cold_logs" not in str(p)
    ]
    value["raw_shard_hashes"] += [
        reference(raw / name)
        for name in ["ledger.json", "gpu_lease.json", "panel.json"]
        if (raw / name).is_file()
    ]
    value["field_principles"] = {
        k: "Bind owned no-generation work to durable bytes; fixtures and exposed inputs earn zero independent science."
        for k in value
    }
    value["field_principles"]["model_invocation_counts"] = (
        "Model-load counters identify actual loads. The separate generation operation records zero attempts without denying those loads. Original flat counters remain in the capture checkpoint."
    )
    return value


def replay(value: Json) -> None:
    """Cold-reduce original checkpoints so edited aggregates cannot qualify."""
    for ref in (
        value["raw_shard_hashes"] + value["checkpoint_references"] + value["code_config_hashes"]
    ):
        checked(ref)
    verify_references([value["validation_receipts"], value["repository_health"]])
    raw = Path(value["checkpoint_references"][0]["path"]).parent
    plan, result, receipts = [
        json.loads((raw / name).read_text())
        for name in ["plan.json", "capture.json", "validation_receipts.json"]
    ]
    expected = build(plan, result, raw, receipts["receipts"], value["duration_s"])
    for key in expected:
        if (
            key not in {"coverage_statement_counts", "field_principles"}
            and value.get(key) != expected[key]
        ):
            raise ValueError("cold_reduction_drift:" + key)


def terminal(path: Path) -> Json:
    """Use unchanged terminal readers on exactly the candidate or published bytes."""
    replay(json.loads(path.read_text()))
    receipts = run_commands(
        ROOT,
        [
            CommandSpec(
                name,
                (str(ROOT / ".venv/bin/python"), "-u", script, flag, str(path)),
                "terminal",
                60,
            )
            for name, script, flag in [
                ("adversarial", "scripts/adversarial_verify.py", "--json"),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            ]
        ],
        log_dir=path.parent / "terminal_logs"
        if path.parent.name != "results"
        else path.parent / "raw" / NAME / "published_logs",
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Private replay routes and one live child end in checked atomic publication."""
    started = time.monotonic()
    s.progress("start", started)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    routes = parser.add_mutually_exclusive_group()
    routes.add_argument("--cold-replay", type=Path)
    routes.add_argument("--runtime-child", type=Path)
    routes.add_argument("--fixture", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            s.progress("cold_replay_passed", started)
            return 0
        if args.runtime_child:
            if not args.runtime_child.is_file():
                raise ValueError("public_input_missing")
            runtime.worker(args.runtime_child, args.output)
            return 0
        output = args.output.absolute()
        if args.fixture and output.is_relative_to(ROOT):
            raise ValueError("fixture_output_must_be_private")
        raw = output.parent / "raw" / output.stem
        raw.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="carnot-8022-") as directory:
            scratch = Path(directory)
            specs = commands(scratch)
            bundle = json.loads(args.fixture.read_text()) if args.fixture else {}
            plan = bundle["plan"] if args.fixture else authenticate(args.root)
            retained = []
            existing = {}
            if not args.fixture and (raw / "capture.json").is_file():
                frozen = json.loads((raw / "validation_commands.json").read_text())
                for ref in frozen["code_config_hashes"]:
                    if Path(ref["path"]).name in {
                        "fixed_answer_likelihood_8022.py",
                        "likelihood_runtime_8022.py",
                    }:
                        checked(ref)
                if json.loads((raw / "public.json").read_text()) != plan.get("public", {}):
                    raise ValueError("scoring_inputs_changed")
                existing = json.loads((raw / "capture.json").read_text())
                old_receipts = json.loads((raw / "validation_receipts.json").read_text())[
                    "receipts"
                ]
                retained = [r for r in old_receipts if r["scope"] == "repository_health"]
                archive = raw / "prior_reporting" / canonical_hash(frozen).split(":")[1]
                shutil.copytree(
                    raw,
                    archive,
                    ignore=shutil.ignore_patterns("prior_reporting"),
                    dirs_exist_ok=True,
                )
                specs = [x for x in specs if x.scope != "repository_health"]
            atomic_json(raw / "plan.json", plan)
            atomic_json(raw / "public.json", plan.get("public", {}))
            atomic_json(
                raw / "validation_commands.json",
                dict(
                    methods=s.METHODS,
                    commands=[asdict(x) for x in specs],
                    code_config_hashes=[reference(ROOT / p) for p in OWNED + [TEST]],
                ),
            )
            s.progress("before_runtime_subprocess", started, 0, 8)
            result = existing or bundle.get("result", {})
            if not args.fixture and not existing and all(c["passed"] for c in plan["checks"]):
                child = run_commands(
                    ROOT,
                    [
                        CommandSpec(
                            "teacher_forced_preflight",
                            (
                                str(ROOT / ".venv/bin/python"),
                                "-u",
                                CLI,
                                "--runtime-child",
                                str(raw / "public.json"),
                                "--output",
                                str(raw / "capture.json"),
                            ),
                            "runtime",
                            600,
                        )
                    ],
                    log_dir=raw / "runtime_logs",
                    heartbeat_s=60,
                    extra_env=dict(CARNOT_FORCE_LIVE="1", JAX_PLATFORMS="cpu"),
                )[0]
                result = (
                    json.loads((raw / "capture.json").read_text())
                    if (raw / "capture.json").is_file()
                    else {}
                )
                result["child_receipt"] = child
                if not child["passed"]:
                    result.setdefault("checks", []).append(
                        dict(
                            upstream_id="exp8022-runtime-child",
                            path=child["log_path"],
                            hash=child["log_sha256"],
                            artifact_field="exit_code",
                            expected=0,
                            observed=child["exit_code"],
                            passed=False,
                        )
                    )
            if args.fixture:
                result.setdefault("checks", []).append(
                    dict(
                        upstream_id="private_fixture",
                        path=str(args.fixture),
                        hash=sha256_file(args.fixture),
                        artifact_field="live_inference",
                        expected=True,
                        observed=False,
                        passed=False,
                    )
                )
            atomic_json(raw / "capture.json", result)
            s.progress(
                "after_runtime_before_owned_checks",
                started,
                len(result.get("qualification", {}).get("rows", [])),
            )
            receipts = (
                []
                if args.fixture or args.root != ROOT
                else run_commands(
                    ROOT,
                    specs,
                    log_dir=raw / "validation_logs",
                    heartbeat_s=10,
                    extra_env=dict(
                        CARNOT_8022_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                        COVERAGE_FILE=str(scratch / ".coverage"),
                        JAX_PLATFORMS="cpu",
                    ),
                )
            )
            receipts += retained
            atomic_json(raw / "validation_receipts.json", dict(receipts=receipts))
            value = build(plan, result, raw, receipts, time.monotonic() - started)
            if (scratch / "coverage.json").is_file():
                value["coverage_statement_counts"] = json.loads(
                    (scratch / "coverage.json").read_text()
                )["totals"]
            candidate = raw / (NAME + ".json")
            atomic_json(candidate, value)
            s.progress("before_cold_reduction", started)
            cold = run_commands(
                ROOT,
                [
                    CommandSpec(
                        "cold_reduction",
                        (
                            str(ROOT / ".venv/bin/python"),
                            "-u",
                            CLI,
                            "--cold-replay",
                            str(candidate),
                        ),
                        "terminal",
                        60,
                    )
                ],
                log_dir=raw / "cold_logs",
            )
            if not all(r["passed"] for r in cold):
                raise ValueError("cold_reduction_failed")
            publication = publish_primary(output, value, terminal)
            atomic_json(raw / "terminal_validation.json", dict(publication, cold_receipts=cold))
            report = terminal(output)
            atomic_json(raw / "published_validation.json", report)
            readers = reader_receipt(
                TASK,
                output.parent,
                field="likelihood_protocol_ready_score",
                expected=value["likelihood_protocol_ready_score"],
            )
            atomic_json(raw / "primary_readers.json", readers)
            if not report["passed"] or not readers["passed"]:
                raise ValueError("published_readers_failed")
        s.progress("complete", started, len(value["rows"]))
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        print(f"[exp8022] rejected={type(error).__name__}:{error}", flush=True)
        return 1
